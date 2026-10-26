//! `prism bench`: stage throughput, with FFmpeg's closest built-in filters as reference points.
//!
//! Every scenario decodes the same clip and discards the result (FFmpeg's null muxer), so the
//! numbers measure processing, not encoding. Prism's fps excludes model loading; the wall time
//! includes it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use prism_core::models::{Catalog, ModelKind};
use prism_core::plan::{PlanContext, plan};
use prism_core::settings::*;
use prism_core::{JobSettings, paths};
use prism_engine::process::command;
use prism_engine::{Control, Tools, runner};
use serde::Serialize;

#[derive(Args, Debug)]
pub struct BenchArgs {
    /// Benchmark this video instead of a generated clip.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// Size of the generated clip.
    #[arg(long, default_value = "1920x1080")]
    pub size: String,
    /// Frames in the generated clip.
    #[arg(long, default_value_t = 120)]
    pub frames: u32,
    /// Frames used for network (ONNX) scenarios, which are much slower.
    #[arg(long, default_value_t = 24)]
    pub ai_frames: u32,
    /// Skip the FFmpeg reference runs.
    #[arg(long)]
    pub no_compare: bool,
    /// Only run scenarios whose name contains this text.
    #[arg(long)]
    pub only: Option<String>,
    /// Inference device: auto, cpu, dml[:N], cuda[:N].
    #[arg(long, default_value = "auto")]
    pub device: String,
    /// Write results as JSON.
    #[arg(long)]
    pub json: Option<PathBuf>,
}

#[derive(Serialize)]
struct Row {
    scenario: String,
    frames: u64,
    prism_fps: f64,
    prism_wall_s: f64,
    reference: Option<String>,
    reference_fps: Option<f64>,
    speedup: Option<f64>,
    notes: String,
}

struct Scenario {
    name: String,
    settings: JobSettings,
    /// FFmpeg filter chain doing the closest equivalent work.
    reference: Option<String>,
    ai: bool,
}

fn base() -> JobSettings {
    let mut s = JobSettings::default();
    s.export.encoder = "null".into();
    s.export.audio.mode = AudioMode::None;
    s
}

fn manual(model: &str, scale: f64) -> JobSettings {
    let mut s = base();
    s.enhancement.enabled = true;
    s.enhancement.model = model.into();
    s.enhancement.param_mode = ParamMode::Manual;
    s.enhancement.recover_original = 0;
    s.output.resolution = Resolution::Scale(scale);
    s
}

fn scenarios(cat: &Catalog, fps: f64) -> Vec<Scenario> {
    let mut v = vec![Scenario { name: "decode only".into(), settings: base(), reference: None, ai: false }];
    v.push(Scenario { name: "plain resize 2x (routed to FFmpeg)".into(), settings: manual("prism-classic", 2.0), reference: Some("scale=iw*2:ih*2:flags=lanczos".into()), ai: false });
    let mut ring = manual("prism-classic", 2.0);
    ring.enhancement.dehalo = 20;
    v.push(Scenario { name: "classic 2x (Lanczos + anti-ringing)".into(), settings: ring, reference: Some("scale=iw*2:ih*2:flags=lanczos".into()), ai: false });
    let mut enh = manual("prism-classic", 2.0);
    enh.enhancement.noise = 50;
    enh.enhancement.sharpen = 30;
    enh.enhancement.details = 20;
    enh.enhancement.compression = 30;
    v.push(Scenario {
        name: "classic enhance 2x (denoise, deblock, sharpen)".into(),
        settings: enh,
        reference: Some("nlmeans=s=3:p=5:r=9,deblock,scale=iw*2:ih*2:flags=lanczos,cas=0.4".into()),
        ai: false,
    });
    let mut ring4 = manual("prism-classic", 4.0);
    ring4.enhancement.dehalo = 20;
    v.push(Scenario { name: "classic 4x (Lanczos + anti-ringing)".into(), settings: ring4, reference: Some("scale=iw*4:ih*4:flags=lanczos".into()), ai: false });
    let mut flow = base();
    flow.interpolation.enabled = true;
    flow.interpolation.output_fps = Some(fps * 2.0);
    v.push(Scenario {
        name: "interpolate 2x (flow)".into(),
        settings: flow.clone(),
        reference: Some(format!("minterpolate=fps={}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1", fps * 2.0)),
        ai: false,
    });
    let mut blend = flow;
    blend.interpolation.model = "prism-blend".into();
    v.push(Scenario { name: "interpolate 2x (blend, routed to FFmpeg)".into(), settings: blend, reference: Some(format!("framerate=fps={}", fps * 2.0)), ai: false });
    let mut hdr = base();
    hdr.hdr.enabled = true;
    hdr.export.encoder = "null-10".into();
    v.push(Scenario { name: "sdr to hdr10".into(), settings: hdr, reference: None, ai: false });
    let mut grain = base();
    grain.grain.enabled = true;
    grain.grain.amount = 40;
    v.push(Scenario { name: "film grain".into(), settings: grain, reference: Some("noise=alls=10:allf=t".into()), ai: false });
    for m in cat.of_kind(ModelKind::Enhancement).filter(|m| m.is_onnx() && m.is_available(&cat.dir)) {
        let scale = *m.scales.first().unwrap_or(&4) as f64;
        v.push(Scenario { name: format!("network {} {}x", m.id, scale), settings: manual(&m.id, scale), reference: None, ai: true });
    }
    v
}

fn generate_clip(tools: &Tools, size: &str, frames: u32, dir: &Path) -> Result<PathBuf> {
    let path = dir.join(format!("bench-{size}-{frames}.mp4"));
    if path.exists() {
        return Ok(path);
    }
    std::fs::create_dir_all(dir)?;
    // Moving test pattern with grain, encoded like typical delivery video.
    let src = format!("testsrc2=size={size}:rate=30,noise=alls=8:allf=t,format=yuv420p");
    let o = command(&tools.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i", &src, "-frames:v", &frames.to_string(), "-c:v", "libx264", "-preset", "fast", "-crf", "20"])
        .arg(&path)
        .output()?;
    if !o.status.success() {
        bail!("could not generate the benchmark clip: {}", String::from_utf8_lossy(&o.stderr));
    }
    Ok(path)
}

fn reference_fps(tools: &Tools, input: &Path, chain: &str, frames: u64, limit_frames: Option<u64>) -> Result<f64> {
    let mut cmd = command(&tools.ffmpeg);
    cmd.args(["-hide_banner", "-nostdin", "-loglevel", "error", "-i"]).arg(input);
    if let Some(n) = limit_frames {
        cmd.args(["-frames:v", &n.to_string()]);
    }
    cmd.args(["-vf", chain, "-an", "-f", "null", "-"]);
    let t = Instant::now();
    let o = cmd.output()?;
    if !o.status.success() {
        bail!("{}", String::from_utf8_lossy(&o.stderr).trim());
    }
    Ok(frames as f64 / t.elapsed().as_secs_f64())
}

pub fn run(args: BenchArgs) -> Result<()> {
    let tools = Tools::locate()?;
    let cat = Catalog::load(&paths::models_dir());
    let input = match &args.input {
        Some(p) => p.clone(),
        None => generate_clip(&tools, &args.size, args.frames, &paths::temp_root().join("bench"))?,
    };
    let info = tools.probe(&input)?;
    let fps = info.fps.as_f64();
    println!("Input: {} ({}x{}, {} frames, {} fps)", input.display(), info.width, info.height, info.frame_count, info.fps);
    match prism_engine::ai::runtime::runtime() {
        Ok(rt) => println!("ONNX Runtime: {} [{}]", rt.library.display(), rt.providers.join(", ")),
        Err(e) => println!("ONNX Runtime: unavailable ({e})"),
    }
    println!("CPU threads: {}\n", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    prism_engine::init_threads(0);
    let encoders = tools.encoders()?;
    let mut rows = Vec::new();
    println!("{:<46} {:>7} {:>9} {:>8}   {:<10} {:>9}", "scenario", "frames", "prism fps", "wall s", "ffmpeg ref", "speedup");
    for sc in scenarios(&cat, fps) {
        if let Some(o) = &args.only {
            if !sc.name.contains(o.as_str()) {
                continue;
            }
        }
        let mut s = sc.settings.clone();
        s.device.backend = match args.device.split(':').next().unwrap_or("auto") {
            "cpu" => Backend::Cpu,
            "dml" | "directml" => Backend::DirectMl,
            "cuda" => Backend::Cuda,
            "coreml" => Backend::CoreMl,
            _ => Backend::Auto,
        };
        if let Some(i) = args.device.split_once(':').and_then(|(_, i)| i.parse().ok()) {
            s.device.device_index = i;
        }
        let limit = if sc.ai { Some(args.ai_frames as u64) } else { None };
        if let Some(n) = limit {
            s.trim = Some(TrimRange { start: 0.0, end: n as f64 / fps });
        }
        let exists = |p: &Path| p.exists();
        let ctx = PlanContext {
            catalog: &cat,
            temp_dir: paths::temp_root().join(format!("bench-job-{}", std::process::id())),
            encoders_available: Some(&encoders),
            preview: None,
            exists: &exists,
        };
        let p = plan(&info, &s, &ctx).with_context(|| format!("planning '{}'", sc.name))?;
        let control = Control::new();
        let notes = std::sync::Mutex::new(Vec::new());
        let on_event = |e: runner::Event| {
            if let runner::Event::Info(m) = e {
                notes.lock().unwrap().push(m);
            }
        };
        let t = Instant::now();
        let r = match runner::run_plan(&p, &tools, &cat, &control, &on_event) {
            Ok(r) => r,
            Err(e) => {
                println!("{:<46} failed: {e:#}", sc.name);
                continue;
            }
        };
        let wall = t.elapsed().as_secs_f64();
        let (ref_fps, speedup) = match (&sc.reference, args.no_compare) {
            (Some(chain), false) => match reference_fps(&tools, &input, chain, r.frames, limit) {
                Ok(f) => (Some(f), Some(r.fps / f)),
                Err(e) => {
                    eprintln!("reference for '{}' failed: {e}", sc.name);
                    (None, None)
                }
            },
            _ => (None, None),
        };
        println!(
            "{:<46} {:>7} {:>9.1} {:>8.2}   {:<10} {:>9}",
            sc.name,
            r.frames,
            r.fps,
            wall,
            ref_fps.map(|f| format!("{f:.1} fps")).unwrap_or_else(|| "-".into()),
            speedup.map(|s| format!("{s:.2}x")).unwrap_or_else(|| "-".into())
        );
        let notes = notes.into_inner().unwrap().join("; ");
        rows.push(Row { scenario: sc.name, frames: r.frames, prism_fps: r.fps, prism_wall_s: wall, reference: sc.reference, reference_fps: ref_fps, speedup, notes });
    }
    if let Some(path) = args.json {
        std::fs::write(&path, serde_json::to_vec_pretty(&rows)?).map_err(|e| anyhow!("writing {}: {e}", path.display()))?;
        println!("\nResults written to {}", path.display());
    }
    Ok(())
}
