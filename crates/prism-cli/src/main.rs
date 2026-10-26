//! `prism`: command-line front end for Prism Video Enhancer.

mod args;
mod bench;

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use args::ProcessArgs;
use clap::{Parser, Subcommand};
use prism_core::encoders;
use prism_core::models::{Catalog, Implementation, ModelKind};
use prism_core::plan::{Pass, PlanContext, PreviewRange, plan, shell_join};
use prism_core::presets::PresetStore;
use prism_core::progress::format_duration;
use prism_core::{JobSettings, VideoInfo, paths, timecode};
use prism_engine::ai::convert;
use prism_engine::queue::{JobState, Queue};
use prism_engine::runner::{self, Event};
use prism_engine::{Control, Tools};

#[derive(Parser)]
#[command(name = "prism", version, about = "Prism Video Enhancer: upscaling, frame interpolation and restoration on FFmpeg", long_about = None)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
    /// Only print errors and the final result.
    #[arg(long, short, global = true)]
    quiet: bool,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show what Prism reads from a video.
    Probe {
        input: PathBuf,
        #[arg(long)]
        json: bool,
    },
    /// List, import or download models.
    Models {
        #[command(subcommand)]
        action: Option<ModelsCmd>,
    },
    /// List the encoders this FFmpeg build provides.
    Encoders {
        /// Test-encode with each hardware encoder to see which work on this machine.
        #[arg(long)]
        probe: bool,
    },
    /// Show FFmpeg, ONNX Runtime and inference-provider information.
    Devices,
    /// Show the passes and FFmpeg commands a job would run.
    Plan {
        input: PathBuf,
        #[command(flatten)]
        opts: ProcessArgs,
        #[arg(long)]
        json: bool,
    },
    /// Process a video.
    Run {
        input: PathBuf,
        #[command(flatten)]
        opts: ProcessArgs,
    },
    /// Render a short processed clip starting at a given time.
    Preview {
        input: PathBuf,
        /// Start time (seconds or timecode).
        #[arg(long, default_value = "0")]
        at: String,
        /// Length in seconds.
        #[arg(long, default_value_t = 2.0)]
        duration: f64,
        #[command(flatten)]
        opts: ProcessArgs,
    },
    /// List presets, or show one as JSON.
    Presets { name: Option<String> },
    /// Run a JSON batch file through the export queue.
    Batch {
        file: PathBuf,
        /// Parallel exports.
        #[arg(long, default_value_t = 1)]
        workers: usize,
    },
    /// Measure each stage's throughput and compare with FFmpeg's equivalents.
    Bench(bench::BenchArgs),
}

#[derive(Subcommand)]
enum ModelsCmd {
    /// List the catalog and which models are installed.
    List,
    /// Install a local model file: an .onnx file as is, or a Real-ESRGAN PyTorch checkpoint
    /// (.pth) converted to ONNX.
    Import {
        weights: PathBuf,
        /// Catalog id to install as (guessed from the file name if omitted).
        #[arg(long)]
        id: Option<String>,
        /// Write only the FP32 model.
        #[arg(long)]
        fp32_only: bool,
        /// Models folder (default: the configured models folder).
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Download a catalog model's published weights and import them.
    Fetch {
        /// Catalog id, or "all".
        id: String,
        #[arg(long)]
        fp32_only: bool,
    },
    /// Forget tuned GPU tile sizes so the next run measures again.
    Retune,
}

fn main() {
    // Exit quietly when the reader goes away (`prism models list | head`), like other Unix tools.
    #[cfg(unix)]
    // SAFETY: restoring the default disposition of SIGPIPE before any threads start.
    unsafe {
        libc::signal(libc::SIGPIPE, libc::SIG_DFL);
    }
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        if runner::is_cancelled(&e) {
            eprintln!("cancelled");
            std::process::exit(130);
        }
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<()> {
    let quiet = cli.quiet;
    match cli.cmd {
        Cmd::Probe { input, json } => probe(&input, json),
        Cmd::Models { action } => models(action.unwrap_or(ModelsCmd::List)),
        Cmd::Encoders { probe } => list_encoders(probe),
        Cmd::Devices => devices(),
        Cmd::Plan { input, opts, json } => show_plan(&input, &opts, json),
        Cmd::Run { input, opts } => process(&input, &opts, None, quiet),
        Cmd::Preview { input, at, duration, opts } => {
            let tools = Tools::locate()?;
            let info = tools.probe(&input)?;
            let start = timecode::parse(&at, info.fps.as_f64()).ok_or_else(|| anyhow!("bad time '{at}'"))?;
            process(&input, &opts, Some(PreviewRange { start, duration }), quiet)
        }
        Cmd::Presets { name } => presets(name),
        Cmd::Batch { file, workers } => batch(&file, workers, quiet),
        Cmd::Bench(args) => bench::run(args),
    }
}

fn probe(input: &Path, json: bool) -> Result<()> {
    let tools = Tools::locate()?;
    let info = tools.probe(input)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&info)?);
        return Ok(());
    }
    println!("{}", info.path.display());
    println!("  container   {}", info.container);
    println!("  video       {} {}x{} SAR {}  {} fps{}  {}", info.codec, info.width, info.height, info.sar.to_ffmpeg(), info.fps, if info.vfr { " (variable)" } else { "" }, info.pix_fmt);
    println!("  frames      {}{}  duration {:.3}s", info.frame_count, if info.frame_count_exact { "" } else { " (estimated)" }, info.duration);
    println!(
        "  colour      {:?} {:?} primaries {} transfer {}{}",
        info.color.range,
        info.color.matrix,
        info.color.primaries,
        info.color.transfer,
        if info.color.inferred { " (partly inferred)" } else { "" }
    );
    println!("  scan        {:?}{}", info.field_order, if info.rotation != 0 { format!("  rotation {}°", info.rotation) } else { String::new() });
    for a in &info.audio {
        println!("  audio {}     {} {} ch {} Hz{}", a.index, a.codec, a.channels, a.sample_rate, a.language.as_deref().map(|l| format!(" [{l}]")).unwrap_or_default());
    }
    if info.subtitles > 0 {
        println!("  subtitles   {}", info.subtitles);
    }
    Ok(())
}

fn catalog() -> Catalog {
    Catalog::load(&paths::models_dir())
}

fn models(cmd: ModelsCmd) -> Result<()> {
    match cmd {
        ModelsCmd::List => {
            let cat = catalog();
            println!("Models folder: {}\n", cat.dir.display());
            for kind in [ModelKind::Enhancement, ModelKind::Interpolation, ModelKind::Stabilization, ModelKind::ParamEstimation, ModelKind::MotionDeblur, ModelKind::Hdr] {
                println!("{kind:?}");
                for m in cat.of_kind(kind) {
                    let status = match &m.implementation {
                        Implementation::Native => "built in".to_string(),
                        Implementation::Ffmpeg { filter } => format!("FFmpeg {filter}"),
                        Implementation::Onnx { source, .. } => match m.onnx_file(&cat.dir, true) {
                            Some(p) => format!("installed ({})", p.file_name().unwrap().to_string_lossy()),
                            None if source.is_some() => format!("not installed: prism models fetch {}", m.id),
                            None => format!("not installed: place {}.onnx in the models folder", m.id),
                        },
                    };
                    println!("  {:<22} {:<16} {}", m.id, m.name, status);
                }
                println!();
            }
            Ok(())
        }
        ModelsCmd::Import { weights, id, fp32_only, dir } => {
            let dir = dir.unwrap_or_else(paths::models_dir);
            let file_name = weights.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            let id = match id {
                Some(id) => id,
                // A checkpoint's name identifies the Real-ESRGAN variant; an ONNX file installs
                // under its own name.
                None => convert::catalog_id_for(&file_name)
                    .map(str::to_string)
                    .or_else(|| weights.extension().is_some_and(|e| e.eq_ignore_ascii_case("onnx")).then(|| weights.file_stem().unwrap_or_default().to_string_lossy().to_string()))
                    .ok_or_else(|| anyhow!("cannot guess the model id from '{file_name}'; pass --id"))?,
            };
            import(&weights, &dir, &id, fp32_only)
        }
        ModelsCmd::Retune => {
            prism_engine::ai::runtime::clear_tile_cache()?;
            println!("Tile sizes will be measured again on the next run.");
            Ok(())
        }
        ModelsCmd::Fetch { id, fp32_only } => {
            let cat = catalog();
            let ids: Vec<String> = if id == "all" {
                cat.models.iter().filter(|m| matches!(&m.implementation, Implementation::Onnx { source: Some(_), .. })).map(|m| m.id.clone()).collect()
            } else {
                vec![id]
            };
            for id in ids {
                let t = Instant::now();
                let files = prism_engine::ai::fetch::fetch(&cat, &id, fp32_only, &|line| println!("{line}"))?;
                report_installed(&files, t);
            }
            Ok(())
        }
    }
}

fn import(weights: &Path, dir: &Path, id: &str, fp32_only: bool) -> Result<()> {
    let t = Instant::now();
    let files = prism_engine::ai::fetch::install_file(weights, dir, id, fp32_only)?;
    report_installed(&files, t);
    Ok(())
}

fn report_installed(files: &[PathBuf], t: Instant) {
    for f in files {
        let mb = std::fs::metadata(f).map(|m| m.len() as f64 / 1e6).unwrap_or(0.0);
        println!("installed {} ({mb:.1} MB)", f.display());
    }
    println!("done in {:.2}s", t.elapsed().as_secs_f64());
}

fn list_encoders(probe: bool) -> Result<()> {
    let tools = Tools::locate()?;
    let available = tools.encoders()?;
    println!("{:<16} {:<8} {:<22} {:<12} {:<6} containers", "id", "family", "profile", "ffmpeg", "bits");
    for e in encoders::video_encoders() {
        if !available.contains(e.codec) || e.family == "Null" {
            continue;
        }
        let status = if probe && e.is_hardware() {
            if tools.test_encoder(e.codec, e.pix_fmt) { "  [works]" } else { "  [not usable here]" }
        } else {
            ""
        };
        println!("{:<16} {:<8} {:<22} {:<12} {:<6} {}{}", e.id, e.family, e.profile, e.codec, e.bit_depth, e.containers.join(","), status);
    }
    Ok(())
}

fn devices() -> Result<()> {
    match Tools::locate() {
        Ok(t) => println!("FFmpeg:        {}\n               {}", t.ffmpeg.display(), t.version().unwrap_or_default()),
        Err(e) => println!("FFmpeg:        {e}"),
    }
    match prism_engine::ai::runtime::runtime() {
        Ok(rt) => println!("ONNX Runtime:  {}\n  providers:   {}", rt.library.display(), rt.providers.join(", ")),
        Err(e) => println!("ONNX Runtime:  unavailable ({e})"),
    }
    println!("CPU threads:   {}", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    println!("Models folder: {}", paths::models_dir().display());
    Ok(())
}

fn job_context<'a>(cat: &'a Catalog, tools: &Tools, preview: Option<PreviewRange>, exists: &'a dyn Fn(&Path) -> bool) -> Result<(PlanContext<'a>, std::collections::HashSet<String>)> {
    let enc = tools.encoders()?;
    let id = format!("job-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0));
    Ok((PlanContext { catalog: cat, temp_dir: paths::temp_root().join(id), encoders_available: None, preview, exists }, enc))
}

fn build_plan(input: &Path, opts: &ProcessArgs, preview: Option<PreviewRange>) -> Result<(Tools, Catalog, VideoInfo, prism_core::JobPlan)> {
    let tools = Tools::locate()?;
    let info = tools.probe(input)?;
    let settings = opts.settings(info.fps.as_f64())?;
    let cat = catalog();
    let exists = |p: &Path| p.exists();
    let (mut ctx, available) = job_context(&cat, &tools, preview, &exists)?;
    ctx.encoders_available = Some(&available);
    let p = plan(&info, &settings, &ctx)?;
    drop(ctx);
    Ok((tools, cat, info, p))
}

fn show_plan(input: &Path, opts: &ProcessArgs, json: bool) -> Result<()> {
    let (tools, _cat, _info, p) = build_plan(input, opts, None)?;
    if json {
        println!("{}", serde_json::to_string_pretty(&p)?);
        return Ok(());
    }
    print_plan(&p, &tools);
    Ok(())
}

fn print_plan(p: &prism_core::JobPlan, tools: &Tools) {
    println!("Output:   {}", p.output.display());
    println!("Size:     {}x{} -> {}x{}  ({} frames at {} fps -> {} frames at {} fps)", p.geometry.source.0, p.geometry.source.1, p.geometry.output.0, p.geometry.output.1, p.frames_in, p.fps_in, p.frames_out, p.fps_out);
    println!("Encoder:  {}", p.encoder);
    println!("Metadata: {}", p.description);
    for w in &p.warnings {
        println!("Warning:  {w}");
    }
    let ff = tools.ffmpeg.display().to_string();
    for (i, pass) in p.passes.iter().enumerate() {
        println!("\nPass {}/{}: {}", i + 1, p.passes.len(), pass.label());
        match pass {
            Pass::Ffmpeg(f) => println!("  {}", shell_join(&ff, &tools.adapt_args(&f.args))),
            Pass::Engine(e) => {
                println!("  decode: {}", shell_join(&ff, &tools.adapt_args(&e.decode)));
                println!("          -> {} {}x{}", e.input.pix_fmt(), e.input.width, e.input.height);
                for s in &e.stages {
                    println!("  stage:  {}", s.label());
                }
                println!("          -> {} {}x{}", e.output.pix_fmt(), e.output.width, e.output.height);
                println!("  encode: {}", shell_join(&ff, &tools.adapt_args(&e.encode)));
            }
        }
    }
}

/// Live progress line on stderr.
fn progress_printer(quiet: bool) -> impl Fn(Event) + Sync {
    let label = Mutex::new(String::new());
    move |ev: Event| {
        if quiet {
            if let Event::Warning(w) = ev {
                eprintln!("warning: {w}");
            }
            return;
        }
        match ev {
            Event::PassStarted { index, count, label: l } => {
                *label.lock().unwrap() = if count > 1 { format!("{l} {}/{count}", index + 1) } else { l };
                eprintln!();
            }
            Event::Progress(p) => {
                let l = label.lock().unwrap();
                eprint!("\r[{l}] frame {}/{} {:5.1} fps  eta {}  {:3}%   ", p.frame, p.total, p.fps, format_duration(p.eta_secs), p.percent);
                let _ = std::io::stderr().flush();
            }
            Event::Info(m) => eprintln!("\n{m}"),
            Event::Warning(w) => eprintln!("\nwarning: {w}"),
        }
    }
}

fn process(input: &Path, opts: &ProcessArgs, preview: Option<PreviewRange>, quiet: bool) -> Result<()> {
    let (tools, cat, _info, p) = build_plan(input, opts, preview)?;
    prism_engine::init_threads(p.device.threads);
    if !quiet {
        eprintln!("{} -> {}", input.display(), p.output.display());
    }
    let control = Control::new();
    let c2 = control.clone();
    let _ = ctrlc_hook(move || c2.cancel());
    let report = runner::run_plan(&p, &tools, &cat, &control, &progress_printer(quiet))?;
    if !quiet {
        eprintln!();
        for (k, v) in &report.stats {
            eprintln!("{k}: {v}");
        }
        if !report.timings.is_empty() {
            let t: Vec<String> = report.timings.iter().map(|(k, v)| format!("{k} {v:.1}")).collect();
            eprintln!("busy ms/frame (concurrent steps): {}", t.join(", "));
        }
    }
    println!("{} ({} frames in {:.1}s, {:.1} fps)", report.output.display(), report.frames, report.elapsed_secs, report.fps);
    Ok(())
}

/// Cancels the job on Ctrl+C so FFmpeg children are cleaned up and partial output removed.
fn ctrlc_hook(f: impl Fn() + Send + 'static) -> Result<()> {
    static HOOK: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>> = std::sync::OnceLock::new();
    let f = Mutex::new(f);
    let _ = HOOK.set(Box::new(move || (f.lock().unwrap())()));
    #[cfg(windows)]
    {
        unsafe extern "system" fn handler(_: u32) -> i32 {
            if let Some(h) = HOOK.get() {
                h();
            }
            1
        }
        unsafe extern "system" {
            fn SetConsoleCtrlHandler(handler: Option<unsafe extern "system" fn(u32) -> i32>, add: i32) -> i32;
        }
        // SAFETY: registers a handler that only calls our thread-safe hook.
        unsafe { SetConsoleCtrlHandler(Some(handler), 1) };
    }
    #[cfg(unix)]
    {
        use std::sync::atomic::{AtomicBool, Ordering};
        static HIT: AtomicBool = AtomicBool::new(false);
        extern "C" fn handler(_: libc::c_int) {
            HIT.store(true, Ordering::SeqCst);
            // A second Ctrl+C terminates immediately.
            // SAFETY: signal() is async-signal-safe.
            unsafe { libc::signal(libc::SIGINT, libc::SIG_DFL) };
        }
        // SAFETY: the handler only stores to an atomic and resets the disposition.
        unsafe { libc::signal(libc::SIGINT, handler as *const () as libc::sighandler_t) };
        // The hook takes a lock, which a signal handler must not, so a thread runs it.
        std::thread::spawn(|| {
            loop {
                if HIT.swap(false, Ordering::SeqCst) {
                    if let Some(h) = HOOK.get() {
                        h();
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
        });
    }
    Ok(())
}

fn presets(name: Option<String>) -> Result<()> {
    let store = PresetStore::new(paths::presets_dir());
    let (list, errors) = store.list();
    for e in errors {
        eprintln!("warning: {e}");
    }
    match name {
        Some(n) => {
            let p = list.into_iter().find(|p| p.name.eq_ignore_ascii_case(&n)).ok_or_else(|| anyhow!("no preset named '{n}'"))?;
            println!("{}", serde_json::to_string_pretty(&p)?);
        }
        None => {
            for p in list {
                println!("{:<32} {}{}", p.name, p.description, if p.builtin { "" } else { "  (user)" });
            }
            println!("\nUser presets live in {}", store.dir().display());
        }
    }
    Ok(())
}

#[derive(serde::Deserialize)]
struct BatchJob {
    input: PathBuf,
    #[serde(default)]
    preset: Option<String>,
    #[serde(default)]
    settings: Option<JobSettings>,
    #[serde(default)]
    output: Option<PathBuf>,
    #[serde(default = "normal_priority")]
    priority: u8,
}

fn normal_priority() -> u8 {
    prism_engine::queue::PRIORITY_NORMAL
}

fn batch(file: &Path, workers: usize, quiet: bool) -> Result<()> {
    let jobs: Vec<BatchJob> = serde_json::from_str(&std::fs::read_to_string(file)?).context("parsing batch file")?;
    let tools = Tools::locate()?;
    let cat = Arc::new(catalog());
    let available = tools.encoders()?;
    let queue = Queue::new(tools.clone(), cat.clone(), workers);
    let store = PresetStore::new(paths::presets_dir());
    let exists = |p: &Path| p.exists();
    for (i, j) in jobs.iter().enumerate() {
        let info = tools.probe(&j.input)?;
        let mut s = j.settings.clone().unwrap_or_default();
        if let Some(name) = &j.preset {
            store.find(name).ok_or_else(|| anyhow!("no preset named '{name}'"))?.apply(&mut s);
        }
        if let Some(o) = &j.output {
            s.export.output_path = Some(o.clone());
        }
        let ctx = PlanContext {
            catalog: &cat,
            temp_dir: paths::temp_root().join(format!("batch-{}-{i}", std::process::id())),
            encoders_available: Some(&available),
            preview: None,
            exists: &exists,
        };
        let p = plan(&info, &s, &ctx).with_context(|| format!("planning {}", j.input.display()))?;
        let name = j.input.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        queue.submit(&name, p, j.priority);
    }
    let start = Instant::now();
    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        let views = queue.jobs();
        if !quiet {
            let line: Vec<String> = views.iter().map(|v| format!("{}:{:?} {}%", v.name, v.state, v.progress.percent)).collect();
            eprint!("\r{}   ", line.join(" | "));
        }
        if views.iter().all(|v| v.state.is_finished()) {
            eprintln!();
            let mut failed = 0;
            for v in views {
                match v.state {
                    JobState::Done => println!("done   {} -> {}", v.name, v.output.display()),
                    _ => {
                        failed += 1;
                        println!("{:?} {}: {}", v.state, v.name, v.error.unwrap_or(v.message));
                    }
                }
            }
            println!("batch finished in {:.1}s", start.elapsed().as_secs_f64());
            if failed > 0 {
                bail!("{failed} job(s) did not complete");
            }
            return Ok(());
        }
    }
}
