//! Frame access for the viewer: single-frame grabs, paced playback, thumbnails and instant
//! single-frame processing through the engine.

use std::io::Read;
use std::path::Path;
use std::process::{Child, Stdio};
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use crossbeam_channel::{Receiver, bounded};
use prism_core::models::Catalog;
use prism_core::plan::{PlanContext, PreviewRange, plan};
use prism_core::settings::{AudioMode, HdrTransfer};
use prism_core::{JobSettings, VideoInfo, paths};
use prism_engine::process::command;
use prism_engine::{Control, Frame, Tools, runner};

#[derive(Clone)]
pub struct RgbaImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RgbaImage {
    pub fn to_color_image(&self) -> egui::ColorImage {
        egui::ColorImage::from_rgba_unmultiplied([self.width, self.height], &self.data)
    }
}

/// Decoded (display-oriented) size of a video, fitted inside `max_w x max_h`.
pub fn display_size(info: &VideoInfo, max_w: usize, max_h: usize) -> (usize, usize) {
    let (mut w, mut h) = (info.width as f64 * info.sar.as_f64(), info.height as f64);
    if info.rotation.rem_euclid(180) == 90 {
        (w, h) = (h, w);
    }
    let k = (max_w as f64 / w).min(max_h as f64 / h).min(1.0);
    let even = |v: f64| (((v * k) / 2.0).round() as usize * 2).max(2);
    (even(w), even(h))
}

fn read_frames(child: &mut Child, w: usize, h: usize, max: usize) -> Result<Vec<RgbaImage>> {
    let mut out = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
    let mut frames = Vec::new();
    let size = w * h * 4;
    while frames.len() < max {
        let mut data = vec![0u8; size];
        let mut filled = 0;
        while filled < size {
            match out.read(&mut data[filled..])? {
                0 => break,
                n => filled += n,
            }
        }
        if filled < size {
            break;
        }
        frames.push(RgbaImage { width: w, height: h, data });
    }
    let _ = child.kill();
    let _ = child.wait();
    Ok(frames)
}

/// One frame at `t` seconds, scaled to `w x h`.
pub fn grab(tools: &Tools, path: &Path, t: f64, w: usize, h: usize) -> Result<RgbaImage> {
    let mut child = command(&tools.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{:.4}", t.max(0.0)), "-i"])
        .arg(path)
        .args(["-frames:v", "1", "-vf", &format!("scale={w}:{h}:flags=bicubic"), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    read_frames(&mut child, w, h, 1)?.into_iter().next().ok_or_else(|| anyhow!("no frame at {t:.3}s"))
}

/// Evenly spaced thumbnails across the video.
pub fn thumbnails(tools: &Tools, info: &VideoInfo, count: usize, height: usize) -> Result<Vec<RgbaImage>> {
    let (dw, dh) = display_size(info, 4096, 4096);
    let w = ((dw as f64 * height as f64 / dh as f64) / 2.0).round() as usize * 2;
    let rate = count as f64 / info.duration.max(0.1);
    // Keyframe-only decoding is fast for long videos but yields too few frames for short ones.
    let skip: &[&str] = if info.duration > 120.0 { &["-skip_frame", "nokey"] } else { &[] };
    let mut child = command(&tools.ffmpeg)
        .args(["-hide_banner", "-loglevel", "error"])
        .args(skip)
        .arg("-i")
        .arg(&info.path)
        .args(["-vf", &format!("fps={rate:.6},scale={w}:{height}"), "-frames:v", &count.to_string(), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    read_frames(&mut child, w, height, count)
}

/// Sequential decoding for playback; frames arrive in order through a small queue.
pub struct Player {
    pub fps: f64,
    rx: Receiver<RgbaImage>,
    child: Child,
    pub delivered: u64,
}

impl Player {
    pub fn start(tools: &Tools, path: &Path, t: f64, w: usize, h: usize, fps: f64) -> Result<Player> {
        let mut child = command(&tools.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-ss", &format!("{:.4}", t.max(0.0)), "-i"])
            .arg(path)
            .args(["-vf", &format!("scale={w}:{h}:flags=bilinear"), "-f", "rawvideo", "-pix_fmt", "rgba", "pipe:1"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()?;
        let mut out = child.stdout.take().ok_or_else(|| anyhow!("no stdout"))?;
        let (tx, rx) = bounded(6);
        std::thread::spawn(move || {
            let size = w * h * 4;
            loop {
                let mut data = vec![0u8; size];
                let mut filled = 0;
                while filled < size {
                    match out.read(&mut data[filled..]) {
                        Ok(0) | Err(_) => return,
                        Ok(n) => filled += n,
                    }
                }
                if tx.send(RgbaImage { width: w, height: h, data }).is_err() {
                    return;
                }
            }
        });
        Ok(Player { fps, rx, child, delivered: 0 })
    }

    /// The newest frame due at `elapsed` seconds of playback, dropping late ones.
    pub fn frame_for(&mut self, elapsed: f64) -> Option<RgbaImage> {
        let due = (elapsed * self.fps).floor() as u64;
        let mut latest = None;
        while self.delivered <= due {
            match self.rx.try_recv() {
                Ok(f) => {
                    self.delivered += 1;
                    latest = Some(f);
                }
                Err(_) => break,
            }
        }
        latest
    }
}

impl Drop for Player {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

fn to_u8(v: f32) -> u8 {
    (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8
}

/// Converts an engine frame to RGBA for display. HDR frames are tone-mapped for an SDR screen.
pub fn frame_to_rgba(f: &Frame, hdr: Option<HdrTransfer>) -> RgbaImage {
    let n = f.pixels();
    let mut data = vec![255u8; n * 4];
    let [r, g, b] = f.planes();
    // BT.2020 -> BT.709 primaries (linear).
    const M: [[f32; 3]; 3] = [[1.660_491, -0.587_641, -0.072_850], [-0.124_550, 1.132_900, -0.008_349], [-0.018_151, -0.100_579, 1.118_730]];
    for i in 0..n {
        let (mut cr, mut cg, mut cb) = (r[i], g[i], b[i]);
        if let Some(t) = hdr {
            let dec = |e: f32| -> f32 {
                match t {
                    HdrTransfer::Pq => prism_engine::ops::color::pq_decode(e) * 10000.0,
                    // Approximate HLG decode to nits (1000-nit display).
                    HdrTransfer::Hlg => {
                        let e = e.clamp(0.0, 1.0);
                        let lin = if e <= 0.5 { e * e / 3.0 } else { (((e - 0.559_910_7) / 0.178_832_77).exp() + 0.284_668_92) / 12.0 };
                        lin.powf(1.2) * 1000.0
                    }
                }
            };
            let (lr, lg, lb) = (dec(cr) / 203.0, dec(cg) / 203.0, dec(cb) / 203.0);
            let m = |row: [f32; 3]| row[0] * lr + row[1] * lg + row[2] * lb;
            let tm = |x: f32| (x.max(0.0) / (1.0 + x.max(0.0) / 4.0)).powf(1.0 / 2.4);
            cr = tm(m(M[0]));
            cg = tm(m(M[1]));
            cb = tm(m(M[2]));
        }
        data[4 * i] = to_u8(cr);
        data[4 * i + 1] = to_u8(cg);
        data[4 * i + 2] = to_u8(cb);
    }
    RgbaImage { width: f.width, height: f.height, data }
}

/// Processes the single frame at `t` through the enhancement-type stages (no interpolation or
/// stabilization, which need neighbouring frames or an analysis pass).
pub fn render_instant(tools: &Tools, catalog: &Catalog, info: &VideoInfo, settings: &JobSettings, t: f64, control: &Arc<Control>) -> Result<RgbaImage> {
    let mut s = settings.clone();
    s.interpolation.enabled = false;
    s.stabilization.enabled = false;
    s.export.encoder = if s.hdr.enabled { "null-10".into() } else { "null".into() };
    s.export.container = "null".into();
    s.export.audio.mode = AudioMode::None;
    s.export.output_path = None;
    if !s.needs_engine() {
        bail!("no enhancement stages are active");
    }
    let fps = info.fps.as_f64().max(1.0);
    let exists = |_: &Path| false;
    let ctx = PlanContext {
        catalog,
        temp_dir: paths::temp_root().join(format!("instant-{}-{}", std::process::id(), (t * 1000.0) as u64)),
        encoders_available: None,
        preview: Some(PreviewRange { start: t, duration: 1.5 / fps }),
        exists: &exists,
    };
    let p = plan(info, &s, &ctx)?;
    let frames = runner::render_frames(&p, tools, catalog, control)?;
    let f = frames.first().ok_or_else(|| anyhow!("no frame decoded at {t:.3}s"))?;
    Ok(frame_to_rgba(f, s.hdr.enabled.then_some(s.hdr.transfer)))
}
