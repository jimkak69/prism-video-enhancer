//! Processing options shared by `plan`, `run`, `preview` and `batch`, mapped onto `JobSettings`.

use std::path::PathBuf;

use anyhow::{Context, Result, anyhow, bail};
use clap::Args;
use prism_core::presets::PresetStore;
use prism_core::settings::*;
use prism_core::{JobSettings, paths, timecode};

#[derive(Args, Debug, Clone, Default)]
pub struct ProcessArgs {
    /// Start from a preset (see `prism presets`).
    #[arg(long)]
    pub preset: Option<String>,
    /// Start from a JobSettings JSON file (e.g. exported by the GUI).
    #[arg(long)]
    pub settings: Option<PathBuf>,

    // ---- Enhancement
    /// Output resolution: 2x, 4x, 720p, 1080p, 4k, 8k, 2560x1440, original. Enables enhancement.
    #[arg(long, value_name = "RES")]
    pub upscale: Option<String>,
    /// How a fixed output box is filled: fit, crop, pad, stretch.
    #[arg(long, value_name = "MODE")]
    pub fit: Option<String>,
    /// Enhancement model id (see `prism models`). Enables enhancement.
    #[arg(long)]
    pub model: Option<String>,
    /// Enable enhancement with the current/default model at the original size.
    #[arg(long)]
    pub enhance: bool,
    /// Parameter mode: auto, relative, manual.
    #[arg(long)]
    pub mode: Option<String>,
    #[arg(long, value_name = "0-100")]
    pub compression: Option<i32>,
    #[arg(long, value_name = "0-100")]
    pub details: Option<i32>,
    #[arg(long, value_name = "0-100")]
    pub sharpen: Option<i32>,
    #[arg(long, value_name = "0-100")]
    pub denoise: Option<i32>,
    #[arg(long, value_name = "0-100")]
    pub dehalo: Option<i32>,
    /// Anti-alias (negative) or deblur (positive).
    #[arg(long, value_name = "-100-100", allow_hyphen_values = true)]
    pub deblur: Option<i32>,
    #[arg(long, value_name = "0-100")]
    pub add_noise: Option<i32>,
    /// Blend back the plainly resampled original (default 20).
    #[arg(long, value_name = "0-100")]
    pub recover: Option<i32>,
    /// Focus fix: off, normal, strong.
    #[arg(long)]
    pub focus_fix: Option<String>,
    /// Second enhancement model (runs after the first).
    #[arg(long)]
    pub second_model: Option<String>,
    /// Intermediate scale for two-stage enhancement: auto, 1x, 2x, 4x.
    #[arg(long)]
    pub intermediate: Option<String>,

    // ---- Frame interpolation
    /// Output frame rate (enables interpolation).
    #[arg(long)]
    pub fps: Option<f64>,
    /// Slow-motion factor 1-16 (enables interpolation).
    #[arg(long)]
    pub slowmo: Option<f64>,
    /// Interpolation model: prism-flow, prism-blend, rife.
    #[arg(long)]
    pub fi_model: Option<String>,
    /// Keep duplicated frames instead of re-timing them.
    #[arg(long)]
    pub no_dedup: bool,
    /// Duplicate sensitivity 0-100 (default 10).
    #[arg(long)]
    pub dup_threshold: Option<f64>,
    /// Change the frame rate by dropping/repeating frames (no interpolation).
    #[arg(long)]
    pub retime: Option<f64>,

    // ---- Other filters
    /// Stabilize: autocrop or fullframe.
    #[arg(long, num_args = 0..=1, default_missing_value = "autocrop")]
    pub stabilize: Option<String>,
    #[arg(long, value_name = "0-100")]
    pub smoothness: Option<i32>,
    /// Deinterlace: same, double, telecine.
    #[arg(long, num_args = 0..=1, default_missing_value = "same")]
    pub deinterlace: Option<String>,
    /// Field order: auto, tff, bff.
    #[arg(long)]
    pub field_order: Option<String>,
    /// Motion deblur strength 0-100.
    #[arg(long, value_name = "0-100")]
    pub motion_deblur: Option<i32>,
    /// SDR to HDR: pq or hlg.
    #[arg(long, num_args = 0..=1, default_missing_value = "pq")]
    pub hdr: Option<String>,
    #[arg(long, value_name = "NITS")]
    pub hdr_peak: Option<f64>,
    /// Film grain amount 0-100.
    #[arg(long, value_name = "0-100")]
    pub grain: Option<i32>,
    #[arg(long, value_name = "0-5")]
    pub grain_size: Option<f64>,
    /// Trim, e.g. 10-20.5 or 00:00:10:00-00:00:20:00.
    #[arg(long, value_name = "START-END")]
    pub trim: Option<String>,
    /// Crop WxH+X+Y (after rotation).
    #[arg(long)]
    pub crop: Option<String>,
    /// Rotate clockwise: 90, 180, 270.
    #[arg(long)]
    pub rotate: Option<u32>,
    /// 3D LUT file applied first.
    #[arg(long)]
    pub lut: Option<PathBuf>,

    // ---- Export
    /// Encoder id (see `prism encoders`).
    #[arg(long)]
    pub encoder: Option<String>,
    /// high, medium, low (constant quality).
    #[arg(long)]
    pub quality: Option<String>,
    /// Target bitrate in Mbit/s (instead of constant quality).
    #[arg(long)]
    pub bitrate: Option<f64>,
    /// Container extension (mp4, mkv, mov, webm).
    #[arg(long)]
    pub container: Option<String>,
    /// Audio: copy, none, or codec[:kbps] (aac, ac3, opus, flac, pcm, vorbis).
    #[arg(long)]
    pub audio: Option<String>,
    /// Output file.
    #[arg(short, long)]
    pub output: Option<PathBuf>,
    /// Output folder (file name derived from the input).
    #[arg(long)]
    pub output_dir: Option<PathBuf>,
    #[arg(long)]
    pub overwrite: bool,

    // ---- Device
    /// auto, cpu, gpu (this OS's GPU provider), dml[:N], cuda[:N] or coreml.
    #[arg(long)]
    pub device: Option<String>,
    /// Fraction of GPU memory for tiles (0.1-1).
    #[arg(long)]
    pub vram: Option<f64>,
    /// Fixed tile edge in pixels.
    #[arg(long)]
    pub tile: Option<u32>,
    /// Prefer FP32 model files.
    #[arg(long)]
    pub fp32: bool,
    /// Worker threads for native kernels (0 = all).
    #[arg(long)]
    pub threads: Option<usize>,
    /// Let FFmpeg decode on the GPU.
    #[arg(long)]
    pub hwdec: bool,
}

fn parse_crop(s: &str) -> Result<CropRect> {
    let (wh, xy) = s.split_once('+').ok_or_else(|| anyhow!("crop must look like WxH+X+Y"))?;
    let (w, h) = wh.split_once('x').ok_or_else(|| anyhow!("crop must look like WxH+X+Y"))?;
    let (x, y) = xy.split_once('+').ok_or_else(|| anyhow!("crop must look like WxH+X+Y"))?;
    Ok(CropRect { width: w.parse()?, height: h.parse()?, x: x.parse()?, y: y.parse()? })
}

impl ProcessArgs {
    pub fn settings(&self, fps_hint: f64) -> Result<JobSettings> {
        let mut s = match (&self.settings, &self.preset) {
            (Some(path), _) => serde_json::from_str(&std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?).context("parsing settings JSON")?,
            (None, Some(name)) => {
                let store = PresetStore::new(paths::presets_dir());
                let p = store.find(name).ok_or_else(|| anyhow!("no preset named '{name}' (see `prism presets`)"))?;
                let mut s = JobSettings::default();
                p.apply(&mut s);
                s
            }
            (None, None) => JobSettings::default(),
        };
        let e = &mut s.enhancement;
        if let Some(r) = &self.upscale {
            s.output.resolution = Resolution::parse(r).ok_or_else(|| anyhow!("unrecognised resolution '{r}'"))?;
            e.enabled = true;
        }
        if let Some(m) = &self.model {
            e.model = m.clone();
            e.enabled = true;
        }
        if self.enhance {
            e.enabled = true;
        }
        if let Some(m) = &self.mode {
            e.param_mode = match m.as_str() {
                "auto" => ParamMode::Auto,
                "relative" | "relative-to-auto" => ParamMode::RelativeToAuto,
                "manual" => ParamMode::Manual,
                _ => bail!("mode must be auto, relative or manual"),
            };
        }
        let sliders = [
            (self.compression, &mut e.compression),
            (self.details, &mut e.details),
            (self.sharpen, &mut e.sharpen),
            (self.denoise, &mut e.noise),
            (self.dehalo, &mut e.dehalo),
            (self.deblur, &mut e.deblur),
            (self.add_noise, &mut e.add_noise),
            (self.recover, &mut e.recover_original),
        ];
        let mut any_slider = false;
        for (v, slot) in sliders {
            if let Some(v) = v {
                *slot = v;
                any_slider = true;
            }
        }
        // Explicit slider values without a mode mean "use these values".
        if any_slider && self.mode.is_none() && e.param_mode == ParamMode::Auto {
            e.param_mode = ParamMode::Manual;
        }
        if let Some(f) = &self.focus_fix {
            e.focus_fix = match f.as_str() {
                "off" => FocusFix::Off,
                "normal" => FocusFix::Normal,
                "strong" => FocusFix::Strong,
                _ => bail!("focus fix must be off, normal or strong"),
            };
        }
        if let Some(m) = &self.second_model {
            s.second_enhancement.enabled = true;
            s.second_enhancement.settings.enabled = true;
            s.second_enhancement.settings.model = m.clone();
        }
        if let Some(i) = &self.intermediate {
            s.second_enhancement.intermediate = match i.as_str() {
                "auto" => IntermediateScale::Auto,
                "1x" => IntermediateScale::X1,
                "2x" => IntermediateScale::X2,
                "4x" => IntermediateScale::X4,
                _ => bail!("intermediate must be auto, 1x, 2x or 4x"),
            };
        }
        if let Some(r) = &self.fit {
            s.output.fit = match r.as_str() {
                "fit" => FitMode::Fit,
                "crop" => FitMode::Crop,
                "pad" => FitMode::Pad,
                "stretch" => FitMode::Stretch,
                _ => bail!("fit must be fit, crop, pad or stretch"),
            };
        }
        let fi = &mut s.interpolation;
        if let Some(f) = self.fps {
            fi.enabled = true;
            fi.output_fps = Some(f);
        }
        if let Some(m) = self.slowmo {
            fi.enabled = true;
            fi.slowmo = m;
        }
        if let Some(m) = &self.fi_model {
            fi.enabled = true;
            fi.model = m.clone();
        }
        if self.no_dedup {
            fi.replace_duplicates = false;
        }
        if let Some(t) = self.dup_threshold {
            fi.duplicate_threshold = t;
        }
        if let Some(r) = self.retime {
            s.output.fps = Some(r);
        }
        if let Some(m) = &self.stabilize {
            s.stabilization.enabled = true;
            s.stabilization.mode = match m.as_str() {
                "autocrop" | "auto" => StabilizationMode::AutoCrop,
                "fullframe" | "full" => StabilizationMode::FullFrame,
                _ => bail!("stabilize must be autocrop or fullframe"),
            };
        }
        if let Some(v) = self.smoothness {
            s.stabilization.smoothness = v;
        }
        if let Some(m) = &self.deinterlace {
            s.deinterlace.enabled = true;
            s.deinterlace.mode = match m.as_str() {
                "same" => DeinterlaceMode::SameRate,
                "double" => DeinterlaceMode::DoubleRate,
                "telecine" | "ivtc" => DeinterlaceMode::Telecine,
                _ => bail!("deinterlace must be same, double or telecine"),
            };
        }
        if let Some(f) = &self.field_order {
            let o = match f.as_str() {
                "auto" => FieldOrderSetting::Auto,
                "tff" => FieldOrderSetting::TopFirst,
                "bff" => FieldOrderSetting::BottomFirst,
                _ => bail!("field order must be auto, tff or bff"),
            };
            s.deinterlace.field_order = o;
            s.enhancement.field_order = o;
        }
        if let Some(v) = self.motion_deblur {
            s.motion_deblur.enabled = v > 0;
            s.motion_deblur.strength = v;
        }
        if let Some(t) = &self.hdr {
            s.hdr.enabled = true;
            s.hdr.transfer = match t.as_str() {
                "pq" | "hdr10" => HdrTransfer::Pq,
                "hlg" => HdrTransfer::Hlg,
                _ => bail!("hdr must be pq or hlg"),
            };
        }
        if let Some(p) = self.hdr_peak {
            s.hdr.peak_nits = p;
        }
        if let Some(g) = self.grain {
            s.grain.enabled = g > 0;
            s.grain.amount = g;
        }
        if let Some(g) = self.grain_size {
            s.grain.size = g;
        }
        if let Some(t) = &self.trim {
            let (a, b) = t.split_once('-').ok_or_else(|| anyhow!("trim must look like START-END"))?;
            let start = timecode::parse(a, fps_hint).ok_or_else(|| anyhow!("bad trim start '{a}'"))?;
            let end = timecode::parse(b, fps_hint).ok_or_else(|| anyhow!("bad trim end '{b}'"))?;
            if end <= start {
                bail!("trim end must be after its start");
            }
            s.trim = Some(TrimRange { start, end });
        }
        if let Some(c) = &self.crop {
            s.crop = Some(parse_crop(c)?);
        }
        if let Some(r) = self.rotate {
            s.rotation = match r {
                0 => 0,
                90 => 1,
                180 => 2,
                270 => 3,
                _ => bail!("rotate must be 90, 180 or 270"),
            };
        }
        if let Some(l) = &self.lut {
            s.export.lut = Some(l.clone());
        }
        let x = &mut s.export;
        if let Some(enc) = &self.encoder {
            if prism_core::encoders::find(enc).is_none() {
                bail!("unknown encoder '{enc}' (see `prism encoders`)");
            }
            x.encoder = enc.clone();
        }
        if let Some(q) = &self.quality {
            x.rate = RateControl::Quality(match q.as_str() {
                "high" => Quality::High,
                "medium" | "mid" => Quality::Medium,
                "low" => Quality::Low,
                _ => bail!("quality must be high, medium or low"),
            });
        }
        if let Some(b) = self.bitrate {
            x.rate = RateControl::Bitrate(b);
        }
        if let Some(c) = &self.container {
            x.container = c.trim_start_matches('.').to_string();
        }
        if let Some(a) = &self.audio {
            let (codec, kbps) = a.split_once(':').map(|(c, k)| (c, k.parse::<u32>().ok())).unwrap_or((a.as_str(), None));
            match codec {
                "copy" => x.audio.mode = AudioMode::Copy,
                "none" | "off" => x.audio.mode = AudioMode::None,
                c => {
                    if prism_core::encoders::audio_codec(c).is_none() {
                        bail!("unknown audio codec '{c}'");
                    }
                    x.audio.mode = AudioMode::Convert;
                    x.audio.codec = c.into();
                    x.audio.bitrate_kbps = kbps.unwrap_or(0);
                }
            }
        }
        if let Some(o) = &self.output {
            x.output_path = Some(o.clone());
            if let Some(ext) = o.extension().and_then(|e| e.to_str()) {
                if self.container.is_none() {
                    x.container = ext.to_ascii_lowercase();
                }
            }
        }
        if let Some(d) = &self.output_dir {
            x.output_dir = Some(d.clone());
        }
        if self.overwrite {
            x.overwrite = true;
        }
        let d = &mut s.device;
        if let Some(dev) = &self.device {
            let (kind, idx) = dev.split_once(':').map(|(k, i)| (k, i.parse::<i32>().ok())).unwrap_or((dev.as_str(), None));
            d.backend = match kind {
                "auto" => Backend::Auto,
                "cpu" => Backend::Cpu,
                "gpu" if cfg!(target_os = "macos") => Backend::CoreMl,
                "gpu" if !cfg!(windows) => Backend::Cuda,
                "dml" | "directml" | "gpu" => Backend::DirectMl,
                "cuda" => Backend::Cuda,
                "coreml" => Backend::CoreMl,
                _ => bail!("device must be auto, cpu, gpu, dml[:N], cuda[:N] or coreml"),
            };
            if let Some(i) = idx {
                d.device_index = i;
            }
        }
        if let Some(v) = self.vram {
            d.max_memory = v;
        }
        if let Some(t) = self.tile {
            d.tile = Some(t);
        }
        if self.fp32 {
            d.fp16 = false;
        }
        if let Some(t) = self.threads {
            d.threads = t;
        }
        if self.hwdec {
            d.hw_decode = true;
        }
        s.sanitize();
        Ok(s)
    }
}
