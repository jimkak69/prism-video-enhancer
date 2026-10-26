//! Job settings: the complete state of every filter for one output.
//!
//! This is the serialisable equivalent of the reference product's filter-controller state. Presets,
//! project tabs and CLI flags all produce a [`JobSettings`]; the planner turns it into commands.
//! Slider values use the same 0..100 conventions the UI shows.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::models;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct JobSettings {
    pub trim: Option<TrimRange>,
    /// Clockwise quarter turns, 0..=3.
    pub rotation: u8,
    pub deinterlace: DeinterlaceSettings,
    pub correction: CorrectionSettings,
    pub crop: Option<CropRect>,
    pub stabilization: StabilizationSettings,
    pub motion_deblur: MotionDeblurSettings,
    pub interpolation: InterpolationSettings,
    pub enhancement: EnhancementSettings,
    pub second_enhancement: SecondEnhancementSettings,
    pub hdr: HdrSettings,
    pub grain: GrainSettings,
    pub output: OutputSettings,
    pub export: ExportSettings,
    pub device: DeviceSettings,
}

impl Default for JobSettings {
    fn default() -> Self {
        JobSettings {
            trim: None,
            rotation: 0,
            deinterlace: Default::default(),
            correction: Default::default(),
            crop: None,
            stabilization: Default::default(),
            motion_deblur: Default::default(),
            interpolation: Default::default(),
            enhancement: Default::default(),
            second_enhancement: Default::default(),
            hdr: Default::default(),
            grain: Default::default(),
            output: Default::default(),
            export: Default::default(),
            device: Default::default(),
        }
    }
}

impl JobSettings {
    /// True when any stage that needs the Rust frame pipeline is active.
    pub fn needs_engine(&self) -> bool {
        self.enhancement.enabled || self.second_enhancement.active() || self.interpolation.enabled || self.hdr.enabled || self.grain.active() || self.motion_deblur.enabled
    }

    /// Clamps every value into its documented range.
    pub fn sanitize(&mut self) {
        self.rotation %= 4;
        self.enhancement.sanitize();
        self.second_enhancement.settings.sanitize();
        self.interpolation.slowmo = self.interpolation.slowmo.clamp(1.0, 16.0);
        if let Some(f) = self.interpolation.output_fps {
            self.interpolation.output_fps = (f.is_finite() && f > 0.0).then_some(f.clamp(1.0, 480.0));
        }
        self.interpolation.duplicate_threshold = self.interpolation.duplicate_threshold.clamp(0.0, 100.0);
        self.stabilization.smoothness = self.stabilization.smoothness.clamp(0, 100);
        self.stabilization.reduce_jitter = self.stabilization.reduce_jitter.min(5);
        self.motion_deblur.strength = self.motion_deblur.strength.clamp(0, 100);
        self.hdr.sdr_ip = self.hdr.sdr_ip.clamp(0.45, 0.85);
        self.hdr.brightness = self.hdr.brightness.clamp(-100, 100);
        self.hdr.saturation = self.hdr.saturation.clamp(-100, 100);
        self.hdr.peak_nits = self.hdr.peak_nits.clamp(400.0, 10000.0);
        self.grain.amount = self.grain.amount.clamp(0, 100);
        self.grain.size = self.grain.size.clamp(0.0, 5.0);
        self.device.max_memory = self.device.max_memory.clamp(0.1, 1.0);
        if let Some(t) = &mut self.trim {
            t.start = t.start.max(0.0);
            t.end = t.end.max(t.start);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TrimRange {
    pub start: f64,
    pub end: f64,
}

impl TrimRange {
    pub fn duration(&self) -> f64 {
        (self.end - self.start).max(0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CropRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FieldOrderSetting {
    #[default]
    Auto,
    TopFirst,
    BottomFirst,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum DeinterlaceMode {
    /// One frame per frame (keeps the frame rate).
    #[default]
    SameRate,
    /// One frame per field (doubles the frame rate).
    DoubleRate,
    /// Inverse telecine: field matching, deinterlace leftovers, drop duplicates.
    Telecine,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct DeinterlaceSettings {
    pub enabled: bool,
    pub mode: DeinterlaceMode,
    pub field_order: FieldOrderSetting,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct CorrectionSettings {
    /// Resample non-square pixels to square pixels.
    pub square_pixels: bool,
    /// Override the source colour description (FFmpeg names), e.g. when tags are missing.
    pub color_override: Option<ColorOverride>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorOverride {
    pub range: String,
    pub primaries: String,
    pub transfer: String,
    pub matrix: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StabilizationMode {
    /// Zoom in just enough to hide moving borders.
    #[default]
    AutoCrop,
    /// Keep the full frame and fill exposed borders from neighbouring frames.
    FullFrame,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StabilizationSettings {
    pub enabled: bool,
    pub mode: StabilizationMode,
    /// 0..100; higher follows the camera path less closely.
    pub smoothness: i32,
    /// Rolling-shutter correction (not supported by libvidstab; reported as a plan warning).
    pub rolling_shutter: bool,
    /// 0..5 extra jitter-reduction iterations.
    pub reduce_jitter: u8,
}

impl Default for StabilizationSettings {
    fn default() -> Self {
        StabilizationSettings { enabled: false, mode: StabilizationMode::AutoCrop, smoothness: 50, rolling_shutter: false, reduce_jitter: 0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MotionDeblurSettings {
    pub enabled: bool,
    pub strength: i32,
}

impl Default for MotionDeblurSettings {
    fn default() -> Self {
        MotionDeblurSettings { enabled: false, strength: 50 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct InterpolationSettings {
    pub enabled: bool,
    pub model: String,
    /// Slow-motion factor (1 = real time), 1..16.
    pub slowmo: f64,
    /// Output frame rate; `None` keeps the input rate.
    pub output_fps: Option<f64>,
    /// Replace runs of duplicated frames with interpolated ones.
    pub replace_duplicates: bool,
    /// Duplicate sensitivity 0..100 (10 = 1% mean difference).
    pub duplicate_threshold: f64,
    /// Do not interpolate across scene cuts.
    pub scene_detection: bool,
}

impl Default for InterpolationSettings {
    fn default() -> Self {
        InterpolationSettings {
            enabled: false,
            model: models::DEFAULT_INTERPOLATION.into(),
            slowmo: 1.0,
            output_fps: None,
            replace_duplicates: true,
            duplicate_threshold: 10.0,
            scene_detection: true,
        }
    }
}

impl InterpolationSettings {
    /// Mean-absolute-difference threshold below which two frames count as duplicates.
    pub fn duplicate_mad(&self) -> f32 {
        if self.replace_duplicates { (self.duplicate_threshold / 1000.0) as f32 } else { -1.0 }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum VideoType {
    #[default]
    Progressive,
    Interlaced,
    InterlacedProgressive,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ParamMode {
    /// Values come from the estimator.
    #[default]
    Auto,
    /// Slider values are offsets added to the estimate.
    RelativeToAuto,
    /// Slider values are used as is.
    Manual,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FocusFix {
    #[default]
    Off,
    /// Downscale 2x before enhancement so the model re-synthesises soft detail.
    Normal,
    /// Downscale 4x before enhancement.
    Strong,
}

impl FocusFix {
    pub fn factor(self) -> u32 {
        match self {
            FocusFix::Off => 1,
            FocusFix::Normal => 2,
            FocusFix::Strong => 4,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EnhancementSettings {
    pub enabled: bool,
    pub model: String,
    pub video_type: VideoType,
    pub field_order: FieldOrderSetting,
    pub param_mode: ParamMode,
    /// Revert compression (deblock/dering), 0..100.
    pub compression: i32,
    /// Recover details (local contrast), 0..100.
    pub details: i32,
    /// Sharpen, 0..100.
    pub sharpen: i32,
    /// Reduce noise, 0..100.
    pub noise: i32,
    /// Dehalo (anti-ringing), 0..100.
    pub dehalo: i32,
    /// Anti-alias (negative) / deblur (positive), -100..100.
    pub deblur: i32,
    /// Noise added before the model to avoid plastic-looking output, 0..100.
    pub add_noise: i32,
    /// Blend back the plainly resampled source, 0..100.
    pub recover_original: i32,
    pub focus_fix: FocusFix,
}

impl Default for EnhancementSettings {
    fn default() -> Self {
        EnhancementSettings {
            enabled: false,
            model: models::DEFAULT_ENHANCEMENT.into(),
            video_type: VideoType::Progressive,
            field_order: FieldOrderSetting::Auto,
            param_mode: ParamMode::Auto,
            compression: 0,
            details: 0,
            sharpen: 0,
            noise: 0,
            dehalo: 0,
            deblur: 0,
            add_noise: 0,
            recover_original: 20,
            focus_fix: FocusFix::Off,
        }
    }
}

impl EnhancementSettings {
    fn sanitize(&mut self) {
        let lo = if self.param_mode == ParamMode::RelativeToAuto { -100 } else { 0 };
        for v in [&mut self.compression, &mut self.details, &mut self.sharpen, &mut self.noise, &mut self.dehalo] {
            *v = (*v).clamp(lo, 100);
        }
        self.deblur = self.deblur.clamp(-100, 100);
        self.add_noise = self.add_noise.clamp(0, 100);
        self.recover_original = self.recover_original.clamp(0, 100);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum IntermediateScale {
    #[default]
    Auto,
    X1,
    X2,
    X4,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct SecondEnhancementSettings {
    pub enabled: bool,
    pub intermediate: IntermediateScale,
    pub settings: EnhancementSettings,
}

impl SecondEnhancementSettings {
    pub fn active(&self) -> bool {
        self.enabled && self.settings.enabled
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HdrTransfer {
    /// HDR10 (SMPTE ST 2084).
    #[default]
    Pq,
    /// Hybrid Log-Gamma (ARIB STD-B67).
    Hlg,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HdrSettings {
    pub enabled: bool,
    pub transfer: HdrTransfer,
    /// SDR inflection point (0.45..0.85): signal level above which highlights are expanded.
    pub sdr_ip: f64,
    /// Overall HDR brightness adjustment, -100..100.
    pub brightness: i32,
    /// Saturation adjustment, -100..100.
    pub saturation: i32,
    /// Target mastering peak in nits.
    pub peak_nits: f64,
}

impl Default for HdrSettings {
    fn default() -> Self {
        HdrSettings { enabled: false, transfer: HdrTransfer::Pq, sdr_ip: 0.65, brightness: 0, saturation: 0, peak_nits: 1000.0 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GrainSettings {
    pub enabled: bool,
    /// 0..100.
    pub amount: i32,
    /// Grain size 0..5 (0 = per-pixel noise).
    pub size: f64,
}

impl Default for GrainSettings {
    fn default() -> Self {
        GrainSettings { enabled: false, amount: 20, size: 1.0 }
    }
}

impl GrainSettings {
    pub fn active(&self) -> bool {
        self.enabled && self.amount > 0
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum Resolution {
    #[default]
    Original,
    /// Multiply both dimensions.
    Scale(f64),
    /// Fit into a named or custom box (see [`FitMode`]).
    Box { width: u32, height: u32 },
    /// Target height; width follows the display aspect ratio.
    Height(u32),
}

impl Resolution {
    pub fn hd() -> Self {
        Resolution::Box { width: 1280, height: 720 }
    }
    pub fn fhd() -> Self {
        Resolution::Box { width: 1920, height: 1080 }
    }
    pub fn uhd4k() -> Self {
        Resolution::Box { width: 3840, height: 2160 }
    }
    pub fn uhd8k() -> Self {
        Resolution::Box { width: 7680, height: 4320 }
    }

    /// Parses `2x`, `4x`, `1080p`, `4k`, `8k`, `1920x1080`, `original`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim().to_ascii_lowercase();
        match s.as_str() {
            "original" | "orig" | "1x" => return Some(Resolution::Original),
            "hd" | "720p" => return Some(Resolution::hd()),
            "fhd" | "1080p" => return Some(Resolution::fhd()),
            "4k" | "uhd" | "2160p" => return Some(Resolution::uhd4k()),
            "8k" | "4320p" => return Some(Resolution::uhd8k()),
            _ => {}
        }
        if let Some(f) = s.strip_suffix('x').and_then(|v| v.parse::<f64>().ok()) {
            return (f > 0.0 && f <= 16.0).then_some(Resolution::Scale(f));
        }
        if let Some(h) = s.strip_suffix('p').and_then(|v| v.parse::<u32>().ok()) {
            return (h >= 16).then_some(Resolution::Height(h));
        }
        if let Some((w, h)) = s.split_once('x') {
            let (w, h) = (w.parse::<u32>().ok()?, h.parse::<u32>().ok()?);
            return (w >= 16 && h >= 16).then_some(Resolution::Box { width: w, height: h });
        }
        None
    }

    pub fn label(&self) -> String {
        match self {
            Resolution::Original => "Original".into(),
            Resolution::Scale(f) => format!("{f}x"),
            Resolution::Box { width, height } => format!("{width}x{height}"),
            Resolution::Height(h) => format!("{h}p"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FitMode {
    /// Keep the aspect ratio; the output is as large as fits in the box (no bars, no cropping).
    #[default]
    Fit,
    /// Fill the box and crop the overflow.
    Crop,
    /// Fit inside the box and add black bars to reach it exactly.
    Pad,
    /// Stretch to the box.
    Stretch,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
#[serde(default)]
pub struct OutputSettings {
    pub resolution: Resolution,
    pub fit: FitMode,
    /// Frame-rate change without interpolation (drops or repeats frames).
    pub fps: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Quality {
    #[default]
    High,
    Medium,
    Low,
}

impl Quality {
    pub fn index(self) -> usize {
        match self {
            Quality::High => 0,
            Quality::Medium => 1,
            Quality::Low => 2,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum RateControl {
    /// Constant quality (CRF / CQ / QP) at the given level.
    Quality(Quality),
    /// Target bitrate in Mbit/s.
    Bitrate(f64),
}

impl Default for RateControl {
    fn default() -> Self {
        RateControl::Quality(Quality::High)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AudioMode {
    None,
    #[default]
    Copy,
    Convert,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioSettings {
    pub mode: AudioMode,
    /// Codec id from [`crate::encoders::audio_codecs`] when converting.
    pub codec: String,
    pub bitrate_kbps: u32,
}

impl Default for AudioSettings {
    fn default() -> Self {
        AudioSettings { mode: AudioMode::Copy, codec: "aac".into(), bitrate_kbps: 320 }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExportSettings {
    /// Encoder id from [`crate::encoders::video_encoders`].
    pub encoder: String,
    pub rate: RateControl,
    /// Container extension (`mp4`, `mkv`, `mov`, ...). Empty selects the encoder's first.
    pub container: String,
    pub audio: AudioSettings,
    /// Output folder; `None` writes next to the input.
    pub output_dir: Option<PathBuf>,
    /// Exact output path (overrides naming).
    pub output_path: Option<PathBuf>,
    pub overwrite: bool,
    /// 3D LUT applied first in the chain (`.cube`, `.3dl`, ...).
    pub lut: Option<PathBuf>,
    /// Copy subtitles when the container supports it.
    pub subtitles: bool,
    /// Frame rate for image-sequence inputs/outputs.
    pub image_sequence_fps: f64,
}

impl Default for ExportSettings {
    fn default() -> Self {
        ExportSettings {
            encoder: "h264-x264".into(),
            rate: RateControl::default(),
            container: String::new(),
            audio: AudioSettings::default(),
            output_dir: None,
            output_path: None,
            overwrite: false,
            lut: None,
            subtitles: true,
            image_sequence_fps: 25.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Backend {
    /// GPU when an ONNX Runtime with a GPU provider is available, otherwise CPU.
    #[default]
    Auto,
    Cpu,
    DirectMl,
    Cuda,
    /// Core ML (GPU and Neural Engine) on macOS.
    CoreMl,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeviceSettings {
    pub backend: Backend,
    /// Adapter index for GPU backends; -1 picks the default adapter.
    pub device_index: i32,
    /// Fraction of GPU memory a tile may use (0.1..1), like the reference `vram` option.
    pub max_memory: f64,
    /// Fixed tile edge in input pixels; `None` lets the tiler choose.
    pub tile: Option<u32>,
    /// Prefer FP16 model files when present.
    pub fp16: bool,
    /// CPU worker threads for native kernels (0 = all cores).
    pub threads: usize,
    /// Let FFmpeg use hardware decoding.
    pub hw_decode: bool,
}

impl Default for DeviceSettings {
    fn default() -> Self {
        DeviceSettings { backend: Backend::Auto, device_index: -1, max_memory: 0.7, tile: None, fp16: true, threads: 0, hw_decode: false }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolution_parsing() {
        assert_eq!(Resolution::parse("4k"), Some(Resolution::uhd4k()));
        assert_eq!(Resolution::parse("2x"), Some(Resolution::Scale(2.0)));
        assert_eq!(Resolution::parse("720p"), Some(Resolution::hd()));
        assert_eq!(Resolution::parse("1440p"), Some(Resolution::Height(1440)));
        assert_eq!(Resolution::parse("2560x1440"), Some(Resolution::Box { width: 2560, height: 1440 }));
        assert_eq!(Resolution::parse("abc"), None);
    }

    #[test]
    fn settings_round_trip_and_forward_compat() {
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        s.interpolation.slowmo = 4.0;
        let json = serde_json::to_string(&s).unwrap();
        let back: JobSettings = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
        // Unknown and missing fields must not break older/newer files.
        let partial: JobSettings = serde_json::from_str(r#"{"enhancement":{"enabled":true,"future":1}}"#).unwrap();
        assert!(partial.enhancement.enabled);
        assert_eq!(partial.enhancement.recover_original, 20);
    }

    #[test]
    fn sanitize_clamps() {
        let mut s = JobSettings::default();
        s.enhancement.noise = 500;
        s.enhancement.details = -40;
        s.interpolation.slowmo = 64.0;
        s.rotation = 7;
        s.sanitize();
        assert_eq!(s.enhancement.noise, 100);
        assert_eq!(s.enhancement.details, 0);
        assert_eq!(s.interpolation.slowmo, 16.0);
        assert_eq!(s.rotation, 3);
    }
}
