//! Job planning: media facts + [`JobSettings`] → an ordered list of passes.
//!
//! Filter order follows the reference product:
//! `Trim → Rotation → Deinterlace → Correction → Crop → Stabilization → MotionDeblur →
//!  FrameInterpolation → Enhancement → SecondEnhancement → HDR → Grain → VideoSettings → Export`.
//!
//! Work up to stabilization runs inside the decoding FFmpeg process, the AI/image stages run in the
//! Rust engine, and output sizing, colour tagging and encoding run inside the encoding FFmpeg process.
//! When no Rust stage is active the whole job is a single FFmpeg command. Stabilization adds an
//! analysis pass that runs first.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Result, anyhow, bail};
use serde::{Deserialize, Serialize};

use crate::encoders::{self, VideoEncoder};
use crate::media::{Chroma, ColorInfo, ColorRange, Matrix, Rational, VideoInfo};
use crate::models::{Catalog, ModelKind};
use crate::naming;
use crate::settings::*;

/// Name of the libvidstab transform file inside the job's temporary directory.
pub const STAB_FILE: &str = "stab.trf";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RawLayout {
    /// Planar Y, U, V with 2x2 chroma subsampling.
    Yuv420,
    /// Planar G, B, R (FFmpeg `gbrp*`).
    Rgb,
}

/// Raw video exchanged with FFmpeg over a pipe.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RawFormat {
    pub width: u32,
    pub height: u32,
    pub layout: RawLayout,
    /// 8, 10 or 16 significant bits; anything above 8 travels as little-endian u16.
    pub bits: u8,
    pub matrix: Matrix,
    pub range: ColorRange,
}

impl RawFormat {
    pub fn pix_fmt(&self) -> &'static str {
        match (self.layout, self.bits) {
            (RawLayout::Yuv420, 8) => "yuv420p",
            (RawLayout::Yuv420, 10) => "yuv420p10le",
            (RawLayout::Yuv420, _) => "yuv420p16le",
            (RawLayout::Rgb, 8) => "gbrp",
            (RawLayout::Rgb, _) => "gbrp16le",
        }
    }

    pub fn bytes_per_sample(&self) -> usize {
        if self.bits > 8 { 2 } else { 1 }
    }

    pub fn chroma_size(&self) -> (usize, usize) {
        (self.width.div_ceil(2) as usize, self.height.div_ceil(2) as usize)
    }

    pub fn frame_bytes(&self) -> usize {
        let px = self.width as usize * self.height as usize;
        let samples = match self.layout {
            RawLayout::Yuv420 => {
                let (cw, ch) = self.chroma_size();
                px + 2 * cw * ch
            }
            RawLayout::Rgb => 3 * px,
        };
        samples * self.bytes_per_sample()
    }
}

/// Slider values normalised to the engine's ranges (0..1, deblur -1..1).
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct EnhanceParams {
    pub compression: f32,
    pub details: f32,
    pub sharpen: f32,
    pub noise: f32,
    pub dehalo: f32,
    pub deblur: f32,
    pub add_noise: f32,
    pub recover_original: f32,
}

impl EnhanceParams {
    fn from_settings(e: &EnhancementSettings) -> Self {
        let f = |v: i32| v as f32 / 100.0;
        EnhanceParams {
            compression: f(e.compression),
            details: f(e.details),
            sharpen: f(e.sharpen),
            noise: f(e.noise),
            dehalo: f(e.dehalo),
            deblur: f(e.deblur),
            add_noise: f(e.add_noise),
            recover_original: f(e.recover_original),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnhanceSpec {
    pub model: String,
    pub in_width: u32,
    pub in_height: u32,
    pub out_width: u32,
    pub out_height: u32,
    /// Focus fix: shrink the input by this factor before the model runs.
    pub pre_downscale: u32,
    pub params: EnhanceParams,
    pub mode: ParamMode,
    /// Frames sampled by the estimator (0 in manual mode).
    pub estimate_frames: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterpolateSpec {
    pub model: String,
    pub in_fps: Rational,
    pub out_fps: Rational,
    pub slowmo: f64,
    /// Mean absolute difference below which frames are duplicates; negative disables.
    pub duplicate_mad: f32,
    pub scene_detection: bool,
    pub frames_in: u64,
    pub frames_out: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HdrSpec {
    pub transfer: HdrTransfer,
    pub sdr_ip: f32,
    pub brightness: f32,
    pub saturation: f32,
    pub peak_nits: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum StageSpec {
    MotionDeblur { strength: f32 },
    Interpolate(InterpolateSpec),
    Enhance(EnhanceSpec),
    Hdr(HdrSpec),
    Grain { amount: f32, size: f32 },
}

impl StageSpec {
    pub fn label(&self) -> String {
        match self {
            StageSpec::MotionDeblur { .. } => "motion deblur".into(),
            StageSpec::Interpolate(i) => format!("interpolate ({})", i.model),
            StageSpec::Enhance(e) => format!("enhance ({}) {}x{} -> {}x{}", e.model, e.in_width, e.in_height, e.out_width, e.out_height),
            StageSpec::Hdr(_) => "sdr to hdr".into(),
            StageSpec::Grain { .. } => "grain".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FfmpegPass {
    pub label: String,
    pub args: Vec<String>,
    pub frames: u64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EnginePass {
    pub label: String,
    /// FFmpeg arguments that write `input` frames to stdout.
    pub decode: Vec<String>,
    pub input: RawFormat,
    pub stages: Vec<StageSpec>,
    /// FFmpeg arguments that read `output` frames from stdin.
    pub encode: Vec<String>,
    pub output: RawFormat,
    pub frames_in: u64,
    pub frames_out: u64,
    pub fps_in: Rational,
    pub fps_out: Rational,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Pass {
    Ffmpeg(FfmpegPass),
    Engine(EnginePass),
}

impl Pass {
    pub fn label(&self) -> &str {
        match self {
            Pass::Ffmpeg(p) => &p.label,
            Pass::Engine(p) => &p.label,
        }
    }

    pub fn frames(&self) -> u64 {
        match self {
            Pass::Ffmpeg(p) => p.frames,
            Pass::Engine(p) => p.frames_out,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Geometry {
    /// Decoded source size (after display-matrix rotation).
    pub source: (u32, u32),
    /// Size entering the Rust stages (after rotation, square pixels and crop).
    pub process: (u32, u32),
    /// Picture size before padding/cropping to the output box.
    pub content: (u32, u32),
    /// Encoded size.
    pub output: (u32, u32),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct JobPlan {
    pub input: PathBuf,
    pub output: PathBuf,
    pub output_is_sequence: bool,
    pub temp_dir: PathBuf,
    pub passes: Vec<Pass>,
    pub description: String,
    pub prefix: String,
    pub warnings: Vec<String>,
    pub geometry: Geometry,
    pub fps_in: Rational,
    pub fps_out: Rational,
    pub frames_in: u64,
    pub frames_out: u64,
    /// Seconds of source covered by the job.
    pub source_duration: f64,
    pub device: DeviceSettings,
    pub models_dir: PathBuf,
    pub is_preview: bool,
    pub encoder: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PreviewRange {
    pub start: f64,
    pub duration: f64,
}

pub struct PlanContext<'a> {
    pub catalog: &'a Catalog,
    /// Per-job scratch directory (created by the runner). FFmpeg runs with this as its cwd.
    pub temp_dir: PathBuf,
    /// FFmpeg encoder names present in the build; `None` skips the check.
    pub encoders_available: Option<&'a HashSet<String>>,
    pub preview: Option<PreviewRange>,
    pub exists: &'a dyn Fn(&Path) -> bool,
}

/// Encoder used for preview renders: fast, visually lossless, decodable while growing.
fn preview_encoder(high_bit_depth: bool) -> &'static VideoEncoder {
    static X264: std::sync::OnceLock<VideoEncoder> = std::sync::OnceLock::new();
    if high_bit_depth {
        return encoders::find("ffv1-10").expect("ffv1-10 in table");
    }
    X264.get_or_init(|| VideoEncoder {
        id: "preview",
        family: "H.264",
        profile: "Preview",
        codec: "libx264",
        args: &["-preset", "ultrafast", "-tune", "fastdecode", "-g", "12"],
        bitrate_args: &[],
        rate: encoders::RateSpec::Quality { flag: "-crf", values: [12, 12, 12] },
        ..encoders::find("h264-x264").expect("x264 in table").clone()
    })
}

pub fn plan(info: &VideoInfo, settings: &JobSettings, ctx: &PlanContext) -> Result<JobPlan> {
    let mut s = settings.clone();
    s.sanitize();
    let mut warnings = Vec::new();
    let input = std::path::absolute(&info.path).unwrap_or_else(|_| info.path.clone());
    let is_preview = ctx.preview.is_some();

    // ---- time range
    let range = match ctx.preview {
        Some(p) => Some(TrimRange { start: p.start.max(0.0), end: p.start.max(0.0) + p.duration.max(0.0) }),
        None => s.trim,
    };
    let range = match range {
        Some(r) => {
            let end = if info.duration > 0.0 { r.end.min(info.duration) } else { r.end };
            if end - r.start <= 0.0 {
                bail!("time range {:.3}s..{:.3}s is outside the video (duration {:.3}s)", r.start, r.end, info.duration);
            }
            Some(TrimRange { start: r.start, end })
        }
        None => None,
    };
    let mut frames: u64 = match range {
        Some(r) => info.frames_in(r.start, r.end).max(1),
        None => info.frame_count.max(1),
    };
    let source_duration = range.map(|r| r.duration()).unwrap_or(info.duration);
    let mut fps = info.fps;

    // ---- geometry and decode-side filters
    let (mut w, mut h) = (info.width, info.height);
    let mut sar = info.sar;
    if info.rotation.rem_euclid(180) == 90 {
        (w, h) = (h, w);
        sar = Rational::new(sar.den, sar.num);
    }
    let source = (w, h);
    let mut pre: Vec<String> = Vec::new();
    match s.rotation {
        1 => pre.push("transpose=clock".into()),
        2 => pre.push("hflip,vflip".into()),
        3 => pre.push("transpose=cclock".into()),
        _ => {}
    }
    if s.rotation % 2 == 1 {
        (w, h) = (h, w);
        sar = Rational::new(sar.den, sar.num);
    }

    let deinterlace = s.deinterlace.enabled || (s.enhancement.enabled && s.enhancement.video_type == VideoType::Interlaced);
    if deinterlace {
        let order = if s.deinterlace.enabled { s.deinterlace.field_order } else { s.enhancement.field_order };
        let parity = match order {
            FieldOrderSetting::TopFirst => "tff",
            FieldOrderSetting::BottomFirst => "bff",
            FieldOrderSetting::Auto => match info.field_order {
                crate::media::FieldOrder::TopFirst => "tff",
                crate::media::FieldOrder::BottomFirst => "bff",
                _ => "auto",
            },
        };
        match s.deinterlace.mode {
            DeinterlaceMode::SameRate => pre.push(format!("bwdif=mode=send_frame:parity={parity}:deint=all")),
            DeinterlaceMode::DoubleRate => {
                pre.push(format!("bwdif=mode=send_field:parity={parity}:deint=all"));
                fps = fps.mul(Rational::new(2, 1));
                frames *= 2;
            }
            DeinterlaceMode::Telecine => {
                let order = if parity == "auto" { "auto" } else { parity };
                pre.push(format!("fieldmatch=order={order}:combmatch=full,bwdif=mode=send_frame:parity={parity}:deint=interlaced,decimate"));
                fps = fps.mul(Rational::new(4, 5));
                frames = (frames * 4).div_ceil(5);
            }
        }
    }

    let mut src_color = info.color.clone();
    if let Some(o) = &s.correction.color_override {
        pre.push(format!("setparams=range={}:color_primaries={}:color_trc={}:colorspace={}", o.range, o.primaries, o.transfer, o.matrix));
        src_color = ColorInfo {
            range: if matches!(o.range.as_str(), "pc" | "full" | "jpeg") { ColorRange::Full } else { ColorRange::Limited },
            matrix: match o.matrix.as_str() {
                "bt2020nc" | "bt2020c" => Matrix::Bt2020Nc,
                "smpte170m" | "bt470bg" | "bt601" => Matrix::Bt601,
                _ => Matrix::Bt709,
            },
            primaries: o.primaries.clone(),
            transfer: o.transfer.clone(),
            inferred: false,
        };
    }

    let resizes = !matches!(s.output.resolution, Resolution::Original);
    if sar != Rational::ONE && (s.correction.square_pixels || resizes) {
        let nw = even(w as f64 * sar.as_f64());
        pre.push(format!("scale=w={nw}:h={h}:flags=lanczos,setsar=1"));
        w = nw;
        sar = Rational::ONE;
    }

    if let Some(c) = s.crop {
        let (cx, cy, cw, ch) = (c.x & !1, c.y & !1, c.width & !1, c.height & !1);
        if cw < 16 || ch < 16 || cx + cw > w || cy + ch > h {
            bail!("crop {}x{}+{}+{} does not fit the {}x{} frame", c.width, c.height, c.x, c.y, w, h);
        }
        if (cw, ch) != (w, h) {
            pre.push(format!("crop=w={cw}:h={ch}:x={cx}:y={cy}"));
            (w, h) = (cw, ch);
        }
    }

    let lut = match &s.export.lut {
        Some(p) => Some(format!("lut3d=file={}", escape_filter_path(p)?)),
        None => None,
    };

    let mut stab_transform = None;
    let mut detect_chain = None;
    if s.stabilization.enabled {
        let st = &s.stabilization;
        let shakiness = (5 + st.reduce_jitter as u32).min(10);
        let mut chain = pre.clone();
        chain.push(format!("vidstabdetect=shakiness={shakiness}:accuracy=15:stepsize=6:result={STAB_FILE}"));
        detect_chain = Some(chain.join(","));
        let smoothing = 1 + (st.smoothness as f64 * 0.5).round() as u32;
        stab_transform = Some(match st.mode {
            StabilizationMode::AutoCrop => format!("vidstabtransform=input={STAB_FILE}:smoothing={smoothing}:optzoom=1:zoomspeed=0.25:crop=black:interpol=bicubic,unsharp=5:5:0.5:3:3:0.25"),
            StabilizationMode::FullFrame => format!("vidstabtransform=input={STAB_FILE}:smoothing={smoothing}:optzoom=0:crop=keep:interpol=bicubic,unsharp=5:5:0.5:3:3:0.25"),
        });
        if st.rolling_shutter {
            warnings.push("Rolling-shutter correction is not available with libvidstab; ignored.".into());
        }
    }
    let process = (w, h);
    let frames_stab = frames;

    // ---- frame rate
    let interpolating = s.interpolation.enabled;
    let mut decode_fps = None;
    let mut encode_fps = None;
    if !interpolating {
        if let Some(target) = s.output.fps {
            let fo = Rational::from_f64(target);
            let ratio = fo.as_f64() / fps.as_f64();
            if (ratio - 1.0).abs() > 1e-6 {
                let filter = format!("fps={}", fo.to_ffmpeg());
                if ratio < 1.0 { decode_fps = Some(filter) } else { encode_fps = Some(filter) }
            }
        }
    }
    if let Some(f) = &decode_fps {
        let fo = Rational::parse(f.trim_start_matches("fps=")).unwrap();
        frames = (frames as f64 * fo.as_f64() / fps.as_f64()).round().max(1.0) as u64;
        fps = fo;
    }
    let fps_engine_in = fps;
    let frames_engine_in = frames;

    // ---- output geometry
    let (content, output) = output_geometry(process, &s.output);

    // ---- encoder and container
    let hdr_out = s.hdr.enabled;
    let hdr_pass = !hdr_out && src_color.is_hdr();
    let mut enc: &'static VideoEncoder = if is_preview {
        preview_encoder(hdr_out || hdr_pass)
    } else {
        encoders::find(&s.export.encoder).ok_or_else(|| anyhow!("unknown encoder '{}'", s.export.encoder))?
    };
    if (hdr_out || hdr_pass) && enc.bit_depth < 10 {
        match encoders::deeper_variant(enc, 10) {
            Some(deeper) => {
                warnings.push(format!("HDR output needs at least 10 bits; using {} {} instead of {}.", deeper.family, deeper.profile, enc.profile));
                enc = deeper;
            }
            None => warnings.push(format!("{} {} is 8-bit; HDR output will band. Choose a 10-bit encoder.", enc.family, enc.profile)),
        }
    }
    if let Some(av) = ctx.encoders_available {
        if !av.contains(enc.codec) {
            bail!("encoder '{}' needs FFmpeg's {} which this FFmpeg build does not provide", enc.id, enc.codec);
        }
    }
    if output.0 > enc.max_size.0 || output.1 > enc.max_size.1 {
        bail!("{}x{} exceeds the {}x{} limit of {} {}", output.0, output.1, enc.max_size.0, enc.max_size.1, enc.family, enc.profile);
    }
    let container: String = if is_preview {
        "mkv".into()
    } else if s.export.container.is_empty() {
        enc.default_container().into()
    } else if enc.containers.contains(&s.export.container.as_str()) {
        s.export.container.clone()
    } else {
        warnings.push(format!("{} cannot be written to .{}; using .{}.", enc.family, s.export.container, enc.default_container()));
        enc.default_container().into()
    };

    // ---- colour
    let mut out_color = if hdr_out {
        match s.hdr.transfer {
            HdrTransfer::Pq => ColorInfo::hdr10(),
            HdrTransfer::Hlg => ColorInfo::hlg(),
        }
    } else {
        let mut c = src_color.clone();
        if c.matrix == Matrix::Bt601 && output.1 >= 720 {
            c.matrix = Matrix::Bt709;
        }
        c
    };
    if enc.pix_fmt.starts_with("yuvj") {
        out_color.range = ColorRange::Full;
    }

    // ---- engine stages
    let mut stages = Vec::new();
    let mut prefix_parts: Vec<String> = Vec::new();
    let mut info_parts: Vec<String> = Vec::new();
    if s.stabilization.enabled {
        prefix_parts.push("stab".into());
        info_parts.push(format!("Stabilized ({:?}, smoothness {})", s.stabilization.mode, s.stabilization.smoothness));
    }
    if s.motion_deblur.enabled {
        stages.push(StageSpec::MotionDeblur { strength: s.motion_deblur.strength as f32 / 100.0 });
        prefix_parts.push("dbl".into());
        info_parts.push(format!("motion deblur at {}", s.motion_deblur.strength));
    }
    // Frame blending is FFmpeg's `framerate` filter (8-bit SIMD) unless a Rust stage must run first.
    let mut ffmpeg_interp = None;
    if interpolating {
        let m = ctx.catalog.get(&s.interpolation.model).filter(|m| m.kind == ModelKind::Interpolation).ok_or_else(|| anyhow!("unknown interpolation model '{}'", s.interpolation.model))?;
        if !m.is_available(&ctx.catalog.dir) {
            bail!("interpolation model '{}' is not installed (expected {}.onnx in {})", m.id, m.id, ctx.catalog.dir.display());
        }
        let out_fps = s.interpolation.output_fps.map(Rational::from_f64).unwrap_or(fps);
        let frames_out = ((frames as f64) * s.interpolation.slowmo * out_fps.as_f64() / fps.as_f64()).round().max(1.0) as u64;
        if m.id == "prism-blend" && !s.motion_deblur.enabled {
            let mut chain = Vec::new();
            if s.interpolation.replace_duplicates {
                chain.push("mpdecimate".to_string());
            }
            if (s.interpolation.slowmo - 1.0).abs() > 1e-9 {
                chain.push(format!("setpts=PTS*{}", s.interpolation.slowmo));
            }
            let scene = if s.interpolation.scene_detection { "" } else { ":flags=0" };
            chain.push(format!("framerate=fps={}{scene}", out_fps.to_ffmpeg()));
            ffmpeg_interp = Some(chain.join(","));
        } else {
            stages.push(StageSpec::Interpolate(InterpolateSpec {
                model: m.id.clone(),
                in_fps: fps,
                out_fps,
                slowmo: s.interpolation.slowmo,
                duplicate_mad: s.interpolation.duplicate_mad(),
                scene_detection: s.interpolation.scene_detection,
                frames_in: frames,
                frames_out,
            }));
        }
        prefix_parts.push(m.prefix.clone());
        let mut d = match (s.interpolation.slowmo != 1.0, out_fps != fps) {
            (true, true) => format!("slow motion {}% and frame rate {} fps", s.interpolation.slowmo * 100.0, out_fps),
            (true, false) => format!("slow motion {}%", s.interpolation.slowmo * 100.0),
            (false, true) => format!("frame rate changed to {} fps", out_fps),
            (false, false) => "frames processed".into(),
        };
        d += &format!(" using {}", m.id);
        if s.interpolation.replace_duplicates {
            d += " replacing duplicate frames";
        }
        info_parts.push(d);
        frames = frames_out;
        fps = out_fps;
    }
    let (fps_engine_in, frames_engine_in) = if ffmpeg_interp.is_some() { (fps, frames) } else { (fps_engine_in, frames_engine_in) };

    let mut cur = process;
    // Classic with nothing but resizing to do is FFmpeg's Lanczos scaler (in the encode chain).
    let resize_only = |e: &EnhancementSettings| {
        e.model == crate::models::DEFAULT_ENHANCEMENT
            && e.param_mode == ParamMode::Manual
            && [e.compression, e.details, e.sharpen, e.noise, e.dehalo, e.deblur, e.add_noise].iter().all(|v| *v == 0)
            && e.focus_fix == FocusFix::Off
            && !s.grain.active()
    };
    let first = if s.enhancement.enabled { Some(&s.enhancement) } else if s.second_enhancement.active() { Some(&s.second_enhancement.settings) } else { None };
    let second = if s.enhancement.enabled && s.second_enhancement.active() { Some(&s.second_enhancement.settings) } else { None };
    if let Some(e1) = first {
        let target1 = match second {
            Some(_) => {
                let final_scale = (content.0 as f64 / process.0 as f64).max(content.1 as f64 / process.1 as f64);
                let k = match s.second_enhancement.intermediate {
                    IntermediateScale::Auto => {
                        if final_scale >= 8.0 {
                            4.0
                        } else if final_scale >= 4.0 {
                            2.0
                        } else {
                            1.0
                        }
                    }
                    IntermediateScale::X1 => 1.0,
                    IntermediateScale::X2 => 2.0,
                    IntermediateScale::X4 => 4.0,
                };
                let t = (even(process.0 as f64 * k), even(process.1 as f64 * k));
                if t.0 > content.0 || t.1 > content.1 { content } else { t }
            }
            None => content,
        };
        let skip_all = resize_only(e1) && second.is_none_or(resize_only);
        if !skip_all {
            stages.push(StageSpec::Enhance(enhance_spec(e1, cur, target1, ctx)?));
            cur = target1;
        }
        push_enhance_info(e1, ctx, &mut prefix_parts, &mut info_parts);
        if let Some(e2) = second {
            if !skip_all {
                stages.push(StageSpec::Enhance(enhance_spec(e2, cur, content, ctx)?));
                cur = content;
            }
            push_enhance_info(e2, ctx, &mut prefix_parts, &mut info_parts);
        }
    }
    if hdr_out {
        stages.push(StageSpec::Hdr(HdrSpec {
            transfer: s.hdr.transfer,
            sdr_ip: s.hdr.sdr_ip as f32,
            brightness: s.hdr.brightness as f32 / 100.0,
            saturation: s.hdr.saturation as f32 / 100.0,
            peak_nits: s.hdr.peak_nits as f32,
        }));
        prefix_parts.push("hdr".into());
        info_parts.push(format!("converted to HDR ({:?}, {} nits)", s.hdr.transfer, s.hdr.peak_nits));
        if src_color.is_hdr() {
            warnings.push("The source is already HDR; SDR-to-HDR expects an SDR source.".into());
        }
    }
    if s.grain.active() {
        stages.push(StageSpec::Grain { amount: s.grain.amount as f32 / 100.0 * 0.06, size: s.grain.size as f32 });
        info_parts.push(format!("grain {} size {}", s.grain.amount, s.grain.size));
    }
    if let Some(f) = &encode_fps {
        let fo = Rational::parse(f.trim_start_matches("fps=")).unwrap();
        frames = (frames as f64 * fo.as_f64() / fps.as_f64()).round().max(1.0) as u64;
        fps = fo;
    }
    if content != process {
        info_parts.push(format!("resized to {}x{}", output.0, output.1));
    }
    if info.has_alpha {
        warnings.push("The source has an alpha channel; it is not preserved.".into());
    }
    if info.vfr {
        warnings.push(format!("Variable frame rate source; frames are resampled to a constant {} fps.", info.fps));
    }

    let prefix = if prefix_parts.is_empty() { "prism".to_string() } else { prefix_parts.join("_") };
    let description = if info_parts.is_empty() { "Processed with Prism".to_string() } else { sentence(&info_parts) };

    // ---- output path
    let sequence = enc.image_sequence;
    let discard = container == "null";
    let output_path = if is_preview {
        ctx.temp_dir.join("preview.mkv")
    } else if discard {
        PathBuf::from("-")
    } else {
        let p = naming::output_path(&input, &s.export, &prefix, &container, sequence, ctx.exists);
        std::path::absolute(&p).unwrap_or(p)
    };

    // ---- audio
    let slowmo = interpolating && (s.interpolation.slowmo - 1.0).abs() > 1e-9;
    let audio = audio_plan(info, &s.export, &container, slowmo, is_preview || sequence || discard, &mut warnings);
    let subtitles = !is_preview && !sequence && s.export.subtitles && info.subtitles > 0 && container == "mkv";

    // ---- passes
    let mut passes = Vec::new();
    let range_args = |args: &mut Vec<String>| {
        if let Some(r) = range {
            args.extend(["-ss".into(), fmt_secs(r.start), "-t".into(), fmt_secs(r.duration())]);
        }
    };
    let hw = |args: &mut Vec<String>| {
        if s.device.hw_decode {
            args.extend(["-hwaccel".into(), "auto".into()]);
        }
    };
    if let Some(chain) = detect_chain {
        let mut a = strs(&["-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-stats", "-stats_period", "0.5"]);
        hw(&mut a);
        range_args(&mut a);
        a.extend(["-i".into(), path_arg(&input), "-map".into(), "0:v:0".into(), "-vf".into(), chain, "-f".into(), "null".into(), "-".into()]);
        passes.push(Pass::Ffmpeg(FfmpegPass { label: "Analyzing camera motion".into(), args: a, frames: frames_stab }));
    }

    let mut decode_chain: Vec<String> = lut.into_iter().collect();
    decode_chain.extend(pre.iter().cloned());
    decode_chain.extend(stab_transform);
    decode_chain.extend(decode_fps);
    decode_chain.extend(ffmpeg_interp);
    let out_tags = color_args(&out_color, enc);
    let mut tail: Vec<String> = Vec::new();
    if !sequence && !discard {
        tail.extend(["-metadata".into(), format!("comment={description}")]);
    }
    if matches!(container.as_str(), "mp4" | "mov") {
        tail.extend(["-movflags".into(), "+faststart".into()]);
    }
    tail.extend(["-max_muxing_queue_size".into(), "4096".into()]);
    if discard {
        tail.extend(["-f".into(), "null".into()]);
    }
    tail.push(path_arg(&output_path));

    if stages.is_empty() {
        // Single FFmpeg command.
        let mut a = strs(&["-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-stats", "-stats_period", "0.5"]);
        hw(&mut a);
        range_args(&mut a);
        a.extend(["-i".into(), path_arg(&input), "-map".into(), "0:v:0".into()]);
        a.extend(audio.maps(0));
        if subtitles {
            a.extend(strs(&["-map", "0:s?", "-c:s", "copy"]));
        }
        let mut vf = decode_chain;
        vf.extend(encode_fps.clone());
        let in_rgb = matches!(info.chroma, Chroma::Rgb);
        vf.extend(convert_filters(cur, content, output, &s.output, sar, Some((&src_color, in_rgb)), &out_color, enc));
        a.extend(["-vf".into(), vf.join(",")]);
        if info.vfr {
            a.extend(["-fps_mode".into(), "cfr".into(), "-r".into(), fps.to_ffmpeg()]);
        }
        a.extend(enc.video_args(s.export.rate, output.0, output.1, fps.as_f64(), &container));
        a.extend(out_tags);
        a.extend(audio.codec_args());
        a.extend(strs(&["-map_metadata", "0"]));
        a.extend(tail);
        passes.push(Pass::Ffmpeg(FfmpegPass { label: if is_preview { "Rendering preview".into() } else { "Exporting".into() }, args: a, frames }));
    } else {
        let in_layout = if s.export.lut.is_some() || !matches!(info.chroma, Chroma::Yuv420 | Chroma::Gray) { RawLayout::Rgb } else { RawLayout::Yuv420 };
        let input_raw = RawFormat {
            width: process.0,
            height: process.1,
            layout: in_layout,
            bits: transport_bits(in_layout, info.bit_depth),
            matrix: src_color.matrix,
            range: src_color.range,
        };
        let out_layout = if enc.chroma == Chroma::Yuv420 { RawLayout::Yuv420 } else { RawLayout::Rgb };
        let output_raw = RawFormat {
            width: cur.0,
            height: cur.1,
            layout: out_layout,
            bits: transport_bits(out_layout, enc.bit_depth),
            matrix: out_color.matrix,
            range: out_color.range,
        };
        let fps_engine_out = stages
            .iter()
            .find_map(|st| match st {
                StageSpec::Interpolate(i) => Some(i.out_fps),
                _ => None,
            })
            .unwrap_or(fps_engine_in);
        let frames_engine_out = stages
            .iter()
            .find_map(|st| match st {
                StageSpec::Interpolate(i) => Some(i.frames_out),
                _ => None,
            })
            .unwrap_or(frames_engine_in);

        let mut d = strs(&["-hide_banner", "-nostdin", "-loglevel", "error"]);
        hw(&mut d);
        range_args(&mut d);
        d.extend(["-i".into(), path_arg(&input), "-map".into(), "0:v:0".into()]);
        if !decode_chain.is_empty() {
            d.extend(["-vf".into(), decode_chain.join(",")]);
        }
        if info.vfr {
            d.extend(["-fps_mode".into(), "cfr".into(), "-r".into(), fps_engine_in.to_ffmpeg()]);
        } else {
            d.extend(strs(&["-fps_mode", "passthrough"]));
        }
        d.extend(["-pix_fmt".into(), input_raw.pix_fmt().into()]);
        d.extend(strs(&["-f", "rawvideo", "-an", "-sn", "-dn", "pipe:1"]));

        let mut e = strs(&["-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo"]);
        e.extend(["-pix_fmt".into(), output_raw.pix_fmt().into(), "-video_size".into(), format!("{}x{}", cur.0, cur.1)]);
        e.extend(["-framerate".into(), fps_engine_out.to_ffmpeg(), "-i".into(), "pipe:0".into()]);
        let second_input = audio.uses_input() || subtitles;
        if second_input {
            range_args(&mut e);
            e.extend(["-i".into(), path_arg(&input)]);
        }
        e.extend(strs(&["-map", "0:v:0"]));
        e.extend(audio.maps(1));
        if subtitles {
            e.extend(strs(&["-map", "1:s?", "-c:s", "copy"]));
        }
        let mut vf = Vec::new();
        if out_layout == RawLayout::Yuv420 {
            vf.push(setparams(&out_color));
        }
        vf.extend(encode_fps);
        vf.extend(convert_filters(cur, content, output, &s.output, sar, None, &out_color, enc));
        e.extend(["-vf".into(), vf.join(",")]);
        e.extend(enc.video_args(s.export.rate, output.0, output.1, fps.as_f64(), &container));
        e.extend(out_tags);
        e.extend(audio.codec_args());
        if second_input {
            e.extend(strs(&["-map_metadata", "1"]));
        }
        e.extend(tail);
        passes.push(Pass::Engine(EnginePass {
            label: if is_preview { "Rendering preview".into() } else { "Enhancing".into() },
            decode: d,
            input: input_raw,
            stages,
            encode: e,
            output: output_raw,
            frames_in: frames_engine_in,
            frames_out: frames_engine_out,
            fps_in: fps_engine_in,
            fps_out: fps_engine_out,
        }));
    }

    Ok(JobPlan {
        input,
        output: output_path,
        output_is_sequence: sequence,
        temp_dir: ctx.temp_dir.clone(),
        passes,
        description,
        prefix,
        warnings,
        geometry: Geometry { source, process, content, output },
        fps_in: info.fps,
        fps_out: fps,
        frames_in: match range {
            Some(r) => info.frames_in(r.start, r.end).max(1),
            None => info.frame_count,
        },
        frames_out: frames,
        source_duration,
        device: s.device.clone(),
        models_dir: ctx.catalog.dir.clone(),
        is_preview,
        encoder: enc.id.to_string(),
    })
}

fn enhance_spec(e: &EnhancementSettings, from: (u32, u32), to: (u32, u32), ctx: &PlanContext) -> Result<EnhanceSpec> {
    let m = ctx.catalog.get(&e.model).filter(|m| m.kind == ModelKind::Enhancement).ok_or_else(|| anyhow!("unknown enhancement model '{}'", e.model))?;
    if !m.is_available(&ctx.catalog.dir) {
        bail!("enhancement model '{}' is not installed: run `prism models import <weights>` or place {}.onnx in {}", m.id, m.id, ctx.catalog.dir.display());
    }
    let mut pre = e.focus_fix.factor();
    while pre > 1 && (from.0 / pre < 32 || from.1 / pre < 32) {
        pre /= 2;
    }
    Ok(EnhanceSpec {
        model: m.id.clone(),
        in_width: from.0,
        in_height: from.1,
        out_width: to.0,
        out_height: to.1,
        pre_downscale: pre,
        params: EnhanceParams::from_settings(e),
        mode: e.param_mode,
        estimate_frames: if e.param_mode == ParamMode::Manual { 0 } else { 8 },
    })
}

fn push_enhance_info(e: &EnhancementSettings, ctx: &PlanContext, prefix: &mut Vec<String>, info: &mut Vec<String>) {
    let m = ctx.catalog.get(&e.model);
    prefix.push(m.map(|m| m.prefix.clone()).unwrap_or_else(|| e.model.replace('-', "")));
    let mode = match e.param_mode {
        ParamMode::Auto => "auto",
        ParamMode::RelativeToAuto => "relative to auto",
        ParamMode::Manual => "manual",
    };
    let mut s = if e.video_type == VideoType::Interlaced { format!("Deinterlaced and enhanced using {}", e.model) } else { format!("Enhanced using {}", e.model) };
    s += &format!(" (mode: {mode}; compression {}, details {}, sharpen {}, noise {}, dehalo {}, deblur {}", e.compression, e.details, e.sharpen, e.noise, e.dehalo, e.deblur);
    if e.add_noise > 0 {
        s += &format!(", add noise {}", e.add_noise);
    }
    if e.recover_original > 0 {
        s += &format!(", recover original detail {}", e.recover_original);
    }
    if e.focus_fix != FocusFix::Off {
        s += &format!(", focus fix {:?}", e.focus_fix);
    }
    s.push(')');
    info.push(s);
}

/// Content size and final output size for a processed frame of `process` pixels.
pub fn output_geometry(process: (u32, u32), out: &OutputSettings) -> ((u32, u32), (u32, u32)) {
    let (w, h) = (process.0 as f64, process.1 as f64);
    match out.resolution {
        Resolution::Original => (process, process),
        Resolution::Scale(f) => {
            let c = (even(w * f), even(h * f));
            (c, c)
        }
        Resolution::Height(th) => {
            let c = (even(th as f64 * w / h), even(th as f64));
            (c, c)
        }
        Resolution::Box { width, height } => {
            let (bw, bh) = (width as f64, height as f64);
            let fit = (bw / w).min(bh / h);
            let fill = (bw / w).max(bh / h);
            let boxed = (even(bw), even(bh));
            match out.fit {
                FitMode::Fit => {
                    let c = (even(w * fit), even(h * fit));
                    (c, c)
                }
                FitMode::Pad => ((even(w * fit).min(boxed.0), even(h * fit).min(boxed.1)), boxed),
                FitMode::Crop => ((even(w * fill).max(boxed.0), even(h * fill).max(boxed.1)), boxed),
                FitMode::Stretch => (boxed, boxed),
            }
        }
    }
}

/// Encode-side sizing and colour conversion. `source` is set for the single-FFmpeg path, where
/// frames still carry the source colour description.
#[allow(clippy::too_many_arguments)]
fn convert_filters(cur: (u32, u32), content: (u32, u32), output: (u32, u32), out: &OutputSettings, sar: Rational, source: Option<(&ColorInfo, bool)>, out_color: &ColorInfo, enc: &VideoEncoder) -> Vec<String> {
    let mut vf = Vec::new();
    let out_rgb = enc.chroma == Chroma::Rgb;
    let mut scale_opts: Vec<String> = Vec::new();
    if cur != content {
        scale_opts.push(format!("w={}:h={}:flags=lanczos", content.0, content.1));
    }
    match source {
        Some((src, src_rgb)) => {
            if !src_rgb && (src.matrix != out_color.matrix || src.range != out_color.range || out_rgb) {
                scale_opts.push(format!("in_color_matrix={}:in_range={}", src.matrix.swscale_name(), src.range.ffmpeg_name()));
            }
            if !out_rgb && (src_rgb || src.matrix != out_color.matrix || src.range != out_color.range) {
                scale_opts.push(format!("out_color_matrix={}:out_range={}", out_color.matrix.swscale_name(), out_color.range.ffmpeg_name()));
            }
        }
        // Engine path: RGB frames need an explicit matrix for YUV encoders (swscale's default is BT.601).
        None => {
            if enc.chroma != Chroma::Yuv420 && !out_rgb {
                scale_opts.push(format!("out_color_matrix={}:out_range={}", out_color.matrix.swscale_name(), out_color.range.ffmpeg_name()));
            }
        }
    }
    if !scale_opts.is_empty() {
        vf.push(format!("scale={}", scale_opts.join(":")));
    }
    if content != output {
        match out.fit {
            FitMode::Pad => vf.push(format!("pad=w={}:h={}:x=(ow-iw)/2:y=(oh-ih)/2:color=black", output.0, output.1)),
            _ => vf.push(format!("crop=w={}:h={}", output.0, output.1)),
        }
    }
    vf.push(format!("setsar={}", sar.to_ffmpeg()));
    vf.push(format!("format={}", enc.pix_fmt));
    vf
}

fn setparams(c: &ColorInfo) -> String {
    format!("setparams=range={}:color_primaries={}:color_trc={}:colorspace={}", c.range.ffmpeg_name(), c.primaries, c.transfer, c.matrix.ffmpeg_colorspace())
}

fn color_args(c: &ColorInfo, enc: &VideoEncoder) -> Vec<String> {
    if enc.chroma == Chroma::Rgb {
        return Vec::new();
    }
    vec![
        "-color_primaries".into(),
        c.primaries.clone(),
        "-color_trc".into(),
        c.transfer.clone(),
        "-colorspace".into(),
        c.matrix.ffmpeg_colorspace().into(),
        "-color_range".into(),
        c.range.ffmpeg_name().into(),
    ]
}

fn transport_bits(layout: RawLayout, depth: u32) -> u8 {
    match (layout, depth) {
        (_, 0..=8) => 8,
        (RawLayout::Yuv420, 9..=10) => 10,
        _ => 16,
    }
}

#[derive(Debug, Clone, Default)]
struct AudioPlan {
    /// Audio streams to map (indices into the input's audio streams); empty means no audio.
    streams: Vec<u32>,
    /// Per-output-stream codec args after the maps.
    codec: Vec<String>,
}

impl AudioPlan {
    fn uses_input(&self) -> bool {
        !self.streams.is_empty()
    }

    fn maps(&self, input: usize) -> Vec<String> {
        if self.streams.is_empty() {
            return Vec::new();
        }
        self.streams.iter().flat_map(|i| ["-map".to_string(), format!("{input}:a:{i}")]).collect()
    }

    fn codec_args(&self) -> Vec<String> {
        if self.streams.is_empty() { vec!["-an".into()] } else { self.codec.clone() }
    }
}

fn audio_plan(info: &VideoInfo, export: &ExportSettings, container: &str, slowmo: bool, disabled: bool, warnings: &mut Vec<String>) -> AudioPlan {
    if disabled || info.audio.is_empty() || export.audio.mode == AudioMode::None {
        return AudioPlan::default();
    }
    if slowmo {
        warnings.push("Audio is removed because slow motion changes the duration.".into());
        return AudioPlan::default();
    }
    let streams: Vec<u32> = info.audio.iter().map(|a| a.index).collect();
    let convert = |codec: &encoders::AudioCodec, bitrate: u32| -> Vec<String> {
        let mut v = vec!["-c:a".to_string(), codec.encoder.to_string()];
        if let Some((min, max, def)) = codec.bitrate {
            let b = if bitrate == 0 { def } else { bitrate.clamp(min, max) };
            v.extend(["-b:a".into(), format!("{b}k")]);
        }
        if codec.id == "vorbis" || codec.id == "ac3" {
            v.extend(["-ac".into(), "2".into()]);
        }
        v
    };
    match export.audio.mode {
        AudioMode::Copy => {
            if let Some(bad) = info.audio.iter().find(|a| !encoders::audio_copy_compatible(&a.codec, container)) {
                let fallback = encoders::default_audio_for(container);
                warnings.push(format!("{} audio cannot be copied into .{container}; converting to {}.", bad.codec, fallback.name));
                return AudioPlan { streams, codec: convert(fallback, 0) };
            }
            AudioPlan { streams, codec: vec!["-c:a".into(), "copy".into()] }
        }
        AudioMode::Convert => {
            let mut codec = encoders::audio_codec(&export.audio.codec).unwrap_or_else(|| encoders::default_audio_for(container));
            if !codec.containers.contains(&container) {
                let fallback = encoders::default_audio_for(container);
                warnings.push(format!("{} audio is not supported in .{container}; using {}.", codec.name, fallback.name));
                codec = fallback;
            }
            AudioPlan { streams, codec: convert(codec, export.audio.bitrate_kbps) }
        }
        AudioMode::None => AudioPlan::default(),
    }
}

fn sentence(parts: &[String]) -> String {
    match parts.len() {
        0 => String::new(),
        1 => parts[0].clone(),
        2 => format!("{}; and {}", parts[0], parts[1]),
        n => format!("{}; and {}", parts[..n - 1].join("; "), parts[n - 1]),
    }
}

fn even(v: f64) -> u32 {
    (((v / 2.0).round() as u32) * 2).max(2)
}

fn strs(v: &[&str]) -> Vec<String> {
    v.iter().map(|s| s.to_string()).collect()
}

fn fmt_secs(v: f64) -> String {
    format!("{v:.6}")
}

fn path_arg(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

/// Quotes a path for use as a filter option value (e.g. `lut3d=file=...`).
pub fn escape_filter_path(p: &Path) -> Result<String> {
    let s = std::path::absolute(p).unwrap_or_else(|_| p.to_path_buf()).to_string_lossy().replace('\\', "/");
    if s.contains('\'') {
        bail!("paths containing an apostrophe cannot be used in filters: {s}");
    }
    Ok(format!("'{}'", s.replace(':', "\\:")))
}

/// Renders argv for display in a shell-like form.
pub fn shell_join(program: &str, args: &[String]) -> String {
    let q = |a: &str| {
        if a.is_empty() || a.contains([' ', '"', '\'', '&', '|', '<', '>', ';', '(', ')']) { format!("\"{}\"", a.replace('"', "\\\"")) } else { a.to_string() }
    };
    std::iter::once(q(program)).chain(args.iter().map(|a| q(a))).collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::media::{AudioStream, FieldOrder};

    fn video(w: u32, h: u32) -> VideoInfo {
        VideoInfo {
            path: PathBuf::from("C:/in/clip.mp4"),
            container: "mov,mp4".into(),
            codec: "h264".into(),
            width: w,
            height: h,
            sar: Rational::ONE,
            fps: Rational::new(24000, 1001),
            vfr: false,
            duration: 10.0,
            start_time: 0.0,
            frame_count: 240,
            frame_count_exact: true,
            pix_fmt: "yuv420p".into(),
            bit_depth: 8,
            chroma: Chroma::Yuv420,
            has_alpha: false,
            color: ColorInfo::sdr_for_height(h),
            field_order: FieldOrder::Progressive,
            rotation: 0,
            audio: vec![AudioStream { index: 0, codec: "aac".into(), channels: 2, sample_rate: 48000, language: None }],
            subtitles: 0,
            bitrate: None,
        }
    }

    fn ctx<'a>(cat: &'a Catalog, preview: Option<PreviewRange>) -> PlanContext<'a> {
        PlanContext { catalog: cat, temp_dir: PathBuf::from("C:/tmp/job1"), encoders_available: None, preview, exists: &|_| false }
    }

    fn engine(p: &JobPlan) -> &EnginePass {
        p.passes.iter().find_map(|p| if let Pass::Engine(e) = p { Some(e) } else { None }).expect("engine pass")
    }

    #[test]
    fn transcode_only_is_one_ffmpeg_pass() {
        let cat = Catalog::load(Path::new("none"));
        let p = plan(&video(1920, 1080), &JobSettings::default(), &ctx(&cat, None)).unwrap();
        assert_eq!(p.passes.len(), 1);
        let Pass::Ffmpeg(f) = &p.passes[0] else { panic!() };
        assert!(f.args.windows(2).any(|w| w == ["-c:a", "copy"]));
        assert!(f.args.windows(2).any(|w| w == ["-map", "0:a:0"]));
        assert_eq!(p.output, PathBuf::from("C:/in/clip_prism.mp4"));
        assert_eq!(p.frames_out, 240);
    }

    #[test]
    fn upscale_uses_engine_with_yuv_transport() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        s.output.resolution = Resolution::uhd4k();
        let p = plan(&video(960, 540), &s, &ctx(&cat, None)).unwrap();
        let e = engine(&p);
        assert_eq!(e.input.pix_fmt(), "yuv420p");
        assert_eq!((e.output.width, e.output.height), (3840, 2160));
        assert_eq!(p.geometry.output, (3840, 2160));
        assert!(e.encode.iter().any(|a| a == "pipe:0"));
        assert!(e.encode.windows(2).any(|w| w == ["-map", "1:a:0"]), "audio from the source as second input");
        assert!(e.decode.windows(2).any(|w| w == ["-fps_mode", "passthrough"]));
        let StageSpec::Enhance(sp) = &e.stages[0] else { panic!() };
        assert_eq!(sp.estimate_frames, 8);
        assert_eq!(p.output.file_name().unwrap(), "clip_cls.mp4");
        // SD source upgraded to the BT.709 matrix for an HD output.
        assert_eq!(e.output.matrix, Matrix::Bt709);
        assert_eq!(e.input.matrix, Matrix::Bt601);
    }

    #[test]
    fn interpolation_frame_maths_and_slowmo_audio() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.interpolation.enabled = true;
        s.interpolation.slowmo = 2.0;
        s.interpolation.output_fps = Some(60.0);
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        let e = engine(&p);
        // 240 frames * 2 * 60 / 23.976
        assert_eq!(e.frames_out, (240.0f64 * 2.0 * 60.0 / (24000.0 / 1001.0)).round() as u64);
        assert_eq!(e.fps_out, Rational::new(60, 1));
        assert!(e.encode.iter().any(|a| a == "-an"));
        assert!(p.warnings.iter().any(|w| w.contains("slow motion")));
    }

    #[test]
    fn stabilization_adds_analysis_pass_first() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.stabilization.enabled = true;
        s.enhancement.enabled = true;
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        assert_eq!(p.passes.len(), 2);
        let Pass::Ffmpeg(a) = &p.passes[0] else { panic!() };
        assert!(a.args.iter().any(|x| x.contains("vidstabdetect")));
        assert!(a.args.ends_with(&["-f".into(), "null".into(), "-".into()]));
        assert!(engine(&p).decode.iter().any(|x| x.contains("vidstabtransform")));
        assert!(p.prefix.starts_with("stab"));
    }

    #[test]
    fn hdr_upgrades_encoder_and_tags_output() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.hdr.enabled = true;
        s.export.encoder = "h265-x265".into();
        let p = plan(&video(1920, 1080), &s, &ctx(&cat, None)).unwrap();
        assert_eq!(p.encoder, "h265-x265-10");
        let e = engine(&p);
        assert_eq!(e.output.pix_fmt(), "yuv420p10le");
        assert_eq!(e.output.matrix, Matrix::Bt2020Nc);
        assert!(e.encode.windows(2).any(|w| w == ["-color_trc", "smpte2084"]));
    }

    #[test]
    fn prores_uses_rgb_transport_with_explicit_matrix() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        s.export.encoder = "prores-hq".into();
        let p = plan(&video(1920, 1080), &s, &ctx(&cat, None)).unwrap();
        let e = engine(&p);
        assert_eq!(e.output.pix_fmt(), "gbrp16le");
        let vf = &e.encode[e.encode.iter().position(|a| a == "-vf").unwrap() + 1];
        assert!(vf.contains("out_color_matrix=bt709"), "{vf}");
        assert!(vf.ends_with("format=yuv422p10le"));
        assert_eq!(p.output.extension().unwrap(), "mov");
    }

    #[test]
    fn preview_targets_temp_dir_without_audio() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        let p = plan(&video(1280, 720), &s, &ctx(&cat, Some(PreviewRange { start: 3.0, duration: 2.0 }))).unwrap();
        assert!(p.is_preview);
        assert_eq!(p.output, PathBuf::from("C:/tmp/job1/preview.mkv"));
        let e = engine(&p);
        assert!(e.decode.windows(2).any(|w| w == ["-ss", "3.000000"]));
        assert_eq!(e.frames_in, 48);
        assert!(e.encode.iter().any(|a| a == "-an"));
    }

    #[test]
    fn geometry_modes() {
        let o = |fit, w, h| OutputSettings { resolution: Resolution::Box { width: w, height: h }, fit, fps: None };
        assert_eq!(output_geometry((640, 480), &o(FitMode::Fit, 3840, 2160)), ((2880, 2160), (2880, 2160)));
        assert_eq!(output_geometry((640, 480), &o(FitMode::Pad, 3840, 2160)), ((2880, 2160), (3840, 2160)));
        assert_eq!(output_geometry((640, 480), &o(FitMode::Crop, 3840, 2160)), ((3840, 2880), (3840, 2160)));
        assert_eq!(output_geometry((720, 480), &OutputSettings { resolution: Resolution::Scale(2.0), ..Default::default() }), ((1440, 960), (1440, 960)));
    }

    #[test]
    fn missing_onnx_model_fails_closed() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        s.enhancement.model = "resr-general-v3".into();
        let err = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap_err().to_string();
        assert!(err.contains("not installed"), "{err}");
    }

    #[test]
    fn rotation_crop_and_deinterlace_double_rate() {
        let cat = Catalog::load(Path::new("none"));
        let mut s = JobSettings::default();
        s.rotation = 1;
        s.crop = Some(CropRect { x: 11, y: 0, width: 701, height: 1280 });
        s.deinterlace.enabled = true;
        s.deinterlace.mode = DeinterlaceMode::DoubleRate;
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        assert_eq!(p.geometry.process, (700, 1280));
        assert_eq!(p.frames_out, 480);
        let Pass::Ffmpeg(f) = &p.passes[0] else { panic!() };
        let vf = &f.args[f.args.iter().position(|a| a == "-vf").unwrap() + 1];
        assert!(vf.starts_with("transpose=clock,bwdif=mode=send_field"), "{vf}");
        assert!(vf.contains("crop=w=700:h=1280:x=10:y=0"), "{vf}");
    }

    #[test]
    fn trivial_work_is_routed_to_ffmpeg_filters() {
        let cat = Catalog::load(Path::new("none"));
        // Blend interpolation becomes FFmpeg's framerate filter.
        let mut s = JobSettings::default();
        s.interpolation.enabled = true;
        s.interpolation.model = "prism-blend".into();
        s.interpolation.output_fps = Some(60.0);
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        assert_eq!(p.passes.len(), 1);
        let Pass::Ffmpeg(f) = &p.passes[0] else { panic!("expected a single FFmpeg pass") };
        let vf = &f.args[f.args.iter().position(|a| a == "-vf").unwrap() + 1];
        assert!(vf.starts_with("mpdecimate,framerate=fps=60"), "{vf}");
        assert_eq!(p.fps_out, Rational::new(60, 1));
        // Classic with every slider at zero is a plain Lanczos resize.
        let mut s = JobSettings::default();
        s.enhancement.enabled = true;
        s.enhancement.param_mode = ParamMode::Manual;
        s.output.resolution = Resolution::Scale(2.0);
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        let Pass::Ffmpeg(f) = &p.passes[0] else { panic!("expected a single FFmpeg pass") };
        assert!(f.args.iter().any(|a| a.contains("scale=w=2560:h=1440:flags=lanczos")));
        // Blend after a Rust stage stays in the engine; a non-zero slider keeps Classic in Rust.
        s.enhancement.sharpen = 10;
        s.interpolation.enabled = true;
        s.interpolation.model = "prism-blend".into();
        let p = plan(&video(1280, 720), &s, &ctx(&cat, None)).unwrap();
        let e = engine(&p);
        assert!(matches!(e.stages[0], StageSpec::Enhance(_)));
        assert!(e.decode.iter().any(|a| a.contains("framerate")), "blend still runs in the decoder before enhancement");
    }

    #[test]
    fn filter_path_escaping() {
        assert_eq!(escape_filter_path(Path::new("C:\\luts\\a b.cube")).unwrap(), "'C\\:/luts/a b.cube'");
    }
}
