//! Media facts about an input video, parsed once from `ffprobe -show_streams -show_format -of json`.
//!
//! The engine runs ffprobe; this module owns the interpretation of its output so every
//! consumer (planner, GUI, CLI) works from the same parsed facts.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// An exact rational number, used for frame rates and aspect ratios.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Rational {
    pub num: i64,
    pub den: i64,
}

impl Rational {
    pub const ONE: Rational = Rational { num: 1, den: 1 };

    pub fn new(num: i64, den: i64) -> Self {
        if den == 0 {
            return Rational { num: 0, den: 1 };
        }
        let g = gcd(num.unsigned_abs(), den.unsigned_abs()).max(1) as i64;
        let sign = if den < 0 { -1 } else { 1 };
        Rational { num: sign * num / g, den: sign * den / g }
    }

    pub fn as_f64(self) -> f64 {
        if self.den == 0 { 0.0 } else { self.num as f64 / self.den as f64 }
    }

    pub fn is_valid(self) -> bool {
        self.num > 0 && self.den > 0
    }

    /// Parses `30000/1001`, `30000:1001`, `25` or `29.97`.
    pub fn parse(s: &str) -> Option<Self> {
        let s = s.trim();
        if let Some((a, b)) = s.split_once(['/', ':']) {
            let (a, b) = (a.trim().parse::<i64>().ok()?, b.trim().parse::<i64>().ok()?);
            let r = Rational::new(a, b);
            return r.is_valid().then_some(r);
        }
        let v: f64 = s.parse().ok()?;
        (v > 0.0 && v.is_finite()).then(|| Rational::from_f64(v))
    }

    /// Converts a decimal rate to a rational, recognising the NTSC family exactly.
    pub fn from_f64(v: f64) -> Self {
        for base in [24i64, 30, 48, 60, 120, 240] {
            let ntsc = base as f64 * 1000.0 / 1001.0;
            if (v - ntsc).abs() < 0.0015 * base as f64 / 24.0 {
                return Rational::new(base * 1000, 1001);
            }
        }
        if (v - v.round()).abs() < 1e-6 {
            return Rational::new(v.round() as i64, 1);
        }
        Rational::new((v * 1000.0).round() as i64, 1000)
    }

    /// FFmpeg's `num/den` syntax.
    pub fn to_ffmpeg(self) -> String {
        if self.den == 1 { self.num.to_string() } else { format!("{}/{}", self.num, self.den) }
    }

    pub fn mul(self, o: Rational) -> Rational {
        Rational::new(self.num * o.num, self.den * o.den)
    }
}

impl std::fmt::Display for Rational {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let v = self.as_f64();
        if self.den == 1 { write!(f, "{}", self.num) } else { write!(f, "{:.3}", v) }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ColorRange {
    #[default]
    Limited,
    Full,
}

impl ColorRange {
    pub fn ffmpeg_name(self) -> &'static str {
        match self {
            ColorRange::Limited => "tv",
            ColorRange::Full => "pc",
        }
    }
}

/// YCbCr matrix coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Matrix {
    Bt601,
    #[default]
    Bt709,
    Bt2020Nc,
}

impl Matrix {
    /// (Kr, Kb) luma coefficients.
    pub fn coefficients(self) -> (f32, f32) {
        match self {
            Matrix::Bt601 => (0.299, 0.114),
            Matrix::Bt709 => (0.2126, 0.0722),
            Matrix::Bt2020Nc => (0.2627, 0.0593),
        }
    }

    pub fn ffmpeg_colorspace(self) -> &'static str {
        match self {
            Matrix::Bt601 => "smpte170m",
            Matrix::Bt709 => "bt709",
            Matrix::Bt2020Nc => "bt2020nc",
        }
    }

    /// Name accepted by swscale's `in_color_matrix`/`out_color_matrix`.
    pub fn swscale_name(self) -> &'static str {
        match self {
            Matrix::Bt601 => "bt601",
            Matrix::Bt709 => "bt709",
            Matrix::Bt2020Nc => "bt2020",
        }
    }
}

/// Colour description of a stream. Unknown tags are resolved once, at probe time,
/// using the same height heuristic FFmpeg applies, and the original tags are kept.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ColorInfo {
    pub range: ColorRange,
    pub matrix: Matrix,
    /// FFmpeg primaries name (`bt709`, `bt2020`, `smpte170m`, ...).
    pub primaries: String,
    /// FFmpeg transfer name (`bt709`, `smpte2084`, `arib-std-b67`, ...).
    pub transfer: String,
    /// True when one or more of the above was inferred rather than tagged.
    pub inferred: bool,
}

impl Default for ColorInfo {
    fn default() -> Self {
        ColorInfo {
            range: ColorRange::Limited,
            matrix: Matrix::Bt709,
            primaries: "bt709".into(),
            transfer: "bt709".into(),
            inferred: true,
        }
    }
}

impl ColorInfo {
    pub fn sdr_for_height(height: u32) -> Self {
        if height >= 720 {
            ColorInfo::default()
        } else {
            ColorInfo {
                range: ColorRange::Limited,
                matrix: Matrix::Bt601,
                primaries: "smpte170m".into(),
                transfer: "bt709".into(),
                inferred: true,
            }
        }
    }

    pub fn hdr10() -> Self {
        ColorInfo {
            range: ColorRange::Limited,
            matrix: Matrix::Bt2020Nc,
            primaries: "bt2020".into(),
            transfer: "smpte2084".into(),
            inferred: false,
        }
    }

    pub fn hlg() -> Self {
        ColorInfo { transfer: "arib-std-b67".into(), ..ColorInfo::hdr10() }
    }

    pub fn is_hdr(&self) -> bool {
        matches!(self.transfer.as_str(), "smpte2084" | "arib-std-b67")
    }

    fn from_tags(height: u32, range: Option<&str>, space: Option<&str>, prim: Option<&str>, trc: Option<&str>) -> Self {
        let mut c = ColorInfo::sdr_for_height(height);
        fn known(v: Option<&str>) -> Option<&str> {
            v.filter(|s| !s.is_empty() && !matches!(*s, "unknown" | "unspecified" | "reserved"))
        }
        let mut inferred = false;
        match known(range) {
            Some("pc") | Some("jpeg") => c.range = ColorRange::Full,
            Some(_) => c.range = ColorRange::Limited,
            None => inferred = true,
        }
        match known(space) {
            Some("bt709") => c.matrix = Matrix::Bt709,
            Some("bt2020nc") | Some("bt2020c") => c.matrix = Matrix::Bt2020Nc,
            Some("smpte170m") | Some("bt470bg") | Some("fcc") | Some("smpte240m") => c.matrix = Matrix::Bt601,
            Some(_) | None => inferred = true,
        }
        match known(prim) {
            Some(p) => c.primaries = p.to_string(),
            None => inferred = true,
        }
        match known(trc) {
            Some(t) => c.transfer = t.to_string(),
            None => inferred = true,
        }
        c.inferred = inferred;
        c
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum FieldOrder {
    #[default]
    Progressive,
    TopFirst,
    BottomFirst,
    Unknown,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Chroma {
    #[default]
    Yuv420,
    Yuv422,
    Yuv444,
    Rgb,
    Gray,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AudioStream {
    /// Index among the file's audio streams (`0:a:N`).
    pub index: u32,
    pub codec: String,
    pub channels: u32,
    pub sample_rate: u32,
    pub language: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct VideoInfo {
    pub path: PathBuf,
    pub container: String,
    pub codec: String,
    pub width: u32,
    pub height: u32,
    /// Sample (pixel) aspect ratio.
    pub sar: Rational,
    pub fps: Rational,
    /// Average and nominal frame rates differ, so timestamps are not evenly spaced.
    pub vfr: bool,
    pub duration: f64,
    pub start_time: f64,
    /// Exact count when the container reports it, otherwise `duration * fps` rounded.
    pub frame_count: u64,
    pub frame_count_exact: bool,
    pub pix_fmt: String,
    pub bit_depth: u32,
    pub chroma: Chroma,
    pub has_alpha: bool,
    pub color: ColorInfo,
    pub field_order: FieldOrder,
    /// Display-matrix rotation in degrees (counter-clockwise, as ffprobe reports it).
    pub rotation: i32,
    pub audio: Vec<AudioStream>,
    pub subtitles: u32,
    pub bitrate: Option<u64>,
}

impl VideoInfo {
    /// Parses ffprobe JSON (`-show_streams -show_format`). The first video stream is used.
    pub fn from_ffprobe_json(path: &Path, json: &Value) -> Result<Self> {
        let streams = json.get("streams").and_then(Value::as_array).ok_or_else(|| anyhow!("ffprobe output has no streams"))?;
        let format = json.get("format");
        let video = streams
            .iter()
            .find(|s| s.get("codec_type").and_then(Value::as_str) == Some("video") && !is_attached_pic(s))
            .ok_or_else(|| anyhow!("no video stream in {}", path.display()))?;

        let s = |v: &Value, k: &str| v.get(k).and_then(Value::as_str).map(str::to_string);
        let width = video.get("width").and_then(Value::as_u64).context("video stream has no width")? as u32;
        let height = video.get("height").and_then(Value::as_u64).context("video stream has no height")? as u32;
        let pix_fmt = s(video, "pix_fmt").unwrap_or_else(|| "yuv420p".into());
        let rate = |k: &str| s(video, k).and_then(|v| Rational::parse(&v)).filter(|r| r.is_valid() && r.as_f64() < 1000.0);
        let (avg, nominal) = (rate("avg_frame_rate"), rate("r_frame_rate"));
        let fps = avg.or(nominal).unwrap_or(Rational::new(25, 1));
        let vfr = matches!((avg, nominal), (Some(a), Some(r)) if (a.as_f64() - r.as_f64()).abs() > 0.01 * r.as_f64());
        let sar = s(video, "sample_aspect_ratio").and_then(|v| Rational::parse(&v)).filter(|r| r.is_valid()).unwrap_or(Rational::ONE);
        let parse_f = |v: Option<String>| v.and_then(|x| x.parse::<f64>().ok()).filter(|x| x.is_finite());
        let duration = parse_f(s(video, "duration")).or_else(|| format.and_then(|f| parse_f(s(f, "duration")))).unwrap_or(0.0);
        let start_time = parse_f(s(video, "start_time")).unwrap_or(0.0);
        let nb_frames = s(video, "nb_frames").and_then(|v| v.parse::<u64>().ok()).filter(|n| *n > 0);
        let (frame_count, frame_count_exact) = match nb_frames {
            Some(n) => (n, true),
            None => (((duration * fps.as_f64()).round() as u64).max(1), false),
        };
        let field_order = match s(video, "field_order").as_deref() {
            None | Some("progressive") => FieldOrder::Progressive,
            Some("tt") | Some("tb") => FieldOrder::TopFirst,
            Some("bb") | Some("bt") => FieldOrder::BottomFirst,
            Some(_) => FieldOrder::Unknown,
        };
        let rotation = rotation_of(video);
        let (bit_depth, chroma, has_alpha) = pix_fmt_traits(&pix_fmt);
        let color = ColorInfo::from_tags(
            height,
            s(video, "color_range").as_deref(),
            s(video, "color_space").as_deref(),
            s(video, "color_primaries").as_deref(),
            s(video, "color_transfer").as_deref(),
        );
        let mut audio = Vec::new();
        let mut subtitles = 0;
        for st in streams {
            match st.get("codec_type").and_then(Value::as_str) {
                Some("audio") => audio.push(AudioStream {
                    index: audio.len() as u32,
                    codec: s(st, "codec_name").unwrap_or_default(),
                    channels: st.get("channels").and_then(Value::as_u64).unwrap_or(2) as u32,
                    sample_rate: s(st, "sample_rate").and_then(|v| v.parse().ok()).unwrap_or(48000),
                    language: st.get("tags").and_then(|t| t.get("language")).and_then(Value::as_str).map(str::to_string),
                }),
                Some("subtitle") => subtitles += 1,
                _ => {}
            }
        }
        Ok(VideoInfo {
            path: path.to_path_buf(),
            container: format.and_then(|f| s(f, "format_name")).unwrap_or_default(),
            codec: s(video, "codec_name").unwrap_or_default(),
            width,
            height,
            sar,
            fps,
            vfr,
            duration,
            start_time,
            frame_count,
            frame_count_exact,
            pix_fmt,
            bit_depth,
            chroma,
            has_alpha,
            color,
            field_order,
            rotation,
            audio,
            subtitles,
            bitrate: format.and_then(|f| s(f, "bit_rate")).and_then(|v| v.parse().ok()),
        })
    }

    pub fn is_interlaced(&self) -> bool {
        matches!(self.field_order, FieldOrder::TopFirst | FieldOrder::BottomFirst)
    }

    /// Display aspect ratio including SAR.
    pub fn display_aspect(&self) -> f64 {
        self.width as f64 * self.sar.as_f64() / self.height.max(1) as f64
    }

    /// Number of frames in `[start, end)` seconds of this video.
    pub fn frames_in(&self, start: f64, end: f64) -> u64 {
        (((end - start).max(0.0)) * self.fps.as_f64()).round() as u64
    }
}

fn is_attached_pic(stream: &Value) -> bool {
    stream.get("disposition").and_then(|d| d.get("attached_pic")).and_then(Value::as_i64) == Some(1)
}

fn rotation_of(stream: &Value) -> i32 {
    if let Some(list) = stream.get("side_data_list").and_then(Value::as_array) {
        for sd in list {
            if let Some(r) = sd.get("rotation").and_then(Value::as_f64) {
                return r.round() as i32;
            }
        }
    }
    stream
        .get("tags")
        .and_then(|t| t.get("rotate"))
        .and_then(Value::as_str)
        .and_then(|r| r.parse::<i32>().ok())
        .map(|r| -r)
        .unwrap_or(0)
}

/// Bit depth, chroma layout and alpha presence implied by an FFmpeg pixel-format name.
pub fn pix_fmt_traits(pix_fmt: &str) -> (u32, Chroma, bool) {
    let p = pix_fmt.to_ascii_lowercase();
    let depth = if p.contains("f32") {
        32
    } else if p.contains("f16") {
        16
    } else if ["p16", "48", "64", "16le", "16be", "p016", "y416"].iter().any(|k| p.contains(k)) {
        16
    } else if ["p14"].iter().any(|k| p.contains(k)) {
        14
    } else if ["p12", "12le", "12be", "p012", "y212"].iter().any(|k| p.contains(k)) {
        12
    } else if ["p10", "10le", "10be", "p010", "v210", "x2rgb10", "x2bgr10", "y210", "v410", "xv30"].iter().any(|k| p.contains(k)) {
        10
    } else {
        8
    };
    let chroma = if p.starts_with("gray") || p.starts_with("ya") || p.starts_with("mono") {
        Chroma::Gray
    } else if p.starts_with("gbr") || p.contains("rgb") || p.contains("bgr") || p.starts_with("argb") || p.starts_with("abgr") {
        Chroma::Rgb
    } else if p.contains("444") || p.starts_with("ayuv") || p.starts_with("vuya") || p.starts_with("v410") || p.starts_with("xv3") || p.starts_with("y41") {
        Chroma::Yuv444
    } else if p.contains("422") || p.starts_with("yuyv") || p.starts_with("uyvy") || p.starts_with("v210") || p.starts_with("y21") || p.starts_with("nv16") || p.starts_with("p21") {
        Chroma::Yuv422
    } else {
        Chroma::Yuv420
    };
    let alpha = p.starts_with("yuva") || p.starts_with("gbrap") || p.contains("rgba") || p.contains("bgra") || p.starts_with("argb") || p.starts_with("abgr") || p.starts_with("ya") || p.starts_with("ayuv") || p.starts_with("vuya") || p.contains("rgba64");
    (depth, chroma, alpha)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rational_parsing_and_ntsc() {
        assert_eq!(Rational::parse("30000/1001"), Some(Rational::new(30000, 1001)));
        assert_eq!(Rational::parse("29.97"), Some(Rational::new(30000, 1001)));
        assert_eq!(Rational::parse("23.976"), Some(Rational::new(24000, 1001)));
        assert_eq!(Rational::parse("59.94"), Some(Rational::new(60000, 1001)));
        assert_eq!(Rational::parse("25"), Some(Rational::new(25, 1)));
        assert_eq!(Rational::parse("0/0"), None);
        assert_eq!(Rational::new(50, 2).to_ffmpeg(), "25");
    }

    #[test]
    fn pix_fmt_classification() {
        assert_eq!(pix_fmt_traits("yuv420p"), (8, Chroma::Yuv420, false));
        assert_eq!(pix_fmt_traits("yuv420p10le"), (10, Chroma::Yuv420, false));
        assert_eq!(pix_fmt_traits("p010le"), (10, Chroma::Yuv420, false));
        assert_eq!(pix_fmt_traits("yuv422p10le"), (10, Chroma::Yuv422, false));
        assert_eq!(pix_fmt_traits("yuva444p12le"), (12, Chroma::Yuv444, true));
        assert_eq!(pix_fmt_traits("rgb48le"), (16, Chroma::Rgb, false));
        assert_eq!(pix_fmt_traits("rgba"), (8, Chroma::Rgb, true));
        assert_eq!(pix_fmt_traits("gbrpf32le"), (32, Chroma::Rgb, false));
        assert_eq!(pix_fmt_traits("gray"), (8, Chroma::Gray, false));
    }

    #[test]
    fn parses_ffprobe_json() {
        let json: Value = serde_json::from_str(
            r#"{"streams":[
                {"index":0,"codec_type":"video","codec_name":"h264","width":1920,"height":1080,"pix_fmt":"yuv420p",
                 "sample_aspect_ratio":"1:1","avg_frame_rate":"24000/1001","r_frame_rate":"24000/1001",
                 "nb_frames":"240","duration":"10.010000","field_order":"progressive","color_space":"bt709",
                 "color_range":"tv","side_data_list":[{"rotation":-90}]},
                {"index":1,"codec_type":"audio","codec_name":"aac","channels":2,"sample_rate":"48000","tags":{"language":"eng"}},
                {"index":2,"codec_type":"subtitle","codec_name":"subrip"}],
              "format":{"format_name":"mov,mp4,m4a,3gp,3g2,mj2","duration":"10.010000","bit_rate":"8000000"}}"#,
        )
        .unwrap();
        let info = VideoInfo::from_ffprobe_json(Path::new("a.mp4"), &json).unwrap();
        assert_eq!((info.width, info.height), (1920, 1080));
        assert_eq!(info.fps, Rational::new(24000, 1001));
        assert_eq!(info.frame_count, 240);
        assert!(info.frame_count_exact);
        assert_eq!(info.rotation, -90);
        assert_eq!(info.audio.len(), 1);
        assert_eq!(info.audio[0].language.as_deref(), Some("eng"));
        assert_eq!(info.subtitles, 1);
        assert_eq!(info.color.matrix, Matrix::Bt709);
        assert!(info.color.inferred, "primaries and transfer were untagged");
    }
}
