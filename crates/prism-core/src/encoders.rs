//! Encoder and audio-codec tables for stock FFmpeg.
//!
//! Same concepts as the reference product's encoder table (per-entry FFmpeg options, quality
//! levels, bitrate template, size and bit-depth limits, container list, hardware vendor) but written
//! for a stock FFmpeg build, so software encoders (x264, x265, SVT-AV1, libvpx) are included.

use std::collections::HashSet;

use serde::Serialize;

use crate::media::Chroma;
use crate::settings::RateControl;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Vendor {
    Software,
    Nvidia,
    Amd,
    Intel,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum RateSpec {
    /// Constant quality: `flag value` with values for High/Medium/Low; bitrate mode allowed.
    Quality { flag: &'static str, values: [u32; 3] },
    /// AMF/QSV-style constant QP applied to several flags at once.
    MultiQp { flags: &'static [&'static str], values: [u32; 3] },
    /// Quality fixed by the profile (ProRes, DNxHR) or lossless (FFV1, PNG, ...).
    Fixed,
}

#[derive(Debug, Clone, Serialize)]
pub struct VideoEncoder {
    pub id: &'static str,
    /// Codec family shown first in pickers ("H.264", "ProRes", ...).
    pub family: &'static str,
    /// Variant shown second ("x264 High", "NVENC Main10", ...).
    pub profile: &'static str,
    /// FFmpeg encoder name.
    pub codec: &'static str,
    pub vendor: Vendor,
    pub args: &'static [&'static str],
    /// Bitrate-mode arguments; `{B}`, `{MAX}` and `{BUF}` are Mbit/s placeholders.
    pub bitrate_args: &'static [&'static str],
    pub rate: RateSpec,
    pub pix_fmt: &'static str,
    pub bit_depth: u32,
    pub chroma: Chroma,
    pub containers: &'static [&'static str],
    pub image_sequence: bool,
    pub max_size: (u32, u32),
    /// Bits per pixel used to suggest a bitrate (`bpp * w * h * fps`).
    pub suggested_bpp: f64,
    /// Width/height must be multiples of this.
    pub alignment: u32,
}

impl VideoEncoder {
    pub fn is_hardware(&self) -> bool {
        self.vendor != Vendor::Software
    }

    pub fn supports_bitrate(&self) -> bool {
        !self.bitrate_args.is_empty()
    }

    pub fn default_container(&self) -> &'static str {
        self.containers[0]
    }

    pub fn suggested_bitrate_mbps(&self, width: u32, height: u32, fps: f64) -> f64 {
        let mbps = self.suggested_bpp * width as f64 * height as f64 * fps / 1e6;
        (mbps * 2.0).round() / 2.0
    }

    /// `-c:v ...` arguments for the given rate control. `container` decides container-only tags.
    pub fn video_args(&self, rate: RateControl, width: u32, height: u32, fps: f64, container: &str) -> Vec<String> {
        let mut out: Vec<String> = vec!["-c:v".into(), self.codec.into()];
        out.extend(self.args.iter().map(|s| s.to_string()));
        match (rate, self.rate) {
            (RateControl::Bitrate(mbps), _) if self.supports_bitrate() => {
                let b = if mbps > 0.0 { mbps } else { self.suggested_bitrate_mbps(width, height, fps) };
                for a in self.bitrate_args {
                    out.push(a.replace("{B}", &fmt_mbps(b)).replace("{MAX}", &fmt_mbps(b * 1.5)).replace("{BUF}", &fmt_mbps(b * 2.0)));
                }
            }
            (rc, RateSpec::Quality { flag, values }) => {
                let q = quality_of(rc);
                out.push(flag.into());
                out.push(values[q].to_string());
            }
            (rc, RateSpec::MultiQp { flags, values }) => {
                let q = quality_of(rc);
                for f in flags {
                    out.push((*f).into());
                    out.push(values[q].to_string());
                }
            }
            (_, RateSpec::Fixed) => {}
        }
        if matches!(self.codec, "libx265" | "hevc_nvenc" | "hevc_amf" | "hevc_qsv") && matches!(container, "mp4" | "mov") {
            out.extend(["-tag:v".into(), "hvc1".into()]);
        }
        out.extend(["-pix_fmt".into(), self.pix_fmt.into()]);
        out
    }
}

fn quality_of(rc: RateControl) -> usize {
    match rc {
        RateControl::Quality(q) => q.index(),
        RateControl::Bitrate(_) => 0,
    }
}

fn fmt_mbps(v: f64) -> String {
    if (v - v.round()).abs() < 1e-9 { format!("{}M", v.round() as i64) } else { format!("{}k", (v * 1000.0).round() as i64) }
}

const VIDEO_CONTAINERS: &[&str] = &["mp4", "mkv", "mov"];
const BR_SW: &[&str] = &["-b:v", "{B}", "-maxrate", "{MAX}", "-bufsize", "{BUF}"];
const BR_NV: &[&str] = &["-rc", "vbr", "-b:v", "{B}", "-maxrate", "{MAX}", "-bufsize", "{BUF}"];
const BR_AMF: &[&str] = &["-rc", "vbr_peak", "-b:v", "{B}", "-maxrate", "{MAX}", "-bufsize", "{BUF}"];
/// AMF constant-QP mode sets the QP of each frame type; the bitrate mode overrides `-rc cqp`.
const AMF_QP: RateSpec = RateSpec::MultiQp { flags: &["-qp_i", "-qp_p", "-qp_b"], values: [20, 24, 28] };
const MAX_SW: (u32, u32) = (16384, 16384);
const MAX_HW: (u32, u32) = (8192, 8192);

macro_rules! enc {
    ($id:expr, $fam:expr, $prof:expr, $codec:expr, $vendor:expr, $args:expr, $br:expr, $rate:expr, $pix:expr, $depth:expr, $chroma:expr, $conts:expr, $seq:expr, $max:expr, $bpp:expr, $align:expr) => {
        VideoEncoder {
            id: $id,
            family: $fam,
            profile: $prof,
            codec: $codec,
            vendor: $vendor,
            args: $args,
            bitrate_args: $br,
            rate: $rate,
            pix_fmt: $pix,
            bit_depth: $depth,
            chroma: $chroma,
            containers: $conts,
            image_sequence: $seq,
            max_size: $max,
            suggested_bpp: $bpp,
            alignment: $align,
        }
    };
}

pub fn video_encoders() -> &'static [VideoEncoder] {
    use Chroma::*;
    use RateSpec::*;
    use Vendor::*;
    static TABLE: std::sync::OnceLock<Vec<VideoEncoder>> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        vec![
            // Software
            enc!("h264-x264", "H.264", "x264 High", "libx264", Software, &["-preset", "medium", "-profile:v", "high"], BR_SW, Quality { flag: "-crf", values: [16, 20, 24] }, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, MAX_SW, 0.10, 2),
            enc!("h265-x265", "H.265", "x265 Main", "libx265", Software, &["-preset", "medium", "-x265-params", "log-level=error"], BR_SW, Quality { flag: "-crf", values: [18, 22, 26] }, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, MAX_SW, 0.06, 2),
            enc!("h265-x265-10", "H.265", "x265 Main10", "libx265", Software, &["-preset", "medium", "-profile:v", "main10", "-x265-params", "log-level=error"], BR_SW, Quality { flag: "-crf", values: [18, 22, 26] }, "yuv420p10le", 10, Yuv420, VIDEO_CONTAINERS, false, MAX_SW, 0.07, 2),
            enc!("av1-svt", "AV1", "SVT-AV1", "libsvtav1", Software, &["-preset", "7"], &["-b:v", "{B}"], Quality { flag: "-crf", values: [24, 30, 36] }, "yuv420p", 8, Yuv420, &["mp4", "mkv", "webm"], false, MAX_SW, 0.05, 2),
            enc!("av1-svt-10", "AV1", "SVT-AV1 10-bit", "libsvtav1", Software, &["-preset", "7"], &["-b:v", "{B}"], Quality { flag: "-crf", values: [24, 30, 36] }, "yuv420p10le", 10, Yuv420, &["mp4", "mkv", "webm"], false, MAX_SW, 0.05, 2),
            enc!("vp9", "VP9", "libvpx good", "libvpx-vp9", Software, &["-deadline", "good", "-cpu-used", "2", "-row-mt", "1", "-b:v", "0"], &["-b:v", "{B}"], Quality { flag: "-crf", values: [24, 31, 37] }, "yuv420p", 8, Yuv420, &["webm", "mkv", "mp4"], false, MAX_SW, 0.06, 2),
            enc!("prores-proxy", "ProRes", "422 Proxy", "prores_ks", Software, &["-profile:v", "0", "-vendor", "apl0"], &[], Fixed, "yuv422p10le", 10, Yuv422, &["mov", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("prores-lt", "ProRes", "422 LT", "prores_ks", Software, &["-profile:v", "1", "-vendor", "apl0"], &[], Fixed, "yuv422p10le", 10, Yuv422, &["mov", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("prores-std", "ProRes", "422", "prores_ks", Software, &["-profile:v", "2", "-vendor", "apl0"], &[], Fixed, "yuv422p10le", 10, Yuv422, &["mov", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("prores-hq", "ProRes", "422 HQ", "prores_ks", Software, &["-profile:v", "3", "-vendor", "apl0"], &[], Fixed, "yuv422p10le", 10, Yuv422, &["mov", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("prores-4444", "ProRes", "4444", "prores_ks", Software, &["-profile:v", "4", "-vendor", "apl0"], &[], Fixed, "yuv444p10le", 10, Yuv444, &["mov", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("dnxhr-hq", "DNxHR", "HQ 8-bit", "dnxhd", Software, &["-profile:v", "dnxhr_hq"], &[], Fixed, "yuv422p", 8, Yuv422, &["mov", "mxf", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("dnxhr-hqx", "DNxHR", "HQX 10-bit", "dnxhd", Software, &["-profile:v", "dnxhr_hqx"], &[], Fixed, "yuv422p10le", 10, Yuv422, &["mov", "mxf", "mkv"], false, MAX_SW, 0.0, 2),
            enc!("ffv1-8", "FFV1", "Lossless 8-bit 4:2:0", "ffv1", Software, &["-level", "3", "-slices", "16", "-slicecrc", "1", "-g", "1"], &[], Fixed, "yuv420p", 8, Yuv420, &["mkv", "mov", "avi"], false, MAX_SW, 0.0, 2),
            enc!("ffv1-10", "FFV1", "Lossless 10-bit 4:2:0", "ffv1", Software, &["-level", "3", "-slices", "16", "-slicecrc", "1", "-g", "1"], &[], Fixed, "yuv420p10le", 10, Yuv420, &["mkv", "mov", "avi"], false, MAX_SW, 0.0, 2),
            enc!("ffv1-16-rgb", "FFV1", "Lossless 16-bit RGB", "ffv1", Software, &["-level", "3", "-slices", "16", "-slicecrc", "1", "-g", "1"], &[], Fixed, "gbrp16le", 16, Rgb, &["mkv"], false, MAX_SW, 0.0, 1),
            // NVIDIA
            enc!("h264-nvenc", "H.264", "NVENC High", "h264_nvenc", Nvidia, &["-preset", "p6", "-tune", "hq", "-profile:v", "high", "-spatial-aq", "1", "-rc-lookahead", "20", "-bf", "3"], BR_NV, Quality { flag: "-cq", values: [19, 23, 27] }, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, (4096, 4096), 0.10, 2),
            enc!("hevc-nvenc", "H.265", "NVENC Main", "hevc_nvenc", Nvidia, &["-preset", "p6", "-tune", "hq", "-spatial-aq", "1", "-rc-lookahead", "20"], BR_NV, Quality { flag: "-cq", values: [20, 24, 28] }, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, MAX_HW, 0.06, 2),
            enc!("hevc-nvenc-10", "H.265", "NVENC Main10", "hevc_nvenc", Nvidia, &["-preset", "p6", "-tune", "hq", "-profile:v", "main10", "-spatial-aq", "1", "-rc-lookahead", "20"], BR_NV, Quality { flag: "-cq", values: [20, 24, 28] }, "p010le", 10, Yuv420, VIDEO_CONTAINERS, false, MAX_HW, 0.07, 2),
            enc!("av1-nvenc", "AV1", "NVENC", "av1_nvenc", Nvidia, &["-preset", "p6", "-tune", "hq", "-spatial-aq", "1"], BR_NV, Quality { flag: "-cq", values: [24, 30, 36] }, "yuv420p", 8, Yuv420, &["mp4", "mkv", "webm"], false, MAX_HW, 0.05, 2),
            enc!("av1-nvenc-10", "AV1", "NVENC 10-bit", "av1_nvenc", Nvidia, &["-preset", "p6", "-tune", "hq", "-spatial-aq", "1"], BR_NV, Quality { flag: "-cq", values: [24, 30, 36] }, "p010le", 10, Yuv420, &["mp4", "mkv", "webm"], false, MAX_HW, 0.05, 2),
            // AMD
            enc!("h264-amf", "H.264", "AMF High", "h264_amf", Amd, &["-quality", "quality", "-profile:v", "high", "-rc", "cqp"], BR_AMF, AMF_QP, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, (4096, 4096), 0.10, 2),
            enc!("hevc-amf", "H.265", "AMF Main", "hevc_amf", Amd, &["-quality", "quality", "-rc", "cqp"], BR_AMF, AMF_QP, "yuv420p", 8, Yuv420, VIDEO_CONTAINERS, false, MAX_HW, 0.06, 2),
            enc!("av1-amf", "AV1", "AMF", "av1_amf", Amd, &["-quality", "quality", "-rc", "cqp"], BR_AMF, AMF_QP, "yuv420p", 8, Yuv420, &["mp4", "mkv", "webm"], false, MAX_HW, 0.05, 2),
            // Intel
            enc!("h264-qsv", "H.264", "QSV High", "h264_qsv", Intel, &["-preset", "slower", "-profile:v", "high"], BR_SW, Quality { flag: "-global_quality", values: [20, 24, 28] }, "nv12", 8, Yuv420, VIDEO_CONTAINERS, false, (4096, 4096), 0.10, 2),
            enc!("hevc-qsv", "H.265", "QSV Main", "hevc_qsv", Intel, &["-preset", "slower"], BR_SW, Quality { flag: "-global_quality", values: [21, 25, 29] }, "nv12", 8, Yuv420, VIDEO_CONTAINERS, false, MAX_HW, 0.06, 2),
            enc!("av1-qsv", "AV1", "QSV", "av1_qsv", Intel, &["-preset", "slower"], BR_SW, Quality { flag: "-global_quality", values: [24, 30, 36] }, "nv12", 8, Yuv420, &["mp4", "mkv", "webm"], false, MAX_HW, 0.05, 2),
            // Image sequences
            enc!("png-8", "PNG", "8-bit", "png", Software, &[], &[], Fixed, "rgb24", 8, Rgb, &["png"], true, MAX_SW, 0.0, 1),
            enc!("png-16", "PNG", "16-bit", "png", Software, &[], &[], Fixed, "rgb48be", 16, Rgb, &["png"], true, MAX_SW, 0.0, 1),
            enc!("tiff-16", "TIFF", "16-bit deflate", "tiff", Software, &["-compression_algo", "deflate"], &[], Fixed, "rgb48le", 16, Rgb, &["tiff"], true, MAX_SW, 0.0, 1),
            enc!("jpeg", "JPEG", "Quality 95", "mjpeg", Software, &["-q:v", "2"], &[], Fixed, "yuvj444p", 8, Yuv444, &["jpg"], true, MAX_SW, 0.0, 1),
            enc!("exr", "EXR", "Half float", "exr", Software, &["-compression", "zip1", "-format", "half"], &[], Fixed, "gbrpf32le", 16, Rgb, &["exr"], true, MAX_SW, 0.0, 1),
            enc!("dpx-10", "DPX", "10-bit", "dpx", Software, &[], &[], Fixed, "gbrp10le", 10, Rgb, &["dpx"], true, MAX_SW, 0.0, 1),
            // Discards frames; used for benchmarks and dry runs.
            enc!("null", "Null", "Discard", "wrapped_avframe", Software, &[], &[], Fixed, "yuv420p", 8, Yuv420, &["null"], false, MAX_SW, 0.0, 2),
            enc!("null-10", "Null", "Discard 10-bit", "wrapped_avframe", Software, &[], &[], Fixed, "yuv420p10le", 10, Yuv420, &["null"], false, MAX_SW, 0.0, 2),
        ]
    })
}

pub fn find(id: &str) -> Option<&'static VideoEncoder> {
    video_encoders().iter().find(|e| e.id == id)
}

/// Encoders whose FFmpeg implementation is present in `available` (from `ffmpeg -encoders`).
pub fn usable<'a>(available: &'a HashSet<String>) -> impl Iterator<Item = &'static VideoEncoder> + 'a {
    video_encoders().iter().filter(move |e| available.contains(e.codec))
}

/// Same family and vendor with at least `min_depth` bits, used to upgrade 8-bit encoders for HDR.
pub fn deeper_variant(enc: &VideoEncoder, min_depth: u32) -> Option<&'static VideoEncoder> {
    video_encoders()
        .iter()
        .filter(|e| e.family == enc.family && e.vendor == enc.vendor && e.bit_depth >= min_depth && !e.image_sequence)
        .min_by_key(|e| e.bit_depth)
}

#[derive(Debug, Clone, Serialize)]
pub struct AudioCodec {
    pub id: &'static str,
    pub name: &'static str,
    pub encoder: &'static str,
    pub containers: &'static [&'static str],
    /// (min, max, default) kbit/s; `None` for lossless/PCM.
    pub bitrate: Option<(u32, u32, u32)>,
}

pub fn audio_codecs() -> &'static [AudioCodec] {
    &[
        AudioCodec { id: "aac", name: "AAC", encoder: "aac", containers: &["mp4", "mov", "mkv"], bitrate: Some((32, 512, 320)) },
        AudioCodec { id: "ac3", name: "AC-3", encoder: "ac3", containers: &["mp4", "mov", "mkv"], bitrate: Some((160, 640, 448)) },
        AudioCodec { id: "opus", name: "Opus", encoder: "libopus", containers: &["mkv", "webm", "mp4"], bitrate: Some((32, 512, 192)) },
        AudioCodec { id: "flac", name: "FLAC", encoder: "flac", containers: &["mkv", "mp4"], bitrate: None },
        AudioCodec { id: "pcm", name: "PCM 24-bit", encoder: "pcm_s24le", containers: &["mov", "mkv", "mxf"], bitrate: None },
        AudioCodec { id: "vorbis", name: "Vorbis", encoder: "libvorbis", containers: &["webm", "mkv"], bitrate: Some((64, 500, 192)) },
    ]
}

pub fn audio_codec(id: &str) -> Option<&'static AudioCodec> {
    audio_codecs().iter().find(|c| c.id == id)
}

/// Whether a stream with FFmpeg codec name `codec` can be stream-copied into `container`.
pub fn audio_copy_compatible(codec: &str, container: &str) -> bool {
    match container {
        "mkv" => true,
        "mp4" => matches!(codec, "aac" | "mp3" | "ac3" | "eac3" | "opus" | "flac" | "alac" | "dts" | "truehd"),
        "mov" => matches!(codec, "aac" | "mp3" | "ac3" | "eac3" | "alac" | "pcm_s16le" | "pcm_s16be" | "pcm_s24le" | "pcm_s24be" | "pcm_s32le" | "pcm_f32le"),
        "webm" => matches!(codec, "opus" | "vorbis"),
        "mxf" => codec.starts_with("pcm_"),
        "avi" => matches!(codec, "mp3" | "ac3" | "pcm_s16le" | "pcm_s24le"),
        _ => false,
    }
}

/// Default conversion codec for a container when copying is impossible.
pub fn default_audio_for(container: &str) -> &'static AudioCodec {
    match container {
        "webm" => audio_codec("opus").unwrap(),
        "mxf" => audio_codec("pcm").unwrap(),
        _ => audio_codec("aac").unwrap(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::settings::Quality;

    #[test]
    fn ids_are_unique_and_containers_present() {
        let mut seen = HashSet::new();
        for e in video_encoders() {
            assert!(seen.insert(e.id), "duplicate encoder id {}", e.id);
            assert!(!e.containers.is_empty());
        }
    }

    #[test]
    fn quality_and_bitrate_args() {
        let x264 = find("h264-x264").unwrap();
        let a = x264.video_args(RateControl::Quality(Quality::Medium), 1920, 1080, 30.0, "mp4");
        assert!(a.windows(2).any(|w| w == ["-crf", "20"]));
        assert!(a.windows(2).any(|w| w == ["-pix_fmt", "yuv420p"]));
        let b = x264.video_args(RateControl::Bitrate(20.0), 1920, 1080, 30.0, "mp4");
        assert!(b.windows(2).any(|w| w == ["-b:v", "20M"]));
        assert!(b.windows(2).any(|w| w == ["-maxrate", "30M"]));
        let hevc = find("hevc-nvenc-10").unwrap();
        let h = hevc.video_args(RateControl::Quality(Quality::High), 3840, 2160, 60.0, "mov");
        assert!(h.windows(2).any(|w| w == ["-tag:v", "hvc1"]));
        let amf = find("h264-amf").unwrap();
        let m = amf.video_args(RateControl::Quality(Quality::Low), 1920, 1080, 30.0, "mkv");
        assert!(m.windows(2).any(|w| w == ["-rc", "cqp"]));
        assert!(m.windows(2).any(|w| w == ["-qp_p", "28"]));
    }

    #[test]
    fn hdr_upgrade_and_audio_rules() {
        assert_eq!(deeper_variant(find("h265-x265").unwrap(), 10).unwrap().id, "h265-x265-10");
        assert_eq!(deeper_variant(find("h264-x264").unwrap(), 10).map(|e| e.id), None);
        assert!(audio_copy_compatible("aac", "mp4"));
        assert!(!audio_copy_compatible("pcm_s16le", "mp4"));
        assert!(audio_copy_compatible("pcm_s16le", "mkv"));
        assert_eq!(default_audio_for("webm").id, "opus");
    }
}
