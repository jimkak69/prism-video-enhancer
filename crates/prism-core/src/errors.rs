//! Maps FFmpeg and engine messages to a category and user-facing advice.

use std::sync::OnceLock;

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct Diagnosis {
    /// Short category, e.g. "Out of memory".
    pub category: String,
    /// Advice lines, most specific first, without duplicates.
    pub advice: Vec<String>,
}

impl Diagnosis {
    pub fn is_empty(&self) -> bool {
        self.advice.is_empty()
    }
}

struct Rule {
    re: Regex,
    category: &'static str,
    advice: &'static str,
}

fn rules() -> &'static [Rule] {
    static RULES: OnceLock<Vec<Rule>> = OnceLock::new();
    RULES.get_or_init(|| {
        let r = |p: &str, c: &'static str, a: &'static str| Rule { re: Regex::new(p).unwrap(), category: c, advice: a };
        vec![
            r(r"Unrecognized option '?([^'.]+)'?", "Command error", "FFmpeg does not know the option '$1'. The FFmpeg build may be too old."),
            r(r"Value (\S+) for parameter '(\w+)' out of range", "Command error", "Invalid value $1 for option $2."),
            r(r"Error setting option (\w+) to value (\S+)", "Command error", "Could not set option $1 to $2."),
            r(r"codec not currently supported in container|Could not find tag for codec", "Codec error", "The container does not support this codec. Choose another container (mkv accepts almost everything) or set audio to Convert."),
            r(r"Unsupported channel layout", "Audio error", "The audio channel layout is not supported by the encoder. Convert the audio or set it to None."),
            r(r"No space left on device", "No space left", "The output drive is full. Free some space or choose another output folder."),
            r(r"Unknown encoder '([^']+)'|Encoder not found|Codec not supported", "Codec error", "The encoder is not available in this FFmpeg build. Choose a different encoder."),
            r(r"No capable devices found|OpenEncodeSessionEx failed", "Encoder error", "The GPU encoder could not start. Some GPUs allow only a few concurrent encode sessions; lower the number of parallel exports or use a software encoder."),
            r(r"Driver does not support the required nvenc API version\. Required: ([\d.]+) Found: ([\d.]+)", "Driver error", "Update the NVIDIA driver: NVENC API $1 is required, the driver provides $2."),
            r(r"minimum required Nvidia driver for nvenc is ([\d.]+)", "Driver error", "Update the NVIDIA driver to $1 or newer."),
            r(r"Picture size (\d+x\d+) is invalid", "Resolution error", "Invalid output resolution $1. Change the output size or encoder."),
            r(r"(Width|Height) (\d+) exceeds (\d+)", "Resolution error", "$1 $2 exceeds the encoder limit of $3. Lower the output resolution or pick a software encoder."),
            r(r"Frame Dimension less than the minimum supported value", "Resolution error", "The output is smaller than the encoder minimum. Increase the output size."),
            r(r"Error while opening encoder|Could not open encoder before EOF", "Encoder error", "The encoder rejected its settings. Try a different resolution, bitrate or encoder."),
            r(r"Cannot allocate memory|out of memory|E_OUTOFMEMORY|bad_alloc", "Out of memory", "The process ran out of memory. Lower the GPU memory fraction or tile size, reduce parallel exports, or close other applications."),
            r(r"Impossible to convert between the formats", "Colorspace error", "FFmpeg cannot convert between these pixel formats. Reset colour overrides to automatic."),
            r(r"Invalid data found when processing input|moov atom not found", "Input error", "The input file is damaged or incomplete."),
            r(r"Permission denied|Access is denied", "Permission error", "Prism cannot write to the output location. Choose another folder."),
            r(r"vidstab|Transformation file", "Stabilization error", "Camera-motion analysis failed. Try a lower smoothness or disable stabilization for this clip."),
            r(r":AIERROR:\s*(.+)", "AI error", "Engine error: $1"),
        ]
    })
}

/// Analyses captured output lines; later matches add advice, the first category wins.
pub fn diagnose<'a>(lines: impl IntoIterator<Item = &'a str>) -> Diagnosis {
    let mut d = Diagnosis::default();
    for line in lines {
        for rule in rules() {
            if let Some(c) = rule.re.captures(line) {
                let mut advice = String::new();
                c.expand(rule.advice, &mut advice);
                if !d.advice.contains(&advice) {
                    d.advice.push(advice);
                }
                if d.category.is_empty() {
                    d.category = rule.category.into();
                }
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diagnoses_common_failures() {
        let d = diagnose(["[mp4 @ 0x1] Could not find tag for codec pcm_s16le in stream #1, codec not currently supported in container"]);
        assert_eq!(d.category, "Codec error");
        let d = diagnose(["Width 9000 exceeds 8192", "No space left on device"]);
        assert_eq!(d.category, "Resolution error");
        assert_eq!(d.advice.len(), 2);
        assert!(d.advice[0].contains("9000"));
        let d = diagnose([":AIERROR: tile inference failed"]);
        assert_eq!(d.advice, vec!["Engine error: tile inference failed".to_string()]);
        assert!(diagnose(["all good"]).is_empty());
    }
}
