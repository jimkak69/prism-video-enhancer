//! Progress tracking: FPS smoothing, ETA and percent, plus parsing of FFmpeg `frame= fps=` lines.

use std::sync::OnceLock;
use std::time::Instant;

use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Progress {
    /// 1-based pass number and pass count.
    pub pass: usize,
    pub passes: usize,
    pub frame: u64,
    pub total: u64,
    /// Smoothed frames per second.
    pub fps: f64,
    pub eta_secs: f64,
    pub elapsed_secs: f64,
    pub percent: u32,
    /// All frames are written; muxing and cleanup remain.
    pub finalizing: bool,
}

/// Turns a stream of frame counts into smoothed progress.
///
/// FFmpeg's own fps column is unreliable, so speed is derived from frame deltas over wall time and
/// smoothed with `fps = 0.9 * previous + 0.1 * new`. The first interval (start-up) is ignored.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    total: u64,
    start: Instant,
    last: Option<(u64, Instant)>,
    samples: u32,
    fps: f64,
}

impl ProgressTracker {
    pub fn new(total: u64) -> Self {
        ProgressTracker { total: total.max(1), start: Instant::now(), last: None, samples: 0, fps: 0.0 }
    }

    pub fn total(&self) -> u64 {
        self.total
    }

    pub fn update(&mut self, frame: u64, now: Instant) -> Progress {
        if let Some((pf, pt)) = self.last {
            let dt = now.duration_since(pt).as_secs_f64();
            if self.samples > 1 && frame > pf && dt > 0.0 {
                let inst = (frame - pf) as f64 / dt;
                self.fps = if self.fps > 0.0 { 0.9 * self.fps + 0.1 * inst } else { inst };
            }
        }
        self.last = Some((frame, now));
        self.samples += 1;
        let frame_c = frame.min(self.total);
        let eta = if self.fps > 0.0 { (self.total - frame_c) as f64 / self.fps } else { 0.0 };
        Progress {
            pass: 0,
            passes: 0,
            frame,
            total: self.total,
            fps: self.fps,
            eta_secs: eta,
            elapsed_secs: now.duration_since(self.start).as_secs_f64(),
            percent: ((100 * frame_c) / self.total) as u32,
            finalizing: frame >= self.total,
        }
    }
}

/// Parses an FFmpeg stats line (`frame=  766 fps= 59 q=4.0 ...`) into (frame, fps).
pub fn parse_ffmpeg_stats(line: &str) -> Option<(u64, f64)> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let re = RE.get_or_init(|| Regex::new(r"frame=\s*(\d+)\s+fps=\s*([0-9.]+)").unwrap());
    let c = re.captures_iter(line).last()?;
    Some((c[1].parse().ok()?, c[2].parse().unwrap_or(0.0)))
}

pub fn format_duration(secs: f64) -> String {
    if !secs.is_finite() || secs <= 0.0 {
        return "--:--".into();
    }
    let s = secs.round() as u64;
    if s >= 3600 { format!("{}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60) } else { format!("{:02}:{:02}", s / 60, s % 60) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn parses_stats_lines() {
        assert_eq!(parse_ffmpeg_stats("frame=  766 fps= 59 q=4.0 Lsize=     2kB"), Some((766, 59.0)));
        assert_eq!(parse_ffmpeg_stats("frame=12 fps=0.0 q=0.0 size=0kB\rframe=  40 fps= 7.5 q=1"), Some((40, 7.5)));
        assert_eq!(parse_ffmpeg_stats("Stream mapping:"), None);
    }

    #[test]
    fn smoothing_eta_and_percent() {
        let t0 = Instant::now();
        let mut p = ProgressTracker::new(100);
        p.update(0, t0);
        // The start-up interval is ignored.
        let a = p.update(10, t0 + Duration::from_secs(1));
        assert_eq!(a.fps, 0.0);
        let b = p.update(30, t0 + Duration::from_secs(2));
        assert!((b.fps - 20.0).abs() < 1e-9);
        let c = p.update(40, t0 + Duration::from_secs(3));
        assert!((c.fps - (0.9 * 20.0 + 0.1 * 10.0)).abs() < 1e-9);
        assert_eq!(c.percent, 40);
        assert!((c.eta_secs - 60.0 / 19.0).abs() < 1e-9);
        assert!(p.update(100, t0 + Duration::from_secs(9)).finalizing);
    }

    #[test]
    fn durations() {
        assert_eq!(format_duration(65.2), "01:05");
        assert_eq!(format_duration(3725.0), "1:02:05");
        assert_eq!(format_duration(0.0), "--:--");
    }
}
