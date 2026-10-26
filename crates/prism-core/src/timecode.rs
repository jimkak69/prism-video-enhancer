//! Timecode formatting and parsing.

/// `HH:MM:SS:FF` (SMPTE, non-drop) for `secs` at `fps`.
pub fn smpte(secs: f64, fps: f64) -> String {
    let fps_i = fps.round().max(1.0) as u64;
    let total = (secs.max(0.0) * fps).round() as u64;
    let ff = total % fps_i;
    let s = total / fps_i;
    format!("{:02}:{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60, ff)
}

/// Parses `83.5`, `1:23.5`, `0:01:23.5` or SMPTE `00:01:23:12` (needs `fps`) into seconds.
pub fn parse(text: &str, fps: f64) -> Option<f64> {
    let parts: Vec<&str> = text.trim().split(':').collect();
    let num = |s: &str| s.trim().parse::<f64>().ok().filter(|v| v.is_finite() && *v >= 0.0);
    match parts.as_slice() {
        [s] => num(s),
        [m, s] => Some(num(m)? * 60.0 + num(s)?),
        [h, m, s] => Some(num(h)? * 3600.0 + num(m)? * 60.0 + num(s)?),
        [h, m, s, f] if fps > 0.0 => Some(num(h)? * 3600.0 + num(m)? * 60.0 + num(s)? + num(f)? / fps.round()),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        assert_eq!(smpte(83.5, 24.0), "00:01:23:12");
        assert_eq!(parse("00:01:23:12", 24.0), Some(83.5));
        assert_eq!(parse("1:23.5", 24.0), Some(83.5));
        assert_eq!(parse("83.5", 0.0), Some(83.5));
        assert_eq!(parse("x", 24.0), None);
    }
}
