//! Locating and interrogating FFmpeg.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use prism_core::VideoInfo;

use crate::process::command;

#[derive(Debug, Clone)]
pub struct Tools {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    /// FFmpeg release (major, minor); `None` for builds that report no version number
    /// (git snapshots), which are treated as current.
    pub version: Option<(u32, u32)>,
}

/// Parses the release from `ffmpeg -version` ("ffmpeg version 4.4.2-0ubuntu0.22.04.1",
/// "ffmpeg version n7.1-...").
pub fn parse_version(banner: &str) -> Option<(u32, u32)> {
    let v = banner.split("ffmpeg version ").nth(1)?.split_whitespace().next()?.trim_start_matches('n');
    let mut parts = v.split(|c: char| !c.is_ascii_digit());
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().and_then(|m| m.parse().ok()).unwrap_or(0);
    // Date-stamped git builds ("2024-10-10-git-...") are not release numbers.
    (major < 100).then_some((major, minor))
}

fn exe(name: &str) -> String {
    if cfg!(windows) { format!("{name}.exe") } else { name.to_string() }
}

fn search_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path).map(|d| d.join(exe(name))).find(|p| p.is_file())
}

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("PRISM_FFMPEG").map(PathBuf::from) {
        v.push(if p.is_dir() { p.join(exe("ffmpeg")) } else { p });
    }
    if let Some(dir) = prism_core::paths::exe_dir() {
        v.push(dir.join(exe("ffmpeg")));
        v.push(dir.join("ffmpeg").join("bin").join(exe("ffmpeg")));
    }
    if let Some(p) = search_path("ffmpeg") {
        v.push(p);
    }
    if cfg!(windows) {
        if let Some(local) = std::env::var_os("LOCALAPPDATA").map(PathBuf::from) {
            // winget installs (Gyan, BtbN) unpack into versioned folders.
            if let Ok(pkgs) = std::fs::read_dir(local.join("Microsoft").join("WinGet").join("Packages")) {
                for pkg in pkgs.flatten().filter(|p| p.file_name().to_string_lossy().contains("FFmpeg")) {
                    if let Ok(inner) = std::fs::read_dir(pkg.path()) {
                        for d in inner.flatten() {
                            v.push(d.path().join("bin").join("ffmpeg.exe"));
                        }
                    }
                }
            }
        }
        for d in ["C:\\ffmpeg\\bin", "C:\\Program Files\\ffmpeg\\bin", "C:\\ProgramData\\chocolatey\\bin"] {
            v.push(Path::new(d).join("ffmpeg.exe"));
        }
        if let Some(home) = std::env::var_os("USERPROFILE") {
            v.push(PathBuf::from(home).join("scoop").join("shims").join("ffmpeg.exe"));
        }
    } else {
        for d in ["/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"] {
            v.push(Path::new(d).join("ffmpeg"));
        }
    }
    v
}

impl Tools {
    /// Finds `ffmpeg` and the `ffprobe` next to it.
    pub fn locate() -> Result<Tools> {
        for ff in candidates() {
            if !ff.is_file() {
                continue;
            }
            let probe = ff.with_file_name(exe("ffprobe"));
            if probe.is_file() {
                let version = command(&ff).args(["-hide_banner", "-version"]).output().ok().and_then(|o| parse_version(&String::from_utf8_lossy(&o.stdout)));
                return Ok(Tools { ffmpeg: ff, ffprobe: probe, version });
            }
        }
        let how = if cfg!(windows) {
            "`winget install Gyan.FFmpeg`"
        } else if cfg!(target_os = "macos") {
            "`brew install ffmpeg`"
        } else {
            "your package manager, e.g. `sudo apt install ffmpeg`"
        };
        bail!("FFmpeg was not found. Install it with {how}, or set PRISM_FFMPEG to ffmpeg's path.")
    }

    /// Arguments for this FFmpeg build. Plans use the current option names; FFmpeg before 5.1
    /// spells `-fps_mode` as `-vsync` (same values), and before 4.4 has no `-stats_period`
    /// (its default interval is the same 0.5 s).
    pub fn adapt_args(&self, args: &[String]) -> Vec<String> {
        let Some(v) = self.version else { return args.to_vec() };
        let mut out = Vec::with_capacity(args.len());
        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "-fps_mode" if v < (5, 1) => out.push("-vsync".to_string()),
                "-stats_period" if v < (4, 4) => {
                    i += 2;
                    continue;
                }
                a => out.push(a.to_string()),
            }
            i += 1;
        }
        out
    }

    pub fn version(&self) -> Result<String> {
        let o = command(&self.ffmpeg).args(["-hide_banner", "-version"]).output()?;
        Ok(String::from_utf8_lossy(&o.stdout).lines().next().unwrap_or_default().to_string())
    }

    pub fn probe(&self, path: &Path) -> Result<VideoInfo> {
        if !path.exists() {
            bail!("{} does not exist", path.display());
        }
        let o = command(&self.ffprobe)
            .args(["-v", "error", "-show_streams", "-show_format", "-of", "json"])
            .arg(path)
            .output()
            .with_context(|| format!("running {}", self.ffprobe.display()))?;
        if !o.status.success() {
            bail!("ffprobe could not read {}: {}", path.display(), String::from_utf8_lossy(&o.stderr).trim());
        }
        let json: serde_json::Value = serde_json::from_slice(&o.stdout).context("parsing ffprobe output")?;
        let mut info = VideoInfo::from_ffprobe_json(path, &json)?;
        if !info.frame_count_exact && info.duration <= 0.0 {
            return Err(anyhow!("cannot determine the length of {}", path.display()));
        }
        // Containers without frame counts (mkv, webm, ts): count packets when the file is small.
        if !info.frame_count_exact && std::fs::metadata(path).map(|m| m.len() < 512 << 20).unwrap_or(false) {
            if let Ok(n) = self.count_frames(path) {
                info.frame_count = n;
                info.frame_count_exact = true;
            }
        }
        Ok(info)
    }

    fn count_frames(&self, path: &Path) -> Result<u64> {
        let o = command(&self.ffprobe)
            .args(["-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"])
            .arg(path)
            .output()?;
        String::from_utf8_lossy(&o.stdout).trim().trim_end_matches(',').parse::<u64>().map_err(|e| anyhow!("{e}"))
    }

    fn list(&self, flag: &str) -> Result<HashSet<String>> {
        let o = command(&self.ffmpeg).args(["-hide_banner", flag]).output()?;
        let text = String::from_utf8_lossy(&o.stdout);
        // Lines look like " V....D libx264    description"; the name is the second column.
        Ok(text.lines().filter_map(|l| {
            let mut it = l.split_whitespace();
            let flags = it.next()?;
            let name = it.next()?;
            (flags.len() >= 2 && !flags.contains('=') && flags != "------").then(|| name.to_string())
        }).collect())
    }

    /// FFmpeg encoder names in this build (cached per binary).
    pub fn encoders(&self) -> Result<HashSet<String>> {
        static CACHE: OnceLock<Mutex<Vec<(PathBuf, HashSet<String>)>>> = OnceLock::new();
        let cache = CACHE.get_or_init(Default::default);
        if let Some((_, s)) = cache.lock().unwrap().iter().find(|(p, _)| *p == self.ffmpeg) {
            return Ok(s.clone());
        }
        let s = self.list("-encoders")?;
        cache.lock().unwrap().push((self.ffmpeg.clone(), s.clone()));
        Ok(s)
    }

    pub fn filters(&self) -> Result<HashSet<String>> {
        self.list("-filters")
    }

    /// Encodes a few frames to check that a (hardware) encoder actually works on this machine.
    pub fn test_encoder(&self, codec: &str, pix_fmt: &str) -> bool {
        command(&self.ffmpeg)
            .args(["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=256x256:rate=10:duration=0.3", "-c:v", codec, "-pix_fmt", pix_fmt, "-f", "null", "-"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }
}

#[cfg(test)]
mod version_tests {
    use super::*;

    fn tools(version: Option<(u32, u32)>) -> Tools {
        Tools { ffmpeg: PathBuf::from("ffmpeg"), ffprobe: PathBuf::from("ffprobe"), version }
    }

    #[test]
    fn parses_release_banners() {
        assert_eq!(parse_version("ffmpeg version 4.4.2-0ubuntu0.22.04.1 Copyright (c)"), Some((4, 4)));
        assert_eq!(parse_version("ffmpeg version n7.1-3-g0123 Copyright"), Some((7, 1)));
        assert_eq!(parse_version("ffmpeg version 9.0.2-full_build-www.gyan.dev Copyright"), Some((9, 0)));
        assert_eq!(parse_version("ffmpeg version 2024-10-10-git-abc-full_build Copyright"), None);
        assert_eq!(parse_version("ffmpeg version N-117000-gabc Copyright"), None);
    }

    #[test]
    fn rewrites_options_for_old_ffmpeg() {
        let args: Vec<String> = ["-stats", "-stats_period", "0.5", "-i", "a.mp4", "-fps_mode", "cfr", "-r", "30", "out.mp4"].iter().map(|s| s.to_string()).collect();
        assert_eq!(tools(Some((7, 1))).adapt_args(&args), args);
        assert_eq!(tools(None).adapt_args(&args), args);
        let v44 = tools(Some((4, 4))).adapt_args(&args);
        assert_eq!(v44, ["-stats", "-stats_period", "0.5", "-i", "a.mp4", "-vsync", "cfr", "-r", "30", "out.mp4"]);
        let v43 = tools(Some((4, 3))).adapt_args(&args);
        assert_eq!(v43, ["-stats", "-i", "a.mp4", "-vsync", "cfr", "-r", "30", "out.mp4"]);
    }
}
