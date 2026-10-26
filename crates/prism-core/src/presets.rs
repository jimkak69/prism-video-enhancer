//! Presets: named filter states. Built-ins live in code; user presets are one JSON file each.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};

use crate::settings::*;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Preset {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub favorite: bool,
    #[serde(skip)]
    pub builtin: bool,
    pub settings: JobSettings,
}

impl Preset {
    /// Captures the filter state of `s`, leaving out per-clip choices (trim, crop, output path).
    pub fn capture(name: &str, description: &str, s: &JobSettings) -> Preset {
        let mut settings = s.clone();
        settings.trim = None;
        settings.crop = None;
        settings.export.output_path = None;
        Preset { name: name.into(), description: description.into(), favorite: false, builtin: false, settings }
    }

    /// Applies the preset to `target`, keeping its per-clip choices and device preferences.
    pub fn apply(&self, target: &mut JobSettings) {
        let keep_trim = target.trim;
        let keep_crop = target.crop;
        let keep_path = target.export.output_path.clone();
        let keep_dir = target.export.output_dir.clone();
        let keep_device = target.device.clone();
        *target = self.settings.clone();
        target.trim = keep_trim;
        target.crop = keep_crop;
        target.export.output_path = keep_path;
        if target.export.output_dir.is_none() {
            target.export.output_dir = keep_dir;
        }
        target.device = keep_device;
    }
}

fn builtin(name: &str, description: &str, f: impl FnOnce(&mut JobSettings)) -> Preset {
    let mut settings = JobSettings::default();
    f(&mut settings);
    Preset { name: name.into(), description: description.into(), favorite: false, builtin: true, settings }
}

pub fn builtins() -> Vec<Preset> {
    vec![
        builtin("Upscale to 4K", "Enhance and upscale to 3840x2160 (aspect ratio kept).", |s| {
            s.enhancement.enabled = true;
            s.output.resolution = Resolution::uhd4k();
        }),
        builtin("Upscale to FHD", "Enhance and upscale to 1920x1080 (aspect ratio kept).", |s| {
            s.enhancement.enabled = true;
            s.output.resolution = Resolution::fhd();
        }),
        builtin("2x Upscale", "Enhance at twice the resolution.", |s| {
            s.enhancement.enabled = true;
            s.output.resolution = Resolution::Scale(2.0);
        }),
        builtin("Convert to 60 fps", "Motion-compensated frame-rate conversion to 60 fps.", |s| {
            s.interpolation.enabled = true;
            s.interpolation.output_fps = Some(60.0);
        }),
        builtin("4x slow motion", "Four times slower with interpolated frames.", |s| {
            s.interpolation.enabled = true;
            s.interpolation.slowmo = 4.0;
        }),
        builtin("8x super slow motion", "Eight times slower with interpolated frames.", |s| {
            s.interpolation.enabled = true;
            s.interpolation.slowmo = 8.0;
        }),
        builtin("Auto crop stabilization", "Smooth camera shake and zoom to hide borders.", |s| {
            s.stabilization.enabled = true;
            s.stabilization.mode = StabilizationMode::AutoCrop;
        }),
        builtin("Deinterlace and upscale to FHD", "Deinterlace TV footage and enhance to 1080p.", |s| {
            s.enhancement.enabled = true;
            s.enhancement.video_type = VideoType::Interlaced;
            s.output.resolution = Resolution::fhd();
        }),
        builtin("Clean up noise", "Strong denoise and deblocking at the original size.", |s| {
            s.enhancement.enabled = true;
            s.enhancement.param_mode = ParamMode::Manual;
            s.enhancement.noise = 60;
            s.enhancement.compression = 50;
            s.enhancement.details = 20;
            s.enhancement.recover_original = 0;
        }),
        builtin("SDR to HDR10", "Inverse tone mapping to HDR10 with a 10-bit HEVC export.", |s| {
            s.hdr.enabled = true;
            s.export.encoder = "h265-x265-10".into();
        }),
    ]
}

pub struct PresetStore {
    dir: PathBuf,
}

impl PresetStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        PresetStore { dir: dir.into() }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Built-ins followed by user presets (alphabetical). Unreadable files are reported, not fatal.
    pub fn list(&self) -> (Vec<Preset>, Vec<String>) {
        let mut out = builtins();
        let mut errors = Vec::new();
        let mut user = Vec::new();
        if let Ok(entries) = std::fs::read_dir(&self.dir) {
            for e in entries.flatten() {
                let p = e.path();
                if p.extension().is_some_and(|x| x == "json") {
                    match std::fs::read_to_string(&p).map_err(anyhow::Error::from).and_then(|t| Ok(serde_json::from_str::<Preset>(&t)?)) {
                        Ok(preset) => user.push(preset),
                        Err(err) => errors.push(format!("{}: {err}", p.display())),
                    }
                }
            }
        }
        user.sort_by_key(|p| p.name.to_lowercase());
        out.extend(user);
        (out, errors)
    }

    pub fn find(&self, name: &str) -> Option<Preset> {
        self.list().0.into_iter().find(|p| p.name.eq_ignore_ascii_case(name))
    }

    pub fn save(&self, preset: &Preset) -> Result<PathBuf> {
        if builtins().iter().any(|b| b.name.eq_ignore_ascii_case(&preset.name)) {
            bail!("'{}' is a built-in preset name", preset.name);
        }
        std::fs::create_dir_all(&self.dir).with_context(|| format!("creating {}", self.dir.display()))?;
        let path = self.path_for(&preset.name);
        write_atomic(&path, &serde_json::to_vec_pretty(preset)?)?;
        Ok(path)
    }

    pub fn delete(&self, name: &str) -> Result<()> {
        let path = self.path_for(name);
        std::fs::remove_file(&path).with_context(|| format!("deleting {}", path.display()))
    }

    fn path_for(&self, name: &str) -> PathBuf {
        let safe: String = name.chars().map(|c| if c.is_alphanumeric() || " -_()".contains(c) { c } else { '_' }).collect();
        self.dir.join(format!("{}.json", safe.trim()))
    }
}

/// Writes through a temporary file and renames, so readers never see a partial file.
pub fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_keeps_per_clip_choices() {
        let mut target = JobSettings::default();
        target.trim = Some(TrimRange { start: 1.0, end: 2.0 });
        target.device.backend = Backend::Cpu;
        let p = builtins().into_iter().find(|p| p.name == "Upscale to 4K").unwrap();
        p.apply(&mut target);
        assert!(target.enhancement.enabled);
        assert_eq!(target.trim, Some(TrimRange { start: 1.0, end: 2.0 }));
        assert_eq!(target.device.backend, Backend::Cpu);
    }

    #[test]
    fn store_round_trip() {
        let dir = std::env::temp_dir().join(format!("prism-presets-{}", std::process::id()));
        let store = PresetStore::new(&dir);
        let mut s = JobSettings::default();
        s.grain.enabled = true;
        s.trim = Some(TrimRange { start: 0.0, end: 1.0 });
        let p = Preset::capture("My: grain", "", &s);
        assert!(p.settings.trim.is_none());
        store.save(&p).unwrap();
        let found = store.find("my: grain").unwrap();
        assert!(found.settings.grain.enabled);
        assert!(store.save(&Preset::capture("Upscale to 4K", "", &s)).is_err());
        store.delete("My: grain").unwrap();
        assert!(store.find("My: grain").is_none());
        std::fs::remove_dir_all(dir).ok();
    }
}
