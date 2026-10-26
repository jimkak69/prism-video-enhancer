//! Locations of models, presets, logs and scratch space.
//!
//! Overrides: `PRISM_MODEL_DIR`, `PRISM_DATA_DIR`. Otherwise a `models` folder next to the
//! executable wins (portable installs), then the per-user data directory.

use std::path::PathBuf;

pub const APP_DIR: &str = "Prism Video Enhancer";

pub fn data_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PRISM_DATA_DIR") {
        return PathBuf::from(p);
    }
    dirs::data_dir().unwrap_or_else(std::env::temp_dir).join(APP_DIR)
}

pub fn models_dir() -> PathBuf {
    if let Some(p) = std::env::var_os("PRISM_MODEL_DIR") {
        return PathBuf::from(p);
    }
    if let Some(dir) = exe_dir().map(|d| d.join("models")).filter(|d| d.is_dir()) {
        return dir;
    }
    data_dir().join("models")
}

pub fn presets_dir() -> PathBuf {
    data_dir().join("presets")
}

pub fn logs_dir() -> PathBuf {
    data_dir().join("logs")
}

pub fn settings_file() -> PathBuf {
    data_dir().join("settings.json")
}

/// Scratch root; each job gets its own subfolder.
pub fn temp_root() -> PathBuf {
    std::env::temp_dir().join("prism-video-enhancer")
}

pub fn exe_dir() -> Option<PathBuf> {
    std::env::current_exe().ok().and_then(|p| p.parent().map(|p| p.to_path_buf()))
}
