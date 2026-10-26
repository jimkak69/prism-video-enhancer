//! App preferences (processing device, preview defaults, panel state), saved as JSON in the
//! data directory.

use std::path::PathBuf;

use prism_core::paths;
use prism_core::presets::write_atomic;
use prism_core::settings::DeviceSettings;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ViewMode {
    Single,
    Split,
    SideBySide,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Applied to every preview and export.
    pub device: DeviceSettings,
    /// Preview length in seconds; negative renders In → Out.
    pub preview_seconds: f64,
    pub live_render: bool,
    pub view: ViewMode,
    pub show_panel: bool,
    /// Built-in presets marked as favourites (user presets keep the flag in their file).
    pub favorites: Vec<String>,
    pub export_workers: usize,
    pub model_list_view: bool,
    pub show_time_remaining: bool,
    pub last_project: Option<PathBuf>,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            device: DeviceSettings::default(),
            preview_seconds: 2.0,
            live_render: true,
            view: ViewMode::Split,
            show_panel: true,
            favorites: Vec::new(),
            export_workers: 1,
            model_list_view: false,
            show_time_remaining: true,
            last_project: None,
        }
    }
}

impl Prefs {
    pub fn load() -> Prefs {
        std::fs::read_to_string(paths::settings_file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
    }

    pub fn save(&self) -> anyhow::Result<()> {
        let path = paths::settings_file();
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        write_atomic(&path, &serde_json::to_vec_pretty(self)?)
    }
}
