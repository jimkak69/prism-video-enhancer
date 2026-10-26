//! Projects: sources, their comparison tabs (named filter states) and export history.
//! Saved as `<project dir>/project.prism.json`.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::presets::write_atomic;
use crate::settings::JobSettings;

pub const PROJECT_FILE: &str = "project.prism.json";
const VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tab {
    pub name: String,
    pub settings: JobSettings,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExportStatus {
    Queued,
    Running,
    Paused,
    Done,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportRecord {
    pub output: PathBuf,
    pub tab: String,
    pub status: ExportStatus,
    #[serde(default)]
    pub message: String,
    #[serde(default)]
    pub elapsed_secs: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Source {
    pub path: PathBuf,
    pub tabs: Vec<Tab>,
    pub active_tab: usize,
    #[serde(default)]
    pub exports: Vec<ExportRecord>,
}

impl Source {
    pub fn new(path: PathBuf, settings: JobSettings) -> Self {
        Source { path, tabs: vec![Tab { name: "Enhanced".into(), settings }], active_tab: 0, exports: Vec::new() }
    }

    pub fn active(&self) -> &Tab {
        &self.tabs[self.active_tab.min(self.tabs.len() - 1)]
    }

    pub fn active_mut(&mut self) -> &mut Tab {
        let i = self.active_tab.min(self.tabs.len() - 1);
        &mut self.tabs[i]
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Project {
    pub version: u32,
    pub name: String,
    pub sources: Vec<Source>,
    pub selected: Option<usize>,
}

impl Default for Project {
    fn default() -> Self {
        Project { version: VERSION, name: "Default".into(), sources: Vec::new(), selected: None }
    }
}

impl Project {
    pub fn load(dir: &Path) -> Result<Project> {
        let path = dir.join(PROJECT_FILE);
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        let mut p: Project = serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
        for s in &mut p.sources {
            if s.tabs.is_empty() {
                s.tabs.push(Tab { name: "Enhanced".into(), settings: JobSettings::default() });
            }
            s.active_tab = s.active_tab.min(s.tabs.len() - 1);
        }
        p.selected = p.selected.filter(|i| *i < p.sources.len());
        Ok(p)
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        write_atomic(&dir.join(PROJECT_FILE), &serde_json::to_vec_pretty(self)?)
    }

    pub fn add_source(&mut self, path: PathBuf, settings: JobSettings) -> usize {
        if let Some(i) = self.sources.iter().position(|s| s.path == path) {
            self.selected = Some(i);
            return i;
        }
        self.sources.push(Source::new(path, settings));
        let i = self.sources.len() - 1;
        self.selected = Some(i);
        i
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_and_load_repairs_indices() {
        let dir = std::env::temp_dir().join(format!("prism-proj-{}", std::process::id()));
        let mut p = Project::default();
        let i = p.add_source(PathBuf::from("a.mp4"), JobSettings::default());
        assert_eq!(p.add_source(PathBuf::from("a.mp4"), JobSettings::default()), i);
        p.sources[0].active_tab = 9;
        p.save(&dir).unwrap();
        let q = Project::load(&dir).unwrap();
        assert_eq!(q.sources.len(), 1);
        assert_eq!(q.sources[0].active_tab, 0);
        std::fs::remove_dir_all(dir).ok();
    }
}
