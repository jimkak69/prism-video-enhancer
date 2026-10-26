//! Application state, background work and the main window: menu row, header, comparison tabs,
//! viewer, transport, timeline, bottom bar and the right pane. Popovers and dialogs are in
//! `panels.rs`.

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, unbounded};
use egui::{Align, Align2, Color32, CursorIcon, Layout, Pos2, Rect, Sense, Stroke, StrokeKind, TextureHandle, TextureOptions, Ui, Vec2, pos2, vec2};
use prism_core::models::Catalog;
use prism_core::plan::{PlanContext, PreviewRange, plan};
use prism_core::presets::{Preset, PresetStore};
use prism_core::progress::Progress;
use prism_core::project::{Project, Source, Tab};
use prism_core::settings::{FocusFix, ParamMode, Resolution, TrimRange};
use prism_core::{JobSettings, VideoInfo, paths, timecode};
use prism_engine::queue::Queue;
use prism_engine::{Control, Tools, runner};

use crate::icons::{self, Icon};
use crate::media::{self, Player, RgbaImage};
use crate::prefs::{Prefs, ViewMode};
use crate::settings_ui::{self, Action, PaneContext};
use crate::theme::{self, ACCENT, BG, HOVER, LABEL, MUTED, PANEL, RAISED, TEXT};
use crate::widgets;

pub(crate) const VIDEO_EXTS: &[&str] = &["mp4", "mov", "mkv", "avi", "webm", "m4v", "mts", "m2ts", "ts", "mxf", "wmv", "flv", "mpg", "mpeg", "vob", "3gp", "y4m"];
const RIGHT_W: f32 = 360.0;
const VIEWER_BG: Color32 = Color32::from_rgb(0x12, 0x12, 0x12);

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RightTab {
    Adjustments,
    Codec,
}

/// What one side of the viewer shows.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Content {
    Original,
    Tab(usize),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Popover {
    Sources,
    Queue,
    Presets,
    InputSettings,
    Projects,
}

pub(crate) enum Dialog {
    Preferences,
    Models,
    About,
    Command { title: String, text: String },
    NewPreset { name: String, description: String },
    RenamePreset { from: String, name: String },
    NewProject { name: String },
    CloseAll,
}

pub(crate) struct Tex {
    pub handle: TextureHandle,
    pub key: String,
}

pub(crate) struct Pending<T> {
    pub rx: Receiver<T>,
    pub key: String,
}

/// The processed picture for one viewer side.
#[derive(Default)]
pub(crate) struct Slot {
    pub tex: Option<Tex>,
    pub pending: Option<(Pending<Result<RgbaImage, String>>, Option<Arc<Control>>)>,
    pub note: String,
    /// Key of the last render that failed, so it is not retried every frame.
    pub failed: String,
}

impl Slot {
    fn cancel(&mut self) {
        if let Some((_, Some(c))) = self.pending.take() {
            c.cancel();
        }
    }
}

pub(crate) struct PreviewClip {
    pub source: PathBuf,
    pub settings_hash: u64,
    pub start: f64,
    pub duration: f64,
    pub path: PathBuf,
    /// Output seconds per source second (slow motion).
    pub stretch: f64,
    pub fps: f64,
    pub size: (usize, usize),
}

pub(crate) struct PreviewJob {
    pub rx: Receiver<Result<PreviewClip, String>>,
    pub progress: Arc<Mutex<Progress>>,
    pub control: Arc<Control>,
    pub total: usize,
    pub done: usize,
    /// Stop was requested; the render thread is winding down.
    pub stopping: bool,
}

struct SidePlayer {
    player: Player,
    stretch: f64,
}

struct Playback {
    started: Instant,
    from: f64,
    original: Player,
    sides: [Option<SidePlayer>; 2],
}

/// Crop being edited, normalised to the displayed frame.
#[derive(Clone, Copy)]
pub(crate) struct CropEdit {
    pub rect: Rect,
    drag: Option<(u8, Pos2, Rect)>,
}

pub struct PrismApp {
    pub(crate) tools: Option<Tools>,
    pub(crate) catalog: Arc<Catalog>,
    pub(crate) encoders: HashSet<String>,
    pub(crate) queue: Option<Arc<Queue>>,
    pub(crate) runtime_status: Arc<Mutex<String>>,
    pub(crate) prefs: Prefs,
    pub(crate) project: Project,
    pub(crate) project_dir: PathBuf,
    dirty_since: Option<Instant>,
    prefs_dirty: bool,
    pub(crate) infos: HashMap<PathBuf, VideoInfo>,
    pub(crate) probes: Vec<(PathBuf, Receiver<Result<VideoInfo, String>>, bool)>,
    pub(crate) source_thumbs: HashMap<PathBuf, TextureHandle>,
    source_thumb_rx: Vec<(PathBuf, Receiver<Option<RgbaImage>>)>,
    pub(crate) message: Option<(String, bool, Instant)>,
    pub(crate) show_original: bool,
    pub(crate) compare: Option<usize>,
    zoom: f32,
    center: Vec2,
    split: f32,
    /// The split divider is being dragged.
    split_drag: bool,
    fit_scale: f32,
    pub(crate) full_preview: bool,
    pub(crate) playhead: f64,
    playback: Option<Playback>,
    original: Option<Tex>,
    grab: Option<Pending<Result<RgbaImage, String>>>,
    slots: [Slot; 2],
    clips: Vec<PreviewClip>,
    pub(crate) preview_job: Option<PreviewJob>,
    thumbs: Vec<TextureHandle>,
    thumbs_for: Option<PathBuf>,
    thumbs_rx: Option<Receiver<Vec<RgbaImage>>>,
    timeline_zoom: f32,
    timeline_start: f64,
    pub(crate) right_tab: RightTab,
    pub(crate) popover: Option<Popover>,
    pub(crate) popover_opened: bool,
    pub(crate) dialog: Option<Dialog>,
    pub(crate) presets: PresetStore,
    pub(crate) preset_list: Vec<Preset>,
    pub(crate) presets_custom: bool,
    pub(crate) fetching: Option<(String, Receiver<Result<Vec<PathBuf>, String>>)>,
    estimating: Option<(bool, Receiver<Result<prism_core::plan::EnhanceParams, String>>)>,
    clipboard: Option<JobSettings>,
    pub(crate) cropping: Option<CropEdit>,
    renaming_tab: Option<(usize, String)>,
    timecode_edit: Option<String>,
    maximized_once: bool,
    pub(crate) queue_newest_first: bool,
    pub(crate) queue_selected_only: bool,
    pub(crate) presets_button: Rect,
    /// Anchors for popovers, recorded while laying out the frame.
    pub(crate) central_rect: Rect,
    pub(crate) bottom_top: f32,
    pub(crate) crumb_anchor: Pos2,
}

/// New sources start with enhancement on (Classic, Auto parameters).
pub(crate) fn default_settings() -> JobSettings {
    let mut s = JobSettings::default();
    s.enhancement.enabled = true;
    s
}

fn settings_hash(s: &JobSettings) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    serde_json::to_string(s).unwrap_or_default().hash(&mut h);
    h.finish()
}

pub(crate) fn spawn<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static, ctx: &egui::Context) -> Receiver<T> {
    let (tx, rx) = unbounded();
    let ctx = ctx.clone();
    std::thread::spawn(move || {
        let _ = tx.send(f());
        ctx.request_repaint();
    });
    rx
}

/// Short descriptions of the active filters, for the breadcrumb under the tabs.
pub(crate) fn adjustments(s: &JobSettings, catalog: &Catalog) -> Vec<(&'static str, String)> {
    let model_name = |id: &str| catalog.get(id).map(|m| m.name.clone()).unwrap_or_else(|| id.to_string());
    let mode = |m: ParamMode| match m {
        ParamMode::Auto => "Auto",
        ParamMode::RelativeToAuto => "Relative",
        ParamMode::Manual => "Manual",
    };
    let mut out = Vec::new();
    if s.enhancement.enabled {
        let mut d = format!("{}, {}", model_name(&s.enhancement.model), mode(s.enhancement.param_mode));
        if s.enhancement.focus_fix != FocusFix::Off {
            d.push_str(", focus fix");
        }
        out.push(("Enhancement", d));
    } else if s.output.resolution != Resolution::Original {
        out.push(("Resize", s.output.resolution.label()));
    }
    if s.second_enhancement.enabled {
        out.push(("2nd Enhancement", model_name(&s.second_enhancement.settings.model)));
    }
    if s.interpolation.enabled {
        let mut d = model_name(&s.interpolation.model);
        if let Some(f) = s.interpolation.output_fps {
            d.push_str(&format!(", {f} fps"));
        }
        if s.interpolation.slowmo > 1.0 {
            d.push_str(&format!(", {}x slow motion", s.interpolation.slowmo));
        }
        out.push(("Frame interpolation", d));
    }
    if s.stabilization.enabled {
        out.push(("Stabilization", format!("strength {}", s.stabilization.smoothness)));
    }
    if s.motion_deblur.enabled {
        out.push(("Motion deblur", format!("strength {}", s.motion_deblur.strength)));
    }
    if s.hdr.enabled {
        out.push(("HDR", match s.hdr.transfer {
            prism_core::settings::HdrTransfer::Pq => "PQ".into(),
            prism_core::settings::HdrTransfer::Hlg => "HLG".into(),
        }));
    }
    if s.grain.enabled {
        out.push(("Grain", format!("{}", s.grain.amount)));
    }
    out
}

impl PrismApp {
    pub fn new(cc: &eframe::CreationContext<'_>, files: Vec<PathBuf>) -> Self {
        theme::apply(&cc.egui_ctx);
        let prefs = Prefs::load();
        let tools = Tools::locate();
        let catalog = Arc::new(Catalog::load(&paths::models_dir()));
        let (tools, mut message) = match tools {
            Ok(t) => (Some(t), None),
            Err(e) => (None, Some((format!("{e:#}"), true, Instant::now()))),
        };
        let encoders = tools.as_ref().and_then(|t| t.encoders().ok()).unwrap_or_default();
        let queue = tools.as_ref().map(|t| Queue::new(t.clone(), catalog.clone(), prefs.export_workers.max(1)));
        let runtime_status = Arc::new(Mutex::new(String::from("Loading ONNX Runtime…")));
        {
            let status = runtime_status.clone();
            let ctx = cc.egui_ctx.clone();
            std::thread::spawn(move || {
                let s = match prism_engine::ai::runtime::runtime() {
                    Ok(rt) => format!("ONNX Runtime: {}", rt.providers.join(", ")),
                    Err(_) => "ONNX Runtime unavailable: network models are disabled; Classic and Flow still work".into(),
                };
                *status.lock().unwrap() = s;
                ctx.request_repaint();
            });
        }
        let default_dir = paths::data_dir().join("projects").join("default");
        let project_dir = prefs.last_project.clone().filter(|d| d.join(prism_core::project::PROJECT_FILE).is_file()).unwrap_or(default_dir);
        let project = Project::load(&project_dir).unwrap_or_default();
        let presets = PresetStore::new(paths::presets_dir());
        let (preset_list, errs) = presets.list();
        if let Some(e) = errs.first() {
            message.get_or_insert((format!("Could not read a preset: {e}"), true, Instant::now()));
        }
        let mut app = PrismApp {
            tools,
            catalog,
            encoders,
            queue,
            runtime_status,
            prefs,
            project,
            project_dir,
            dirty_since: None,
            prefs_dirty: false,
            infos: HashMap::new(),
            probes: Vec::new(),
            source_thumbs: HashMap::new(),
            source_thumb_rx: Vec::new(),
            message,
            show_original: false,
            compare: None,
            zoom: 1.0,
            center: Vec2::splat(0.5),
            split: 0.5,
            split_drag: false,
            fit_scale: 1.0,
            full_preview: false,
            playhead: 0.0,
            playback: None,
            original: None,
            grab: None,
            slots: Default::default(),
            clips: Vec::new(),
            preview_job: None,
            thumbs: Vec::new(),
            thumbs_for: None,
            thumbs_rx: None,
            timeline_zoom: 1.0,
            timeline_start: 0.0,
            right_tab: RightTab::Adjustments,
            popover: None,
            popover_opened: false,
            dialog: None,
            presets,
            preset_list,
            presets_custom: false,
            fetching: None,
            estimating: None,
            clipboard: None,
            cropping: None,
            renaming_tab: None,
            timecode_edit: None,
            maximized_once: false,
            queue_newest_first: true,
            queue_selected_only: false,
            presets_button: Rect::NOTHING,
            central_rect: Rect::NOTHING,
            bottom_top: 0.0,
            crumb_anchor: Pos2::ZERO,
        };
        app.restore_sources(&cc.egui_ctx);
        for p in files {
            app.add_source(p, true, &cc.egui_ctx);
        }
        app
    }

    fn restore_sources(&mut self, ctx: &egui::Context) {
        let restored: Vec<PathBuf> = self.project.sources.iter().map(|s| s.path.clone()).collect();
        for p in restored {
            self.add_source(p, false, ctx);
        }
    }

    pub(crate) fn mark_dirty(&mut self) {
        self.dirty_since.get_or_insert(Instant::now());
    }

    pub(crate) fn prefs_changed(&mut self) {
        self.prefs_dirty = true;
    }

    pub(crate) fn notify(&mut self, text: impl Into<String>, error: bool) {
        self.message = Some((text.into(), error, Instant::now()));
    }

    /// Probes `path` in the background and adds it to the project. `select` is set for files the
    /// user just opened; restored project sources keep the saved selection.
    pub(crate) fn add_source(&mut self, path: PathBuf, select: bool, ctx: &egui::Context) {
        let Some(tools) = self.tools.clone() else { return };
        if self.infos.contains_key(&path) {
            if let Some(i) = self.project.sources.iter().position(|s| s.path == path) {
                if select {
                    self.select_source(i);
                }
            }
            return;
        }
        if let Some(p) = self.probes.iter_mut().find(|p| p.0 == path) {
            p.2 |= select;
            return;
        }
        let p = path.clone();
        self.probes.push((path, spawn(move || tools.probe(&p).map_err(|e| format!("{e:#}")), ctx), select));
    }

    pub(crate) fn import_dialog(&mut self, ctx: &egui::Context) {
        if let Some(files) = rfd::FileDialog::new().add_filter("Video", VIDEO_EXTS).pick_files() {
            for f in files {
                self.add_source(f, true, ctx);
            }
        }
    }

    pub(crate) fn select_source(&mut self, i: usize) {
        if self.project.selected != Some(i) && i < self.project.sources.len() {
            self.project.selected = Some(i);
            self.reset_view_for_source();
            self.mark_dirty();
        }
    }

    pub(crate) fn remove_source(&mut self, i: usize) {
        if i >= self.project.sources.len() {
            return;
        }
        let path = self.project.sources.remove(i).path;
        self.infos.remove(&path);
        self.source_thumbs.remove(&path);
        self.clips.retain(|c| c.source != path);
        self.project.selected = if self.project.sources.is_empty() { None } else { Some(i.min(self.project.sources.len() - 1)) };
        self.reset_view_for_source();
        self.mark_dirty();
    }

    pub(crate) fn clear_sources(&mut self) {
        self.project.sources.clear();
        self.project.selected = None;
        self.infos.clear();
        self.source_thumbs.clear();
        self.clips.clear();
        self.reset_view_for_source();
        self.mark_dirty();
    }

    pub(crate) fn selected_source(&self) -> Option<&Source> {
        self.project.selected.and_then(|i| self.project.sources.get(i))
    }

    pub(crate) fn selected_info(&self) -> Option<&VideoInfo> {
        self.infos.get(&self.selected_source()?.path)
    }

    pub(crate) fn active_settings(&self) -> Option<&JobSettings> {
        Some(&self.selected_source()?.active().settings)
    }

    pub(crate) fn active_settings_mut(&mut self) -> Option<&mut JobSettings> {
        let i = self.project.selected?;
        Some(&mut self.project.sources.get_mut(i)?.active_mut().settings)
    }

    /// Runs `f` on the settings of every tab of the selected source (input properties such as
    /// rotation and crop belong to the source, not to one tab).
    pub(crate) fn for_all_tabs(&mut self, f: impl Fn(&mut JobSettings)) {
        if let Some(i) = self.project.selected {
            for t in &mut self.project.sources[i].tabs {
                f(&mut t.settings);
            }
            self.mark_dirty();
        }
    }

    /// Settings of tab `i` as they will run: the processing device comes from preferences.
    fn run_settings(&self, tab: usize) -> Option<JobSettings> {
        let src = self.selected_source()?;
        let mut s = src.tabs.get(tab)?.settings.clone();
        s.device = self.prefs.device.clone();
        Some(s)
    }

    fn reset_view_for_source(&mut self) {
        self.playback = None;
        self.original = None;
        for s in &mut self.slots {
            s.cancel();
            s.tex = None;
            s.note.clear();
        }
        self.playhead = 0.0;
        self.zoom = 1.0;
        self.center = Vec2::splat(0.5);
        self.thumbs.clear();
        self.thumbs_for = None;
        self.compare = None;
        self.show_original = false;
        self.cropping = None;
        self.timeline_zoom = 1.0;
        self.timeline_start = 0.0;
    }

    /// What each viewer side shows: index 0 is the left side (split and side-by-side only),
    /// index 1 the main/right side, whose tab the right pane edits.
    pub(crate) fn contents(&self) -> [Option<Content>; 2] {
        let Some(src) = self.selected_source() else { return [None, None] };
        let active = Content::Tab(src.active_tab.min(src.tabs.len() - 1));
        if self.cropping.is_some() {
            return [None, Some(Content::Original)];
        }
        match self.prefs.view {
            ViewMode::Single => [None, Some(if self.show_original { Content::Original } else { active })],
            _ => [Some(self.compare.filter(|i| *i < src.tabs.len()).map(Content::Tab).unwrap_or(Content::Original)), Some(active)],
        }
    }

    // ---------------------------------------------------------------- background results

    fn poll(&mut self, ctx: &egui::Context) {
        let mut done = Vec::new();
        for (i, (path, rx, select)) in self.probes.iter().enumerate() {
            if let Ok(r) = rx.try_recv() {
                done.push((i, path.clone(), r, *select));
            }
        }
        for (i, path, r, select) in done.into_iter().rev() {
            self.probes.remove(i);
            let before = self.project.selected;
            match r {
                Ok(info) => {
                    if let Some(tools) = self.tools.clone() {
                        let (p, inf) = (path.clone(), info.clone());
                        let (w, h) = media::display_size(&inf, 128, 72);
                        let t = (inf.duration * 0.1).min(2.0);
                        self.source_thumb_rx.push((path.clone(), spawn(move || media::grab(&tools, &p, t, w, h).ok(), ctx)));
                    }
                    self.infos.insert(path.clone(), info);
                    let index = match self.project.sources.iter().position(|s| s.path == path) {
                        Some(i) => i,
                        None => self.project.add_source(path, default_settings()),
                    };
                    self.project.selected = if select || before.is_none() { Some(index) } else { before };
                    self.mark_dirty();
                }
                Err(e) => {
                    let selected_path = before.and_then(|i| self.project.sources.get(i)).map(|s| s.path.clone());
                    self.project.sources.retain(|s| s.path != path);
                    self.project.selected = selected_path.and_then(|p| self.project.sources.iter().position(|s| s.path == p)).or(if self.project.sources.is_empty() { None } else { Some(0) });
                    self.notify(e, true);
                }
            }
            if before != self.project.selected {
                self.reset_view_for_source();
            }
        }
        let mut thumbs_done = Vec::new();
        for (i, (path, rx)) in self.source_thumb_rx.iter().enumerate() {
            if let Ok(r) = rx.try_recv() {
                thumbs_done.push(i);
                if let Some(im) = r {
                    self.source_thumbs.insert(path.clone(), ctx.load_texture(format!("src-{}", path.display()), im.to_color_image(), TextureOptions::LINEAR));
                }
            }
        }
        for i in thumbs_done.into_iter().rev() {
            self.source_thumb_rx.remove(i);
        }
        if let Some(rx) = &self.thumbs_rx {
            if let Ok(images) = rx.try_recv() {
                self.thumbs = images.iter().enumerate().map(|(i, im)| ctx.load_texture(format!("thumb{i}"), im.to_color_image(), TextureOptions::LINEAR)).collect();
                self.thumbs_rx = None;
            }
        }
        if let Some(p) = &self.grab {
            if let Ok(r) = p.rx.try_recv() {
                let key = p.key.clone();
                self.grab = None;
                if let Ok(im) = r {
                    set_tex(ctx, &mut self.original, "original", &im, key);
                }
            }
        }
        for side in 0..2 {
            let slot = &mut self.slots[side];
            if let Some((p, _)) = &slot.pending {
                if let Ok(r) = p.rx.try_recv() {
                    let key = p.key.clone();
                    slot.pending = None;
                    match r {
                        Ok(im) => {
                            set_tex(ctx, &mut slot.tex, if side == 0 { "left" } else { "right" }, &im, key);
                            slot.note.clear();
                            slot.failed.clear();
                        }
                        Err(e) => {
                            slot.tex = None;
                            slot.note = e;
                            slot.failed = key;
                        }
                    }
                }
            }
        }
        if let Some(job) = &mut self.preview_job {
            let mut finished = false;
            while let Ok(r) = job.rx.try_recv() {
                job.done += 1;
                match r {
                    Ok(clip) => {
                        self.clips.retain(|c| !(c.source == clip.source && c.settings_hash == clip.settings_hash && (c.start - clip.start).abs() < 1e-3));
                        self.clips.push(clip);
                        if self.clips.len() > 12 {
                            self.clips.remove(0);
                        }
                        for s in &mut self.slots {
                            s.tex = None;
                        }
                    }
                    Err(e) => {
                        if e != "Cancelled" {
                            self.message = Some((e, true, Instant::now()));
                        }
                        finished = true;
                    }
                }
                if job.done >= job.total {
                    finished = true;
                }
            }
            if finished {
                self.preview_job = None;
            }
        }
        if let Some((id, rx)) = &self.fetching {
            if let Ok(r) = rx.try_recv() {
                let id = id.clone();
                self.fetching = None;
                match r {
                    Ok(_) => {
                        self.catalog = Arc::new(Catalog::load(&paths::models_dir()));
                        self.notify(format!("Installed model {id}"), false);
                    }
                    Err(e) => self.notify(format!("Could not install {id}: {e}"), true),
                }
            }
        }
        if let Some((second, rx)) = &self.estimating {
            if let Ok(r) = rx.try_recv() {
                let second = *second;
                self.estimating = None;
                match r {
                    Ok(p) => {
                        if let Some(s) = self.active_settings_mut() {
                            let e = if second { &mut s.second_enhancement.settings } else { &mut s.enhancement };
                            let v = |x: f32| (x * 100.0).round() as i32;
                            e.compression = v(p.compression);
                            e.details = v(p.details);
                            e.sharpen = v(p.sharpen);
                            e.noise = v(p.noise);
                            e.dehalo = v(p.dehalo);
                            e.deblur = v(p.deblur);
                        }
                        self.mark_dirty();
                    }
                    Err(e) => self.notify(format!("Could not estimate parameters: {e}"), true),
                }
            }
        }
        if self.dirty_since.is_some_and(|t| t.elapsed() > Duration::from_millis(1500)) {
            self.dirty_since = None;
            if let Err(e) = self.project.save(&self.project_dir) {
                self.notify(format!("Could not save the project: {e:#}"), true);
            }
        }
        if self.prefs_dirty {
            self.prefs_dirty = false;
            if let Err(e) = self.prefs.save() {
                self.notify(format!("Could not save preferences: {e:#}"), true);
            }
        }
        if self.message.as_ref().is_some_and(|m| !m.1 && m.2.elapsed() > Duration::from_secs(5)) {
            self.message = None;
        }
    }

    fn clip_for(&self, source: &Path, hash: u64, t: f64) -> Option<&PreviewClip> {
        self.clips.iter().rev().find(|c| c.source == source && c.settings_hash == hash && t >= c.start - 1e-6 && t <= c.start + c.duration)
    }

    /// Starts grabs, instant renders and thumbnails the current view needs.
    fn schedule(&mut self, ctx: &egui::Context, viewer_px: (usize, usize)) {
        let (Some(tools), Some(info)) = (self.tools.clone(), self.selected_info().cloned()) else { return };
        if self.thumbs_for.as_ref() != Some(&info.path) && self.thumbs_rx.is_none() {
            self.thumbs_for = Some(info.path.clone());
            let (t, i) = (tools.clone(), info.clone());
            self.thumbs_rx = Some(spawn(move || media::thumbnails(&t, &i, 30, 54).unwrap_or_default(), ctx));
        }
        if self.playback.is_some() {
            return;
        }
        let fps = info.fps.as_f64().max(1.0);
        let t = (self.playhead * fps).round() / fps;
        // Grab at the size on screen; zooming in requests more pixels, in powers of two.
        let z = 2f32.powf(self.zoom.log2().ceil().max(0.0));
        let (gw, gh) = media::display_size(&info, ((viewer_px.0 as f32 * z) as usize).clamp(320, 4096), ((viewer_px.1 as f32 * z) as usize).clamp(180, 4096));
        let okey = format!("{}@{t:.4}@{gw}", info.path.display());
        if self.original.as_ref().is_none_or(|o| o.key != okey) && self.grab.as_ref().is_none_or(|g| g.key != okey) {
            let (tl, path) = (tools.clone(), info.path.clone());
            self.grab = Some(Pending { rx: spawn(move || media::grab(&tl, &path, t, gw, gh).map_err(|e| format!("{e:#}")), ctx), key: okey });
        }
        let contents = self.contents();
        for side in 0..2 {
            let Some(Content::Tab(i)) = contents[side] else {
                self.slots[side].cancel();
                self.slots[side].tex = None;
                continue;
            };
            let Some(settings) = self.run_settings(i) else { continue };
            if adjustments(&settings, &self.catalog).is_empty() {
                self.slots[side].cancel();
                self.slots[side].tex = None;
                continue;
            }
            let hash = settings_hash(&settings);
            let clip = self.clip_for(&info.path, hash, t).map(|c| ((t - c.start) * c.stretch, c.path.clone(), c.size));
            let slot = &mut self.slots[side];
            // A rendered preview covering the playhead takes precedence over live frames.
            let (key, job): (String, Box<dyn FnOnce() -> Result<RgbaImage, String> + Send>) = match clip {
                Some((pt, path, size)) => {
                    let (tl, (w, h)) = (tools.clone(), fit_size(size, 3840, 2160));
                    (format!("clip:{hash}@{pt:.4}"), Box::new(move || media::grab(&tl, &path, pt, w, h).map_err(|e| format!("{e:#}"))))
                }
                None if !self.prefs.live_render => {
                    slot.cancel();
                    slot.tex = None;
                    slot.note = "Live render is off".into();
                    continue;
                }
                None => {
                    let (tl, cat, inf) = (tools.clone(), self.catalog.clone(), info.clone());
                    let control = Control::new();
                    let c2 = control.clone();
                    let key = format!("instant:{}:{hash}:{t:.4}", info.path.display());
                    if slot.tex.as_ref().is_some_and(|p| p.key == key) || slot.pending.as_ref().is_some_and(|(p, _)| p.key == key) || slot.failed == key {
                        continue;
                    }
                    slot.cancel();
                    slot.tex = None;
                    let rx = spawn(move || media::render_instant(&tl, &cat, &inf, &settings, t, &c2).map_err(|e| if runner::is_cancelled(&e) { "Cancelled".into() } else { format!("{e:#}") }), ctx);
                    slot.pending = Some((Pending { rx, key }, Some(control)));
                    continue;
                }
            };
            if slot.tex.as_ref().is_some_and(|p| p.key == key) || slot.pending.as_ref().is_some_and(|(p, _)| p.key == key) || slot.failed == key {
                continue;
            }
            slot.cancel();
            slot.pending = Some((Pending { rx: spawn(job, ctx), key }, None));
        }
    }

    // ---------------------------------------------------------------- actions

    pub(crate) fn start_preview_render(&mut self, ctx: &egui::Context) {
        let (Some(tools), Some(info)) = (self.tools.clone(), self.selected_info().cloned()) else { return };
        if let Some(job) = self.preview_job.take() {
            job.control.cancel();
        }
        let mut jobs: Vec<(JobSettings, u64)> = Vec::new();
        for c in self.contents().into_iter().flatten() {
            if let Content::Tab(i) = c {
                if let Some(s) = self.run_settings(i) {
                    let h = settings_hash(&s);
                    if !adjustments(&s, &self.catalog).is_empty() && !jobs.iter().any(|j| j.1 == h) {
                        jobs.push((s, h));
                    }
                }
            }
        }
        if jobs.is_empty() {
            self.notify("Nothing to render: the visible tabs have no adjustments.", false);
            return;
        }
        let trim = self.active_settings().and_then(|s| s.trim);
        let (start, duration) = if self.prefs.preview_seconds < 0.0 {
            let tr = trim.unwrap_or(TrimRange { start: 0.0, end: info.duration });
            (tr.start, (tr.end - tr.start).max(0.1))
        } else {
            let start = self.playhead.min((info.duration - 0.1).max(0.0));
            (start, self.prefs.preview_seconds.min(info.duration - start).max(0.1))
        };
        let catalog = self.catalog.clone();
        let control = Control::new();
        let progress = Arc::new(Mutex::new(Progress::default()));
        let (c2, p2) = (control.clone(), progress.clone());
        let total = jobs.len();
        let (tx, rx) = unbounded();
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            for (settings, hash) in jobs {
                let r = (|| -> Result<PreviewClip, String> {
                    let exists = |_: &Path| false;
                    let temp = paths::temp_root().join(format!("preview-{}-{}", std::process::id(), hash));
                    let _ = std::fs::remove_dir_all(&temp);
                    let pctx = PlanContext { catalog: &catalog, temp_dir: temp, encoders_available: None, preview: Some(PreviewRange { start, duration }), exists: &exists };
                    let p = plan(&info, &settings, &pctx).map_err(|e| format!("{e:#}"))?;
                    let on_event = |e: runner::Event| {
                        if let runner::Event::Progress(pr) = e {
                            *p2.lock().unwrap() = pr;
                        }
                    };
                    runner::run_plan(&p, &tools, &catalog, &c2, &on_event).map_err(|e| if runner::is_cancelled(&e) { "Cancelled".into() } else { format!("{e:#}") })?;
                    let stretch = if settings.interpolation.enabled { settings.interpolation.slowmo.max(1.0) } else { 1.0 };
                    Ok(PreviewClip { source: info.path.clone(), settings_hash: hash, start, duration, path: p.output.clone(), stretch, fps: p.fps_out.as_f64(), size: (p.geometry.output.0 as usize, p.geometry.output.1 as usize) })
                })();
                let stop = r.is_err();
                let _ = tx.send(r);
                ctx2.request_repaint();
                if stop {
                    break;
                }
            }
        });
        self.preview_job = Some(PreviewJob { rx, progress, control, total, done: 0, stopping: false });
    }

    /// Plans an export of the active tab (`output` overrides the file name).
    pub(crate) fn export_plan(&self, output: Option<PathBuf>, preview: bool) -> Result<(prism_core::plan::JobPlan, String), String> {
        let (Some(info), Some(src)) = (self.selected_info(), self.selected_source()) else { return Err("No video selected".into()) };
        let tab = src.active();
        let mut s = tab.settings.clone();
        s.device = self.prefs.device.clone();
        if let Some(o) = output {
            if let Some(ext) = o.extension().and_then(|e| e.to_str()) {
                s.export.container = ext.to_ascii_lowercase();
            }
            s.export.output_path = Some(o);
        }
        let exists = |p: &Path| p.exists();
        let stamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        let ctx = PlanContext {
            catalog: &self.catalog,
            temp_dir: paths::temp_root().join(format!("export-{}-{stamp}", std::process::id())),
            encoders_available: Some(&self.encoders),
            preview: preview.then(|| PreviewRange { start: self.playhead, duration: self.prefs.preview_seconds.max(0.1) }),
            exists: &exists,
        };
        let name = format!("{} · {}", info.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), tab.name);
        plan(info, &s, &ctx).map(|p| (p, name)).map_err(|e| format!("{e:#}"))
    }

    pub(crate) fn export(&mut self, output: Option<PathBuf>) {
        let Some(queue) = self.queue.clone() else { return };
        match self.export_plan(output, false) {
            Ok((p, name)) => {
                queue.submit(&name, p, prism_engine::queue::PRIORITY_NORMAL);
                self.popover = Some(Popover::Queue);
                self.popover_opened = true;
            }
            Err(e) => self.notify(e, true),
        }
    }

    pub(crate) fn export_as(&mut self) {
        let Some(src) = self.selected_source() else { return };
        let s = &src.active().settings;
        let e = prism_core::encoders::find(&s.export.encoder);
        let container = if s.export.container.is_empty() { e.map(|e| e.default_container().to_string()).unwrap_or_else(|| "mp4".into()) } else { s.export.container.clone() };
        let stem = src.path.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
        let mut d = rfd::FileDialog::new().set_file_name(format!("{stem}_prism.{container}")).add_filter(container.clone(), &[container.as_str()]);
        if let Some(dir) = s.export.output_dir.clone().or_else(|| src.path.parent().map(Path::to_path_buf)) {
            d = d.set_directory(dir);
        }
        if let Some(p) = d.save_file() {
            self.export(Some(p));
        }
    }

    pub(crate) fn show_command(&mut self, preview: bool) {
        let Some(tools) = &self.tools else { return };
        match self.export_plan(None, preview) {
            Ok((p, _)) => {
                let ff = tools.ffmpeg.display().to_string();
                let mut text = String::new();
                for (i, pass) in p.passes.iter().enumerate() {
                    text.push_str(&format!("# Pass {} — {}\n", i + 1, pass.label()));
                    match pass {
                        prism_core::plan::Pass::Ffmpeg(f) => text.push_str(&prism_core::plan::shell_join(&ff, &tools.adapt_args(&f.args))),
                        prism_core::plan::Pass::Engine(e) => {
                            let stages: Vec<String> = e.stages.iter().map(|s| format!("{s:?}").split(['(', ' ', '{']).next().unwrap_or("").to_string()).collect();
                            text.push_str(&prism_core::plan::shell_join(&ff, &tools.adapt_args(&e.decode)));
                            text.push_str(&format!("\n  | prism engine [{}]\n  | ", stages.join(" → ")));
                            text.push_str(&prism_core::plan::shell_join(&ff, &tools.adapt_args(&e.encode)));
                        }
                    }
                    text.push_str("\n\n");
                }
                for w in &p.warnings {
                    text.push_str(&format!("# warning: {w}\n"));
                }
                self.dialog = Some(Dialog::Command { title: if preview { "Preview command".into() } else { "Export command".into() }, text });
            }
            Err(e) => self.notify(e, true),
        }
    }

    fn toggle_play(&mut self) {
        if self.playback.take().is_some() {
            return;
        }
        let (Some(tools), Some(info)) = (self.tools.clone(), self.selected_info().cloned()) else { return };
        let (w, h) = media::display_size(&info, 1280, 720);
        let fps = info.fps.as_f64().max(1.0);
        let trim = self.active_settings().and_then(|s| s.trim);
        let end = trim.map(|t| t.end).unwrap_or(info.duration);
        if self.playhead >= end - 1.0 / fps {
            self.playhead = trim.map(|t| t.start).unwrap_or(0.0);
        }
        let Ok(original) = Player::start(&tools, &info.path, self.playhead, w, h, fps) else { return };
        let mut sides: [Option<SidePlayer>; 2] = [None, None];
        for (side, c) in self.contents().into_iter().enumerate() {
            let Some(Content::Tab(i)) = c else { continue };
            let Some(s) = self.run_settings(i) else { continue };
            let hash = settings_hash(&s);
            if let Some(clip) = self.clips.iter().rev().find(|c| c.source == info.path && c.settings_hash == hash && self.playhead >= c.start && self.playhead < c.start + c.duration) {
                let (pw, ph) = fit_size(clip.size, 1280, 720);
                if let Ok(player) = Player::start(&tools, &clip.path, (self.playhead - clip.start) * clip.stretch, pw, ph, clip.fps) {
                    sides[side] = Some(SidePlayer { player, stretch: clip.stretch });
                }
            }
        }
        self.playback = Some(Playback { started: Instant::now(), from: self.playhead, original, sides });
    }

    fn update_playback(&mut self, ctx: &egui::Context) {
        let Some(pb) = &mut self.playback else { return };
        let Some(info) = self.project.selected.and_then(|i| self.project.sources.get(i)).and_then(|s| self.infos.get(&s.path)) else { return };
        let end = self.project.selected.and_then(|i| self.project.sources.get(i)).and_then(|s| s.active().settings.trim).map(|t| t.end).unwrap_or(info.duration);
        let elapsed = pb.started.elapsed().as_secs_f64();
        let t = pb.from + elapsed;
        if t >= end {
            self.playback = None;
            return;
        }
        self.playhead = t;
        if let Some(f) = pb.original.frame_for(elapsed) {
            set_tex(ctx, &mut self.original, "original", &f, format!("play@{t}"));
        }
        for side in 0..2 {
            if let Some(sp) = &mut pb.sides[side] {
                if let Some(f) = sp.player.frame_for(elapsed * sp.stretch) {
                    set_tex(ctx, &mut self.slots[side].tex, if side == 0 { "left" } else { "right" }, &f, format!("play@{t}"));
                }
            }
        }
        ctx.request_repaint_after(Duration::from_millis(8));
    }

    fn step(&mut self, frames: f64) {
        let Some(info) = self.selected_info() else { return };
        let fps = info.fps.as_f64().max(1.0);
        let dur = info.duration;
        self.playback = None;
        self.playhead = ((self.playhead + frames / fps) * fps).round().clamp(0.0, (dur * fps).floor()) / fps;
    }

    fn jump(&mut self, to_out: bool) {
        let Some(info) = self.selected_info() else { return };
        let dur = info.duration;
        let trim = self.active_settings().and_then(|s| s.trim);
        self.playback = None;
        self.playhead = if to_out { trim.map(|t| t.end).unwrap_or(dur) } else { trim.map(|t| t.start).unwrap_or(0.0) };
    }

    pub(crate) fn mark(&mut self, out: bool) {
        let Some(info) = self.selected_info() else { return };
        let (dur, fps) = (info.duration, info.fps.as_f64().max(1.0));
        let t = self.playhead;
        let trim = self.active_settings().and_then(|s| s.trim).unwrap_or(TrimRange { start: 0.0, end: dur });
        let new = if out { TrimRange { start: trim.start.min(t - 1.0 / fps).max(0.0), end: t.max(1.0 / fps) } } else { TrimRange { start: t, end: trim.end.max(t + 1.0 / fps).min(dur) } };
        self.for_all_tabs(|s| s.trim = Some(new));
    }

    pub(crate) fn clear_trim(&mut self) {
        self.for_all_tabs(|s| s.trim = None);
    }

    pub(crate) fn set_view(&mut self, v: ViewMode) {
        if self.prefs.view != v {
            self.prefs.view = v;
            self.prefs_changed();
        }
        if v == ViewMode::Single {
            self.show_original = false;
        }
    }

    pub(crate) fn set_zoom_percent(&mut self, pct: Option<f32>) {
        match pct {
            None => {
                self.zoom = 1.0;
                self.center = Vec2::splat(0.5);
            }
            Some(p) => self.zoom = (p / 100.0 / self.fit_scale.max(1e-3)).clamp(0.05, 64.0),
        }
    }

    pub(crate) fn toggle_crop(&mut self) {
        if let Some(c) = self.cropping.take() {
            self.apply_crop(c.rect);
            return;
        }
        let Some(info) = self.selected_info() else { return };
        let Some(s) = self.active_settings() else { return };
        if s.rotation != 0 {
            self.notify("Reset the rotation in Input settings before cropping.", true);
            return;
        }
        let (w, h) = crop_space(info, s.correction.square_pixels);
        let rect = s.crop.map(|c| Rect::from_min_size(pos2(c.x as f32 / w as f32, c.y as f32 / h as f32), vec2(c.width as f32 / w as f32, c.height as f32 / h as f32))).unwrap_or(Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)));
        self.cropping = Some(CropEdit { rect, drag: None });
        self.playback = None;
    }

    fn apply_crop(&mut self, r: Rect) {
        let Some(info) = self.selected_info() else { return };
        let square = self.active_settings().is_some_and(|s| s.correction.square_pixels);
        let (w, h) = crop_space(info, square);
        let even = |v: f32, max: u32| ((v.round() as u32) & !1).min(max);
        let x = even(r.min.x * w as f32, w);
        let y = even(r.min.y * h as f32, h);
        let cw = even(r.width() * w as f32, w - x).max(16);
        let ch = even(r.height() * h as f32, h - y).max(16);
        let crop = if x == 0 && y == 0 && cw >= w - 1 && ch >= h - 1 { None } else { Some(prism_core::settings::CropRect { x, y, width: cw, height: ch }) };
        self.for_all_tabs(|s| s.crop = crop);
        for s in &mut self.slots {
            s.tex = None;
        }
    }

    pub(crate) fn copy_settings(&mut self) {
        self.clipboard = self.active_settings().cloned();
    }

    pub(crate) fn paste_settings(&mut self) {
        let Some(c) = self.clipboard.clone() else { return };
        let p = Preset::capture("clipboard", "", &c);
        if let Some(s) = self.active_settings_mut() {
            p.apply(s);
        }
        self.mark_dirty();
    }

    pub(crate) fn apply_to_all_sources(&mut self) {
        let Some(si) = self.project.selected else { return };
        let tab = self.project.sources[si].active().clone();
        let p = Preset::capture(&tab.name, "", &tab.settings);
        for (i, src) in self.project.sources.iter_mut().enumerate() {
            if i != si {
                p.apply(&mut src.active_mut().settings);
            }
        }
        self.mark_dirty();
        self.notify(format!("Applied “{}” to all sources", tab.name), false);
    }

    pub(crate) fn new_tab(&mut self, duplicate: bool) {
        let Some(si) = self.project.selected else { return };
        let src = &mut self.project.sources[si];
        let n = src.tabs.len() + 1;
        let settings = if duplicate { src.active().settings.clone() } else { input_only(&src.active().settings) };
        let name = if duplicate { format!("{} copy", src.active().name) } else { format!("Untitled {n}") };
        src.tabs.push(Tab { name, settings });
        src.active_tab = src.tabs.len() - 1;
        self.show_original = false;
        self.mark_dirty();
    }

    pub(crate) fn close_tab(&mut self, i: usize) {
        let Some(si) = self.project.selected else { return };
        let src = &mut self.project.sources[si];
        if src.tabs.len() <= 1 {
            src.tabs[0] = Tab { name: "Untitled 1".into(), settings: input_only(&src.tabs[0].settings) };
        } else {
            src.tabs.remove(i);
            if src.active_tab >= i && src.active_tab > 0 {
                src.active_tab -= 1;
            }
            src.active_tab = src.active_tab.min(src.tabs.len() - 1);
        }
        self.compare = match self.compare {
            Some(c) if c == i => None,
            Some(c) if c > i => Some(c - 1),
            c => c,
        };
        self.mark_dirty();
    }

    pub(crate) fn close_all_tabs(&mut self) {
        let Some(si) = self.project.selected else { return };
        let src = &mut self.project.sources[si];
        let keep = input_only(&src.active().settings);
        src.tabs = vec![Tab { name: "Untitled 1".into(), settings: keep }];
        src.active_tab = 0;
        self.compare = None;
        self.mark_dirty();
    }

    pub(crate) fn estimate(&mut self, second: bool, ctx: &egui::Context) {
        let (Some(tools), Some(info)) = (self.tools.clone(), self.selected_info().cloned()) else { return };
        if self.estimating.is_some() {
            return;
        }
        let t = self.playhead;
        self.estimating = Some((
            second,
            spawn(
                move || {
                    let (w, h) = media::display_size(&info, 1280, 720);
                    let im = media::grab(&tools, &info.path, t, w, h).map_err(|e| format!("{e:#}"))?;
                    let mut f = prism_engine::Frame::zeroed(im.width, im.height);
                    {
                        let [r, g, b] = f.planes_mut();
                        for i in 0..im.width * im.height {
                            r[i] = im.data[4 * i] as f32 / 255.0;
                            g[i] = im.data[4 * i + 1] as f32 / 255.0;
                            b[i] = im.data[4 * i + 2] as f32 / 255.0;
                        }
                    }
                    Ok(prism_engine::ops::analysis::estimate_params(&[&f]))
                },
                ctx,
            ),
        ));
    }

    pub(crate) fn switch_project(&mut self, dir: PathBuf, ctx: &egui::Context) {
        let _ = self.project.save(&self.project_dir);
        match Project::load(&dir) {
            Ok(p) => self.project = p,
            Err(_) => {
                self.project = Project::default();
                self.project.name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "Untitled".into());
            }
        }
        self.project_dir = dir;
        self.infos.clear();
        self.source_thumbs.clear();
        self.clips.clear();
        self.reset_view_for_source();
        self.prefs.last_project = Some(self.project_dir.clone());
        self.prefs_changed();
        self.mark_dirty();
        self.restore_sources(ctx);
    }

    fn handle_actions(&mut self, actions: Vec<Action>, ctx: &egui::Context) {
        for a in actions {
            match a {
                Action::FetchModel(id) => self.fetch_model(id, ctx),
                Action::BrowseLut => {
                    if let Some(p) = rfd::FileDialog::new().add_filter("3D LUT", &["cube", "3dl", "dat", "m3d", "csp"]).pick_file() {
                        if let Some(s) = self.active_settings_mut() {
                            s.export.lut = Some(p);
                        }
                        self.mark_dirty();
                    }
                }
                Action::BrowseOutputDir => {
                    if let Some(d) = rfd::FileDialog::new().pick_folder() {
                        if let Some(s) = self.active_settings_mut() {
                            s.export.output_dir = Some(d);
                        }
                        self.mark_dirty();
                    }
                }
                Action::Estimate(second) => self.estimate(second, ctx),
                Action::ImportModel(id) => self.import_model(id, ctx),
            }
        }
    }

    pub(crate) fn fetch_model(&mut self, id: String, ctx: &egui::Context) {
        if self.fetching.is_none() {
            let cat = self.catalog.clone();
            let id2 = id.clone();
            self.fetching = Some((id, spawn(move || prism_engine::ai::fetch::fetch(&cat, &id2, false, &|_| {}).map_err(|e| format!("{e:#}")), ctx)));
        }
    }

    /// Installs a local .onnx file (or a Real-ESRGAN checkpoint) as model `id`.
    pub(crate) fn import_model(&mut self, id: String, ctx: &egui::Context) {
        if self.fetching.is_some() {
            return;
        }
        let Some(file) = rfd::FileDialog::new().set_title(format!("Model file for {id}")).add_filter("ONNX model", &["onnx"]).add_filter("PyTorch checkpoint", &["pth"]).pick_file() else { return };
        let dir = self.catalog.dir.clone();
        let id2 = id.clone();
        self.fetching = Some((id, spawn(move || prism_engine::ai::fetch::install_file(&file, &dir, &id2, false).map_err(|e| format!("{e:#}")), ctx)));
    }

    // ---------------------------------------------------------------- window chrome

    /// Resize handles along the edges of the frameless window.
    fn window_edges(&mut self, ctx: &egui::Context) {
        // macOS keeps the native frame, which resizes itself.
        if cfg!(target_os = "macos") {
            return;
        }
        if ctx.input(|i| i.viewport().maximized.unwrap_or(false) || i.viewport().fullscreen.unwrap_or(false)) {
            return;
        }
        let Some(pos) = ctx.input(|i| i.pointer.hover_pos()) else { return };
        let r = ctx.content_rect();
        let m = 5.0;
        let (l, rt, t, b) = (pos.x - r.left() < m, r.right() - pos.x < m, pos.y - r.top() < m, r.bottom() - pos.y < m);
        use egui::viewport::ResizeDirection as D;
        let dir = match (l, rt, t, b) {
            (true, _, true, _) => Some((D::NorthWest, CursorIcon::ResizeNwSe)),
            (_, true, true, _) => Some((D::NorthEast, CursorIcon::ResizeNeSw)),
            (true, _, _, true) => Some((D::SouthWest, CursorIcon::ResizeNeSw)),
            (_, true, _, true) => Some((D::SouthEast, CursorIcon::ResizeNwSe)),
            (true, ..) => Some((D::West, CursorIcon::ResizeHorizontal)),
            (_, true, ..) => Some((D::East, CursorIcon::ResizeHorizontal)),
            (_, _, true, _) => Some((D::North, CursorIcon::ResizeVertical)),
            (.., true) => Some((D::South, CursorIcon::ResizeVertical)),
            _ => None,
        };
        if let Some((d, cursor)) = dir {
            ctx.set_cursor_icon(cursor);
            if ctx.input(|i| i.pointer.primary_pressed()) {
                ctx.send_viewport_cmd(egui::ViewportCommand::BeginResize(d));
            }
        }
    }

    fn menu_row(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let full = ui.max_rect();
        // On macOS the native traffic-light buttons sit over the left end of this row.
        let native_buttons = cfg!(target_os = "macos");
        let mut menus_right = full.left();
        ui.horizontal_centered(|ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_space(if native_buttons { 80.0 } else { 12.0 });
            let (r, _) = ui.allocate_exact_size(vec2(18.0, 18.0), Sense::hover());
            icons::paint(ui.painter(), r, Icon::Logo, ACCENT);
            ui.add_space(8.0);
            egui::MenuBar::new().ui(ui, |ui| {
                ui.style_mut().spacing.button_padding = vec2(9.0, 4.0);
                menus_right = self.menus(ui);
            });
        });
        // Drag area between the menus and the window buttons (the menu bar itself spans the
        // whole row, so its rect cannot be used).
        let buttons_w = if native_buttons { 0.0 } else { 46.0 * 3.0 };
        let drag = Rect::from_min_max(pos2(menus_right + 8.0, full.top()), pos2((full.right() - buttons_w).max(menus_right + 8.0), full.bottom()));
        let resp = ui.interact(drag, ui.scope_id().with("drag"), Sense::click_and_drag());
        if resp.drag_started() {
            ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
        }
        let maximized = ctx.input(|i| i.viewport().maximized.unwrap_or(false));
        if resp.double_clicked() {
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(!maximized));
        }
        let mut x = full.right() - buttons_w;
        let buttons = if native_buttons { Vec::new() } else { vec![(Icon::Minimize, 0), (if maximized { Icon::Restore } else { Icon::Maximize }, 1), (Icon::Close, 2)] };
        for (icon, cmd) in buttons {
            let r = Rect::from_min_size(pos2(x, full.top()), vec2(46.0, full.height()));
            let resp = ui.interact(r, ui.scope_id().with(("winbtn", cmd)), Sense::click());
            if resp.hovered() {
                ui.painter().rect_filled(r, 0.0, if cmd == 2 { Color32::from_rgb(0xC4, 0x2B, 0x1C) } else { Color32::from_white_alpha(18) });
            }
            icons::paint_at(ui.painter(), r.center(), if cmd == 2 { 12.0 } else { 14.0 }, icon, TEXT);
            if resp.clicked() {
                ctx.send_viewport_cmd(match cmd {
                    0 => egui::ViewportCommand::Minimized(true),
                    1 => egui::ViewportCommand::Maximized(!maximized),
                    _ => egui::ViewportCommand::Close,
                });
            }
            x += 46.0;
        }
    }

    /// The menus; returns the right edge of the last one.
    fn menus(&mut self, ui: &mut Ui) -> f32 {
        let ctx = ui.ctx().clone();
        let has_src = self.selected_source().is_some();
        let item = |ui: &mut Ui, text: &str, shortcut: &str, enabled: bool| -> bool { ui.add_enabled(enabled, egui::Button::new(text).shortcut_text(egui::RichText::new(key_label(shortcut)).color(MUTED))).clicked() };
        ui.menu_button("File", |ui| {
            ui.set_min_width(250.0);
            if item(ui, "New project…", "Ctrl+N", true) {
                self.dialog = Some(Dialog::NewProject { name: String::new() });
            }
            if item(ui, "Open project…", "Ctrl+O", true) {
                self.open_project_dialog(&ctx);
            }
            if item(ui, "Save project", "Ctrl+S", true) {
                self.save_project();
            }
            if item(ui, "Save project as…", "Ctrl+Shift+S", true) {
                self.save_project_as(&ctx);
            }
            ui.separator();
            if item(ui, "Import…", "Ctrl+I", self.tools.is_some()) {
                self.import_dialog(&ctx);
            }
            if item(ui, "Close input", "Ctrl+Backspace", has_src) {
                if let Some(i) = self.project.selected {
                    self.remove_source(i);
                }
            }
            if item(ui, "Close all inputs", "Ctrl+Shift+Backspace", has_src) {
                self.dialog = Some(Dialog::CloseAll);
            }
            ui.separator();
            if item(ui, "Preferences", "Ctrl+,", true) {
                self.dialog = Some(Dialog::Preferences);
            }
            if item(ui, "Model manager", "", true) {
                self.dialog = Some(Dialog::Models);
            }
            ui.separator();
            if item(ui, "Exit", "Ctrl+Q", true) {
                ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            }
        });
        ui.menu_button("Edit", |ui| {
            ui.set_min_width(250.0);
            if item(ui, "Copy settings", "Ctrl+C", has_src) {
                self.copy_settings();
            }
            if item(ui, "Paste settings", "Ctrl+V", has_src && self.clipboard.is_some()) {
                self.paste_settings();
            }
            ui.separator();
            if item(ui, "Mark video in", "I", has_src) {
                self.mark(false);
            }
            if item(ui, "Mark video out", "O", has_src) {
                self.mark(true);
            }
            if item(ui, "Clear in and out", "Shift+R", has_src) {
                self.clear_trim();
            }
            ui.separator();
            if item(ui, if self.cropping.is_some() { "Apply crop" } else { "Crop video" }, "Ctrl+Shift+C", has_src) {
                self.toggle_crop();
            }
            ui.separator();
            if item(ui, "Close all tabs", "Ctrl+Shift+W", has_src) {
                self.close_all_tabs();
            }
        });
        ui.menu_button("View", |ui| {
            ui.set_min_width(250.0);
            let playing = self.playback.is_some();
            if item(ui, "Back to in", "Shift+Left", has_src) {
                self.jump(false);
            }
            if item(ui, "Step backward", "Left", has_src) {
                self.step(-1.0);
            }
            if item(ui, if playing { "Pause" } else { "Play" }, "Space", has_src) {
                self.toggle_play();
            }
            if item(ui, "Step forward", "Right", has_src) {
                self.step(1.0);
            }
            if item(ui, "Forward to out", "Shift+Right", has_src) {
                self.jump(true);
            }
            ui.separator();
            if item(ui, "Previous video", "Up", has_src) {
                self.select_relative(-1);
            }
            if item(ui, "Next video", "Down", has_src) {
                self.select_relative(1);
            }
            ui.separator();
            for (label, sc, pct) in [("Fit", "Ctrl+0", None), ("100%", "Ctrl+1", Some(100.0)), ("200%", "Ctrl+2", Some(200.0)), ("400%", "Ctrl+4", Some(400.0))] {
                if item(ui, label, sc, has_src) {
                    self.set_zoom_percent(pct);
                }
            }
            ui.separator();
            for (label, sc, v) in [("Single view", "Shift+1", ViewMode::Single), ("Split view", "Shift+2", ViewMode::Split), ("Side by side view", "Shift+3", ViewMode::SideBySide)] {
                if item(ui, label, sc, true) {
                    self.set_view(v);
                }
            }
            ui.separator();
            if item(ui, if self.prefs.show_panel { "Hide filter panel" } else { "Show filter panel" }, "Tab", true) {
                self.prefs.show_panel = !self.prefs.show_panel;
                self.prefs_changed();
            }
            if item(ui, if self.full_preview { "Exit full screen preview" } else { "Full screen preview" }, "F", has_src) {
                self.full_preview = !self.full_preview;
            }
        });
        ui.menu_button("Process", |ui| {
            ui.set_min_width(250.0);
            if item(ui, &format!("Render preview ({})", preview_len_label(self.prefs.preview_seconds)), "Ctrl+P", has_src) {
                self.start_preview_render(&ctx);
            }
            if item(ui, "Stop preview render", "Esc", self.preview_job.as_ref().is_some_and(|j| !j.stopping)) {
                self.stop_preview();
            }
            if item(ui, "Export", "Ctrl+E", has_src) {
                self.export(None);
            }
            if item(ui, "Export as…", "Ctrl+Shift+X", has_src) {
                self.export_as();
            }
            ui.separator();
            if item(ui, "Show preview command", "Ctrl+Shift+P", has_src) {
                self.show_command(true);
            }
            if item(ui, "Show export command", "Ctrl+Shift+E", has_src) {
                self.show_command(false);
            }
            ui.separator();
            if item(ui, "Export queue", "", true) {
                self.popover = Some(Popover::Queue);
                self.popover_opened = true;
            }
        });
        let help = ui.menu_button("Help", |ui| {
            ui.set_min_width(220.0);
            if item(ui, "About Prism Video Enhancer", "", true) {
                self.dialog = Some(Dialog::About);
            }
            ui.separator();
            if item(ui, "Open data folder", "", true) {
                open_path(&paths::data_dir());
            }
            if item(ui, "Open models folder", "", true) {
                let _ = std::fs::create_dir_all(paths::models_dir());
                open_path(&paths::models_dir());
            }
            if item(ui, "Open temporary folder", "", true) {
                let _ = std::fs::create_dir_all(paths::temp_root());
                open_path(&paths::temp_root());
            }
        });
        help.response.rect.right()
    }

    fn select_relative(&mut self, d: isize) {
        let n = self.project.sources.len() as isize;
        if n == 0 {
            return;
        }
        let i = self.project.selected.map(|i| i as isize).unwrap_or(0);
        self.select_source((i + d).clamp(0, n - 1) as usize);
    }

    pub(crate) fn save_project(&mut self) {
        match self.project.save(&self.project_dir) {
            Ok(_) => self.notify(format!("Saved project “{}”", self.project.name), false),
            Err(e) => self.notify(format!("{e:#}"), true),
        }
    }

    pub(crate) fn save_project_as(&mut self, ctx: &egui::Context) {
        if let Some(dir) = rfd::FileDialog::new().set_title("Save project in folder").pick_folder() {
            self.project.name = dir.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| self.project.name.clone());
            match self.project.save(&dir) {
                Ok(_) => {
                    self.project_dir = dir;
                    self.prefs.last_project = Some(self.project_dir.clone());
                    self.prefs_changed();
                }
                Err(e) => self.notify(format!("{e:#}"), true),
            }
            ctx.request_repaint();
        }
    }

    pub(crate) fn open_project_dialog(&mut self, ctx: &egui::Context) {
        if let Some(f) = rfd::FileDialog::new().add_filter("Prism project", &["json"]).set_directory(paths::data_dir().join("projects")).pick_file() {
            if let Some(dir) = f.parent() {
                self.switch_project(dir.to_path_buf(), ctx);
            }
        }
    }

    fn header(&mut self, ui: &mut Ui) {
        let full = ui.max_rect();
        let has_src = self.selected_source().is_some();
        // Left: wordmark and project breadcrumb.
        child(ui, full, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_space(16.0);
            let p = ui.painter().clone();
            let x = ui.cursor().min.x;
            let r = p.text(pos2(x, full.center().y), Align2::LEFT_CENTER, "Prism", theme::semibold(16.0), TEXT);
            let r2 = p.text(pos2(r.right() + 6.0, full.center().y + 1.0), Align2::LEFT_CENTER, env!("CARGO_PKG_VERSION").rsplit_once('.').map(|v| v.0).unwrap_or("1"), theme::body(14.0), MUTED);
            ui.add_space(r2.right() - x + 18.0);
            let (div, _) = ui.allocate_exact_size(vec2(1.0, full.height()), Sense::hover());
            p.rect_filled(div, 0.0, RAISED);
            ui.add_space(16.0);
            let crumb = format!("Projects   ›   {}", self.project.name);
            let galley = p.layout_no_wrap(crumb, theme::body(13.0), TEXT);
            let (cr, resp) = ui.allocate_exact_size(galley.size() + vec2(12.0, 16.0), Sense::click());
            if resp.hovered() {
                p.rect_filled(cr, 4.0, HOVER);
            }
            let projects_w = p.layout_no_wrap("Projects".into(), theme::body(13.0), MUTED).size().x;
            p.text(pos2(cr.left() + 6.0, cr.center().y), Align2::LEFT_CENTER, "Projects", theme::body(13.0), MUTED);
            p.text(pos2(cr.left() + 6.0 + projects_w + 8.0, cr.center().y), Align2::LEFT_CENTER, "›", theme::body(13.0), MUTED);
            p.text(pos2(cr.left() + 6.0 + projects_w + 22.0, cr.center().y), Align2::LEFT_CENTER, &self.project.name, theme::body(13.0), TEXT);
            self.crumb_anchor = pos2(cr.left(), full.bottom() + 4.0);
            if resp.clicked() {
                self.popover = if self.popover == Some(Popover::Projects) { None } else { Some(Popover::Projects) };
                self.popover_opened = true;
            }
        });
        // Centre: view modes.
        let cw = 48.0 * 4.0;
        let centre = Rect::from_center_size(pos2(full.center().x - if self.prefs.show_panel { RIGHT_W / 2.0 } else { 0.0 }, full.center().y), vec2(cw, full.height()));
        child(ui, centre, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_enabled_ui(has_src && self.cropping.is_none(), |ui| {
                let s = vec2(48.0, full.height());
                if widgets::tip(widgets::icon_button(ui, Icon::FullScreen, s, 17.0, self.full_preview), "Full screen preview", "Hide the panels and show a larger view of the video (F).").clicked() {
                    self.full_preview = !self.full_preview;
                }
                let v = self.prefs.view;
                if widgets::tip(widgets::icon_button(ui, Icon::SingleView, s, 17.0, v == ViewMode::Single), "Single view", "Shows one tab. Press and hold on the video to see the original (Shift+1).").clicked() {
                    self.set_view(ViewMode::Single);
                }
                if widgets::tip(widgets::icon_button(ui, Icon::SplitView, s, 17.0, v == ViewMode::Split), "Split view", "Shows two tabs on either side of a divider you can drag (Shift+2).").clicked() {
                    self.set_view(ViewMode::Split);
                }
                if widgets::tip(widgets::icon_button(ui, Icon::SideBySide, s, 17.0, v == ViewMode::SideBySide), "Side-by-side view", "Shows two tabs next to each other (Shift+3).").clicked() {
                    self.set_view(ViewMode::SideBySide);
                }
            });
        });
        // Right: presets, input settings, panel toggle.
        child(ui, full, Layout::right_to_left(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            let s = vec2(48.0, full.height());
            if widgets::tip(widgets::icon_button(ui, Icon::Panel, s, 17.0, !self.prefs.show_panel), "Filter panel", "Show or hide the adjustments panel (Tab).").clicked() {
                self.prefs.show_panel = !self.prefs.show_panel;
                self.prefs_changed();
            }
            let (d, _) = ui.allocate_exact_size(vec2(1.0, full.height()), Sense::hover());
            ui.painter().rect_filled(d, 0.0, RAISED);
            ui.add_enabled_ui(has_src, |ui| {
                if widgets::tip(widgets::icon_button(ui, Icon::Gear, s, 17.0, self.popover == Some(Popover::InputSettings)), "Input settings", "Rotation, deinterlacing, pixel aspect and colour overrides for the selected video.").clicked() {
                    self.popover = if self.popover == Some(Popover::InputSettings) { None } else { Some(Popover::InputSettings) };
                    self.popover_opened = true;
                }
                let (d, _) = ui.allocate_exact_size(vec2(1.0, full.height()), Sense::hover());
                ui.painter().rect_filled(d, 0.0, RAISED);
                let r = widgets::icon_text_button(ui, Icon::Presets, "Presets", full.height(), self.popover == Some(Popover::Presets));
                self.presets_button = r.rect;
                if r.clicked() {
                    self.popover = if self.popover == Some(Popover::Presets) { None } else { Some(Popover::Presets) };
                    self.popover_opened = true;
                }
            });
        });
    }

    // ---------------------------------------------------------------- tabs and breadcrumb

    fn tab_strip(&mut self, ui: &mut Ui, rect: Rect, side: usize) {
        let Some(si) = self.project.selected else { return };
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, PANEL);
        p.rect_filled(Rect::from_min_max(pos2(rect.left(), rect.bottom() - 1.0), rect.max), 0.0, RAISED);
        let single = self.prefs.view == ViewMode::Single || self.cropping.is_some();
        let src = &self.project.sources[si];
        let selected = match (single, side) {
            (true, _) => if self.show_original { Content::Original } else { Content::Tab(src.active_tab) },
            (false, 0) => self.compare.filter(|i| *i < src.tabs.len()).map(Content::Tab).unwrap_or(Content::Original),
            _ => Content::Tab(src.active_tab),
        };
        let mut items: Vec<Content> = Vec::new();
        if single || side == 0 {
            items.push(Content::Original);
        }
        items.extend((0..src.tabs.len()).map(Content::Tab));
        let mut x = rect.left();
        let mut clicked = None;
        let mut close = None;
        let mut rename = None;
        let mut menu_action = None;
        for c in items {
            let (name, closable) = match c {
                Content::Original => ("Original".to_string(), false),
                Content::Tab(i) => (src.tabs[i].name.clone(), true),
            };
            if let (Content::Tab(i), Some((ri, _))) = (c, &self.renaming_tab) {
                if *ri == i && side == if single { 0 } else { 1 } {
                    let r = Rect::from_min_size(pos2(x, rect.top()), vec2(150.0, rect.height() - 1.0));
                    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(r.shrink2(vec2(6.0, 4.0))));
                    let (_, text) = self.renaming_tab.as_mut().unwrap();
                    let te = child.add(egui::TextEdit::singleline(text).desired_width(r.width() - 12.0));
                    if te.lost_focus() {
                        rename = Some(text.trim().to_string());
                    } else {
                        te.request_focus();
                    }
                    x = r.right();
                    continue;
                }
            }
            let galley = p.layout_no_wrap(name.clone(), theme::body(13.0), TEXT);
            let icon_w = if closable { 22.0 } else { 0.0 };
            let close_w = if closable { 22.0 } else { 0.0 };
            let w = (16.0 + icon_w + galley.size().x + close_w + 14.0).min(220.0);
            let r = Rect::from_min_size(pos2(x, rect.top()), vec2(w, rect.height() - 1.0));
            if r.left() > rect.right() {
                break;
            }
            let resp = ui.interact(r, ui.scope_id().with(("tab", side, format!("{c:?}"))), Sense::click());
            let is_sel = c == selected;
            if is_sel {
                p.rect_filled(r, 0.0, RAISED);
                p.rect_filled(Rect::from_min_size(r.min, vec2(r.width(), 2.0)), 0.0, ACCENT);
            } else if resp.hovered() {
                p.rect_filled(r, 0.0, Color32::from_rgb(0x26, 0x26, 0x26));
            }
            let color = if is_sel { TEXT } else { MUTED };
            let mut tx = r.left() + 14.0;
            if closable {
                icons::paint_at(&p, pos2(tx + 7.0, r.center().y), 13.0, Icon::Cube, color);
                tx += icon_w;
            }
            p.with_clip_rect(Rect::from_min_max(pos2(tx, r.top()), pos2(r.right() - close_w - 6.0, r.bottom()))).galley(pos2(tx, r.center().y - galley.size().y / 2.0), galley, color);
            if let Content::Tab(i) = c {
                if is_sel || resp.hovered() {
                    let cr = Rect::from_center_size(pos2(r.right() - 16.0, r.center().y), Vec2::splat(18.0));
                    let cresp = ui.interact(cr, ui.scope_id().with(("tabclose", side, i)), Sense::click());
                    if cresp.hovered() {
                        p.rect_filled(cr, 3.0, HOVER);
                    }
                    icons::paint_at(&p, cr.center(), 11.0, Icon::Close, if cresp.hovered() { TEXT } else { MUTED });
                    if cresp.clicked() {
                        close = Some(i);
                    }
                }
                if resp.double_clicked() {
                    self.renaming_tab = Some((i, src.tabs[i].name.clone()));
                }
                resp.context_menu(|ui| {
                    if ui.button("Rename").clicked() {
                        menu_action = Some(("rename", i));
                    }
                    if ui.button("Duplicate tab").clicked() {
                        menu_action = Some(("duplicate", i));
                    }
                    if ui.button("Save as preset…").clicked() {
                        menu_action = Some(("preset", i));
                    }
                    if ui.button("Apply to all sources").clicked() {
                        menu_action = Some(("all", i));
                    }
                    ui.separator();
                    if ui.button("Close tab").clicked() {
                        menu_action = Some(("close", i));
                    }
                });
            }
            if resp.clicked() && close.is_none() {
                clicked = Some(c);
            }
            x = r.right() + 1.0;
        }
        // "+" adds a tab.
        let pr = Rect::from_min_size(pos2(x + 2.0, rect.top()), vec2(rect.height() - 1.0, rect.height() - 1.0));
        if (side == 1 || single) && pr.right() <= rect.right() + 1.0 {
            let presp = ui.interact(pr, ui.scope_id().with(("tabplus", side)), Sense::click());
            if presp.hovered() {
                p.rect_filled(pr, 0.0, HOVER);
            }
            icons::paint_at(&p, pr.center(), 14.0, Icon::Plus, TEXT);
            if widgets::tip(presp, "New tab", "Adds a tab to compare another set of adjustments.").clicked() {
                self.new_tab(false);
            }
        }
        if let Some(name) = rename {
            if let Some((i, _)) = self.renaming_tab.take() {
                if !name.is_empty() {
                    self.project.sources[si].tabs[i].name = name;
                    self.mark_dirty();
                }
            }
        }
        if let Some(i) = close {
            self.close_tab(i);
        } else if let Some(c) = clicked {
            match (single, side, c) {
                (true, _, Content::Original) => self.show_original = true,
                (false, 0, Content::Original) => self.compare = None,
                (false, 0, Content::Tab(i)) => self.compare = Some(i),
                (_, _, Content::Tab(i)) => {
                    self.show_original = false;
                    self.project.sources[si].active_tab = i;
                    self.mark_dirty();
                }
                _ => {}
            }
        }
        if let Some((what, i)) = menu_action {
            match what {
                "rename" => self.renaming_tab = Some((i, self.project.sources[si].tabs[i].name.clone())),
                "duplicate" => {
                    self.project.sources[si].active_tab = i;
                    self.new_tab(true);
                }
                "preset" => {
                    self.project.sources[si].active_tab = i;
                    self.dialog = Some(Dialog::NewPreset { name: self.project.sources[si].tabs[i].name.clone(), description: String::new() });
                }
                "all" => {
                    self.project.sources[si].active_tab = i;
                    self.apply_to_all_sources();
                }
                _ => self.close_tab(i),
            }
        }
    }

    fn breadcrumb(&mut self, ui: &mut Ui, rect: Rect, content: Option<Content>) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, Color32::from_rgb(0x1B, 0x1B, 0x1B));
        let Some(src) = self.selected_source() else { return };
        // "fx" badge with an arrow-shaped end.
        let badge = Rect::from_min_size(rect.min, vec2(34.0, rect.height()));
        p.add(egui::Shape::convex_polygon(vec![badge.left_top(), badge.right_top() - vec2(8.0, 0.0), pos2(badge.right(), badge.center().y), badge.right_bottom() - vec2(8.0, 0.0), badge.left_bottom()], PANEL, Stroke::NONE));
        p.text(pos2(badge.left() + 12.0, badge.center().y), Align2::LEFT_CENTER, "ƒx", theme::body(13.0).clone(), LABEL);
        let items = match content {
            Some(Content::Tab(i)) => adjustments(&src.tabs[i].settings, &self.catalog),
            _ => Vec::new(),
        };
        let mut x = badge.right() + 10.0;
        if items.is_empty() {
            p.text(pos2(x, rect.center().y), Align2::LEFT_CENTER, "No adjustments", theme::body(12.5), LABEL);
        }
        for (name, detail) in items {
            let r = p.text(pos2(x, rect.center().y), Align2::LEFT_CENTER, name, theme::body(12.5), TEXT);
            let r2 = p.text(pos2(r.right() + 6.0, rect.center().y), Align2::LEFT_CENTER, detail, theme::body(12.5), MUTED);
            x = r2.right() + 18.0;
            if x > rect.right() - 40.0 {
                break;
            }
        }
        if let Some(Content::Tab(i)) = content {
            let er = Rect::from_center_size(pos2(rect.right() - 18.0, rect.center().y), Vec2::splat(22.0));
            let resp = ui.interact(er, ui.scope_id().with(("crumbmenu", rect.left() as i32)), Sense::click());
            if resp.hovered() {
                p.rect_filled(er, 4.0, HOVER);
            }
            icons::paint_at(&p, er.center(), 14.0, Icon::Ellipsis, TEXT);
            let mut action = None;
            egui::Popup::menu(&resp).show(|ui| {
                ui.set_min_width(190.0);
                for (label, a) in [("Rename tab", 0), ("Duplicate tab", 1), ("Save as preset…", 2), ("Apply to all sources", 3), ("Reset adjustments", 4), ("Close tab", 5)] {
                    if a == 4 {
                        ui.separator();
                    }
                    if ui.button(label).clicked() {
                        action = Some(a);
                    }
                }
            });
            if let (Some(a), Some(si)) = (action, self.project.selected) {
                self.project.sources[si].active_tab = i;
                match a {
                    0 => self.renaming_tab = Some((i, self.project.sources[si].tabs[i].name.clone())),
                    1 => self.new_tab(true),
                    2 => self.dialog = Some(Dialog::NewPreset { name: self.project.sources[si].tabs[i].name.clone(), description: String::new() }),
                    3 => self.apply_to_all_sources(),
                    4 => {
                        let s = &mut self.project.sources[si].tabs[i].settings;
                        *s = input_only(s);
                        self.mark_dirty();
                    }
                    _ => self.close_tab(i),
                }
            }
        }
    }

    // ---------------------------------------------------------------- viewer

    fn drop_zone(&mut self, ui: &mut Ui, rect: Rect) {
        let p = ui.painter_at(rect);
        let zone = Rect::from_center_size(rect.center(), vec2(rect.width().min(560.0) - 40.0, 260.0));
        let hovering = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());
        let stroke = Stroke::new(1.5, if hovering { ACCENT } else { Color32::from_rgb(0x3A, 0x3A, 0x3A) });
        // Dashed outline.
        let pts = [zone.left_top(), zone.right_top(), zone.right_bottom(), zone.left_bottom(), zone.left_top()];
        for w in pts.windows(2) {
            p.add(egui::Shape::dashed_line(&[w[0], w[1]], stroke, 8.0, 6.0));
        }
        icons::paint_at(&p, zone.center() - vec2(0.0, 70.0), 40.0, Icon::Film, if hovering { ACCENT } else { LABEL });
        p.text(zone.center() - vec2(0.0, 22.0), Align2::CENTER_CENTER, "Drag and drop videos here", theme::semibold(17.0), TEXT);
        p.text(zone.center() + vec2(0.0, 4.0), Align2::CENTER_CENTER, "MP4, MOV, MKV, AVI, WebM, MXF and other FFmpeg formats", theme::body(12.5), MUTED);
        let br = Rect::from_center_size(zone.center() + vec2(0.0, 56.0), vec2(140.0, 36.0));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(br));
        let enabled = self.tools.is_some();
        if child.add_enabled_ui(enabled, |ui| widgets::primary_button(ui, Some(Icon::Folder), "Browse", br.size())).inner.clicked() {
            let ctx = ui.ctx().clone();
            self.import_dialog(&ctx);
        }
        if !self.probes.is_empty() {
            p.text(zone.center_bottom() + vec2(0.0, 24.0), Align2::CENTER_CENTER, "Reading video…", theme::body(12.5), MUTED);
        }
    }

    fn draw_content(&self, p: &egui::Painter, side: usize, content: Option<Content>, img: Rect, uv: Rect, peek: bool) {
        let Some(orig) = &self.original else { return };
        let draw = |id: egui::TextureId, size: Vec2| {
            // Fit the texture's own aspect inside the shared image rect.
            let k = (img.width() / size.x).min(img.height() / size.y);
            let r = Rect::from_center_size(img.center(), size * k);
            p.image(id, r, uv, Color32::WHITE);
        };
        match content {
            None | Some(Content::Original) => draw(orig.handle.id(), orig.handle.size_vec2()),
            Some(Content::Tab(i)) => {
                if peek {
                    draw(orig.handle.id(), orig.handle.size_vec2());
                    widgets::pill(p, img.center(), "Original");
                    return;
                }
                let slot = &self.slots[side];
                match &slot.tex {
                    Some(t) => draw(t.handle.id(), t.handle.size_vec2()),
                    None => {
                        draw(orig.handle.id(), orig.handle.size_vec2());
                        let has_adj = self.selected_source().is_some_and(|s| s.tabs.get(i).is_some_and(|t| !adjustments(&t.settings, &self.catalog).is_empty()));
                        if has_adj {
                            let text = if slot.pending.is_some() { "Processing…" } else { "Unprocessed" };
                            widgets::pill(p, img.center(), text);
                        }
                    }
                }
            }
        }
    }

    fn viewer(&mut self, ui: &mut Ui, rect: Rect) {
        let resp = ui.interact(rect, ui.scope_id().with("viewer"), Sense::click_and_drag());
        let painter = ui.painter_at(rect);
        painter.rect_filled(rect, 0.0, VIEWER_BG);
        if self.project.sources.is_empty() {
            self.drop_zone(ui, rect);
            return;
        }
        let Some(orig) = &self.original else {
            let text = if self.probes.is_empty() { "Loading frame…" } else { "Reading video…" };
            painter.text(rect.center(), Align2::CENTER_CENTER, text, theme::body(15.0), MUTED);
            return;
        };
        let contents = self.contents();
        let base = orig.handle.size_vec2();
        let side_mode = self.prefs.view == ViewMode::SideBySide && self.cropping.is_none();
        let split_mode = self.prefs.view == ViewMode::Split && self.cropping.is_none();
        let area = if side_mode { Rect::from_min_size(rect.min, vec2(rect.width() / 2.0, rect.height())) } else { rect };
        let area = area.shrink(8.0);
        let fit = (area.width() / base.x).min(area.height() / base.y);
        let img = Rect::from_center_size(area.center(), base * fit);
        let ppp = ui.ctx().pixels_per_point();
        if let Some(info) = self.selected_info() {
            let (dw, _) = media::display_size(info, 1 << 20, 1 << 20);
            self.fit_scale = img.width() * ppp / dw.max(1) as f32;
        }
        // Zoom around the cursor; drag pans (or moves the split divider / crop).
        if let Some(pos) = resp.hover_pos() {
            let scroll = ui.input(|i| i.smooth_scroll_delta.y);
            if scroll != 0.0 && self.cropping.is_none() {
                let local = side_local(pos, img, side_mode, rect);
                let before = self.uv_at(local);
                self.zoom = (self.zoom * (1.0 + scroll * 0.0015)).clamp(1.0, 64.0);
                let after = self.uv_at(local);
                self.center += before - after;
            }
        }
        let split_x = img.min.x + img.width() * self.split;
        let (top, bottom) = (img.min.y.max(rect.min.y), img.max.y.min(rect.max.y));
        let handle = pos2(split_x, (top + bottom) / 2.0);
        // The divider is grabbed by where the button went down, not where the pointer is once
        // egui's drag threshold is passed (a quick grab is already past a narrow hit zone).
        let on_divider = |p: Pos2| split_mode && (p.distance(handle) < 20.0 || ((p.x - split_x).abs() < 16.0 && p.y >= top && p.y <= bottom));
        let near_split = resp.hover_pos().is_some_and(on_divider);
        if !resp.is_pointer_button_down_on() {
            self.split_drag = false;
        } else if !self.split_drag && ui.input(|i| i.pointer.press_origin()).is_some_and(on_divider) {
            self.split_drag = true;
        }
        if let Some(mut crop) = self.cropping {
            self.crop_overlay(ui, &resp, img, &mut crop);
            self.cropping = Some(crop);
        } else if self.split_drag {
            if let Some(p) = resp.interact_pointer_pos() {
                self.split = ((p.x - img.min.x) / img.width()).clamp(0.0, 1.0);
            }
        } else if resp.dragged() {
            self.center -= resp.drag_delta() / img.size() / self.zoom;
        }
        let dragging_split = self.split_drag;
        let half = 0.5 / self.zoom;
        self.center = self.center.clamp(Vec2::splat(half), Vec2::splat(1.0 - half));
        let uv = if self.cropping.is_some() { Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)) } else { Rect::from_min_max(pos2(self.center.x - half, self.center.y - half), pos2(self.center.x + half, self.center.y + half)) };
        let peek = self.prefs.view == ViewMode::Single && resp.is_pointer_button_down_on() && resp.drag_delta() == Vec2::ZERO && ui.input(|i| i.pointer.press_start_time().is_some_and(|t| i.time - t > 0.15));
        let clip_img = painter.with_clip_rect(rect);
        if side_mode {
            let img2 = img.translate(vec2(rect.width() / 2.0, 0.0));
            self.draw_content(&clip_img.with_clip_rect(Rect::from_min_max(rect.min, pos2(rect.center().x, rect.max.y))), 0, contents[0], img, uv, false);
            self.draw_content(&clip_img.with_clip_rect(Rect::from_min_max(pos2(rect.center().x, rect.min.y), rect.max)), 1, contents[1], img2, uv, false);
            painter.rect_filled(Rect::from_center_size(pos2(rect.center().x, rect.center().y), vec2(1.0, rect.height())), 0.0, BG);
        } else if split_mode {
            self.draw_content(&clip_img, 1, contents[1], img, uv, false);
            let left = Rect::from_min_max(img.min, pos2(split_x, img.max.y));
            self.draw_content(&clip_img.with_clip_rect(left.intersect(rect)), 0, contents[0], img, uv, false);
            // Paint where the divider is now (it may have moved this frame).
            let split_x = img.min.x + img.width() * self.split;
            painter.line_segment([pos2(split_x, top), pos2(split_x, bottom)], Stroke::new(1.5, Color32::WHITE));
            let hc = pos2(split_x, handle.y);
            painter.circle_filled(hc, 13.0, Color32::from_rgba_unmultiplied(0x21, 0x21, 0x21, 230));
            painter.circle_stroke(hc, 13.0, Stroke::new(1.5, Color32::WHITE));
            icons::paint_at(&painter, hc - vec2(4.0, 0.0), 9.0, Icon::ChevronLeft, Color32::WHITE);
            icons::paint_at(&painter, hc + vec2(4.0, 0.0), 9.0, Icon::ChevronRight, Color32::WHITE);
            if near_split || dragging_split {
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            }
        } else {
            self.draw_content(&clip_img, 1, contents[1], img, uv, peek);
            if let Some(crop) = &self.cropping {
                paint_crop(&painter, img, crop);
            }
        }
        if self.cropping.is_some() {
            self.crop_toolbar(ui, rect);
        }
    }

    fn uv_at(&self, local: Vec2) -> Vec2 {
        self.center + (local - Vec2::splat(0.5)) / self.zoom
    }

    fn crop_overlay(&mut self, ui: &mut Ui, resp: &egui::Response, img: Rect, crop: &mut CropEdit) {
        let to_screen = |p: Pos2| pos2(img.left() + p.x * img.width(), img.top() + p.y * img.height());
        let screen = Rect::from_min_max(to_screen(crop.rect.min), to_screen(crop.rect.max));
        let hover = resp.hover_pos();
        // Handle ids: 0..=7 corners/edges clockwise from top-left, 8 = move.
        let handles = [screen.left_top(), screen.center_top(), screen.right_top(), screen.right_center(), screen.right_bottom(), screen.center_bottom(), screen.left_bottom(), screen.left_center()];
        let hit = |p: Pos2| handles.iter().position(|h| h.distance(p) < 12.0).map(|i| i as u8).or_else(|| screen.contains(p).then_some(8));
        if let Some(h) = hover.and_then(hit) {
            ui.ctx().set_cursor_icon(match h {
                0 | 4 => CursorIcon::ResizeNwSe,
                2 | 6 => CursorIcon::ResizeNeSw,
                1 | 5 => CursorIcon::ResizeVertical,
                3 | 7 => CursorIcon::ResizeHorizontal,
                _ => CursorIcon::Move,
            });
        }
        if resp.drag_started() {
            if let Some(p) = resp.interact_pointer_pos() {
                crop.drag = hit(p).map(|h| (h, p, crop.rect));
            }
        }
        if let (Some((h, start, orig)), Some(p)) = (crop.drag, resp.interact_pointer_pos()) {
            let d = (p - start) / img.size();
            let mut r = orig;
            let min = 0.05;
            match h {
                8 => {
                    let dx = d.x.clamp(-orig.min.x, 1.0 - orig.max.x);
                    let dy = d.y.clamp(-orig.min.y, 1.0 - orig.max.y);
                    r = orig.translate(vec2(dx, dy));
                }
                _ => {
                    if matches!(h, 0 | 6 | 7) {
                        r.min.x = (orig.min.x + d.x).clamp(0.0, orig.max.x - min);
                    }
                    if matches!(h, 2 | 3 | 4) {
                        r.max.x = (orig.max.x + d.x).clamp(orig.min.x + min, 1.0);
                    }
                    if matches!(h, 0 | 1 | 2) {
                        r.min.y = (orig.min.y + d.y).clamp(0.0, orig.max.y - min);
                    }
                    if matches!(h, 4 | 5 | 6) {
                        r.max.y = (orig.max.y + d.y).clamp(orig.min.y + min, 1.0);
                    }
                }
            }
            crop.rect = r;
        }
        if resp.drag_stopped() {
            crop.drag = None;
        }
    }

    fn crop_toolbar(&mut self, ui: &mut Ui, rect: Rect) {
        let bar = Rect::from_center_size(pos2(rect.center().x, rect.bottom() - 34.0), vec2(330.0, 40.0));
        ui.painter().rect_filled(bar, 6.0, Color32::from_rgba_unmultiplied(0x21, 0x21, 0x21, 240));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(bar.shrink(6.0)).layout(Layout::left_to_right(Align::Center)));
        if widgets::button(&mut child, "Reset", vec2(90.0, 28.0)).clicked() {
            if let Some(c) = &mut self.cropping {
                c.rect = Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0));
            }
        }
        if widgets::button(&mut child, "Cancel", vec2(90.0, 28.0)).clicked() {
            self.cropping = None;
        }
        if widgets::primary_button(&mut child, None, "Apply crop", vec2(110.0, 28.0)).clicked() {
            self.toggle_crop();
        }
    }

    // ---------------------------------------------------------------- transport and timeline

    fn transport(&mut self, ui: &mut Ui) {
        let Some(info) = self.selected_info().cloned() else { return };
        let fps = info.fps.as_f64().max(1.0);
        let full = ui.max_rect();
        let row = Rect::from_min_size(full.min, vec2(full.width(), 48.0));
        ui.painter().rect_filled(row, 0.0, PANEL);
        // Left: timecode.
        let tc = Rect::from_min_size(row.min + vec2(12.0, 8.0), vec2(108.0, 32.0));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(tc).layout(Layout::left_to_right(Align::Center)));
        match &mut self.timecode_edit {
            Some(text) => {
                let te = child.add(egui::TextEdit::singleline(text).font(egui::FontId::monospace(13.0)).desired_width(104.0));
                if te.lost_focus() {
                    if child.input(|i| i.key_pressed(egui::Key::Enter)) {
                        if let Some(t) = timecode::parse(text, fps) {
                            self.playback = None;
                            self.playhead = t.clamp(0.0, info.duration);
                        }
                    }
                    self.timecode_edit = None;
                } else {
                    te.request_focus();
                }
            }
            None => {
                let r = child.add(egui::Label::new(egui::RichText::new(timecode::smpte(self.playhead, fps)).font(egui::FontId::monospace(13.5)).color(TEXT)).sense(Sense::click()));
                if widgets::tip(r, "Timecode", "Click to type a time (seconds, mm:ss or hh:mm:ss:ff).").clicked() {
                    self.timecode_edit = Some(timecode::smpte(self.playhead, fps));
                }
            }
        }
        ui.painter().rect_filled(Rect::from_min_size(pos2(row.left() + 128.0, row.top()), vec2(1.0, row.height())), 0.0, RAISED);
        // Centre: transport.
        let playing = self.playback.is_some();
        let bw = if full.width() < 900.0 { 36.0 } else { 44.0 };
        let centre = Rect::from_center_size(pos2(row.center().x, row.center().y), vec2(bw * 5.0, 48.0));
        let mut c = ui.new_child(egui::UiBuilder::new().max_rect(centre).layout(Layout::left_to_right(Align::Center)));
        c.spacing_mut().item_spacing.x = 0.0;
        let s = vec2(bw, 48.0);
        if widgets::tip(widgets::icon_button(&mut c, Icon::JumpIn, s, 15.0, false), "Back to in", "Shift+Left").clicked() {
            self.jump(false);
        }
        if widgets::tip(widgets::icon_button(&mut c, Icon::StepBack, s, 15.0, false), "Step backward", "Left").clicked() {
            self.step(-1.0);
        }
        if widgets::tip(widgets::icon_button(&mut c, if playing { Icon::Pause } else { Icon::Play }, s, 16.0, false), if playing { "Pause" } else { "Play" }, "Space").clicked() {
            self.toggle_play();
        }
        if widgets::tip(widgets::icon_button(&mut c, Icon::StepForward, s, 15.0, false), "Step forward", "Right").clicked() {
            self.step(1.0);
        }
        if widgets::tip(widgets::icon_button(&mut c, Icon::JumpOut, s, 15.0, false), "Forward to out", "Shift+Right").clicked() {
            self.jump(true);
        }
        // Right: crop, zoom, live render, render button.
        let right = Rect::from_min_max(pos2(centre.right() + 8.0, row.top()), row.max);
        let mut r = ui.new_child(egui::UiBuilder::new().max_rect(right).layout(Layout::right_to_left(Align::Center)));
        r.spacing_mut().item_spacing.x = 0.0;
        self.render_button(&mut r);
        r.add_space(12.0);
        let mut live = self.prefs.live_render;
        if widgets::tip(widgets::switch(&mut r, &mut live), "Live render", "Processes the paused frame automatically. Interpolation and stabilization need a rendered preview.").changed() {
            self.prefs.live_render = live;
            self.prefs_changed();
        }
        r.add_space(8.0);
        r.label(egui::RichText::new("Live render").color(LABEL).size(12.5));
        r.add_space(14.0);
        let pct = (self.fit_scale * self.zoom * 100.0).round();
        let zoom_text = if (self.zoom - 1.0).abs() < 1e-3 { "Fit".to_string() } else { format!("{pct:.0}%") };
        r.allocate_ui(vec2(84.0, 28.0), |ui| {
            widgets::combo(ui, "zoom", zoom_text, |ui| {
                if ui.selectable_label(false, "Fit").clicked() {
                    self.set_zoom_percent(None);
                }
                for p in [25.0, 50.0, 100.0, 200.0, 400.0, 800.0] {
                    if ui.selectable_label(false, format!("{p}%")).clicked() {
                        self.set_zoom_percent(Some(p));
                    }
                }
            });
        });
        r.add_space(4.0);
        if widgets::tip(widgets::icon_button(&mut r, Icon::Crop, vec2(40.0, 48.0), 15.0, self.cropping.is_some()), "Crop", "Crop the input video (Ctrl+Shift+C).").clicked() {
            self.toggle_crop();
        }
        // Timeline below.
        let tl = Rect::from_min_max(pos2(full.left(), row.bottom()), full.max);
        self.timeline(ui, tl, &info);
    }

    fn render_button(&mut self, ui: &mut Ui) {
        let h = 48.0;
        if self.preview_job.is_some() {
            self.render_progress(ui, h);
            return;
        }
        let chev_w = 28.0;
        let main_w = 104.0;
        let (rect, _) = ui.allocate_exact_size(vec2(main_w + chev_w, h), Sense::hover());
        let main = Rect::from_min_size(rect.min, vec2(main_w, h));
        let chev = Rect::from_min_max(pos2(main.right(), rect.top()), rect.max);
        let enabled = ui.is_enabled();
        let p = ui.painter().clone();
        let mresp = ui.interact(main, ui.scope_id().with("render-main"), Sense::click());
        let cresp = ui.interact(chev, ui.scope_id().with("render-chev"), Sense::click());
        let fill = |hot: bool| if !enabled { theme::ACCENT_DIM } else if hot { theme::ACCENT_HOVER } else { ACCENT };
        p.rect_filled(main, 0.0, fill(mresp.hovered()));
        p.rect_filled(chev, 0.0, fill(cresp.hovered()));
        p.rect_filled(Rect::from_min_size(pos2(chev.left(), chev.top() + 10.0), vec2(1.0, h - 20.0)), 0.0, theme::ON_ACCENT.gamma_multiply(0.3));
        p.text(main.center(), Align2::CENTER_CENTER, format!("Render {}", preview_len_label(self.prefs.preview_seconds)), theme::semibold(13.0), theme::ON_ACCENT);
        icons::paint_at(&p, chev.center(), 12.0, Icon::ChevronDown, theme::ON_ACCENT);
        if widgets::tip(mresp, "Render preview", "Processes a short clip from the playhead with every filter, including interpolation and stabilization (Ctrl+P).").clicked() {
            let ctx = ui.ctx().clone();
            self.start_preview_render(&ctx);
        }
        let mut pick = None;
        egui::Popup::menu(&cresp).align(egui::RectAlign::TOP_END).show(|ui| {
            ui.set_min_width(150.0);
            ui.label(egui::RichText::new("Preview length").color(MUTED).small());
            for s in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, -1.0] {
                if ui.selectable_label(self.prefs.preview_seconds == s, preview_len_label(s)).clicked() {
                    pick = Some(s);
                }
            }
        });
        if let Some(s) = pick {
            self.prefs.preview_seconds = s;
            self.prefs_changed();
        }
    }

    /// While a preview renders: a progress bar and a separate Stop button.
    fn render_progress(&mut self, ui: &mut Ui, h: f32) {
        let Some(job) = &self.preview_job else { return };
        let pr = *job.progress.lock().unwrap();
        let frac = ((job.done as f32 + pr.percent as f32 / 100.0) / job.total.max(1) as f32).clamp(0.0, 1.0);
        let stopping = job.stopping;
        let (stop, _) = ui.allocate_exact_size(vec2(76.0, h), Sense::hover());
        let (bar, _) = ui.allocate_exact_size(vec2(112.0, h), Sense::hover());
        let p = ui.painter().clone();
        // Progress.
        p.rect_filled(bar, 0.0, theme::ACCENT_DIM);
        p.rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * frac, h)), 0.0, ACCENT);
        let label = if stopping { "Stopping…".to_string() } else { format!("Rendering {}%", (frac * 100.0).round()) };
        p.text(bar.center() - vec2(0.0, 6.0), Align2::CENTER_CENTER, label, theme::semibold(12.5), theme::ON_ACCENT);
        let detail = if job.total > 1 { format!("{}/{} · {:.1} fps", (job.done + 1).min(job.total), job.total, pr.fps) } else { format!("{:.1} fps", pr.fps) };
        p.text(bar.center() + vec2(0.0, 9.0), Align2::CENTER_CENTER, detail, theme::body(10.5), theme::ON_ACCENT);
        ui.interact(bar, ui.scope_id().with("render-bar"), Sense::hover()).on_hover_text("Rendering the preview clip with every filter.");
        // Stop.
        let resp = ui.interact(stop, ui.scope_id().with("render-stop"), Sense::click());
        p.rect_filled(stop, 0.0, if resp.hovered() && !stopping { theme::SELECTED } else { RAISED });
        let red = theme::ERROR;
        icons::paint_at(&p, pos2(stop.left() + 22.0, stop.center().y), 13.0, Icon::Stop, if stopping { MUTED } else { red });
        p.text(pos2(stop.left() + 34.0, stop.center().y), Align2::LEFT_CENTER, "Stop", theme::semibold(13.0), if stopping { MUTED } else { TEXT });
        if widgets::tip(resp, "Stop rendering", "Cancels the preview render (Esc). Finished clips are kept.").clicked() {
            self.stop_preview();
        }
    }

    pub(crate) fn stop_preview(&mut self) {
        if let Some(job) = &mut self.preview_job {
            job.control.cancel();
            job.stopping = true;
        }
    }

    fn timeline(&mut self, ui: &mut Ui, rect: Rect, info: &VideoInfo) {
        let p = ui.painter_at(rect);
        p.rect_filled(rect, 0.0, PANEL);
        let inner = Rect::from_min_max(pos2(rect.left() + 16.0, rect.top()), pos2(rect.right() - 16.0, rect.bottom() - 6.0));
        let ruler = Rect::from_min_size(inner.min, vec2(inner.width(), 24.0));
        let track = Rect::from_min_size(pos2(inner.left(), ruler.bottom() + 2.0), vec2(inner.width(), 8.0));
        let strip = Rect::from_min_max(pos2(inner.left(), track.bottom() + 4.0), inner.max);
        let dur = info.duration.max(1e-3);
        let fps = info.fps.as_f64().max(1.0);
        let span = dur / self.timeline_zoom as f64;
        // Keep the playhead in view while playing or scrubbing past the edge.
        if self.playhead < self.timeline_start || self.playhead > self.timeline_start + span {
            self.timeline_start = (self.playhead - span * 0.1).max(0.0);
        }
        self.timeline_start = self.timeline_start.clamp(0.0, (dur - span).max(0.0));
        let t0 = self.timeline_start;
        let x_of = |t: f64| inner.left() + ((t - t0) / span) as f32 * inner.width();
        let t_of = |x: f32| (t0 + ((x - inner.left()) / inner.width()) as f64 * span).clamp(0.0, dur);
        // Ruler.
        let step = [1.0 / fps, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0].into_iter().find(|s| (s / span) as f32 * inner.width() > 90.0).unwrap_or(7200.0);
        let mut t = (t0 / step).floor() * step;
        while t <= t0 + span + step {
            let x = x_of(t);
            if x >= inner.left() - 1.0 && x <= inner.right() + 1.0 {
                p.line_segment([pos2(x, ruler.bottom() - 8.0), pos2(x, ruler.bottom())], Stroke::new(1.0, Color32::from_rgb(0x54, 0x54, 0x54)));
                for k in 1..4 {
                    let xs = x_of(t + step * k as f64 / 4.0);
                    p.line_segment([pos2(xs, ruler.bottom() - 4.0), pos2(xs, ruler.bottom())], Stroke::new(1.0, Color32::from_rgb(0x3A, 0x3A, 0x3A)));
                }
                let label = if step < 1.0 { timecode::smpte(t, fps) } else { hms(t) };
                p.text(pos2(x + 4.0, ruler.top() + 3.0), Align2::LEFT_TOP, label, theme::body(10.5), MUTED);
            }
            t += step;
        }
        // Preview clips of the visible tabs.
        p.rect_filled(track, 2.0, Color32::from_rgb(0x2B, 0x2B, 0x2B));
        let hashes: Vec<(usize, u64)> = self.contents().iter().enumerate().filter_map(|(side, c)| match c {
            Some(Content::Tab(i)) => self.run_settings(*i).map(|s| (side, settings_hash(&s))),
            _ => None,
        }).collect();
        for c in self.clips.iter().filter(|c| c.source == info.path) {
            if let Some((side, _)) = hashes.iter().find(|h| h.1 == c.settings_hash) {
                let r = Rect::from_min_max(pos2(x_of(c.start), track.top() + 1.0), pos2(x_of(c.start + c.duration), track.bottom() - 1.0)).intersect(track);
                p.rect_filled(r, 2.0, if *side == 1 { ACCENT } else { Color32::from_rgb(0x9A, 0x9A, 0x9A) });
            }
        }
        // Filmstrip.
        p.rect_filled(strip, 3.0, Color32::from_rgb(0x2B, 0x2B, 0x2B));
        if !self.thumbs.is_empty() {
            let n = self.thumbs.len();
            let clip = p.with_clip_rect(strip);
            let aspect = self.thumbs[0].aspect_ratio();
            let tile_w = strip.height() * aspect;
            // Tiles at their natural width; each shows the thumbnail nearest its time.
            let mut x = strip.left();
            while x < strip.right() {
                let tm = t_of(x + tile_w / 2.0);
                let i = ((tm / dur) * n as f64).floor().clamp(0.0, n as f64 - 1.0) as usize;
                let r = Rect::from_min_size(pos2(x, strip.top()), vec2(tile_w, strip.height()));
                clip.image(self.thumbs[i].id(), r, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
                x += tile_w;
            }
        }
        // In/out: dim outside, bracket handles.
        let trim = self.active_settings().and_then(|s| s.trim);
        if let Some(tr) = trim {
            let (xi, xo) = (x_of(tr.start), x_of(tr.end));
            let shade = Color32::from_black_alpha(160);
            if xi > strip.left() {
                p.rect_filled(Rect::from_min_max(strip.min, pos2(xi.min(strip.right()), strip.bottom())), 0.0, shade);
            }
            if xo < strip.right() {
                p.rect_filled(Rect::from_min_max(pos2(xo.max(strip.left()), strip.top()), strip.max), 0.0, shade);
            }
            p.rect_stroke(Rect::from_min_max(pos2(xi, strip.top()), pos2(xo, strip.bottom())), 3.0, Stroke::new(2.0, ACCENT), StrokeKind::Inside);
        }
        // Playhead.
        let px = x_of(self.playhead);
        if px >= inner.left() - 1.0 && px <= inner.right() + 1.0 {
            p.line_segment([pos2(px, ruler.top() + 6.0), pos2(px, strip.bottom())], Stroke::new(1.5, ACCENT));
            p.add(egui::Shape::convex_polygon(vec![pos2(px - 5.0, ruler.top()), pos2(px + 5.0, ruler.top()), pos2(px + 5.0, ruler.top() + 7.0), pos2(px, ruler.top() + 12.0), pos2(px - 5.0, ruler.top() + 7.0)], ACCENT, Stroke::NONE));
        }
        // Interaction: scrub, or drag an in/out edge.
        let resp = ui.interact(inner, ui.scope_id().with("timeline"), Sense::click_and_drag());
        if let Some(h) = resp.hover_pos() {
            if trim.is_some_and(|tr| (h.x - x_of(tr.start)).abs() < 6.0 || (h.x - x_of(tr.end)).abs() < 6.0) && h.y > strip.top() {
                ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            }
            // Ctrl+scroll arrives as a zoom gesture.
            let zoom = ui.input(|i| i.zoom_delta());
            if zoom != 1.0 {
                self.timeline_zoom = (self.timeline_zoom * zoom).clamp(1.0, 200.0);
            }
        }
        let handle_id = ui.scope_id().with("trim-handle");
        if resp.drag_started() {
            let h = resp.interact_pointer_pos().and_then(|pos| {
                let tr = trim?;
                if pos.y < strip.top() {
                    None
                } else if (pos.x - x_of(tr.start)).abs() < 6.0 {
                    Some(0u8)
                } else if (pos.x - x_of(tr.end)).abs() < 6.0 {
                    Some(1u8)
                } else {
                    None
                }
            });
            ui.memory_mut(|m| m.data.insert_temp(handle_id, h));
        }
        let handle = ui.memory(|m| m.data.get_temp::<Option<u8>>(handle_id)).flatten();
        if (resp.dragged() || resp.clicked()) && resp.interact_pointer_pos().is_some() {
            let pos = resp.interact_pointer_pos().unwrap();
            let t = (t_of(pos.x) * fps).round() / fps;
            match (handle, trim) {
                (Some(0), Some(tr)) if resp.dragged() => {
                    let tr = TrimRange { start: t.min(tr.end - 1.0 / fps), end: tr.end };
                    self.for_all_tabs(|s| s.trim = Some(tr));
                }
                (Some(1), Some(tr)) if resp.dragged() => {
                    let tr = TrimRange { start: tr.start, end: t.max(tr.start + 1.0 / fps) };
                    self.for_all_tabs(|s| s.trim = Some(tr));
                }
                _ => {
                    self.playback = None;
                    self.playhead = t;
                }
            }
        }
        if resp.drag_stopped() {
            ui.memory_mut(|m| m.data.insert_temp::<Option<u8>>(handle_id, None));
        }
    }

    fn bottom_bar(&mut self, ui: &mut Ui) {
        let full = ui.max_rect();
        self.bottom_top = full.top();
        let has_src = self.selected_source().is_some();
        child(ui, full, Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_space(8.0);
            ui.add_enabled_ui(has_src, |ui| {
                let s = vec2(40.0, full.height());
                if widgets::tip(widgets::icon_button(ui, Icon::MarkIn, s, 15.0, false), "Mark in", "Marks the playhead as the start of the export (I).").clicked() {
                    self.mark(false);
                }
                if widgets::tip(widgets::icon_button(ui, Icon::MarkOut, s, 15.0, false), "Mark out", "Marks the playhead as the end of the export (O).").clicked() {
                    self.mark(true);
                }
                let has_trim = self.active_settings().is_some_and(|s| s.trim.is_some());
                ui.add_enabled_ui(has_trim, |ui| {
                    if widgets::tip(widgets::icon_button(ui, Icon::ClearInOut, s, 15.0, false), "Clear in / out", "Resets in and out to the full length (Shift+R).").clicked() {
                        self.clear_trim();
                    }
                });
                if widgets::tip(widgets::icon_button(ui, Icon::ZoomReset, s, 15.0, false), "Reset timeline zoom", "Shows the whole video on the timeline.").clicked() {
                    self.timeline_zoom = 1.0;
                }
                ui.add_space(10.0);
                if widgets::small_icon(ui, Icon::Minus, MUTED).clicked() {
                    self.timeline_zoom = (self.timeline_zoom * 0.8).max(1.0);
                }
                ui.add_space(4.0);
                ui.spacing_mut().slider_width = 110.0;
                widgets::tip(ui.add(egui::Slider::new(&mut self.timeline_zoom, 1.0..=200.0).logarithmic(true).show_value(false).handle_shape(egui::style::HandleShape::Circle)), "Timeline zoom", "Drag to zoom in and out of the timeline (Ctrl+scroll on the timeline).");
                ui.add_space(4.0);
                if widgets::small_icon(ui, Icon::Plus, MUTED).clicked() {
                    self.timeline_zoom = (self.timeline_zoom * 1.25).min(200.0);
                }
            });
        });
        child(ui, full, Layout::right_to_left(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 0.0;
            ui.add_space(4.0);
            let running = self.queue.as_ref().map(|q| q.jobs().iter().filter(|j| !j.state.is_finished()).count()).unwrap_or(0);
            let r = widgets::icon_text_button(ui, Icon::Queue, "Export queue", full.height(), self.popover == Some(Popover::Queue));
            if running > 0 {
                widgets::count_badge(ui.painter(), pos2(r.rect.left() + 32.0, r.rect.top() + 14.0), running);
            }
            if r.clicked() {
                self.popover = if self.popover == Some(Popover::Queue) { None } else { Some(Popover::Queue) };
                self.popover_opened = true;
            }
            let n = self.project.sources.len();
            let r = widgets::icon_text_button(ui, Icon::Sources, "Sources", full.height(), self.popover == Some(Popover::Sources));
            if n > 1 {
                widgets::count_badge(ui.painter(), pos2(r.rect.left() + 32.0, r.rect.top() + 14.0), n);
            }
            if r.clicked() {
                self.popover = if self.popover == Some(Popover::Sources) { None } else { Some(Popover::Sources) };
                self.popover_opened = true;
            }
        });
    }

    // ---------------------------------------------------------------- right pane

    fn right_pane(&mut self, ui: &mut Ui) {
        let full = ui.max_rect();
        // Header: Adjustments | Codec settings, crop.
        let header = Rect::from_min_size(full.min, vec2(full.width(), 44.0));
        let p = ui.painter().clone();
        let mut x = header.left() + 18.0;
        for (tab, label) in [(RightTab::Adjustments, "Adjustments"), (RightTab::Codec, "Codec settings")] {
            let sel = self.right_tab == tab;
            let galley = p.layout_no_wrap(label.to_string(), if sel { theme::semibold(13.5) } else { theme::body(13.5) }, TEXT);
            let r = Rect::from_min_size(pos2(x, header.top()), vec2(galley.size().x + 4.0, header.height()));
            let resp = ui.interact(r, ui.scope_id().with(("rtab", label)), Sense::click());
            let color = if sel { TEXT } else if resp.hovered() { LABEL } else { MUTED };
            p.galley(pos2(r.left() + 2.0, r.center().y - galley.size().y / 2.0), galley, color);
            if sel {
                p.rect_filled(Rect::from_min_size(pos2(r.left(), r.bottom() - 2.0), vec2(r.width(), 2.0)), 0.0, ACCENT);
            }
            if resp.clicked() {
                self.right_tab = tab;
            }
            x = r.right() + 18.0;
        }
        let cr = Rect::from_min_size(pos2(header.right() - 48.0, header.top()), vec2(48.0, header.height()));
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(cr));
        child.add_enabled_ui(self.selected_source().is_some(), |ui| {
            if widgets::tip(widgets::icon_button(ui, Icon::Crop, cr.size(), 15.0, self.cropping.is_some()), "Crop video", "Crop the input video (Ctrl+Shift+C).").clicked() {
                self.toggle_crop();
            }
        });
        p.rect_filled(Rect::from_min_size(pos2(full.left(), header.bottom()), vec2(full.width(), 1.0)), 0.0, RAISED);
        // Footer: Quick export | Export as…
        let footer = Rect::from_min_max(pos2(full.left(), full.bottom() - 49.0), full.max);
        p.rect_filled(Rect::from_min_size(footer.min, vec2(footer.width(), 1.0)), 0.0, RAISED);
        let fb = Rect::from_min_max(pos2(footer.left(), footer.top() + 1.0), footer.max);
        let mut fchild = ui.new_child(egui::UiBuilder::new().max_rect(fb).layout(Layout::left_to_right(Align::Center)));
        fchild.spacing_mut().item_spacing.x = 0.0;
        let can_export = self.selected_source().is_some() && self.cropping.is_none() && self.queue.is_some();
        fchild.add_enabled_ui(can_export, |ui| {
            let qw = (fb.width() * 0.44).round();
            let r = widgets::icon_text_button(ui, Icon::ExportOut, "Quick export", fb.height(), false);
            let pad = qw - r.rect.width();
            if pad > 0.0 {
                ui.add_space(pad);
            }
            if widgets::tip(r, "Quick export", "Exports next to the input (or to the chosen output folder) with a generated name (Ctrl+E).").clicked() {
                self.export(None);
            }
            let w = ui.available_width();
            if widgets::tip(widgets::primary_button(ui, Some(Icon::Folder), "Export as…", vec2(w, fb.height())), "Export as", "Choose a location and name for the exported file.").clicked() {
                self.export_as();
            }
        });
        // Body.
        let body = Rect::from_min_max(pos2(full.left(), header.bottom() + 1.0), pos2(full.right(), footer.top()));
        let mut bui = ui.new_child(egui::UiBuilder::new().max_rect(body).layout(Layout::top_down(Align::Min)));
        let Some(si) = self.project.selected else {
            settings_ui::empty(&mut bui, "Import a video to adjust it.");
            return;
        };
        if self.show_original && self.prefs.view == ViewMode::Single && self.right_tab == RightTab::Adjustments {
            settings_ui::empty(&mut bui, "The Original tab shows the video without adjustments. Select a tab, or add one with +, to adjust it.");
            return;
        }
        let info = self.infos.get(&self.project.sources[si].path).cloned();
        let catalog = self.catalog.clone();
        let encoders = std::mem::take(&mut self.encoders);
        let fetching = self.fetching.as_ref().map(|f| f.0.clone());
        let before = self.project.sources[si].active().settings.clone();
        let tab = self.right_tab;
        let frame = self.original.as_ref().map(|t| (t.handle.id(), t.handle.size_vec2()));
        let mut list_view = self.prefs.model_list_view;
        let estimating = self.estimating.is_some();
        let actions = {
            let mut cx = PaneContext { catalog: &catalog, encoders: &encoders, info: info.as_ref(), fetching: fetching.as_deref(), estimating, frame, list_view: &mut list_view };
            let settings = &mut self.project.sources[si].active_mut().settings;
            egui::ScrollArea::vertical()
                .id_salt(("pane", tab as u8))
                .auto_shrink([false, false])
                .show(&mut bui, |ui| {
                    ui.set_width(ui.available_width());
                    let a = match tab {
                        RightTab::Adjustments => settings_ui::filters_tab(ui, settings, &mut cx),
                        RightTab::Codec => settings_ui::export_tab(ui, settings, &cx),
                    };
                    ui.add_space(16.0);
                    a
                })
                .inner
        };
        self.encoders = encoders;
        if list_view != self.prefs.model_list_view {
            self.prefs.model_list_view = list_view;
            self.prefs_changed();
        }
        if self.project.sources[si].active().settings != before {
            self.mark_dirty();
        }
        let ctx = ui.ctx().clone();
        self.handle_actions(actions, &ctx);
    }

    // ---------------------------------------------------------------- keyboard

    fn shortcuts(&mut self, ctx: &egui::Context) {
        use egui::{Key, KeyboardShortcut as K, Modifiers as M};
        let cs = M::COMMAND | M::SHIFT;
        let fire = |m: M, k: Key| ctx.input_mut(|i| i.consume_shortcut(&K::new(m, k)));
        let has_src = self.project.selected.is_some();
        if fire(M::COMMAND, Key::N) {
            self.dialog = Some(Dialog::NewProject { name: String::new() });
        }
        if fire(M::COMMAND, Key::O) {
            self.open_project_dialog(ctx);
        }
        if fire(cs, Key::S) {
            self.save_project_as(ctx);
        }
        if fire(M::COMMAND, Key::S) {
            self.save_project();
        }
        if fire(M::COMMAND, Key::I) {
            self.import_dialog(ctx);
        }
        if fire(M::COMMAND, Key::Comma) {
            self.dialog = Some(Dialog::Preferences);
        }
        if fire(M::COMMAND, Key::Q) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
        if !has_src || ctx.egui_wants_keyboard_input() {
            return;
        }
        if fire(cs, Key::Backspace) {
            self.dialog = Some(Dialog::CloseAll);
        }
        if fire(M::COMMAND, Key::Backspace) {
            if let Some(i) = self.project.selected {
                self.remove_source(i);
            }
        }
        if fire(cs, Key::C) {
            self.toggle_crop();
        }
        if fire(cs, Key::W) {
            self.close_all_tabs();
        }
        if fire(cs, Key::P) {
            self.show_command(true);
        }
        if fire(cs, Key::E) {
            self.show_command(false);
        }
        if fire(cs, Key::X) {
            self.export_as();
        }
        if fire(M::COMMAND, Key::P) {
            self.start_preview_render(ctx);
        }
        if fire(M::COMMAND, Key::E) {
            self.export(None);
        }
        if fire(M::COMMAND, Key::T) {
            self.new_tab(false);
        }
        for (k, pct) in [(Key::Num0, None), (Key::Num1, Some(100.0)), (Key::Num2, Some(200.0)), (Key::Num4, Some(400.0))] {
            if fire(M::COMMAND, k) {
                self.set_zoom_percent(pct);
            }
        }
        // Copy/paste arrive as events rather than key presses.
        let (copy, paste) = ctx.input(|i| (i.events.iter().any(|e| matches!(e, egui::Event::Copy)), i.events.iter().any(|e| matches!(e, egui::Event::Paste(_)))));
        if copy {
            self.copy_settings();
            self.notify("Copied the tab's settings", false);
        }
        if paste && self.clipboard.is_some() {
            self.paste_settings();
        }
        for (k, v) in [(Key::Num1, ViewMode::Single), (Key::Num2, ViewMode::Split), (Key::Num3, ViewMode::SideBySide)] {
            if fire(M::SHIFT, k) {
                self.set_view(v);
            }
        }
        if fire(M::SHIFT, Key::ArrowLeft) {
            self.jump(false);
        }
        if fire(M::SHIFT, Key::ArrowRight) {
            self.jump(true);
        }
        if fire(M::SHIFT, Key::R) {
            self.clear_trim();
        }
        if fire(M::NONE, Key::Space) {
            self.toggle_play();
        }
        if fire(M::NONE, Key::ArrowLeft) {
            self.step(-1.0);
        }
        if fire(M::NONE, Key::ArrowRight) {
            self.step(1.0);
        }
        if fire(M::NONE, Key::ArrowUp) {
            self.select_relative(-1);
        }
        if fire(M::NONE, Key::ArrowDown) {
            self.select_relative(1);
        }
        if fire(M::NONE, Key::I) {
            self.mark(false);
        }
        if fire(M::NONE, Key::O) {
            self.mark(true);
        }
        if fire(M::NONE, Key::F) {
            self.full_preview = !self.full_preview;
        }
        if fire(M::NONE, Key::Tab) {
            self.prefs.show_panel = !self.prefs.show_panel;
            self.prefs_changed();
        }
        // Escape belongs to an open menu or dropdown first.
        if !egui::Popup::is_any_open(ctx) && fire(M::NONE, Key::Escape) {
            if self.preview_job.is_some() {
                self.stop_preview();
            } else if self.cropping.is_some() {
                self.cropping = None;
            } else if self.popover.is_some() {
                self.popover = None;
            } else {
                self.full_preview = false;
            }
        }
    }
}

/// Settings with only the per-input choices kept (trim, crop, rotation, deinterlace, export).
fn input_only(s: &JobSettings) -> JobSettings {
    let mut n = JobSettings::default();
    n.trim = s.trim;
    n.crop = s.crop;
    n.rotation = s.rotation;
    n.deinterlace = s.deinterlace.clone();
    n.correction = s.correction.clone();
    n.export = s.export.clone();
    n
}

/// Pixel space crops are expressed in: the decoded frame after display rotation (and square
/// pixels when that conversion is on).
fn crop_space(info: &VideoInfo, square: bool) -> (u32, u32) {
    let mut w = if square { (info.width as f64 * info.sar.as_f64()).round() as u32 } else { info.width };
    let mut h = info.height;
    if info.rotation.rem_euclid(180) == 90 {
        std::mem::swap(&mut w, &mut h);
    }
    (w, h)
}

fn paint_crop(p: &egui::Painter, img: Rect, crop: &CropEdit) {
    let to_screen = |q: Pos2| pos2(img.left() + q.x * img.width(), img.top() + q.y * img.height());
    let r = Rect::from_min_max(to_screen(crop.rect.min), to_screen(crop.rect.max));
    let shade = Color32::from_black_alpha(150);
    p.rect_filled(Rect::from_min_max(img.min, pos2(img.max.x, r.min.y)), 0.0, shade);
    p.rect_filled(Rect::from_min_max(pos2(img.min.x, r.max.y), img.max), 0.0, shade);
    p.rect_filled(Rect::from_min_max(pos2(img.min.x, r.min.y), pos2(r.min.x, r.max.y)), 0.0, shade);
    p.rect_filled(Rect::from_min_max(pos2(r.max.x, r.min.y), pos2(img.max.x, r.max.y)), 0.0, shade);
    p.rect_stroke(r, 0.0, Stroke::new(1.5, Color32::WHITE), StrokeKind::Middle);
    for k in 1..3 {
        let fx = r.left() + r.width() * k as f32 / 3.0;
        let fy = r.top() + r.height() * k as f32 / 3.0;
        p.line_segment([pos2(fx, r.top()), pos2(fx, r.bottom())], Stroke::new(1.0, Color32::from_white_alpha(70)));
        p.line_segment([pos2(r.left(), fy), pos2(r.right(), fy)], Stroke::new(1.0, Color32::from_white_alpha(70)));
    }
    for h in [r.left_top(), r.center_top(), r.right_top(), r.right_center(), r.right_bottom(), r.center_bottom(), r.left_bottom(), r.left_center()] {
        p.rect_filled(Rect::from_center_size(h, Vec2::splat(9.0)), 1.0, Color32::WHITE);
    }
    let label = format!("{:.0}% × {:.0}%", crop.rect.width() * 100.0, crop.rect.height() * 100.0);
    p.text(r.left_top() + vec2(6.0, 6.0), Align2::LEFT_TOP, label, theme::body(11.5), Color32::WHITE);
}

/// Shortcut text for menus: egui's COMMAND modifier is Cmd on macOS.
pub(crate) fn key_label(s: &str) -> String {
    if cfg!(target_os = "macos") { s.replace("Ctrl+", "Cmd+") } else { s.to_string() }
}

/// Wording for revealing a file in the platform's file manager.
pub(crate) const REVEAL_LABEL: &str = if cfg!(windows) {
    "Show in Explorer"
} else if cfg!(target_os = "macos") {
    "Show in Finder"
} else {
    "Show in file manager"
};

pub(crate) fn preview_len_label(s: f64) -> String {
    if s < 0.0 { "In → Out".into() } else { format!("{s}s") }
}

fn hms(t: f64) -> String {
    let s = t.max(0.0).round() as u64;
    format!("{:02}:{:02}:{:02}", s / 3600, (s / 60) % 60, s % 60)
}

fn side_local(pos: Pos2, img: Rect, side: bool, full: Rect) -> Vec2 {
    let mut p = pos;
    if side && pos.x > full.center().x {
        p.x -= full.width() / 2.0;
    }
    ((p - img.min) / img.size()).clamp(Vec2::ZERO, Vec2::splat(1.0))
}

/// Runs `add` in a child placed at `rect`, without reserving space in the parent.
pub(crate) fn child<R>(ui: &mut Ui, rect: Rect, layout: Layout, add: impl FnOnce(&mut Ui) -> R) -> R {
    let mut c = ui.new_child(egui::UiBuilder::new().max_rect(rect).layout(layout));
    add(&mut c)
}

fn fit_size(size: (usize, usize), max_w: usize, max_h: usize) -> (usize, usize) {
    let k = (max_w as f64 / size.0 as f64).min(max_h as f64 / size.1 as f64).min(1.0);
    let even = |v: usize| ((v as f64 * k / 2.0).round() as usize * 2).max(2);
    (even(size.0), even(size.1))
}

fn set_tex(ctx: &egui::Context, slot: &mut Option<Tex>, name: &str, im: &RgbaImage, key: String) {
    let image = im.to_color_image();
    match slot {
        Some(t) if t.handle.size() == [im.width, im.height] => {
            t.handle.set(image, TextureOptions::LINEAR);
            t.key = key;
        }
        _ => *slot = Some(Tex { handle: ctx.load_texture(name, image, TextureOptions::LINEAR), key }),
    }
}

pub(crate) fn reveal(path: &Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer").arg(format!("/select,{}", path.display())).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg("-R").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        if let Some(dir) = path.parent() {
            let _ = std::process::Command::new("xdg-open").arg(dir).spawn();
        }
    }
}

pub(crate) fn open_path(path: &Path) {
    #[cfg(windows)]
    {
        let _ = std::process::Command::new("explorer").arg(path).spawn();
    }
    #[cfg(target_os = "macos")]
    {
        let _ = std::process::Command::new("open").arg(path).spawn();
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        let _ = std::process::Command::new("xdg-open").arg(path).spawn();
    }
}

impl eframe::App for PrismApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        // Maximize on the first frame: the creation-time flag is ignored for hidden windows.
        if !self.maximized_once {
            self.maximized_once = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Maximized(true));
        }
        self.poll(&ctx);
        if self.dialog.is_none() {
            self.shortcuts(&ctx);
        }
        self.update_playback(&ctx);
        self.window_edges(&ctx);
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().map(|f| f.path().to_path_buf()).collect());
        for p in dropped {
            self.add_source(p, true, &ctx);
        }
        egui::Panel::top("menu").exact_size(32.0).show_separator_line(false).frame(egui::Frame::new().fill(Color32::from_rgb(0x2B, 0x2B, 0x2B))).show(ui, |ui| self.menu_row(ui));
        egui::Panel::top("header").exact_size(48.0).frame(egui::Frame::new().fill(PANEL)).show(ui, |ui| self.header(ui));
        let has_src = self.selected_source().is_some();
        if self.prefs.show_panel && !self.full_preview {
            egui::Panel::right("right").exact_size(RIGHT_W).resizable(false).frame(egui::Frame::new().fill(PANEL)).show(ui, |ui| self.right_pane(ui));
        }
        if !self.full_preview {
            egui::Panel::bottom("bottom").exact_size(49.0).frame(egui::Frame::new().fill(PANEL)).show(ui, |ui| self.bottom_bar(ui));
        }
        if has_src {
            egui::Panel::bottom("transport").exact_size(48.0 + 104.0).frame(egui::Frame::new().fill(PANEL)).show(ui, |ui| self.transport(ui));
        }
        let mut viewer_px = (1280, 720);
        if self.full_preview {
            self.bottom_top = ctx.content_rect().bottom();
        }
        egui::CentralPanel::default().frame(egui::Frame::new().fill(VIEWER_BG)).show(ui, |ui| {
            let full = ui.max_rect();
            self.central_rect = full;
            let mut top = full.top();
            if has_src && !self.full_preview {
                let tabs_h = 34.0;
                let crumb_h = 30.0;
                let contents = self.contents();
                if self.prefs.view == ViewMode::Single || self.cropping.is_some() {
                    self.tab_strip(ui, Rect::from_min_size(full.min, vec2(full.width(), tabs_h)), 0);
                    self.breadcrumb(ui, Rect::from_min_size(pos2(full.left(), full.top() + tabs_h), vec2(full.width(), crumb_h)), contents[1]);
                } else {
                    let half = full.width() / 2.0;
                    self.tab_strip(ui, Rect::from_min_size(full.min, vec2(half, tabs_h)), 0);
                    self.tab_strip(ui, Rect::from_min_size(pos2(full.left() + half, full.top()), vec2(half, tabs_h)), 1);
                    ui.painter().rect_filled(Rect::from_min_size(pos2(full.left() + half, full.top()), vec2(1.0, tabs_h + crumb_h)), 0.0, BG);
                    self.breadcrumb(ui, Rect::from_min_size(pos2(full.left(), full.top() + tabs_h), vec2(half, crumb_h)), contents[0]);
                    self.breadcrumb(ui, Rect::from_min_size(pos2(full.left() + half + 1.0, full.top() + tabs_h), vec2(half - 1.0, crumb_h)), contents[1]);
                }
                top += tabs_h + crumb_h;
            }
            let rect = Rect::from_min_max(pos2(full.left(), top), full.max);
            let ppp = ui.ctx().pixels_per_point();
            viewer_px = ((rect.width() * ppp) as usize, (rect.height() * ppp) as usize);
            self.viewer(ui, rect);
        });
        self.popovers(&ctx);
        self.dialogs(&ctx);
        self.toast(&ctx);
        self.schedule(&ctx, viewer_px);
        if self.preview_job.is_some() || !self.probes.is_empty() || self.slots.iter().any(|s| s.pending.is_some()) || self.grab.is_some() || self.queue.as_ref().is_some_and(|q| !q.is_idle()) || self.fetching.is_some() {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
    }

    fn on_exit(&mut self) {
        let _ = self.project.save(&self.project_dir);
        let _ = self.prefs.save();
        if let Some(job) = &self.preview_job {
            job.control.cancel();
        }
    }
}
