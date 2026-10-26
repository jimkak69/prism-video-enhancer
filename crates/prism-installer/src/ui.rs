//! The setup window: options, progress and result, for installing or uninstalling.

use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

use egui::{Color32, CornerRadius, RichText, Stroke, Vec2, vec2};

use crate::setup::{self, APP_NAME, Options, VERSION};

#[path = "../../prism-app/src/appicon.rs"]
#[allow(dead_code)]
mod appicon;

const BG: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);
const RAISED: Color32 = Color32::from_rgb(0x2B, 0x2B, 0x2B);
const TEXT: Color32 = Color32::from_rgb(0xF2, 0xF2, 0xF2);
const MUTED: Color32 = Color32::from_rgb(0x97, 0x97, 0x97);
const ACCENT: Color32 = Color32::from_rgb(0x2E, 0xCC, 0x71);
const ON_ACCENT: Color32 = Color32::from_rgb(0x08, 0x1A, 0x0F);
const WARNING: Color32 = Color32::from_rgb(0xFF, 0xCE, 0x21);
const ERROR: Color32 = Color32::from_rgb(0xFF, 0x4D, 0x6A);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    Install,
    Uninstall,
}

enum Page {
    Options,
    Working,
    Done(Option<PathBuf>),
    Failed(String),
}

struct SetupApp {
    mode: Mode,
    page: Page,
    opts: Options,
    launch: bool,
    remove_data: bool,
    ffmpeg: Option<PathBuf>,
    existing: Option<String>,
    progress: Arc<Mutex<(u64, u64)>>,
    rx: Option<Receiver<Result<Option<PathBuf>, String>>>,
}

pub fn run(mode: Mode) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title(if mode == Mode::Install { "Prism Video Enhancer Setup" } else { "Uninstall Prism Video Enhancer" })
            .with_icon(appicon::icon_data(64))
            .with_inner_size([600.0, 430.0])
            .with_resizable(false),
        centered: true,
        ..Default::default()
    };
    eframe::run_native(
        "Prism Setup",
        options,
        Box::new(move |cc| {
            style(&cc.egui_ctx);
            let opts = Options::default();
            let existing = setup::existing_version(&opts.dir);
            Ok(Box::new(SetupApp { mode, page: Page::Options, opts, launch: true, remove_data: false, ffmpeg: setup::find_ffmpeg(), existing, progress: Arc::new(Mutex::new((0, 0))), rx: None }))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn style(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    let dir = std::env::var_os("WINDIR").map(PathBuf::from).unwrap_or_else(|| r"C:\Windows".into()).join("Fonts");
    if let Ok(bytes) = std::fs::read(dir.join("segoeui.ttf")) {
        fonts.font_data.insert("segoe".into(), Arc::new(egui::FontData::from_owned(bytes)));
        fonts.families.entry(egui::FontFamily::Proportional).or_default().insert(0, "segoe".into());
    }
    ctx.set_fonts(fonts);
    let mut v = egui::Visuals::dark();
    v.panel_fill = BG;
    v.override_text_color = Some(TEXT);
    v.selection.bg_fill = Color32::from_rgb(0x1E, 0x6B, 0x41);
    for w in [&mut v.widgets.inactive, &mut v.widgets.hovered, &mut v.widgets.active] {
        w.corner_radius = CornerRadius::same(4);
        w.bg_stroke = Stroke::NONE;
    }
    v.widgets.inactive.weak_bg_fill = RAISED;
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.set_theme(egui::Theme::Dark);
    ctx.style_mut_of(egui::Theme::Dark, |s| {
        s.text_styles.insert(egui::TextStyle::Body, egui::FontId::proportional(14.0));
        s.text_styles.insert(egui::TextStyle::Button, egui::FontId::proportional(14.0));
        s.spacing.item_spacing = vec2(8.0, 8.0);
        s.spacing.button_padding = vec2(14.0, 6.0);
    });
}

fn primary(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add_sized([130.0, 34.0], egui::Button::new(RichText::new(text).color(ON_ACCENT).strong()).fill(ACCENT))
}

fn secondary(ui: &mut egui::Ui, text: &str) -> egui::Response {
    ui.add_sized([110.0, 34.0], egui::Button::new(text).fill(RAISED))
}

impl SetupApp {
    fn start(&mut self, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let (mode, opts, remove, progress, ctx2) = (self.mode, self.opts.clone(), self.remove_data, self.progress.clone(), ctx.clone());
        std::thread::spawn(move || {
            let r = match mode {
                Mode::Install => setup::install(&opts, &|done, total| {
                    *progress.lock().unwrap() = (done, total);
                    ctx2.request_repaint();
                })
                .map(Some),
                Mode::Uninstall => setup::uninstall(&setup::installed_dir(), remove).map(|_| None),
            };
            let _ = tx.send(r.map_err(|e| format!("{e:#}")));
            ctx2.request_repaint();
        });
        self.rx = Some(rx);
        self.page = Page::Working;
    }

    fn header(&self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            let (r, _) = ui.allocate_exact_size(Vec2::splat(52.0), egui::Sense::hover());
            let tex = ui.ctx().data_mut(|d| d.get_temp::<egui::TextureHandle>(egui::Id::unique("setup-logo")));
            let tex = tex.unwrap_or_else(|| {
                let t = ui.ctx().load_texture("setup-logo", egui::ColorImage::from_rgba_unmultiplied([128, 128], &appicon::rgba(128)), egui::TextureOptions::LINEAR);
                ui.ctx().data_mut(|d| d.insert_temp(egui::Id::unique("setup-logo"), t.clone()));
                t
            });
            ui.painter().image(tex.id(), r, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
            ui.vertical(|ui| {
                ui.add_space(4.0);
                ui.label(RichText::new(APP_NAME).size(22.0).strong());
                let sub = match self.mode {
                    Mode::Install => format!("Version {VERSION} · installs for this Windows user"),
                    Mode::Uninstall => format!("Uninstall from {}", setup::installed_dir().display()),
                };
                ui.label(RichText::new(sub).color(MUTED));
            });
        });
        ui.add_space(10.0);
        ui.separator();
        ui.add_space(6.0);
    }

    fn options_page(&mut self, ui: &mut egui::Ui) {
        match self.mode {
            Mode::Install => {
                ui.label(RichText::new("Install location").color(MUTED));
                ui.horizontal(|ui| {
                    let mut text = self.opts.dir.display().to_string();
                    if ui.add(egui::TextEdit::singleline(&mut text).desired_width(430.0)).changed() {
                        self.opts.dir = PathBuf::from(text);
                        self.existing = setup::existing_version(&self.opts.dir);
                    }
                    if secondary(ui, "Browse…").clicked() {
                        if let Some(d) = rfd::FileDialog::new().set_directory(&self.opts.dir).pick_folder() {
                            self.opts.dir = d.join(APP_NAME);
                            self.existing = setup::existing_version(&self.opts.dir);
                        }
                    }
                });
                if let Some(v) = &self.existing {
                    ui.label(RichText::new(format!("Version {v} is installed here and will be updated. Settings and models are kept.")).color(MUTED).small());
                }
                ui.add_space(4.0);
                ui.checkbox(&mut self.opts.start_menu, "Start menu shortcut");
                ui.checkbox(&mut self.opts.desktop_shortcut, "Desktop shortcut");
                ui.checkbox(&mut self.opts.add_to_path, "Add the prism command-line tool to PATH");
                ui.add_space(6.0);
                match &self.ffmpeg {
                    Some(p) => {
                        ui.label(RichText::new(format!("FFmpeg found: {}", p.display())).color(MUTED).small());
                    }
                    None => {
                        ui.label(RichText::new("FFmpeg was not found. Prism needs it to read and write video.").color(WARNING));
                        ui.horizontal(|ui| {
                            ui.label(RichText::new("Install it from a terminal:  winget install Gyan.FFmpeg").color(MUTED).small());
                            if ui.small_button("Copy").clicked() {
                                ui.ctx().copy_text("winget install Gyan.FFmpeg".into());
                            }
                        });
                    }
                }
            }
            Mode::Uninstall => {
                ui.label(format!("This removes {APP_NAME}, its shortcuts and its entry in Settings › Apps."));
                ui.add_space(6.0);
                ui.checkbox(&mut self.remove_data, "Also delete settings, projects, presets and downloaded models");
                ui.label(RichText::new(format!("They are in {}", prism_core::paths::data_dir().display())).color(MUTED).small());
            }
        }
    }
}

impl eframe::App for SetupApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(rx) = &self.rx {
            if let Ok(r) = rx.try_recv() {
                self.rx = None;
                self.page = match r {
                    Ok(app) => Page::Done(app),
                    Err(e) => Page::Failed(e),
                };
            }
        }
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(24)).show(ui, |ui| {
            self.header(ui);
            let footer_h = 44.0;
            let body_h = ui.available_height() - footer_h;
            ui.allocate_ui(vec2(ui.available_width(), body_h), |ui| {
                ui.set_min_height(body_h);
                match &self.page {
                    Page::Options => self.options_page(ui),
                    Page::Working => {
                        let (done, total) = *self.progress.lock().unwrap();
                        ui.add_space(20.0);
                        ui.label(if self.mode == Mode::Install { "Installing…" } else { "Removing…" });
                        if self.mode == Mode::Install {
                            let frac = if total > 0 { done as f32 / total as f32 } else { 0.0 };
                            ui.add(egui::ProgressBar::new(frac).fill(ACCENT).desired_width(ui.available_width()).show_percentage());
                        } else {
                            ui.spinner();
                        }
                    }
                    Page::Done(_) => {
                        ui.add_space(20.0);
                        let text = match self.mode {
                            Mode::Install => format!("{APP_NAME} is installed."),
                            Mode::Uninstall => format!("{APP_NAME} was removed."),
                        };
                        ui.label(RichText::new(text).size(17.0).color(ACCENT));
                        if self.mode == Mode::Install {
                            ui.checkbox(&mut self.launch, "Open Prism Video Enhancer now");
                            if self.ffmpeg.is_none() {
                                ui.label(RichText::new("Remember to install FFmpeg:  winget install Gyan.FFmpeg").color(WARNING));
                            }
                        }
                    }
                    Page::Failed(e) => {
                        ui.add_space(12.0);
                        ui.label(RichText::new("Setup could not finish").size(17.0).color(ERROR));
                        egui::ScrollArea::vertical().max_height(150.0).show(ui, |ui| {
                            ui.label(RichText::new(e).color(TEXT));
                        });
                    }
                }
            });
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| match &self.page {
                Page::Options => {
                    let label = match (self.mode, self.existing.is_some()) {
                        (Mode::Install, false) => "Install",
                        (Mode::Install, true) => "Update",
                        (Mode::Uninstall, _) => "Uninstall",
                    };
                    if primary(ui, label).clicked() {
                        self.start(&ctx);
                    }
                    if secondary(ui, "Cancel").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
                Page::Working => {}
                Page::Done(app) => {
                    if primary(ui, "Finish").clicked() {
                        if let (true, Some(app)) = (self.launch && self.mode == Mode::Install, app) {
                            let _ = std::process::Command::new(app).current_dir(app.parent().unwrap_or(&self.opts.dir)).spawn();
                        }
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                }
                Page::Failed(_) => {
                    if secondary(ui, "Close").clicked() {
                        ctx.send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    if primary(ui, "Try again").clicked() {
                        self.page = Page::Options;
                    }
                }
            });
        });
    }
}
