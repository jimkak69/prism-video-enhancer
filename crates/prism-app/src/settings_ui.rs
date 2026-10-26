//! The right pane: "Adjustments" (one section per filter, in processing-panel order) and
//! "Codec settings".

use std::collections::HashSet;

use egui::{Align2, Color32, Id, Rect, Sense, Stroke, StrokeKind, TextureId, Ui, pos2, vec2};
use prism_core::encoders::{self, Vendor};
use prism_core::models::{Catalog, Implementation, ModelInfo, ModelKind};
use prism_core::settings::*;
use prism_core::{JobSettings, VideoInfo};

use crate::icons::{self, Icon};
use crate::theme::{self, ACCENT, LABEL, MUTED, RAISED, TEXT, WARNING};
use crate::widgets::{self, combo, field_label, full, row, section_header, segmented, slider};

/// Requests the pane cannot fulfil itself.
pub enum Action {
    FetchModel(String),
    BrowseLut,
    BrowseOutputDir,
    /// Estimate parameters on the current frame (`true` for the second enhancement).
    Estimate(bool),
    /// Install a local model file as the given catalog id.
    ImportModel(String),
}

pub struct PaneContext<'a> {
    pub catalog: &'a Catalog,
    pub encoders: &'a HashSet<String>,
    pub info: Option<&'a VideoInfo>,
    pub fetching: Option<&'a str>,
    pub estimating: bool,
    /// Current source frame, used as card art in the model picker.
    pub frame: Option<(TextureId, egui::Vec2)>,
    pub list_view: &'a mut bool,
}

fn expanded_state(ui: &Ui, key: &str, default: bool) -> bool {
    ui.data(|d| d.get_temp::<bool>(Id::unique(("pane-section", key)))).unwrap_or(default)
}

fn set_expanded(ui: &Ui, key: &str, v: bool) {
    ui.data_mut(|d| d.insert_temp(Id::unique(("pane-section", key)), v));
}

/// A filter section: header plus body. `always` is shown even when collapsed (the controls the
/// reference keeps visible, such as output resolution and frame rate).
fn section(ui: &mut Ui, key: &str, title: &str, tooltip: (&str, &str), active: &mut bool, always: impl FnOnce(&mut Ui), body: impl FnOnce(&mut Ui)) {
    let mut expanded = expanded_state(ui, key, *active);
    section_header(ui, title, tooltip, &mut expanded, Some(active), false);
    set_expanded(ui, key, expanded);
    always(ui);
    if expanded {
        ui.add_enabled_ui(*active, |ui| body(ui));
    }
    ui.add_space(10.0);
    widgets::divider(ui);
}

fn gradient(p: &egui::Painter, rect: Rect, top: Color32, bottom: Color32) {
    let mut mesh = egui::Mesh::default();
    mesh.colored_vertex(rect.left_top(), top);
    mesh.colored_vertex(rect.right_top(), top);
    mesh.colored_vertex(rect.left_bottom(), bottom);
    mesh.colored_vertex(rect.right_bottom(), bottom);
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(1, 3, 2);
    p.add(mesh);
}

fn card_hue(i: usize) -> Color32 {
    const HUES: [Color32; 6] = [
        Color32::from_rgb(0x2E, 0xCC, 0x71),
        Color32::from_rgb(0x3D, 0x8B, 0xFF),
        Color32::from_rgb(0xFF, 0x9F, 0x43),
        Color32::from_rgb(0xE0, 0x56, 0xFD),
        Color32::from_rgb(0x00, 0xC2, 0xD1),
        Color32::from_rgb(0xFF, 0x5E, 0x7E),
    ];
    HUES[i % HUES.len()]
}

fn wrapped(p: &egui::Painter, text: &str, font: egui::FontId, color: Color32, width: f32, italics: bool) -> std::sync::Arc<egui::Galley> {
    let mut job = egui::text::LayoutJob::default();
    job.wrap.max_width = width;
    job.wrap.max_rows = 3;
    job.append(text, 0.0, egui::TextFormat { font_id: font, color, italics, ..Default::default() });
    p.layout_job(job)
}

/// One model card: frame art tinted per model, name and description at the bottom.
fn paint_card(p: &egui::Painter, rect: Rect, m: &ModelInfo, index: usize, selected: bool, installed: bool, frame: Option<(TextureId, egui::Vec2)>) {
    let hue = card_hue(index);
    match frame {
        Some((tex, size)) => {
            // Centre crop of the frame matching the card's aspect.
            let (fa, ca) = (size.x / size.y, rect.width() / rect.height());
            let uv = if fa > ca {
                let w = ca / fa;
                Rect::from_min_max(pos2(0.5 - w / 2.0, 0.0), pos2(0.5 + w / 2.0, 1.0))
            } else {
                let h = fa / ca;
                Rect::from_min_max(pos2(0.0, 0.5 - h / 2.0), pos2(1.0, 0.5 + h / 2.0))
            };
            p.image(tex, rect, uv, Color32::WHITE);
            p.rect_filled(rect, 4.0, hue.gamma_multiply(0.22));
        }
        None => gradient(p, rect, hue.gamma_multiply(0.55), theme::RAISED),
    }
    let shade = Rect::from_min_max(pos2(rect.left(), rect.top() + rect.height() * 0.35), rect.max);
    gradient(p, shade, Color32::TRANSPARENT, Color32::from_black_alpha(225));
    let name = p.layout_no_wrap(m.name.clone(), theme::semibold(13.0), Color32::WHITE);
    let desc = wrapped(p, &m.description, theme::body(11.0), Color32::from_rgb(0xD0, 0xD0, 0xD0), rect.width() - 16.0, true);
    let desc_h = desc.size().y.min(44.0);
    let name_pos = pos2(rect.left() + 8.0, rect.bottom() - 8.0 - desc_h - name.size().y - 2.0);
    p.galley(name_pos, name, Color32::WHITE);
    p.with_clip_rect(Rect::from_min_max(pos2(rect.left(), rect.bottom() - 8.0 - desc_h), rect.max)).galley(pos2(rect.left() + 8.0, rect.bottom() - 8.0 - desc_h), desc, Color32::WHITE);
    if !installed {
        let chip = Rect::from_min_size(rect.left_top() + vec2(6.0, 6.0), vec2(20.0, 20.0));
        p.rect_filled(chip, 4.0, Color32::from_black_alpha(170));
        icons::paint_at(p, chip.center(), 12.0, Icon::Download, Color32::WHITE);
    }
    if selected {
        p.rect_stroke(rect, 4.0, Stroke::new(2.0, ACCENT), StrokeKind::Inside);
        let c = rect.right_top() + vec2(-14.0, 14.0);
        p.circle_filled(c, 8.0, ACCENT);
        icons::paint_at(p, c, 10.0, Icon::Check, theme::ON_ACCENT);
    } else {
        p.rect_stroke(rect, 4.0, Stroke::new(1.0, Color32::from_white_alpha(18)), StrokeKind::Inside);
    }
}

/// "AI model" picker: a carousel of cards (or a list), with count, dots and arrows.
fn model_picker(ui: &mut Ui, key: &str, current: &mut String, kind: ModelKind, cx: &mut PaneContext, actions: &mut Vec<Action>) {
    let models: Vec<&ModelInfo> = cx.catalog.of_kind(kind).collect();
    if models.is_empty() {
        return;
    }
    let idx = models.iter().position(|m| m.id == *current).unwrap_or(0);
    // Header: label + view toggles.
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(widgets::LEFT);
        ui.label(egui::RichText::new("AI model").color(LABEL));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.add_space(widgets::RIGHT - 4.0);
            ui.spacing_mut().item_spacing.x = 2.0;
            if widgets::small_icon(ui, Icon::List, if *cx.list_view { TEXT } else { MUTED }).on_hover_text("List").clicked() {
                *cx.list_view = true;
            }
            if widgets::small_icon(ui, Icon::Carousel, if *cx.list_view { MUTED } else { TEXT }).on_hover_text("Cards").clicked() {
                *cx.list_view = false;
            }
        });
    });
    let dir = &cx.catalog.dir;
    if *cx.list_view {
        full(ui, |ui| {
            for (i, m) in models.iter().enumerate() {
                let w = ui.available_width();
                let (rect, resp) = ui.allocate_exact_size(vec2(w, 42.0), Sense::click());
                let selected = i == idx;
                let p = ui.painter();
                p.rect_filled(rect, 4.0, if selected { theme::HOVER } else if resp.hovered() { RAISED } else { Color32::TRANSPARENT });
                if selected {
                    p.rect_filled(Rect::from_min_size(rect.min, vec2(3.0, rect.height())), 1.0, ACCENT);
                }
                p.text(rect.left_top() + vec2(12.0, 5.0), Align2::LEFT_TOP, &m.name, theme::semibold(13.0), TEXT);
                let desc = wrapped(p, &m.description, theme::body(11.0), MUTED, w - 24.0, false);
                p.with_clip_rect(rect).galley(rect.left_top() + vec2(12.0, 23.0), desc, MUTED);
                if !m.is_available(dir) {
                    icons::paint_at(p, pos2(rect.right() - 14.0, rect.top() + 12.0), 12.0, Icon::Download, MUTED);
                }
                if resp.clicked() {
                    *current = m.id.clone();
                }
            }
        });
    } else {
        let w = ui.available_width();
        let (area, _) = ui.allocate_exact_size(vec2(w, 154.0), Sense::hover());
        let card = vec2(150.0, 146.0);
        let centre = area.center();
        let painter = ui.painter_at(Rect::from_min_max(pos2(area.left() + 4.0, area.top()), pos2(area.right() - 4.0, area.bottom())));
        let step = card.x + 10.0;
        let n = models.len() as isize;
        for off in -2isize..=2 {
            if n < 2 && off != 0 {
                continue;
            }
            let i = ((idx as isize + off).rem_euclid(n)) as usize;
            if off != 0 && n <= 2 && off.abs() > 1 {
                continue;
            }
            let r = Rect::from_center_size(centre + vec2(off as f32 * step, 0.0), card);
            if !painter.clip_rect().intersects(r) {
                continue;
            }
            let m = models[i];
            paint_card(&painter, r, m, i, off == 0, m.is_available(dir), cx.frame);
            let resp = ui.interact(r.intersect(painter.clip_rect()), ui.scope_id().with((key, "card", off)), Sense::click());
            if off == 0 {
                widgets::tip(resp, &m.name, &m.description);
            } else if resp.clicked() {
                *current = m.id.clone();
            }
        }
        // Count, dots and arrows.
        ui.horizontal(|ui| {
            ui.add_space(widgets::LEFT);
            let w = ui.available_width() - widgets::RIGHT;
            let (r, _) = ui.allocate_exact_size(vec2(w, 22.0), Sense::hover());
            let p = ui.painter();
            p.text(r.left_center(), Align2::LEFT_CENTER, format!("{} models", models.len()), theme::body(11.5), MUTED);
            let dots = models.len().min(12);
            let x0 = r.center().x - (dots as f32 - 1.0) * 4.0;
            for d in 0..dots {
                let sel = d == idx.min(dots - 1);
                p.circle_filled(pos2(x0 + d as f32 * 8.0, r.center().y), if sel { 2.5 } else { 2.0 }, if sel { TEXT } else { theme::DISABLED });
            }
            let mut child = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_max(pos2(r.right() - 48.0, r.top()), r.max)).layout(egui::Layout::left_to_right(egui::Align::Center)));
            child.spacing_mut().item_spacing.x = 2.0;
            if widgets::small_icon(&mut child, Icon::ChevronLeft, TEXT).on_hover_text("Previous model").clicked() {
                *current = models[(idx + models.len() - 1) % models.len()].id.clone();
            }
            if widgets::small_icon(&mut child, Icon::ChevronRight, TEXT).on_hover_text("Next model").clicked() {
                *current = models[(idx + 1) % models.len()].id.clone();
            }
        });
    }
    if let Some(m) = cx.catalog.get(current) {
        if !m.is_available(dir) {
            ui.add_space(4.0);
            full(ui, |ui| ui.horizontal(|ui| {
                ui.set_min_height(28.0);
                icons::paint_at(ui.painter(), ui.cursor().min + vec2(7.0, 14.0), 13.0, Icon::Warning, WARNING);
                ui.add_space(20.0);
                ui.label(egui::RichText::new("Not installed").color(WARNING));
                ui.add_space(4.0);
                if cx.fetching == Some(m.id.as_str()) {
                    ui.spinner();
                    ui.label(egui::RichText::new("Installing…").color(MUTED));
                } else {
                    ui.add_enabled_ui(cx.fetching.is_none(), |ui| {
                        if matches!(&m.implementation, Implementation::Onnx { source: Some(_), .. }) && widgets::button(ui, "Download", vec2(84.0, 26.0)).on_hover_text(source_tip(m)).clicked() {
                            actions.push(Action::FetchModel(m.id.clone()));
                        }
                        if widgets::button(ui, "Import…", vec2(70.0, 26.0)).on_hover_text("Install a model file you already have (.onnx)").clicked() {
                            actions.push(Action::ImportModel(m.id.clone()));
                        }
                    });
                }
            }));
        }
    }
}

/// Source and licence shown on the Download button.
pub fn source_tip(m: &ModelInfo) -> String {
    match &m.implementation {
        Implementation::Onnx { source: Some(src), license, .. } => format!("Downloads {}
License: {license}", src.url()),
        _ => String::new(),
    }
}

fn resolution_label(r: &Resolution, info: Option<&VideoInfo>) -> String {
    let src = info.map(|i| (i.width, i.height));
    match (r, src) {
        (Resolution::Original, Some((w, h))) => format!("{w}x{h} (Original)"),
        (Resolution::Original, None) => "Original".into(),
        (Resolution::Scale(s), Some((w, h))) => format!("{}x{} ({}x Upscale)", (w as f64 * s).round(), (h as f64 * s).round(), s),
        (Resolution::Scale(s), None) => format!("{s}x Upscale"),
        (Resolution::Box { width, height }, _) => match (*width, *height) {
            (640, 480) => "640x480 (SD NTSC)".into(),
            (768, 576) => "768x576 (SD PAL)".into(),
            (1280, 720) => "1280x720 (HD)".into(),
            (1920, 1080) => "1920x1080 (FHD)".into(),
            (3840, 2160) => "3840x2160 (4K)".into(),
            (7680, 4320) => "7680x4320 (8K)".into(),
            (w, h) => format!("{w}x{h} (Custom)"),
        },
        (Resolution::Height(h), _) => format!("{h}p"),
    }
}

const BOXES: [(u32, u32); 6] = [(640, 480), (768, 576), (1280, 720), (1920, 1080), (3840, 2160), (7680, 4320)];

fn output_resolution(ui: &mut Ui, out: &mut OutputSettings, info: Option<&VideoInfo>, enabled: bool) {
    field_label(ui, "Output resolution", ("Output resolution", "Size of the exported video. Upscaling without enhancement uses a plain Lanczos resize."));
    let custom = matches!(out.resolution, Resolution::Box { width, height } if !BOXES.contains(&(width, height)));
    ui.add_enabled_ui(enabled, |ui| {
        full(ui, |ui| {
            combo(ui, "out-res", resolution_label(&out.resolution, info), |ui| {
                let mut opts = vec![Resolution::Original, Resolution::Scale(2.0), Resolution::Scale(4.0)];
                opts.extend(BOXES.iter().map(|(w, h)| Resolution::Box { width: *w, height: *h }));
                for r in opts {
                    let label = resolution_label(&r, info);
                    ui.selectable_value(&mut out.resolution, r, label);
                }
                if ui.selectable_label(custom, "Custom resolution").clicked() {
                    let (w, h) = info.map(|i| (i.width * 2, i.height * 2)).unwrap_or((2560, 1440));
                    out.resolution = Resolution::Box { width: w, height: h };
                }
            });
        });
        if let Resolution::Box { width, height } = &mut out.resolution {
            if custom {
                row(ui, "Size", ("Custom resolution", "Width and height of the output box."), |ui| {
                    ui.add(egui::DragValue::new(width).range(16..=16384).prefix("W  "));
                    ui.add(egui::DragValue::new(height).range(16..=16384).prefix("H  "));
                });
            }
            let src_aspect = info.map(|i| i.width as f64 * i.sar.as_f64() / i.height as f64);
            let box_aspect = *width as f64 / *height as f64;
            if src_aspect.is_none_or(|a| (a - box_aspect).abs() > 0.01) {
                row(ui, "Crop setting", ("Crop setting", "How the picture fits a box with a different aspect ratio."), |ui| {
                    let name = |f: FitMode| match f {
                        FitMode::Pad => "Letterbox/Pillarbox",
                        FitMode::Crop => "Crop to fill",
                        FitMode::Fit => "Fit inside",
                        FitMode::Stretch => "Stretch",
                    };
                    combo(ui, "fit-mode", name(out.fit), |ui| {
                        for f in [FitMode::Pad, FitMode::Crop, FitMode::Fit, FitMode::Stretch] {
                            ui.selectable_value(&mut out.fit, f, name(f));
                        }
                    });
                });
            }
        }
    });
}

fn params_ui(ui: &mut Ui, e: &mut EnhancementSettings, second: bool, cx: &PaneContext, actions: &mut Vec<Action>) {
    let mut enabled = e.param_mode != ParamMode::Auto;
    row(ui, "Parameters", ("AI model parameters", "Off: parameters are estimated from the video. On: adjust them relative to the estimate, or set them manually."), |ui| {
        if widgets::check(ui, &mut enabled, "Enable parameters").changed() {
            e.param_mode = if enabled { ParamMode::RelativeToAuto } else { ParamMode::Auto };
        }
    });
    if !enabled {
        return;
    }
    row(ui, "Mode", ("Parameter mode", "Relative: offsets added to the automatic estimate. Manual: absolute values."), |ui| {
        let w = ui.available_width();
        ui.allocate_ui(vec2(w - if e.param_mode == ParamMode::Manual { 84.0 } else { 0.0 }, 28.0), |ui| {
            segmented(ui, &mut e.param_mode, &[(ParamMode::RelativeToAuto, "Relative"), (ParamMode::Manual, "Manual")]);
        });
        if e.param_mode == ParamMode::Manual {
            let label = if cx.estimating { "Estimating…" } else { "Estimate" };
            if ui.add_enabled_ui(!cx.estimating, |ui| widgets::button(ui, label, vec2(76.0, 28.0))).inner.on_hover_text("Estimate values from the current frame").clicked() {
                actions.push(Action::Estimate(second));
            }
        }
    });
    let lo = if e.param_mode == ParamMode::RelativeToAuto { -100 } else { 0 };
    let rows: [(&str, &str, &mut i32, i32); 6] = [
        ("Fix compression", "Removes block and ringing artifacts before upscaling.", &mut e.compression, lo),
        ("Improve detail", "Boosts local contrast after upscaling.", &mut e.details, lo),
        ("Sharpen", "Contrast-adaptive sharpening of the result.", &mut e.sharpen, lo),
        ("Reduce noise", "Edge-preserving spatial and motion-adaptive temporal denoise.", &mut e.noise, lo),
        ("Dehalo", "Pulls overshoot around edges back into the local range.", &mut e.dehalo, lo),
        ("Anti-alias/deblur", "Negative values soften aliasing, positive values sharpen before upscaling.", &mut e.deblur, -100),
    ];
    for (label, text, v, lo) in rows {
        row(ui, label, (label, text), |ui| slider(ui, v, lo..=100, 0));
    }
}

fn enhancement_body(ui: &mut Ui, s: &mut JobSettings, cx: &mut PaneContext, actions: &mut Vec<Action>) {
    let e = &mut s.enhancement;
    field_label(ui, "Video type", ("Input video type", "Interlaced sources are deinterlaced before enhancement. Interlaced progressive is interlaced content stored as progressive frames."));
    full(ui, |ui| {
        segmented(ui, &mut e.video_type, &[(VideoType::Progressive, "Progressive"), (VideoType::Interlaced, "Interlaced"), (VideoType::InterlacedProgressive, "Interlaced prog.")]);
    });
    if e.video_type == VideoType::Interlaced {
        row(ui, "Field order", ("Field order", "Which field comes first; Auto reads it from the file."), |ui| {
            let name = |f: FieldOrderSetting| match f {
                FieldOrderSetting::Auto => "Auto",
                FieldOrderSetting::TopFirst => "Top field first",
                FieldOrderSetting::BottomFirst => "Bottom field first",
            };
            combo(ui, "field-order", name(e.field_order), |ui| {
                for f in [FieldOrderSetting::Auto, FieldOrderSetting::TopFirst, FieldOrderSetting::BottomFirst] {
                    ui.selectable_value(&mut e.field_order, f, name(f));
                }
            });
        });
    }
    model_picker(ui, "enh1", &mut e.model, ModelKind::Enhancement, cx, actions);
    row(ui, "Add noise", ("Add noise", "Adds fine noise before upscaling to avoid a plastic look."), |ui| slider(ui, &mut e.add_noise, 0..=100, 0));
    let focus = e.focus_fix != FocusFix::Off;
    ui.add_enabled_ui(!focus, |ui| {
        row(ui, "Recover detail", ("Recover detail", "Blends back a plain resample of the source to keep original texture."), |ui| slider(ui, &mut e.recover_original, 0..=100, 20));
    });
    row(ui, "Focus fix", ("Focus fix", "For very soft footage: downscales first, then enhances back to size."), |ui| {
        segmented(ui, &mut e.focus_fix, &[(FocusFix::Off, "Off"), (FocusFix::Normal, "Normal"), (FocusFix::Strong, "Strong")]);
    });
    row(ui, "Grain", ("Activate grain", "Adds film grain after enhancement."), |ui| {
        widgets::check(ui, &mut s.grain.enabled, "Apply grain");
    });
    if s.grain.enabled {
        grain_rows(ui, &mut s.grain);
    }
    params_ui(ui, &mut s.enhancement, false, cx, actions);
    if !s.second_enhancement.enabled {
        ui.add_space(4.0);
        full(ui, |ui| {
            if widgets::icon_text_button(ui, Icon::Plus, "Add second enhancement", 30.0, false).clicked() {
                s.second_enhancement.enabled = true;
                s.second_enhancement.settings.enabled = true;
            }
        });
    }
}

fn grain_rows(ui: &mut Ui, g: &mut GrainSettings) {
    row(ui, "Grain amount", ("Grain amount", "Strength of the added grain."), |ui| slider(ui, &mut g.amount, 0..=100, 0));
    row(ui, "Grain size", ("Grain size", "Size of the grain particles."), |ui| widgets::slider_f64(ui, &mut g.size, 0.0..=5.0, 10.0, 1.0));
}

pub fn filters_tab(ui: &mut Ui, s: &mut JobSettings, cx: &mut PaneContext) -> Vec<Action> {
    let mut actions = Vec::new();
    let info = cx.info;
    // Enhancement (output resolution always visible).
    {
        let mut on = s.enhancement.enabled;
        let mut expanded = expanded_state(ui, "enhancement", on);
        section_header(ui, "Enhancement", ("Enhancement", "Upscaling and restoration with an AI or classic model."), &mut expanded, Some(&mut on), false);
        set_expanded(ui, "enhancement", expanded);
        s.enhancement.enabled = on;
        output_resolution(ui, &mut s.output, info, true);
        if expanded {
            ui.add_enabled_ui(on, |ui| enhancement_body(ui, s, cx, &mut actions));
        }
        ui.add_space(10.0);
        widgets::divider(ui);
    }
    if s.second_enhancement.enabled {
        let mut on = true;
        let mut body_actions = Vec::new();
        section(
            ui,
            "second",
            "Second enhancement",
            ("Second enhancement", "A second model pass, for example a denoiser before an upscaler. Switch off to remove it."),
            &mut on,
            |_| {},
            |ui| {
                let se = &mut s.second_enhancement;
                row(ui, "Intermediate", ("Intermediate resolution", "Size between the first and second enhancement."), |ui| {
                    segmented(ui, &mut se.intermediate, &[(IntermediateScale::Auto, "Auto"), (IntermediateScale::X1, "1x"), (IntermediateScale::X2, "2x"), (IntermediateScale::X4, "4x")]);
                });
                model_picker(ui, "enh2", &mut se.settings.model, ModelKind::Enhancement, cx, &mut body_actions);
                row(ui, "Add noise", ("Add noise", ""), |ui| slider(ui, &mut se.settings.add_noise, 0..=100, 0));
                row(ui, "Recover detail", ("Recover original detail", ""), |ui| slider(ui, &mut se.settings.recover_original, 0..=100, 20));
                params_ui(ui, &mut se.settings, true, cx, &mut body_actions);
            },
        );
        actions.extend(body_actions);
        s.second_enhancement.enabled = on;
        s.second_enhancement.settings.enabled = on;
    }
    // Frame interpolation (frame rate and slow motion always visible).
    {
        let fi = &mut s.interpolation;
        let mut expanded = expanded_state(ui, "interp", fi.enabled);
        section_header(ui, "Frame interpolation", ("Frame interpolation", "Creates new frames for higher frame rates and slow motion."), &mut expanded, Some(&mut fi.enabled), false);
        let orig = info.map(|i| i.fps.as_f64());
        let fps_label = |f: Option<f64>| match (f, orig) {
            (None, Some(o)) => format!("{} FPS (Original)", fmt_fps(o)),
            (None, None) => "Original".to_string(),
            (Some(f), _) => format!("{} FPS", fmt_fps(f)),
        };
        let before = (fi.output_fps, fi.slowmo);
        field_label(ui, "Frame rate", ("Output frame rate", "Frame rate of the exported video."));
        full(ui, |ui| {
            combo(ui, "fi-fps", fps_label(fi.output_fps), |ui| {
                ui.selectable_value(&mut fi.output_fps, None, fps_label(None));
                for f in [23.976, 24.0, 25.0, 29.97, 30.0, 48.0, 50.0, 59.94, 60.0, 90.0, 120.0] {
                    ui.selectable_value(&mut fi.output_fps, Some(f), format!("{} FPS", fmt_fps(f)));
                }
            });
        });
        field_label(ui, "Slow motion", ("Slow motion", "Stretches time; each source second lasts this many seconds."));
        full(ui, |ui| {
            let text = if fi.slowmo <= 1.0 { "None".to_string() } else { format!("{}x", fi.slowmo) };
            combo(ui, "fi-slow", text, |ui| {
                for f in [1.0, 2.0, 4.0, 8.0, 16.0] {
                    ui.selectable_value(&mut fi.slowmo, f, if f == 1.0 { "None".to_string() } else { format!("{f}x") });
                }
            });
        });
        // Picking a new rate or slow motion switches the filter on.
        if (fi.output_fps, fi.slowmo) != before && (fi.output_fps.is_some() || fi.slowmo > 1.0) {
            fi.enabled = true;
            expanded = true;
        }
        set_expanded(ui, "interp", expanded);
        if expanded {
            ui.add_enabled_ui(fi.enabled, |ui| {
                model_picker(ui, "fi", &mut fi.model, ModelKind::Interpolation, cx, &mut actions);
                row(ui, "Duplicate frames", ("Replace duplicate frames", "Replaces repeated frames with interpolated ones."), |ui| {
                    widgets::check(ui, &mut fi.replace_duplicates, "Replace");
                });
                ui.add_enabled_ui(fi.replace_duplicates, |ui| {
                    row(ui, "Sensitivity", ("Sensitivity", "How different frames may be and still count as duplicates."), |ui| widgets::slider_f64(ui, &mut fi.duplicate_threshold, 0.0..=100.0, 1.0, 10.0));
                });
                row(ui, "Scene cuts", ("Scene detection", "Repeats frames across hard cuts instead of morphing between shots."), |ui| {
                    widgets::check(ui, &mut fi.scene_detection, "Detect");
                });
            });
        }
        ui.add_space(10.0);
        widgets::divider(ui);
    }
    let mut on = s.stabilization.enabled;
    section(
        ui,
        "stab",
        "Stabilization",
        ("Stabilization", "Smooths camera shake using a separate analysis pass."),
        &mut on,
        |_| {},
        |ui| {
            let st = &mut s.stabilization;
            row(ui, "Method", ("Method", "Auto crop removes moving borders; full frame keeps the whole picture."), |ui| {
                segmented(ui, &mut st.mode, &[(StabilizationMode::AutoCrop, "Auto crop"), (StabilizationMode::FullFrame, "Full frame")]);
            });
            row(ui, "Strength", ("Strength", "How far the camera path is smoothed."), |ui| slider(ui, &mut st.smoothness, 0..=100, 50));
            row(ui, "Jittery motion", ("Fix jittery motion", "Raises motion-detection sensitivity for small, fast shake."), |ui| {
                let mut j = st.reduce_jitter > 0;
                if widgets::check(ui, &mut j, "Apply correction").changed() {
                    st.reduce_jitter = if j { 3 } else { 0 };
                }
            });
            row(ui, "Rolling shutter", ("Rolling shutter correction", "Reduces skew from rolling-shutter cameras."), |ui| {
                widgets::check(ui, &mut st.rolling_shutter, "Apply correction");
            });
            note(ui, "Stabilization won't live render. Click Render to preview the stabilized result.");
        },
    );
    s.stabilization.enabled = on;
    let mut on = s.motion_deblur.enabled;
    let mut body_actions = Vec::new();
    section(
        ui,
        "deblur",
        "Motion deblur",
        ("Motion deblur", "Reduces blur from camera or subject motion."),
        &mut on,
        |_| {},
        |ui| {
            let mut model = cx.catalog.of_kind(ModelKind::MotionDeblur).next().map(|m| m.id.clone()).unwrap_or_default();
            model_picker(ui, "deblur", &mut model, ModelKind::MotionDeblur, cx, &mut body_actions);
            row(ui, "Strength", ("Strength", ""), |ui| slider(ui, &mut s.motion_deblur.strength, 0..=100, 50));
        },
    );
    actions.extend(body_actions);
    s.motion_deblur.enabled = on;
    let mut on = s.hdr.enabled;
    section(
        ui,
        "hdr",
        "SDR to HDR",
        ("SDR to HDR", "Expands standard-range video to HDR10 (PQ) or HLG."),
        &mut on,
        |_| {},
        |ui| {
            let h = &mut s.hdr;
            row(ui, "Transfer", ("Transfer function", "PQ for HDR10 displays; HLG for broadcast."), |ui| {
                segmented(ui, &mut h.transfer, &[(HdrTransfer::Pq, "PQ"), (HdrTransfer::Hlg, "HLG")]);
            });
            row(ui, "Exposure", ("Adjust exposure", ""), |ui| slider(ui, &mut h.brightness, -100..=100, 0));
            row(ui, "Saturation", ("Boost saturation", ""), |ui| slider(ui, &mut h.saturation, -100..=100, 0));
            row(ui, "Highlight thresh.", ("Highlight threshold", "Signal level where highlight expansion starts."), |ui| widgets::slider_f64(ui, &mut h.sdr_ip, 0.45..=0.85, 100.0, 0.58));
            row(ui, "Peak brightness", ("Peak brightness", "Mastering peak in nits."), |ui| {
                combo(ui, "hdr-peak", format!("{} nits", h.peak_nits), |ui| {
                    for p in [600.0, 1000.0, 2000.0, 4000.0] {
                        ui.selectable_value(&mut h.peak_nits, p, format!("{p} nits"));
                    }
                });
            });
        },
    );
    s.hdr.enabled = on;
    let mut on = s.grain.enabled;
    section(ui, "grain", "Grain", ("Grain", "Adds film grain to the final picture."), &mut on, |_| {}, |ui| grain_rows(ui, &mut s.grain));
    s.grain.enabled = on;
    actions
}

fn note(ui: &mut Ui, text: &str) {
    ui.add_space(2.0);
    full(ui, |ui| {
        ui.label(egui::RichText::new(text).color(MUTED).small());
    });
}

fn fmt_fps(f: f64) -> String {
    let s = format!("{f:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn vendor_name(v: Vendor) -> &'static str {
    match v {
        Vendor::Software => "CPU (software)",
        Vendor::Nvidia => "NVIDIA",
        Vendor::Amd => "AMD",
        Vendor::Intel => "Intel",
    }
}

pub fn export_tab(ui: &mut Ui, s: &mut JobSettings, cx: &PaneContext) -> Vec<Action> {
    let mut actions = Vec::new();
    let x = &mut s.export;
    ui.add_space(6.0);
    full(ui, |ui| {
        ui.label(egui::RichText::new("Set your preview and export settings here").color(MUTED));
    });
    ui.add_space(4.0);
    let usable: Vec<&'static encoders::VideoEncoder> = encoders::video_encoders().iter().filter(|e| cx.encoders.contains(e.codec) && e.family != "Null").collect();
    let current = encoders::find(&x.encoder).filter(|e| usable.iter().any(|u| u.id == e.id)).or_else(|| usable.first().copied());
    let Some(current) = current else {
        note(ui, "FFmpeg reports no usable encoders.");
        return actions;
    };
    let mut seq = current.image_sequence;
    row(ui, "Output type", ("Output type", "A video file or one image per frame."), |ui| {
        if segmented(ui, &mut seq, &[(false, "Video"), (true, "Image seq.")]) {
            if let Some(e) = usable.iter().find(|e| e.image_sequence == seq) {
                x.encoder = e.id.to_string();
                x.container.clear();
            }
        }
    });
    let current = encoders::find(&x.encoder).unwrap_or(current);
    let pool: Vec<&&encoders::VideoEncoder> = usable.iter().filter(|e| e.image_sequence == current.image_sequence).collect();
    let mut families: Vec<&str> = Vec::new();
    for e in &pool {
        if !families.contains(&e.family) {
            families.push(e.family);
        }
    }
    let family_label = if current.image_sequence { "File type" } else { "Codec" };
    row(ui, family_label, (family_label, ""), |ui| {
        combo(ui, "codec-family", current.family, |ui| {
            for f in &families {
                if ui.selectable_label(*f == current.family, *f).clicked() {
                    if let Some(e) = pool.iter().find(|e| e.family == *f && e.vendor == current.vendor).or_else(|| pool.iter().find(|e| e.family == *f)) {
                        x.encoder = e.id.to_string();
                        x.container.clear();
                    }
                }
            }
        });
    });
    let current = encoders::find(&x.encoder).unwrap_or(current);
    if !current.image_sequence {
        let vendors: Vec<Vendor> = {
            let mut v = Vec::new();
            for e in pool.iter().filter(|e| e.family == current.family) {
                if !v.contains(&e.vendor) {
                    v.push(e.vendor);
                }
            }
            v
        };
        row(ui, "Encoder", ("Encoder GPU", "Hardware encoders are much faster; software encoders compress better."), |ui| {
            combo(ui, "codec-vendor", vendor_name(current.vendor), |ui| {
                for v in &vendors {
                    if ui.selectable_label(*v == current.vendor, vendor_name(*v)).clicked() {
                        if let Some(e) = pool.iter().find(|e| e.family == current.family && e.vendor == *v) {
                            x.encoder = e.id.to_string();
                        }
                    }
                }
            });
        });
    }
    let current = encoders::find(&x.encoder).unwrap_or(current);
    let variants: Vec<&&&encoders::VideoEncoder> = pool.iter().filter(|e| e.family == current.family && e.vendor == current.vendor).collect();
    let profile_label = if current.image_sequence { "Bit depth" } else { "Profile" };
    row(ui, profile_label, (profile_label, ""), |ui| {
        combo(ui, "codec-profile", current.profile, |ui| {
            for e in &variants {
                if ui.selectable_label(e.id == current.id, e.profile).clicked() {
                    x.encoder = e.id.to_string();
                }
            }
        });
    });
    let current = encoders::find(&x.encoder).unwrap_or(current);
    if !matches!(current.rate, encoders::RateSpec::Fixed) {
        let mut constant = matches!(x.rate, RateControl::Bitrate(_));
        if current.supports_bitrate() {
            row(ui, "Bitrate", ("Video bitrate", "Dynamic targets a quality level; constant targets a bitrate."), |ui| {
                if segmented(ui, &mut constant, &[(false, "Dynamic"), (true, "Constant")]) {
                    x.rate = if constant { RateControl::Bitrate(0.0) } else { RateControl::Quality(Quality::High) };
                }
            });
        } else if constant {
            x.rate = RateControl::Quality(Quality::High);
        }
        match &mut x.rate {
            RateControl::Quality(q) => {
                row(ui, "Quality level", ("Dynamic compression quality level", ""), |ui| {
                    segmented(ui, q, &[(Quality::Low, "Low"), (Quality::Medium, "Medium"), (Quality::High, "High")]);
                });
            }
            RateControl::Bitrate(b) => {
                row(ui, "Target bitrate", ("Target bitrate", "Mbit/s. 0 picks a bitrate from the output size and frame rate."), |ui| {
                    ui.add(egui::DragValue::new(b).range(0.0..=1000.0).speed(0.5).suffix(" Mbit/s"));
                    if *b == 0.0 {
                        ui.label(egui::RichText::new("Auto").color(MUTED));
                    }
                });
            }
        }
    }
    if !current.image_sequence {
        let has_audio = cx.info.is_none_or(|i| !i.audio.is_empty());
        ui.add_enabled_ui(has_audio, |ui| {
            let r = row(ui, "Audio mode", ("Audio mode", if has_audio { "Copy keeps the original audio; convert re-encodes it." } else { "This option is disabled because the input video has no audio." }), |ui| {
                let name = |m: AudioMode| match m {
                    AudioMode::Copy => "Copy",
                    AudioMode::Convert => "Convert",
                    AudioMode::None => "No audio",
                };
                combo(ui, "audio-mode", name(x.audio.mode), |ui| {
                    for m in [AudioMode::Copy, AudioMode::Convert, AudioMode::None] {
                        ui.selectable_value(&mut x.audio.mode, m, name(m));
                    }
                });
            });
            let _ = r;
            if x.audio.mode == AudioMode::Convert {
                row(ui, "Audio codec", ("Audio codec", ""), |ui| {
                    combo(ui, "audio-codec", encoders::audio_codec(&x.audio.codec).map(|c| c.name).unwrap_or("AAC"), |ui| {
                        for c in encoders::audio_codecs() {
                            ui.selectable_value(&mut x.audio.codec, c.id.to_string(), c.name);
                        }
                    });
                });
                if let Some((lo, hi, def)) = encoders::audio_codec(&x.audio.codec).and_then(|c| c.bitrate) {
                    if x.audio.bitrate_kbps == 0 {
                        x.audio.bitrate_kbps = def;
                    }
                    row(ui, "Audio bitrate", ("Audio bitrate", ""), |ui| {
                        ui.add(egui::DragValue::new(&mut x.audio.bitrate_kbps).range(lo..=hi).suffix(" kbit/s"));
                    });
                }
            }
        });
        row(ui, "Container", ("Container", ""), |ui| {
            let cur = if x.container.is_empty() { current.default_container().to_string() } else { x.container.clone() };
            combo(ui, "container", cur.to_uppercase(), |ui| {
                for c in current.containers {
                    if ui.selectable_label(cur == *c, c.to_uppercase()).clicked() {
                        x.container = c.to_string();
                    }
                }
            });
        });
        row(ui, "Subtitles", ("Subtitles", "Copies subtitle streams when the container supports them."), |ui| {
            widgets::check(ui, &mut x.subtitles, "Copy subtitles");
        });
    }
    ui.add_space(6.0);
    widgets::divider(ui);
    ui.add_space(6.0);
    row(ui, "Apply LUT", ("Apply LUT", "Applies a .cube (or .3dl) file to the output colour."), |ui| {
        let text = x.lut.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "None".into());
        ui.add(egui::Label::new(egui::RichText::new(text).color(LABEL)).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if x.lut.is_some() && widgets::small_icon(ui, Icon::Close, MUTED).on_hover_text("Remove").clicked() {
                x.lut = None;
            }
            if widgets::small_icon(ui, Icon::Folder, TEXT).on_hover_text("Browse…").clicked() {
                actions.push(Action::BrowseLut);
            }
        });
    });
    row(ui, "Output folder", ("Output folder", "Where exports are written. Empty writes next to the input."), |ui| {
        let text = x.output_dir.as_ref().map(|d| d.display().to_string()).unwrap_or_else(|| "Same as input".into());
        ui.add(egui::Label::new(egui::RichText::new(text).color(LABEL)).truncate());
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if x.output_dir.is_some() && widgets::small_icon(ui, Icon::Close, MUTED).on_hover_text("Use the input's folder").clicked() {
                x.output_dir = None;
            }
            if widgets::small_icon(ui, Icon::Folder, TEXT).on_hover_text("Browse…").clicked() {
                actions.push(Action::BrowseOutputDir);
            }
        });
    });
    row(ui, "Existing files", ("Existing files", "Off adds a number to the name instead of replacing a file."), |ui| {
        widgets::check(ui, &mut x.overwrite, "Overwrite");
    });
    actions
}

/// A placeholder line when there is nothing to adjust.
pub fn empty(ui: &mut Ui, text: &str) {
    ui.add_space(24.0);
    full(ui, |ui| {
        ui.label(egui::RichText::new(text).color(MUTED));
    });
}
