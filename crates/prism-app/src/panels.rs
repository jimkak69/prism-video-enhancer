//! Popovers (sources, export queue, presets, input settings, projects), modal dialogs and the
//! message toast.

use std::path::PathBuf;

use egui::{Align, Align2, Color32, Id, Layout, Order, Pos2, Rect, RichText, ScrollArea, Sense, Stroke, Ui, pos2, vec2};
use prism_core::models::{Implementation, ModelKind};
use prism_core::presets::Preset;
use prism_core::progress::format_duration;
use prism_core::settings::{Backend, ColorOverride, DeinterlaceMode, FieldOrderSetting};
use prism_core::paths;
use prism_engine::queue::JobState;

use crate::app::{Dialog, Popover, PrismApp, REVEAL_LABEL, open_path, preview_len_label, reveal};
use crate::icons::{self, Icon};
use crate::theme::{self, ACCENT, ERROR, HOVER, LABEL, MUTED, PANEL, RAISED, TEXT, WARNING};
use crate::widgets::{self, segmented};

fn frame() -> egui::Frame {
    egui::Frame::new()
        .fill(PANEL)
        .stroke(Stroke::new(1.0, HOVER))
        .corner_radius(6)
        .shadow(egui::epaint::Shadow { offset: [0, 6], blur: 24, spread: 0, color: Color32::from_black_alpha(150) })
        .inner_margin(egui::Margin::same(0))
}

/// Shows a floating panel; returns its rect.
fn popover(ctx: &egui::Context, id: &str, pos: Pos2, pivot: Align2, width: f32, add: impl FnOnce(&mut Ui)) -> Rect {
    egui::Area::new(Id::unique(("popover", id)))
        .order(Order::Foreground)
        .fixed_pos(pos)
        .pivot(pivot)
        .constrain(true)
        .show(ctx, |ui| {
            frame().show(ui, |ui| {
                ui.set_width(width);
                add(ui);
            });
        })
        .response
        .rect
}

fn title_row(ui: &mut Ui, title: &str, right: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        ui.set_height(44.0);
        ui.add_space(16.0);
        ui.label(RichText::new(title).font(theme::semibold(14.0)));
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add_space(10.0);
            right(ui);
        });
    });
    widgets::divider(ui);
}

fn details(info: &prism_core::VideoInfo) -> String {
    format!("{}x{} · {} fps · {} · {}", info.width, info.height, fmt_fps(info.fps.as_f64()), format_duration(info.duration), info.codec)
}

fn fmt_fps(f: f64) -> String {
    let s = format!("{f:.3}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl PrismApp {
    pub(crate) fn popovers(&mut self, ctx: &egui::Context) {
        let Some(which) = self.popover else { return };
        let c = self.central_rect;
        let rect = match which {
            Popover::Sources => popover(ctx, "sources", pos2(c.left() + 8.0, self.bottom_top - 8.0), Align2::LEFT_BOTTOM, 470.0, |ui| self.sources_panel(ui)),
            Popover::Queue => popover(ctx, "queue", pos2(c.right() - 8.0, self.bottom_top - 8.0), Align2::RIGHT_BOTTOM, 580.0, |ui| self.queue_panel(ui)),
            Popover::Presets => popover(ctx, "presets", pos2(self.presets_button.right(), self.presets_button.bottom() + 4.0), Align2::RIGHT_TOP, 330.0, |ui| self.presets_panel(ui)),
            Popover::InputSettings => popover(ctx, "input", pos2(c.right() - 12.0, c.top() + 12.0), Align2::RIGHT_TOP, 360.0, |ui| self.input_panel(ui)),
            Popover::Projects => popover(ctx, "projects", self.crumb_anchor, Align2::LEFT_TOP, 300.0, |ui| self.projects_panel(ui)),
        };
        // Close on a click outside, unless the click landed in another floating layer (a
        // dropdown opened from inside the popover) or this frame's click opened it.
        let outside = ctx.input(|i| i.pointer.any_click() && i.pointer.interact_pos().is_some_and(|p| !rect.contains(p)));
        let on_background = ctx.input(|i| i.pointer.interact_pos()).and_then(|p| ctx.layer_id_at(p)).is_none_or(|l| l.order == Order::Background);
        if outside && on_background && !self.popover_opened && self.dialog.is_none() {
            self.popover = None;
        }
        self.popover_opened = false;
    }

    fn sources_panel(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        let mut import = false;
        title_row(ui, &format!("Sources ({})", self.project.sources.len()), |ui| {
            if widgets::icon_text_button(ui, Icon::Plus, "Import…", 30.0, false).clicked() {
                import = true;
            }
        });
        if import {
            self.import_dialog(&ctx);
        }
        let mut select = None;
        let mut remove = None;
        ScrollArea::vertical().max_height(380.0).auto_shrink([false, true]).show(ui, |ui| {
            ui.add_space(4.0);
            if self.project.sources.is_empty() && self.probes.is_empty() {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.label(RichText::new("No sources yet. Import videos or drop them on the window.").color(MUTED));
                });
            }
            for (i, s) in self.project.sources.iter().enumerate() {
                let w = ui.available_width();
                let (r, resp) = ui.allocate_exact_size(vec2(w, 66.0), Sense::click());
                let p = ui.painter();
                let selected = self.project.selected == Some(i);
                if selected {
                    p.rect_filled(r.shrink2(vec2(6.0, 2.0)), 4.0, HOVER);
                    p.rect_filled(Rect::from_min_size(r.min + vec2(6.0, 8.0), vec2(3.0, r.height() - 16.0)), 1.0, ACCENT);
                } else if resp.hovered() {
                    p.rect_filled(r.shrink2(vec2(6.0, 2.0)), 4.0, RAISED);
                }
                let thumb = Rect::from_min_size(r.min + vec2(16.0, 10.0), vec2(82.0, 46.0));
                match self.source_thumbs.get(&s.path) {
                    Some(t) => {
                        p.image(t.id(), thumb, Rect::from_min_max(Pos2::ZERO, pos2(1.0, 1.0)), Color32::WHITE);
                    }
                    None => {
                        p.rect_filled(thumb, 3.0, RAISED);
                        icons::paint_at(p, thumb.center(), 18.0, Icon::Film, MUTED);
                    }
                }
                let name = s.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                let tx = thumb.right() + 12.0;
                let clip = p.with_clip_rect(Rect::from_min_max(pos2(tx, r.top()), pos2(r.right() - 40.0, r.bottom())));
                clip.text(pos2(tx, r.top() + 10.0), Align2::LEFT_TOP, name, theme::semibold(13.0), TEXT);
                let detail = self.infos.get(&s.path).map(details).unwrap_or_else(|| "Reading…".into());
                clip.text(pos2(tx, r.top() + 29.0), Align2::LEFT_TOP, detail, theme::body(11.5), MUTED);
                let tabs = format!("{} tab{}", s.tabs.len(), if s.tabs.len() == 1 { "" } else { "s" });
                clip.text(pos2(tx, r.top() + 45.0), Align2::LEFT_TOP, tabs, theme::body(10.5), theme::DISABLED);
                let xr = Rect::from_center_size(pos2(r.right() - 24.0, r.center().y), vec2(22.0, 22.0));
                let xresp = ui.interact(xr, ui.scope_id().with(("src-close", i)), Sense::click());
                if xresp.hovered() {
                    ui.painter().rect_filled(xr, 4.0, theme::SELECTED);
                }
                icons::paint_at(ui.painter(), xr.center(), 11.0, Icon::Close, if xresp.hovered() { TEXT } else { MUTED });
                if xresp.on_hover_text("Close input").clicked() {
                    remove = Some(i);
                } else if resp.clicked() {
                    select = Some(i);
                }
                let path = s.path.clone();
                resp.context_menu(|ui| {
                    if ui.button(REVEAL_LABEL).clicked() {
                        reveal(&path);
                    }
                    if ui.button("Close input").clicked() {
                        remove = Some(i);
                    }
                });
            }
            for p in &self.probes {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.spinner();
                    ui.label(RichText::new(p.0.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).color(MUTED));
                });
            }
            ui.add_space(6.0);
        });
        if let Some(i) = remove {
            self.remove_source(i);
        } else if let Some(i) = select {
            self.select_source(i);
        }
    }

    fn queue_panel(&mut self, ui: &mut Ui) {
        let Some(queue) = self.queue.clone() else { return };
        let mut selected_only = self.queue_selected_only;
        let mut newest = self.queue_newest_first;
        let mut remaining = self.prefs.show_time_remaining;
        title_row(ui, "Export queue", |ui| {
            if widgets::icon_text_button(ui, Icon::Close, "Clear finished", 30.0, false).on_hover_text("Removes finished, failed and cancelled exports from the list").clicked() {
                queue.clear_finished();
            }
        });
        ui.horizontal(|ui| {
            ui.add_space(16.0);
            ui.allocate_ui(vec2(180.0, 30.0), |ui| segmented(ui, &mut selected_only, &[(false, "All sources"), (true, "Selected")]));
            ui.add_space(6.0);
            ui.allocate_ui(vec2(150.0, 30.0), |ui| segmented(ui, &mut newest, &[(true, "Newest"), (false, "Oldest")]));
            ui.add_space(6.0);
            ui.allocate_ui(vec2(170.0, 30.0), |ui| segmented(ui, &mut remaining, &[(true, "Remaining"), (false, "Elapsed")]));
        });
        if remaining != self.prefs.show_time_remaining {
            self.prefs.show_time_remaining = remaining;
            self.prefs_changed();
        }
        self.queue_selected_only = selected_only;
        self.queue_newest_first = newest;
        ui.add_space(4.0);
        let selected_name = self.selected_source().and_then(|s| s.path.file_name()).map(|n| n.to_string_lossy().to_string());
        let mut jobs = queue.jobs();
        if selected_only {
            jobs.retain(|j| selected_name.as_ref().is_some_and(|n| j.name.starts_with(n.as_str())));
        }
        if newest {
            jobs.reverse();
        }
        ScrollArea::vertical().max_height(400.0).auto_shrink([false, true]).show(ui, |ui| {
            if jobs.is_empty() {
                ui.add_space(8.0);
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.label(RichText::new("No exports yet. Quick export or Export as… adds the selected video with its active tab.").color(MUTED));
                });
                ui.add_space(12.0);
            }
            for j in &jobs {
                let w = ui.available_width();
                let (r, resp) = ui.allocate_exact_size(vec2(w, 74.0), Sense::click());
                let p = ui.painter();
                p.rect_filled(r.shrink2(vec2(8.0, 3.0)), 5.0, if resp.hovered() { HOVER } else { RAISED });
                let x = r.left() + 20.0;
                let (state, color) = match j.state {
                    JobState::Waiting => ("Queued", MUTED),
                    JobState::Processing => ("Processing", ACCENT),
                    JobState::Paused => ("Paused", WARNING),
                    JobState::Done => ("Done", ACCENT),
                    JobState::Error => ("Failed", ERROR),
                    JobState::Cancelled => ("Stopped", MUTED),
                };
                let clip = p.with_clip_rect(Rect::from_min_max(r.min, pos2(r.right() - 110.0, r.bottom())));
                clip.text(pos2(x, r.top() + 12.0), Align2::LEFT_TOP, &j.name, theme::semibold(13.0), TEXT);
                let out = j.output.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                clip.text(pos2(x, r.top() + 30.0), Align2::LEFT_TOP, out, theme::body(11.5), MUTED);
                let pr = &j.progress;
                let time = if remaining { format!("{} remaining", format_duration(pr.eta_secs)) } else { format!("{} elapsed", format_duration(pr.elapsed_secs)) };
                let status = match j.state {
                    JobState::Processing => format!("{state} · {}% · {:.1} fps · {time}{}", pr.percent, pr.fps, if pr.passes > 1 { format!(" · pass {}/{}", pr.pass, pr.passes) } else { String::new() }),
                    JobState::Done => format!("{state} · {}", format_duration(pr.elapsed_secs)),
                    JobState::Error => format!("{state} · {}", j.error.as_deref().unwrap_or(&j.message).lines().next().unwrap_or("")),
                    _ => state.to_string(),
                };
                clip.text(pos2(x, r.top() + 47.0), Align2::LEFT_TOP, status, theme::body(11.5), color);
                let bar = Rect::from_min_size(pos2(x, r.bottom() - 11.0), vec2(r.width() - 40.0, 3.0));
                p.rect_filled(bar, 1.5, theme::SELECTED);
                let frac = if j.state == JobState::Done { 1.0 } else { pr.percent as f32 / 100.0 };
                p.rect_filled(Rect::from_min_size(bar.min, vec2(bar.width() * frac, bar.height())), 1.5, if j.state == JobState::Error { ERROR } else { ACCENT });
                // Actions.
                let mut child = ui.new_child(egui::UiBuilder::new().max_rect(Rect::from_min_max(pos2(r.right() - 108.0, r.top() + 8.0), pos2(r.right() - 14.0, r.top() + 34.0))).layout(Layout::right_to_left(Align::Center)));
                child.spacing_mut().item_spacing.x = 2.0;
                match j.state {
                    JobState::Processing => {
                        if widgets::small_icon(&mut child, Icon::Stop, TEXT).on_hover_text("Stop processing").clicked() {
                            queue.cancel(j.id);
                        }
                        if widgets::small_icon(&mut child, Icon::Pause, TEXT).on_hover_text("Pause process").clicked() {
                            queue.pause(j.id);
                        }
                    }
                    JobState::Paused => {
                        if widgets::small_icon(&mut child, Icon::Stop, TEXT).on_hover_text("Stop processing").clicked() {
                            queue.cancel(j.id);
                        }
                        if widgets::small_icon(&mut child, Icon::Play, TEXT).on_hover_text("Resume process").clicked() {
                            queue.resume(j.id);
                        }
                    }
                    JobState::Waiting => {
                        if widgets::small_icon(&mut child, Icon::Close, TEXT).on_hover_text("Skip processing").clicked() {
                            queue.cancel(j.id);
                        }
                    }
                    JobState::Done => {
                        if widgets::small_icon(&mut child, Icon::Folder, TEXT).on_hover_text(REVEAL_LABEL).clicked() {
                            reveal(&j.output);
                        }
                        if widgets::small_icon(&mut child, Icon::Play, TEXT).on_hover_text("Open in external player").clicked() {
                            open_path(&j.output);
                        }
                    }
                    JobState::Error => {
                        let msg = j.error.clone().unwrap_or_else(|| j.message.clone());
                        widgets::small_icon(&mut child, Icon::Warning, ERROR).on_hover_ui(|ui| {
                            ui.set_max_width(420.0);
                            ui.label(RichText::new(&msg).color(TEXT));
                        });
                    }
                    JobState::Cancelled => {}
                }
                let (log, err) = (j.log.join("\n"), j.error.clone());
                resp.context_menu(|ui| {
                    if ui.button("Copy log").clicked() {
                        ui.ctx().copy_text(if log.is_empty() { err.clone().unwrap_or_default() } else { log.clone() });
                    }
                });
            }
            ui.add_space(6.0);
        });
    }

    fn presets_panel(&mut self, ui: &mut Ui) {
        let mut close = false;
        title_row(ui, "Preset", |ui| {
            if widgets::small_icon(ui, Icon::Close, MUTED).clicked() {
                close = true;
            }
        });
        ui.add_space(8.0);
        let mut custom = self.presets_custom;
        widgets::full(ui, |ui| segmented(ui, &mut custom, &[(true, "Custom"), (false, "Recommended")]));
        self.presets_custom = custom;
        ui.add_space(6.0);
        let favs = self.prefs.favorites.clone();
        let is_fav = |p: &Preset| if p.builtin { favs.iter().any(|f| f == &p.name) } else { p.favorite };
        let mut list: Vec<Preset> = self.preset_list.iter().filter(|p| p.builtin != custom).cloned().collect();
        list.sort_by_key(|p| !is_fav(p));
        let mut apply = None;
        let mut toggle_fav = None;
        let mut action: Option<(&str, Preset)> = None;
        ScrollArea::vertical().max_height(360.0).auto_shrink([false, true]).show(ui, |ui| {
            if list.is_empty() {
                ui.horizontal(|ui| {
                    ui.add_space(16.0);
                    ui.label(RichText::new("No custom presets yet. Use Create new to save the active tab's settings.").color(MUTED));
                });
            }
            for p in &list {
                let w = ui.available_width();
                let (r, resp) = ui.allocate_exact_size(vec2(w, 46.0), Sense::click());
                let painter = ui.painter();
                if resp.hovered() {
                    painter.rect_filled(r.shrink2(vec2(6.0, 1.0)), 4.0, HOVER);
                }
                let clip = painter.with_clip_rect(Rect::from_min_max(r.min, pos2(r.right() - 40.0, r.bottom())));
                clip.text(pos2(r.left() + 16.0, r.top() + 8.0), Align2::LEFT_TOP, &p.name, theme::body(13.0), TEXT);
                clip.text(pos2(r.left() + 16.0, r.top() + 26.0), Align2::LEFT_TOP, &p.description, theme::body(11.0), MUTED);
                let sr = Rect::from_center_size(pos2(r.right() - 24.0, r.center().y), vec2(22.0, 22.0));
                let sresp = ui.interact(sr, ui.scope_id().with(("fav", &p.name)), Sense::click());
                let fav = is_fav(p);
                icons::paint_at(ui.painter(), sr.center(), 13.0, if fav { Icon::StarFilled } else { Icon::Star }, if fav { theme::WARNING } else if sresp.hovered() { TEXT } else { MUTED });
                if sresp.on_hover_text(if fav { "Remove from favourites" } else { "Add to favourites" }).clicked() {
                    toggle_fav = Some(p.clone());
                } else if resp.clicked() {
                    apply = Some(p.clone());
                }
                if !p.builtin {
                    resp.context_menu(|ui| {
                        if ui.button("Update with current settings").clicked() {
                            action = Some(("update", p.clone()));
                        }
                        if ui.button("Rename preset…").clicked() {
                            action = Some(("rename", p.clone()));
                        }
                        if ui.button("Delete preset").clicked() {
                            action = Some(("delete", p.clone()));
                        }
                    });
                }
            }
        });
        ui.add_space(6.0);
        widgets::divider(ui);
        ui.add_space(8.0);
        widgets::full(ui, |ui| {
            ui.add_enabled_ui(self.selected_source().is_some(), |ui| {
                if widgets::button(ui, "Create new", vec2(ui.available_width(), 30.0)).on_hover_text("Saves the active tab's settings as a preset").clicked() {
                    let name = self.selected_source().map(|s| s.active().name.clone()).unwrap_or_default();
                    self.dialog = Some(Dialog::NewPreset { name, description: String::new() });
                }
            });
        });
        ui.add_space(10.0);
        if let Some(p) = apply {
            if let Some(s) = self.active_settings_mut() {
                p.apply(s);
            }
            if let Some(i) = self.project.selected {
                let tab = self.project.sources[i].active_mut();
                if tab.name.starts_with("Untitled") {
                    tab.name = p.name.clone();
                }
            }
            self.mark_dirty();
        }
        if let Some(mut p) = toggle_fav {
            if p.builtin {
                if let Some(i) = self.prefs.favorites.iter().position(|f| f == &p.name) {
                    self.prefs.favorites.remove(i);
                } else {
                    self.prefs.favorites.push(p.name.clone());
                }
                self.prefs_changed();
            } else {
                p.favorite = !p.favorite;
                if let Err(e) = self.presets.save(&p) {
                    self.notify(format!("{e:#}"), true);
                }
                self.preset_list = self.presets.list().0;
            }
        }
        if let Some((what, p)) = action {
            match what {
                "update" => {
                    if let Some(s) = self.active_settings().cloned() {
                        let mut n = Preset::capture(&p.name, &p.description, &s);
                        n.favorite = p.favorite;
                        if let Err(e) = self.presets.save(&n) {
                            self.notify(format!("{e:#}"), true);
                        }
                    }
                }
                "rename" => self.dialog = Some(Dialog::RenamePreset { from: p.name.clone(), name: p.name.clone() }),
                _ => {
                    if let Err(e) = self.presets.delete(&p.name) {
                        self.notify(format!("{e:#}"), true);
                    }
                }
            }
            self.preset_list = self.presets.list().0;
        }
        if close {
            self.popover = None;
        }
    }

    fn input_panel(&mut self, ui: &mut Ui) {
        let mut close = false;
        title_row(ui, "Input settings", |ui| {
            if widgets::small_icon(ui, Icon::Close, MUTED).clicked() {
                close = true;
            }
        });
        let (Some(info), Some(s)) = (self.selected_info().cloned(), self.active_settings().cloned()) else { return };
        ui.add_space(6.0);
        widgets::full(ui, |ui| {
            ui.label(RichText::new(info.path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()).font(theme::semibold(13.0)));
            ui.label(RichText::new(details(&info)).color(MUTED).small());
            let c = &info.color;
            ui.label(RichText::new(format!("{} · {} · {:?} range · {}-bit {}", c.primaries, c.transfer, c.range, info.bit_depth, info.pix_fmt)).color(MUTED).small());
        });
        ui.add_space(6.0);
        let mut n = s.clone();
        widgets::row(ui, "Rotation", ("Rotation", "Rotates the input clockwise. Changing it resets the crop."), |ui| {
            segmented(ui, &mut n.rotation, &[(0u8, "0°"), (1, "90°"), (2, "180°"), (3, "270°")]);
        });
        let mut mode = if !n.deinterlace.enabled { 0 } else if n.deinterlace.mode == DeinterlaceMode::DoubleRate { 2 } else { 1 };
        let telecine = n.deinterlace.enabled && n.deinterlace.mode == DeinterlaceMode::Telecine;
        widgets::row(ui, "Deinterlace", ("Deinterlace", "Off, one frame per frame, or one frame per field (double rate)."), |ui| {
            if segmented(ui, &mut mode, &[(0, "Off"), (1, "Same rate"), (2, "Double rate")]) {
                n.deinterlace.enabled = mode != 0;
                n.deinterlace.mode = if mode == 2 { DeinterlaceMode::DoubleRate } else { DeinterlaceMode::SameRate };
            }
        });
        let mut tc = telecine;
        widgets::row(ui, "Telecined", ("Telecined video", "Reverses 3:2 pulldown back to 23.976 fps before processing."), |ui| {
            if widgets::check(ui, &mut tc, "Inverse telecine").changed() {
                n.deinterlace.enabled = tc || mode != 0;
                n.deinterlace.mode = if tc { DeinterlaceMode::Telecine } else { DeinterlaceMode::SameRate };
            }
        });
        if n.deinterlace.enabled {
            widgets::row(ui, "Field order", ("Field order", ""), |ui| {
                let name = |f: FieldOrderSetting| match f {
                    FieldOrderSetting::Auto => "Auto",
                    FieldOrderSetting::TopFirst => "Top field first",
                    FieldOrderSetting::BottomFirst => "Bottom field first",
                };
                widgets::combo(ui, "in-field", name(n.deinterlace.field_order), |ui| {
                    for f in [FieldOrderSetting::Auto, FieldOrderSetting::TopFirst, FieldOrderSetting::BottomFirst] {
                        ui.selectable_value(&mut n.deinterlace.field_order, f, name(f));
                    }
                });
            });
        }
        widgets::row(ui, "Pixel type", ("Pixel type", "Converts anamorphic video to square pixels."), |ui| {
            widgets::check(ui, &mut n.correction.square_pixels, "Square pixels");
        });
        let mut over = n.correction.color_override.is_some();
        widgets::row(ui, "Colour", ("Override input colourspace", "Changes how the input's colours are interpreted for processing."), |ui| {
            if widgets::check(ui, &mut over, "Override").changed() {
                n.correction.color_override = over.then(|| ColorOverride {
                    range: if matches!(info.color.range, prism_core::media::ColorRange::Full) { "pc".into() } else { "tv".into() },
                    primaries: info.color.primaries.clone(),
                    transfer: info.color.transfer.clone(),
                    matrix: match info.color.matrix {
                        prism_core::media::Matrix::Bt601 => "smpte170m".into(),
                        prism_core::media::Matrix::Bt709 => "bt709".into(),
                        prism_core::media::Matrix::Bt2020Nc => "bt2020nc".into(),
                    },
                });
            }
        });
        if let Some(o) = &mut n.correction.color_override {
            let rows: [(&str, &mut String, &[&str]); 4] = [
                ("Range", &mut o.range, &["tv", "pc"]),
                ("Primaries", &mut o.primaries, &["bt709", "bt470bg", "smpte170m", "bt2020"]),
                ("Transfer", &mut o.transfer, &["bt709", "smpte170m", "smpte2084", "arib-std-b67", "iec61966-2-1"]),
                ("Matrix", &mut o.matrix, &["bt709", "bt470bg", "smpte170m", "bt2020nc"]),
            ];
            for (label, v, opts) in rows {
                widgets::row(ui, label, ("", ""), |ui| {
                    widgets::combo(ui, ("in-color", label), v.clone(), |ui| {
                        for o in opts {
                            ui.selectable_value(v, o.to_string(), *o);
                        }
                    });
                });
            }
        }
        ui.add_space(10.0);
        if n != s {
            let rotated = n.rotation != s.rotation;
            let (rot, de, cor) = (n.rotation, n.deinterlace.clone(), n.correction.clone());
            self.for_all_tabs(|t| {
                t.rotation = rot;
                t.deinterlace = de.clone();
                t.correction = cor.clone();
                if rotated {
                    t.crop = None;
                }
            });
        }
        if close {
            self.popover = None;
        }
    }

    fn projects_panel(&mut self, ui: &mut Ui) {
        let ctx = ui.ctx().clone();
        title_row(ui, "Projects", |_| {});
        let root = paths::data_dir().join("projects");
        let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).map(|rd| rd.flatten().map(|e| e.path()).filter(|p| p.join(prism_core::project::PROJECT_FILE).is_file()).collect()).unwrap_or_default();
        if !dirs.contains(&self.project_dir) {
            dirs.insert(0, self.project_dir.clone());
        }
        let mut switch = None;
        ScrollArea::vertical().max_height(300.0).auto_shrink([false, true]).show(ui, |ui| {
            for d in &dirs {
                let w = ui.available_width();
                let (r, resp) = ui.allocate_exact_size(vec2(w, 36.0), Sense::click());
                let current = *d == self.project_dir;
                if current || resp.hovered() {
                    ui.painter().rect_filled(r.shrink2(vec2(6.0, 1.0)), 4.0, if current { HOVER } else { RAISED });
                }
                let name = if current { self.project.name.clone() } else { d.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default() };
                ui.painter().text(pos2(r.left() + 16.0, r.center().y), Align2::LEFT_CENTER, name, theme::body(13.0), if current { TEXT } else { LABEL });
                if current {
                    icons::paint_at(ui.painter(), pos2(r.right() - 22.0, r.center().y), 12.0, Icon::Check, ACCENT);
                }
                if resp.on_hover_text(d.display().to_string()).clicked() && !current {
                    switch = Some(d.clone());
                }
            }
        });
        widgets::divider(ui);
        ui.add_space(4.0);
        let mut close = false;
        for (label, a) in [("New project…", 0), ("Open project…", 1), ("Save project as…", 2)] {
            widgets::full(ui, |ui| {
                if widgets::icon_text_button(ui, if a == 0 { Icon::Plus } else { Icon::Folder }, label, 30.0, false).clicked() {
                    close = true;
                    match a {
                        0 => self.dialog = Some(Dialog::NewProject { name: String::new() }),
                        1 => self.open_project_dialog(&ctx),
                        _ => self.save_project_as(&ctx),
                    }
                }
            });
        }
        ui.add_space(6.0);
        if let Some(d) = switch {
            self.switch_project(d, &ctx);
            close = true;
        }
        if close {
            self.popover = None;
        }
    }

    // ---------------------------------------------------------------- dialogs

    pub(crate) fn dialogs(&mut self, ctx: &egui::Context) {
        let Some(dialog) = self.dialog.take() else { return };
        let mut keep = true;
        let modal = egui::Modal::new(Id::unique("dialog")).frame(egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.0, HOVER)).corner_radius(8).inner_margin(20));
        let mut dialog = dialog;
        let resp = modal.show(ctx, |ui| match &mut dialog {
            Dialog::Preferences => {
                ui.set_width(540.0);
                self.preferences(ui, &mut keep);
            }
            Dialog::Models => {
                ui.set_width(660.0);
                self.models(ui, &mut keep);
            }
            Dialog::About => {
                ui.set_width(440.0);
                ui.horizontal(|ui| {
                    let (r, _) = ui.allocate_exact_size(vec2(44.0, 44.0), Sense::hover());
                    icons::paint(ui.painter(), r, Icon::Logo, ACCENT);
                    ui.vertical(|ui| {
                        ui.label(RichText::new("Prism Video Enhancer").font(theme::semibold(18.0)));
                        ui.label(RichText::new(format!("Version {}", env!("CARGO_PKG_VERSION"))).color(MUTED));
                    });
                });
                ui.add_space(10.0);
                ui.label(RichText::new("Video upscaling, restoration, frame interpolation, stabilization and SDR-to-HDR on stock FFmpeg, native Rust kernels and ONNX Runtime.").color(LABEL));
                ui.add_space(8.0);
                if let Some(t) = &self.tools {
                    ui.label(RichText::new(format!("FFmpeg: {}", t.ffmpeg.display())).color(MUTED).small());
                }
                ui.label(RichText::new(self.runtime_status.lock().unwrap().clone()).color(MUTED).small());
                ui.label(RichText::new("Network models are converted from their openly licensed releases (Real-ESRGAN: BSD-3-Clause).").color(MUTED).small());
                ui.add_space(12.0);
                if widgets::button(ui, "Close", vec2(100.0, 30.0)).clicked() {
                    keep = false;
                }
            }
            Dialog::Command { title, text } => {
                ui.set_width(760.0);
                ui.label(RichText::new(title.as_str()).font(theme::semibold(16.0)));
                ui.add_space(8.0);
                ScrollArea::vertical().max_height(420.0).show(ui, |ui| {
                    ui.add(egui::TextEdit::multiline(text).font(egui::TextStyle::Monospace).desired_width(f32::INFINITY).interactive(true));
                });
                ui.add_space(10.0);
                ui.horizontal(|ui| {
                    if widgets::primary_button(ui, None, "Copy", vec2(100.0, 30.0)).clicked() {
                        ui.ctx().copy_text(text.clone());
                    }
                    if widgets::button(ui, "Close", vec2(100.0, 30.0)).clicked() {
                        keep = false;
                    }
                });
            }
            Dialog::NewPreset { name, description } => {
                ui.set_width(380.0);
                ui.label(RichText::new("Create preset").font(theme::semibold(16.0)));
                ui.add_space(10.0);
                ui.label(RichText::new("Name").color(LABEL));
                ui.add(egui::TextEdit::singleline(name).desired_width(f32::INFINITY)).request_focus();
                ui.label(RichText::new("Description").color(LABEL));
                ui.add(egui::TextEdit::singleline(description).desired_width(f32::INFINITY));
                ui.add_space(4.0);
                ui.label(RichText::new("Saves the active tab's filters and export settings. Trim, crop and output paths are not included.").color(MUTED).small());
                let trimmed = name.trim().to_string();
                let error = if trimmed.is_empty() {
                    Some("Enter a name")
                } else if trimmed.eq_ignore_ascii_case("none") {
                    Some("A preset cannot be called 'None'")
                } else if self.preset_list.iter().any(|p| p.name.eq_ignore_ascii_case(&trimmed)) {
                    Some("A preset with this name already exists")
                } else {
                    None
                };
                if let Some(e) = error.filter(|_| !trimmed.is_empty()) {
                    ui.label(RichText::new(e).color(ERROR).small());
                }
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled_ui(error.is_none(), |ui| widgets::primary_button(ui, None, "Create", vec2(100.0, 30.0))).inner.clicked() {
                        if let Some(s) = self.active_settings() {
                            let p = Preset::capture(&trimmed, description.trim(), s);
                            match self.presets.save(&p) {
                                Ok(_) => {
                                    self.preset_list = self.presets.list().0;
                                    self.presets_custom = true;
                                }
                                Err(e) => self.notify(format!("{e:#}"), true),
                            }
                        }
                        keep = false;
                    }
                    if widgets::button(ui, "Cancel", vec2(100.0, 30.0)).clicked() {
                        keep = false;
                    }
                });
            }
            Dialog::RenamePreset { from, name } => {
                ui.set_width(360.0);
                ui.label(RichText::new("Rename preset").font(theme::semibold(16.0)));
                ui.add_space(10.0);
                ui.add(egui::TextEdit::singleline(name).desired_width(f32::INFINITY)).request_focus();
                let trimmed = name.trim().to_string();
                let ok = !trimmed.is_empty() && !trimmed.eq_ignore_ascii_case("none") && (trimmed.eq_ignore_ascii_case(from) || !self.preset_list.iter().any(|p| p.name.eq_ignore_ascii_case(&trimmed)));
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled_ui(ok, |ui| widgets::primary_button(ui, None, "Rename", vec2(100.0, 30.0))).inner.clicked() {
                        if let Some(mut p) = self.presets.find(from) {
                            p.name = trimmed.clone();
                            let r = self.presets.save(&p).and_then(|_| if trimmed.eq_ignore_ascii_case(from) { Ok(()) } else { self.presets.delete(from) });
                            if let Err(e) = r {
                                self.notify(format!("{e:#}"), true);
                            }
                            self.preset_list = self.presets.list().0;
                        }
                        keep = false;
                    }
                    if widgets::button(ui, "Cancel", vec2(100.0, 30.0)).clicked() {
                        keep = false;
                    }
                });
            }
            Dialog::NewProject { name } => {
                ui.set_width(360.0);
                ui.label(RichText::new("New project").font(theme::semibold(16.0)));
                ui.add_space(10.0);
                ui.add(egui::TextEdit::singleline(name).hint_text("Project name").desired_width(f32::INFINITY)).request_focus();
                let safe: String = name.trim().chars().map(|c| if c.is_alphanumeric() || " -_()".contains(c) { c } else { '_' }).collect();
                let dir = paths::data_dir().join("projects").join(safe.trim());
                let exists = dir.join(prism_core::project::PROJECT_FILE).exists();
                if exists {
                    ui.label(RichText::new("A project with this name already exists").color(ERROR).small());
                }
                ui.add_space(12.0);
                ui.horizontal(|ui| {
                    if ui.add_enabled_ui(!safe.trim().is_empty() && !exists, |ui| widgets::primary_button(ui, None, "Create", vec2(100.0, 30.0))).inner.clicked() {
                        let _ = self.project.save(&self.project_dir);
                        let mut p = prism_core::project::Project::default();
                        p.name = name.trim().to_string();
                        match p.save(&dir) {
                            Ok(_) => self.switch_project(dir.clone(), ui.ctx()),
                            Err(e) => self.notify(format!("{e:#}"), true),
                        }
                        keep = false;
                    }
                    if widgets::button(ui, "Cancel", vec2(100.0, 30.0)).clicked() {
                        keep = false;
                    }
                });
            }
            Dialog::CloseAll => {
                ui.set_width(380.0);
                ui.label(RichText::new("Close all sources").font(theme::semibold(16.0)));
                ui.add_space(8.0);
                ui.label(RichText::new("Are you sure you want to close all sources? Their tabs and settings are removed from the project; exports keep running.").color(LABEL));
                ui.add_space(14.0);
                ui.horizontal(|ui| {
                    if widgets::primary_button(ui, None, "Close all", vec2(110.0, 30.0)).clicked() {
                        self.clear_sources();
                        keep = false;
                    }
                    if widgets::button(ui, "Cancel", vec2(100.0, 30.0)).clicked() {
                        keep = false;
                    }
                });
            }
        });
        if resp.should_close() {
            keep = false;
        }
        // A handler may have opened another dialog; that one wins.
        if keep && self.dialog.is_none() {
            self.dialog = Some(dialog);
        }
    }

    fn preferences(&mut self, ui: &mut Ui, keep: &mut bool) {
        ui.label(RichText::new("Preferences").font(theme::semibold(18.0)));
        ui.add_space(6.0);
        let before = serde_json::to_string(&self.prefs).unwrap_or_default();
        ScrollArea::vertical().max_height(520.0).show(ui, |ui| {
            let heading = |ui: &mut Ui, t: &str| {
                ui.add_space(10.0);
                ui.label(RichText::new(t).font(theme::semibold(14.0)).color(TEXT));
                widgets::divider(ui);
                ui.add_space(2.0);
            };
            heading(ui, "Processing");
            ui.label(RichText::new(self.runtime_status.lock().unwrap().clone()).color(MUTED).small());
            let d = &mut self.prefs.device;
            widgets::row(ui, "AI processor", ("AI processor", "Where network models run. Auto uses the GPU when ONNX Runtime offers one."), |ui| {
                let name = |b: Backend| match b {
                    Backend::Auto => "Auto",
                    Backend::DirectMl => "GPU (DirectML)",
                    Backend::Cuda => "GPU (CUDA)",
                    Backend::CoreMl => "GPU / Neural Engine (Core ML)",
                    Backend::Cpu => "CPU",
                };
                widgets::combo(ui, "pref-backend", name(d.backend), |ui| {
                    // Only the GPU providers ONNX Runtime offers on this OS.
                    let mut list = vec![Backend::Auto];
                    if cfg!(windows) {
                        list.extend([Backend::DirectMl, Backend::Cuda]);
                    } else if cfg!(target_os = "macos") {
                        list.push(Backend::CoreMl);
                    } else {
                        list.push(Backend::Cuda);
                    }
                    list.push(Backend::Cpu);
                    for b in list {
                        ui.selectable_value(&mut d.backend, b, name(b));
                    }
                });
            });
            widgets::row(ui, "GPU", ("GPU index", "Adapter index for GPU backends; -1 uses the default adapter."), |ui| {
                ui.add(egui::DragValue::new(&mut d.device_index).range(-1..=8));
            });
            widgets::row(ui, "Max memory", ("Max memory usage", "Share of GPU memory one tile may use."), |ui| widgets::slider_f64(ui, &mut d.max_memory, 0.1..=1.0, 100.0, 0.7));
            widgets::row(ui, "Precision", ("FP16", "Half-precision models are faster on most GPUs."), |ui| {
                widgets::check(ui, &mut d.fp16, "Prefer FP16 models");
            });
            widgets::row(ui, "Decoding", ("Hardware decoding", "Lets FFmpeg decode on the GPU."), |ui| {
                widgets::check(ui, &mut d.hw_decode, "Hardware decoding");
            });
            widgets::row(ui, "CPU threads", ("CPU threads", "Threads for native kernels; 0 uses every core."), |ui| {
                ui.add(egui::DragValue::new(&mut d.threads).range(0..=256));
            });
            heading(ui, "Preview");
            let p = &mut self.prefs;
            widgets::row(ui, "Render length", ("Preview length", ""), |ui| {
                widgets::combo(ui, "pref-len", preview_len_label(p.preview_seconds), |ui| {
                    for s in [0.5, 1.0, 2.0, 5.0, 10.0, 30.0, -1.0] {
                        ui.selectable_value(&mut p.preview_seconds, s, preview_len_label(s));
                    }
                });
            });
            widgets::row(ui, "Live render", ("Live render", "Processes the paused frame automatically."), |ui| {
                widgets::check(ui, &mut p.live_render, "Enabled");
            });
            heading(ui, "Export");
            let mut workers = p.export_workers as i32;
            widgets::row(ui, "Simultaneous", ("Simultaneous exports", "How many exports run at once."), |ui| {
                ui.add(egui::DragValue::new(&mut workers).range(1..=4));
            });
            if workers as usize != p.export_workers {
                p.export_workers = workers as usize;
                if let Some(q) = &self.queue {
                    q.set_workers(p.export_workers);
                }
            }
            heading(ui, "Folders");
            for (label, path) in [("Data", paths::data_dir()), ("Models", paths::models_dir()), ("Temporary", paths::temp_root())] {
                widgets::row(ui, label, ("", ""), |ui| {
                    ui.add(egui::Label::new(RichText::new(path.display().to_string()).color(LABEL).small()).truncate());
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        if widgets::small_icon(ui, Icon::Folder, TEXT).on_hover_text("Open").clicked() {
                            let _ = std::fs::create_dir_all(&path);
                            open_path(&path);
                        }
                    });
                });
            }
        });
        if serde_json::to_string(&self.prefs).unwrap_or_default() != before {
            self.prefs_changed();
        }
        ui.add_space(12.0);
        if widgets::primary_button(ui, None, "Done", vec2(100.0, 30.0)).clicked() {
            *keep = false;
        }
    }

    fn models(&mut self, ui: &mut Ui, keep: &mut bool) {
        let ctx = ui.ctx().clone();
        ui.label(RichText::new("Model manager").font(theme::semibold(18.0)));
        ui.label(RichText::new(self.runtime_status.lock().unwrap().clone()).color(MUTED).small());
        ui.add_space(8.0);
        let dir = self.catalog.dir.clone();
        let mut fetch = None;
        let mut import = None;
        ScrollArea::vertical().max_height(460.0).show(ui, |ui| {
            for (kind, title) in [(ModelKind::Enhancement, "Enhancement"), (ModelKind::Interpolation, "Frame interpolation"), (ModelKind::MotionDeblur, "Motion deblur"), (ModelKind::Stabilization, "Stabilization"), (ModelKind::Hdr, "SDR to HDR")] {
                let models: Vec<_> = self.catalog.of_kind(kind).cloned().collect();
                if models.is_empty() {
                    continue;
                }
                ui.add_space(8.0);
                ui.label(RichText::new(title).font(theme::semibold(13.5)));
                widgets::divider(ui);
                for m in models {
                    ui.horizontal(|ui| {
                        ui.set_min_height(40.0);
                        ui.vertical(|ui| {
                            ui.set_width(360.0);
                            ui.label(RichText::new(&m.name).color(TEXT));
                            ui.add(egui::Label::new(RichText::new(&m.description).color(MUTED).small()).truncate());
                        });
                        let (status, color) = match &m.implementation {
                            Implementation::Native => ("Built in", LABEL),
                            Implementation::Ffmpeg { .. } => ("FFmpeg", LABEL),
                            Implementation::Onnx { .. } if m.is_available(&dir) => ("Installed", ACCENT),
                            Implementation::Onnx { .. } => ("Not installed", WARNING),
                        };
                        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                            if let Implementation::Onnx { source, .. } = &m.implementation {
                                if !m.is_available(&dir) {
                                    if self.fetching.as_ref().is_some_and(|f| f.0 == m.id) {
                                        ui.spinner();
                                    } else {
                                        ui.add_enabled_ui(self.fetching.is_none(), |ui| {
                                            if widgets::button(ui, "Import…", vec2(70.0, 26.0)).on_hover_text("Install a model file you already have (.onnx)").clicked() {
                                                import = Some(m.id.clone());
                                            }
                                            if source.is_some() && widgets::button(ui, "Download", vec2(84.0, 26.0)).on_hover_text(crate::settings_ui::source_tip(&m)).clicked() {
                                                fetch = Some(m.id.clone());
                                            }
                                        });
                                    }
                                }
                            }
                            ui.label(RichText::new(status).color(color));
                        });
                    });
                }
            }
        });
        if let Some(id) = fetch {
            self.fetch_model(id, &ctx);
        }
        if let Some(id) = import {
            self.import_model(id, &ctx);
        }
        ui.add_space(12.0);
        ui.horizontal(|ui| {
            if widgets::primary_button(ui, None, "Done", vec2(100.0, 30.0)).clicked() {
                *keep = false;
            }
            if widgets::button(ui, "Open models folder", vec2(150.0, 30.0)).clicked() {
                let _ = std::fs::create_dir_all(&dir);
                open_path(&dir);
            }
        });
    }

    pub(crate) fn toast(&mut self, ctx: &egui::Context) {
        let Some((text, error, _)) = self.message.clone() else { return };
        let c = self.central_rect;
        let mut close = false;
        egui::Area::new(Id::unique("toast")).order(Order::Foreground).fixed_pos(pos2(c.center().x, self.bottom_top - 16.0)).pivot(Align2::CENTER_BOTTOM).show(ctx, |ui| {
            egui::Frame::new().fill(theme::RAISED).stroke(Stroke::new(1.0, if error { ERROR } else { ACCENT })).corner_radius(6).inner_margin(egui::Margin::symmetric(14, 10)).show(ui, |ui| {
                ui.set_max_width(560.0);
                ui.horizontal(|ui| {
                    icons::paint_at(ui.painter(), ui.cursor().min + vec2(8.0, 10.0), 14.0, if error { Icon::Warning } else { Icon::Check }, if error { ERROR } else { ACCENT });
                    ui.add_space(22.0);
                    ScrollArea::vertical().max_height(160.0).show(ui, |ui| {
                        ui.add(egui::Label::new(RichText::new(&text).color(TEXT)).wrap());
                    });
                    if widgets::small_icon(ui, Icon::Close, MUTED).clicked() {
                        close = true;
                    }
                });
            });
        });
        if close {
            self.message = None;
        }
    }
}
