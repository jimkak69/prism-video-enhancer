//! Controls in the app's style: switches, segmented bars, sliders, section headers and buttons.

use std::ops::RangeInclusive;

use egui::{Align, Align2, Color32, CornerRadius, Layout, Pos2, Rect, Response, Sense, Stroke, StrokeKind, Ui, Vec2, pos2, vec2};

use crate::icons::{self, Icon};
use crate::theme::{self, ACCENT, ACCENT_HOVER, DISABLED, HOVER, LABEL, MUTED, ON_ACCENT, RAISED, SELECTED, TEXT};

/// Width of the label column in pane rows.
pub const LABEL_W: f32 = 104.0;
pub const ROW_H: f32 = 32.0;
pub const LEFT: f32 = 18.0;
pub const RIGHT: f32 = 16.0;

/// Adds a title + text tooltip to `resp`.
pub fn tip(resp: Response, title: &str, text: &str) -> Response {
    if title.is_empty() && text.is_empty() {
        return resp;
    }
    resp.on_hover_ui(|ui| {
        ui.set_max_width(280.0);
        if !title.is_empty() {
            ui.label(egui::RichText::new(title).font(theme::semibold(13.0)));
        }
        if !text.is_empty() {
            ui.label(egui::RichText::new(crate::app::key_label(text)).color(LABEL));
        }
    })
}

/// A transparent button holding one icon; `selected` keeps the hover fill.
pub fn icon_button(ui: &mut Ui, icon: Icon, size: Vec2, icon_size: f32, selected: bool) -> Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let enabled = ui.is_enabled();
    if selected || (enabled && resp.hovered()) {
        ui.painter().rect_filled(rect, 0.0, HOVER);
    }
    let color = if !enabled { DISABLED } else if selected || resp.hovered() { Color32::WHITE } else { TEXT };
    icons::paint_at(ui.painter(), rect.center(), icon_size, icon, color);
    resp
}

/// A small square icon button with rounded hover, used inside rows and lists.
pub fn small_icon(ui: &mut Ui, icon: Icon, color: Color32) -> Response {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(22.0), Sense::click());
    if resp.hovered() && ui.is_enabled() {
        ui.painter().rect_filled(rect, 4.0, HOVER);
    }
    let c = if !ui.is_enabled() { DISABLED } else if resp.hovered() { Color32::WHITE } else { color };
    icons::paint_at(ui.painter(), rect.center(), 13.0, icon, c);
    resp
}

/// A transparent button with an icon and a label (header and bottom bar actions).
pub fn icon_text_button(ui: &mut Ui, icon: Icon, text: &str, height: f32, selected: bool) -> Response {
    let font = theme::body(13.0);
    let galley = ui.painter().layout_no_wrap(text.to_string(), font, TEXT);
    let w = 16.0 + 16.0 + 8.0 + galley.size().x + 16.0;
    let (rect, resp) = ui.allocate_exact_size(vec2(w, height), Sense::click());
    let enabled = ui.is_enabled();
    if selected || (enabled && resp.hovered()) {
        ui.painter().rect_filled(rect, 0.0, HOVER);
    }
    let color = if enabled { TEXT } else { DISABLED };
    icons::paint_at(ui.painter(), pos2(rect.left() + 24.0, rect.center().y), 15.0, icon, color);
    ui.painter().galley(pos2(rect.left() + 40.0, rect.center().y - galley.size().y / 2.0), galley, color);
    resp
}

/// The filled accent button (Export, Render, Create).
pub fn primary_button(ui: &mut Ui, icon: Option<Icon>, text: &str, size: Vec2) -> Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let enabled = ui.is_enabled();
    let fill = if !enabled { theme::ACCENT_DIM } else if resp.is_pointer_button_down_on() { theme::ACCENT_DIM } else if resp.hovered() { ACCENT_HOVER } else { ACCENT };
    ui.painter().rect_filled(rect, 0.0, fill);
    let fg = if enabled { ON_ACCENT } else { Color32::from_rgb(0x9C, 0xB8, 0xA6) };
    let galley = ui.painter().layout_no_wrap(text.to_string(), theme::semibold(13.0), fg);
    let icon_w = if icon.is_some() { 22.0 } else { 0.0 };
    let x = rect.center().x - (galley.size().x + icon_w) / 2.0;
    if let Some(i) = icon {
        icons::paint_at(ui.painter(), pos2(x + 8.0, rect.center().y), 15.0, i, fg);
    }
    ui.painter().galley(pos2(x + icon_w, rect.center().y - galley.size().y / 2.0), galley, fg);
    resp
}

/// A neutral filled button (dialogs, secondary actions).
pub fn button(ui: &mut Ui, text: &str, size: Vec2) -> Response {
    let (rect, resp) = ui.allocate_exact_size(size, Sense::click());
    let enabled = ui.is_enabled();
    let fill = if enabled && resp.hovered() { SELECTED } else { RAISED };
    ui.painter().rect_filled(rect, 4.0, fill);
    ui.painter().text(rect.center(), Align2::CENTER_CENTER, text, theme::body(13.0), if enabled { TEXT } else { DISABLED });
    resp
}

/// On/off switch.
pub fn switch(ui: &mut Ui, on: &mut bool) -> Response {
    let (rect, mut resp) = ui.allocate_exact_size(vec2(30.0, 16.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let t = ui.ctx().animate_bool_with_time(resp.id, *on, 0.12);
    let enabled = ui.is_enabled();
    let track = if !enabled {
        RAISED
    } else if *on {
        ACCENT
    } else if resp.hovered() {
        Color32::from_rgb(0x60, 0x60, 0x60)
    } else {
        Color32::from_rgb(0x54, 0x54, 0x54)
    };
    ui.painter().rect_filled(rect, 8.0, track);
    let knob = if *on { Color32::WHITE } else { Color32::from_rgb(0xB0, 0xB0, 0xB0) };
    let x = egui::lerp(rect.left() + 8.0..=rect.right() - 8.0, t);
    ui.painter().circle_filled(pos2(x, rect.center().y), 5.5, if enabled { knob } else { DISABLED });
    resp
}

/// A bar of mutually exclusive options filling the available width.
pub fn segmented<T: PartialEq + Clone>(ui: &mut Ui, value: &mut T, options: &[(T, &str)]) -> bool {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 28.0), Sense::hover());
    ui.painter().rect_filled(rect, 4.0, RAISED);
    let cell = rect.width() / options.len().max(1) as f32;
    let mut changed = false;
    let enabled = ui.is_enabled();
    for (i, (v, label)) in options.iter().enumerate() {
        let r = Rect::from_min_size(pos2(rect.left() + i as f32 * cell, rect.top()), vec2(cell, rect.height()));
        let resp = ui.interact(r, ui.scope_id().with(("seg", i, label)), Sense::click());
        let selected = *value == *v;
        if selected {
            ui.painter().rect_filled(r.shrink(2.0), 3.0, SELECTED);
        } else if resp.hovered() && enabled {
            ui.painter().rect_filled(r.shrink(2.0), 3.0, HOVER);
        }
        let color = if !enabled { DISABLED } else if selected { TEXT } else { MUTED };
        let galley = ui.painter().layout_no_wrap(label.to_string(), theme::body(12.5), color);
        let pos = r.center() - galley.size() / 2.0;
        ui.painter().with_clip_rect(r.shrink(3.0)).galley(pos, galley, color);
        if resp.clicked() && !selected {
            *value = v.clone();
            changed = true;
        }
    }
    changed
}

/// A checkbox with its label.
pub fn check(ui: &mut Ui, on: &mut bool, text: &str) -> Response {
    let galley = ui.painter().layout_no_wrap(text.to_string(), theme::body(13.0), TEXT);
    let w = 14.0 + if text.is_empty() { 0.0 } else { 8.0 + galley.size().x };
    let (rect, mut resp) = ui.allocate_exact_size(vec2(w, 20.0), Sense::click());
    if resp.clicked() {
        *on = !*on;
        resp.mark_changed();
    }
    let enabled = ui.is_enabled();
    let b = Rect::from_center_size(pos2(rect.left() + 7.0, rect.center().y), Vec2::splat(14.0));
    if *on {
        ui.painter().rect_filled(b, 2.0, if enabled { ACCENT } else { DISABLED });
        icons::paint_at(ui.painter(), b.center(), 11.0, Icon::Check, ON_ACCENT);
    } else {
        let c = if !enabled { DISABLED } else if resp.hovered() { TEXT } else { MUTED };
        ui.painter().rect_stroke(b, 2.0, Stroke::new(1.0, c), StrokeKind::Inside);
    }
    if !text.is_empty() {
        let c = if enabled { LABEL } else { DISABLED };
        ui.painter().galley(pos2(b.right() + 8.0, rect.center().y - galley.size().y / 2.0), galley, c);
    }
    resp
}

/// A horizontal slider with the value at the right. Double-click resets to `default`.
pub fn slider(ui: &mut Ui, value: &mut i32, range: RangeInclusive<i32>, default: i32) -> Response {
    let w = ui.available_width();
    let (rect, mut resp) = ui.allocate_exact_size(vec2(w, ROW_H), Sense::click_and_drag());
    let (lo, hi) = (*range.start(), *range.end());
    let track = Rect::from_min_max(pos2(rect.left() + 6.0, rect.center().y - 1.0), pos2(rect.right() - 40.0, rect.center().y + 1.0));
    let enabled = ui.is_enabled();
    let x_of = |v: i32| egui::remap_clamp(v as f32, lo as f32..=hi as f32, track.x_range());
    if enabled {
        if resp.double_clicked() {
            *value = default;
            resp.mark_changed();
        } else if let Some(p) = resp.interact_pointer_pos().filter(|_| resp.dragged() || resp.clicked()) {
            let v = egui::remap_clamp(p.x, track.x_range(), lo as f32..=hi as f32).round() as i32;
            if v != *value {
                *value = v;
                resp.mark_changed();
            }
        }
    }
    let p = ui.painter();
    p.rect_filled(track, 1.0, if enabled { Color32::from_rgb(0x54, 0x54, 0x54) } else { RAISED });
    let zero = x_of(0.clamp(lo, hi));
    let kx = x_of(*value);
    let fill = Rect::from_min_max(pos2(zero.min(kx), track.top()), pos2(zero.max(kx), track.bottom()));
    p.rect_filled(fill, 1.0, if enabled { TEXT } else { DISABLED });
    let hot = resp.hovered() || resp.dragged();
    p.circle_filled(pos2(kx, track.center().y), 6.0, theme::PANEL);
    p.circle_stroke(pos2(kx, track.center().y), if hot { 6.0 } else { 5.0 }, Stroke::new(1.8, if enabled { TEXT } else { DISABLED }));
    p.text(pos2(rect.right(), rect.center().y), Align2::RIGHT_CENTER, value.to_string(), theme::body(13.0), if enabled { TEXT } else { DISABLED });
    resp
}

/// Float slider shown as an integer percentage-like value with `scale`.
pub fn slider_f64(ui: &mut Ui, value: &mut f64, range: RangeInclusive<f64>, scale: f64, default: f64) -> Response {
    let mut v = (*value * scale).round() as i32;
    let r = slider(ui, &mut v, ((range.start() * scale).round() as i32)..=((range.end() * scale).round() as i32), (default * scale).round() as i32);
    if r.changed() {
        *value = v as f64 / scale;
    }
    r
}

/// A pane row: label column on the left, the control filling the rest.
pub fn row<R>(ui: &mut Ui, label: &str, tooltip: (&str, &str), add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 0.0;
        ui.add_space(LEFT);
        let (rect, resp) = ui.allocate_exact_size(vec2(LABEL_W, ROW_H), Sense::hover());
        let color = if !ui.is_enabled() { DISABLED } else if resp.hovered() { TEXT } else { LABEL };
        ui.painter().text(rect.left_center(), Align2::LEFT_CENTER, label, theme::body(13.0), color);
        tip(resp, tooltip.0, tooltip.1);
        let w = (ui.available_width() - RIGHT).max(40.0);
        ui.allocate_ui_with_layout(vec2(w, ROW_H), Layout::left_to_right(Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 8.0;
            add(ui)
        })
        .inner
    })
    .inner
}

/// A label on its own line above a full-width control.
pub fn field_label(ui: &mut Ui, text: &str, tooltip: (&str, &str)) {
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.add_space(LEFT);
        let galley = ui.painter().layout_no_wrap(text.to_string(), theme::body(13.0), LABEL);
        let (rect, resp) = ui.allocate_exact_size(galley.size() + vec2(0.0, 4.0), Sense::hover());
        let color = if !ui.is_enabled() { DISABLED } else if resp.hovered() { TEXT } else { LABEL };
        ui.painter().galley(rect.min + vec2(0.0, 2.0), galley, color);
        tip(resp, tooltip.0, tooltip.1);
    });
}

/// Runs `add` in a full-width area inside the pane margins.
pub fn full<R>(ui: &mut Ui, add: impl FnOnce(&mut Ui) -> R) -> R {
    ui.horizontal(|ui| {
        ui.add_space(LEFT);
        let w = (ui.available_width() - RIGHT).max(40.0);
        ui.allocate_ui_with_layout(vec2(w, 28.0), Layout::top_down(Align::Min), |ui| {
            ui.set_width(w);
            add(ui)
        })
        .inner
    })
    .inner
}

/// A dropdown filling the available width.
pub fn combo<R>(ui: &mut Ui, id: impl egui::AsIdSalt, text: impl Into<String>, add: impl FnOnce(&mut Ui) -> R) -> Option<R> {
    let w = ui.available_width();
    egui::ComboBox::from_id_salt(id)
        .width(w)
        .selected_text(text.into())
        .icon(|ui, rect, _v, _open| {
            let r = Rect::from_center_size(pos2(rect.right() - 2.0, rect.center().y), Vec2::splat(12.0));
            icons::paint(ui.painter(), r, Icon::ChevronDown, MUTED);
        })
        .show_ui(ui, |ui| {
            ui.set_min_width(w - 8.0);
            add(ui)
        })
        .inner
}

/// The header of a filter section: chevron, title and an on/off switch (or a lock).
/// Returns true when the active state changed.
pub fn section_header(ui: &mut Ui, title: &str, tooltip: (&str, &str), expanded: &mut bool, active: Option<&mut bool>, locked: bool) -> bool {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 48.0), Sense::hover());
    let mut changed = false;
    let title_galley = ui.painter().layout_no_wrap(title.to_string(), theme::body(13.5), TEXT);
    let title_rect = Rect::from_min_max(pos2(rect.left() + 12.0, rect.top()), pos2(rect.left() + 34.0 + title_galley.size().x + 6.0, rect.bottom()));
    let title_resp = tip(ui.interact(title_rect, ui.scope_id().with((title, "title")), Sense::click()), tooltip.0, tooltip.1);
    let filler = Rect::from_min_max(pos2(title_rect.right(), rect.top()), pos2(rect.right() - 56.0, rect.bottom()));
    let filler_resp = ui.interact(filler, ui.scope_id().with((title, "fill")), Sense::click());
    let hot = title_resp.hovered() || filler_resp.hovered();
    let chev = if *expanded { Icon::ChevronDown } else { Icon::ChevronRight };
    icons::paint_at(ui.painter(), pos2(rect.left() + 22.0, rect.center().y), 11.0, chev, if hot { Color32::WHITE } else { LABEL });
    let enabled = ui.is_enabled();
    ui.painter().galley(pos2(rect.left() + 34.0, rect.center().y - title_galley.size().y / 2.0), title_galley, if enabled { TEXT } else { DISABLED });
    if title_resp.clicked() {
        *expanded = !*expanded;
    }
    let switch_rect = Rect::from_center_size(pos2(rect.right() - RIGHT - 15.0, rect.center().y), vec2(30.0, 16.0));
    if locked {
        icons::paint_at(ui.painter(), switch_rect.right_center() - vec2(7.0, 0.0), 13.0, Icon::Lock, LABEL);
    } else if let Some(on) = active {
        if filler_resp.clicked() {
            *expanded = !*expanded;
            if !*on {
                *on = true;
                *expanded = true;
                changed = true;
            }
        }
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(switch_rect));
        if switch(&mut child, on).changed() {
            changed = true;
            if *on {
                *expanded = true;
            }
        }
    } else if filler_resp.clicked() {
        *expanded = !*expanded;
    }
    changed
}

/// A horizontal 1 px divider across the available width.
pub fn divider(ui: &mut Ui) {
    let w = ui.available_width();
    let (rect, _) = ui.allocate_exact_size(vec2(w, 1.0), Sense::hover());
    ui.painter().rect_filled(rect, 0.0, theme::RAISED);
}

/// A centred dark label over the video ("Unprocessed").
pub fn pill(p: &egui::Painter, centre: Pos2, text: &str) {
    let galley = p.layout_no_wrap(text.to_string(), theme::semibold(13.0), TEXT);
    let r = Rect::from_center_size(centre, galley.size() + vec2(36.0, 18.0));
    p.rect_filled(r, CornerRadius::same(3), Color32::from_rgba_unmultiplied(0x21, 0x21, 0x21, 235));
    p.galley(r.center() - galley.size() / 2.0, galley, TEXT);
}

/// A small numeric badge (counts on bottom bar buttons).
pub fn count_badge(p: &egui::Painter, centre: Pos2, n: usize) {
    let text = n.to_string();
    let galley = p.layout_no_wrap(text, theme::semibold(10.0), ON_ACCENT);
    let r = Rect::from_center_size(centre, vec2((galley.size().x + 8.0).max(15.0), 15.0));
    p.rect_filled(r, 7.5, ACCENT);
    p.galley(r.center() - galley.size() / 2.0, galley, ON_ACCENT);
}
