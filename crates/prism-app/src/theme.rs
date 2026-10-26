//! Palette, fonts and base style. Neutral greys with a green accent.

use std::sync::Arc;

use egui::{Color32, CornerRadius, FontFamily, FontId, Stroke, TextStyle, Visuals};

/// Window and viewer background.
pub const BG: Color32 = Color32::from_rgb(0x17, 0x17, 0x17);
/// Panels (header, right pane, bars).
pub const PANEL: Color32 = Color32::from_rgb(0x21, 0x21, 0x21);
/// Inputs, segmented controls, dividers.
pub const RAISED: Color32 = Color32::from_rgb(0x2B, 0x2B, 0x2B);
/// Menu row, hovered controls.
pub const HOVER: Color32 = Color32::from_rgb(0x35, 0x35, 0x35);
/// Selected segment, slider track.
pub const SELECTED: Color32 = Color32::from_rgb(0x48, 0x48, 0x48);
pub const BORDER: Color32 = Color32::from_rgb(0x2B, 0x2B, 0x2B);
pub const TEXT: Color32 = Color32::from_rgb(0xF2, 0xF2, 0xF2);
pub const LABEL: Color32 = Color32::from_rgb(0xB0, 0xB0, 0xB0);
pub const MUTED: Color32 = Color32::from_rgb(0x97, 0x97, 0x97);
pub const DISABLED: Color32 = Color32::from_rgb(0x54, 0x54, 0x54);
/// Primary accent: actions, selection, playhead.
pub const ACCENT: Color32 = Color32::from_rgb(0x2E, 0xCC, 0x71);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(0x45, 0xDB, 0x86);
pub const ACCENT_DIM: Color32 = Color32::from_rgb(0x1E, 0x6B, 0x41);
/// Text drawn on the accent.
pub const ON_ACCENT: Color32 = Color32::from_rgb(0x08, 0x1A, 0x0F);
pub const ERROR: Color32 = Color32::from_rgb(0xFF, 0x4D, 0x6A);
pub const WARNING: Color32 = Color32::from_rgb(0xFF, 0xCE, 0x21);

pub const SEMIBOLD: &str = "semibold";

pub fn semibold(size: f32) -> FontId {
    FontId::new(size, FontFamily::Name(SEMIBOLD.into()))
}

pub fn body(size: f32) -> FontId {
    FontId::proportional(size)
}

/// System UI fonts tried in order, as (regular, semibold) paths. Fonts are read from the
/// system and not redistributed; egui's bundled fonts stay as fallbacks for other scripts.
fn font_candidates() -> Vec<(std::path::PathBuf, std::path::PathBuf)> {
    use std::path::PathBuf;
    if cfg!(windows) {
        let dir = std::env::var_os("WINDIR").map(PathBuf::from).unwrap_or_else(|| r"C:\Windows".into()).join("Fonts");
        vec![(dir.join("segoeui.ttf"), dir.join("seguisb.ttf"))]
    } else if cfg!(target_os = "macos") {
        let s = PathBuf::from("/System/Library/Fonts/Supplemental");
        vec![(s.join("Arial.ttf"), s.join("Arial Bold.ttf"))]
    } else {
        [
            ("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf", "/usr/share/fonts/truetype/noto/NotoSans-SemiBold.ttf"),
            ("/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/noto/NotoSans-SemiBold.ttf"),
            ("/usr/share/fonts/google-noto/NotoSans-Regular.ttf", "/usr/share/fonts/google-noto/NotoSans-SemiBold.ttf"),
            ("/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", "/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf"),
            ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            ("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"),
        ]
        .into_iter()
        .map(|(r, b)| (PathBuf::from(r), PathBuf::from(b)))
        .collect()
    }
}

fn fonts() -> egui::FontDefinitions {
    let mut f = egui::FontDefinitions::default();
    let mut proportional = f.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let mut bold = proportional.clone();
    if let Some((regular, semibold)) = font_candidates().into_iter().find(|(r, _)| r.is_file()) {
        if let Ok(bytes) = std::fs::read(&regular) {
            f.font_data.insert("system".into(), Arc::new(egui::FontData::from_owned(bytes)));
            proportional.insert(0, "system".into());
            bold.insert(0, "system".into());
        }
        if let Ok(bytes) = std::fs::read(&semibold) {
            f.font_data.insert("system-sb".into(), Arc::new(egui::FontData::from_owned(bytes)));
            bold.insert(0, "system-sb".into());
        }
    }
    f.families.insert(FontFamily::Proportional, proportional);
    f.families.insert(FontFamily::Name(SEMIBOLD.into()), bold);
    f
}

pub fn apply(ctx: &egui::Context) {
    ctx.set_fonts(fonts());
    let mut v = Visuals::dark();
    v.panel_fill = PANEL;
    v.window_fill = PANEL;
    v.extreme_bg_color = RAISED;
    v.faint_bg_color = RAISED;
    v.code_bg_color = BG;
    v.override_text_color = Some(TEXT);
    v.hyperlink_color = ACCENT;
    v.selection.bg_fill = ACCENT_DIM;
    v.selection.stroke = Stroke::new(1.0, ACCENT);
    v.window_stroke = Stroke::new(1.0, HOVER);
    v.window_corner_radius = CornerRadius::same(6);
    v.menu_corner_radius = CornerRadius::same(4);
    v.popup_shadow = egui::epaint::Shadow { offset: [0, 4], blur: 16, spread: 0, color: Color32::from_black_alpha(140) };
    v.window_shadow = egui::epaint::Shadow { offset: [0, 8], blur: 28, spread: 0, color: Color32::from_black_alpha(160) };
    v.widgets.noninteractive.bg_fill = PANEL;
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = Stroke::new(1.0, LABEL);
    for (w, fill) in [(&mut v.widgets.inactive, RAISED), (&mut v.widgets.hovered, HOVER), (&mut v.widgets.active, SELECTED), (&mut v.widgets.open, HOVER)] {
        w.bg_fill = fill;
        w.weak_bg_fill = fill;
        w.bg_stroke = Stroke::NONE;
        w.corner_radius = CornerRadius::same(4);
        w.expansion = 0.0;
    }
    v.widgets.inactive.fg_stroke = Stroke::new(1.0, TEXT);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.widgets.active.fg_stroke = Stroke::new(1.0, Color32::WHITE);
    v.slider_trailing_fill = true;
    ctx.set_visuals_of(egui::Theme::Dark, v);
    ctx.style_mut_of(egui::Theme::Dark, |s| {
        s.text_styles = [
            (TextStyle::Small, FontId::proportional(11.0)),
            (TextStyle::Body, FontId::proportional(13.0)),
            (TextStyle::Button, FontId::proportional(13.0)),
            (TextStyle::Heading, semibold(17.0)),
            (TextStyle::Monospace, FontId::monospace(12.0)),
        ]
        .into();
        s.spacing.item_spacing = egui::vec2(8.0, 6.0);
        s.spacing.button_padding = egui::vec2(10.0, 5.0);
        s.spacing.interact_size = egui::vec2(40.0, 26.0);
        s.spacing.menu_margin = egui::Margin::same(4);
        s.spacing.combo_height = 320.0;
        s.spacing.scroll = egui::style::ScrollStyle::thin();
        s.interaction.tooltip_delay = 0.35;
    });
    ctx.set_theme(egui::Theme::Dark);
}
