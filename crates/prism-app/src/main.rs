//! Prism Video Enhancer desktop app.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod appicon;
mod icons;
mod media;
mod panels;
mod prefs;
mod settings_ui;
mod theme;
mod widgets;

fn main() -> eframe::Result {
    let mut args = std::env::args_os().skip(1).peekable();
    // Packaging helper: `prism-app --write-icons <dir>` writes the icon files and exits.
    if args.peek().is_some_and(|a| a == "--write-icons") {
        let dir = std::path::PathBuf::from(args.nth(1).unwrap_or_else(|| ".".into()));
        if let Err(e) = appicon::write_icons(&dir) {
            eprintln!("could not write icons: {e}");
            std::process::exit(1);
        }
        return Ok(());
    }
    prism_engine::init_threads(0);
    let files: Vec<std::path::PathBuf> = args.map(Into::into).collect();
    let viewport = egui::ViewportBuilder::default()
        .with_title("Prism Video Enhancer")
        .with_app_id("prism-video-enhancer")
        .with_icon(appicon::icon_data(256))
        .with_inner_size([1440.0, 860.0])
        .with_min_inner_size([1100.0, 680.0])
        .with_drag_and_drop(true);
    // macOS keeps its native traffic-light buttons over a full-height content view; elsewhere
    // the app draws its own title bar (menu row and window buttons).
    let viewport = if cfg!(target_os = "macos") {
        viewport.with_fullsize_content_view(true).with_titlebar_shown(false).with_title_shown(false)
    } else {
        viewport.with_decorations(false)
    };
    let options = eframe::NativeOptions { viewport, centered: true, ..Default::default() };
    eframe::run_native("Prism Video Enhancer", options, Box::new(move |cc| Ok(Box::new(app::PrismApp::new(cc, files)))))
}
