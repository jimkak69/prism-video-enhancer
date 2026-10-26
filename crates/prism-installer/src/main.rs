//! Prism Setup: a per-user Windows installer that carries the app as a zip archive appended
//! to its own executable.
//!
//! prism-setup                                        install with a window
//! prism-setup --silent [--dir D] [--no-desktop] [--add-to-path]
//! prism-setup --uninstall [--silent] [--remove-data]  (run as the installed uninstall.exe)
//! prism-setup --make-installer <payload dir> <out.exe>  (packaging: append the payload)
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[cfg(windows)]
mod setup;
#[cfg(windows)]
mod ui;

#[cfg(windows)]
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let has = |f: &str| args.iter().any(|a| a == f);
    let value = |f: &str| args.iter().position(|a| a == f).and_then(|i| args.get(i + 1)).cloned();
    let result = if has("--make-installer") {
        let i = args.iter().position(|a| a == "--make-installer").unwrap();
        match (args.get(i + 1), args.get(i + 2)) {
            (Some(payload), Some(out)) => setup::make_installer(payload.as_ref(), out.as_ref()).map(|n| println!("wrote {out} ({n} files)")),
            _ => Err(anyhow::anyhow!("usage: prism-setup --make-installer <payload dir> <out.exe>")),
        }
    } else if has("--uninstall") {
        if has("--silent") {
            setup::uninstall(&setup::installed_dir(), has("--remove-data"))
        } else {
            ui::run(ui::Mode::Uninstall)
        }
    } else if has("--silent") {
        let mut o = setup::Options::default();
        if let Some(d) = value("--dir") {
            o.dir = d.into();
        }
        o.desktop_shortcut = !has("--no-desktop");
        o.start_menu = !has("--no-start-menu");
        o.add_to_path = has("--add-to-path");
        setup::install(&o, &|_, _| {}).map(|_| ())
    } else {
        ui::run(ui::Mode::Install)
    };
    if let Err(e) = result {
        // Silent and packaging modes report on stderr; the window shows its own errors.
        eprintln!("prism-setup: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("prism-setup is the Windows installer. On Linux use the .deb or tar.gz package; on macOS the .dmg.");
    std::process::exit(1);
}
