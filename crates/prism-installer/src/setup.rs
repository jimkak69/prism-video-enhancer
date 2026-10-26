//! Install, uninstall and packaging steps, without UI.
//!
//! Installs per user (no administrator rights): files under
//! `%LOCALAPPDATA%\Programs\Prism Video Enhancer`, Start menu and desktop shortcuts, an entry in
//! Settings > Apps (HKCU), and optionally the folder on the user PATH. `install.json` records
//! what was created so the uninstaller removes exactly that.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};

pub const APP_NAME: &str = "Prism Video Enhancer";
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
const REG_KEY: &str = r"HKCU\Software\Microsoft\Windows\CurrentVersion\Uninstall\PrismVideoEnhancer";
const MANIFEST: &str = "install.json";
const UNINSTALLER: &str = "uninstall.exe";
const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const DETACHED_PROCESS: u32 = 0x0000_0008;

#[derive(Debug, Clone)]
pub struct Options {
    pub dir: PathBuf,
    pub desktop_shortcut: bool,
    pub start_menu: bool,
    pub add_to_path: bool,
}

impl Default for Options {
    fn default() -> Self {
        Options { dir: default_dir(), desktop_shortcut: true, start_menu: true, add_to_path: false }
    }
}

/// What an installation created, kept next to the uninstaller.
#[derive(Debug, Default, Serialize, Deserialize)]
struct Manifest {
    version: String,
    /// Files relative to the install folder.
    files: Vec<String>,
    shortcuts: Vec<PathBuf>,
    /// Exact text appended to the user PATH (with its separator), if setup added it.
    path_entry: Option<String>,
}

pub fn default_dir() -> PathBuf {
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    local.join("Programs").join(APP_NAME)
}

/// The folder the running uninstaller lives in.
pub fn installed_dir() -> PathBuf {
    std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf)).unwrap_or_else(default_dir)
}

/// Version recorded by an existing installation in `dir`.
pub fn existing_version(dir: &Path) -> Option<String> {
    let text = fs::read_to_string(dir.join(MANIFEST)).ok()?;
    serde_json::from_str::<Manifest>(&text).ok().map(|m| m.version)
}

fn no_window(c: &mut Command) -> &mut Command {
    c.creation_flags(CREATE_NO_WINDOW)
}

/// Runs a PowerShell script and returns its standard output.
fn powershell(script: &str) -> Result<String> {
    let out = no_window(Command::new("powershell.exe").args(["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", script]))
        .output()
        .context("running PowerShell")?;
    if !out.status.success() {
        bail!("PowerShell failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// A PowerShell single-quoted literal.
fn q(s: impl AsRef<str>) -> String {
    format!("'{}'", s.as_ref().replace('\'', "''"))
}

fn reg(args: &[&str]) -> Result<()> {
    let out = no_window(Command::new("reg.exe").args(args)).output().context("running reg.exe")?;
    if !out.status.success() {
        bail!("reg.exe {}: {}", args.first().unwrap_or(&""), String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

/// FFmpeg on PATH or in a winget install, as Prism looks for it.
pub fn find_ffmpeg() -> Option<PathBuf> {
    if let Some(path) = std::env::var_os("PATH") {
        if let Some(p) = std::env::split_paths(&path).map(|d| d.join("ffmpeg.exe")).find(|p| p.is_file()) {
            return Some(p);
        }
    }
    let local = PathBuf::from(std::env::var_os("LOCALAPPDATA")?);
    for pkg in fs::read_dir(local.join("Microsoft").join("WinGet").join("Packages")).ok()?.flatten() {
        if pkg.file_name().to_string_lossy().contains("FFmpeg") {
            for d in fs::read_dir(pkg.path()).ok()?.flatten() {
                let p = d.path().join("bin").join("ffmpeg.exe");
                if p.is_file() {
                    return Some(p);
                }
            }
        }
    }
    None
}

/// Where the appended archive begins: the setup program is everything before it. The writer
/// records absolute offsets, so this is the first local header, not `ZipArchive::offset`.
fn payload_start(zip: &mut zip::ZipArchive<File>) -> Result<u64> {
    let mut start = u64::MAX;
    for i in 0..zip.len() {
        start = start.min(zip.by_index_raw(i)?.header_start());
    }
    if start == u64::MAX || start == 0 {
        bail!("the setup program has no payload before its archive");
    }
    Ok(start)
}

fn open_payload() -> Result<zip::ZipArchive<File>> {
    let exe = std::env::current_exe().context("locating the setup program")?;
    let f = File::open(&exe).with_context(|| format!("opening {}", exe.display()))?;
    zip::ZipArchive::new(f).map_err(|_| anyhow!("this setup program carries no application files (build it with --make-installer)"))
}

/// Fails when an installed program is running: Windows will not replace a running executable.
fn ensure_not_running(dir: &Path) -> Result<()> {
    for name in ["prism-app.exe", "prism.exe"] {
        let p = dir.join(name);
        if p.is_file() {
            if let Err(e) = OpenOptions::new().write(true).open(&p) {
                if e.raw_os_error() == Some(32) {
                    bail!("{APP_NAME} is running ({name}). Close it and try again.");
                }
            }
        }
    }
    Ok(())
}

fn create_shortcut(lnk: &Path, target: &Path, icon: &Path) -> Result<()> {
    if let Some(d) = lnk.parent() {
        fs::create_dir_all(d)?;
    }
    let script = format!(
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut({}); $s.TargetPath = {}; $s.WorkingDirectory = {}; $s.IconLocation = {}; $s.Description = {}; $s.Save()",
        q(lnk.to_string_lossy()),
        q(target.to_string_lossy()),
        q(target.parent().unwrap_or(Path::new(".")).to_string_lossy()),
        q(format!("{},0", icon.to_string_lossy())),
        q("Upscale, restore and interpolate video"),
    );
    powershell(&script).map(|_| ())
}

fn start_menu_dir() -> Result<PathBuf> {
    let appdata = std::env::var_os("APPDATA").ok_or_else(|| anyhow!("APPDATA is not set"))?;
    Ok(PathBuf::from(appdata).join(r"Microsoft\Windows\Start Menu\Programs"))
}

/// The user PATH as stored: its registry type and unexpanded text (`%USERPROFILE%` and similar
/// stay as written).
fn read_user_path() -> Result<(String, String)> {
    let out = powershell(concat!(
        r"$k = Get-Item 'HKCU:\Environment'; $v = $k.GetValue('Path', '', 'DoNotExpandEnvironmentNames'); ",
        r"$t = if ($k.GetValueNames() -contains 'Path') { $k.GetValueKind('Path') } else { 'ExpandString' }; ",
        r"Write-Output $t; Write-Output $v",
    ))?;
    let mut lines = out.lines();
    let kind = lines.next().unwrap_or("ExpandString").trim().to_string();
    Ok((kind, lines.next().unwrap_or("").to_string()))
}

/// Writes the user PATH with the given registry type and tells running programs.
fn write_user_path(kind: &str, value: &str) -> Result<()> {
    let kind = if kind == "String" { "String" } else { "ExpandString" };
    // WM_SETTINGCHANGE ("Environment") lets Explorer and new terminals pick up the change.
    let script = format!(
        concat!(
            r"Set-ItemProperty -Path 'HKCU:\Environment' -Name Path -Value {} -Type {}; ",
            r#"Add-Type -Namespace PrismSetup -Name Env -MemberDefinition '[DllImport("user32.dll", CharSet=CharSet.Unicode)] "#,
            r"public static extern IntPtr SendMessageTimeout(IntPtr h, uint m, UIntPtr w, string l, uint f, uint t, out UIntPtr r);'; ",
            r"$r = [UIntPtr]::Zero; [void][PrismSetup.Env]::SendMessageTimeout([IntPtr]0xffff, 0x1A, [UIntPtr]::Zero, 'Environment', 2, 5000, [ref]$r)",
        ),
        q(value),
        kind
    );
    powershell(&script).map(|_| ())
}

fn path_has(value: &str, dir: &str) -> bool {
    value.split(';').any(|p| p.eq_ignore_ascii_case(dir))
}

/// Appends `dir` to the user PATH unless present; returns the exact text appended.
fn add_to_user_path(dir: &str) -> Result<Option<String>> {
    let (kind, current) = read_user_path()?;
    if path_has(&current, dir) {
        return Ok(None);
    }
    let added = if current.is_empty() || current.ends_with(';') { dir.to_string() } else { format!(";{dir}") };
    write_user_path(&kind, &format!("{current}{added}"))?;
    Ok(Some(added))
}

/// Removes the text setup appended, leaving the rest of PATH byte for byte as it is.
fn remove_from_user_path(added: &str, dir: &str) -> Result<()> {
    let (kind, current) = read_user_path()?;
    // Only a whole-entry match counts: never part of a longer folder name.
    let whole = |i: usize| {
        let end = i + added.len();
        let after_ok = end == current.len() || current[end..].starts_with(';');
        let before_ok = added.starts_with(';') || i == 0 || current[..i].ends_with(';');
        after_ok && before_ok
    };
    let new = match current.rmatch_indices(added).map(|(i, _)| i).find(|&i| whole(i)) {
        Some(i) => format!("{}{}", &current[..i], &current[i + added.len()..]),
        None => current.split(';').filter(|p| !p.eq_ignore_ascii_case(dir)).collect::<Vec<_>>().join(";"),
    };
    if new != current {
        write_user_path(&kind, &new)?;
    }
    Ok(())
}

fn dir_size_kb(dir: &Path) -> u64 {
    fs::read_dir(dir).map(|rd| rd.flatten().map(|e| if e.path().is_dir() { dir_size_kb(&e.path()) * 1024 } else { e.metadata().map(|m| m.len()).unwrap_or(0) }).sum::<u64>() / 1024).unwrap_or(0)
}

/// Installs the payload into `o.dir`. `progress(done, total)` is called with byte counts.
/// Returns the path of the installed app.
pub fn install(o: &Options, progress: &(dyn Fn(u64, u64) + Sync)) -> Result<PathBuf> {
    let mut zip = open_payload()?;
    let dir = &o.dir;
    ensure_not_running(dir)?;
    fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let total: u64 = (0..zip.len()).filter_map(|i| zip.by_index_raw(i).ok().map(|f| f.size())).sum();
    let mut done = 0u64;
    let previous: Option<Manifest> = fs::read_to_string(dir.join(MANIFEST)).ok().and_then(|t| serde_json::from_str(&t).ok());
    let mut manifest = Manifest { version: VERSION.into(), ..Default::default() };
    for i in 0..zip.len() {
        let mut entry = zip.by_index(i)?;
        let Some(rel) = entry.enclosed_name() else { bail!("unsafe path in payload: {}", entry.name()) };
        let dest = dir.join(&rel);
        if entry.is_dir() {
            fs::create_dir_all(&dest)?;
            continue;
        }
        if let Some(parent) = dest.parent() {
            fs::create_dir_all(parent)?;
        }
        // Write beside the target and swap in, so an interrupted install never leaves a
        // truncated executable.
        let part = dest.with_extension("part");
        {
            let mut out = File::create(&part).with_context(|| format!("writing {}", part.display()))?;
            let mut buf = vec![0u8; 1 << 20];
            loop {
                let n = entry.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                out.write_all(&buf[..n])?;
                done += n as u64;
                progress(done, total);
            }
        }
        if dest.exists() {
            fs::remove_file(&dest).with_context(|| format!("replacing {}", dest.display()))?;
        }
        fs::rename(&part, &dest)?;
        manifest.files.push(rel.to_string_lossy().replace('\\', "/"));
    }
    // The uninstaller is this program without the payload.
    let offset = payload_start(&mut zip)?;
    drop(zip);
    {
        let mut src = File::open(std::env::current_exe()?)?;
        let mut stub = vec![0u8; offset as usize];
        src.read_exact(&mut stub)?;
        let un = dir.join(UNINSTALLER);
        if un.exists() && un != std::env::current_exe()? {
            fs::remove_file(&un)?;
        }
        fs::write(&un, stub)?;
    }
    let app = dir.join("prism-app.exe");
    let icon = dir.join("prism.ico");
    if o.start_menu {
        let lnk = start_menu_dir()?.join(format!("{APP_NAME}.lnk"));
        create_shortcut(&lnk, &app, &icon)?;
        manifest.shortcuts.push(lnk);
    }
    if o.desktop_shortcut {
        let desktop = PathBuf::from(powershell("[Environment]::GetFolderPath('Desktop')")?);
        let lnk = desktop.join(format!("{APP_NAME}.lnk"));
        create_shortcut(&lnk, &app, &icon)?;
        manifest.shortcuts.push(lnk);
    }
    // PATH: an update keeps ownership of an entry an earlier install added; unticking the
    // option on update removes that entry.
    let d = dir.to_string_lossy().to_string();
    let earlier = previous.as_ref().and_then(|p| p.path_entry.clone());
    if o.add_to_path {
        manifest.path_entry = add_to_user_path(&d)?.or(earlier);
    } else if let Some(entry) = earlier {
        remove_from_user_path(&entry, &d)?;
    }
    let un = dir.join(UNINSTALLER).to_string_lossy().to_string();
    let size = dir_size_kb(dir).to_string();
    let dir_s = dir.to_string_lossy().to_string();
    let icon_s = icon.to_string_lossy().to_string();
    for (name, kind, value) in [
        ("DisplayName", "REG_SZ", APP_NAME.to_string()),
        ("DisplayVersion", "REG_SZ", VERSION.to_string()),
        ("Publisher", "REG_SZ", APP_NAME.to_string()),
        ("InstallLocation", "REG_SZ", dir_s),
        ("DisplayIcon", "REG_SZ", icon_s),
        ("UninstallString", "REG_SZ", format!("\"{un}\" --uninstall")),
        ("QuietUninstallString", "REG_SZ", format!("\"{un}\" --uninstall --silent")),
        ("NoModify", "REG_DWORD", "1".into()),
        ("NoRepair", "REG_DWORD", "1".into()),
        ("EstimatedSize", "REG_DWORD", size),
    ] {
        reg(&["add", REG_KEY, "/v", name, "/t", kind, "/d", &value, "/f"])?;
    }
    fs::write(dir.join(MANIFEST), serde_json::to_vec_pretty(&manifest)?)?;
    Ok(app)
}

/// Removes an installation from `dir`. `remove_data` also deletes settings, projects, presets
/// and downloaded models.
pub fn uninstall(dir: &Path, remove_data: bool) -> Result<()> {
    ensure_not_running(dir)?;
    let manifest: Manifest = fs::read_to_string(dir.join(MANIFEST)).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    if manifest.version.is_empty() && !dir.join("prism-app.exe").exists() {
        bail!("no {APP_NAME} installation found in {}", dir.display());
    }
    for lnk in &manifest.shortcuts {
        let _ = fs::remove_file(lnk);
    }
    let _ = reg(&["delete", REG_KEY, "/f"]);
    if let Some(entry) = &manifest.path_entry {
        let _ = remove_from_user_path(entry, &dir.to_string_lossy());
    }
    let mut files = manifest.files.clone();
    if files.is_empty() {
        files = ["prism-app.exe", "prism.exe", "prism.ico", "README.md"].map(String::from).to_vec();
    }
    for f in &files {
        let _ = fs::remove_file(dir.join(f));
    }
    let _ = fs::remove_file(dir.join(MANIFEST));
    if remove_data {
        let _ = fs::remove_dir_all(prism_core::paths::data_dir());
        let _ = fs::remove_dir_all(prism_core::paths::temp_root());
    }
    let me = std::env::current_exe().ok();
    if me.as_ref().is_some_and(|m| m.starts_with(dir)) {
        // A running program cannot delete itself; a detached shell removes the folder once
        // this process has exited.
        let cmd = format!("ping 127.0.0.1 -n 3 > nul & rd /s /q \"{}\"", dir.display());
        Command::new("cmd.exe").raw_arg("/c").raw_arg(&cmd).creation_flags(CREATE_NO_WINDOW | DETACHED_PROCESS).spawn().context("scheduling removal")?;
    } else {
        let _ = fs::remove_file(dir.join(UNINSTALLER));
        let _ = fs::remove_dir(dir);
    }
    Ok(())
}

fn add_dir(zw: &mut zip::ZipWriter<File>, root: &Path, dir: &Path, count: &mut usize) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)?.flatten().collect();
    entries.sort_by_key(|e| e.file_name());
    for e in entries {
        let p = e.path();
        let rel = p.strip_prefix(root)?.to_string_lossy().replace('\\', "/");
        if p.is_dir() {
            zw.add_directory(rel, zip::write::SimpleFileOptions::default())?;
            add_dir(zw, root, &p, count)?;
        } else {
            let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated).large_file(fs::metadata(&p)?.len() >= u32::MAX as u64);
            zw.start_file(rel, opts)?;
            std::io::copy(&mut File::open(&p)?, zw)?;
            *count += 1;
        }
    }
    Ok(())
}

/// Writes `out`: this program followed by a zip of `payload`. Returns the number of files.
pub fn make_installer(payload: &Path, out: &Path) -> Result<usize> {
    if !payload.join("prism-app.exe").is_file() {
        bail!("{} has no prism-app.exe", payload.display());
    }
    let me = std::env::current_exe()?;
    let mut stub = fs::read(&me)?;
    // Building from an installer that already carries a payload: keep only the program.
    if let Ok(mut z) = zip::ZipArchive::new(File::open(&me)?) {
        stub.truncate(payload_start(&mut z)? as usize);
    }
    let mut f = File::create(out).with_context(|| format!("creating {}", out.display()))?;
    f.write_all(&stub)?;
    f.seek(SeekFrom::End(0))?;
    let mut zw = zip::ZipWriter::new(f);
    let mut count = 0;
    add_dir(&mut zw, payload, payload, &mut count)?;
    zw.finish()?;
    Ok(count)
}
