//! Installing network weights: downloading a catalog model's published files, or importing a
//! local file, into the models folder as `<id>.onnx` (plus `<id>.fp16.onnx` for converted
//! checkpoints).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use prism_core::models::{Catalog, Implementation, WeightsSource};
use prism_core::paths;

use crate::ai::convert;
use crate::process::command;

/// Removes a scratch folder when dropped, including after an error.
struct Scratch(PathBuf);

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scratch(id: &str) -> Result<Scratch> {
    let dir = paths::temp_root().join(format!("fetch-{id}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(Scratch(dir))
}

fn download(url: &str, dest: &Path) -> Result<()> {
    let curl = if cfg!(windows) { "curl.exe" } else { "curl" };
    let out = command(Path::new(curl))
        .args(["-sS", "-L", "--fail", "--retry", "2", "-o"])
        .arg(dest)
        .arg(url)
        .output()
        .context("running curl")?;
    if !out.status.success() {
        bail!("download of {url} failed: {}", String::from_utf8_lossy(&out.stderr).trim());
    }
    Ok(())
}

/// Archive extractors to try, in order. Windows 10 and later ship a libarchive `tar` in
/// System32 that reads 7z and zip; a GNU `tar` earlier on PATH (Git Bash) cannot.
fn extractors() -> Vec<(PathBuf, bool)> {
    let mut v = Vec::new();
    if cfg!(windows) {
        let root = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from("C:\\Windows"));
        v.push((root.join("System32").join("tar.exe"), false));
    }
    v.push((PathBuf::from("bsdtar"), false));
    v.push((PathBuf::from("7z"), true));
    v.push((PathBuf::from("7za"), true));
    v.push((PathBuf::from("tar"), false));
    v
}

/// Extracts `member` from `archive` into `out` and returns its path.
fn extract(archive: &Path, member: &str, out: &Path) -> Result<PathBuf> {
    std::fs::create_dir_all(out)?;
    let target = out.join(member);
    let mut tried = Vec::new();
    for (tool, seven_zip) in extractors() {
        let mut c = command(&tool);
        if seven_zip {
            c.arg("x").arg(archive).arg(format!("-o{}", out.display())).arg(member).arg("-y");
        } else {
            c.arg("-xf").arg(archive).arg("-C").arg(out).arg(member);
        }
        match c.output() {
            Ok(o) if o.status.success() && target.is_file() => return Ok(target),
            Ok(o) => tried.push(format!("{}: {}", tool.display(), String::from_utf8_lossy(&o.stderr).trim())),
            Err(e) => tried.push(format!("{}: {e}", tool.display())),
        }
    }
    bail!("could not extract {member} from {}; install 7-Zip or bsdtar. Tried:\n  {}", archive.display(), tried.join("\n  "))
}

/// Copies a ready ONNX file into `dir` as `<id>.onnx`, through a temporary name so a partial
/// copy never looks installed.
fn install_onnx(src: &Path, dir: &Path, id: &str) -> Result<PathBuf> {
    let mut head = [0u8; 1];
    let len = std::fs::metadata(src).with_context(|| format!("reading {}", src.display()))?.len();
    std::io::Read::read_exact(&mut std::fs::File::open(src)?, &mut head)?;
    // An ONNX ModelProto starts with field 1 (ir_version) as a varint: tag byte 0x08.
    if len < 1024 || head[0] != 0x08 {
        bail!("{} is not an ONNX model", src.display());
    }
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    let dest = dir.join(format!("{id}.onnx"));
    let part = dir.join(format!("{id}.onnx.part"));
    std::fs::copy(src, &part).with_context(|| format!("copying to {}", part.display()))?;
    std::fs::rename(&part, &dest).with_context(|| format!("installing {}", dest.display()))?;
    Ok(dest)
}

/// Installs a local file for model `id`: an `.onnx` file is copied as is; a PyTorch checkpoint
/// is converted (FP32, plus FP16 unless `fp32_only`). Returns the installed files.
pub fn install_file(src: &Path, dir: &Path, id: &str, fp32_only: bool) -> Result<Vec<PathBuf>> {
    let ext = src.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
    if ext == "onnx" {
        return Ok(vec![install_onnx(src, dir, id)?]);
    }
    let mut out = vec![convert::convert(src, dir, id, false)?.path];
    if !fp32_only {
        out.push(convert::convert(src, dir, id, true)?.path);
    }
    Ok(out)
}

/// Downloads catalog model `id` from its published source and installs it. `log` receives
/// progress lines.
pub fn fetch(catalog: &Catalog, id: &str, fp32_only: bool, log: &dyn Fn(&str)) -> Result<Vec<PathBuf>> {
    let m = catalog.get(id).ok_or_else(|| anyhow!("unknown model '{id}'"))?;
    let Implementation::Onnx { source: Some(source), license, .. } = &m.implementation else {
        bail!("'{id}' has no downloadable weights");
    };
    let tmp = scratch(id)?;
    log(&format!("Downloading {}\n  license: {license}", source.url()));
    match source {
        WeightsSource::Checkpoint { url } => {
            let file = tmp.0.join(format!("{id}.pth"));
            download(url, &file)?;
            install_file(&file, &catalog.dir, id, fp32_only)
        }
        WeightsSource::OnnxInArchive { url, member } => {
            let name = url.rsplit('/').next().filter(|n| !n.is_empty()).unwrap_or("archive");
            let archive = tmp.0.join(name);
            download(url, &archive)?;
            log(&format!("Extracting {member}"));
            let onnx = extract(&archive, member, &tmp.0.join("x"))?;
            Ok(vec![install_onnx(&onnx, &catalog.dir, id)?])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dir(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("prism-fetch-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn installs_onnx_files_whole_and_rejects_other_files() {
        let src = dir("src");
        let models = dir("models");
        // Minimal bytes that look like a ModelProto: field 1 (ir_version = 8), then padding.
        let mut onnx = vec![0x08, 0x08];
        onnx.resize(4096, 0);
        std::fs::write(src.join("net.onnx"), &onnx).unwrap();
        let out = install_file(&src.join("net.onnx"), &models, "rife", false).unwrap();
        assert_eq!(out, vec![models.join("rife.onnx")]);
        assert_eq!(std::fs::read(models.join("rife.onnx")).unwrap(), onnx);
        assert!(!models.join("rife.onnx.part").exists());

        std::fs::write(src.join("page.onnx"), "<html>not found</html>".repeat(100)).unwrap();
        assert!(install_file(&src.join("page.onnx"), &models, "other", false).is_err());
        assert!(!models.join("other.onnx").exists() && !models.join("other.onnx.part").exists());
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&models);
    }
}
