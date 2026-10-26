//! Output naming: `<dir>/<input stem>_<prefix>.<ext>`, `<stem>_<n>_<prefix>.<ext>` on collision,
//! and `<dir>/<stem>_<prefix>/%06d.<ext>` for image sequences.

use std::path::{Path, PathBuf};

use crate::settings::ExportSettings;

/// Digits used for image-sequence frame numbers.
pub const SEQUENCE_DIGITS: usize = 6;

pub fn output_path(input: &Path, export: &ExportSettings, prefix: &str, ext: &str, sequence: bool, exists: &dyn Fn(&Path) -> bool) -> PathBuf {
    if let Some(p) = &export.output_path {
        return if sequence { p.join(format!("%0{SEQUENCE_DIGITS}d.{ext}")) } else { p.clone() };
    }
    let dir = export.output_dir.clone().unwrap_or_else(|| input.parent().map(Path::to_path_buf).unwrap_or_default());
    let stem = input.file_stem().map(|s| s.to_string_lossy().to_string()).unwrap_or_else(|| "output".into());
    let prefix = if prefix.is_empty() { "prism" } else { prefix };
    let make = |n: Option<u32>| -> PathBuf {
        let name = match n {
            None => format!("{stem}_{prefix}"),
            Some(n) => format!("{stem}_{n}_{prefix}"),
        };
        if sequence { dir.join(name) } else { dir.join(format!("{name}.{ext}")) }
    };
    let mut candidate = make(None);
    if !export.overwrite {
        let mut n = 1;
        while exists(&candidate) || candidate == input {
            candidate = make(Some(n));
            n += 1;
        }
    }
    if sequence { candidate.join(format!("%0{SEQUENCE_DIGITS}d.{ext}")) } else { candidate }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collision_numbering() {
        let input = Path::new("C:/v/clip.mov");
        let export = ExportSettings::default();
        let taken = [PathBuf::from("C:/v/clip_rg3.mp4"), PathBuf::from("C:/v/clip_1_rg3.mp4")];
        let p = output_path(input, &export, "rg3", "mp4", false, &|p| taken.iter().any(|t| t == p));
        assert_eq!(p, PathBuf::from("C:/v/clip_2_rg3.mp4"));
    }

    #[test]
    fn sequences_get_a_folder_and_pattern() {
        let mut export = ExportSettings::default();
        export.output_dir = Some(PathBuf::from("D:/out"));
        let p = output_path(Path::new("a/b.mp4"), &export, "", "png", true, &|_| false);
        assert_eq!(p, PathBuf::from("D:/out/b_prism").join("%06d.png"));
    }
}
