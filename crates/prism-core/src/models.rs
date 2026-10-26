//! Model catalog.
//!
//! Same taxonomy as the reference product (enhancement, frame interpolation, parameter estimation,
//! camera pose, stabilization) but every entry is either a native Rust implementation, an FFmpeg
//! filter, or an openly licensed network that the user converts locally (`prism models import`).
//! Additional ONNX files dropped into the models directory are listed as custom models.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelKind {
    Enhancement,
    Interpolation,
    ParamEstimation,
    CameraPose,
    Stabilization,
    MotionDeblur,
    Hdr,
}

/// Network architecture of an ONNX model, which decides how weights are converted and how the
/// network is fed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetArch {
    /// Real-ESRGAN compact VGG-style network (`SRVGGNetCompact`).
    SrVggCompact,
    /// ESRGAN residual-in-residual dense blocks (`RRDBNet`).
    Rrdb,
    /// RIFE-style interpolation network taking two frames and a timestep.
    Rife,
    /// Any single-input image-to-image network.
    Generic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Implementation {
    /// Native Rust kernels; always available.
    Native,
    /// An ONNX network stored as `<models dir>/<id>.onnx` (or `<id>.fp16.onnx`).
    Onnx {
        arch: NetArch,
        /// Where openly licensed weights are published, if they are.
        source: Option<WeightsSource>,
        license: String,
    },
    /// An FFmpeg filter.
    Ffmpeg { filter: String },
}

/// Where a network's weights are published and how they are packaged.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightsSource {
    /// A PyTorch checkpoint, converted to ONNX (FP32 and FP16) on install.
    Checkpoint { url: String },
    /// A ready ONNX file inside an archive (7z, zip or tar); `member` is its path there.
    OnnxInArchive { url: String, member: String },
}

impl WeightsSource {
    pub fn url(&self) -> &str {
        match self {
            WeightsSource::Checkpoint { url } | WeightsSource::OnnxInArchive { url, .. } => url,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Content {
    Any,
    LiveAction,
    Animation,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Stable identifier used in settings and presets.
    pub id: String,
    pub name: String,
    pub description: String,
    pub kind: ModelKind,
    /// Native scales of the network. Empty means any scale (resampling based).
    pub scales: Vec<u32>,
    pub implementation: Implementation,
    /// Short tag appended to output file names.
    pub prefix: String,
    pub content: Content,
    /// Relative cost per output pixel (1 = Classic). Used for estimates and recommendations.
    pub cost: f32,
    /// Lower sorts first in pickers.
    pub display_priority: u32,
    pub custom: bool,
}

impl ModelInfo {
    pub fn is_onnx(&self) -> bool {
        matches!(self.implementation, Implementation::Onnx { .. })
    }

    pub fn arch(&self) -> Option<NetArch> {
        match self.implementation {
            Implementation::Onnx { arch, .. } => Some(arch),
            _ => None,
        }
    }

    /// Smallest native scale that reaches `needed`, or the largest available.
    pub fn pick_scale(&self, needed: f64) -> u32 {
        if self.scales.is_empty() {
            return needed.ceil().max(1.0) as u32;
        }
        let mut sorted = self.scales.clone();
        sorted.sort_unstable();
        sorted.iter().copied().find(|s| *s as f64 + 1e-6 >= needed).unwrap_or(*sorted.last().unwrap())
    }

    /// ONNX file for this model in `dir`, preferring FP16 when requested and present.
    pub fn onnx_file(&self, dir: &Path, prefer_fp16: bool) -> Option<PathBuf> {
        if !self.is_onnx() {
            return None;
        }
        let fp16 = dir.join(format!("{}.fp16.onnx", self.id));
        let fp32 = dir.join(format!("{}.onnx", self.id));
        let order = if prefer_fp16 { [fp16, fp32] } else { [fp32, fp16] };
        order.into_iter().find(|p| p.is_file())
    }

    pub fn is_available(&self, dir: &Path) -> bool {
        !self.is_onnx() || self.onnx_file(dir, true).is_some()
    }
}

/// Optional sidecar (`<name>.json`) describing a custom ONNX model.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct CustomModelSidecar {
    pub name: Option<String>,
    pub description: Option<String>,
    pub kind: Option<ModelKind>,
    pub scale: Option<u32>,
    pub arch: Option<NetArch>,
}

#[derive(Debug, Clone)]
pub struct Catalog {
    pub models: Vec<ModelInfo>,
    pub dir: PathBuf,
}

pub const DEFAULT_ENHANCEMENT: &str = "prism-classic";
pub const DEFAULT_INTERPOLATION: &str = "prism-flow";

impl Catalog {
    /// Built-in catalog plus custom ONNX files found in `dir`.
    pub fn load(dir: &Path) -> Self {
        let mut models = builtin();
        if let Ok(entries) = std::fs::read_dir(dir) {
            let mut files: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|e| e == "onnx")).collect();
            files.sort();
            for path in files {
                let stem = path.file_stem().unwrap().to_string_lossy().to_string();
                let id = stem.strip_suffix(".fp16").unwrap_or(&stem).to_string();
                if models.iter().any(|m| m.id == id) {
                    continue;
                }
                let sidecar: CustomModelSidecar = std::fs::read_to_string(dir.join(format!("{id}.json")))
                    .ok()
                    .and_then(|s| serde_json::from_str(&s).ok())
                    .unwrap_or_default();
                let kind = sidecar.kind.unwrap_or(ModelKind::Enhancement);
                models.push(ModelInfo {
                    name: sidecar.name.unwrap_or_else(|| id.clone()),
                    description: sidecar.description.unwrap_or_else(|| format!("Custom ONNX model ({})", path.display())),
                    kind,
                    scales: sidecar.scale.map(|s| vec![s]).unwrap_or_default(),
                    implementation: Implementation::Onnx {
                        arch: sidecar.arch.unwrap_or(if kind == ModelKind::Interpolation { NetArch::Rife } else { NetArch::Generic }),
                        source: None,
                        license: "user supplied".into(),
                    },
                    prefix: id.chars().filter(|c| c.is_ascii_alphanumeric()).take(8).collect(),
                    content: Content::Any,
                    cost: 20.0,
                    display_priority: 100,
                    custom: true,
                    id,
                });
            }
        }
        Catalog { models, dir: dir.to_path_buf() }
    }

    pub fn get(&self, id: &str) -> Option<&ModelInfo> {
        self.models.iter().find(|m| m.id == id)
    }

    pub fn of_kind(&self, kind: ModelKind) -> impl Iterator<Item = &ModelInfo> {
        let mut v: Vec<&ModelInfo> = self.models.iter().filter(|m| m.kind == kind).collect();
        v.sort_by_key(|m| m.display_priority);
        v.into_iter()
    }

    pub fn is_available(&self, id: &str) -> bool {
        self.get(id).is_some_and(|m| m.is_available(&self.dir))
    }

    /// Enhancement models suited to the content and target height, best first, available only.
    pub fn recommend(&self, content: Content, target_height: u32) -> Vec<&ModelInfo> {
        let mut v: Vec<&ModelInfo> = self
            .of_kind(ModelKind::Enhancement)
            .filter(|m| m.is_available(&self.dir))
            .filter(|m| content == Content::Any || m.content == Content::Any || m.content == content)
            .collect();
        // Heavy networks are impractical for very large outputs; prefer cheaper ones there.
        let budget = if target_height > 2160 { 10.0 } else if target_height > 1080 { 60.0 } else { f32::MAX };
        v.sort_by(|a, b| {
            let fa = a.cost <= budget;
            let fb = b.cost <= budget;
            fb.cmp(&fa).then(content_rank(b, content).cmp(&content_rank(a, content))).then(a.display_priority.cmp(&b.display_priority))
        });
        v
    }
}

fn content_rank(m: &ModelInfo, c: Content) -> u8 {
    if m.content == c { 2 } else if m.content == Content::Any { 1 } else { 0 }
}

fn onnx(arch: NetArch, url: &str, license: &str) -> Implementation {
    Implementation::Onnx { arch, source: Some(WeightsSource::Checkpoint { url: url.into() }), license: license.into() }
}

#[allow(clippy::too_many_arguments)]
fn model(id: &str, name: &str, desc: &str, kind: ModelKind, scales: &[u32], imp: Implementation, prefix: &str, content: Content, cost: f32, prio: u32) -> ModelInfo {
    ModelInfo {
        id: id.into(),
        name: name.into(),
        description: desc.into(),
        kind,
        scales: scales.to_vec(),
        implementation: imp,
        prefix: prefix.into(),
        content,
        cost,
        display_priority: prio,
        custom: false,
    }
}

const RESR: &str = "BSD-3-Clause (Real-ESRGAN, Xintao Wang et al.)";

pub fn builtin() -> Vec<ModelInfo> {
    use ModelKind::*;
    vec![
        model(
            "prism-classic",
            "Classic",
            "Lanczos-3 resampling with edge-aware denoise, deblocking, anti-ringing and adaptive sharpening. CPU, very fast, nothing to download.",
            Enhancement,
            &[],
            Implementation::Native,
            "cls",
            Content::Any,
            1.0,
            10,
        ),
        model(
            "resr-general-v3",
            "General",
            "Compact Real-ESRGAN network for real-world footage (4x). Good speed/quality balance on GPU.",
            Enhancement,
            &[4],
            onnx(NetArch::SrVggCompact, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth", RESR),
            "rg3",
            Content::LiveAction,
            40.0,
            20,
        ),
        model(
            "resr-general-wdn-v3",
            "General Soft",
            "Real-ESRGAN general network trained with stronger denoising (4x). For noisy or compressed sources.",
            Enhancement,
            &[4],
            onnx(NetArch::SrVggCompact, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth", RESR),
            "rgd3",
            Content::LiveAction,
            40.0,
            30,
        ),
        model(
            "resr-anime-v3",
            "Animation",
            "Real-ESRGAN network for animation and line art (4x). The fastest network in the catalog.",
            Enhancement,
            &[4],
            onnx(NetArch::SrVggCompact, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth", RESR),
            "ra3",
            Content::Animation,
            20.0,
            40,
        ),
        model(
            "resr-x4plus",
            "Detail",
            "Full Real-ESRGAN RRDB network (4x). Highest detail, about 10x slower than General.",
            Enhancement,
            &[4],
            onnx(NetArch::Rrdb, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", RESR),
            "rx4",
            Content::LiveAction,
            400.0,
            50,
        ),
        model(
            "resr-x2plus",
            "Detail 2x",
            "Full Real-ESRGAN RRDB network (2x).",
            Enhancement,
            &[2],
            onnx(NetArch::Rrdb, "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", RESR),
            "rx2",
            Content::LiveAction,
            150.0,
            60,
        ),
        model(
            "prism-flow",
            "Flow",
            "Native motion-compensated interpolation: hierarchical bilateral motion search, occlusion-aware blending, scene-cut and duplicate-frame handling.",
            Interpolation,
            &[],
            Implementation::Native,
            "flow",
            Content::Any,
            1.0,
            10,
        ),
        model("prism-blend", "Blend", "Weighted blending of neighbouring frames. Fastest; ghosts on motion.", Interpolation, &[], Implementation::Native, "blend", Content::Any, 0.1, 20),
        model(
            "rife",
            "RIFE",
            "RIFE v4.26 optical-flow network on the GPU. Handles complex and fast motion better than Flow; slower.",
            Interpolation,
            &[],
            Implementation::Onnx {
                arch: NetArch::Rife,
                // The vs-mlrt "rife_v2" export pads internally and takes img0, img1 and a
                // timestep plane as one 7-channel input.
                source: Some(WeightsSource::OnnxInArchive {
                    url: "https://github.com/AmusementClub/vs-mlrt/releases/download/external-models/rife_v4.26.7z".into(),
                    member: "rife_v2/rife_v4.26.onnx".into(),
                }),
                license: "MIT (Practical-RIFE weights, hzwer); ONNX export by vs-mlrt".into(),
            },
            "rife",
            Content::Any,
            30.0,
            30,
        ),
        model("prism-estimate", "Auto parameters", "Estimates noise, blockiness and softness from sample frames to set enhancement sliders.", ParamEstimation, &[], Implementation::Native, "est", Content::Any, 0.1, 10),
        model("vidstab-detect", "Camera motion", "libvidstab motion detection (FFmpeg vidstabdetect).", CameraPose, &[], Implementation::Ffmpeg { filter: "vidstabdetect".into() }, "stab", Content::Any, 1.0, 10),
        model("vidstab", "Stabilize", "libvidstab smoothing and transform (FFmpeg vidstabtransform).", Stabilization, &[], Implementation::Ffmpeg { filter: "vidstabtransform".into() }, "stab", Content::Any, 1.0, 10),
        model("prism-deblur", "Motion deblur", "Edge-preserving deconvolution approximation for mild motion blur.", MotionDeblur, &[], Implementation::Native, "dbl", Content::Any, 2.0, 10),
        model("prism-itm", "SDR to HDR", "Inverse tone mapping from SDR BT.709 to HDR10 (PQ) or HLG in BT.2020.", Hdr, &[], Implementation::Native, "hdr", Content::Any, 1.0, 10),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scale_selection() {
        let cat = Catalog::load(Path::new("does-not-exist"));
        let g = cat.get("resr-general-v3").unwrap();
        assert_eq!(g.pick_scale(1.5), 4);
        assert_eq!(g.pick_scale(8.0), 4);
        let c = cat.get("prism-classic").unwrap();
        assert_eq!(c.pick_scale(2.7), 3);
        assert!(c.is_available(Path::new("x")));
        assert!(!g.is_available(Path::new("does-not-exist")));
    }

    #[test]
    fn recommendation_falls_back_to_available() {
        let cat = Catalog::load(Path::new("does-not-exist"));
        let rec = cat.recommend(Content::Animation, 2160);
        assert_eq!(rec[0].id, "prism-classic");
    }

    #[test]
    fn custom_models_are_discovered() {
        let dir = std::env::temp_dir().join(format!("prism-cat-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("mynet.fp16.onnx"), b"x").unwrap();
        std::fs::write(dir.join("mynet.json"), br#"{"name":"My Net","scale":2}"#).unwrap();
        std::fs::write(dir.join("resr-anime-v3.onnx"), b"x").unwrap();
        let cat = Catalog::load(&dir);
        let m = cat.get("mynet").unwrap();
        assert!(m.custom);
        assert_eq!(m.name, "My Net");
        assert_eq!(m.scales, vec![2]);
        assert!(cat.is_available("resr-anime-v3"));
        assert_eq!(cat.models.iter().filter(|m| m.id == "resr-anime-v3").count(), 1);
        std::fs::remove_dir_all(&dir).ok();
    }
}
