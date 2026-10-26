//! Converts Real-ESRGAN PyTorch checkpoints (`.pth`) to ONNX without PyTorch.
//!
//! Supported architectures (detected from the weight names and shapes):
//! - `SRVGGNetCompact` (realesr-general-x4v3, realesr-general-wdn-x4v3, realesr-animevideov3):
//!   conv + PReLU stack, pixel shuffle, plus a nearest-upsampled copy of the input.
//! - `RRDBNet` (RealESRGAN_x4plus, RealESRGAN_x2plus): residual-in-residual dense blocks with
//!   nearest-neighbour upsampling; the x2 variant pixel-unshuffles its input.
//!
//! Graph inputs/outputs are FP32 `[batch, 3, height, width]`. With `fp16`, weights are stored as
//! FP16 and the graph casts at its edges, so callers always exchange FP32.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use prism_core::models::NetArch;

use super::onnx_proto::{Attr, Dim, FLOAT, FLOAT16, Graph};
use super::pickle::{StateDict, Tensor, load_state_dict};

const OPSET: i64 = 13;

#[derive(Debug, Clone)]
pub struct Converted {
    pub arch: NetArch,
    pub scale: u32,
    pub path: PathBuf,
    pub bytes: usize,
    pub description: String,
}

fn get<'a>(sd: &'a StateDict, key: &str) -> Result<&'a Tensor> {
    sd.get(key).ok_or_else(|| anyhow!("checkpoint is missing '{key}'"))
}

pub fn detect(sd: &StateDict) -> Option<NetArch> {
    if sd.contains_key("conv_first.weight") && sd.contains_key("body.0.rdb1.conv1.weight") {
        Some(NetArch::Rrdb)
    } else if sd.get("body.0.weight").is_some_and(|t| t.shape.len() == 4 && t.shape[1] == 3) {
        Some(NetArch::SrVggCompact)
    } else {
        None
    }
}

fn image_dims() -> Vec<Dim> {
    vec![Dim::Named("batch"), Dim::Fixed(3), Dim::Named("height"), Dim::Named("width")]
}

fn out_dims() -> Vec<Dim> {
    vec![Dim::Named("batch"), Dim::Fixed(3), Dim::Named("out_height"), Dim::Named("out_width")]
}

/// Builds the SRVGGNetCompact graph. Returns (onnx bytes, scale).
pub fn build_srvgg(sd: &StateDict, fp16: bool) -> Result<(Vec<u8>, u32, String)> {
    let mut g = Graph::new(fp16);
    let input = g.input("input", image_dims());
    let x0 = if fp16 { g.cast(&input, FLOAT16) } else { input.clone() };
    let mut x = x0.clone();
    let mut i = 0;
    let mut convs = 0;
    let mut feat = 0;
    let mut last_out = 0;
    loop {
        let wkey = format!("body.{i}.weight");
        let Some(w) = sd.get(&wkey) else { break };
        if w.shape.len() == 4 {
            let b = get(sd, &format!("body.{i}.bias"))?;
            let wn = g.weight("conv_w", &w.shape, &w.data);
            let bn = g.weight("conv_b", &b.shape, &b.data);
            x = g.conv3x3(&x, &wn, &bn);
            convs += 1;
            last_out = w.shape[0];
            if feat == 0 {
                feat = w.shape[0];
            }
        } else if w.shape.len() == 1 {
            let slope = g.weight("prelu", &[w.shape[0], 1, 1], &w.data);
            x = g.node("PRelu", &[&x, &slope], vec![]);
        } else {
            bail!("unexpected tensor shape {:?} for {wkey}", w.shape);
        }
        i += 1;
    }
    if convs < 2 {
        bail!("not an SRVGGNetCompact checkpoint (found {convs} convolutions)");
    }
    let scale = ((last_out / 3) as f64).sqrt().round() as u32;
    if scale == 0 || 3 * (scale * scale) as usize != last_out {
        bail!("last convolution has {last_out} channels; expected 3 * scale^2");
    }
    x = g.node("DepthToSpace", &[&x], vec![Attr::Int("blocksize", scale as i64), Attr::Str("mode", "CRD".into())]);
    let scales = g.const_f32("scales", &[4], &[1.0, 1.0, scale as f32, scale as f32]);
    let base = g.node(
        "Resize",
        &[&x0, "", &scales],
        vec![Attr::Str("mode", "nearest".into()), Attr::Str("coordinate_transformation_mode", "asymmetric".into()), Attr::Str("nearest_mode", "floor".into())],
    );
    let mut y = g.node("Add", &[&x, &base], vec![]);
    if fp16 {
        y = g.cast(&y, FLOAT);
    }
    g.output(&y, "output", out_dims());
    let desc = format!("SRVGGNetCompact x{scale}, {feat} features, {} body convolutions{}", convs - 2, if fp16 { ", FP16" } else { "" });
    Ok((g.encode(OPSET, &desc), scale, desc))
}

/// Builds the RRDBNet graph. Returns (onnx bytes, scale).
pub fn build_rrdb(sd: &StateDict, fp16: bool) -> Result<(Vec<u8>, u32, String)> {
    let first = get(sd, "conv_first.weight")?;
    let in_ch = first.shape[1];
    let (scale, unshuffle) = match in_ch {
        3 => (4u32, 1usize),
        12 => (2, 2),
        48 => (1, 4),
        c => bail!("unsupported RRDBNet input channels {c}"),
    };
    let feat = first.shape[0];
    let mut blocks = 0;
    while sd.contains_key(&format!("body.{blocks}.rdb1.conv1.weight")) {
        blocks += 1;
    }
    let mut g = Graph::new(fp16);
    let input = g.input("input", image_dims());
    let mut x = if fp16 { g.cast(&input, FLOAT16) } else { input.clone() };

    let conv = |g: &mut Graph, x: &str, name: &str| -> Result<String> {
        let w = get(sd, &format!("{name}.weight"))?;
        let b = get(sd, &format!("{name}.bias"))?;
        let wn = g.weight("w", &w.shape, &w.data);
        let bn = g.weight("b", &b.shape, &b.data);
        Ok(g.conv3x3(x, &wn, &bn))
    };
    let lrelu = |g: &mut Graph, x: &str| g.node("LeakyRelu", &[x], vec![Attr::Float("alpha", 0.2)]);

    if unshuffle > 1 {
        // torch.pixel_unshuffle orders channels c*r^2 + (i*r + j); ONNX SpaceToDepth orders
        // (i*r + j)*C + c. Permute conv_first's input channels to match SpaceToDepth.
        x = g.node("SpaceToDepth", &[&x], vec![Attr::Int("blocksize", unshuffle as i64)]);
        let r2 = unshuffle * unshuffle;
        let c = 3;
        let (o, k) = (first.shape[0], first.shape[2] * first.shape[3]);
        let mut wperm = vec![0.0f32; first.data.len()];
        for oc in 0..o {
            for onnx_ch in 0..c * r2 {
                let (kk, cc) = (onnx_ch / c, onnx_ch % c);
                let torch_ch = cc * r2 + kk;
                let src = &first.data[(oc * in_ch + torch_ch) * k..][..k];
                wperm[(oc * in_ch + onnx_ch) * k..][..k].copy_from_slice(src);
            }
        }
        let b = get(sd, "conv_first.bias")?;
        let wn = g.weight("w", &first.shape, &wperm);
        let bn = g.weight("b", &b.shape, &b.data);
        x = g.conv3x3(&x, &wn, &bn);
    } else {
        x = conv(&mut g, &x, "conv_first")?;
    }
    let feat_in = x.clone();
    let rscale = g.weight("res_scale", &[1], &[0.2]);
    let mut h = x.clone();
    for bi in 0..blocks {
        let block_in = h.clone();
        for rdb in 1..=3 {
            let rdb_in = h.clone();
            let mut cat: Vec<String> = vec![rdb_in.clone()];
            for ci in 1..=5 {
                let refs: Vec<&str> = cat.iter().map(String::as_str).collect();
                let xin = if refs.len() == 1 { refs[0].to_string() } else { g.node("Concat", &refs, vec![Attr::Int("axis", 1)]) };
                let y = conv(&mut g, &xin, &format!("body.{bi}.rdb{rdb}.conv{ci}"))?;
                if ci < 5 {
                    let y = lrelu(&mut g, &y);
                    cat.push(y);
                } else {
                    let scaled = g.node("Mul", &[&y, &rscale], vec![]);
                    h = g.node("Add", &[&scaled, &rdb_in], vec![]);
                }
            }
        }
        let scaled = g.node("Mul", &[&h, &rscale], vec![]);
        h = g.node("Add", &[&scaled, &block_in], vec![]);
    }
    let body = conv(&mut g, &h, "conv_body")?;
    let mut f = g.node("Add", &[&feat_in, &body], vec![]);
    let up = g.const_f32("scales", &[4], &[1.0, 1.0, 2.0, 2.0]);
    for name in ["conv_up1", "conv_up2"] {
        let r = g.node("Resize", &[&f, "", &up], vec![Attr::Str("mode", "nearest".into()), Attr::Str("coordinate_transformation_mode", "asymmetric".into()), Attr::Str("nearest_mode", "floor".into())]);
        let c = conv(&mut g, &r, name)?;
        f = lrelu(&mut g, &c);
    }
    let hr = conv(&mut g, &f, "conv_hr")?;
    let hr = lrelu(&mut g, &hr);
    let mut out = conv(&mut g, &hr, "conv_last")?;
    if fp16 {
        out = g.cast(&out, FLOAT);
    }
    g.output(&out, "output", out_dims());
    let desc = format!("RRDBNet x{scale}, {feat} features, {blocks} blocks{}", if fp16 { ", FP16" } else { "" });
    Ok((g.encode(OPSET, &desc), scale, desc))
}

/// Converts `pth` into `<out_dir>/<id>.onnx` (or `<id>.fp16.onnx`).
pub fn convert(pth: &Path, out_dir: &Path, id: &str, fp16: bool) -> Result<Converted> {
    let sd = load_state_dict(pth)?;
    let arch = detect(&sd).ok_or_else(|| anyhow!("{} is neither an SRVGGNetCompact nor an RRDBNet checkpoint", pth.display()))?;
    let (bytes, scale, description) = match arch {
        NetArch::SrVggCompact => build_srvgg(&sd, fp16)?,
        NetArch::Rrdb => build_rrdb(&sd, fp16)?,
        _ => unreachable!(),
    };
    std::fs::create_dir_all(out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    let path = out_dir.join(if fp16 { format!("{id}.fp16.onnx") } else { format!("{id}.onnx") });
    std::fs::write(&path, &bytes).with_context(|| format!("writing {}", path.display()))?;
    Ok(Converted { arch, scale, path, bytes: bytes.len(), description })
}

/// Guesses a catalog id from a checkpoint file name.
pub fn catalog_id_for(file_name: &str) -> Option<&'static str> {
    let n = file_name.to_ascii_lowercase();
    if n.contains("general-wdn-x4v3") {
        Some("resr-general-wdn-v3")
    } else if n.contains("general-x4v3") {
        Some("resr-general-v3")
    } else if n.contains("animevideov3") {
        Some("resr-anime-v3")
    } else if n.contains("x4plus") && !n.contains("anime") {
        Some("resr-x4plus")
    } else if n.contains("x2plus") {
        Some("resr-x2plus")
    } else {
        None
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::ops::noise::Rng;

    fn rand_tensor(rng: &mut Rng, shape: &[usize], scale: f32) -> Tensor {
        let n = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: (0..n).map(|_| rng.gauss() * scale).collect() }
    }

    /// A tiny random SRVGGNetCompact state dict: `feat` features, `convs` body convolutions.
    pub fn tiny_srvgg(feat: usize, convs: usize, scale: usize) -> StateDict {
        let mut rng = Rng::new(42);
        let mut sd = StateDict::new();
        let mut idx = 0;
        let mut conv = |sd: &mut StateDict, idx: &mut usize, o: usize, i: usize| {
            sd.insert(format!("body.{idx}.weight"), rand_tensor(&mut rng, &[o, i, 3, 3], 0.2));
            sd.insert(format!("body.{idx}.bias"), rand_tensor(&mut rng, &[o], 0.05));
            *idx += 1;
        };
        let prelu = |sd: &mut StateDict, idx: &mut usize| {
            sd.insert(format!("body.{idx}.weight"), Tensor { shape: vec![feat], data: vec![0.25; feat] });
            *idx += 1;
        };
        conv(&mut sd, &mut idx, feat, 3);
        prelu(&mut sd, &mut idx);
        for _ in 0..convs {
            conv(&mut sd, &mut idx, feat, feat);
            prelu(&mut sd, &mut idx);
        }
        conv(&mut sd, &mut idx, 3 * scale * scale, feat);
        sd
    }

    /// Reference forward pass for SRVGGNetCompact (naive, for tests).
    pub fn reference_srvgg(sd: &StateDict, input: &[f32], w: usize, h: usize) -> (Vec<f32>, usize) {
        let mut x = input.to_vec();
        let mut ch = 3;
        let mut i = 0;
        while let Some(t) = sd.get(&format!("body.{i}.weight")) {
            if t.shape.len() == 4 {
                let b = &sd[&format!("body.{i}.bias")];
                let (o, inc) = (t.shape[0], t.shape[1]);
                let mut y = vec![0.0f32; o * w * h];
                for oc in 0..o {
                    for yy in 0..h {
                        for xx in 0..w {
                            let mut acc = b.data[oc];
                            for ic in 0..inc {
                                for ky in 0..3 {
                                    for kx in 0..3 {
                                        let sy = yy as isize + ky as isize - 1;
                                        let sx = xx as isize + kx as isize - 1;
                                        if sy < 0 || sx < 0 || sy >= h as isize || sx >= w as isize {
                                            continue;
                                        }
                                        acc += t.data[((oc * inc + ic) * 3 + ky) * 3 + kx] * x[(ic * h + sy as usize) * w + sx as usize];
                                    }
                                }
                            }
                            y[(oc * h + yy) * w + xx] = acc;
                        }
                    }
                }
                x = y;
                ch = o;
            } else {
                for c in 0..ch {
                    for v in &mut x[c * w * h..(c + 1) * w * h] {
                        if *v < 0.0 {
                            *v *= t.data[c];
                        }
                    }
                }
            }
            i += 1;
        }
        let s = ((ch / 3) as f64).sqrt() as usize;
        let (ow, oh) = (w * s, h * s);
        let mut out = vec![0.0f32; 3 * ow * oh];
        for c in 0..3 {
            for oy in 0..oh {
                for ox in 0..ow {
                    let src_c = c * s * s + (oy % s) * s + (ox % s);
                    out[(c * oh + oy) * ow + ox] = x[(src_c * h + oy / s) * w + ox / s] + input[(c * h + oy / s) * w + ox / s];
                }
            }
        }
        (out, s)
    }

    #[test]
    fn detects_and_builds_srvgg() {
        let sd = tiny_srvgg(4, 2, 2);
        assert_eq!(detect(&sd), Some(NetArch::SrVggCompact));
        let (bytes, scale, _) = build_srvgg(&sd, false).unwrap();
        assert_eq!(scale, 2);
        assert!(bytes.len() > 100);
        let (bytes16, _, _) = build_srvgg(&sd, true).unwrap();
        assert!(bytes16.len() < bytes.len());
    }

    #[test]
    fn catalog_ids() {
        assert_eq!(catalog_id_for("realesr-general-x4v3.pth"), Some("resr-general-v3"));
        assert_eq!(catalog_id_for("realesr-general-wdn-x4v3.pth"), Some("resr-general-wdn-v3"));
        assert_eq!(catalog_id_for("RealESRGAN_x4plus.pth"), Some("resr-x4plus"));
        assert_eq!(catalog_id_for("RealESRGAN_x4plus_anime_6B.pth"), None);
    }
}
