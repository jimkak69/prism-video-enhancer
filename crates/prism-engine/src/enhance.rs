//! Enhancement: pre-processing, upscaling (network or Lanczos), post-processing.
//!
//! Slider semantics follow the reference product: revert compression, reduce noise, anti-alias /
//! deblur and add noise act before upscaling; dehalo, recover details and sharpen act after it;
//! recover original detail blends back a plain resample of the untouched input.
//!
//! The work is split into two pipeline stages so the upscaler (often a GPU network) and the
//! output-resolution post-processing (CPU) run concurrently on consecutive frames. A side channel
//! carries each frame's inputs needed by the post stage.

use std::sync::{Arc, OnceLock};

use anyhow::{Result, anyhow};
use crossbeam_channel::{Receiver, Sender, unbounded};
use prism_core::models::{Catalog, NetArch};
use prism_core::plan::{EnhanceParams, EnhanceSpec};
use prism_core::settings::{DeviceSettings, ParamMode};
use rayon::prelude::*;

use crate::ai::runtime::OnnxUpscaler;
use crate::frame::{Frame, Scratch};
use crate::ops::analysis::estimate_params;
use crate::ops::filters::{box_blur, cas, deblock, gaussian_blur, guided_smooth, local_minmax, unsharp};
use crate::ops::noise::add_noise;
use crate::ops::resample::{Filter, clamp_to_upsampled, resize, resize_blend_into};

enum Upscaler {
    Classic,
    Network(Box<OnnxUpscaler>),
}

/// Per-frame data the post stage needs from the front stage.
struct Side {
    /// Untouched input, for "recover original detail".
    original: Option<Frame>,
    /// Pre-processed frame at work resolution, whose local range bounds the dehalo clamp.
    work: Option<Frame>,
}

/// Parameters resolved once (manual values, or the estimate in auto modes).
type SharedParams = Arc<OnceLock<EnhanceParams>>;

/// Front half: estimation, pre-processing and upscaling.
pub struct EnhanceFront {
    spec: EnhanceSpec,
    upscaler: Upscaler,
    params: SharedParams,
    pending: Vec<Frame>,
    temporal: Option<Frame>,
    frame_no: u64,
    work: (usize, usize),
    side: Sender<Side>,
    pub description: String,
}

/// Back half: dehalo, details, sharpening and recover-original at output resolution.
pub struct EnhanceBack {
    spec: EnhanceSpec,
    params: SharedParams,
    side: Receiver<Side>,
    classic: bool,
}

/// Builds the two halves for `spec`.
pub fn build(spec: &EnhanceSpec, catalog: &Catalog, device: &DeviceSettings) -> Result<(EnhanceFront, EnhanceBack)> {
    let model = catalog.get(&spec.model).ok_or_else(|| anyhow!("unknown model {}", spec.model))?;
    let pre = spec.pre_downscale.max(1) as usize;
    let work = ((spec.in_width as usize / pre).max(1), (spec.in_height as usize / pre).max(1));
    let (upscaler, description) = if model.is_onnx() {
        // FP16 files only pay off on GPU providers; the CPU provider lacks FP16 kernels.
        let gpu = device.backend != prism_core::settings::Backend::Cpu;
        let path = model.onnx_file(&catalog.dir, device.fp16 && gpu).ok_or_else(|| anyhow!("model file for {} not found in {}", model.id, catalog.dir.display()))?;
        let up = OnnxUpscaler::new(&path, work.0, work.1, device, model.arch() == Some(NetArch::Rrdb))?;
        let d = up.description.clone();
        (Upscaler::Network(Box::new(up)), d)
    } else {
        (Upscaler::Classic, "Lanczos-3 + native enhancement (CPU)".to_string())
    };
    let params: SharedParams = Arc::new(OnceLock::new());
    if spec.mode == ParamMode::Manual || spec.estimate_frames == 0 {
        let _ = params.set(spec.params);
    }
    let classic = matches!(upscaler, Upscaler::Classic);
    let (tx, rx) = unbounded();
    Ok((
        EnhanceFront { spec: spec.clone(), upscaler, params: params.clone(), pending: Vec::new(), temporal: None, frame_no: 0, work, side: tx, description },
        EnhanceBack { spec: spec.clone(), params, side: rx, classic },
    ))
}

impl EnhanceFront {
    pub fn params(&self) -> Option<EnhanceParams> {
        self.params.get().copied()
    }

    pub fn push(&mut self, frame: Frame, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        if self.params.get().is_none() {
            self.pending.push(frame);
            if self.pending.len() < self.spec.estimate_frames as usize {
                return Ok(());
            }
            self.resolve_params();
            for f in std::mem::take(&mut self.pending) {
                let o = self.process(f)?;
                out(o)?;
            }
            return Ok(());
        }
        let o = self.process(frame)?;
        out(o)
    }

    pub fn finish(&mut self, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        if self.params.get().is_none() {
            self.resolve_params();
        }
        for f in std::mem::take(&mut self.pending) {
            let o = self.process(f)?;
            out(o)?;
        }
        Ok(())
    }

    fn resolve_params(&mut self) {
        let refs: Vec<&Frame> = self.pending.iter().collect();
        let est = estimate_params(&refs);
        let s = self.spec.params;
        let p = match self.spec.mode {
            ParamMode::Manual => s,
            ParamMode::Auto => EnhanceParams { add_noise: s.add_noise, recover_original: s.recover_original, deblur: s.deblur, ..est },
            ParamMode::RelativeToAuto => {
                let c = |a: f32, b: f32| (a + b).clamp(0.0, 1.0);
                EnhanceParams {
                    compression: c(est.compression, s.compression),
                    details: c(est.details, s.details),
                    sharpen: c(est.sharpen, s.sharpen),
                    noise: c(est.noise, s.noise),
                    dehalo: c(est.dehalo, s.dehalo),
                    deblur: s.deblur,
                    add_noise: s.add_noise,
                    recover_original: s.recover_original,
                }
            }
        };
        let _ = self.params.set(p);
    }

    fn process(&mut self, input: Frame) -> Result<Frame> {
        let p = self.params.get().copied().unwrap_or(self.spec.params);
        let (ow, oh) = (self.spec.out_width as usize, self.spec.out_height as usize);
        self.frame_no += 1;
        let classic = matches!(self.upscaler, Upscaler::Classic);
        let upscaling = ow > self.work.0 || oh > self.work.1;
        let dehalo = if classic { (0.45 + 0.55 * p.dehalo).min(1.0) } else { p.dehalo };

        let original = if p.recover_original > 0.0 { Some(input.duplicate()) } else { None };
        let mut x = if (input.width, input.height) != self.work { resize(&input, self.work.0, self.work.1, Filter::Lanczos3) } else { input };
        let (w, h) = (x.width, x.height);

        if p.compression > 0.0 {
            x.planes_mut().into_par_iter().for_each(|pl| deblock(pl, w, h, p.compression));
        }
        if p.noise > 0.0 {
            let sigma = 0.025 * p.noise;
            let eps = (2.0 * sigma) * (2.0 * sigma);
            let r = if p.noise > 0.5 { 3 } else { 2 };
            let mut tmp = Scratch::new(w * h);
            for c in 0..3 {
                guided_smooth(x.plane(c), &mut tmp, w, h, r, eps);
                x.plane_mut(c).copy_from_slice(&tmp);
            }
            self.temporal_denoise(&mut x, sigma, p.noise);
        }
        if p.deblur != 0.0 {
            let mut tmp = Scratch::new(w * h);
            for c in 0..3 {
                if p.deblur > 0.0 {
                    unsharp(x.plane(c), &mut tmp, w, h, 1.0, 0.9 * p.deblur);
                } else {
                    gaussian_blur(x.plane(c), &mut tmp, w, h, 0.3 + 0.6 * -p.deblur);
                }
                x.plane_mut(c).copy_from_slice(&tmp);
            }
        }
        if p.add_noise > 0.0 {
            add_noise(&mut x, 0.02 * p.add_noise, self.frame_no);
        }
        let work = if dehalo > 0.0 && upscaling { Some(x.duplicate()) } else { None };
        let y = match &mut self.upscaler {
            Upscaler::Classic => resize(&x, ow, oh, Filter::Lanczos3),
            Upscaler::Network(up) => {
                let big = up.upscale(&x)?;
                if (big.width, big.height) != (ow, oh) { resize(&big, ow, oh, Filter::Lanczos3) } else { big }
            }
        };
        self.side.send(Side { original, work }).map_err(|_| anyhow!("enhancement post stage stopped"))?;
        Ok(y)
    }

    /// Recursive motion-adaptive temporal filter: static areas average over time, moving areas
    /// follow the current frame.
    fn temporal_denoise(&mut self, x: &mut Frame, sigma: f32, strength: f32) {
        let (w, h) = (x.width, x.height);
        let Some(prev) = &self.temporal else {
            self.temporal = Some(x.duplicate());
            return;
        };
        if (prev.width, prev.height) != (w, h) {
            self.temporal = Some(x.duplicate());
            return;
        }
        // Motion measure on a lightly blurred difference so noise alone does not look like motion.
        let mut diff = Scratch::new(w * h);
        {
            let (cur, old) = (x.planes(), prev.planes());
            diff.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                for (i, d) in row.iter_mut().enumerate() {
                    let k = y * w + i;
                    *d = ((cur[0][k] - old[0][k]) + (cur[1][k] - old[1][k]) + (cur[2][k] - old[2][k])).abs() / 3.0;
                }
            });
        }
        let mut motion = Scratch::new(w * h);
        box_blur(&diff, &mut motion, w, h, 1);
        let min_alpha = 1.0 - 0.6 * strength.clamp(0.0, 1.0);
        let thr = 2.5 * sigma + 0.004;
        let old = prev.planes();
        for (c, plane) in x.planes_mut().into_iter().enumerate() {
            let o = old[c];
            plane.par_iter_mut().zip(o.par_iter()).zip(motion.par_iter()).for_each(|((v, ov), m)| {
                let alpha = (m / thr).clamp(min_alpha, 1.0);
                *v = ov + (*v - ov) * alpha;
            });
        }
        self.temporal = Some(x.duplicate());
    }
}

impl EnhanceBack {
    pub fn process(&mut self, mut y: Frame) -> Result<Frame> {
        let side = self.side.recv().map_err(|_| anyhow!("enhancement front stage stopped"))?;
        let p = self.params.get().copied().unwrap_or(self.spec.params);
        let (ow, oh) = (y.width, y.height);
        let dehalo = if self.classic { (0.45 + 0.55 * p.dehalo).min(1.0) } else { p.dehalo };
        if let Some(work) = &side.work {
            let (w, h) = (work.width, work.height);
            let mut mn = Scratch::new(w * h);
            let mut mx = Scratch::new(w * h);
            for c in 0..3 {
                local_minmax(work.plane(c), w, h, &mut mn, &mut mx);
                clamp_to_upsampled(y.plane_mut(c), ow, oh, &mn, &mx, w, h, dehalo, 0.004);
            }
        }
        let scale = side.work.as_ref().map(|wk| ow as f32 / wk.width as f32).unwrap_or(ow as f32 / self.spec.in_width as f32).max(1.0);
        if p.details > 0.0 || p.sharpen > 0.0 {
            luma_detail(&mut y, 1.6 * scale, 0.6 * p.details, p.sharpen);
        }
        if let Some(orig) = &side.original {
            resize_blend_into(orig, &mut y, p.recover_original, Filter::Lanczos3);
        }
        Ok(y)
    }
}

/// Local-contrast boost (`details`, unsharp at `sigma`) and adaptive sharpening (`sharpen`)
/// applied to luma only; the luma change is added to R, G and B so hue and saturation are kept.
/// One plane instead of three, which matters at 4K.
fn luma_detail(y: &mut Frame, sigma: f32, details: f32, sharpen: f32) {
    let (w, h) = (y.width, y.height);
    let luma = y.gray();
    let mut a = Scratch::new(w * h);
    let mut b = Scratch::new(w * h);
    let mut cur: &[f32] = &luma;
    if details > 0.0 {
        unsharp(cur, &mut a, w, h, sigma, details);
        cur = &a;
    }
    if sharpen > 0.0 {
        cas(cur, &mut b, w, h, sharpen);
        cur = &b;
    }
    let cur = cur;
    for plane in y.planes_mut() {
        plane.par_chunks_mut(w).zip(cur.par_chunks(w).zip(luma.par_chunks(w))).for_each(|(p, (n, o))| {
            for i in 0..p.len() {
                p[i] += n[i] - o[i];
            }
        });
    }
}

/// Both halves in one object, for single-threaded use (tests, instant previews).
pub struct Enhancer {
    front: EnhanceFront,
    back: EnhanceBack,
}

impl Enhancer {
    pub fn new(spec: &EnhanceSpec, catalog: &Catalog, device: &DeviceSettings) -> Result<Enhancer> {
        let (front, back) = build(spec, catalog, device)?;
        Ok(Enhancer { front, back })
    }

    pub fn params(&self) -> Option<EnhanceParams> {
        self.front.params()
    }

    pub fn push(&mut self, frame: Frame, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        let back = &mut self.back;
        self.front.push(frame, &mut |f| out(back.process(f)?))
    }

    pub fn finish(&mut self, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        let back = &mut self.back;
        self.front.finish(&mut |f| out(back.process(f)?))
    }

    pub fn process(&mut self, frame: Frame) -> Result<Frame> {
        let f = self.front.process(frame)?;
        self.back.process(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::noise::noise_plane;
    use std::path::Path;

    fn spec(w: u32, h: u32, ow: u32, oh: u32, params: EnhanceParams, mode: ParamMode) -> EnhanceSpec {
        EnhanceSpec { model: "prism-classic".into(), in_width: w, in_height: h, out_width: ow, out_height: oh, pre_downscale: 1, params, mode, estimate_frames: if mode == ParamMode::Manual { 0 } else { 2 } }
    }

    fn noisy(w: usize, h: usize, sigma: f32, seed: u64) -> Frame {
        let mut f = Frame::uninit(w, h);
        let mut n = vec![0.0; w * h];
        for c in 0..3 {
            noise_plane(&mut n, w, h, seed + c as u64, 0.0);
            for (i, v) in f.plane_mut(c).iter_mut().enumerate() {
                let x = (i % w) as f32;
                *v = 0.3 + 0.4 * (x / w as f32) + n[i] * sigma;
            }
        }
        f
    }

    fn residual_noise(f: &Frame) -> f32 {
        let (w, h) = (f.width, f.height);
        let mut s = 0.0;
        let mut n = 0;
        for y in 4..h - 4 {
            for x in 4..w - 4 {
                let clean = 0.3 + 0.4 * (x as f32 / w as f32);
                s += (f.plane(1)[y * w + x] - clean).powi(2);
                n += 1;
            }
        }
        (s / n as f32).sqrt()
    }

    #[test]
    fn classic_upscale_has_target_size_and_denoise_reduces_noise() {
        let cat = Catalog::load(Path::new("none"));
        let dev = DeviceSettings::default();
        let manual = EnhanceParams { noise: 0.8, recover_original: 0.0, ..Default::default() };
        let mut e = Enhancer::new(&spec(64, 48, 64, 48, manual, ParamMode::Manual), &cat, &dev).unwrap();
        let mut outs = Vec::new();
        for s in 0..4 {
            e.push(noisy(64, 48, 0.02, s * 10), &mut |f| {
                outs.push(f);
                Ok(())
            })
            .unwrap();
        }
        let before = residual_noise(&noisy(64, 48, 0.02, 30));
        let after = residual_noise(outs.last().unwrap());
        assert!(after < before * 0.5, "noise {before} -> {after}");

        let all = EnhanceParams { noise: 0.3, details: 0.5, sharpen: 0.5, dehalo: 0.5, compression: 0.5, recover_original: 0.2, ..Default::default() };
        let mut up = Enhancer::new(&spec(64, 48, 256, 192, all, ParamMode::Manual), &cat, &dev).unwrap();
        let out = up.process(noisy(64, 48, 0.0, 1)).unwrap();
        assert_eq!((out.width, out.height), (256, 192));
        assert!(out.data().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn auto_mode_buffers_then_estimates() {
        let cat = Catalog::load(Path::new("none"));
        let mut e = Enhancer::new(&spec(64, 48, 128, 96, EnhanceParams::default(), ParamMode::Auto), &cat, &DeviceSettings::default()).unwrap();
        let mut n = 0;
        let mut sink = |_f: Frame| -> Result<()> {
            n += 1;
            Ok(())
        };
        e.push(noisy(64, 48, 0.02, 1), &mut sink).unwrap();
        assert!(e.params().is_none());
        e.push(noisy(64, 48, 0.02, 2), &mut sink).unwrap();
        assert!(e.params().unwrap().noise > 0.3, "{:?}", e.params());
        e.finish(&mut sink).unwrap();
        assert_eq!(n, 2);
    }
}
