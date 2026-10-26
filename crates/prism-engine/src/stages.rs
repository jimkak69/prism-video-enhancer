//! Engine stages built from plan [`StageSpec`]s.

use anyhow::{Result, anyhow, bail};
use prism_core::models::{Catalog, Implementation};
use prism_core::plan::{HdrSpec, StageSpec};
use prism_core::settings::DeviceSettings;

use crate::ai::runtime::OnnxInterpolator;
use crate::enhance::{self, EnhanceBack, EnhanceFront};
use crate::frame::{Frame, Scratch};
use crate::interp::{Interpolator, Method};
use crate::ops::color::inverse_tone_map;
use crate::ops::filters::{clamp_towards, local_minmax, unsharp};
use crate::ops::noise::add_grain;

pub type Emit<'a> = dyn FnMut(Frame) -> Result<()> + 'a;

pub trait Stage: Send {
    fn name(&self) -> String;
    /// What actually runs (backend, tiles, provider), for logs.
    fn describe(&self) -> String;
    fn push(&mut self, frame: Frame, out: &mut Emit) -> Result<()>;
    fn finish(&mut self, _out: &mut Emit) -> Result<()> {
        Ok(())
    }
    /// Counters worth reporting at the end (e.g. scene cuts).
    fn stats(&self) -> Vec<(String, u64)> {
        Vec::new()
    }
}

pub struct EnhanceStage(EnhanceFront);

impl Stage for EnhanceStage {
    fn name(&self) -> String {
        "enhance".into()
    }
    fn describe(&self) -> String {
        match self.0.params() {
            Some(p) => format!("{}; noise {:.2} compression {:.2} details {:.2} sharpen {:.2}", self.0.description, p.noise, p.compression, p.details, p.sharpen),
            None => self.0.description.clone(),
        }
    }
    fn push(&mut self, frame: Frame, out: &mut Emit) -> Result<()> {
        self.0.push(frame, out)
    }
    fn finish(&mut self, out: &mut Emit) -> Result<()> {
        self.0.finish(out)
    }
}

/// Output-resolution post-processing of an enhancement, on its own thread.
pub struct EnhancePostStage(EnhanceBack);

impl Stage for EnhancePostStage {
    fn name(&self) -> String {
        "enhance post".into()
    }
    fn describe(&self) -> String {
        "dehalo, details, sharpen, recover original (CPU)".into()
    }
    fn push(&mut self, frame: Frame, out: &mut Emit) -> Result<()> {
        let f = self.0.process(frame)?;
        out(f)
    }
}

pub struct InterpolateStage {
    inner: Interpolator,
    description: String,
}

impl Stage for InterpolateStage {
    fn name(&self) -> String {
        "interpolate".into()
    }
    fn describe(&self) -> String {
        self.description.clone()
    }
    fn push(&mut self, frame: Frame, out: &mut Emit) -> Result<()> {
        self.inner.push(frame, out)
    }
    fn finish(&mut self, out: &mut Emit) -> Result<()> {
        self.inner.finish(out)
    }
    fn stats(&self) -> Vec<(String, u64)> {
        vec![("scene cuts".into(), self.inner.scene_cuts), ("duplicates replaced".into(), self.inner.duplicates), ("frames out".into(), self.inner.emitted())]
    }
}

/// Unsharp deconvolution approximation with an anti-halo clamp.
pub struct DeblurStage {
    strength: f32,
}

impl Stage for DeblurStage {
    fn name(&self) -> String {
        "motion deblur".into()
    }
    fn describe(&self) -> String {
        format!("unsharp deconvolution, strength {:.2} (CPU)", self.strength)
    }
    fn push(&mut self, mut frame: Frame, out: &mut Emit) -> Result<()> {
        let (w, h) = (frame.width, frame.height);
        let mut tmp = Scratch::new(w * h);
        let mut mn = Scratch::new(w * h);
        let mut mx = Scratch::new(w * h);
        for c in 0..3 {
            local_minmax(frame.plane(c), w, h, &mut mn, &mut mx);
            unsharp(frame.plane(c), &mut tmp, w, h, 1.6, 1.4 * self.strength);
            frame.plane_mut(c).copy_from_slice(&tmp);
            clamp_towards(frame.plane_mut(c), &mn, &mx, 0.7, 0.02);
        }
        out(frame)
    }
}

pub struct HdrStage {
    spec: HdrSpec,
}

impl Stage for HdrStage {
    fn name(&self) -> String {
        "sdr to hdr".into()
    }
    fn describe(&self) -> String {
        format!("inverse tone mapping to {:?}, peak {} nits (CPU)", self.spec.transfer, self.spec.peak_nits)
    }
    fn push(&mut self, mut frame: Frame, out: &mut Emit) -> Result<()> {
        inverse_tone_map(&mut frame, &self.spec);
        out(frame)
    }
}

pub struct GrainStage {
    amount: f32,
    size: f32,
    n: u64,
}

impl Stage for GrainStage {
    fn name(&self) -> String {
        "grain".into()
    }
    fn describe(&self) -> String {
        format!("film grain {:.3} size {:.1} (CPU)", self.amount, self.size)
    }
    fn push(&mut self, mut frame: Frame, out: &mut Emit) -> Result<()> {
        self.n += 1;
        add_grain(&mut frame, self.amount, self.size, self.n);
        out(frame)
    }
}

/// Builds the stage(s) for frames of `w x h` entering `spec`. Enhancement yields two stages
/// (upscale, then post-processing) so they run concurrently.
pub fn build(spec: &StageSpec, w: usize, h: usize, catalog: &Catalog, device: &DeviceSettings) -> Result<Vec<Box<dyn Stage>>> {
    Ok(match spec {
        StageSpec::Enhance(e) => {
            if (e.in_width as usize, e.in_height as usize) != (w, h) {
                bail!("enhance stage expects {}x{} but receives {w}x{h}", e.in_width, e.in_height);
            }
            let (front, back) = enhance::build(e, catalog, device)?;
            return Ok(vec![Box::new(EnhanceStage(front)), Box::new(EnhancePostStage(back))]);
        }
        StageSpec::Interpolate(i) => {
            let model = catalog.get(&i.model).ok_or_else(|| anyhow!("unknown model {}", i.model))?;
            let (method, description) = match (&model.implementation, model.id.as_str()) {
                (Implementation::Onnx { .. }, _) => {
                    let path = model.onnx_file(&catalog.dir, device.fp16).ok_or_else(|| anyhow!("{} not installed", model.id))?;
                    let m = OnnxInterpolator::new(&path, w, h, device)?;
                    let d = m.description.clone();
                    (Method::Onnx(Box::new(m)), d)
                }
                (_, "prism-blend") => (Method::Blend, "frame blending (CPU)".to_string()),
                _ => (Method::Flow, format!("bidirectional block motion, max {} px wide, occlusion-aware warp (CPU)", crate::interp::ME_WIDTH)),
            };
            let description = format!("{description}; {} -> {} fps, slow motion x{}", i.in_fps, i.out_fps, i.slowmo);
            vec![Box::new(InterpolateStage { inner: Interpolator::new(i, method), description })]
        }
        StageSpec::MotionDeblur { strength } => vec![Box::new(DeblurStage { strength: *strength })],
        StageSpec::Hdr(h) => vec![Box::new(HdrStage { spec: h.clone() })],
        StageSpec::Grain { amount, size } => vec![Box::new(GrainStage { amount: *amount, size: *size, n: 0 })],
    })
}

/// Size of frames leaving a stage given its input size.
pub fn output_size(spec: &StageSpec, w: usize, h: usize) -> (usize, usize) {
    match spec {
        StageSpec::Enhance(e) => (e.out_width as usize, e.out_height as usize),
        _ => (w, h),
    }
}
