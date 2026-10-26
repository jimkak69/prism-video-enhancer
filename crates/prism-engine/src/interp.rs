//! Frame interpolation: motion estimation, warping and the frame-rate scheduler.
//!
//! Motion ("Flow" model) is estimated on downsampled luma with hierarchical block matching in
//! both directions. Intermediate frames warp both neighbours toward time `t` using the Super
//! SloMo approximation of intermediate flow, and blend them with visibility weights derived
//! from forward/backward consistency, so content appearing or disappearing comes from the frame
//! where it is visible. The scheduler handles arbitrary rate ratios, slow motion, scene cuts
//! (no blending across cuts) and runs of duplicated frames (re-timed by interpolation).

use std::sync::Arc;

use anyhow::Result;
use prism_core::plan::InterpolateSpec;
use rayon::prelude::*;

use crate::ai::runtime::OnnxInterpolator;
use crate::frame::Frame;
use crate::ops::analysis::{SmallLuma, mad};
use crate::ops::resample::sample_bilinear;

const BLOCK: usize = 8;
/// Motion is estimated on luma at most this wide.
pub const ME_WIDTH: usize = 1280;
/// Longer runs of identical frames are treated as genuinely static content.
const MAX_DUP_RUN: u32 = 6;
const EPS: f64 = 1e-6;

struct Level {
    data: Vec<f32>,
    w: usize,
    h: usize,
}

fn downsample(l: &Level) -> Level {
    let (w, h) = (l.w / 2, l.h / 2);
    let mut data = vec![0.0; w * h];
    data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        for (x, o) in row.iter_mut().enumerate() {
            let i = 2 * y * l.w + 2 * x;
            *o = 0.25 * (l.data[i] + l.data[i + 1] + l.data[i + l.w] + l.data[i + l.w + 1]);
        }
    });
    Level { data, w, h }
}

fn pyramid(base: &SmallLuma) -> Vec<Level> {
    let mut v = vec![Level { data: base.data.clone(), w: base.width, h: base.height }];
    while v.len() < 5 {
        let l = v.last().unwrap();
        if l.w / 2 < 4 * BLOCK || l.h / 2 < 4 * BLOCK {
            break;
        }
        let d = downsample(l);
        v.push(d);
    }
    v
}

/// Block motion field; vectors in pixels of its level, block centres at `b * BLOCK + BLOCK / 2`.
#[derive(Clone)]
pub struct Field {
    pub bw: usize,
    pub bh: usize,
    pub v: Vec<[f32; 2]>,
    /// Mean absolute difference of the best match per block.
    pub cost: Vec<f32>,
}

impl Field {
    /// Bilinear vector at pixel position (x, y) of the field's level.
    #[inline]
    pub fn at(&self, x: f32, y: f32) -> [f32; 2] {
        let u = ((x - (BLOCK as f32 / 2.0 - 0.5)) / BLOCK as f32).clamp(0.0, (self.bw - 1) as f32);
        let v = ((y - (BLOCK as f32 / 2.0 - 0.5)) / BLOCK as f32).clamp(0.0, (self.bh - 1) as f32);
        let (u0, v0) = (u as usize, v as usize);
        let (u1, v1) = ((u0 + 1).min(self.bw - 1), (v0 + 1).min(self.bh - 1));
        let (fu, fv) = (u - u0 as f32, v - v0 as f32);
        let g = |bx: usize, by: usize| self.v[by * self.bw + bx];
        let (a, b, c, d) = (g(u0, v0), g(u1, v0), g(u0, v1), g(u1, v1));
        let mut out = [0.0; 2];
        for k in 0..2 {
            let top = a[k] + (b[k] - a[k]) * fu;
            let bot = c[k] + (d[k] - c[k]) * fu;
            out[k] = top + (bot - top) * fv;
        }
        out
    }
}

#[inline]
fn sad_int(a: &Level, b: &Level, x0: usize, y0: usize, dx: isize, dy: isize) -> f32 {
    let mut s = 0.0f32;
    let inside = x0 as isize + dx >= 0 && y0 as isize + dy >= 0 && x0 as isize + dx + BLOCK as isize <= b.w as isize && y0 as isize + dy + BLOCK as isize <= b.h as isize;
    if inside {
        for j in 0..BLOCK {
            let ra = &a.data[(y0 + j) * a.w + x0..][..BLOCK];
            let rb = &b.data[((y0 + j) as isize + dy) as usize * b.w + (x0 as isize + dx) as usize..][..BLOCK];
            for i in 0..BLOCK {
                s += (ra[i] - rb[i]).abs();
            }
        }
    } else {
        for j in 0..BLOCK {
            let yb = ((y0 + j) as isize + dy).clamp(0, b.h as isize - 1) as usize;
            for i in 0..BLOCK {
                let xb = ((x0 + i) as isize + dx).clamp(0, b.w as isize - 1) as usize;
                s += (a.data[(y0 + j) * a.w + x0 + i] - b.data[yb * b.w + xb]).abs();
            }
        }
    }
    s / (BLOCK * BLOCK) as f32
}

#[inline]
fn sad_frac(a: &Level, b: &Level, x0: usize, y0: usize, dx: f32, dy: f32) -> f32 {
    let mut s = 0.0f32;
    for j in 0..BLOCK {
        for i in 0..BLOCK {
            let va = a.data[(y0 + j) * a.w + x0 + i];
            let vb = sample_bilinear(&b.data, b.w, b.h, (x0 + i) as f32 + dx, (y0 + j) as f32 + dy);
            s += (va - vb).abs();
        }
    }
    s / (BLOCK * BLOCK) as f32
}

fn median3(mut v: [f32; 9], n: usize) -> f32 {
    let s = &mut v[..n];
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[n / 2]
}

fn median_filter(f: &mut Field) {
    let src = f.v.clone();
    let (bw, bh) = (f.bw, f.bh);
    f.v.par_chunks_mut(bw).enumerate().for_each(|(by, row)| {
        for (bx, o) in row.iter_mut().enumerate() {
            let mut xs = [0.0f32; 9];
            let mut ys = [0.0f32; 9];
            let mut n = 0;
            for dy in -1isize..=1 {
                for dx in -1isize..=1 {
                    let (x, y) = (bx as isize + dx, by as isize + dy);
                    if x >= 0 && y >= 0 && (x as usize) < bw && (y as usize) < bh {
                        let v = src[y as usize * bw + x as usize];
                        xs[n] = v[0];
                        ys[n] = v[1];
                        n += 1;
                    }
                }
            }
            *o = [median3(xs, n), median3(ys, n)];
        }
    });
}

/// Estimates motion from `a` to `b` (for each block of `a`, where it moved to in `b`).
fn estimate(pa: &[Level], pb: &[Level]) -> Field {
    let mut coarse: Option<Field> = None;
    for lvl in (0..pa.len()).rev() {
        let (a, b) = (&pa[lvl], &pb[lvl]);
        let (bw, bh) = ((a.w / BLOCK).max(1), (a.h / BLOCK).max(1));
        let lambda = 0.004f32;
        let results: Vec<([f32; 2], f32)> = (0..bw * bh)
            .into_par_iter()
            .map(|i| {
                let (bx, by) = (i % bw, i / bw);
                let (x0, y0) = ((bx * BLOCK).min(a.w - BLOCK.min(a.w)), (by * BLOCK).min(a.h - BLOCK.min(a.h)));
                let mut best = ([0.0f32, 0.0f32], f32::MAX);
                match &coarse {
                    None => {
                        let r = 6isize;
                        for dy in -r..=r {
                            for dx in -r..=r {
                                let c = sad_int(a, b, x0, y0, dx, dy) + lambda * 0.25 * (dx.abs() + dy.abs()) as f32;
                                if c < best.1 {
                                    best = ([dx as f32, dy as f32], c);
                                }
                            }
                        }
                    }
                    Some(cf) => {
                        let (pbx, pby) = ((bx / 2).min(cf.bw - 1), (by / 2).min(cf.bh - 1));
                        let parent = cf.v[pby * cf.bw + pbx];
                        let pred = [parent[0] * 2.0, parent[1] * 2.0];
                        let mut cands: Vec<[isize; 2]> = vec![[0, 0]];
                        for (ox, oy) in [(0isize, 0isize), (-1, 0), (1, 0), (0, -1), (0, 1)] {
                            let (nx, ny) = (pbx as isize + ox, pby as isize + oy);
                            if nx >= 0 && ny >= 0 && (nx as usize) < cf.bw && (ny as usize) < cf.bh {
                                let v = cf.v[ny as usize * cf.bw + nx as usize];
                                let c = [(v[0] * 2.0).round() as isize, (v[1] * 2.0).round() as isize];
                                if !cands.contains(&c) {
                                    cands.push(c);
                                }
                            }
                        }
                        for c in cands {
                            for dy in -1..=1 {
                                for dx in -1..=1 {
                                    let (vx, vy) = (c[0] + dx, c[1] + dy);
                                    let pen = lambda * ((vx as f32 - pred[0]).abs() + (vy as f32 - pred[1]).abs());
                                    let cost = sad_int(a, b, x0, y0, vx, vy) + pen;
                                    if cost < best.1 {
                                        best = ([vx as f32, vy as f32], cost);
                                    }
                                }
                            }
                        }
                    }
                }
                if lvl == 0 {
                    // Half-pixel refinement.
                    let base = best.0;
                    let mut refined = (base, sad_int(a, b, x0, y0, base[0] as isize, base[1] as isize));
                    for (hx, hy) in [(-0.5f32, 0.0f32), (0.5, 0.0), (0.0, -0.5), (0.0, 0.5), (-0.5, -0.5), (0.5, -0.5), (-0.5, 0.5), (0.5, 0.5)] {
                        let c = sad_frac(a, b, x0, y0, base[0] + hx, base[1] + hy);
                        if c < refined.1 {
                            refined = ([base[0] + hx, base[1] + hy], c);
                        }
                    }
                    refined
                } else {
                    best
                }
            })
            .collect();
        let mut f = Field { bw, bh, v: results.iter().map(|r| r.0).collect(), cost: results.iter().map(|r| r.1).collect() };
        median_filter(&mut f);
        coarse = Some(f);
    }
    coarse.unwrap()
}

/// Motion between two frames in both directions plus per-block occlusion measures.
pub struct PairMotion {
    fwd: Field,
    bwd: Field,
    occ_a: Vec<f32>,
    occ_b: Vec<f32>,
    /// Full-resolution pixels per motion pixel.
    factor: f32,
    /// Mean compensated residual of the forward field.
    pub residual: f32,
}

fn consistency(f: &Field, g: &Field) -> Vec<f32> {
    (0..f.bw * f.bh)
        .into_par_iter()
        .map(|i| {
            let (bx, by) = (i % f.bw, i / f.bw);
            let (cx, cy) = ((bx * BLOCK) as f32 + BLOCK as f32 / 2.0 - 0.5, (by * BLOCK) as f32 + BLOCK as f32 / 2.0 - 0.5);
            let v = f.v[i];
            let back = g.at(cx + v[0], cy + v[1]);
            ((v[0] + back[0]).powi(2) + (v[1] + back[1]).powi(2)).sqrt()
        })
        .collect()
}

impl PairMotion {
    pub fn estimate(a: &SmallLuma, b: &SmallLuma) -> PairMotion {
        let (pa, pb) = (pyramid(a), pyramid(b));
        let (fwd, bwd) = rayon::join(|| estimate(&pa, &pb), || estimate(&pb, &pa));
        let (occ_a, occ_b) = rayon::join(|| consistency(&fwd, &bwd), || consistency(&bwd, &fwd));
        let residual = fwd.cost.iter().sum::<f32>() / fwd.cost.len().max(1) as f32;
        PairMotion { fwd, bwd, occ_a, occ_b, factor: a.factor as f32, residual }
    }

    fn occ_at(&self, occ: &[f32], x: f32, y: f32) -> f32 {
        let f = &self.fwd;
        let u = ((x - (BLOCK as f32 / 2.0 - 0.5)) / BLOCK as f32).clamp(0.0, (f.bw - 1) as f32);
        let v = ((y - (BLOCK as f32 / 2.0 - 0.5)) / BLOCK as f32).clamp(0.0, (f.bh - 1) as f32);
        sample_bilinear(occ, f.bw, f.bh, u, v)
    }

    /// Synthesises the frame at time `t` between `a` (t = 0) and `b` (t = 1).
    pub fn warp(&self, a: &Frame, b: &Frame, t: f32) -> Frame {
        let (w, h) = (a.width, a.height);
        let mut out = Frame::uninit(w, h);
        let f = self.factor;
        let inv = 1.0 / f;
        let (pa, pb) = (a.planes(), b.planes());
        let tau = 1.0f32;
        let n = w * h;
        let data = out.data_mut();
        let (r, rest) = data.split_at_mut(n);
        let (g, bl) = rest.split_at_mut(n);
        r.par_chunks_mut(w).zip(g.par_chunks_mut(w)).zip(bl.par_chunks_mut(w)).enumerate().for_each(|(y, ((rr, gr), br))| {
            let sy = (y as f32 + 0.5) * inv - 0.5;
            for x in 0..w {
                let sx = (x as f32 + 0.5) * inv - 0.5;
                let fv = self.fwd.at(sx, sy);
                let gv = self.bwd.at(sx, sy);
                // Intermediate flows (full-resolution pixels).
                let t0 = [(-(1.0 - t) * t * fv[0] + t * t * gv[0]) * f, (-(1.0 - t) * t * fv[1] + t * t * gv[1]) * f];
                let t1 = [((1.0 - t) * (1.0 - t) * fv[0] - t * (1.0 - t) * gv[0]) * f, ((1.0 - t) * (1.0 - t) * fv[1] - t * (1.0 - t) * gv[1]) * f];
                let (x0, y0) = (x as f32 + t0[0], y as f32 + t0[1]);
                let (x1, y1) = (x as f32 + t1[0], y as f32 + t1[1]);
                let c0 = self.occ_at(&self.occ_a, (x0 + 0.5) * inv - 0.5, (y0 + 0.5) * inv - 0.5) / tau;
                let c1 = self.occ_at(&self.occ_b, (x1 + 0.5) * inv - 0.5, (y1 + 0.5) * inv - 0.5) / tau;
                let w0 = (1.0 - t) / (1.0 + c0 * c0);
                let w1 = t / (1.0 + c1 * c1);
                let norm = 1.0 / (w0 + w1).max(1e-6);
                let (w0, w1) = (w0 * norm, w1 * norm);
                rr[x] = w0 * sample_bilinear(pa[0], w, h, x0, y0) + w1 * sample_bilinear(pb[0], w, h, x1, y1);
                gr[x] = w0 * sample_bilinear(pa[1], w, h, x0, y0) + w1 * sample_bilinear(pb[1], w, h, x1, y1);
                br[x] = w0 * sample_bilinear(pa[2], w, h, x0, y0) + w1 * sample_bilinear(pb[2], w, h, x1, y1);
            }
        });
        out
    }
}

pub fn blend(a: &Frame, b: &Frame, t: f32) -> Frame {
    let mut out = Frame::uninit(a.width, a.height);
    out.data_mut().par_chunks_mut(1 << 14).zip(a.data().par_chunks(1 << 14).zip(b.data().par_chunks(1 << 14))).for_each(|(o, (x, y))| {
        for i in 0..o.len() {
            o[i] = x[i] + (y[i] - x[i]) * t;
        }
    });
    out
}

pub enum Method {
    Flow,
    Blend,
    Onnx(Box<OnnxInterpolator>),
}

struct Anchor {
    pos: f64,
    frame: Arc<Frame>,
    luma: SmallLuma,
}

/// Emits frames at the output rate, interpolating between "anchor" (non-duplicate) input frames.
pub struct Interpolator {
    method: Method,
    /// Input frames advanced per output frame.
    step: f64,
    next_out: u64,
    in_count: u64,
    anchor: Option<Anchor>,
    dup_run: u32,
    dup_mad: f32,
    scene_detection: bool,
    pub scene_cuts: u64,
    pub duplicates: u64,
}

impl Interpolator {
    pub fn new(spec: &InterpolateSpec, method: Method) -> Interpolator {
        let step = spec.in_fps.as_f64() / (spec.out_fps.as_f64() * spec.slowmo);
        Interpolator {
            method,
            step,
            next_out: 0,
            in_count: 0,
            anchor: None,
            dup_run: 0,
            dup_mad: spec.duplicate_mad,
            scene_detection: spec.scene_detection,
            scene_cuts: 0,
            duplicates: 0,
        }
    }

    pub fn push(&mut self, frame: Frame, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        let pos = self.in_count as f64;
        self.in_count += 1;
        let luma = SmallLuma::from_frame(&frame, ME_WIDTH);
        let frame = Arc::new(frame);
        let Some(prev) = self.anchor.take() else {
            self.anchor = Some(Anchor { pos, frame, luma });
            return Ok(());
        };
        if self.dup_mad >= 0.0 && self.dup_run < MAX_DUP_RUN && mad(&prev.luma.data, &luma.data) < self.dup_mad {
            self.dup_run += 1;
            self.duplicates += 1;
            self.anchor = Some(prev);
            return Ok(());
        }
        self.dup_run = 0;
        let next = Anchor { pos, frame, luma };
        self.emit_between(&prev, &next, out)?;
        self.anchor = Some(next);
        Ok(())
    }

    fn emit_between(&mut self, a: &Anchor, b: &Anchor, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        let mut motion: Option<PairMotion> = None;
        let mut cut: Option<bool> = None;
        while (self.next_out as f64) * self.step < b.pos - EPS {
            let p = self.next_out as f64 * self.step;
            let t = ((p - a.pos) / (b.pos - a.pos)).clamp(0.0, 1.0) as f32;
            let frame = if (t as f64) < EPS {
                a.frame.duplicate()
            } else {
                match &mut self.method {
                    Method::Blend => blend(&a.frame, &b.frame, t),
                    Method::Flow => {
                        let m = motion.get_or_insert_with(|| PairMotion::estimate(&a.luma, &b.luma));
                        let is_cut = *cut.get_or_insert_with(|| self.scene_detection && m.residual > 0.07 && mad(&a.luma.data, &b.luma.data) > 0.08);
                        if is_cut {
                            if t < 0.5 { a.frame.duplicate() } else { b.frame.duplicate() }
                        } else {
                            m.warp(&a.frame, &b.frame, t)
                        }
                    }
                    Method::Onnx(model) => {
                        let is_cut = *cut.get_or_insert_with(|| self.scene_detection && mad(&a.luma.data, &b.luma.data) > 0.12);
                        if is_cut {
                            if t < 0.5 { a.frame.duplicate() } else { b.frame.duplicate() }
                        } else {
                            model.interpolate(&a.frame, &b.frame, t)?
                        }
                    }
                }
            };
            out(frame)?;
            self.next_out += 1;
        }
        if cut == Some(true) {
            self.scene_cuts += 1;
        }
        Ok(())
    }

    /// Emits the frames after the last anchor (trailing duplicates repeat it).
    pub fn finish(&mut self, out: &mut dyn FnMut(Frame) -> Result<()>) -> Result<()> {
        if let Some(a) = &self.anchor {
            while (self.next_out as f64) * self.step < self.in_count as f64 - EPS {
                out(a.frame.duplicate())?;
                self.next_out += 1;
            }
        }
        Ok(())
    }

    pub fn emitted(&self) -> u64 {
        self.next_out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use prism_core::media::Rational;

    /// A textured frame shifted by (dx, dy) pixels.
    fn textured(w: usize, h: usize, dx: f32, dy: f32) -> Frame {
        let mut f = Frame::uninit(w, h);
        for c in 0..3 {
            let p = f.plane_mut(c);
            for y in 0..h {
                for x in 0..w {
                    let (u, v) = (x as f32 - dx, y as f32 - dy);
                    p[y * w + x] = 0.5 + 0.2 * (u * 0.21 + c as f32).sin() * (v * 0.17).cos() + 0.1 * ((u * 0.05).sin() * (v * 0.07).sin());
                }
            }
        }
        f
    }

    fn spec(in_fps: i64, out_fps: i64, slowmo: f64, dup: f32) -> InterpolateSpec {
        InterpolateSpec {
            model: "prism-flow".into(),
            in_fps: Rational::new(in_fps, 1),
            out_fps: Rational::new(out_fps, 1),
            slowmo,
            duplicate_mad: dup,
            scene_detection: true,
            frames_in: 0,
            frames_out: 0,
        }
    }

    fn err(a: &Frame, b: &Frame, margin: usize) -> f32 {
        let (w, h) = (a.width, a.height);
        let mut s = 0.0;
        let mut n = 0;
        for c in 0..3 {
            for y in margin..h - margin {
                for x in margin..w - margin {
                    s += (a.plane(c)[y * w + x] - b.plane(c)[y * w + x]).abs();
                    n += 1;
                }
            }
        }
        s / n as f32
    }

    #[test]
    fn flow_recovers_translation_midpoint() {
        let (w, h) = (192, 128);
        let a = textured(w, h, 0.0, 0.0);
        let b = textured(w, h, 6.0, -4.0);
        let truth = textured(w, h, 3.0, -2.0);
        let m = PairMotion::estimate(&SmallLuma::from_frame(&a, ME_WIDTH), &SmallLuma::from_frame(&b, ME_WIDTH));
        let mid = m.warp(&a, &b, 0.5);
        let e_flow = err(&mid, &truth, 12);
        let e_blend = err(&blend(&a, &b, 0.5), &truth, 12);
        assert!(e_flow < e_blend * 0.35, "flow {e_flow} vs blend {e_blend}");
        assert!(m.residual < 0.02);
    }

    #[test]
    fn scheduler_counts_for_rate_changes() {
        for (inf, outf, slow, n_in, expected) in [(24, 60, 1.0, 48, 120), (30, 30, 2.0, 30, 60), (60, 30, 1.0, 60, 30), (25, 25, 1.0, 10, 10)] {
            let mut it = Interpolator::new(&spec(inf, outf, slow, -1.0), Method::Blend);
            let mut count = 0;
            let mut sink = |_f: Frame| -> Result<()> {
                count += 1;
                Ok(())
            };
            for i in 0..n_in {
                it.push(Frame::filled(8, 8, [i as f32 / n_in as f32; 3]), &mut sink).unwrap();
            }
            it.finish(&mut sink).unwrap();
            assert_eq!(count, expected, "{inf}->{outf} x{slow}");
        }
    }

    #[test]
    fn duplicates_are_retimed() {
        // Frames 0,0,0,3 (two duplicates) at the same rate become an even ramp 0,1,2,3.
        let mut it = Interpolator::new(&spec(30, 30, 1.0, 0.01), Method::Blend);
        let vals = [0.0f32, 0.0, 0.0, 0.3, 0.4];
        let mut got = Vec::new();
        let mut sink = |f: Frame| -> Result<()> {
            got.push(f.plane(0)[0]);
            Ok(())
        };
        for v in vals {
            it.push(Frame::filled(32, 32, [v; 3]), &mut sink).unwrap();
        }
        it.finish(&mut sink).unwrap();
        assert_eq!(got.len(), 5);
        for (g, e) in got.iter().zip([0.0, 0.1, 0.2, 0.3, 0.4]) {
            assert!((g - e).abs() < 1e-5, "{got:?}");
        }
        assert_eq!(it.duplicates, 2);
    }

    #[test]
    fn scene_cut_is_not_blended() {
        let (w, h) = (128, 96);
        let a = Frame::filled(w, h, [0.1; 3]);
        let mut b = textured(w, h, 0.0, 0.0);
        for v in b.data_mut() {
            *v = 1.0 - *v * 0.3;
        }
        let mut it = Interpolator::new(&spec(30, 60, 1.0, -1.0), Method::Flow);
        let mut got = Vec::new();
        let mut sink = |f: Frame| -> Result<()> {
            got.push(f.plane(0)[w * 40 + 40]);
            Ok(())
        };
        it.push(a, &mut sink).unwrap();
        it.push(b, &mut sink).unwrap();
        it.finish(&mut sink).unwrap();
        assert_eq!(it.scene_cuts, 1);
        assert!((got[1] - 0.1).abs() < 1e-6 || got[1] > 0.6, "middle frame is one side, not a blend: {got:?}");
    }
}
