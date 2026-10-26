//! Separable resampling with precomputed kernels.
//!
//! Each output sample stores a start index and a fixed-length weight window, with out-of-range
//! taps folded onto the edge samples, so the inner loops never bounds-check against the borders.
//! Downscaling widens the kernel (antialiasing). The pass order is chosen by cost.

use rayon::prelude::*;

use crate::frame::{Frame, Scratch};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Filter {
    Lanczos3,
    /// Catmull-Rom; slightly softer than Lanczos, no second ringing lobe.
    Bicubic,
    Bilinear,
}

impl Filter {
    fn radius(self) -> f64 {
        match self {
            Filter::Lanczos3 => 3.0,
            Filter::Bicubic => 2.0,
            Filter::Bilinear => 1.0,
        }
    }

    fn eval(self, x: f64) -> f64 {
        let ax = x.abs();
        match self {
            Filter::Lanczos3 => {
                if ax < 1e-9 {
                    1.0
                } else if ax >= 3.0 {
                    0.0
                } else {
                    let px = std::f64::consts::PI * x;
                    3.0 * px.sin() * (px / 3.0).sin() / (px * px)
                }
            }
            Filter::Bicubic => {
                // Catmull-Rom (B=0, C=0.5).
                if ax < 1.0 {
                    1.5 * ax * ax * ax - 2.5 * ax * ax + 1.0
                } else if ax < 2.0 {
                    -0.5 * ax * ax * ax + 2.5 * ax * ax - 4.0 * ax + 2.0
                } else {
                    0.0
                }
            }
            Filter::Bilinear => (1.0 - ax).max(0.0),
        }
    }
}

pub struct Kernel {
    pub taps: usize,
    pub starts: Vec<usize>,
    pub weights: Vec<f32>,
}

/// Kernels are rebuilt for every frame otherwise (with a `sin` per tap); sizes repeat per job.
pub fn cached_kernel(in_len: usize, out_len: usize, filter: Filter) -> std::sync::Arc<Kernel> {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex, OnceLock};
    static CACHE: OnceLock<Mutex<HashMap<(usize, usize, u8), Arc<Kernel>>>> = OnceLock::new();
    let key = (in_len, out_len, filter as u8);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(k) = cache.lock().unwrap().get(&key) {
        return k.clone();
    }
    let k = Arc::new(Kernel::new(in_len, out_len, filter));
    let mut c = cache.lock().unwrap();
    if c.len() > 256 {
        c.clear();
    }
    c.insert(key, k.clone());
    k
}

impl Kernel {
    pub fn new(in_len: usize, out_len: usize, filter: Filter) -> Kernel {
        assert!(in_len > 0 && out_len > 0);
        let scale = out_len as f64 / in_len as f64;
        let stretch = if scale < 1.0 { 1.0 / scale } else { 1.0 };
        let radius = filter.radius() * stretch;
        let mut windows: Vec<(usize, Vec<f64>)> = Vec::with_capacity(out_len);
        let mut taps = 1;
        for o in 0..out_len {
            let center = (o as f64 + 0.5) / scale - 0.5;
            let lo = (center - radius).ceil() as isize;
            let hi = (center + radius).floor() as isize;
            let mut acc: Vec<(usize, f64)> = Vec::new();
            let mut total = 0.0;
            for i in lo..=hi {
                let wgt = filter.eval((i as f64 - center) / stretch);
                if wgt == 0.0 {
                    continue;
                }
                let idx = i.clamp(0, in_len as isize - 1) as usize;
                total += wgt;
                match acc.last_mut() {
                    Some(last) if last.0 == idx => last.1 += wgt,
                    _ => acc.push((idx, wgt)),
                }
            }
            if acc.is_empty() {
                let idx = center.round().clamp(0.0, in_len as f64 - 1.0) as usize;
                acc.push((idx, 1.0));
                total = 1.0;
            }
            let start = acc.iter().map(|a| a.0).min().unwrap();
            let end = acc.iter().map(|a| a.0).max().unwrap();
            let mut w = vec![0.0; end - start + 1];
            for (idx, wgt) in acc {
                w[idx - start] += wgt / total;
            }
            taps = taps.max(w.len());
            windows.push((start, w));
        }
        let taps = taps.min(in_len);
        let mut starts = Vec::with_capacity(out_len);
        let mut weights = vec![0.0f32; out_len * taps];
        for (o, (start, w)) in windows.into_iter().enumerate() {
            // Shift the window left if it would run past the end; extra slots carry zero weight.
            let s = start.min(in_len - taps);
            for (k, wgt) in w.iter().enumerate() {
                weights[o * taps + (start - s) + k] = *wgt as f32;
            }
            starts.push(s);
        }
        Kernel { taps, starts, weights }
    }
}

fn horizontal(src: &[f32], sw: usize, rows: usize, dst: &mut [f32], k: &Kernel) {
    let dw = k.starts.len();
    let taps = k.taps;
    dst[..rows * dw].par_chunks_mut(dw).enumerate().for_each(|(y, out)| {
        let row = &src[y * sw..(y + 1) * sw];
        for (x, o) in out.iter_mut().enumerate() {
            let s = k.starts[x];
            let w = &k.weights[x * taps..(x + 1) * taps];
            let seg = &row[s..s + taps];
            let mut acc = 0.0f32;
            for t in 0..taps {
                acc += w[t] * seg[t];
            }
            *o = acc;
        }
    });
}

/// Output rows per parallel task in vertical passes. Neighbouring output rows share input rows,
/// so banding keeps those rows in cache instead of refetching them per row.
pub const BAND: usize = 16;

fn vertical(src: &[f32], width: usize, dst: &mut [f32], k: &Kernel) {
    let taps = k.taps;
    dst.par_chunks_mut(width * BAND).enumerate().for_each(|(band, rows)| {
        for (r, out) in rows.chunks_mut(width).enumerate() {
            let y = band * BAND + r;
            let s = k.starts[y];
            let w = &k.weights[y * taps..(y + 1) * taps];
            let first = &src[s * width..(s + 1) * width];
            for (o, v) in out.iter_mut().zip(first) {
                *o = w[0] * v;
            }
            for t in 1..taps {
                let wt = w[t];
                if wt == 0.0 {
                    continue;
                }
                let row = &src[(s + t) * width..(s + t + 1) * width];
                for (o, v) in out.iter_mut().zip(row) {
                    *o += wt * v;
                }
            }
        }
    });
}

/// Resizes one plane with prebuilt kernels.
pub fn resize_plane_with(src: &[f32], sw: usize, sh: usize, dst: &mut [f32], kx: &Kernel, ky: &Kernel) {
    let (dw, dh) = (kx.starts.len(), ky.starts.len());
    assert_eq!(src.len(), sw * sh);
    assert_eq!(dst.len(), dw * dh);
    if (sw, sh) == (dw, dh) {
        dst.copy_from_slice(src);
        return;
    }
    if sw == dw {
        vertical(src, sw, dst, ky);
        return;
    }
    if sh == dh {
        horizontal(src, sw, sh, dst, kx);
        return;
    }
    let h_first = sh * dw * kx.taps + dh * dw * ky.taps;
    let v_first = dh * sw * ky.taps + dh * dw * kx.taps;
    if h_first <= v_first {
        let mut tmp = Scratch::new(sh * dw);
        horizontal(src, sw, sh, &mut tmp, kx);
        vertical(&tmp, dw, dst, ky);
    } else {
        let mut tmp = Scratch::new(dh * sw);
        vertical(src, sw, &mut tmp, ky);
        horizontal(&tmp, sw, dh, dst, kx);
    }
}

/// Resamples `src` to `dst`'s size and blends it into `dst`: `dst += (resampled - dst) * alpha`.
/// Fused so the resampled frame is never materialised at full resolution.
pub fn resize_blend_into(src: &Frame, dst: &mut Frame, alpha: f32, filter: Filter) {
    let (sw, sh, dw, dh) = (src.width, src.height, dst.width, dst.height);
    let kx = cached_kernel(sw, dw, filter);
    let ky = cached_kernel(sh, dh, filter);
    let taps = ky.taps;
    let mut tmp = Scratch::new(sh * dw);
    for c in 0..3 {
        horizontal(src.plane(c), sw, sh, &mut tmp, &kx);
        let tmp = &tmp;
        dst.plane_mut(c).par_chunks_mut(dw * BAND).enumerate().for_each(|(band, rows)| {
            let mut acc = vec![0.0f32; dw];
            for (r, out) in rows.chunks_mut(dw).enumerate() {
                let y = band * BAND + r;
                let s = ky.starts[y];
                let w = &ky.weights[y * taps..(y + 1) * taps];
                acc.fill(0.0);
                for t in 0..taps {
                    let wt = w[t];
                    if wt == 0.0 {
                        continue;
                    }
                    for (a, v) in acc.iter_mut().zip(&tmp[(s + t) * dw..(s + t + 1) * dw]) {
                        *a += wt * v;
                    }
                }
                for (o, a) in out.iter_mut().zip(&acc) {
                    *o += (a - *o) * alpha;
                }
            }
        });
    }
}

/// Clamps `dst` toward the bilinearly upsampled `[lo, hi]` maps (given at `sw x sh`), computing
/// the upsampled bounds row by row instead of materialising them.
pub fn clamp_to_upsampled(dst: &mut [f32], dw: usize, dh: usize, lo: &[f32], hi: &[f32], sw: usize, sh: usize, strength: f32, slack: f32) {
    let sx = sw as f32 / dw as f32;
    let sy = sh as f32 / dh as f32;
    let xs: Vec<(usize, usize, f32)> = (0..dw)
        .map(|x| {
            let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (sw - 1) as f32);
            let x0 = fx as usize;
            (x0, (x0 + 1).min(sw - 1), fx - x0 as f32)
        })
        .collect();
    let row_of = |y: usize| -> (usize, usize, f32) {
        let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (sh - 1) as f32);
        let y0 = fy as usize;
        ((y0), (y0 + 1).min(sh - 1), fy - y0 as f32)
    };
    // Per band: widen the few source rows it touches once, then each output row is a
    // vectorisable vertical blend + clamp.
    dst.par_chunks_mut(dw * BAND).enumerate().for_each(|(band, rows)| {
        let first = row_of(band * BAND).0;
        let last = row_of((band * BAND + rows.len() / dw).saturating_sub(1).min(dh - 1)).1;
        let n = last - first + 1;
        let mut wl = vec![0.0f32; n * dw];
        let mut wh = vec![0.0f32; n * dw];
        for j in 0..n {
            let sr = first + j;
            let (lsrc, hsrc) = (&lo[sr * sw..(sr + 1) * sw], &hi[sr * sw..(sr + 1) * sw]);
            for (x, &(a, b, wx)) in xs.iter().enumerate() {
                wl[j * dw + x] = lsrc[a] + (lsrc[b] - lsrc[a]) * wx;
                wh[j * dw + x] = hsrc[a] + (hsrc[b] - hsrc[a]) * wx;
            }
        }
        for (k, row) in rows.chunks_mut(dw).enumerate() {
            let (y0, y1, wy) = row_of(band * BAND + k);
            let (l0, l1) = (&wl[(y0 - first) * dw..][..dw], &wl[(y1 - first) * dw..][..dw]);
            let (h0, h1) = (&wh[(y0 - first) * dw..][..dw], &wh[(y1 - first) * dw..][..dw]);
            for x in 0..dw {
                let l = l0[x] + (l1[x] - l0[x]) * wy;
                let h = h0[x] + (h1[x] - h0[x]) * wy;
                let d = row[x];
                let c = d.max(l - slack).min(h + slack);
                row[x] = d + (c - d) * strength;
            }
        }
    });
}

pub fn resize_plane(src: &[f32], sw: usize, sh: usize, dst: &mut [f32], dw: usize, dh: usize, filter: Filter) {
    let kx = cached_kernel(sw, dw, filter);
    let ky = cached_kernel(sh, dh, filter);
    resize_plane_with(src, sw, sh, dst, &kx, &ky);
}

pub fn resize(src: &Frame, dw: usize, dh: usize, filter: Filter) -> Frame {
    if (src.width, src.height) == (dw, dh) {
        return src.duplicate();
    }
    let kx = cached_kernel(src.width, dw, filter);
    let ky = cached_kernel(src.height, dh, filter);
    let mut out = Frame::uninit(dw, dh);
    for c in 0..3 {
        resize_plane_with(src.plane(c), src.width, src.height, out.plane_mut(c), &kx, &ky);
    }
    out
}

/// Bilinear sample with clamped edges.
#[inline]
pub fn sample_bilinear(p: &[f32], w: usize, h: usize, x: f32, y: f32) -> f32 {
    let x = x.clamp(0.0, (w - 1) as f32);
    let y = y.clamp(0.0, (h - 1) as f32);
    let x0 = x as usize;
    let y0 = y as usize;
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let fx = x - x0 as f32;
    let fy = y - y0 as f32;
    let a = p[y0 * w + x0] + (p[y0 * w + x1] - p[y0 * w + x0]) * fx;
    let b = p[y1 * w + x0] + (p[y1 * w + x1] - p[y1 * w + x0]) * fx;
    a + (b - a) * fy
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_are_normalised_and_in_range() {
        for (i, o) in [(10, 37), (37, 10), (1, 5), (5, 1), (1920, 3840), (7, 7)] {
            let k = Kernel::new(i, o, Filter::Lanczos3);
            for x in 0..o {
                let s: f32 = k.weights[x * k.taps..(x + 1) * k.taps].iter().sum();
                assert!((s - 1.0).abs() < 1e-4, "{i}->{o} at {x}: {s}");
                assert!(k.starts[x] + k.taps <= i);
            }
        }
    }

    #[test]
    fn constant_stays_constant_and_ramp_is_preserved() {
        let (sw, sh) = (13, 9);
        let ramp: Vec<f32> = (0..sw * sh).map(|i| (i % sw) as f32 / sw as f32).collect();
        let mut out = vec![0.0; 40 * 30];
        resize_plane(&vec![0.5; sw * sh], sw, sh, &mut out, 40, 30, Filter::Lanczos3);
        assert!(out.iter().all(|v| (v - 0.5).abs() < 1e-5));
        resize_plane(&ramp, sw, sh, &mut out, 40, 30, Filter::Bilinear);
        // Monotonic along x in the interior.
        for y in 0..30 {
            for x in 4..35 {
                assert!(out[y * 40 + x + 1] >= out[y * 40 + x] - 1e-6);
            }
        }
    }

    #[test]
    fn downscale_by_two_averages() {
        let src: Vec<f32> = (0..16).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let mut out = vec![0.0; 8];
        resize_plane(&src, 16, 1, &mut out, 8, 1, Filter::Bilinear);
        assert!(out[2..6].iter().all(|v| (v - 0.5).abs() < 1e-5), "{out:?}");
    }
}
