//! Single-plane image filters. All are row-parallel and use clamped (replicated) borders.

use rayon::prelude::*;

use super::resample::BAND;
use crate::frame::Scratch;

#[inline]
fn clampi(v: isize, n: usize) -> usize {
    v.clamp(0, n as isize - 1) as usize
}

/// Box mean of radius `r` (window `2r+1`), separable.
pub fn box_blur(src: &[f32], dst: &mut [f32], w: usize, h: usize, r: usize) {
    if r == 0 {
        dst.copy_from_slice(src);
        return;
    }
    let mut tmp = Scratch::new(w * h);
    let inv = 1.0 / (2 * r + 1) as f32;
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        let mut sum = 0.0f32;
        for i in -(r as isize)..=(r as isize) {
            sum += row[clampi(i, w)];
        }
        for x in 0..w {
            out[x] = sum * inv;
            sum += row[clampi(x as isize + r as isize + 1, w)] - row[clampi(x as isize - r as isize, w)];
        }
    });
    let tmp = &tmp;
    // Running sum down each band of rows: O(1) per pixel and cache-friendly.
    dst.par_chunks_mut(w * BAND).enumerate().for_each(|(band, rows)| {
        let y0 = band * BAND;
        let mut acc = vec![0.0f32; w];
        for j in (y0 as isize - r as isize)..=(y0 as isize + r as isize) {
            for (a, v) in acc.iter_mut().zip(&tmp[clampi(j, h) * w..(clampi(j, h) + 1) * w]) {
                *a += v;
            }
        }
        for (k, out) in rows.chunks_mut(w).enumerate() {
            let y = (y0 + k) as isize;
            for (o, a) in out.iter_mut().zip(&acc) {
                *o = a * inv;
            }
            let add = &tmp[clampi(y + r as isize + 1, h) * w..][..w];
            let sub = &tmp[clampi(y - r as isize, h) * w..][..w];
            for ((a, p), m) in acc.iter_mut().zip(add).zip(sub) {
                *a += p - m;
            }
        }
    });
}

pub fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    let r = (sigma * 3.0).ceil().max(1.0) as usize;
    let mut k: Vec<f32> = (0..=2 * r).map(|i| (-((i as f32 - r as f32).powi(2)) / (2.0 * sigma * sigma)).exp()).collect();
    let s: f32 = k.iter().sum();
    k.iter_mut().for_each(|v| *v /= s);
    k
}

/// Separable Gaussian blur.
pub fn gaussian_blur(src: &[f32], dst: &mut [f32], w: usize, h: usize, sigma: f32) {
    if sigma <= 0.05 {
        dst.copy_from_slice(src);
        return;
    }
    let k = gaussian_kernel(sigma);
    let r = (k.len() / 2) as isize;
    let mut tmp = Scratch::new(w * h);
    tmp.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let mut acc = 0.0;
            if x as isize >= r && (x as isize + r) < w as isize {
                let seg = &row[x - r as usize..x + r as usize + 1];
                for (a, b) in seg.iter().zip(&k) {
                    acc += a * b;
                }
            } else {
                for (i, kv) in k.iter().enumerate() {
                    acc += kv * row[clampi(x as isize + i as isize - r, w)];
                }
            }
            out[x] = acc;
        }
    });
    let tmp = &tmp;
    dst.par_chunks_mut(w * BAND).enumerate().for_each(|(band, rows)| {
        for (j, out) in rows.chunks_mut(w).enumerate() {
            let y = band * BAND + j;
            out.fill(0.0);
            for (i, kv) in k.iter().enumerate() {
                let yy = clampi(y as isize + i as isize - r, h);
                for (o, v) in out.iter_mut().zip(&tmp[yy * w..(yy + 1) * w]) {
                    *o += kv * v;
                }
            }
        }
    });
}

/// Self-guided edge-preserving smoothing (He et al.). `eps` is the variance treated as noise.
pub fn guided_smooth(p: &[f32], out: &mut [f32], w: usize, h: usize, r: usize, eps: f32) {
    let n = w * h;
    let mut sq = Scratch::new(n);
    sq.par_iter_mut().zip(p.par_iter()).for_each(|(s, v)| *s = v * v);
    let mut mean = Scratch::new(n);
    let mut mean_sq = Scratch::new(n);
    box_blur(p, &mut mean, w, h, r);
    box_blur(&sq, &mut mean_sq, w, h, r);
    // a -> sq, b -> mean
    sq.par_iter_mut().zip(mean.par_iter_mut()).zip(mean_sq.par_iter()).for_each(|((a, m), ms)| {
        let var = (ms - *m * *m).max(0.0);
        let av = var / (var + eps);
        *a = av;
        *m -= av * *m;
    });
    let mut ma = Scratch::new(n);
    box_blur(&sq, &mut ma, w, h, r);
    box_blur(&mean, &mut mean_sq, w, h, r);
    out.par_iter_mut().zip(p.par_iter()).zip(ma.par_iter().zip(mean_sq.par_iter())).for_each(|((o, pv), (a, b))| *o = a * pv + b);
}

/// Contrast-adaptive sharpening on a cross neighbourhood: strong where local contrast is low,
/// gentle near strong edges, which keeps halos small. `amount` in 0..1.
pub fn cas(src: &[f32], dst: &mut [f32], w: usize, h: usize, amount: f32) {
    if amount <= 0.0 {
        dst.copy_from_slice(src);
        return;
    }
    let peak = -1.0 / (8.0 + (5.0 - 8.0) * amount.clamp(0.0, 1.0));
    dst.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let up = &src[clampi(y as isize - 1, h) * w..][..w];
        let mid = &src[y * w..][..w];
        let dn = &src[clampi(y as isize + 1, h) * w..][..w];
        for x in 0..w {
            let l = mid[x.saturating_sub(1)];
            let r = mid[(x + 1).min(w - 1)];
            let (a, c, e) = (up[x], mid[x], dn[x]);
            let mn = a.min(l).min(c).min(r).min(e).clamp(0.0, 1.0);
            let mx = a.max(l).max(c).max(r).max(e).clamp(0.0, 1.0);
            let amp = if mx > 1e-5 { (mn.min(1.0 - mx) / mx).clamp(0.0, 1.0).sqrt() } else { 0.0 };
            let wgt = amp * peak;
            out[x] = (c + wgt * (a + l + r + e)) / (1.0 + 4.0 * wgt);
        }
    });
}

/// `dst = src + amount * (src - gaussian(src, sigma))`.
pub fn unsharp(src: &[f32], dst: &mut [f32], w: usize, h: usize, sigma: f32, amount: f32) {
    if amount == 0.0 {
        dst.copy_from_slice(src);
        return;
    }
    let mut blur = Scratch::new(w * h);
    gaussian_blur_any(src, &mut blur, w, h, sigma);
    dst.par_iter_mut().zip(src.par_iter()).zip(blur.par_iter()).for_each(|((d, s), b)| *d = s + amount * (s - b));
}

/// Gaussian blur that stays cheap for large sigma: blurs a downsampled copy and interpolates
/// it back (a wide Gaussian has no high frequencies to lose).
pub fn gaussian_blur_any(src: &[f32], dst: &mut [f32], w: usize, h: usize, sigma: f32) {
    let k = (sigma / 1.5).floor() as usize;
    if k < 2 || w / k < 8 || h / k < 8 {
        gaussian_blur(src, dst, w, h, sigma);
        return;
    }
    use super::resample::{Filter, resize_plane};
    let (sw, sh) = (w.div_ceil(k), h.div_ceil(k));
    let mut small = Scratch::new(sw * sh);
    let mut blurred = Scratch::new(sw * sh);
    resize_plane(src, w, h, &mut small, sw, sh, Filter::Bilinear);
    // Downsampling by k already low-passes by roughly k/2 pixels; blur the remainder.
    let rest = ((sigma / k as f32).powi(2) - 0.25).max(0.09).sqrt();
    gaussian_blur(&small, &mut blurred, sw, sh, rest);
    resize_plane(&blurred, sw, sh, dst, w, h, Filter::Bilinear);
}

/// 3x3 local minimum and maximum.
pub fn local_minmax(src: &[f32], w: usize, h: usize, min_out: &mut [f32], max_out: &mut [f32]) {
    let mut rmin = Scratch::new(w * h);
    let mut rmax = Scratch::new(w * h);
    rmin.par_chunks_mut(w).zip(rmax.par_chunks_mut(w)).enumerate().for_each(|(y, (mn, mx))| {
        let row = &src[y * w..(y + 1) * w];
        for x in 0..w {
            let a = row[x.saturating_sub(1)];
            let b = row[x];
            let c = row[(x + 1).min(w - 1)];
            mn[x] = a.min(b).min(c);
            mx[x] = a.max(b).max(c);
        }
    });
    let (rmin, rmax) = (&rmin, &rmax);
    min_out.par_chunks_mut(w).zip(max_out.par_chunks_mut(w)).enumerate().for_each(|(y, (mn, mx))| {
        let ys = [clampi(y as isize - 1, h), y, clampi(y as isize + 1, h)];
        for x in 0..w {
            mn[x] = rmin[ys[0] * w + x].min(rmin[ys[1] * w + x]).min(rmin[ys[2] * w + x]);
            mx[x] = rmax[ys[0] * w + x].max(rmax[ys[1] * w + x]).max(rmax[ys[2] * w + x]);
        }
    });
}

/// Pulls overshoot back towards `[lo, hi]` with weight `strength` (anti-ringing / dehalo).
pub fn clamp_towards(dst: &mut [f32], lo: &[f32], hi: &[f32], strength: f32, slack: f32) {
    if strength <= 0.0 {
        return;
    }
    dst.par_iter_mut().zip(lo.par_iter().zip(hi.par_iter())).for_each(|(d, (l, h))| {
        let c = d.clamp(l - slack, h + slack);
        *d += (c - *d) * strength;
    });
}

/// Smooths 8x8 block boundaries whose step is small relative to the surrounding texture.
/// `strength` in 0..1 scales both the detection threshold and the correction.
pub fn deblock(plane: &mut [f32], w: usize, h: usize, strength: f32) {
    if strength <= 0.0 || w < 16 || h < 16 {
        return;
    }
    let thr = 0.012 + 0.07 * strength;
    let k = 0.35 + 0.65 * strength;
    let filt = |p1: f32, p0: f32, q0: f32, q1: f32| -> Option<(f32, f32)> {
        let d = q0 - p0;
        if d.abs() < thr && (p1 - p0).abs() < thr * 0.6 && (q1 - q0).abs() < thr * 0.6 {
            let delta = ((q0 - p0) * 4.0 + (p1 - q1)) / 8.0 * k;
            Some((delta, delta * 0.5))
        } else {
            None
        }
    };
    // Vertical edges (x = 8, 16, ...), row-parallel.
    plane.par_chunks_mut(w).for_each(|row| {
        let mut x = 8;
        while x + 1 < w {
            if let Some((d, d2)) = filt(row[x - 2], row[x - 1], row[x], row[x + 1]) {
                row[x - 1] += d;
                row[x] -= d;
                row[x - 2] += d2;
                row[x + 1] -= d2;
            }
            x += 8;
        }
    });
    // Horizontal edges (y = 8, 16, ...): chunks of 8 rows starting at row 4 each contain one edge.
    if h > 12 {
        let tail = &mut plane[4 * w..];
        tail.par_chunks_mut(8 * w).for_each(|chunk| {
            if chunk.len() < 6 * w {
                return;
            }
            let (top, bottom) = chunk.split_at_mut(4 * w);
            for x in 0..w {
                let (p1, p0, q0, q1) = (top[2 * w + x], top[3 * w + x], bottom[x], bottom[w + x]);
                if let Some((d, d2)) = filt(p1, p0, q0, q1) {
                    top[3 * w + x] += d;
                    bottom[x] -= d;
                    top[2 * w + x] += d2;
                    bottom[w + x] -= d2;
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_and_gaussian_preserve_constant_and_mean() {
        let (w, h) = (31, 17);
        let src: Vec<f32> = (0..w * h).map(|i| ((i * 37) % 11) as f32 / 10.0).collect();
        let mean: f32 = src.iter().sum::<f32>() / src.len() as f32;
        let mut out = vec![0.0; w * h];
        box_blur(&vec![0.25; w * h], &mut out, w, h, 3);
        assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-5));
        gaussian_blur(&src, &mut out, w, h, 1.5);
        let m2: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!((mean - m2).abs() < 0.02);
        let var = |v: &[f32], m: f32| v.iter().map(|x| (x - m).powi(2)).sum::<f32>();
        assert!(var(&out, m2) < var(&src, mean) * 0.5);
    }

    #[test]
    fn guided_smoothing_keeps_edges() {
        let (w, h) = (40, 20);
        let mut src = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                let noise = (((x * 7 + y * 13) % 5) as f32 - 2.0) * 0.01;
                src[y * w + x] = if x < w / 2 { 0.2 } else { 0.8 } + noise;
            }
        }
        let mut out = vec![0.0; w * h];
        guided_smooth(&src, &mut out, w, h, 3, 0.01 * 0.01);
        // Edge kept sharp, flat noise reduced.
        assert!(out[10 * w + w / 2 + 2] - out[10 * w + w / 2 - 3] > 0.5);
        let flat = |v: &[f32]| (0..10).map(|x| (v[10 * w + x] - 0.2).abs()).sum::<f32>();
        assert!(flat(&out) < flat(&src));
    }

    #[test]
    fn cas_sharpens_without_overshooting_flat_areas() {
        let (w, h) = (16, 4);
        let mut src = vec![0.5f32; w * h];
        for y in 0..h {
            for x in 8..w {
                src[y * w + x] = 0.6;
            }
        }
        let mut out = vec![0.0; w * h];
        cas(&src, &mut out, w, h, 1.0);
        assert!(out[w + 7] < 0.5 && out[w + 8] > 0.6, "edge contrast increased");
        assert!((out[w + 2] - 0.5).abs() < 1e-6, "flat area untouched");
    }

    #[test]
    fn deblock_reduces_block_steps_only() {
        let (w, h) = (32, 32);
        let mut p = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = ((x / 8) % 2) as f32 * 0.03 + if x >= 24 { 0.5 } else { 0.0 };
            }
        }
        let before_small = (p[5 * w + 8] - p[5 * w + 7]).abs();
        let before_big = (p[5 * w + 24] - p[5 * w + 23]).abs();
        deblock(&mut p, w, h, 0.8);
        assert!((p[5 * w + 8] - p[5 * w + 7]).abs() < before_small * 0.5);
        assert!(((p[5 * w + 24] - p[5 * w + 23]).abs() - before_big).abs() < 1e-6, "real edge kept");
    }
}
