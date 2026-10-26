//! Frame statistics: downsampled luma, differences, and the estimates behind "Auto" parameters.

use prism_core::plan::EnhanceParams;
use rayon::prelude::*;

use crate::frame::Frame;

/// Luma averaged over `factor x factor` blocks.
pub struct SmallLuma {
    pub data: Vec<f32>,
    pub width: usize,
    pub height: usize,
    /// Full-resolution pixels per small pixel.
    pub factor: usize,
}

impl SmallLuma {
    /// Downsamples so the width is at most `max_width`.
    pub fn from_frame(f: &Frame, max_width: usize) -> SmallLuma {
        let factor = f.width.div_ceil(max_width.max(1)).max(1);
        let (w, h) = ((f.width / factor).max(1), (f.height / factor).max(1));
        let [r, g, b] = f.planes();
        let mut data = vec![0.0f32; w * h];
        let inv = 1.0 / (factor * factor) as f32;
        data.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            for (x, out) in row.iter_mut().enumerate() {
                let mut s = 0.0;
                for dy in 0..factor {
                    let o = (y * factor + dy).min(f.height - 1) * f.width;
                    for dx in 0..factor {
                        let i = o + (x * factor + dx).min(f.width - 1);
                        s += 0.299 * r[i] + 0.587 * g[i] + 0.114 * b[i];
                    }
                }
                *out = s * inv;
            }
        });
        SmallLuma { data, width: w, height: h, factor }
    }
}

/// Mean absolute difference of two equally sized planes.
pub fn mad(a: &[f32], b: &[f32]) -> f32 {
    let s: f32 = a.par_chunks(4096).zip(b.par_chunks(4096)).map(|(x, y)| x.iter().zip(y).map(|(p, q)| (p - q).abs()).sum::<f32>()).sum();
    s / a.len().max(1) as f32
}

/// Gaussian noise standard deviation (Immerkær 1996), robust to edges by excluding the
/// strongest gradients.
pub fn estimate_noise(p: &[f32], w: usize, h: usize) -> f32 {
    if w < 8 || h < 8 {
        return 0.0;
    }
    let rows: Vec<(f64, usize)> = (1..h - 1)
        .into_par_iter()
        .map(|y| {
            let (mut s, mut n) = (0.0f64, 0usize);
            for x in 1..w - 1 {
                let at = |dx: isize, dy: isize| p[(y as isize + dy) as usize * w + (x as isize + dx) as usize];
                let gx = (at(1, 0) - at(-1, 0)).abs();
                let gy = (at(0, 1) - at(0, -1)).abs();
                if gx + gy > 0.15 {
                    continue;
                }
                let lap = at(-1, -1) - 2.0 * at(0, -1) + at(1, -1) - 2.0 * at(-1, 0) + 4.0 * at(0, 0) - 2.0 * at(1, 0) + at(-1, 1) - 2.0 * at(0, 1) + at(1, 1);
                s += lap.abs() as f64;
                n += 1;
            }
            (s, n)
        })
        .collect();
    let (s, n) = rows.iter().fold((0.0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    if n < 64 {
        return 0.0;
    }
    ((std::f64::consts::PI / 2.0).sqrt() * s / (6.0 * n as f64)) as f32
}

/// Ratio of gradient energy on the 8-pixel grid to elsewhere; ~1 for clean video, >1.2 for
/// visibly blocky video.
pub fn blockiness(p: &[f32], w: usize, h: usize) -> f32 {
    if w < 32 || h < 32 {
        return 1.0;
    }
    let (on, off, n_on, n_off) = (0..h)
        .into_par_iter()
        .map(|y| {
            let row = &p[y * w..(y + 1) * w];
            let (mut on, mut off, mut n_on, mut n_off) = (0.0f64, 0.0f64, 0usize, 0usize);
            for x in 1..w {
                let d = (row[x] - row[x - 1]).abs() as f64;
                if d > 0.1 {
                    continue;
                }
                if x % 8 == 0 {
                    on += d;
                    n_on += 1;
                } else {
                    off += d;
                    n_off += 1;
                }
            }
            (on, off, n_on, n_off)
        })
        .reduce(|| (0.0, 0.0, 0, 0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2, a.3 + b.3));
    if n_on == 0 || n_off == 0 || off < 1e-9 {
        return 1.0;
    }
    ((on / n_on as f64) / (off / n_off as f64)) as f32
}

/// High-frequency share of gradient energy; low values mean a soft image.
pub fn sharpness(p: &[f32], w: usize, h: usize) -> f32 {
    if w < 8 || h < 8 {
        return 1.0;
    }
    let (fine, coarse) = (2..h - 2)
        .into_par_iter()
        .map(|y| {
            let (mut f, mut c) = (0.0f64, 0.0f64);
            for x in 2..w - 2 {
                let i = y * w + x;
                f += ((p[i + 1] - p[i - 1]).abs() + (p[i + w] - p[i - w]).abs()) as f64;
                c += ((p[i + 2] - p[i - 2]).abs() + (p[i + 2 * w] - p[i - 2 * w]).abs()) as f64;
            }
            (f, c)
        })
        .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));
    if coarse < 1e-9 { 1.0 } else { (fine / coarse) as f32 }
}

/// Estimates enhancement parameters from sample frames (the "Auto" mode).
pub fn estimate_params(frames: &[&Frame]) -> EnhanceParams {
    if frames.is_empty() {
        return EnhanceParams::default();
    }
    let (mut noise, mut block, mut sharp) = (0.0f32, 0.0f32, 0.0f32);
    for f in frames {
        let g = f.gray();
        noise += estimate_noise(&g, f.width, f.height);
        block += blockiness(&g, f.width, f.height);
        sharp += sharpness(&g, f.width, f.height);
    }
    let n = frames.len() as f32;
    let (noise, block, sharp) = (noise / n, block / n, sharp / n);
    // A perfectly sharp edge gives fine/coarse ≈ 0.5; blur pushes it towards 0.5 from above
    // for textured content, so the softness scale is relative.
    let softness = ((0.62 - sharp) / 0.12).clamp(0.0, 1.0);
    EnhanceParams {
        noise: (noise / 0.025).clamp(0.0, 1.0),
        compression: ((block - 1.08) * 2.5).clamp(0.0, 1.0),
        sharpen: (0.1 + 0.4 * softness).clamp(0.0, 0.5),
        details: (0.1 + 0.3 * softness).clamp(0.0, 0.4),
        dehalo: 0.2,
        ..EnhanceParams::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::noise::noise_plane;

    #[test]
    fn noise_estimate_tracks_sigma() {
        let (w, h) = (256, 256);
        let mut n = vec![0.0; w * h];
        noise_plane(&mut n, w, h, 1, 0.0);
        for sigma in [0.005f32, 0.02] {
            let p: Vec<f32> = n.iter().map(|v| 0.5 + v * sigma).collect();
            let est = estimate_noise(&p, w, h);
            assert!((est - sigma).abs() / sigma < 0.25, "sigma {sigma} est {est}");
        }
    }

    #[test]
    fn blockiness_detects_grid() {
        let (w, h) = (64, 64);
        let mut p = vec![0.0f32; w * h];
        for y in 0..h {
            for x in 0..w {
                p[y * w + x] = 0.5 + ((x * 3 + y * 5) % 7) as f32 * 0.002 + ((x / 8) % 3) as f32 * 0.02;
            }
        }
        assert!(blockiness(&p, w, h) > 1.5);
    }
}
