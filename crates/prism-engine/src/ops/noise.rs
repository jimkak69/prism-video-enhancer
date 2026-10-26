//! Deterministic noise and film grain.

use rayon::prelude::*;

use super::filters::gaussian_kernel;
use crate::frame::{Frame, Scratch};

/// SplitMix64: tiny, fast and good enough for visual noise.
#[derive(Clone)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Approximately standard-normal (Irwin–Hall with four uniforms, unit variance).
    #[inline]
    pub fn gauss(&mut self) -> f32 {
        let v = self.next_u64();
        let s = (v & 0xFFFF) + ((v >> 16) & 0xFFFF) + ((v >> 32) & 0xFFFF) + (v >> 48);
        (s as f32 / 65535.0 - 2.0) * 1.732_050_8
    }
}

/// Fills `plane` with unit-variance noise, optionally low-passed to `size` (grain size), keeping
/// unit variance.
pub fn noise_plane(plane: &mut [f32], w: usize, h: usize, seed: u64, size: f32) {
    plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
        let mut rng = Rng::new(seed.wrapping_mul(0x2545_F491_4F6C_DD1D) ^ (y as u64).wrapping_mul(0x9E37_79B9));
        for v in row.iter_mut() {
            *v = rng.gauss();
        }
    });
    if size > 0.05 {
        let sigma = size * 0.6;
        let k = gaussian_kernel(sigma);
        // Variance of separable filtered white noise is (sum k^2)^2.
        let gain = 1.0 / k.iter().map(|v| v * v).sum::<f32>();
        let mut tmp = Scratch::new(w * h);
        super::filters::gaussian_blur(plane, &mut tmp, w, h, sigma);
        plane.par_iter_mut().zip(tmp.par_iter()).for_each(|(p, t)| *p = t * gain);
    }
}

/// Adds monochrome film grain weighted toward mid-tones.
pub fn add_grain(frame: &mut Frame, amount: f32, size: f32, seed: u64) {
    if amount <= 0.0 {
        return;
    }
    let (w, h) = (frame.width, frame.height);
    let mut n = Scratch::new(w * h);
    noise_plane(&mut n, w, h, seed, size);
    let [r, g, b] = frame.planes_mut();
    r.par_chunks_mut(w).zip(g.par_chunks_mut(w)).zip(b.par_chunks_mut(w)).zip(n.par_chunks(w)).for_each(|(((rr, gr), br), nr)| {
        for x in 0..rr.len() {
            let y = (0.299 * rr[x] + 0.587 * gr[x] + 0.114 * br[x]).clamp(0.0, 1.0);
            let wgt = 0.35 + 2.6 * y * (1.0 - y);
            let d = nr[x] * amount * wgt;
            rr[x] += d;
            gr[x] += d;
            br[x] += d;
        }
    });
}

/// Adds independent per-channel Gaussian noise with standard deviation `sigma`.
pub fn add_noise(frame: &mut Frame, sigma: f32, seed: u64) {
    if sigma <= 0.0 {
        return;
    }
    let w = frame.width;
    for (c, plane) in frame.planes_mut().into_iter().enumerate() {
        plane.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
            let mut rng = Rng::new(seed ^ ((c as u64) << 56) ^ (y as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
            for v in row.iter_mut() {
                *v += rng.gauss() * sigma;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stats(v: &[f32]) -> (f32, f32) {
        let m = v.iter().sum::<f32>() / v.len() as f32;
        (m, (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / v.len() as f32).sqrt())
    }

    #[test]
    fn noise_has_unit_variance_with_and_without_grain_size() {
        for size in [0.0f32, 1.0, 2.5] {
            let (w, h) = (256, 256);
            let mut p = vec![0.0; w * h];
            noise_plane(&mut p, w, h, 7, size);
            let (m, s) = stats(&p);
            assert!(m.abs() < 0.05, "mean {m}");
            assert!((s - 1.0).abs() < 0.12, "size {size}: std {s}");
        }
    }

    #[test]
    fn deterministic() {
        let mut a = vec![0.0; 64];
        let mut b = vec![0.0; 64];
        noise_plane(&mut a, 8, 8, 3, 0.0);
        noise_plane(&mut b, 8, 8, 3, 0.0);
        assert_eq!(a, b);
    }
}
