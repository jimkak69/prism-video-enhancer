//! Transfer functions and SDR → HDR inverse tone mapping.

use prism_core::plan::HdrSpec;
use prism_core::settings::HdrTransfer;
use rayon::prelude::*;

use crate::frame::Frame;

/// BT.709 → BT.2020 primaries, linear light.
const BT709_TO_BT2020: [[f32; 3]; 3] = [[0.627_404, 0.329_283, 0.043_313], [0.069_097, 0.919_540, 0.011_362], [0.016_391, 0.088_013, 0.895_595]];

/// SMPTE ST 2084 inverse EOTF: absolute linear (1.0 = 10000 nits) → signal.
#[inline]
pub fn pq_encode(l: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let lp = l.max(0.0).powf(M1);
    ((C1 + C2 * lp) / (1.0 + C3 * lp)).powf(M2)
}

#[inline]
pub fn pq_decode(e: f32) -> f32 {
    const M1: f32 = 0.159_301_76;
    const M2: f32 = 78.84375;
    const C1: f32 = 0.835_937_5;
    const C2: f32 = 18.851_563;
    const C3: f32 = 18.6875;
    let ep = e.max(0.0).powf(1.0 / M2);
    ((ep - C1).max(0.0) / (C2 - C3 * ep)).powf(1.0 / M1)
}

/// ARIB STD-B67 (HLG) OETF on scene-referred linear light in 0..1.
#[inline]
pub fn hlg_encode(e: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 0.284_668_92;
    const C: f32 = 0.559_910_7;
    let e = e.max(0.0);
    if e <= 1.0 / 12.0 { (3.0 * e).sqrt() } else { A * (12.0 * e - B).ln() + C }
}

/// Luminance expansion curve (nits) for SDR relative luminance `y` (1.0 = SDR white).
///
/// Linear (scene-referred white at `ref_nits`) up to the knee, then a monotone cubic Hermite
/// segment that reaches `peak` at SDR white with matching slope at the knee.
pub struct Expansion {
    knee: f32,
    ref_nits: f32,
    peak: f32,
    m1: f32,
}

impl Expansion {
    pub fn new(sdr_ip: f32, brightness: f32, peak: f32) -> Expansion {
        let knee = sdr_ip.clamp(0.3, 0.95).powf(2.4);
        let ref_nits = 203.0 * 2f32.powf(brightness.clamp(-1.0, 1.0));
        let peak = peak.max(ref_nits * 1.05);
        let delta = (peak - ref_nits * knee) / (1.0 - knee);
        let m1 = (2.0 * delta).clamp(0.0, 3.0 * delta);
        Expansion { knee, ref_nits, peak, m1 }
    }

    #[inline]
    pub fn nits(&self, y: f32) -> f32 {
        let y = y.max(0.0);
        if y <= self.knee {
            return y * self.ref_nits;
        }
        if y >= 1.0 {
            return self.peak + (y - 1.0) * self.m1;
        }
        let len = 1.0 - self.knee;
        let t = (y - self.knee) / len;
        let (t2, t3) = (t * t, t * t * t);
        let p0 = self.ref_nits * self.knee;
        (2.0 * t3 - 3.0 * t2 + 1.0) * p0 + (t3 - 2.0 * t2 + t) * len * self.ref_nits + (-2.0 * t3 + 3.0 * t2) * self.peak + (t3 - t2) * len * self.m1
    }
}

/// Transfer-function encoder tabulated over q = L^(1/4) (L in 0..1), where both PQ and HLG are
/// smooth, so linear interpolation is accurate to ~1e-4 and costs two square roots per sample.
struct EncodeLut {
    table: Vec<f32>,
    /// Linear value mapped to 1.0 (10000 nits for PQ, 1000 nits for HLG).
    full_scale: f32,
}

impl EncodeLut {
    const N: usize = 4096;

    fn new(transfer: HdrTransfer) -> EncodeLut {
        let (full_scale, f): (f32, fn(f32) -> f32) = match transfer {
            HdrTransfer::Pq => (10000.0, pq_encode),
            // HLG: nominal 1000-nit display, inverse OOTF approximated with gamma 1.2.
            HdrTransfer::Hlg => (1000.0, |l| hlg_encode(l.min(1.0).powf(1.0 / 1.2))),
        };
        let table = (0..=Self::N + 1).map(|i| {
            let q = (i as f32 / Self::N as f32).min(1.0);
            f(q * q * q * q)
        }).collect();
        EncodeLut { table, full_scale }
    }

    #[inline]
    fn encode(&self, nits: f32) -> f32 {
        let q = (nits / self.full_scale).clamp(0.0, 1.0).sqrt().sqrt() * Self::N as f32;
        let i = q as usize;
        let f = q - i as f32;
        self.table[i] + (self.table[i + 1] - self.table[i]) * f
    }
}

/// SDR BT.709 (BT.1886 gamma 2.4) → HDR BT.2020 PQ or HLG, in place.
pub fn inverse_tone_map(frame: &mut Frame, spec: &HdrSpec) {
    let curve = Expansion::new(spec.sdr_ip, spec.brightness, spec.peak_nits);
    let sat = 1.0 + spec.saturation.clamp(-1.0, 1.0) * 0.5;
    let enc = EncodeLut::new(spec.transfer);
    let w = frame.width;
    // Linearisation LUT with interpolation: powf per sample is the dominant cost otherwise.
    const N: usize = 4096;
    let lut: Vec<f32> = (0..=N + 1).map(|i| (i as f32 / N as f32).min(1.0).powf(2.4)).collect();
    let lin = |v: f32| -> f32 {
        let x = v.clamp(0.0, 1.0) * N as f32;
        let i = x as usize;
        let f = x - i as f32;
        lut[i] + (lut[i + 1] - lut[i]) * f
    };
    let [r, g, b] = frame.planes_mut();
    r.par_chunks_mut(w).zip(g.par_chunks_mut(w)).zip(b.par_chunks_mut(w)).for_each(|((rr, gr), br)| {
        for x in 0..rr.len() {
            let (lr, lg, lb) = (lin(rr[x]), lin(gr[x]), lin(br[x]));
            let y = 0.2126 * lr + 0.7152 * lg + 0.0722 * lb;
            let gain = if y > 1e-6 { curve.nits(y) / y } else { curve.ref_nits };
            let (mut cr, mut cg, mut cb) = (lr * gain, lg * gain, lb * gain);
            let yy = y * gain;
            cr = yy + (cr - yy) * sat;
            cg = yy + (cg - yy) * sat;
            cb = yy + (cb - yy) * sat;
            let m = &BT709_TO_BT2020;
            let o = [
                (m[0][0] * cr + m[0][1] * cg + m[0][2] * cb).max(0.0),
                (m[1][0] * cr + m[1][1] * cg + m[1][2] * cb).max(0.0),
                (m[2][0] * cr + m[2][1] * cg + m[2][2] * cb).max(0.0),
            ];
            rr[x] = enc.encode(o[0]);
            gr[x] = enc.encode(o[1]);
            br[x] = enc.encode(o[2]);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_round_trip_and_reference_points() {
        assert!((pq_encode(1.0) - 1.0).abs() < 1e-4);
        assert!((pq_encode(0.01) - 0.508).abs() < 0.002, "100 nits ≈ 0.508");
        for l in [0.0001f32, 0.01, 0.1, 0.5] {
            assert!((pq_decode(pq_encode(l)) - l).abs() / l < 1e-3);
        }
    }

    #[test]
    fn expansion_is_monotone_and_hits_peak() {
        let c = Expansion::new(0.65, 0.0, 1000.0);
        let mut prev = -1.0;
        for i in 0..=1000 {
            let v = c.nits(i as f32 / 1000.0);
            assert!(v >= prev - 1e-3, "not monotone at {i}");
            prev = v;
        }
        assert!((c.nits(1.0) - 1000.0).abs() < 1e-2);
        assert!((c.nits(0.1) - 20.3).abs() < 1e-3);
    }

    #[test]
    fn encode_lut_matches_exact_curves() {
        for t in [HdrTransfer::Pq, HdrTransfer::Hlg] {
            let lut = EncodeLut::new(t);
            for i in 0..=2000 {
                let nits = (i as f32 / 2000.0).powi(3) * lut.full_scale;
                let exact = match t {
                    HdrTransfer::Pq => pq_encode(nits / 10000.0),
                    HdrTransfer::Hlg => hlg_encode((nits / 1000.0).min(1.0).powf(1.0 / 1.2)),
                };
                assert!((lut.encode(nits) - exact).abs() < 2e-4, "{t:?} at {nits} nits: {} vs {exact}", lut.encode(nits));
            }
        }
    }

    #[test]
    fn hlg_reference() {
        assert!((hlg_encode(1.0) - 1.0).abs() < 1e-3);
        assert!((hlg_encode(1.0 / 12.0) - 0.5).abs() < 1e-4);
    }
}
