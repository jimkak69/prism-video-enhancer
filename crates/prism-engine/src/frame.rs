//! Frames and raw-transport conversion.
//!
//! A [`Frame`] is planar RGB in `f32`, values nominally in `[0, 1]`, in the colour space of the
//! stream (gamma-encoded, not linear). Buffers are recycled through a global pool: large fresh
//! allocations are page-faulted on first touch, which costs more than the conversions themselves.

use std::sync::Mutex;

use prism_core::media::{ColorRange, Matrix};
use prism_core::plan::{RawFormat, RawLayout};
use rayon::prelude::*;

/// Upper bound on idle pooled memory. In-flight frames are bounded by the pipeline's channels;
/// reuse only needs a few spare buffers per frame size.
const POOL_BYTES: usize = 1 << 30;

struct Pool {
    bufs: Vec<Vec<f32>>,
    bytes: usize,
}

static POOL: Mutex<Pool> = Mutex::new(Pool { bufs: Vec::new(), bytes: 0 });

fn take_buffer(len: usize) -> Vec<f32> {
    let mut pool = POOL.lock().unwrap();
    // Exact-size reuse is the common case: every frame of a stage has the same size.
    if let Some(i) = pool.bufs.iter().position(|b| b.capacity() >= len && b.capacity() <= len + len / 4) {
        let mut b = pool.bufs.swap_remove(i);
        pool.bytes -= b.capacity() * 4;
        drop(pool);
        // SAFETY: f32 has no invalid bit patterns and every caller overwrites the full length
        // (or uses `Frame::zeroed`), so exposing stale contents is harmless.
        unsafe { b.set_len(len) };
        return b;
    }
    drop(pool);
    vec![0.0; len]
}

fn return_buffer(b: Vec<f32>) {
    if b.capacity() == 0 {
        return;
    }
    let mut pool = POOL.lock().unwrap();
    let size = b.capacity() * 4;
    while pool.bytes + size > POOL_BYTES && !pool.bufs.is_empty() {
        let old = pool.bufs.remove(0);
        pool.bytes -= old.capacity() * 4;
    }
    if size <= POOL_BYTES {
        pool.bytes += size;
        pool.bufs.push(b);
    }
}

/// Scratch `f32` buffer from the pool, returned on drop.
pub struct Scratch(Vec<f32>);

impl Scratch {
    /// Contents are unspecified; overwrite before reading.
    pub fn new(len: usize) -> Self {
        Scratch(take_buffer(len))
    }
    pub fn zeroed(len: usize) -> Self {
        let mut s = Scratch::new(len);
        s.0.fill(0.0);
        s
    }
}

impl std::ops::Deref for Scratch {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        &self.0
    }
}

impl std::ops::DerefMut for Scratch {
    fn deref_mut(&mut self) -> &mut [f32] {
        &mut self.0
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        return_buffer(std::mem::take(&mut self.0));
    }
}

pub struct Frame {
    pub width: usize,
    pub height: usize,
    data: Vec<f32>,
}

impl Frame {
    /// Frame with unspecified contents (pooled); the caller must write every sample.
    pub fn uninit(width: usize, height: usize) -> Frame {
        Frame { width, height, data: take_buffer(3 * width * height) }
    }

    pub fn zeroed(width: usize, height: usize) -> Frame {
        let mut f = Frame::uninit(width, height);
        f.data.fill(0.0);
        f
    }

    pub fn filled(width: usize, height: usize, rgb: [f32; 3]) -> Frame {
        let mut f = Frame::uninit(width, height);
        for c in 0..3 {
            f.plane_mut(c).fill(rgb[c]);
        }
        f
    }

    pub fn pixels(&self) -> usize {
        self.width * self.height
    }

    pub fn data(&self) -> &[f32] {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut [f32] {
        &mut self.data
    }

    pub fn plane(&self, c: usize) -> &[f32] {
        let n = self.pixels();
        &self.data[c * n..(c + 1) * n]
    }

    pub fn plane_mut(&mut self, c: usize) -> &mut [f32] {
        let n = self.pixels();
        &mut self.data[c * n..(c + 1) * n]
    }

    pub fn planes_mut(&mut self) -> [&mut [f32]; 3] {
        let n = self.pixels();
        let (r, rest) = self.data.split_at_mut(n);
        let (g, b) = rest.split_at_mut(n);
        [r, g, b]
    }

    pub fn planes(&self) -> [&[f32]; 3] {
        [self.plane(0), self.plane(1), self.plane(2)]
    }

    pub fn duplicate(&self) -> Frame {
        let mut f = Frame::uninit(self.width, self.height);
        f.data.par_chunks_mut(1 << 16).zip(self.data.par_chunks(1 << 16)).for_each(|(d, s)| d.copy_from_slice(s));
        f
    }

    /// Mean of the three channels (cheap luma proxy) at full resolution.
    pub fn gray(&self) -> Vec<f32> {
        let [r, g, b] = self.planes();
        let mut out = vec![0.0; self.pixels()];
        out.par_chunks_mut(self.width.max(1)).enumerate().for_each(|(y, row)| {
            let o = y * self.width;
            for x in 0..row.len() {
                row[x] = 0.299 * r[o + x] + 0.587 * g[o + x] + 0.114 * b[o + x];
            }
        });
        out
    }
}

impl Drop for Frame {
    fn drop(&mut self) {
        return_buffer(std::mem::take(&mut self.data));
    }
}

impl std::fmt::Debug for Frame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Frame({}x{})", self.width, self.height)
    }
}

/// YCbCr ↔ R'G'B' coefficients for one matrix/range/bit depth.
#[derive(Debug, Clone, Copy)]
struct Yuv {
    y_off: f32,
    y_scale: f32,
    c_off: f32,
    c_scale: f32,
    kr: f32,
    kb: f32,
    max: f32,
}

impl Yuv {
    fn new(matrix: Matrix, range: ColorRange, bits: u8) -> Yuv {
        let (kr, kb) = matrix.coefficients();
        let m = (1u32 << (bits - 8)) as f32;
        let max = ((1u32 << bits) - 1) as f32;
        match range {
            ColorRange::Limited => Yuv { y_off: 16.0 * m, y_scale: 219.0 * m, c_off: 128.0 * m, c_scale: 224.0 * m, kr, kb, max },
            ColorRange::Full => Yuv { y_off: 0.0, y_scale: max, c_off: 128.0 * m, c_scale: max, kr, kb, max },
        }
    }
}

/// Widens one row of samples (u8, or little-endian u16 when `wide`) to f32.
#[inline]
fn load_row(src: &[u8], wide: bool, dst: &mut [f32]) {
    if wide {
        for (d, s) in dst.iter_mut().zip(src.chunks_exact(2)) {
            *d = u16::from_le_bytes([s[0], s[1]]) as f32;
        }
    } else {
        for (d, s) in dst.iter_mut().zip(src) {
            *d = *s as f32;
        }
    }
}

/// Rounds code values (already scaled to `0..=max`) plus a dither row and stores them.
#[inline]
fn store_row(src: &[f32], dither: &[f32], max: f32, wide: bool, dst: &mut [u8]) {
    if wide {
        for ((d, s), z) in dst.chunks_exact_mut(2).zip(src).zip(dither) {
            let q = (s + z).round().max(0.0).min(max) as u16;
            d.copy_from_slice(&q.to_le_bytes());
        }
    } else {
        for ((d, s), z) in dst.iter_mut().zip(src).zip(dither) {
            *d = (s + z).round().max(0.0).min(max) as u8;
        }
    }
}

/// Converts one raw frame (as produced by FFmpeg with `fmt.pix_fmt()`) into `out`.
///
/// Rows are widened to f32 first and chroma is interpolated row-wise, so the colour maths runs
/// as straight-line code over slices that the compiler vectorises.
pub fn raw_to_frame(bytes: &[u8], fmt: &RawFormat, out: &mut Frame) {
    let (w, h) = (fmt.width as usize, fmt.height as usize);
    assert_eq!(bytes.len(), fmt.frame_bytes(), "raw frame size");
    assert_eq!((out.width, out.height), (w, h));
    let wide = fmt.bits > 8;
    let bps = fmt.bytes_per_sample();
    match fmt.layout {
        RawLayout::Rgb => {
            // gbrp plane order is G, B, R.
            let scale = 1.0 / ((1u32 << if wide { 16 } else { 8 }) - 1) as f32;
            let n = w * h;
            let [r, g, b] = out.planes_mut();
            for (dst, src_plane) in [(g, 0usize), (b, 1), (r, 2)] {
                let src = &bytes[src_plane * n * bps..(src_plane + 1) * n * bps];
                dst.par_chunks_mut(w).enumerate().for_each(|(y, row)| {
                    load_row(&src[y * w * bps..(y + 1) * w * bps], wide, row);
                    for v in row.iter_mut() {
                        *v *= scale;
                    }
                });
            }
        }
        RawLayout::Yuv420 => {
            let k = Yuv::new(fmt.matrix, fmt.range, if wide { fmt.bits } else { 8 });
            let (cw, ch) = fmt.chroma_size();
            let ybytes = &bytes[..w * h * bps];
            let ubytes = &bytes[w * h * bps..(w * h + cw * ch) * bps];
            let vbytes = &bytes[(w * h + cw * ch) * bps..];
            let (iy, ic) = (1.0 / k.y_scale, 1.0 / k.c_scale);
            let (cr_v, cb_u) = (2.0 * (1.0 - k.kr), 2.0 * (1.0 - k.kb));
            let kg_b = k.kb * (1.0 - k.kb) * 2.0 / (1.0 - k.kr - k.kb);
            let kg_r = k.kr * (1.0 - k.kr) * 2.0 / (1.0 - k.kr - k.kb);
            let n = w * h;
            let data = out.data_mut();
            let (rp, rest) = data.split_at_mut(n);
            let (gp, bp) = rest.split_at_mut(n);
            const BAND: usize = 8;
            rp.par_chunks_mut(w * BAND).zip(gp.par_chunks_mut(w * BAND)).zip(bp.par_chunks_mut(w * BAND)).enumerate().for_each(|(band, ((rb, gb), bb))| {
                let (mut yf, mut uf, mut vf) = (vec![0.0f32; w], vec![0.0f32; w], vec![0.0f32; w]);
                let (mut c0, mut c1, mut uc, mut vc) = (vec![0.0f32; cw], vec![0.0f32; cw], vec![0.0f32; cw], vec![0.0f32; cw]);
                for (k_row, ((rr, gr), br)) in rb.chunks_mut(w).zip(gb.chunks_mut(w)).zip(bb.chunks_mut(w)).enumerate() {
                    let y = band * BAND + k_row;
                    // MPEG-2/H.264 siting: chroma co-sited horizontally with even luma, centred vertically.
                    let cyf = (y as f32 - 0.5) * 0.5;
                    let cy0 = cyf.floor().max(0.0) as usize;
                    let cy1 = (cy0 + 1).min(ch - 1);
                    let fy = (cyf - cy0 as f32).clamp(0.0, 1.0);
                    for (plane, dst) in [(ubytes, &mut uc), (vbytes, &mut vc)] {
                        load_row(&plane[cy0 * cw * bps..(cy0 + 1) * cw * bps], wide, &mut c0);
                        load_row(&plane[cy1 * cw * bps..(cy1 + 1) * cw * bps], wide, &mut c1);
                        for ((d, a), b) in dst.iter_mut().zip(&c0).zip(&c1) {
                            *d = (a + (b - a) * fy - k.c_off) * ic;
                        }
                    }
                    for (full, half) in [(&mut uf, &uc), (&mut vf, &vc)] {
                        for x in 0..w / 2 {
                            full[2 * x] = half[x];
                            full[2 * x + 1] = 0.5 * (half[x] + half[(x + 1).min(cw - 1)]);
                        }
                        if w % 2 == 1 {
                            full[w - 1] = half[cw - 1];
                        }
                    }
                    load_row(&ybytes[y * w * bps..(y + 1) * w * bps], wide, &mut yf);
                    for x in 0..w {
                        let yv = (yf[x] - k.y_off) * iy;
                        let (u, v) = (uf[x], vf[x]);
                        rr[x] = yv + cr_v * v;
                        br[x] = yv + cb_u * u;
                        gr[x] = yv - kg_b * u - kg_r * v;
                    }
                }
            });
        }
    }
}

/// 4x4 Bayer matrix, in units of one 8-bit code value.
const BAYER: [[f32; 4]; 4] = [[0.0, 8.0, 2.0, 10.0], [12.0, 4.0, 14.0, 6.0], [3.0, 11.0, 1.0, 9.0], [15.0, 7.0, 13.0, 5.0]];

/// Four rows of `w` zero-centred ordered-dither offsets (all zero when disabled).
fn dither_rows(w: usize, enabled: bool) -> [Vec<f32>; 4] {
    std::array::from_fn(|r| (0..w).map(|x| if enabled { (BAYER[r][x & 3] + 0.5) / 16.0 - 0.5 } else { 0.0 }).collect())
}

/// Converts `frame` into raw bytes for FFmpeg (`fmt.pix_fmt()`), reusing `out`.
pub fn frame_to_raw(frame: &Frame, fmt: &RawFormat, out: &mut Vec<u8>) {
    let (w, h) = (fmt.width as usize, fmt.height as usize);
    assert_eq!((frame.width, frame.height), (w, h), "frame size does not match output format");
    out.resize(fmt.frame_bytes(), 0);
    let wide = fmt.bits > 8;
    let bps = fmt.bytes_per_sample();
    let dither = dither_rows(w.max(1), fmt.bits == 8);
    match fmt.layout {
        RawLayout::Rgb => {
            let max = if wide { 65535.0 } else { 255.0 };
            let n = w * h;
            let (gdst, rest) = out.split_at_mut(n * bps);
            let (bdst, rdst) = rest.split_at_mut(n * bps);
            for (dst, src) in [(gdst, frame.plane(1)), (bdst, frame.plane(2)), (rdst, frame.plane(0))] {
                dst.par_chunks_mut(w * bps).enumerate().for_each(|(y, row)| {
                    let scaled: Vec<f32> = src[y * w..(y + 1) * w].iter().map(|v| v * max).collect();
                    store_row(&scaled, &dither[y & 3], max, wide, row);
                });
            }
        }
        RawLayout::Yuv420 => {
            let k = Yuv::new(fmt.matrix, fmt.range, if wide { fmt.bits } else { 8 });
            let kg = 1.0 - k.kr - k.kb;
            let (cw, ch) = fmt.chroma_size();
            let [r, g, b] = frame.planes();
            let (ydst, rest) = out.split_at_mut(w * h * bps);
            let (udst, vdst) = rest.split_at_mut(cw * ch * bps);
            let cb = 0.5 / (1.0 - k.kb) * 0.5;
            let cr = 0.5 / (1.0 - k.kr) * 0.5;
            // One task per luma row pair: luma is computed once per pixel, and the pair's
            // vertically averaged colour differences feed that pair's chroma row.
            let ychunks = ydst.par_chunks_mut(2 * w * bps);
            ychunks.zip(udst.par_chunks_mut(cw * bps)).zip(vdst.par_chunks_mut(cw * bps)).enumerate().for_each(|(cy, ((yrows, urow), vrow))| {
                let (mut du, mut dv, mut yl, mut code) = (vec![0.0f32; w], vec![0.0f32; w], vec![0.0f32; w], vec![0.0f32; w]);
                let rows = yrows.len() / (w * bps);
                for r_i in 0..2 {
                    let y = (2 * cy + r_i).min(h - 1);
                    let (rs, gs, bs) = (&r[y * w..(y + 1) * w], &g[y * w..(y + 1) * w], &b[y * w..(y + 1) * w]);
                    for x in 0..w {
                        yl[x] = k.kr * rs[x] + kg * gs[x] + k.kb * bs[x];
                        du[x] += (bs[x] - yl[x]) * cb;
                        dv[x] += (rs[x] - yl[x]) * cr;
                    }
                    if r_i < rows {
                        for x in 0..w {
                            code[x] = yl[x] * k.y_scale + k.y_off;
                        }
                        store_row(&code, &dither[y & 3], k.max, wide, &mut yrows[r_i * w * bps..(r_i + 1) * w * bps]);
                    }
                }
                // Left-sited horizontal [1 2 1] filter, then scale to code values.
                let (mut uq, mut vq) = (vec![0.0f32; cw], vec![0.0f32; cw]);
                for cx in 0..cw {
                    let (xa, xb, xc) = ((2 * cx).saturating_sub(1), 2 * cx, (2 * cx + 1).min(w - 1));
                    uq[cx] = (0.25 * du[xa] + 0.5 * du[xb] + 0.25 * du[xc]) * k.c_scale + k.c_off;
                    vq[cx] = (0.25 * dv[xa] + 0.5 * dv[xb] + 0.25 * dv[xc]) * k.c_scale + k.c_off;
                }
                store_row(&uq, &dither[cy & 3][..cw], k.max, wide, urow);
                store_row(&vq, &dither[cy & 3][..cw], k.max, wide, vrow);
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fmt(layout: RawLayout, bits: u8, w: u32, h: u32) -> RawFormat {
        RawFormat { width: w, height: h, layout, bits, matrix: Matrix::Bt709, range: ColorRange::Limited }
    }

    fn test_frame(w: usize, h: usize) -> Frame {
        let mut f = Frame::uninit(w, h);
        for y in 0..h {
            for x in 0..w {
                // Smooth content so 4:2:0 subsampling is nearly lossless.
                let i = y * w + x;
                f.plane_mut(0)[i] = 0.2 + 0.6 * x as f32 / w as f32;
                f.plane_mut(1)[i] = 0.3 + 0.4 * y as f32 / h as f32;
                f.plane_mut(2)[i] = 0.5 + 0.2 * ((x + y) as f32 / (w + h) as f32);
            }
        }
        f
    }

    fn max_err(a: &Frame, b: &Frame) -> f32 {
        a.data().iter().zip(b.data()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    #[test]
    fn rgb_round_trips() {
        for bits in [8u8, 16] {
            let f = fmt(RawLayout::Rgb, bits, 33, 17);
            let src = test_frame(33, 17);
            let mut raw = Vec::new();
            frame_to_raw(&src, &f, &mut raw);
            let mut back = Frame::uninit(33, 17);
            raw_to_frame(&raw, &f, &mut back);
            let tol = if bits == 8 { 1.0 / 255.0 } else { 1e-4 };
            assert!(max_err(&src, &back) <= tol, "bits {bits}: {}", max_err(&src, &back));
        }
    }

    #[test]
    fn yuv420_round_trips_smooth_content() {
        for bits in [8u8, 10, 16] {
            let f = fmt(RawLayout::Yuv420, bits, 64, 36);
            let src = test_frame(64, 36);
            let mut raw = Vec::new();
            frame_to_raw(&src, &f, &mut raw);
            assert_eq!(raw.len(), f.frame_bytes());
            let mut back = Frame::uninit(64, 36);
            raw_to_frame(&raw, &f, &mut back);
            let tol = if bits == 8 { 0.02 } else { 0.012 };
            assert!(max_err(&src, &back) < tol, "bits {bits}: {}", max_err(&src, &back));
        }
    }

    #[test]
    fn yuv_limited_range_levels() {
        // Black and white must land on 16/235 for 8-bit limited range.
        let f = fmt(RawLayout::Yuv420, 8, 4, 2);
        let mut white = Frame::filled(4, 2, [1.0, 1.0, 1.0]);
        let mut raw = Vec::new();
        frame_to_raw(&white, &f, &mut raw);
        assert!(raw[..8].iter().all(|v| *v == 235));
        assert!(raw[8..].iter().all(|v| *v == 128));
        white.data_mut().fill(0.0);
        frame_to_raw(&white, &f, &mut raw);
        assert!(raw[..8].iter().all(|v| *v == 16));
    }

    #[test]
    fn odd_dimensions() {
        let f = fmt(RawLayout::Yuv420, 8, 7, 5);
        let src = test_frame(7, 5);
        let mut raw = Vec::new();
        frame_to_raw(&src, &f, &mut raw);
        let mut back = Frame::uninit(7, 5);
        raw_to_frame(&raw, &f, &mut back);
        assert!(max_err(&src, &back) < 0.05);
    }
}
