//! Tile planning and seam blending for networks run on fixed-size tiles.
//!
//! Every tile has the same size, so a GPU session is compiled once per job. Tiles overlap by at
//! least `overlap` input pixels. Within an overlap each tile discards an outer margin of
//! `overlap / 4` (where the network's zero padding corrupts results) and ramps linearly over the
//! next `overlap / 2`; at the minimum overlap the two ramps sum to exactly one.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TileGrid {
    pub tile_w: usize,
    pub tile_h: usize,
    pub xs: Vec<usize>,
    pub ys: Vec<usize>,
    pub overlap: usize,
    pub width: usize,
    pub height: usize,
}

fn round_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

fn round_down(v: usize, a: usize) -> usize {
    (v / a) * a
}

/// Evenly spread start positions so `n` tiles of `tile` cover `len` with at least `overlap`.
fn positions(len: usize, tile: usize, overlap: usize) -> Vec<usize> {
    if len <= tile {
        return vec![0];
    }
    let step = tile - overlap;
    let n = (len - overlap).div_ceil(step).max(2);
    (0..n).map(|i| ((i * (len - tile)) as f64 / (n - 1) as f64).round() as usize).collect()
}

/// Smallest aligned tile that covers `len` with `n` tiles and the given overlap.
fn balanced(len: usize, n: usize, overlap: usize, align: usize) -> usize {
    round_up((len + (n - 1) * overlap).div_ceil(n), align)
}

/// Plans tiles for a `w x h` frame. `max_pixels` bounds the tile area (memory), `align` the tile
/// size multiple required by the network.
pub fn plan(w: usize, h: usize, max_pixels: usize, overlap: usize, align: usize) -> TileGrid {
    let (pw, ph) = (round_up(w, align), round_up(h, align));
    let single = TileGrid { tile_w: pw, tile_h: ph, xs: vec![0], ys: vec![0], overlap, width: w, height: h };
    if pw * ph <= max_pixels {
        return single;
    }
    let min_tile = round_up(2 * overlap + align, align).max(4 * align);
    // Long side first: keep tiles roughly square but never larger than the padded frame.
    let side = round_down((max_pixels as f64).sqrt() as usize, align).max(min_tile);
    let mut tw = side.min(pw);
    let mut th = round_down(max_pixels / tw, align).max(min_tile).min(ph);
    if th == ph {
        tw = round_down(max_pixels / th, align).max(min_tile).min(pw);
    }
    let nx = if w <= tw { 1 } else { (w - overlap).div_ceil(tw - overlap) };
    let ny = if h <= th { 1 } else { (h - overlap).div_ceil(th - overlap) };
    if nx > 1 {
        tw = balanced(w, nx, overlap, align).min(tw);
    }
    if ny > 1 {
        th = balanced(h, ny, overlap, align).min(th);
    }
    TileGrid { tile_w: tw, tile_h: th, xs: positions(w, tw, overlap), ys: positions(h, th, overlap), overlap, width: w, height: h }
}

impl TileGrid {
    pub fn count(&self) -> usize {
        self.xs.len() * self.ys.len()
    }

    /// Feather weight for output offset `u` inside a tile of `len` output pixels, for a tile
    /// overlap of `overlap` output pixels. Sides without a neighbour keep full weight.
    #[inline]
    pub fn ramp(u: usize, len: usize, overlap: usize, has_left: bool, has_right: bool) -> f32 {
        let margin = (overlap / 4) as f32;
        let ramp = (overlap / 2).max(1) as f32;
        let mut w = 1.0f32;
        if has_left {
            w = w.min((u as f32 + 0.5 - margin) / ramp);
        }
        if has_right {
            w = w.min((len as f32 - u as f32 - 0.5 - margin) / ramp);
        }
        w.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_frames_use_one_padded_tile() {
        let g = plan(320, 180, 1 << 20, 16, 16);
        assert_eq!((g.tile_w, g.tile_h, g.count()), (320, 192, 1));
    }

    #[test]
    fn large_frames_are_covered_with_overlap() {
        for (w, h, max) in [(1920, 1080, 512 * 512), (3840, 2160, 640 * 640), (1000, 300, 256 * 256), (5000, 64, 128 * 128)] {
            let g = plan(w, h, max, 32, 16);
            assert!(g.tile_w * g.tile_h <= max, "{w}x{h}: tile {}x{}", g.tile_w, g.tile_h);
            assert_eq!(g.tile_w % 16, 0);
            for axis in [(&g.xs, g.tile_w, w), (&g.ys, g.tile_h, h)] {
                let (pos, tile, len) = axis;
                if len <= tile {
                    continue;
                }
                assert_eq!(pos[0], 0);
                assert_eq!(pos.last().unwrap() + tile, len);
                for p in pos.windows(2) {
                    assert!(p[0] + tile >= p[1] + 32, "overlap too small at {p:?}");
                }
            }
        }
    }

    #[test]
    fn ramps_discard_margin_and_sum_to_one_at_minimum_overlap() {
        let ov = 32;
        assert_eq!(TileGrid::ramp(0, 100, ov, false, false), 1.0);
        assert_eq!(TileGrid::ramp(5, 100, ov, true, false), 0.0, "margin is discarded");
        assert_eq!(TileGrid::ramp(50, 100, ov, true, true), 1.0);
        // Tile A spans [0, 100), tile B starts at 68 (overlap exactly 32).
        for x in 68..100 {
            let a = TileGrid::ramp(x, 100, ov, false, true);
            let b = TileGrid::ramp(x - 68, 100, ov, true, false);
            assert!((a + b - 1.0).abs() < 1e-6, "x {x}: {a} + {b}");
        }
    }
}
