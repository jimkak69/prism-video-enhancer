//! The app icon, rasterised from the same shape as the in-app logo: a green prism on a dark
//! rounded square. Used for the window icon and, through `--write-icons`, for the installers'
//! PNG and ICO files.

use std::path::Path;

const BG: [f32; 3] = [0x1B as f32 / 255.0, 0x1E as f32 / 255.0, 0x1C as f32 / 255.0];
const GREEN: [f32; 3] = [0x2E as f32 / 255.0, 0xCC as f32 / 255.0, 0x71 as f32 / 255.0];

fn inside_triangle(p: (f32, f32), a: (f32, f32), b: (f32, f32), c: (f32, f32)) -> bool {
    let s = |p: (f32, f32), q: (f32, f32), r: (f32, f32)| (p.0 - r.0) * (q.1 - r.1) - (q.0 - r.0) * (p.1 - r.1);
    let (d1, d2, d3) = (s(p, a, b), s(p, b, c), s(p, c, a));
    !((d1 < 0.0 || d2 < 0.0 || d3 < 0.0) && (d1 > 0.0 || d2 > 0.0 || d3 > 0.0))
}

/// Colour and coverage at a point in the unit square.
fn sample(x: f32, y: f32) -> Option<[f32; 3]> {
    // Rounded square background.
    let r = 0.2;
    let (cx, cy) = (x.clamp(r, 1.0 - r), y.clamp(r, 1.0 - r));
    if (x - cx).powi(2) + (y - cy).powi(2) > r * r {
        return None;
    }
    // Same proportions as icons::Icon::Logo, inset into the square.
    let m = |px: f32, py: f32| (0.14 + px / 16.0 * 0.72, 0.16 + py / 16.0 * 0.72);
    let (a, b, c) = (m(8.0, 1.5), m(15.0, 14.0), m(1.0, 14.0));
    let (ia, ib, ic) = (m(8.0, 5.5), m(11.6, 12.0), m(4.4, 12.0));
    let line = (x - m(8.0, 0.0).0).abs() < 0.6 / 16.0 * 0.72 && y >= ia.1 && y <= ib.1;
    if inside_triangle((x, y), a, b, c) && (!inside_triangle((x, y), ia, ib, ic) || line) {
        Some(GREEN)
    } else {
        Some(BG)
    }
}

/// RGBA pixels of a `size` x `size` icon, 4x4 supersampled.
pub fn rgba(size: u32) -> Vec<u8> {
    let n = size as usize;
    let mut out = vec![0u8; n * n * 4];
    const SS: usize = 4;
    for y in 0..n {
        for x in 0..n {
            let (mut rgb, mut cover) = ([0.0f32; 3], 0.0f32);
            for sy in 0..SS {
                for sx in 0..SS {
                    let u = (x as f32 + (sx as f32 + 0.5) / SS as f32) / size as f32;
                    let v = (y as f32 + (sy as f32 + 0.5) / SS as f32) / size as f32;
                    if let Some(c) = sample(u, v) {
                        for k in 0..3 {
                            rgb[k] += c[k];
                        }
                        cover += 1.0;
                    }
                }
            }
            let i = (y * n + x) * 4;
            if cover > 0.0 {
                for k in 0..3 {
                    out[i + k] = (rgb[k] / cover * 255.0).round() as u8;
                }
                out[i + 3] = (cover / (SS * SS) as f32 * 255.0).round() as u8;
            }
        }
    }
    out
}

pub fn icon_data(size: u32) -> egui::IconData {
    egui::IconData { rgba: rgba(size), width: size, height: size }
}

fn crc32(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c ^= b as u32;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
        }
    }
    !c
}

fn adler32(data: &[u8]) -> u32 {
    let (mut a, mut b) = (1u32, 0u32);
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

/// A PNG with uncompressed (stored) deflate blocks: no compression crate needed for a few
/// small icons.
pub fn png(size: u32) -> Vec<u8> {
    let px = rgba(size);
    let row = size as usize * 4;
    let mut raw = Vec::with_capacity((row + 1) * size as usize);
    for y in 0..size as usize {
        raw.push(0); // filter: none
        raw.extend_from_slice(&px[y * row..(y + 1) * row]);
    }
    let mut z = vec![0x78, 0x01];
    let mut chunks = raw.chunks(65535).peekable();
    while let Some(c) = chunks.next() {
        z.push(if chunks.peek().is_none() { 1 } else { 0 });
        z.extend_from_slice(&(c.len() as u16).to_le_bytes());
        z.extend_from_slice(&(!(c.len() as u16)).to_le_bytes());
        z.extend_from_slice(c);
    }
    z.extend_from_slice(&adler32(&raw).to_be_bytes());
    let mut out = vec![0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A];
    let mut chunk = |kind: &[u8; 4], data: &[u8]| {
        out.extend_from_slice(&(data.len() as u32).to_be_bytes());
        let mut body = kind.to_vec();
        body.extend_from_slice(data);
        out.extend_from_slice(&body);
        out.extend_from_slice(&crc32(&body).to_be_bytes());
    };
    let mut ihdr = Vec::new();
    ihdr.extend_from_slice(&size.to_be_bytes());
    ihdr.extend_from_slice(&size.to_be_bytes());
    ihdr.extend_from_slice(&[8, 6, 0, 0, 0]); // 8-bit RGBA
    chunk(b"IHDR", &ihdr);
    chunk(b"IDAT", &z);
    chunk(b"IEND", &[]);
    out
}

/// A Windows .ico holding PNG images of the given sizes.
pub fn ico(sizes: &[u32]) -> Vec<u8> {
    let images: Vec<Vec<u8>> = sizes.iter().map(|s| png(*s)).collect();
    let mut out = vec![0, 0, 1, 0];
    out.extend_from_slice(&(sizes.len() as u16).to_le_bytes());
    let mut offset = 6 + 16 * sizes.len();
    for (s, img) in sizes.iter().zip(&images) {
        let d = if *s >= 256 { 0 } else { *s as u8 };
        out.extend_from_slice(&[d, d, 0, 0]);
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&32u16.to_le_bytes());
        out.extend_from_slice(&(img.len() as u32).to_le_bytes());
        out.extend_from_slice(&(offset as u32).to_le_bytes());
        offset += img.len();
    }
    for img in images {
        out.extend_from_slice(&img);
    }
    out
}

/// Writes `prism.ico` and `prism-<size>.png` files for packaging.
pub fn write_icons(dir: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    std::fs::write(dir.join("prism.ico"), ico(&[16, 24, 32, 48, 64, 128, 256]))?;
    for s in [16, 32, 48, 64, 128, 256, 512, 1024] {
        std::fs::write(dir.join(format!("prism-{s}.png")), png(s))?;
    }
    Ok(())
}
