//! Kernel micro-benchmarks: `cargo test -p prism-engine --release kernel_bench -- --ignored --nocapture`.

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use prism_core::media::{ColorRange, Matrix};
    use prism_core::plan::{RawFormat, RawLayout};

    use crate::frame::{Frame, frame_to_raw, raw_to_frame};
    use crate::ops::filters::{cas, guided_smooth, local_minmax};
    use crate::ops::resample::{Filter, resize};

    fn time<T>(label: &str, n: usize, mut f: impl FnMut() -> T) {
        let _ = f();
        let t = Instant::now();
        for _ in 0..n {
            std::hint::black_box(f());
        }
        println!("{label:<40} {:8.2} ms", t.elapsed().as_secs_f64() * 1000.0 / n as f64);
    }

    #[test]
    #[ignore]
    fn kernel_bench() {
        let (w, h) = (1920usize, 1080usize);
        let mut src = Frame::uninit(w, h);
        for (i, v) in src.data_mut().iter_mut().enumerate() {
            *v = ((i * 7919) % 1000) as f32 / 1000.0;
        }
        let f_in = RawFormat { width: w as u32, height: h as u32, layout: RawLayout::Yuv420, bits: 8, matrix: Matrix::Bt709, range: ColorRange::Limited };
        let mut raw = Vec::new();
        frame_to_raw(&src, &f_in, &mut raw);
        time("frame_to_raw 1080p yuv420", 20, || frame_to_raw(&src, &f_in, &mut raw));
        let mut back = Frame::uninit(w, h);
        time("raw_to_frame 1080p yuv420", 20, || raw_to_frame(&raw, &f_in, &mut back));
        time("resize lanczos 1080p -> 4K", 10, || resize(&src, 3840, 2160, Filter::Lanczos3));
        time("resize lanczos 1080p -> 720p", 10, || resize(&src, 1280, 720, Filter::Lanczos3));
        {
            use crate::ops::resample::resize_plane;
            let mut hout = vec![0.0f32; 3840 * 1080];
            time("  horizontal only 1920->3840 x1080 (1 plane)", 10, || resize_plane(src.plane(0), w, h, &mut hout, 3840, 1080, Filter::Lanczos3));
            let mut vout = vec![0.0f32; 3840 * 2160];
            time("  vertical only 1080->2160 x3840 (1 plane)", 10, || resize_plane(&hout, 3840, 1080, &mut vout, 3840, 2160, Filter::Lanczos3));
        }
        time("frame alloc 4K (pooled)", 20, || Frame::uninit(3840, 2160));
        let big = resize(&src, 3840, 2160, Filter::Lanczos3);
        let f_out = RawFormat { width: 3840, height: 2160, ..f_in };
        let mut raw4k = Vec::new();
        time("frame_to_raw 4K yuv420", 10, || frame_to_raw(&big, &f_out, &mut raw4k));
        let mut tmp = vec![0.0f32; w * h];
        let mut tmp2 = vec![0.0f32; w * h];
        time("guided_smooth r3 1080p plane", 10, || guided_smooth(src.plane(0), &mut tmp, w, h, 3, 1e-3));
        time("cas 1080p plane", 10, || cas(src.plane(0), &mut tmp, w, h, 0.5));
        time("local_minmax 1080p plane", 10, || local_minmax(src.plane(0), w, h, &mut tmp, &mut tmp2));
        let mut out = Vec::new();
        time("duplicate 4K frame", 10, || out.push(big.duplicate()));
    }

    /// Pipe throughput from an FFmpeg child: std pipes vs crate::process large-buffer pipes.
    #[test]
    #[ignore]
    fn pipe_bench() {
        use std::io::Read;
        let tools = crate::Tools::locate().unwrap();
        let args = ["-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=gray:s=1920x1080:r=30", "-frames:v", "300", "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"];
        let frame = 1920 * 1080 * 3 / 2;
        let mut buf = vec![0u8; frame];
        let mut drain = |r: &mut dyn Read| -> usize {
            let mut total = 0;
            loop {
                match r.read(&mut buf) {
                    Ok(0) | Err(_) => return total,
                    Ok(n) => total += n,
                }
            }
        };
        let t = Instant::now();
        let mut child = std::process::Command::new(&tools.ffmpeg).args(args).stdout(std::process::Stdio::piped()).spawn().unwrap();
        let n = drain(child.stdout.as_mut().unwrap());
        child.wait().unwrap();
        println!("std pipe:   {:.0} MB/s ({} frames/s)", n as f64 / 1e6 / t.elapsed().as_secs_f64(), (n / frame) as f64 / t.elapsed().as_secs_f64());
        let control = crate::Control::new();
        let t = Instant::now();
        let sp = crate::process::Proc::spawn(&tools.ffmpeg, &args.iter().map(|s| s.to_string()).collect::<Vec<_>>(), &std::env::temp_dir(), "bench", false, true, None, &control).unwrap();
        let mut out = sp.stdout.unwrap();
        let n = drain(&mut out);
        sp.proc.wait().unwrap();
        println!("prism pipe: {:.0} MB/s ({} frames/s)", n as f64 / 1e6 / t.elapsed().as_secs_f64(), (n / frame) as f64 / t.elapsed().as_secs_f64());
    }

    /// Times raw inference of an installed model: PRISM_BENCH_MODEL=path.onnx, PRISM_BENCH_DEVICE=dml:1
    #[test]
    #[ignore]
    fn inference_bench() {
        use crate::ai::runtime::OrtModel;
        use prism_core::settings::{Backend, DeviceSettings};
        let Ok(path) = std::env::var("PRISM_BENCH_MODEL") else { return };
        let dev = std::env::var("PRISM_BENCH_DEVICE").unwrap_or_else(|_| "dml:0".into());
        let (kind, idx) = dev.split_once(':').map(|(k, i)| (k.to_string(), i.parse().unwrap_or(0))).unwrap_or((dev.clone(), 0));
        let device = DeviceSettings { backend: if kind == "cpu" { Backend::Cpu } else { Backend::DirectMl }, device_index: idx, ..Default::default() };
        let sides: Vec<usize> = std::env::var("PRISM_BENCH_SIDES").ok().map(|s| s.split(',').filter_map(|v| v.parse().ok()).collect()).unwrap_or_default();
        for side in sides {
            let mut m = OrtModel::load(std::path::Path::new(&path), &device, &[("batch", 1), ("height", side as i64), ("width", side as i64)]).unwrap();
            let input = vec![0.5f32; 3 * side * side];
            let t = Instant::now();
            m.run1(&input, [1, 3, side, side]).unwrap();
            let first = t.elapsed().as_secs_f64() * 1000.0;
            let t = Instant::now();
            for _ in 0..4 {
                m.run1(&input, [1, 3, side, side]).unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 250.0;
            println!("side {side:5}: {ms:8.1} ms/run  {:6.2} Mpx/s  (first run {first:.0} ms)", (side * side) as f64 / ms / 1000.0);
        }
        for (w, h) in [(672usize, 400usize), (992, 576), (1952, 1104)] {
            let mut m = OrtModel::load(std::path::Path::new(&path), &device, &[("batch", 1), ("height", h as i64), ("width", w as i64)]).unwrap();
            let input = vec![0.5f32; 3 * w * h];
            time(&format!("{} {}x{} on {}", std::path::Path::new(&path).file_name().unwrap().to_string_lossy(), w, h, m.provider), 5, || m.run1(&input, [1, 3, h, w]).unwrap().0);
        }
        // Aggregate throughput with several sessions running concurrently.
        let (w, h) = (672usize, 400usize);
        for sessions in [1usize, 2, 3] {
            let models: Vec<_> = (0..sessions).map(|_| OrtModel::load(std::path::Path::new(&path), &device, &[("batch", 1), ("height", h as i64), ("width", w as i64)]).unwrap()).collect();
            let runs = 12;
            let t = Instant::now();
            std::thread::scope(|s| {
                for mut m in models {
                    s.spawn(move || {
                        let input = vec![0.5f32; 3 * w * h];
                        for _ in 0..runs / sessions {
                            m.run1(&input, [1, 3, h, w]).unwrap();
                        }
                    });
                }
            });
            println!("{sessions} concurrent session(s): {:.1} ms per tile", t.elapsed().as_secs_f64() * 1000.0 / runs as f64);
        }
    }
}
