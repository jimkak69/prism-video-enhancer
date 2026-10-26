//! ONNX Runtime integration: dynamic loading, provider selection and tiled inference.
//!
//! ONNX Runtime is loaded at run time, so builds never link it. Search order:
//! `PRISM_ORT_DYLIB`, the library next to the executable (or in `runtime/`), then the copy
//! Windows 11 ships in System32 (ONNX Runtime 1.17 with DirectML) or the usual library folders
//! on Linux and macOS.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use anyhow::{Context, Result, anyhow, bail};
use ort::ep::{self, ExecutionProvider};
use ort::session::Session;
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::value::{TensorRef, ValueType};
use prism_core::settings::{Backend, DeviceSettings};
use rayon::prelude::*;

use super::tiling::{self, TileGrid};
use crate::frame::{Frame, Scratch};

#[derive(Debug, Clone)]
pub struct RuntimeInfo {
    pub library: PathBuf,
    /// Execution providers compiled into the loaded library ("CPU" is always present).
    pub providers: Vec<&'static str>,
}

static RUNTIME: OnceLock<Result<RuntimeInfo, String>> = OnceLock::new();
static ENVIRONMENT: OnceLock<ort::environment::Environment> = OnceLock::new();

fn candidates() -> Vec<PathBuf> {
    let mut v = Vec::new();
    if let Some(p) = std::env::var_os("PRISM_ORT_DYLIB") {
        v.push(PathBuf::from(p));
    }
    let name = if cfg!(windows) {
        "onnxruntime.dll"
    } else if cfg!(target_os = "macos") {
        "libonnxruntime.dylib"
    } else {
        "libonnxruntime.so"
    };
    if let Some(dir) = prism_core::paths::exe_dir() {
        v.push(dir.join(name));
        v.push(dir.join("runtime").join(name));
    }
    if cfg!(windows) {
        if let Some(root) = std::env::var_os("SystemRoot") {
            v.push(PathBuf::from(root).join("System32").join(name));
        }
    } else {
        for d in ["/usr/local/lib", "/usr/lib", "/usr/lib/x86_64-linux-gnu", "/opt/homebrew/lib"] {
            v.push(Path::new(d).join(name));
        }
    }
    v
}

fn init() -> Result<RuntimeInfo, String> {
    let mut errors = Vec::new();
    for path in candidates() {
        if !path.is_file() {
            continue;
        }
        let built = ort::init_from(&path).map_err(|e| e.to_string()).and_then(|b| {
            let mut b = b.with_name("prism");
            // PRISM_ORT_LOG=info|verbose prints ONNX Runtime's own diagnostics (node placement etc.).
            if ort_log_level().is_some() {
                b = b.with_logger(std::sync::Arc::new(|level, _category, _id, _location, message| eprintln!("[onnxruntime {level:?}] {message}")));
            }
            b.build().map_err(|e| e.to_string())
        });
        match built {
            Ok(env) => {
                let _ = ENVIRONMENT.set(env);
                let mut providers = vec!["CPU"];
                if ep::DirectML::is_available().unwrap_or(false) {
                    providers.push("DirectML");
                }
                if ep::CUDA::is_available().unwrap_or(false) {
                    providers.push("CUDA");
                }
                if ep::CoreML::is_available().unwrap_or(false) {
                    providers.push("CoreML");
                }
                return Ok(RuntimeInfo { library: path, providers });
            }
            Err(e) => errors.push(format!("{}: {e}", path.display())),
        }
    }
    Err(if errors.is_empty() {
        format!("ONNX Runtime was not found. Place {} next to the Prism executable (or in runtime/) or set PRISM_ORT_DYLIB.", candidates().get(1).and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| "the ONNX Runtime library".into()))
    } else {
        format!("ONNX Runtime could not be loaded: {}", errors.join("; "))
    })
}

/// Loads ONNX Runtime once per process.
pub fn runtime() -> Result<&'static RuntimeInfo> {
    RUNTIME.get_or_init(init).as_ref().map_err(|e| anyhow!("{e}"))
}

fn ort_log_level() -> Option<ort::logging::LogLevel> {
    match std::env::var("PRISM_ORT_LOG").ok()?.to_ascii_lowercase().as_str() {
        "verbose" => Some(ort::logging::LogLevel::Verbose),
        "info" => Some(ort::logging::LogLevel::Info),
        _ => Some(ort::logging::LogLevel::Warning),
    }
}

fn b(r: std::result::Result<SessionBuilder, ort::Error<SessionBuilder>>) -> Result<SessionBuilder> {
    r.map_err(|e| anyhow!(ort::Error::<()>::from(e)))
}

fn build_session(path: &Path, device: &DeviceSettings, overrides: &[(&str, i64)], provider: &str) -> Result<Session> {
    let env = ENVIRONMENT.get().ok_or_else(|| anyhow!("ONNX Runtime environment is not initialised"))?;
    let mut sb = Session::builder(env).map_err(|e| anyhow!(e))?;
    if let Some(level) = ort_log_level() {
        sb = b(sb.with_log_level(level))?;
    }
    for (name, value) in overrides {
        sb = b(sb.with_dimension_override(*name, *value))?;
    }
    sb = b(sb.with_optimization_level(GraphOptimizationLevel::All))?;
    let device_id = device.device_index.max(0);
    match provider {
        "DirectML" => {
            sb = b(sb.with_memory_pattern(false))?;
            sb = b(sb.with_parallel_execution(false))?;
            sb = b(sb.with_execution_providers([ep::DirectML::default().with_device_id(device_id).build().error_on_failure()]))?;
        }
        "CUDA" => {
            sb = b(sb.with_execution_providers([ep::CUDA::default().with_device_id(device_id).build().error_on_failure()]))?;
        }
        "CoreML" => {
            // Compiled Core ML models are cached so later sessions start quickly.
            let cache = prism_core::paths::data_dir().join("coreml-cache");
            let _ = std::fs::create_dir_all(&cache);
            let coreml = ep::CoreML::default().with_static_input_shapes(true).with_model_cache_dir(cache.display());
            sb = b(sb.with_execution_providers([coreml.build().error_on_failure()]))?;
        }
        _ => {
            let threads = if device.threads > 0 { device.threads } else { std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) };
            sb = b(sb.with_intra_threads(threads))?;
        }
    }
    sb.commit_from_file(path).map_err(|e| anyhow!(e)).with_context(|| format!("loading {}", path.display()))
}

/// A loaded ONNX model with the provider that runs it.
pub struct OrtModel {
    session: Session,
    pub path: PathBuf,
    pub provider: &'static str,
    pub inputs: Vec<(String, Vec<i64>)>,
    pub output: String,
}

impl OrtModel {
    pub fn load(path: &Path, device: &DeviceSettings, overrides: &[(&str, i64)]) -> Result<OrtModel> {
        let rt = runtime()?;
        let order: Vec<&'static str> = match device.backend {
            Backend::Cpu => vec!["CPU"],
            Backend::DirectMl => vec!["DirectML"],
            Backend::Cuda => vec!["CUDA"],
            Backend::CoreMl => vec!["CoreML"],
            Backend::Auto => vec!["CUDA", "DirectML", "CoreML", "CPU"],
        };
        let mut errors = Vec::new();
        for provider in order {
            if !rt.providers.contains(&provider) {
                errors.push(format!("{provider}: not available in {}", rt.library.display()));
                continue;
            }
            match build_session(path, device, overrides, provider) {
                Ok(session) => {
                    let inputs = session
                        .inputs()
                        .iter()
                        .map(|o| {
                            let dims = match o.dtype() {
                                ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
                                _ => Vec::new(),
                            };
                            (o.name().to_string(), dims)
                        })
                        .collect();
                    let output = session.outputs().first().map(|o| o.name().to_string()).ok_or_else(|| anyhow!("model has no outputs"))?;
                    return Ok(OrtModel { session, path: path.to_path_buf(), provider, inputs, output });
                }
                Err(e) => errors.push(format!("{provider}: {e:#}")),
            }
        }
        bail!("could not create an inference session for {}: {}", path.display(), errors.join("; "))
    }

    /// Runs a single-input model on an NCHW tensor, returning (shape, data).
    pub fn run1(&mut self, input: &[f32], shape: [usize; 4]) -> Result<(Vec<usize>, Vec<f32>)> {
        let name = self.inputs.first().map(|i| i.0.clone()).ok_or_else(|| anyhow!("model has no inputs"))?;
        let t = TensorRef::from_array_view((shape, input)).map_err(|e| anyhow!(e))?;
        let outputs = self.session.run(ort::inputs![name.as_str() => t]).map_err(|e| anyhow!(e))?;
        let (s, d) = outputs[self.output.as_str()].try_extract_tensor::<f32>().map_err(|e| anyhow!(e))?;
        Ok((s.iter().map(|v| *v as usize).collect(), d.to_vec()))
    }

    /// Runs a model with several named inputs of arbitrary shapes.
    pub fn run_named(&mut self, inputs: Vec<(String, Vec<usize>, &[f32])>) -> Result<(Vec<usize>, Vec<f32>)> {
        let mut values = Vec::with_capacity(inputs.len());
        for (name, shape, data) in &inputs {
            let t = TensorRef::from_array_view((shape.clone(), *data)).map_err(|e| anyhow!(e))?;
            values.push((std::borrow::Cow::<str>::Owned(name.clone()), ort::session::SessionInputValue::from(t)));
        }
        let outputs = self.session.run(values).map_err(|e| anyhow!(e))?;
        let (s, d) = outputs[self.output.as_str()].try_extract_tensor::<f32>().map_err(|e| anyhow!(e))?;
        Ok((s.iter().map(|v| *v as usize).collect(), d.to_vec()))
    }
}

/// A session shared through the cache below.
pub type SharedModel = Arc<Mutex<OrtModel>>;

const SESSION_CACHE: usize = 3;
static SESSIONS: Mutex<Vec<(String, SharedModel)>> = Mutex::new(Vec::new());

/// Loads a session, reusing a recently loaded one with the same file, shape and device. GPU
/// sessions take ~1 s to compile, which would otherwise be paid for every preview.
pub fn load_shared(path: &Path, device: &DeviceSettings, overrides: &[(&str, i64)]) -> Result<SharedModel> {
    let key = format!("{}|{:?}|{}|{}|{:?}", path.display(), device.backend, device.device_index, device.threads, overrides);
    {
        let mut cache = SESSIONS.lock().unwrap();
        if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
            let entry = cache.remove(i);
            let model = entry.1.clone();
            cache.push(entry);
            return Ok(model);
        }
    }
    let model = Arc::new(Mutex::new(OrtModel::load(path, device, overrides)?));
    let mut cache = SESSIONS.lock().unwrap();
    cache.push((key, model.clone()));
    if cache.len() > SESSION_CACHE {
        cache.remove(0);
    }
    Ok(model)
}

/// Drops cached sessions (frees GPU memory).
pub fn clear_session_cache() {
    SESSIONS.lock().unwrap().clear();
}

fn tile_cache_path() -> PathBuf {
    prism_core::paths::data_dir().join("tile-cache.json")
}

/// Forgets tuned tile sizes so the next run measures again (e.g. after a driver or power change).
pub fn clear_tile_cache() -> std::io::Result<()> {
    match std::fs::remove_file(tile_cache_path()) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(e),
        _ => Ok(()),
    }
}

/// Tile edge (input pixels) with the best throughput for this model and device.
///
/// Throughput per pixel collapses once tiles no longer fit in free GPU memory (other
/// applications may hold most of it), so candidate sizes are timed once and the result cached
/// per model file and device. CPU sessions do not need tuning.
pub fn tuned_tile_side(path: &Path, device: &DeviceSettings, heavy: bool) -> Result<usize> {
    let default = if heavy { 256 } else { 512 };
    if device.backend == Backend::Cpu {
        return Ok(default);
    }
    let len = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let key = format!("{}|{len}|{:?}|{}", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), device.backend, device.device_index);
    let cache_path = tile_cache_path();
    let mut cache: std::collections::HashMap<String, usize> = std::fs::read_to_string(&cache_path).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default();
    if let Some(side) = cache.get(&key) {
        return Ok(*side);
    }
    let candidates: &[usize] = if heavy { &[192, 256, 384] } else { &[256, 384, 512, 768] };
    let mut rates: Vec<(usize, f64)> = Vec::new();
    let mut prev: Option<(usize, f64)> = None;
    for &side in candidates {
        let mut m = OrtModel::load(path, device, &[("batch", 1), ("height", side as i64), ("width", side as i64)])?;
        if m.provider == "CPU" {
            return Ok(default);
        }
        let input = vec![0.5f32; 3 * side * side];
        m.run1(&input, [1, 3, side, side])?;
        // Median of three runs: other applications sharing the GPU make single samples noisy.
        let mut samples = Vec::new();
        for _ in 0..3 {
            let t = std::time::Instant::now();
            m.run1(&input, [1, 3, side, side])?;
            samples.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let ms = samples[1];
        // Useful pixels exclude the overlap shared with neighbouring tiles.
        let useful = (side.saturating_sub(TILE_OVERLAP) as f64).powi(2);
        rates.push((side, useful / ms));
        // Super-linear slowdown means the tile no longer fits; larger ones will be worse.
        if let Some((ps, pms)) = prev {
            if ms > pms * ((side * side) as f64 / (ps * ps) as f64) * 1.6 {
                break;
            }
        }
        prev = Some((side, ms));
    }
    // The largest tile within 15% of the best rate: fewer tiles, fewer seams, less overhead.
    let best = rates.iter().map(|r| r.1).fold(0.0, f64::max);
    let side = rates.iter().filter(|r| r.1 >= best * 0.85).map(|r| r.0).max().unwrap_or(default);
    cache.insert(key, side);
    if let Some(dir) = cache_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let _ = std::fs::write(&cache_path, serde_json::to_vec_pretty(&cache).unwrap_or_default());
    Ok(side)
}

/// Image-to-image network run on fixed-size tiles with feathered seams.
pub struct OnnxUpscaler {
    model: SharedModel,
    /// Tiles over the frame extended by `BORDER` replicated pixels on every side.
    grid: TileGrid,
    width: usize,
    height: usize,
    scale: usize,
    pub description: String,
}

/// Overlap between tiles in input pixels.
pub const TILE_OVERLAP: usize = 32;
/// Replicated border added around the frame so its edges see image-like context instead of the
/// network's zero padding, independent of how the frame is tiled.
const BORDER: usize = TILE_OVERLAP / 2;
const ALIGN: usize = 16;

impl OnnxUpscaler {
    /// Prepares a session for frames of `w x h`. `heavy` marks large networks (RRDB) so tiles are
    /// smaller. The output scale is measured from the model on the first tile.
    pub fn new(path: &Path, w: usize, h: usize, device: &DeviceSettings, heavy: bool) -> Result<OnnxUpscaler> {
        let side = match device.tile {
            Some(t) => t as usize,
            None => tuned_tile_side(path, device, heavy)?,
        };
        // A frame that fits in a slightly larger single tile is cheaper than splitting it.
        let max_pixels = side * side * 5 / 4;
        let grid = tiling::plan(w + 2 * BORDER, h + 2 * BORDER, max_pixels, TILE_OVERLAP, ALIGN);
        let overrides = [("batch", 1i64), ("height", grid.tile_h as i64), ("width", grid.tile_w as i64)];
        let shared = load_shared(path, device, &overrides)?;
        let mut model = shared.lock().unwrap();
        // Measure the scale with one tile-shaped run (also compiles GPU kernels up front).
        let probe = vec![0.5f32; 3 * grid.tile_w * grid.tile_h];
        let (shape, _) = model.run1(&probe, [1, 3, grid.tile_h, grid.tile_w])?;
        if shape.len() != 4 || shape[1] != 3 || shape[3] % grid.tile_w != 0 {
            bail!("unexpected model output shape {shape:?} for input [1, 3, {}, {}]", grid.tile_h, grid.tile_w);
        }
        let scale = shape[3] / grid.tile_w;
        let description = format!(
            "{} on {} ({}x tiles {}x{} x{}, overlap {})",
            path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(),
            model.provider,
            scale,
            grid.tile_w,
            grid.tile_h,
            grid.count(),
            TILE_OVERLAP
        );
        drop(model);
        Ok(OnnxUpscaler { model: shared, grid, width: w, height: h, scale, description })
    }

    pub fn scale(&self) -> usize {
        self.scale
    }

    pub fn provider(&self) -> &'static str {
        self.model.lock().unwrap().provider
    }

    pub fn upscale(&mut self, src: &Frame) -> Result<Frame> {
        let (w, h) = (src.width, src.height);
        if (w, h) != (self.width, self.height) {
            bail!("upscaler prepared for {}x{} but got {w}x{h}", self.width, self.height);
        }
        let s = self.scale;
        let (tw, th) = (self.grid.tile_w, self.grid.tile_h);
        let (vw, vh) = (self.grid.width, self.grid.height);
        let (ow, oh) = (w * s, h * s);
        if self.grid.count() == 1 {
            return self.upscale_single(src);
        }
        let mut out = Frame::zeroed(ow, oh);
        let mut weight = Scratch::zeroed(ow * oh);
        let mut tile = Scratch::new(3 * tw * th);
        let overlap_out = TILE_OVERLAP * s;
        let border_out = (BORDER * s) as isize;
        let xs = self.grid.xs.clone();
        let ys = self.grid.ys.clone();
        for &y0 in &ys {
            for &x0 in &xs {
                // Gather the tile from the virtually bordered frame (clamped source coordinates).
                for c in 0..3 {
                    let plane = src.plane(c);
                    tile[c * tw * th..(c + 1) * tw * th].par_chunks_mut(tw).enumerate().for_each(|(ty, row)| {
                        let sy = (y0 + ty).saturating_sub(BORDER).min(h - 1);
                        let srow = &plane[sy * w..(sy + 1) * w];
                        let first = x0 as isize - BORDER as isize;
                        // Contiguous in-frame span of this row, then replicated edges.
                        let lo = (-first).clamp(0, tw as isize) as usize;
                        let hi = ((w as isize - first).clamp(0, tw as isize)) as usize;
                        if hi > lo {
                            let s0 = (first + lo as isize) as usize;
                            row[lo..hi].copy_from_slice(&srow[s0..s0 + (hi - lo)]);
                        }
                        row[..lo].fill(srow[0]);
                        row[hi..].fill(srow[w - 1]);
                    });
                }
                let (shape, data) = self.model.lock().unwrap().run1(&tile, [1, 3, th, tw])?;
                if shape != [1, 3, th * s, tw * s] {
                    bail!("model returned {shape:?}, expected [1, 3, {}, {}]", th * s, tw * s);
                }
                let (has_left, has_right) = (x0 > 0, x0 + tw < vw);
                let (has_top, has_bottom) = (y0 > 0, y0 + th < vh);
                // Tile output pixel (u, v) lands at frame output (x0*s + u - border, y0*s + v - border).
                let ox = (x0 * s) as isize - border_out;
                let oy = (y0 * s) as isize - border_out;
                let u_lo = (-ox).max(0) as usize;
                let u_hi = ((ow as isize - ox).min((tw * s) as isize)).max(0) as usize;
                let v_lo = (-oy).max(0) as usize;
                let v_hi = ((oh as isize - oy).min((th * s) as isize)).max(0) as usize;
                if u_hi <= u_lo || v_hi <= v_lo {
                    continue;
                }
                let wx: Vec<f32> = (u_lo..u_hi).map(|u| TileGrid::ramp(u, tw * s, overlap_out, has_left, has_right)).collect();
                let (dx0, dy0) = ((ox + u_lo as isize) as usize, (oy + v_lo as isize) as usize);
                let span = u_hi - u_lo;
                for (c, plane) in out.planes_mut().into_iter().enumerate() {
                    let tdata = &data[c * tw * s * th * s..(c + 1) * tw * s * th * s];
                    plane[dy0 * ow..(dy0 + v_hi - v_lo) * ow].par_chunks_mut(ow).enumerate().for_each(|(k, row)| {
                        let v = v_lo + k;
                        let wy = TileGrid::ramp(v, th * s, overlap_out, has_top, has_bottom);
                        let trow = &tdata[v * tw * s + u_lo..v * tw * s + u_hi];
                        for ((o, t), wxv) in row[dx0..dx0 + span].iter_mut().zip(trow).zip(&wx) {
                            *o += t * wxv * wy;
                        }
                    });
                }
                weight[dy0 * ow..(dy0 + v_hi - v_lo) * ow].par_chunks_mut(ow).enumerate().for_each(|(k, row)| {
                    let wy = TileGrid::ramp(v_lo + k, th * s, overlap_out, has_top, has_bottom);
                    for (o, wxv) in row[dx0..dx0 + span].iter_mut().zip(&wx) {
                        *o += wxv * wy;
                    }
                });
            }
        }
        let weight = &weight;
        for plane in out.planes_mut() {
            plane.par_chunks_mut(ow).zip(weight.par_chunks(ow)).for_each(|(row, wrow)| {
                for (o, wv) in row.iter_mut().zip(wrow) {
                    *o = if *wv > 0.0 { *o / wv } else { *o };
                }
            });
        }
        Ok(out)
    }
}

impl OnnxUpscaler {
    /// One tile covers the bordered frame: no blending, copy the interior straight out.
    fn upscale_single(&mut self, src: &Frame) -> Result<Frame> {
        let (w, h) = (src.width, src.height);
        let s = self.scale;
        let (tw, th) = (self.grid.tile_w, self.grid.tile_h);
        let mut tile = Scratch::new(3 * tw * th);
        for c in 0..3 {
            let plane = src.plane(c);
            tile[c * tw * th..(c + 1) * tw * th].par_chunks_mut(tw).enumerate().for_each(|(ty, row)| {
                let sy = ty.saturating_sub(BORDER).min(h - 1);
                let srow = &plane[sy * w..(sy + 1) * w];
                row[..BORDER].fill(srow[0]);
                row[BORDER..BORDER + w].copy_from_slice(srow);
                row[BORDER + w..].fill(srow[w - 1]);
            });
        }
        let (shape, data) = self.model.lock().unwrap().run1(&tile, [1, 3, th, tw])?;
        if shape != [1, 3, th * s, tw * s] {
            bail!("model returned {shape:?}, expected [1, 3, {}, {}]", th * s, tw * s);
        }
        let (ow, oh, b) = (w * s, h * s, BORDER * s);
        let mut out = Frame::uninit(ow, oh);
        for c in 0..3 {
            let tdata = &data[c * tw * s * th * s..(c + 1) * tw * s * th * s];
            out.plane_mut(c).par_chunks_mut(ow).enumerate().for_each(|(y, row)| {
                let start = (y + b) * tw * s + b;
                row.copy_from_slice(&tdata[start..start + ow]);
            });
        }
        Ok(out)
    }
}

/// How a RIFE-style network expects its inputs.
enum RifeLayout {
    /// `img0`, `img1` and a timestep input (scalar or per-pixel plane).
    Separate { img0: String, img1: String, timestep: String, ts_plane: bool },
    /// One tensor: img0 ‖ img1 ‖ timestep plane (7 channels).
    Concat7(String),
}

/// Frame interpolation through a RIFE-style ONNX network, run on whole padded frames.
pub struct OnnxInterpolator {
    model: OrtModel,
    layout: RifeLayout,
    pw: usize,
    ph: usize,
    pub description: String,
}

impl OnnxInterpolator {
    pub fn new(path: &Path, w: usize, h: usize, device: &DeviceSettings) -> Result<OnnxInterpolator> {
        let (pw, ph) = (w.div_ceil(64) * 64, h.div_ceil(64) * 64);
        let overrides = [("batch", 1i64), ("height", ph as i64), ("width", pw as i64)];
        let model = OrtModel::load(path, device, &overrides)?;
        let names: Vec<String> = model.inputs.iter().map(|i| i.0.to_lowercase()).collect();
        let layout = match model.inputs.len() {
            1 => {
                let ch = model.inputs[0].1.get(1).copied().unwrap_or(-1);
                if ch != 7 && ch != -1 {
                    bail!("single-input interpolation model must take 7 channels (img0, img1, timestep); found {ch}");
                }
                RifeLayout::Concat7(model.inputs[0].0.clone())
            }
            3 => {
                let find = |keys: &[&str]| names.iter().position(|n| keys.iter().any(|k| n.contains(k)));
                let ts = find(&["time", "step", "t"]).ok_or_else(|| anyhow!("cannot identify the timestep input among {names:?}"))?;
                let others: Vec<usize> = (0..3).filter(|i| *i != ts).collect();
                let ts_plane = model.inputs[ts].1.len() == 4;
                RifeLayout::Separate { img0: model.inputs[others[0]].0.clone(), img1: model.inputs[others[1]].0.clone(), timestep: model.inputs[ts].0.clone(), ts_plane }
            }
            n => bail!("unsupported interpolation model with {n} inputs"),
        };
        let description = format!("{} on {} ({}x{} padded)", path.file_name().map(|n| n.to_string_lossy()).unwrap_or_default(), model.provider, pw, ph);
        Ok(OnnxInterpolator { model, layout, pw, ph, description })
    }

    fn pad(&self, f: &Frame, dst: &mut [f32]) {
        let (pw, ph, w, h) = (self.pw, self.ph, f.width, f.height);
        for c in 0..3 {
            let plane = f.plane(c);
            dst[c * pw * ph..(c + 1) * pw * ph].par_chunks_mut(pw).enumerate().for_each(|(y, row)| {
                let sy = y.min(h - 1);
                row[..w].copy_from_slice(&plane[sy * w..(sy + 1) * w]);
                let e = plane[sy * w + w - 1];
                row[w..].fill(e);
            });
        }
    }

    pub fn interpolate(&mut self, a: &Frame, bf: &Frame, t: f32) -> Result<Frame> {
        let (pw, ph) = (self.pw, self.ph);
        let n = pw * ph;
        let (shape, data) = match &self.layout {
            RifeLayout::Concat7(name) => {
                let mut buf = vec![0.0f32; 7 * n];
                self.pad(a, &mut buf[..3 * n]);
                self.pad(bf, &mut buf[3 * n..6 * n]);
                buf[6 * n..].fill(t);
                let name = name.clone();
                self.model.run_named(vec![(name, vec![1, 7, ph, pw], &buf)])?
            }
            RifeLayout::Separate { img0, img1, timestep, ts_plane } => {
                let mut i0 = vec![0.0f32; 3 * n];
                let mut i1 = vec![0.0f32; 3 * n];
                self.pad(a, &mut i0);
                self.pad(bf, &mut i1);
                let ts: Vec<f32> = if *ts_plane { vec![t; n] } else { vec![t] };
                let ts_shape = if *ts_plane { vec![1, 1, ph, pw] } else { vec![1] };
                let (img0, img1, timestep) = (img0.clone(), img1.clone(), timestep.clone());
                self.model.run_named(vec![(img0, vec![1, 3, ph, pw], &i0), (img1, vec![1, 3, ph, pw], &i1), (timestep, ts_shape, &ts)])?
            }
        };
        if shape.len() != 4 || shape[1] < 3 || shape[2] != ph || shape[3] != pw {
            bail!("interpolation model returned {shape:?}, expected [1, 3, {ph}, {pw}]");
        }
        let (w, h) = (a.width, a.height);
        let mut out = Frame::uninit(w, h);
        for c in 0..3 {
            let src = &data[c * n..(c + 1) * n];
            out.plane_mut(c).par_chunks_mut(w).enumerate().for_each(|(y, row)| row.copy_from_slice(&src[y * pw..y * pw + w]));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ai::convert::{build_srvgg, tests::{reference_srvgg, tiny_srvgg}};

    /// Runs only where an ONNX Runtime can be loaded (Windows 11 ships one).
    #[test]
    fn converted_srvgg_matches_reference_on_cpu() {
        if runtime().is_err() {
            eprintln!("skipping: {}", runtime().unwrap_err());
            return;
        }
        let sd = tiny_srvgg(4, 2, 2);
        let (bytes, scale, _) = build_srvgg(&sd, false).unwrap();
        let path = std::env::temp_dir().join(format!("prism-srvgg-{}.onnx", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let (w, h) = (9, 7);
        let input: Vec<f32> = (0..3 * w * h).map(|i| ((i * 31) % 17) as f32 / 17.0).collect();
        let (expected, s) = reference_srvgg(&sd, &input, w, h);
        assert_eq!(s, scale as usize);
        let device = DeviceSettings { backend: Backend::Cpu, ..DeviceSettings::default() };
        let mut model = OrtModel::load(&path, &device, &[]).unwrap();
        let (shape, got) = model.run1(&input, [1, 3, h, w]).unwrap();
        assert_eq!(shape, vec![1, 3, h * s, w * s]);
        let err = expected.iter().zip(&got).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "max error {err}");
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn tiled_upscale_matches_whole_frame() {
        if runtime().is_err() {
            return;
        }
        let sd = tiny_srvgg(4, 1, 2);
        let (bytes, _, _) = build_srvgg(&sd, false).unwrap();
        let path = std::env::temp_dir().join(format!("prism-srvgg-tiles-{}.onnx", std::process::id()));
        std::fs::write(&path, &bytes).unwrap();
        let (w, h) = (150, 90);
        let mut f = Frame::uninit(w, h);
        for (i, v) in f.data_mut().iter_mut().enumerate() {
            *v = ((i * 13) % 101) as f32 / 101.0;
        }
        let cpu = DeviceSettings { backend: Backend::Cpu, ..DeviceSettings::default() };
        let whole = OnnxUpscaler::new(&path, w, h, &cpu, false).unwrap().upscale(&f).unwrap();
        let tiled_dev = DeviceSettings { tile: Some(96), ..cpu };
        let mut tiled = OnnxUpscaler::new(&path, w, h, &tiled_dev, false).unwrap();
        assert!(tiled.grid.count() > 1);
        let t = tiled.upscale(&f).unwrap();
        // The tiny network's receptive field (3 convs) is far below the overlap, so seams vanish.
        let err = whole.data().iter().zip(t.data()).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max);
        assert!(err < 1e-4, "max error {err}");
        std::fs::remove_file(path).ok();
    }
}
