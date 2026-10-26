//! Runs a [`JobPlan`]: passes in order, progress events, cleanup.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use prism_core::errors::diagnose;
use prism_core::models::Catalog;
use prism_core::plan::{JobPlan, Pass};
use prism_core::progress::{Progress, ProgressTracker, parse_ffmpeg_stats};

use crate::ffmpeg::Tools;
use crate::frame::Frame;
use crate::pipeline::{self, Sink, process_error};
use crate::process::{Cancelled, Control, Proc};

#[derive(Debug, Clone)]
pub enum Event {
    PassStarted { index: usize, count: usize, label: String },
    Progress(Progress),
    Info(String),
    Warning(String),
}

#[derive(Debug, Clone, Default)]
pub struct RunReport {
    pub output: PathBuf,
    pub elapsed_secs: f64,
    pub frames: u64,
    /// Output frames per second over the main pass.
    pub fps: f64,
    pub stages: Vec<String>,
    pub stats: Vec<(String, u64)>,
    /// Busy milliseconds per output frame for each pipeline step of the engine pass.
    pub timings: Vec<(String, f64)>,
}

pub fn is_cancelled(e: &anyhow::Error) -> bool {
    e.is::<Cancelled>() || e.chain().any(|c| c.is::<Cancelled>())
}

fn remove_output(plan: &JobPlan) {
    if plan.output == Path::new("-") {
        return;
    }
    if plan.output_is_sequence {
        if let Some(dir) = plan.output.parent() {
            let _ = std::fs::remove_dir_all(dir);
        }
    } else {
        let _ = std::fs::remove_file(&plan.output);
    }
}

/// Runs every pass of `plan`. Partial outputs are deleted on failure or cancellation; the
/// scratch directory is deleted afterwards except for previews (which live there).
pub fn run_plan(plan: &JobPlan, tools: &Tools, catalog: &Catalog, control: &Arc<Control>, on_event: &(dyn Fn(Event) + Sync)) -> Result<RunReport> {
    std::fs::create_dir_all(&plan.temp_dir).with_context(|| format!("creating {}", plan.temp_dir.display()))?;
    let out_dir = plan.output.parent().map(Path::to_path_buf).unwrap_or_default();
    if !out_dir.as_os_str().is_empty() {
        std::fs::create_dir_all(&out_dir).with_context(|| format!("creating {}", out_dir.display()))?;
    }
    for w in &plan.warnings {
        on_event(Event::Warning(w.clone()));
    }
    let result = run_passes(plan, tools, catalog, control, on_event);
    if result.is_err() {
        remove_output(plan);
    }
    if !plan.is_preview {
        let _ = std::fs::remove_dir_all(&plan.temp_dir);
    }
    result
}

fn run_passes(plan: &JobPlan, tools: &Tools, catalog: &Catalog, control: &Arc<Control>, on_event: &(dyn Fn(Event) + Sync)) -> Result<RunReport> {
    let start = Instant::now();
    let count = plan.passes.len();
    let mut report = RunReport { output: plan.output.clone(), ..Default::default() };
    for (index, pass) in plan.passes.iter().enumerate() {
        control.checkpoint()?;
        on_event(Event::PassStarted { index, count, label: pass.label().to_string() });
        let tracker = Mutex::new(ProgressTracker::new(pass.frames()));
        let emit = |frame: u64| {
            let mut p = tracker.lock().unwrap().update(frame, Instant::now());
            p.pass = index + 1;
            p.passes = count;
            on_event(Event::Progress(p));
        };
        let pass_start = Instant::now();
        match pass {
            Pass::Ffmpeg(f) => {
                let emit_ref: &(dyn Fn(u64) + Sync) = &emit;
                // The stderr thread is 'static, so progress flows through a channel.
                let (tx, rx) = crossbeam_channel::unbounded::<u64>();
                let handler: crate::process::LineHandler = Box::new(move |line: &str| {
                    if let Some((frame, _)) = parse_ffmpeg_stats(line) {
                        let _ = tx.send(frame);
                    }
                });
                let spawned = Proc::spawn(&tools.ffmpeg, &tools.adapt_args(&f.args), &plan.temp_dir, &f.label, false, false, Some(handler), control)?;
                let proc = spawned.proc;
                let done = std::sync::atomic::AtomicBool::new(false);
                let (status, tail) = std::thread::scope(|s| {
                    s.spawn(|| {
                        while !done.load(std::sync::atomic::Ordering::SeqCst) {
                            if let Ok(frame) = rx.recv_timeout(std::time::Duration::from_millis(100)) {
                                emit_ref(frame);
                            }
                        }
                    });
                    let r = proc.wait();
                    done.store(true, std::sync::atomic::Ordering::SeqCst);
                    r
                })?;
                if control.is_cancelled() {
                    return Err(Cancelled.into());
                }
                if !status.success() {
                    return Err(process_error(&f.label, Some(status), &tail));
                }
                if index + 1 == count {
                    report.frames = f.frames;
                    report.fps = f.frames as f64 / pass_start.elapsed().as_secs_f64().max(1e-3);
                }
            }
            Pass::Engine(e) => {
                let stages = pipeline::build_stages(e, catalog, &plan.device)?;
                for s in &stages {
                    on_event(Event::Info(format!("{}: {}", s.name(), s.describe())));
                }
                let r = pipeline::run(e, tools, &plan.temp_dir, stages, Sink::Encode, control, &emit)?;
                report.frames = r.frames_out;
                report.fps = r.frames_out as f64 / r.elapsed.max(1e-3);
                report.stages = r.stages;
                report.stats = r.stats;
                report.timings = r.timings;
            }
        }
    }
    let discarded = plan.output == Path::new("-");
    if !plan.output_is_sequence && !discarded && !plan.output.exists() {
        return Err(anyhow!("FFmpeg reported success but {} was not written", plan.output.display()));
    }
    report.elapsed_secs = start.elapsed().as_secs_f64();
    Ok(report)
}

/// Renders the frames of an engine plan into memory instead of encoding them (instant previews).
/// Returns (source frames, processed frames). Interpolation stages are skipped by the caller's
/// plan; FFmpeg-only plans return the decoded frames twice.
pub fn render_frames(plan: &JobPlan, tools: &Tools, catalog: &Catalog, control: &Arc<Control>) -> Result<Vec<Frame>> {
    std::fs::create_dir_all(&plan.temp_dir)?;
    let engine = plan.passes.iter().rev().find_map(|p| if let Pass::Engine(e) = p { Some(e.clone()) } else { None });
    let Some(pass) = engine else {
        return Err(anyhow!("this plan has no engine stages to preview"));
    };
    let stages = pipeline::build_stages(&pass, catalog, &plan.device)?;
    let frames = Arc::new(Mutex::new(Vec::new()));
    let r = pipeline::run(&pass, tools, &plan.temp_dir, stages, Sink::Collect(frames.clone()), control, &|_| {});
    let _ = std::fs::remove_dir_all(&plan.temp_dir);
    r?;
    let v = std::mem::take(&mut *frames.lock().unwrap());
    Ok(v)
}

/// Short description of a failure for status displays.
pub fn summarize_error(e: &anyhow::Error) -> String {
    if is_cancelled(e) {
        return "Cancelled".into();
    }
    let text = format!("{e:#}");
    let d = diagnose(text.lines());
    if d.category.is_empty() { "Error".into() } else { d.category }
}
