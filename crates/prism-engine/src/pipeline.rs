//! Executes an engine pass: FFmpeg decode → Rust stages → FFmpeg encode.
//!
//! Every step runs on its own thread connected by small bounded channels, so decoding, each
//! stage (CPU kernels or GPU inference) and encoding overlap. Raw byte buffers are recycled.
//! Pausing suspends both FFmpeg processes; the Rust threads stop at frame boundaries.

use std::io::{Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use crossbeam_channel::{Receiver, Sender, bounded};
use prism_core::errors::diagnose;
use prism_core::models::Catalog;
use prism_core::plan::EnginePass;
use prism_core::settings::DeviceSettings;

use crate::ffmpeg::Tools;
use crate::frame::{Frame, frame_to_raw, raw_to_frame};
use crate::process::{Cancelled, Control, Proc};
use crate::stages::{self, Stage};

/// Where processed frames go.
pub enum Sink {
    /// Into the encoding FFmpeg process described by the pass.
    Encode,
    /// Kept in memory (instant previews).
    Collect(Arc<Mutex<Vec<Frame>>>),
}

#[derive(Debug, Clone, Default)]
pub struct EngineReport {
    pub frames_in: u64,
    pub frames_out: u64,
    pub stages: Vec<String>,
    pub stats: Vec<(String, u64)>,
    /// Busy milliseconds per output frame for each pipeline step (they run concurrently).
    pub timings: Vec<(String, f64)>,
    pub elapsed: f64,
}

/// Formats a process failure with the matching advice.
pub fn process_error(what: &str, status: Option<std::process::ExitStatus>, tail: &[String]) -> anyhow::Error {
    let d = diagnose(tail.iter().map(String::as_str));
    let last: Vec<&String> = tail.iter().rev().take(6).collect::<Vec<_>>().into_iter().rev().collect();
    let mut msg = match status {
        Some(s) => format!("{what} failed ({s})"),
        None => format!("{what} failed"),
    };
    if !d.is_empty() {
        msg += &format!(": {}\n{}", d.category, d.advice.iter().map(|a| format!("- {a}")).collect::<Vec<_>>().join("\n"));
    }
    if !last.is_empty() {
        msg += &format!("\nFFmpeg output:\n{}", last.iter().map(|l| format!("  {l}")).collect::<Vec<_>>().join("\n"));
    }
    anyhow!(msg)
}

/// Builds every stage up front (model loading and GPU compilation happen before decoding starts).
pub fn build_stages(pass: &EnginePass, catalog: &Catalog, device: &DeviceSettings) -> Result<Vec<Box<dyn Stage>>> {
    let (mut w, mut h) = (pass.input.width as usize, pass.input.height as usize);
    let mut out = Vec::new();
    for spec in &pass.stages {
        out.extend(stages::build(spec, w, h, catalog, device).with_context(|| format!("preparing stage '{}'", spec.label()))?);
        (w, h) = stages::output_size(spec, w, h);
    }
    if (w, h) != (pass.output.width as usize, pass.output.height as usize) {
        bail!("stages produce {w}x{h} but the encoder expects {}x{}", pass.output.width, pass.output.height);
    }
    Ok(out)
}

struct Shared {
    failed: AtomicBool,
    error: Mutex<Option<anyhow::Error>>,
}

impl Shared {
    fn fail(&self, e: anyhow::Error) {
        let mut slot = self.error.lock().unwrap();
        if slot.is_none() {
            *slot = Some(e);
        }
        self.failed.store(true, Ordering::SeqCst);
    }
}

/// Runs `pass`. `progress` receives the number of frames delivered to the sink.
pub fn run(pass: &EnginePass, tools: &Tools, cwd: &Path, mut stage_list: Vec<Box<dyn Stage>>, sink: Sink, control: &Arc<Control>, progress: &(dyn Fn(u64) + Sync)) -> Result<EngineReport> {
    let start = Instant::now();
    let descriptions: Vec<String> = stage_list.iter().map(|s| format!("{}: {}", s.name(), s.describe())).collect();
    let decoder = Proc::spawn(&tools.ffmpeg, &tools.adapt_args(&pass.decode), cwd, "decoder", false, true, None, control)?;
    let (dec_proc, mut dec_out) = (decoder.proc, decoder.stdout.ok_or_else(|| anyhow!("decoder has no stdout"))?);
    let (enc_proc, mut enc_in) = match &sink {
        Sink::Encode => {
            let e = Proc::spawn(&tools.ffmpeg, &tools.adapt_args(&pass.encode), cwd, "encoder", true, false, None, control)?;
            (Some(e.proc), Some(e.stdin.ok_or_else(|| anyhow!("encoder has no stdin"))?))
        }
        Sink::Collect(_) => (None, None),
    };

    let shared = Shared { failed: AtomicBool::new(false), error: Mutex::new(None) };
    let finished = AtomicBool::new(false);
    let frames_in = AtomicU64::new(0);
    let frames_out = AtomicU64::new(0);
    let in_bytes = pass.input.frame_bytes();
    let in_fmt = pass.input;
    let out_fmt = pass.output;
    let n_stages = stage_list.len();
    // Busy nanoseconds per pipeline step: unpack, each stage, pack.
    let busy: Vec<AtomicU64> = (0..n_stages + 2).map(|_| AtomicU64::new(0)).collect();
    let step_names: Vec<String> = std::iter::once("unpack".to_string()).chain(stage_list.iter().map(|s| s.name())).chain(std::iter::once("pack".to_string())).collect();

    std::thread::scope(|scope| {
        let shared = &shared;
        let frames_in = &frames_in;
        let frames_out = &frames_out;
        // Raw-buffer recycling between reader and unpacker, and packer and writer.
        let (raw_tx, raw_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = bounded(3);
        let (raw_back_tx, raw_back_rx) = bounded::<Vec<u8>>(4);
        let (packed_tx, packed_rx): (Sender<Vec<u8>>, Receiver<Vec<u8>>) = bounded(3);
        let (packed_back_tx, packed_back_rx) = bounded::<Vec<u8>>(4);

        // Reader: decoder stdout → raw buffers.
        scope.spawn(move || {
            let r = (|| -> Result<()> {
                loop {
                    if shared.failed.load(Ordering::SeqCst) {
                        return Ok(());
                    }
                    let mut buf = raw_back_rx.try_recv().unwrap_or_else(|_| vec![0u8; in_bytes]);
                    buf.resize(in_bytes, 0);
                    let mut filled = 0;
                    while filled < in_bytes {
                        match dec_out.read(&mut buf[filled..]) {
                            Ok(0) => break,
                            Ok(n) => filled += n,
                            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                            Err(e) => return Err(e.into()),
                        }
                    }
                    if filled == 0 {
                        return Ok(());
                    }
                    if filled < in_bytes {
                        bail!("decoder produced a partial frame ({filled} of {in_bytes} bytes)");
                    }
                    if raw_tx.send(buf).is_err() {
                        return Ok(());
                    }
                }
            })();
            if let Err(e) = r {
                shared.fail(e);
            }
        });

        // Stage chain channels: unpack → stage 0 → ... → stage n-1 → pack.
        let mut senders = Vec::new();
        let mut receivers = Vec::new();
        for _ in 0..=n_stages {
            let (tx, rx) = bounded::<Frame>(2);
            senders.push(tx);
            receivers.push(rx);
        }
        let mut senders = senders.into_iter();
        let mut receivers = receivers.into_iter();

        // Unpack: raw → frames.
        let first_tx = senders.next().unwrap();
        let busy = &busy;
        scope.spawn(move || {
            for buf in raw_rx.iter() {
                if shared.failed.load(Ordering::SeqCst) || control.checkpoint().is_err() {
                    break;
                }
                let t = Instant::now();
                let mut f = Frame::uninit(in_fmt.width as usize, in_fmt.height as usize);
                raw_to_frame(&buf, &in_fmt, &mut f);
                busy[0].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                let _ = raw_back_tx.try_send(buf);
                frames_in.fetch_add(1, Ordering::Relaxed);
                if first_tx.send(f).is_err() {
                    break;
                }
            }
        });

        // Stages. Busy time excludes time spent waiting for downstream to accept frames.
        for (i, stage) in stage_list.iter_mut().enumerate() {
            let rx = receivers.next().unwrap();
            let tx = senders.next().unwrap();
            let busy_slot = &busy[i + 1];
            scope.spawn(move || {
                let r = (|| -> Result<()> {
                    let blocked = std::cell::Cell::new(0u64);
                    let mut emit = |f: Frame| -> Result<()> {
                        let t = Instant::now();
                        let r = tx.send(f).map_err(|_| anyhow!("downstream stopped"));
                        blocked.set(blocked.get() + t.elapsed().as_nanos() as u64);
                        r
                    };
                    for f in rx.iter() {
                        control.checkpoint()?;
                        if shared.failed.load(Ordering::SeqCst) {
                            return Ok(());
                        }
                        let t = Instant::now();
                        blocked.set(0);
                        stage.push(f, &mut emit)?;
                        busy_slot.fetch_add((t.elapsed().as_nanos() as u64).saturating_sub(blocked.get()), Ordering::Relaxed);
                    }
                    if !shared.failed.load(Ordering::SeqCst) {
                        let t = Instant::now();
                        blocked.set(0);
                        stage.finish(&mut emit)?;
                        busy_slot.fetch_add((t.elapsed().as_nanos() as u64).saturating_sub(blocked.get()), Ordering::Relaxed);
                    }
                    Ok(())
                })();
                if let Err(e) = r {
                    if !shared.failed.load(Ordering::SeqCst) {
                        shared.fail(e.context(format!("stage '{}'", stage.name())));
                    }
                }
            });
        }

        // Pack: frames → raw (or collect).
        let last_rx = receivers.next().unwrap();
        let collect = match &sink {
            Sink::Collect(v) => Some(v.clone()),
            Sink::Encode => None,
        };
        let finished = &finished;
        let encoding = enc_in.is_some();
        scope.spawn(move || {
            for f in last_rx.iter() {
                if shared.failed.load(Ordering::SeqCst) {
                    break;
                }
                if let Some(v) = &collect {
                    v.lock().unwrap().push(f);
                    frames_out.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let t = Instant::now();
                let mut buf = packed_back_rx.try_recv().unwrap_or_default();
                frame_to_raw(&f, &out_fmt, &mut buf);
                drop(f);
                busy[n_stages + 1].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
                if packed_tx.send(buf).is_err() {
                    break;
                }
            }
            drop(packed_tx);
            if !encoding {
                finished.store(true, Ordering::SeqCst);
            }
        });

        // Writer: raw → encoder stdin. Closing stdin lets the encoder finish the file.
        if let Some(mut stdin) = enc_in.take() {
            scope.spawn(move || {
                let r = (|| -> Result<()> {
                    for buf in packed_rx.iter() {
                        stdin.write_all(&buf).context("writing to the encoder")?;
                        frames_out.fetch_add(1, Ordering::Relaxed);
                        let _ = packed_back_tx.try_send(buf);
                    }
                    stdin.flush().ok();
                    Ok(())
                })();
                drop(stdin);
                if let Err(e) = r {
                    shared.fail(e);
                }
                finished.store(true, Ordering::SeqCst);
            });
        } else {
            drop(packed_rx);
        }

        // Monitor: progress, cancellation, failures. On failure the processes are killed, which
        // unblocks every thread waiting on a pipe; channel ends drop as threads exit.
        let mut last = u64::MAX;
        loop {
            std::thread::sleep(Duration::from_millis(50));
            let n = frames_out.load(Ordering::Relaxed);
            if n != last {
                progress(n);
                last = n;
            }
            if control.is_cancelled() && !shared.failed.load(Ordering::SeqCst) {
                shared.fail(Cancelled.into());
            }
            if shared.failed.load(Ordering::SeqCst) {
                dec_proc.kill();
                if let Some(e) = &enc_proc {
                    e.kill();
                }
                if !control.is_cancelled() {
                    // Release threads parked by a pause so they can observe the failure.
                    control.resume();
                }
                break;
            }
            if finished.load(Ordering::SeqCst) {
                break;
            }
        }
    });

    let failed = shared.failed.load(Ordering::SeqCst);
    let error = shared.error.lock().unwrap().take();
    let (dec_status, dec_tail) = dec_proc.wait()?;
    let enc = match enc_proc {
        Some(p) => Some(p.wait()?),
        None => None,
    };
    if control.is_cancelled() {
        return Err(Cancelled.into());
    }
    if failed {
        let e = error.unwrap_or_else(|| anyhow!("engine failed"));
        // A broken pipe usually means FFmpeg died; its own output explains why.
        if let Some((s, tail)) = &enc {
            if !s.success() {
                return Err(process_error("Encoding", Some(*s), tail).context(format!("{e:#}")));
            }
        }
        if !dec_status.success() && !dec_tail.is_empty() {
            return Err(process_error("Decoding", Some(dec_status), &dec_tail).context(format!("{e:#}")));
        }
        return Err(e);
    }
    if !dec_status.success() {
        return Err(process_error("Decoding", Some(dec_status), &dec_tail));
    }
    if let Some((s, tail)) = enc {
        if !s.success() {
            return Err(process_error("Encoding", Some(s), &tail));
        }
    }
    progress(frames_out.load(Ordering::Relaxed));
    let n_out = frames_out.load(Ordering::Relaxed).max(1) as f64;
    Ok(EngineReport {
        frames_in: frames_in.load(Ordering::Relaxed),
        frames_out: frames_out.load(Ordering::Relaxed),
        stages: descriptions,
        stats: stage_list.iter().flat_map(|s| s.stats()).collect(),
        timings: step_names.into_iter().zip(&busy).map(|(n, b)| (n, b.load(Ordering::Relaxed) as f64 / 1e6 / n_out)).collect(),
        elapsed: start.elapsed().as_secs_f64(),
    })
}
