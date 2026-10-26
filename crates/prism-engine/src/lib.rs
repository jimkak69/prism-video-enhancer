//! Prism engine: the frame pipeline that replaces the reference product's proprietary FFmpeg
//! filters. Raw frames flow from an FFmpeg decoder through Rust stages (enhancement and
//! upscaling, frame interpolation, deblur, HDR, grain) into an FFmpeg encoder.

pub mod ai;
pub mod enhance;
pub mod ffmpeg;
pub mod frame;
pub mod interp;
pub mod ops;
pub mod pipeline;
pub mod process;
pub mod queue;
pub mod runner;
pub mod stages;

pub use ffmpeg::Tools;
pub use frame::Frame;
pub use process::Control;
pub use runner::{Event, RunReport, run_plan};

/// Sizes the global rayon pool. Call once at start-up; 0 uses every logical CPU.
pub fn init_threads(threads: usize) {
    let mut b = rayon::ThreadPoolBuilder::new().thread_name(|i| format!("prism-{i}"));
    if threads > 0 {
        b = b.num_threads(threads);
    }
    let _ = b.build_global();
}

#[cfg(test)]
mod kernel_bench;
