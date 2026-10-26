//! Prism core: domain model and job planning for Prism Video Enhancer.
//!
//! Nothing here touches video data. [`media`] holds parsed facts about an input, [`settings`] the
//! state of every filter, and [`plan`] turns both into FFmpeg commands and Rust engine stages.

pub mod encoders;
pub mod errors;
pub mod media;
pub mod models;
pub mod naming;
pub mod paths;
pub mod plan;
pub mod presets;
pub mod progress;
pub mod project;
pub mod settings;
pub mod timecode;

pub use media::{Rational, VideoInfo};
pub use plan::{JobPlan, Pass, PlanContext, PreviewRange, plan};
pub use settings::JobSettings;
