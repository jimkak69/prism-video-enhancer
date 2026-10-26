//! Neural-network support: weight conversion, tiling and ONNX Runtime inference.

pub mod convert;
pub mod fetch;
mod onnx_proto;
pub mod pickle;
pub mod tiling;

#[cfg(feature = "onnx")]
pub mod runtime;

#[cfg(not(feature = "onnx"))]
pub mod runtime {
    //! Stand-in used when the crate is built without the `onnx` feature.
    use std::path::{Path, PathBuf};

    use anyhow::{Result, bail};
    use prism_core::settings::DeviceSettings;

    use crate::frame::Frame;

    #[derive(Debug, Clone)]
    pub struct RuntimeInfo {
        pub library: PathBuf,
        pub providers: Vec<&'static str>,
    }

    pub fn runtime() -> Result<&'static RuntimeInfo> {
        bail!("this build of Prism has no ONNX support (built without the `onnx` feature)")
    }

    pub struct OnnxUpscaler {
        pub description: String,
    }

    impl OnnxUpscaler {
        pub fn new(_: &Path, _: usize, _: usize, _: &DeviceSettings, _: bool) -> Result<OnnxUpscaler> {
            runtime()?;
            unreachable!()
        }
        pub fn scale(&self) -> usize {
            1
        }
        pub fn provider(&self) -> &'static str {
            "none"
        }
        pub fn upscale(&mut self, _: &Frame) -> Result<Frame> {
            bail!("no ONNX support")
        }
    }

    pub struct OnnxInterpolator {
        pub description: String,
    }

    impl OnnxInterpolator {
        pub fn new(_: &Path, _: usize, _: usize, _: &DeviceSettings) -> Result<OnnxInterpolator> {
            runtime()?;
            unreachable!()
        }
        pub fn interpolate(&mut self, _: &Frame, _: &Frame, _: f32) -> Result<Frame> {
            bail!("no ONNX support")
        }
    }
}
