//! RaBitQ implementation in Rust.

#![forbid(missing_docs)]
mod consts;
pub mod metrics;
pub mod rabitq;
pub mod simd;
pub mod utils;

pub use rabitq::RaBitQ;
