//! RaBitQ implementation in Rust.

#![forbid(missing_docs)]
mod consts;
pub mod metrics;
mod order;
pub mod rabitq;
pub mod simd;
pub mod utils;

pub use rabitq::RaBitQ;
