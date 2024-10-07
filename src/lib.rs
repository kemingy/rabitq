//! RaBitQ implementation in Rust.

#![forbid(missing_docs)]
pub mod consts;
pub mod metrics;
pub mod ord32;
pub mod rabitq;
pub mod rerank;
pub mod simd;
pub mod utils;

pub use rabitq::RaBitQ;
