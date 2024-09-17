//! RaBitQ implementation in Rust.

#![forbid(missing_docs)]
pub mod cache;
mod consts;
pub mod metrics;
mod ord32;
pub mod rabitq;
mod rerank;
pub mod simd;
pub mod utils;

pub use rabitq::RaBitQ;
