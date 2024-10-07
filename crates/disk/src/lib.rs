//! Disk cached RaBitQ implementation
//!
//! 1. Metadata are loaded into memory.
//! 2. Raw vectors are fetched from S3 when needed.
//! 3. Raw vectors are cached on disk with sqlite.

pub mod cache;
pub mod disk;
