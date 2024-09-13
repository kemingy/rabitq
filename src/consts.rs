//! Constants used in the program.

pub(crate) const DEFAULT_X_DOT_PRODUCT: f32 = 0.8;
pub(crate) const EPSILON: f32 = 1.9;
pub(crate) const THETA_LOG_DIM: u32 = 4;
pub(crate) const WINDOWS_SIZE: usize = 12;
pub(crate) const DELTA_SCALAR: f32 = (1 << THETA_LOG_DIM) as f32 - 1.0;
pub(crate) const BLOCK_BYTE_LIMIT: u32 = 1 << 19; // 512KiB
