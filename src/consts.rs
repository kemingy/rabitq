//! Constants used in the program.

/// Default value when the x dot product is divided by zero.
pub const DEFAULT_X_DOT_PRODUCT: f32 = 0.8;
/// $\epsilon$ in the paper.
pub const EPSILON: f32 = 1.9;
/// Asymmetric factor.
pub const THETA_LOG_DIM: u32 = 4;
/// Scalar value.
pub const SCALAR: f32 = 1.0 / ((1 << THETA_LOG_DIM) as f32 - 1.0);
/// Heuristic window size.
pub const WINDOW_SIZE: usize = 12;
