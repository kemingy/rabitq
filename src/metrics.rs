//! Metrics module to provide the insights of RaBitQ query and training.
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::LazyLock;

/// Metrics struct.
#[derive(Debug)]
pub struct Metrics {
    /// rough count
    pub rough: AtomicU64,
    /// precise count
    pub precise: AtomicU64,
}

impl Metrics {
    /// init the metrics
    fn new() -> Self {
        Self {
            rough: AtomicU64::new(0),
            precise: AtomicU64::new(0),
        }
    }

    /// get the instance
    pub fn to_str(&self) -> String {
        format!(
            "rough: {}, precise: {}",
            self.rough.load(Ordering::Relaxed),
            self.precise.load(Ordering::Relaxed)
        )
    }

    /// add rough count
    pub fn add_rough_count(&self, count: u64) {
        self.rough.fetch_add(count, Ordering::Relaxed);
    }

    /// add precise count
    pub fn add_precise_count(&self, count: u64) {
        self.precise.fetch_add(count, Ordering::Relaxed);
    }
}

/// Metrics instance.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
