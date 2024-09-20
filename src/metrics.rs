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
    /// query count
    pub query: AtomicU64,
    /// s3 fetch
    pub s3_fetch: AtomicU64,
}

impl Metrics {
    /// init the metrics
    fn new() -> Self {
        Self {
            rough: AtomicU64::new(0),
            precise: AtomicU64::new(0),
            query: AtomicU64::new(0),
            s3_fetch: AtomicU64::new(0),
        }
    }

    /// get the instance
    pub fn to_str(&self) -> String {
        let rough = self.rough.load(Ordering::Relaxed);
        let precise = self.precise.load(Ordering::Relaxed);
        let s3 = self.s3_fetch.load(Ordering::Relaxed);
        format!(
            "query: {}, rough: {}, precise: {}, ratio: {:.2}, s3 fetch: {}",
            self.query.load(Ordering::Relaxed),
            rough,
            precise,
            rough as f64 / precise as f64,
            s3,
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

    /// add query count
    pub fn add_query_count(&self, count: u64) {
        self.query.fetch_add(count, Ordering::Relaxed);
    }

    /// add s3 fetch count
    pub fn add_s3_fetch_count(&self, count: u64) {
        self.s3_fetch.fetch_add(count, Ordering::Relaxed);
    }
}

/// Metrics instance.
pub static METRICS: LazyLock<Metrics> = LazyLock::new(Metrics::new);
