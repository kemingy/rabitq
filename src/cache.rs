//! Cache layer for the vectors.
//!
//! 1. memory
//! 2. local file
//! 3. S3

use std::io::Write;
use std::path::Path;
// use std::sync::atomic::AtomicU32;
// use std::sync::{Arc, RwLock};
use std::sync::{Arc, OnceLock};

use anyhow::Ok;
use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use bytes::Buf;
use faer::{Col, ColRef, Mat, MatRef};
use foyer::{
    CacheEntry, DirectFsDeviceOptionsBuilder, HybridCache, HybridCacheBuilder, RateLimitPicker,
};

use crate::metrics::METRICS;

// use log::info;

// use quick_cache::sync::{Cache, DefaultLifecycle};
// use quick_cache::{DefaultHashBuilder, OptionsBuilder, UnitWeighter};

// #[derive(Debug)]
// struct CacheShard {
//     vec: Vec<Option<Arc<Col<f32>>>>,
// }

// impl CacheShard {
//     fn new(capacity: usize) -> Self {
//         Self {
//             vec: vec![None; capacity],
//         }
//     }

//     fn get(&self, index: usize) -> Option<Arc<Col<f32>>> {
//         self.vec[index].clone()
//     }

//     fn set(&mut self, index: usize, value: Arc<Col<f32>>) {
//         self.vec[index] = Some(value);
//     }
// }

// #[derive(Debug)]
// struct Cache {
//     limit: u32,
//     shard_num: u32,
//     counter: AtomicU32,
//     hits: AtomicU32,
//     misses: AtomicU32,
//     shards: Box<[RwLock<CacheShard>]>,
// }

// impl Cache {
//     fn new(capacity: usize, limit: u32, shard_num: u32) -> Self {
//         let shard_cap = capacity.saturating_add(shard_num as usize - 1) / shard_num as usize;
//         let shards = (0..shard_num)
//             .map(|_| RwLock::new(CacheShard::new(shard_cap)))
//             .collect::<Vec<_>>();
//         Self {
//             shards: shards.into_boxed_slice(),
//             counter: AtomicU32::new(0),
//             hits: AtomicU32::new(0),
//             misses: AtomicU32::new(0),
//             limit,
//             shard_num,
//         }
//     }

//     fn get(&self, key: &u32) -> Option<Arc<Col<f32>>> {
//         let shard = &self.shards[(key % self.shard_num) as usize]
//             .read()
//             .expect("failed to read shard");
//         let res = shard.get((key / self.shard_num) as usize);
//         if res.is_some() {
//             self.hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         } else {
//             self.misses
//                 .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         }
//         res
//     }

//     fn set(&self, key: u32, value: Arc<Col<f32>>) {
//         let num = self
//             .counter
//             .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
//         if num >= self.limit {
//             info!("cache reaching {} elements", num);
//         }
//         let shard = &mut self.shards[(key % self.shard_num) as usize]
//             .write()
//             .expect("failed to write shard");
//         shard.set((key / self.shard_num) as usize, value);
//     }

//     fn len(&self) -> u32 {
//         self.counter.load(std::sync::atomic::Ordering::Relaxed)
//     }

//     fn hits(&self) -> u32 {
//         self.hits.load(std::sync::atomic::Ordering::Relaxed)
//     }

//     fn misses(&self) -> u32 {
//         self.misses.load(std::sync::atomic::Ordering::Relaxed)
//     }
// }

fn parse_fvecs(bytes: &mut impl Buf) -> Vec<Col<f32>> {
    let mut vecs = Vec::new();
    while bytes.has_remaining() {
        let dim = bytes.get_u32_le() as usize;
        vecs.push(Col::from_fn(dim, |_| bytes.get_f32_le()));
    }
    vecs
}

/// Download rabitq meta data from S3.
pub async fn download_meta_from_s3(bucket: &str, prefix: &str, path: &Path) -> anyhow::Result<()> {
    let s3_config = aws_config::defaults(BehaviorVersion::v2024_03_28())
        .load()
        .await;
    let client = Client::new(&s3_config);
    for filename in [
        "centroids.fvecs",
        "orthogonal.fvecs",
        "factors.fvecs",
        "offsets_ids.ivecs",
        "x_binary_vec.u64vecs",
    ] {
        if path.join(filename).is_file() {
            continue;
        }
        let mut object = client
            .get_object()
            .bucket(bucket)
            .key(format!("{}/{}", prefix, filename))
            .send()
            .await?;
        let mut file = std::fs::File::create(path.join(filename))?;
        while let Some(chunk) = object.body.try_next().await? {
            file.write_all(chunk.as_ref())?;
        }
    }

    Ok(())
}

/// Cached vector.
#[derive(Debug)]
pub struct CachedVector {
    dim: u32,
    offsets: Vec<u32>,
    s3_bucket: Arc<String>,
    s3_prefix: Arc<String>,
    s3_client: Arc<Client>,
    cache: HybridCache<u32, Arc<Mat<f32>>>,
}

impl CachedVector {
    /// init the cached vector.
    pub async fn new(
        dim: u32,
        offsets: Vec<u32>,
        local_path: String,
        s3_bucket: String,
        s3_prefix: String,
        mem_cache_num: u32,
        disk_cache_mb: u32,
    ) -> Self {
        let s3_config = aws_config::defaults(BehaviorVersion::v2024_03_28())
            .load()
            .await;
        let s3_client = Arc::new(Client::new(&s3_config));
        Self {
            dim,
            offsets,
            s3_bucket: Arc::new(s3_bucket),
            s3_prefix: Arc::new(format!("{}/base.fvecs", s3_prefix)),
            cache: HybridCacheBuilder::new()
                .memory(mem_cache_num as usize)
                .storage()
                .with_device_config(
                    DirectFsDeviceOptionsBuilder::new(local_path.clone())
                        .with_capacity(disk_cache_mb as usize * 1024 * 1024)
                        .build(),
                )
                .with_admission_picker(Arc::new(RateLimitPicker::new(
                    disk_cache_mb as usize * 1024 * 1024,
                )))
                .with_flushers(8)
                .with_flush(true)
                .with_reclaimers(8)
                .with_clean_region_threshold(12)
                .build()
                .await
                .expect("failed to create cache"),
            s3_client,
        }
    }

    fn block_range_bytes(&self, index: usize) -> (usize, usize) {
        // let start = 4 * block * (self.dim as usize + 1) * self.num_per_block as usize;
        // let end = if block == self.block_num as usize - 1 {
        //     4 * (self.dim as usize + 1) * self.num as usize
        // } else {
        //     4 * (block + 1) * (self.dim as usize + 1) * self.num_per_block as usize
        // };
        // (start, end - 1)
        (
            (4 * self.offsets[index] * (self.dim + 1)) as usize,
            (4 * self.offsets[index + 1] * (self.dim + 1) - 1) as usize,
        )
    }

    async fn fetch_from_s3(&self, index: usize) -> anyhow::Result<()> {
        let (start, end) = self.block_range_bytes(index);
        let object = self
            .s3_client
            .get_object()
            .bucket(self.s3_bucket.as_ref())
            .key(self.s3_prefix.as_ref())
            .range(format!("bytes={}-{}", start, end))
            .send()
            .await?;
        let mut bytes = object.body.collect().await?;
        let vecs = parse_fvecs(&mut bytes);
        let mat = Mat::from_fn(vecs.len(), vecs[0].nrows(), |i, j| vecs[i][j])
            .transpose()
            .to_owned();
        self.cache.insert(index as u32, Arc::new(mat));
        Ok(())
    }

    /// Get the cache entry.
    pub async fn get_mat(&self, index: u32) -> anyhow::Result<CacheEntry<u32, Arc<Mat<f32>>>> {
        let entry = self
            .cache
            .fetch(index, || {
                let (start, end) = self.block_range_bytes(index as usize);
                let client = self.s3_client.clone();
                let s3_bucket = self.s3_bucket.clone();
                let s3_prefix = self.s3_prefix.clone();
                async move {
                    let object = client
                        .get_object()
                        .bucket(s3_bucket.as_ref())
                        .key(s3_prefix.as_ref())
                        .range(format!("bytes={}-{}", start, end))
                        .send()
                        .await?;
                    METRICS.add_s3_fetch_count(1);
                    let mut bytes = object.body.collect().await?;
                    let vecs = parse_fvecs(&mut bytes);
                    let mat = Mat::from_fn(vecs.len(), vecs[0].nrows(), |i, j| vecs[i][j])
                        .transpose()
                        .to_owned();
                    Ok(Arc::new(mat))
                }
            })
            .await?;
        Ok(entry)
    }

    /// Get the cache stats.
    pub fn get_cache_stats(&self) -> String {
        // format!(
        //     "hits: {}, misses: {}, length: {}",
        //     self.cache.hits(),
        //     self.cache.misses(),
        //     self.cache.len()
        // )
        let stats = self.cache.stats();
        format!(
            "read bytes: {}, write bytes: {}, flush: {}",
            stats.read_bytes.load(std::sync::atomic::Ordering::Relaxed),
            stats.write_bytes.load(std::sync::atomic::Ordering::Relaxed),
            stats.flush_ios.load(std::sync::atomic::Ordering::Relaxed),
        )
    }
}

/// Cached vector instance. Need to be initialized before using.
pub static CACHED_VECTOR: OnceLock<CachedVector> = OnceLock::new();
