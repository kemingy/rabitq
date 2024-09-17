//! Cache layer for the vectors.
//!
//! 1. memory
//! 2. local file
//! 3. S3

use std::io::Write;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use bytes::Buf;
// use foyer::{DirectFsDeviceOptionsBuilder, HybridCache, HybridCacheBuilder};
use faer::{Col, ColRef};
use quick_cache::sync::Cache;

use crate::consts::BLOCK_BYTE_LIMIT;
use crate::simd::l2_squared_distance;

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
    num: u32,
    block_num: u32,
    num_per_block: u32,
    s3_bucket: String,
    s3_key: String,
    s3_client: Client,
    // cache: HybridCache<u32, DVector<f32>>,
    cache: Arc<Cache<u32, Arc<Col<f32>>>>,
}

impl CachedVector {
    /// init the cached vector.
    pub async fn new(
        dim: u32,
        num: u32,
        _local_path: String,
        s3_bucket: String,
        s3_prefix: String,
        mem_cache_num: u32,
        _disk_cache_mb: u32,
    ) -> Self {
        let num_per_block = BLOCK_BYTE_LIMIT / 4 / (dim + 1);
        let block_num = (num as f32 / num_per_block as f32).ceil() as u32;
        let s3_config = aws_config::defaults(BehaviorVersion::v2024_03_28())
            .load()
            .await;
        let s3_client = Client::new(&s3_config);
        Self {
            dim,
            num,
            block_num,
            num_per_block,
            s3_bucket,
            s3_key: format!("{}/base.fvecs", s3_prefix),
            // cache: HybridCacheBuilder::new()
            //     .memory(mem_cache_num as usize * 1024 * 1024)
            //     .storage()
            //     .with_device_config(
            //         DirectFsDeviceOptionsBuilder::new(local_path.clone())
            //             .with_capacity(disk_cache_mb as usize * 1024 * 1024)
            //             .build(),
            //     )
            //     .build()
            //     .await
            //     .expect("failed to create cache"),
            cache: Arc::new(Cache::new(mem_cache_num as usize)),
            s3_client,
        }
    }

    fn block_range_bytes(&self, index: usize) -> (usize, usize) {
        let block = index / self.num_per_block as usize;
        let start = 4 * block * (self.dim as usize + 1) * self.num_per_block as usize;
        let end = if block == self.block_num as usize - 1 {
            4 * (self.dim as usize + 1) * self.num as usize
        } else {
            4 * (block + 1) * (self.dim as usize + 1) * self.num_per_block as usize
        };
        (start, end - 1)
    }

    async fn fetch_from_s3(&self, index: usize) -> anyhow::Result<()> {
        let (start, end) = self.block_range_bytes(index);
        let object = self
            .s3_client
            .get_object()
            .bucket(&self.s3_bucket)
            .key(&self.s3_key)
            .range(format!("bytes={}-{}", start, end))
            .send()
            .await?;
        let mut bytes = object.body.collect().await?;
        let vecs = parse_fvecs(&mut bytes);
        let block_offset = index as u32 / self.num_per_block;
        for (i, vec) in vecs.into_iter().enumerate() {
            self.cache
                .insert(block_offset * self.num_per_block + i as u32, Arc::new(vec));
        }

        Ok(())
    }

    /// Get the L2 squared distance between the vector at the given index and the query.
    pub async fn get_l2_squared_distance(
        &self,
        index: usize,
        query: &ColRef<'_, f32>,
    ) -> anyhow::Result<f32> {
        // if !self.cache.contains(&(index as u32)) {
        //     self.fetch_from_s3(index)
        //         .await
        //         .expect("failed to fetch from s3");
        // }
        // let entry = self
        //     .cache
        //     .get(&(index as u32))
        //     .await?
        //     .expect("entry is empty");

        if let Some(entry) = self.cache.get(&(index as u32)) {
            return Ok(unsafe { l2_squared_distance(&entry.as_ref().as_ref(), query) });
        }
        self.fetch_from_s3(index)
            .await
            .expect("failed to fetch from s3");

        if let Some(entry) = self.cache.get(&(index as u32)) {
            return Ok(unsafe { l2_squared_distance(&entry.as_ref().as_ref(), query) });
        }
        Err(anyhow::anyhow!("failed to get entry"))
    }

    /// Get the cache stats.
    pub fn get_cache_stats(&self) -> String {
        format!("hits: {}, misses: {}, length: {}", self.cache.hits(), self.cache.misses(), self.cache.len())
    }
}

/// Cached vector instance. Need to be initialized before using.
pub static CACHED_VECTOR: OnceLock<CachedVector> = OnceLock::new();
