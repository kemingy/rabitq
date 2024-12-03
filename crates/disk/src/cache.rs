use std::io::Write;
use std::path::Path;
use std::sync::{Arc, Mutex};

use aws_config::BehaviorVersion;
use aws_sdk_s3::Client;
use bytes::Buf;
use rabitq::metrics::METRICS;
use rabitq::utils::l2_squared_distance;
use rusqlite::{Connection, OptionalExtension};

const BLOCK_BYTE_LIMIT: u32 = 1 << 19; // 512KiB

fn parse_fvecs(bytes: &mut impl Buf) -> Vec<Vec<f32>> {
    let mut vecs = Vec::new();
    while bytes.has_remaining() {
        let dim = bytes.get_u32_le() as usize;
        vecs.push((0..dim).map(|_| bytes.get_f32_le()).collect());
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
    num_per_block: u32,
    total_num: u32,
    total_block: u32,
    s3_bucket: Arc<String>,
    s3_key: Arc<String>,
    s3_client: Arc<Client>,
    sqlite_conn: Mutex<Connection>,
}

impl CachedVector {
    /// init the cached vector.
    pub async fn new(
        dim: u32,
        num: u32,
        local_path: String,
        s3_bucket: String,
        s3_prefix: String,
    ) -> Self {
        let s3_config = aws_config::defaults(BehaviorVersion::v2024_03_28())
            .load()
            .await;
        let s3_client = Arc::new(Client::new(&s3_config));
        let num_per_block = BLOCK_BYTE_LIMIT / (4 * (dim + 1));
        let total_num = num;
        let total_block = total_num.div_ceil(num_per_block);
        let sqlite_conn = Connection::open(Path::new(&local_path)).expect("failed to open sqlite");
        sqlite_conn
            .execute(
                "CREATE TABLE IF NOT EXISTS matrix (
                id    INTEGER PRIMARY KEY,
                vec   BLOB
            )",
                (),
            )
            .expect("failed to create table");
        Self {
            dim,
            num_per_block,
            total_num,
            total_block,
            s3_bucket: Arc::new(s3_bucket),
            s3_key: Arc::new(format!("{}/base.fvecs", s3_prefix)),
            s3_client,
            sqlite_conn: Mutex::new(sqlite_conn),
        }
    }

    fn block_range_bytes(&self, block: usize) -> (usize, usize) {
        let start = 4 * block * (self.dim as usize + 1) * self.num_per_block as usize;
        let end = if block == self.total_block as usize - 1 {
            4 * (self.dim as usize + 1) * self.total_num as usize
        } else {
            4 * (block + 1) * (self.dim as usize + 1) * self.num_per_block as usize
        };
        (start, end - 1)
    }

    async fn fetch_from_s3(&self, index: usize, query: &[f32]) -> anyhow::Result<f32> {
        let block = index / self.num_per_block as usize;
        let (start, end) = self.block_range_bytes(block);
        let object = self
            .s3_client
            .get_object()
            .bucket(self.s3_bucket.as_ref())
            .key(self.s3_key.as_ref())
            .range(format!("bytes={}-{}", start, end))
            .send()
            .await?;
        let mut bytes = object.body.collect().await?;
        let vecs = parse_fvecs(&mut bytes);
        METRICS.add_cache_miss_count(1);
        let offset_id = index % self.num_per_block as usize;
        let start_id = index - offset_id;

        {
            let conn = self.sqlite_conn.lock().unwrap();
            let mut statement = conn.prepare(
                "INSERT INTO matrix (id, vec) VALUES (?1, ?2) ON CONFLICT(id) DO NOTHING",
            )?;
            for (i, vec) in vecs.iter().enumerate() {
                statement.execute((start_id + i, bytemuck::cast_slice(vec)))?;
            }
        }

        let distance = l2_squared_distance(&vecs[offset_id], query);

        Ok(distance)
    }

    /// Get the vector l2 square distance.
    pub async fn get_query_vec_distance(&self, query: &[f32], index: u32) -> anyhow::Result<f32> {
        {
            let conn = self.sqlite_conn.lock().unwrap();
            let mut statement = conn.prepare("SELECT vec FROM matrix WHERE id = ?1")?;
            if let Some(raw) = statement
                .query_row([index], |res| res.get::<_, Vec<u8>>(0))
                .optional()?
            {
                let res = l2_squared_distance(bytemuck::cast_slice(&raw), query);
                return Ok(res);
            }
        }
        self.fetch_from_s3(index as usize, query).await
    }
}
