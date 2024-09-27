//! Disk-based RaBitQ implementation.

use core::f32;
use std::path::Path;

use faer::{ColRef, Mat};

use crate::cache::CachedVector;
use crate::consts::{SCALAR, THETA_LOG_DIM};
use crate::metrics::METRICS;
use crate::rabitq::Factor;
use crate::rerank::new_re_ranker;
use crate::utils::{
    asymmetric_binary_dot_product, gen_random_bias, l2_squared_distance, matrix_from_fvecs,
    min_max_residual, project, read_u64_vecs, read_vecs, scalar_quantize, vector_binarize_query,
};

/// RaBitQ struct.
#[derive(Debug)]
pub struct DiskRaBitQ {
    dim: u32,
    cache: CachedVector,
    orthogonal: Mat<f32>,
    centroids: Mat<f32>,
    rand_bias: Vec<f32>,
    offsets: Vec<u32>,
    map_ids: Vec<u32>,
    x_binary_vec: Vec<u64>,
    factors: Vec<Factor>,
}

impl DiskRaBitQ {
    /// Load from dir with cache.
    pub async fn load_from_dir(
        path: &Path,
        cache_path: String,
        s3_bucket: String,
        s3_prefix: String,
    ) -> Self {
        let orthogonal = matrix_from_fvecs(&path.join("orthogonal.fvecs"));
        let centroids = matrix_from_fvecs(&path.join("centroids.fvecs"));

        let offsets_ids =
            read_vecs::<u32>(&path.join("offsets_ids.ivecs")).expect("open offsets_ids error");
        let offsets = offsets_ids.first().expect("offsets is empty").clone();
        let map_ids = offsets_ids.last().expect("map_ids is empty").clone();

        let factors = read_vecs::<f32>(&path.join("factors.fvecs"))
            .expect("open factors error")
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>()
            .chunks_exact(4)
            .map(|f| f.to_vec().into())
            .collect();

        let x_binary_vec = read_u64_vecs(&path.join("x_binary_vec.u64vecs"))
            .expect("open x_binary_vec error")
            .into_iter()
            .flatten()
            .collect();

        let dim = orthogonal.nrows();

        Self {
            dim: dim as u32,
            cache: CachedVector::new(
                dim as u32,
                map_ids.len() as u32,
                cache_path,
                s3_bucket,
                s3_prefix,
            )
            .await,
            orthogonal,
            centroids,
            rand_bias: gen_random_bias(dim),
            offsets,
            map_ids,
            factors,
            x_binary_vec,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn calculate_rough_distance(
        &self,
        y_c_distance_square: f32,
        y_binary_vec: &[u64],
        lower_bound: f32,
        scalar_sum: f32,
        delta: f32,
        cluster_id: usize,
        rough_distances: &mut Vec<(f32, u32)>,
    ) {
        let dist_sqrt = y_c_distance_square.sqrt();
        let binary_offset = y_binary_vec.len() / THETA_LOG_DIM as usize;
        for j in self.offsets[cluster_id]..self.offsets[cluster_id + 1] {
            let ju = j as usize;
            let factor = &self.factors[ju];
            rough_distances.push((
                (factor.center_distance_square
                    + y_c_distance_square
                    + lower_bound * factor.factor_ppc
                    + (2.0
                        * asymmetric_binary_dot_product(
                            &self.x_binary_vec[ju * binary_offset..(ju + 1) * binary_offset],
                            y_binary_vec,
                        ) as f32
                        - scalar_sum)
                        * factor.factor_ip
                        * delta
                    - factor.error_bound * dist_sqrt),
                j,
            ));
        }
    }

    /// Query the topk nearest neighbors for the given query asynchronously.
    pub async fn query_async(
        &self,
        query: &ColRef<'_, f32>,
        probe: usize,
        topk: usize,
        heuristic_rank: bool,
    ) -> Vec<(f32, u32)> {
        let y_projected = project(query, &self.orthogonal.as_ref());
        let k = self.centroids.shape().1;
        let mut lists = Vec::with_capacity(k);
        let mut residual = vec![0f32; self.dim as usize];
        for (i, centroid) in self.centroids.col_iter().enumerate() {
            let dist = l2_squared_distance(&centroid, &y_projected.as_ref());
            lists.push((dist, i));
        }
        let length = probe.min(k);
        lists.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        lists.truncate(length);
        lists.sort_by(|a, b| a.0.total_cmp(&b.0));

        let mut re_ranker = new_re_ranker(query, topk, heuristic_rank);
        let mut rough_distances = Vec::new();
        let mut quantized = vec![0u8; self.dim as usize];
        let mut binary_vec = vec![0u64; self.dim as usize * THETA_LOG_DIM as usize / 64];
        for &(dist, i) in lists[..length].iter() {
            let (lower_bound, upper_bound) =
                min_max_residual(&mut residual, &y_projected.as_ref(), &self.centroids.col(i));
            let delta = (upper_bound - lower_bound) * SCALAR;
            let one_over_delta = delta.recip();
            let scalar_sum = scalar_quantize(
                &mut quantized,
                &residual,
                &self.rand_bias,
                lower_bound,
                one_over_delta,
            );
            binary_vec.iter_mut().for_each(|element| *element = 0);
            vector_binarize_query(&quantized, &mut binary_vec);
            self.calculate_rough_distance(
                dist,
                &binary_vec,
                lower_bound,
                scalar_sum as f32,
                delta,
                i,
                &mut rough_distances,
            );
            re_ranker
                .rank_batch_async(&rough_distances, &self.map_ids, i as u32, &self.cache)
                .await;
            rough_distances.clear();
        }

        METRICS.add_query_count(1);
        re_ranker.get_result()
    }
}
