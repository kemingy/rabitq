//! RaBitQ implementation.

use core::f32;
use std::ops::Range;
use std::path::Path;

use log::debug;
use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView};
use serde::{Deserialize, Serialize};

use crate::consts::{DEFAULT_X_DOT_PRODUCT, EPSILON, THETA_LOG_DIM, WINDOWS_SIZE};
use crate::metrics::METRICS;
use crate::utils::{gen_random_bias, gen_random_qr_orthogonal, matrix_from_fvecs};

/// Convert the vector to binary format and store in a u64 vector.
fn vector_binarize_u64(vec: &DVectorView<f32>) -> Vec<u64> {
    let mut binary = vec![0u64; (vec.len() + 63) / 64];
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            binary[i / 64] |= 1 << (i % 64);
        }
    }
    binary
}

/// Convert the vector to +1/-1 format.
#[inline]
fn vector_binarize_one(vec: &DVectorView<f32>) -> DVector<f32> {
    DVector::from_fn(vec.len(), |i, _| if vec[i] > 0.0 { 1.0 } else { -1.0 })
}

/// Interface of `vector_binarize_query`
fn vector_binarize_query(vec: &DVectorView<u8>, binary: &mut [u64]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                crate::simd::vector_binarize_query(&vec.as_view(), binary);
            }
        } else {
            vector_binarize_query_raw(vec, binary);
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        vector_binarize_query_raw(vec, binary);
    }
}

/// Convert the vector to binary format (one value to multiple bits) and store in a u64 vector.
#[inline]
fn vector_binarize_query_raw(vec: &DVectorView<u8>, binary: &mut [u64]) {
    let length = vec.len();
    for j in 0..THETA_LOG_DIM as usize {
        for i in 0..length {
            binary[(i + j * length) / 64] |= (((vec[i] >> j) & 1) as u64) << (i % 64);
        }
    }
}

/// Calculate the dot product of two binary vectors.
#[inline]
fn binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    for i in 0..x.len() {
        res += (x[i] & y[i]).count_ones();
    }
    res
}

/// Calculate the dot product of two binary vectors with different lengths.
///
/// The length of `y` should be `x.len() * THETA_LOG_DIM`.
fn asymmetric_binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    let length = x.len();
    let mut y_slice = y;
    for i in 0..THETA_LOG_DIM as usize {
        res += binary_dot_product(x, y_slice) << i;
        y_slice = &y_slice[length..];
    }
    res
}

/// Calculate the L2 squared distance between two vectors.
fn l2_squared_distance(
    lhs: &DVectorView<f32>,
    rhs: &DVectorView<f32>,
    residual: &mut DVector<f32>,
) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::l2_squared_distance(lhs, rhs) }
        } else {
            lhs.sub_to(rhs, residual);
            residual.norm_squared()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        lhs.sub_to(rhs, residual);
        residual.norm_squared()
    }
}

// Get the min/max value of a vector.
fn min_max_raw(vec: &DVectorView<f32>) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for v in vec.iter() {
        if *v < min {
            min = *v;
        }
        if *v > max {
            max = *v;
        }
    }
    (min, max)
}

// Interface of `min_max_residual`
fn min_max_residual(
    res: &mut DVector<f32>,
    x: &DVectorView<f32>,
    y: &DVectorView<f32>,
) -> (f32, f32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { crate::simd::min_max_residual(res, x, y) }
        } else {
            x.sub_to(y, res);
            min_max_raw(&res.as_view())
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        x.sub_to(y, &mut res);
        min_max_raw(&res.as_view())
    }
}

// Quantize the query residual vector.
fn scalar_quantize_raw(
    quantized: &mut DVector<u8>,
    vec: &DVectorView<f32>,
    bias: &DVectorView<f32>,
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    let mut sum = 0u32;
    for i in 0..vec.len() {
        let q = ((vec[i] - lower_bound) * multiplier + bias[i]) as u8;
        quantized[i] = q;
        sum += q as u32;
    }
    sum
}

// Interface of `scalar_quantize`
fn scalar_quantize(
    quantized: &mut DVector<u8>,
    vec: &DVectorView<f32>,
    bias: &DVectorView<f32>,
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::scalar_quantize(quantized, vec, lower_bound, multiplier) }
        } else {
            scalar_quantize_raw(quantized, vec, bias, lower_bound, multiplier)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        scalar_quantize_raw(quantized, vec, bias, lower_bound, multiplier)
    }
}

/// Project the vector to the orthogonal matrix.
#[allow(dead_code)]
#[inline]
fn project(vec: &DVectorView<f32>, orthogonal: &DMatrixView<f32>) -> DVector<f32> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            DVector::from_fn(vec.len(), |i, _| unsafe {
                crate::simd::vector_dot_product(vec, &orthogonal.column(i).as_view())
            })
        } else {
            vec.tr_mul(orthogonal).transpose()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        vec.tr_mul(orthogonal).transpose()
    }
}

/// Find the nearest cluster for the given vector.
fn kmeans_nearest_cluster(centroids: &DMatrixView<f32>, vec: &DVectorView<f32>) -> usize {
    let mut min_dist = f32::MAX;
    let mut min_label = 0;
    let mut residual = DVector::<f32>::zeros(vec.len());
    for (j, centroid) in centroids.column_iter().enumerate() {
        let dist = l2_squared_distance(&centroid, vec, &mut residual);
        if dist < min_dist {
            min_dist = dist;
            min_label = j;
        }
    }
    min_label
}

/// Pack the [u64] binary vectors to [u8] and shuffle the low-4 bits and high-4 bits.
fn pack_code_from_binary_vec(dim: usize, binary_vec: &[Vec<u64>], pack_code: &mut [u8]) {
    let aligned_num = (binary_vec.len() + 31) / 32 * 32;
    let flatten_binary_u64 = binary_vec
        .iter()
        .flat_map(|x| x.iter().copied())
        .collect::<Vec<u64>>();
    let binary_vec_u8: &[u8] = bytemuck::cast_slice(&flatten_binary_u64);
    let block_num = 32;
    let quarter = dim / 4;
    let eighth = dim / 8;
    const PERMUTATION: [usize; 16] = [0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15];
    let mut low_bits = [0u8; 32];
    let mut high_bits = [0u8; 32];
    let mut pack_code_slice = pack_code;

    for block in (0..aligned_num).step_by(block_num) {
        for m in (0..quarter).step_by(2) {
            for i in 0..32 {
                let val = binary_vec_u8[(block + i) * eighth + (m >> 1)];
                low_bits[i] = val & 0x0F;
                high_bits[i] = val >> 4;
            }
            for j in 0..16 {
                pack_code_slice[j] =
                    low_bits[PERMUTATION[j]] | (low_bits[PERMUTATION[j] + 16] << 4);
                pack_code_slice[j + 16] =
                    high_bits[PERMUTATION[j]] | (high_bits[PERMUTATION[j] + 16] << 4);
            }
            pack_code_slice = &mut pack_code_slice[32..];
        }
    }
}

/// Build the lookup table for fast scan
fn pack_lookup_table(quantized: &DVectorView<u8>, lookup_table: &mut [u8]) {
    const MASK: [usize; 16] = [3, 3, 2, 3, 1, 3, 2, 3, 0, 3, 2, 3, 1, 3, 2, 3];
    let mut lookup_table_slice = lookup_table;
    for j in 0..(quantized.len() / 4) {
        lookup_table_slice[0] = 0;
        for i in 0..16 {
            lookup_table_slice[i] =
                lookup_table_slice[i - (i & (!i + 1))] + quantized[4 * j + MASK[i]];
        }
        lookup_table_slice = &mut lookup_table_slice[16..];
    }
}

/// RaBitQ struct.
#[derive(Debug, Serialize, Deserialize)]
pub struct RaBitQ {
    dim: u32,
    base: DMatrix<f32>,
    orthogonal: DMatrix<f32>,
    rand_bias: DVector<f32>,
    centroids: DMatrix<f32>,
    offsets: Vec<u32>,
    pack_offsets: Vec<u32>,
    pack_code: Vec<u8>,
    map_ids: Vec<u32>,
    x_binary_vec: Vec<Vec<u64>>,
    x_c_distance_square: Vec<f32>,
    error_bound: Vec<f32>,
    factor_ip: Vec<f32>,
    factor_ppc: Vec<f32>,
}

impl RaBitQ {
    /// Load from a JSON file.
    pub fn load_from_json(path: &impl AsRef<Path>) -> Self {
        serde_json::from_slice(&std::fs::read(path).expect("open json error"))
            .expect("deserialize error")
    }

    /// Dump to a JSON file.
    pub fn dump_to_json(&self, path: &impl AsRef<Path>) {
        std::fs::write(path, serde_json::to_string(&self).expect("serialize error"))
            .expect("write json error");
    }

    /// Build the RaBitQ model from the base and centroids files.
    pub fn from_path(base_path: &Path, centroid_path: &Path) -> Self {
        let base = matrix_from_fvecs(base_path);
        let (n, dim) = base.shape();
        let centroids = matrix_from_fvecs(centroid_path);
        let k = centroids.shape().0;
        debug!("n: {}, dim: {}, k: {}", n, dim, k);
        let orthogonal = gen_random_qr_orthogonal(dim);
        let rand_bias = gen_random_bias(dim);

        // projection
        debug!("projection x & c...");
        let x_projected = (&base * &orthogonal).transpose();
        let centroids = (centroids * &orthogonal).transpose();

        // kmeans
        let dim_sqrt = (dim as f32).sqrt();
        let mut labels = vec![Vec::new(); k];
        let mut x_c_distance_square = vec![0.0; n];
        let mut x_c_distance = vec![0.0; n];
        let mut x_binary_vec: Vec<Vec<u64>> = Vec::with_capacity(n);
        let mut x_signed_vec: Vec<DVector<f32>> = Vec::with_capacity(n);
        let mut x_dot_product = vec![0.0; n];
        for (i, xp) in x_projected.column_iter().enumerate() {
            if i % 5000 == 0 {
                debug!("\t> preprocessing {}...", i);
            }
            let min_label = kmeans_nearest_cluster(&centroids.as_view(), &xp);
            labels[min_label].push(i as u32);
            let x_c_quantized = xp - centroids.column(min_label);
            x_c_distance[i] = x_c_quantized.norm();
            x_c_distance_square[i] = x_c_distance[i].powi(2);
            x_binary_vec.push(vector_binarize_u64(&x_c_quantized.as_view()));
            x_signed_vec.push(vector_binarize_one(&x_c_quantized.as_view()));
            let norm = x_c_distance[i] * dim_sqrt;
            x_dot_product[i] = if norm.is_normal() {
                x_c_quantized.dot(&x_signed_vec[i]) / norm
            } else {
                DEFAULT_X_DOT_PRODUCT
            };
        }

        // factors
        debug!("computing factors...");
        let mut error_bound = Vec::with_capacity(n);
        let mut factor_ip = Vec::with_capacity(n);
        let mut factor_ppc = Vec::with_capacity(n);
        let error_base = 2.0 * EPSILON / (dim as f32 - 1.0).sqrt();
        let one_vec = DVector::from_element(dim, 1.0);
        for i in 0..n {
            let x_c_over_ip = x_c_distance[i] / x_dot_product[i];
            error_bound
                .push(error_base * (x_c_over_ip * x_c_over_ip - x_c_distance_square[i]).sqrt());
            factor_ip.push(-2.0 / dim_sqrt * x_c_over_ip);
            factor_ppc.push(factor_ip[i] * one_vec.dot(&x_signed_vec[i]));
        }

        // sort by labels
        debug!("sort by labels...");
        let mut offsets = vec![0; k + 1];
        for i in 0..k {
            offsets[i + 1] = offsets[i] + labels[i].len() as u32;
        }
        let mut pack_offsets = vec![0; k + 1];
        for i in 0..k {
            pack_offsets[i + 1] =
                pack_offsets[i] + (labels[i].len() + 31) as u32 / 32 * 32 * dim as u32 / 8;
        }
        let flat_labels: Vec<u32> = labels.into_iter().flatten().collect();
        let x_binary_vec: Vec<Vec<u64>> = flat_labels
            .iter()
            .map(|i| x_binary_vec[*i as usize].clone())
            .collect();
        let x_c_distance_square = flat_labels
            .iter()
            .map(|i| x_c_distance_square[*i as usize])
            .collect();
        let error_bound = flat_labels
            .iter()
            .map(|i| error_bound[*i as usize])
            .collect();
        let factor_ip = flat_labels.iter().map(|i| factor_ip[*i as usize]).collect();
        let factor_ppc = flat_labels
            .iter()
            .map(|i| factor_ppc[*i as usize])
            .collect();

        // fast scan
        debug!("build fast scan pack code...");
        let mut pack_code: Vec<u8> = vec![0; pack_offsets[k as usize] as usize];
        for i in 0..k {
            pack_code_from_binary_vec(
                dim,
                &x_binary_vec[offsets[i] as usize..offsets[i + 1] as usize],
                &mut pack_code[(pack_offsets[i] as usize)..(pack_offsets[i + 1] as usize)],
            );
        }

        Self {
            dim: dim as u32,
            base: base.transpose(),
            orthogonal,
            rand_bias,
            offsets,
            pack_offsets,
            pack_code,
            map_ids: flat_labels,
            centroids,
            x_binary_vec,
            x_c_distance_square,
            error_bound,
            factor_ip,
            factor_ppc,
        }
    }

    /// Query the topk nearest neighbors for the given query.
    pub fn query_one(&self, query: &DVector<f32>, probe: usize, topk: usize) -> Vec<(f32, u32)> {
        let y_projected = query.tr_mul(&self.orthogonal).transpose();
        let k = self.centroids.shape().1;
        let mut lists = Vec::with_capacity(k);
        let mut residual = DVector::<f32>::zeros(self.dim as usize);
        for (i, centroid) in self.centroids.column_iter().enumerate() {
            let dist = l2_squared_distance(&centroid, &y_projected.as_view(), &mut residual);
            lists.push((dist, i));
        }
        let length = probe.min(k);
        lists.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        lists.truncate(length);
        lists.sort_by(|a, b| a.0.total_cmp(&b.0));

        let fast_scan = true;
        let mut rough_distances = Vec::new();
        let mut quantized = DVector::<u8>::zeros(self.dim as usize);
        let mut binary_vec = vec![0u64; query.len() * THETA_LOG_DIM as usize / 64];
        let mut lookup_table = vec![0u8; self.dim as usize / 4 * 16];
        for &(dist, i) in lists[..length].iter() {
            let (lower_bound, upper_bound) = min_max_residual(
                &mut residual,
                &y_projected.as_view(),
                &self.centroids.column(i),
            );
            let delta = (upper_bound - lower_bound) / ((1 << THETA_LOG_DIM) as f32 - 1.0);
            let one_over_delta = 1.0 / delta;
            let scalar_sum = scalar_quantize(
                &mut quantized,
                &residual.as_view(),
                &self.rand_bias.as_view(),
                lower_bound,
                one_over_delta,
            );
            let dist_sqrt = dist.sqrt();

            binary_vec.iter_mut().for_each(|element| *element = 0);
            vector_binarize_query(&quantized.as_view(), &mut binary_vec);
            lookup_table.iter_mut().for_each(|element| *element = 0);
            pack_lookup_table(&quantized.as_view(), &mut lookup_table);

            if fast_scan {
                self.calculate_rough_distances_fast_scan(
                    &mut rough_distances,
                    &lookup_table,
                    dist,
                    dist_sqrt,
                    scalar_sum as f32,
                    lower_bound,
                    delta,
                    self.offsets[i]..self.offsets[i + 1],
                );
            } else {
                self.calculate_rough_distances(
                    &mut rough_distances,
                    &binary_vec,
                    dist,
                    dist_sqrt,
                    scalar_sum as f32,
                    lower_bound,
                    delta,
                    self.offsets[i]..self.offsets[i + 1],
                );
            }
        }

        METRICS.add_query_count(1);
        METRICS.add_rough_count(rough_distances.len() as u64);
        self.rerank(query, &rough_distances, topk)
    }

    /// Calculate the rough distances.
    #[allow(clippy::too_many_arguments)]
    fn calculate_rough_distances(
        &self,
        rough_distances: &mut Vec<(f32, u32)>,
        y_binary_vec: &[u64],
        y_c_distance_square: f32,
        y_c_distance: f32,
        scalar_sum: f32,
        lower_bound: f32,
        delta: f32,
        indexes: Range<u32>,
    ) {
        for i in indexes {
            let iu = i as usize;
            rough_distances.push((
                (self.x_c_distance_square[iu]
                    + y_c_distance_square
                    + lower_bound * self.factor_ppc[iu]
                    + (2.0
                        * asymmetric_binary_dot_product(&self.x_binary_vec[iu], y_binary_vec)
                            as f32
                        - scalar_sum)
                        * self.factor_ip[iu]
                        * delta
                    - self.error_bound[iu] * y_c_distance),
                self.map_ids[iu],
            ));
        }
    }

    /// Calculate the rough distances with fast scan.
    #[allow(clippy::too_many_arguments)]
    fn calculate_rough_distances_fast_scan(
        &self,
        rough_distances: &mut Vec<(f32, u32)>,
        lookup_table: &[u8],
        y_c_distance_square: f32,
        y_c_distance: f32,
        scalar_sum: f32,
        lower_bound: f32,
        delta: f32,
        indexes: Range<u32>,
    ) {
        let batch_size = 32;
        let batch_num = (indexes.end - indexes.start) / batch_size;
        let remaining = indexes.end - indexes.start - batch_num * batch_size;
        let mut results = vec![0u16; batch_size as usize];

        for i in indexes.clone().step_by(batch_size as usize) {
            unsafe {
                crate::simd::accumulate_one_block(
                    &self.pack_code[self.pack_offsets[i as usize] as usize
                        ..self.pack_offsets[(i + batch_size) as usize] as usize],
                    lookup_table,
                    &mut results,
                );
            }
            for (j, &res) in results.iter().enumerate() {
                let iu = i as usize + j;
                rough_distances.push((
                    (self.x_c_distance_square[iu]
                        + y_c_distance_square
                        + lower_bound * self.factor_ppc[iu]
                        + (res as f32 - scalar_sum) * self.factor_ip[iu] * delta
                        - y_c_distance * self.error_bound[iu]),
                    self.map_ids[iu],
                ));
            }
        }

        if remaining > 0 {
            let i = (indexes.end - remaining) as usize;
            unsafe {
                crate::simd::accumulate_one_block(
                    &self.pack_code[self.pack_offsets[i] as usize
                        ..self.pack_offsets[i + batch_size as usize] as usize],
                    lookup_table,
                    &mut results,
                );
            }
            for (j, &res) in results.iter().enumerate().take(remaining as usize) {
                let iu = i + j;
                rough_distances.push((
                    (self.x_c_distance_square[iu]
                        + y_c_distance_square
                        + lower_bound * self.factor_ppc[iu]
                        + (res as f32 - scalar_sum) * self.factor_ip[iu] * delta
                        - y_c_distance * self.error_bound[iu]),
                    self.map_ids[iu],
                ));
            }
        }
    }

    /// Rerank the topk nearest neighbors.
    fn rerank(
        &self,
        query: &DVector<f32>,
        rough_distances: &[(f32, u32)],
        topk: usize,
    ) -> Vec<(f32, u32)> {
        let mut threshold = f32::MAX;
        let mut recent_max_accurate = f32::MIN;
        let mut res = Vec::with_capacity(topk);
        let mut count = 0;
        let mut residual = DVector::<f32>::zeros(self.dim as usize);
        for &(rough, u) in rough_distances.iter() {
            if rough < threshold {
                let accurate = l2_squared_distance(
                    &self.base.column(u as usize),
                    &query.as_view(),
                    &mut residual,
                );
                if accurate < threshold {
                    res.push((accurate, u));
                    count += 1;
                    recent_max_accurate = recent_max_accurate.max(accurate);
                    if count == WINDOWS_SIZE {
                        threshold = recent_max_accurate;
                        count = 0;
                        recent_max_accurate = f32::MIN;
                    }
                }
            }
        }

        METRICS.add_precise_count(res.len() as u64);
        let length = topk.min(res.len());
        res.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        res.truncate(length);
        res
    }
}
