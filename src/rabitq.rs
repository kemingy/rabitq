//! RaBitQ implementation.

use std::path::Path;

use log::debug;
use nalgebra::{DMatrix, DVector, DVectorView};
use num_traits::ToPrimitive;
use serde::{Deserialize, Serialize};

use crate::consts::{DEFAULT_X_DOT_PRODUCT, EPSILON, THETA_LOG_DIM, WINDOWS_SIZE};
use crate::metrics::METRICS;
use crate::utils::{gen_random_bias, gen_random_qr_orthogonal, matrix_from_fvecs};

/// Convert the vector to binary format and store in a u64 vector.
fn vector_binarize_u64(vec: &DVector<f32>) -> Vec<u64> {
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
fn vector_binarize_one(vec: &DVector<f32>) -> DVector<f32> {
    DVector::from_fn(vec.len(), |i, _| if vec[i] > 0.0 { 1.0 } else { -1.0 })
}

/// Interface of `vector_binarize_query`
fn vector_binarize_query(vec: &DVector<u8>) -> Vec<u64> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::vector_binarize_query_avx2(&vec.as_view()) }
        } else {
            vector_binarize_query_raw(vec)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        vector_binarize_query_raw(vec)
    }
}

/// Convert the vector to binary format (one value to multiple bits) and store in a u64 vector.
#[inline]
fn vector_binarize_query_raw(vec: &DVector<u8>) -> Vec<u64> {
    let length = vec.len();
    let mut binary = vec![0u64; length * THETA_LOG_DIM as usize / 64];
    for j in 0..THETA_LOG_DIM as usize {
        for i in 0..length {
            binary[(i + j * length) / 64] |= (((vec[i] >> j) & 1) as u64) << (i % 64);
        }
    }
    binary
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
            unsafe { crate::simd::l2_squared_distance_avx2(lhs, rhs) }
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

/// Find the nearest cluster for the given vector.
fn kmeans_nearest_cluster(centroids: &DMatrix<f32>, vec: &DVectorView<f32>) -> usize {
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

/// RaBitQ struct.
#[derive(Debug, Serialize, Deserialize)]
pub struct RaBitQ {
    dim: u32,
    base: DMatrix<f32>,
    orthogonal: DMatrix<f32>,
    rand_bias: DVector<f32>,
    centroids: DMatrix<f32>,
    offsets: Vec<u32>,
    map_ids: Vec<u32>,
    x_binary_vec: Vec<Vec<u64>>,
    x_c_distance_square: Vec<f32>,
    error_bound: Vec<f32>,
    factor_ip: Vec<f32>,
    factor_ppc: Vec<f32>,
}

impl RaBitQ {
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
            let min_label = kmeans_nearest_cluster(&centroids, &xp);
            labels[min_label].push(i as u32);
            let x_c_quantized = xp - centroids.column(min_label);
            x_c_distance[i] = x_c_quantized.norm();
            x_c_distance_square[i] = x_c_distance[i].powi(2);
            x_binary_vec.push(vector_binarize_u64(&x_c_quantized));
            x_signed_vec.push(vector_binarize_one(&x_c_quantized));
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
        let flat_labels: Vec<u32> = labels.into_iter().flatten().collect();
        let x_binary_vec = flat_labels
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

        Self {
            dim: dim as u32,
            base: base.transpose(),
            orthogonal,
            rand_bias,
            offsets,
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
    pub fn query_one(&self, query: &DVector<f32>, probe: usize, topk: usize) -> Vec<i32> {
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

        let mut rough_distances = Vec::new();
        for &(dist, i) in lists[..length].iter() {
            y_projected.sub_to(&self.centroids.column(i), &mut residual);
            let lower_bound = residual.min();
            let upper_bound = residual.max();
            let delta = (upper_bound - lower_bound) / ((1 << THETA_LOG_DIM) as f32 - 1.0);
            let one_over_delta = 1.0 / delta;
            let y_scaled = residual.add_scalar(-lower_bound) * one_over_delta + &self.rand_bias;
            let y_quantized = y_scaled.map(|v| v.to_u8().expect("convert to u8 error"));
            let scalar_sum = y_quantized.iter().fold(0u32, |acc, &v| acc + v as u32);
            let y_binary_vec = vector_binarize_query(&y_quantized);
            let dist_sqrt = dist.sqrt();
            for j in self.offsets[i]..self.offsets[i + 1] {
                let ju = j as usize;
                rough_distances.push((
                    (self.x_c_distance_square[ju]
                        + dist
                        + lower_bound * self.factor_ppc[ju]
                        + (2.0
                            * asymmetric_binary_dot_product(&self.x_binary_vec[ju], &y_binary_vec)
                                as f32
                            - scalar_sum as f32)
                            * self.factor_ip[ju]
                            * delta
                        - self.error_bound[ju] * dist_sqrt),
                    self.map_ids[ju],
                ));
            }
        }

        METRICS.add_rough_count(rough_distances.len() as u64);
        self.rerank(query, &rough_distances, topk)
    }

    /// Rerank the topk nearest neighbors.
    fn rerank(
        &self,
        query: &DVector<f32>,
        rough_distances: &[(f32, u32)],
        topk: usize,
    ) -> Vec<i32> {
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
                    res.push((accurate, u as i32));
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
        res[..length].iter().map(|(_, u)| *u).collect()
    }
}
