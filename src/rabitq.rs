//! RaBitQ implementation.

use core::f32;
use std::collections::BinaryHeap;
use std::path::Path;

use faer::{Col, ColRef, Mat, MatRef, Row};
use log::debug;
// use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView};
use serde::{Deserialize, Serialize};

use crate::consts::{DEFAULT_X_DOT_PRODUCT, EPSILON, THETA_LOG_DIM, WINDOWS_SIZE};
use crate::metrics::METRICS;
use crate::order::Ord32;
use crate::utils::{
    gen_random_bias, gen_random_qr_orthogonal, matrix_from_fvecs, read_u64_vecs, read_vecs,
    write_matrix, write_vecs,
};

/// Convert the vector to binary format and store in a u64 vector.
fn vector_binarize_u64(vec: &ColRef<f32>) -> Vec<u64> {
    let mut binary = vec![0u64; (vec.nrows() + 63) / 64];
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            binary[i / 64] |= 1 << (i % 64);
        }
    }
    binary
}

/// Convert the vector to +1/-1 format.
#[inline]
fn vector_binarize_one(vec: &ColRef<f32>) -> Col<f32> {
    Col::from_fn(vec.nrows(), |i| if vec[i] > 0.0 { 1.0 } else { -1.0 })
}

/// Interface of `vector_binarize_query`
fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                crate::simd::vector_binarize_query(vec, binary);
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
fn vector_binarize_query_raw(vec: &[u8], binary: &mut [u64]) {
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
fn l2_squared_distance(lhs: &ColRef<f32>, rhs: &ColRef<f32>) -> f32 {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { crate::simd::l2_squared_distance(lhs, rhs) }
        } else {
            (lhs - rhs).squared_norm_l2()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        (lhs - rhs).squared_norm_l2()
    }
}

// Get the min/max value of a vector.
fn min_max_raw(res: &mut [f32], x: &ColRef<f32>, y: &ColRef<f32>) -> (f32, f32) {
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    for i in 0..res.len() {
        res[i] = x[i] - y[i];
        if res[i] < min {
            min = res[i];
        }
        if res[i] > max {
            max = res[i];
        }
    }
    (min, max)
}

// Interface of `min_max_residual`
fn min_max_residual(res: &mut [f32], x: &ColRef<f32>, y: &ColRef<f32>) -> (f32, f32) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx") {
            unsafe { crate::simd::min_max_residual(res, x, y) }
        } else {
            min_max_raw(res, x, y)
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        min_max_raw(res, x, y)
    }
}

// Quantize the query residual vector.
fn scalar_quantize_raw(
    quantized: &mut [u8],
    vec: &[f32],
    bias: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    let mut sum = 0u32;
    for i in 0..quantized.len() {
        let q = ((vec[i] - lower_bound) * multiplier + bias[i]) as u8;
        quantized[i] = q;
        sum += q as u32;
    }
    sum
}

// Interface of `scalar_quantize`
fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    bias: &[f32],
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
fn project(vec: &ColRef<f32>, orthogonal: &MatRef<f32>) -> Col<f32> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") {
            Col::from_fn(orthogonal.ncols(), |i| unsafe {
                crate::simd::vector_dot_product(vec, &orthogonal.col(i))
            })
        } else {
            (vec.transpose() * orthogonal).transpose().to_owned()
        }
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    {
        (vec.transpose() * orthogonal).transpose().to_owned()
    }
}

/// Find the nearest cluster for the given vector.
fn kmeans_nearest_cluster(centroids: &MatRef<f32>, vec: &ColRef<f32>) -> usize {
    let mut min_dist = f32::MAX;
    let mut min_label = 0;
    for (j, centroid) in centroids.col_iter().enumerate() {
        let dist = l2_squared_distance(&centroid, vec);
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
    base: Mat<f32>,
    orthogonal: Mat<f32>,
    rand_bias: Vec<f32>,
    centroids: Mat<f32>,
    offsets: Vec<u32>,
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

    /// Load from dir.
    pub fn load_from_dir(path: &Path) -> Self {
        let orthogonal = matrix_from_fvecs(&path.join("orthogonal.fvecs"));
        let centroids = matrix_from_fvecs(&path.join("centroids.fvecs"));

        let offsets_ids =
            read_vecs::<u32>(&path.join("offsets_ids.ivecs")).expect("open offsets_ids error");
        let offsets = offsets_ids.first().expect("offsets is empty").clone();
        let map_ids = offsets_ids.last().expect("map_ids is empty").clone();

        let factors = read_vecs::<f32>(&path.join("factors.fvecs")).expect("open factors error");
        let factor_ip = factors[0].clone();
        let factor_ppc = factors[1].clone();
        let error_bound = factors[2].clone();
        let x_c_distance_square = factors[3].clone();

        let x_binary_vec =
            read_u64_vecs(&path.join("x_binary_vec.u64vecs")).expect("open x_binary_vec error");

        let dim = orthogonal.nrows();
        let base = matrix_from_fvecs(&path.join("base.fvecs"))
            .transpose()
            .to_owned();

        Self {
            dim: dim as u32,
            base,
            orthogonal,
            centroids,
            rand_bias: gen_random_bias(dim),
            offsets,
            map_ids,
            x_binary_vec,
            x_c_distance_square,
            error_bound,
            factor_ip,
            factor_ppc,
        }
    }

    /// Dump to dir.
    pub fn dump_to_dir(&self, path: &Path) {
        std::fs::create_dir_all(path).expect("create dir error");
        write_matrix(&path.join("base.fvecs"), &self.base.transpose()).expect("write base error");
        write_matrix(&path.join("orthogonal.fvecs"), &self.orthogonal.as_ref())
            .expect("write orthogonal error");
        write_matrix(&path.join("centroids.fvecs"), &self.centroids.as_ref())
            .expect("write centroids error");

        write_vecs(
            &path.join("offsets_ids.ivecs"),
            &[&self.offsets, &self.map_ids],
        )
        .expect("write offsets_ids error");
        write_vecs(
            &path.join("factors.fvecs"),
            &[
                &self.factor_ip,
                &self.factor_ppc,
                &self.error_bound,
                &self.x_c_distance_square,
            ],
        )
        .expect("write factors error");
        write_vecs(
            &path.join("x_binary_vec.u64vecs"),
            &self.x_binary_vec.iter().collect::<Vec<_>>(),
        )
        .expect("write x_binary_vec error");
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
        let x_projected = (&base * &orthogonal).transpose().to_owned();
        let centroids = (centroids * &orthogonal).transpose().to_owned();

        // k-means
        let dim_sqrt = (dim as f32).sqrt();
        let mut labels = vec![Vec::new(); k];
        let mut x_c_distance_square = vec![0.0; n];
        let mut x_c_distance = vec![0.0; n];
        let mut x_binary_vec: Vec<Vec<u64>> = Vec::with_capacity(n);
        let mut x_signed_vec: Vec<Col<f32>> = Vec::with_capacity(n);
        let mut x_dot_product = vec![0.0; n];
        for (i, xp) in x_projected.col_iter().enumerate() {
            if i % 5000 == 0 {
                debug!("\t> preprocessing {}...", i);
            }
            let min_label = kmeans_nearest_cluster(&centroids.as_ref(), &xp);
            labels[min_label].push(i as u32);
            let x_c_quantized = xp - centroids.col(min_label);
            x_c_distance[i] = x_c_quantized.norm_l2();
            x_c_distance_square[i] = x_c_distance[i].powi(2);
            x_binary_vec.push(vector_binarize_u64(&x_c_quantized.as_ref()));
            x_signed_vec.push(vector_binarize_one(&x_c_quantized.as_ref()));
            let norm = x_c_distance[i] * dim_sqrt;
            x_dot_product[i] = if norm.is_normal() {
                x_c_quantized.as_ref().adjoint() * &x_signed_vec[i] / norm
            } else {
                DEFAULT_X_DOT_PRODUCT
            };
        }

        // factors
        debug!("computing factors...");
        let mut error_bound = Vec::with_capacity(n);
        let mut factor_ip = Vec::with_capacity(n);
        let mut factor_ppc: Vec<f32> = Vec::with_capacity(n);
        let error_base = 2.0 * EPSILON / (dim as f32 - 1.0).sqrt();
        let one_vec: Row<f32> = Row::ones(dim);
        for i in 0..n {
            let x_c_over_ip = x_c_distance[i] / x_dot_product[i];
            error_bound
                .push(error_base * (x_c_over_ip * x_c_over_ip - x_c_distance_square[i]).sqrt());
            factor_ip.push(-2.0 / dim_sqrt * x_c_over_ip);
            factor_ppc.push(factor_ip[i] * (&one_vec * &x_signed_vec[i]));
        }

        // sort by labels
        debug!("sort by labels...");
        let mut offsets = vec![0; k + 1];
        for i in 0..k {
            offsets[i + 1] = offsets[i] + labels[i].len() as u32;
        }
        let flat_labels: Vec<u32> = labels.into_iter().flatten().collect();
        let base = Mat::from_fn(n, dim, |i, j| base.read(flat_labels[i] as usize, j))
            .transpose()
            .to_owned();
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
            base,
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
    pub fn query(
        &self,
        query: &ColRef<f32>,
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

        let mut rough_distances = Vec::new();
        let mut quantized = vec![0u8; self.dim as usize];
        let mut binary_vec = vec![0u64; self.dim as usize * THETA_LOG_DIM as usize / 64];
        for &(dist, i) in lists[..length].iter() {
            let (lower_bound, upper_bound) =
                min_max_residual(&mut residual, &y_projected.as_ref(), &self.centroids.col(i));
            let delta = (upper_bound - lower_bound) / ((1 << THETA_LOG_DIM) as f32 - 1.0);
            let one_over_delta = 1.0 / delta;
            let scalar_sum = scalar_quantize(
                &mut quantized,
                &residual,
                &self.rand_bias,
                lower_bound,
                one_over_delta,
            );
            binary_vec.iter_mut().for_each(|element| *element = 0);
            vector_binarize_query(&quantized, &mut binary_vec);
            let dist_sqrt = dist.sqrt();
            for j in self.offsets[i]..self.offsets[i + 1] {
                let ju = j as usize;
                rough_distances.push((
                    (self.x_c_distance_square[ju]
                        + dist
                        + lower_bound * self.factor_ppc[ju]
                        + (2.0
                            * asymmetric_binary_dot_product(&self.x_binary_vec[ju], &binary_vec)
                                as f32
                            - scalar_sum as f32)
                            * self.factor_ip[ju]
                            * delta
                        - self.error_bound[ju] * dist_sqrt),
                    j,
                ));
            }
        }

        if heuristic_rank {
            self.heuristic_re_rank(query, &rough_distances, topk)
        } else {
            self.re_rank(query, &rough_distances, topk)
        }
    }

    /// BinaryHeap based re-rank with Ord32.
    fn re_rank(
        &self,
        query: &ColRef<f32>,
        rough_distances: &[(f32, u32)],
        topk: usize,
    ) -> Vec<(f32, u32)> {
        let mut threshold = f32::MAX;
        let mut precise = 0;
        let mut heap: BinaryHeap<(Ord32, u32)> = BinaryHeap::with_capacity(topk);
        for &(rough, u) in rough_distances.iter() {
            if rough < threshold {
                let accurate = l2_squared_distance(&self.base.col(u as usize), query);
                precise += 1;
                if accurate < threshold {
                    heap.push((accurate.into(), self.map_ids[u as usize]));
                    if heap.len() > topk {
                        heap.pop();
                    }
                    if heap.len() == topk {
                        threshold = heap.peek().unwrap().0.into();
                    }
                }
            }
        }

        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
        METRICS.add_query_count(1);

        heap.into_iter().map(|(a, b)| (a.into(), b)).collect()
    }

    /// Heuristic re-rank with a fixed windows size to update the threshold.
    fn heuristic_re_rank(
        &self,
        query: &ColRef<f32>,
        rough_distances: &[(f32, u32)],
        topk: usize,
    ) -> Vec<(f32, u32)> {
        let mut threshold = f32::MAX;
        let mut recent_max_accurate = f32::MIN;
        let mut res = Vec::with_capacity(topk);
        let mut count = 0;
        let mut precise = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < threshold {
                let accurate = l2_squared_distance(&self.base.col(u as usize), query);
                precise += 1;
                if accurate < threshold {
                    res.push((accurate, self.map_ids[u as usize]));
                    count += 1;
                    recent_max_accurate = recent_max_accurate.max(accurate);
                    if count >= WINDOWS_SIZE {
                        threshold = recent_max_accurate;
                        count = 0;
                        recent_max_accurate = f32::MIN;
                    }
                }
            }
        }

        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
        METRICS.add_query_count(1);
        let length = topk.min(res.len());
        res.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        res.truncate(length);
        res
    }
}
