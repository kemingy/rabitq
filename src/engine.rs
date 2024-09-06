//! RaBitQ Engine that provides the interface for training & search.
//!
//! TODO: support insertion & deletion.

use core::f32;
use std::path::Path;

use log::debug;
use nalgebra::{DMatrix, DMatrixView, DVector, DVectorView};
use serde::{Deserialize, Serialize};

use crate::consts::{DEFAULT_X_DOT_PRODUCT, EPSILON, THETA_LOG_DIM};
use crate::rabitq::RaBitQNode;
use crate::utils::{gen_random_bias, gen_random_qr_orthogonal, matrix_from_fvecs, write_matrix};

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

/// RaBitQ Engine struct.
#[derive(Debug, Serialize, Deserialize)]
pub struct RaBitQEngine {
    dim: u32,
    centroids: DMatrix<f32>,
    orthogonal: DMatrix<f32>,
    rand_bias: DVector<f32>,
    nodes: Vec<RaBitQNode>,
}

impl RaBitQEngine {
    /// Build the RaBitQ model from the base and centroids files.
    pub fn new(base_path: &Path, centroids_path: &Path) -> Self {
        let base = matrix_from_fvecs(base_path);
        let (n, dim) = base.shape();
        let centroids = matrix_from_fvecs(centroids_path);
        let k = centroids.shape().0;
        debug!("n: {}, dim: {}, k: {}", n, dim, k);
        let orthogonal = gen_random_qr_orthogonal(dim);
        let rand_bias = gen_random_bias(dim);

        // projection
        debug!("projection x & c...");
        let x_projected = (&base * &orthogonal).transpose();
        let centroids = (centroids * &orthogonal).transpose();

        // k-means
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

        // build k nodes
        debug!("build {k} nodes...");
        let mut nodes = Vec::with_capacity(k);
        for (i, label) in labels.iter().enumerate() {
            nodes.push(RaBitQNode::new(
                centroids.column(i).into(),
                DMatrix::from_row_iterator(
                    label.len(),
                    dim,
                    label
                        .iter()
                        .flat_map(|&i| base.row(i as usize).iter().copied().collect::<Vec<_>>()),
                )
                .transpose(),
                label
                    .iter()
                    .map(|&i| x_binary_vec[i as usize].clone())
                    .collect(),
                label
                    .iter()
                    .map(|&i| x_c_distance_square[i as usize])
                    .collect(),
                label.iter().map(|&i| error_bound[i as usize]).collect(),
                label.iter().map(|&i| factor_ip[i as usize]).collect(),
                label.iter().map(|&i| factor_ppc[i as usize]).collect(),
                label.clone(),
            ))
        }

        Self {
            dim: dim as u32,
            centroids,
            orthogonal,
            rand_bias,
            nodes,
        }
    }

    fn num_of_nodes(&self) -> usize {
        self.nodes.len()
    }

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

    /// Dump to a directory.
    pub fn dump_to_dir(&self, path: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(path)?;
        write_matrix(&path.join("centroids.fvecs"), &self.centroids.as_view())?;
        write_matrix(&path.join("orthogonal.fvecs"), &self.orthogonal.as_view())?;

        Ok(())
    }

    /// Load from a directory.
    pub fn load_from_dir(path: &Path) -> Self {
        let centroids = matrix_from_fvecs(&path.join("centroids.fvecs"));
        let orthogonal = matrix_from_fvecs(&path.join("orthogonal.fvecs"));
        let rand_bias = gen_random_bias(centroids.ncols());
        let nodes = Vec::new();
        Self {
            dim: orthogonal.ncols() as u32,
            centroids,
            orthogonal,
            rand_bias,
            nodes,
        }
    }

    /// Query the RaBitQ model with the given vector.
    pub fn query(&self, query: &DVectorView<f32>, probe: usize, topk: usize) -> Vec<(f32, u32)> {
        let y_projected = query.tr_mul(&self.orthogonal).transpose();
        let k = self.num_of_nodes();
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

        let mut results = Vec::new();
        let mut residual = DVector::<f32>::zeros(self.dim as usize);
        let mut quantized = DVector::<u8>::zeros(self.dim as usize);
        let mut binary_vec = vec![0u64; query.len() * THETA_LOG_DIM as usize / 64];
        let mut rank_threshold = f32::MAX;
        for &(dist, i) in lists[..length].iter() {
            binary_vec.iter_mut().for_each(|element| *element = 0);
            self.nodes[i].query(
                query,
                &y_projected.as_view(),
                dist,
                &mut rank_threshold,
                &self.rand_bias.as_view(),
                &mut results,
                &mut residual,
                &mut quantized,
                &mut binary_vec,
            );
        }
        let length = topk.min(results.len());
        results.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        results.truncate(length);
        results
    }
}
