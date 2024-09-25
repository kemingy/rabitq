//! Utility functions for the project.

use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use faer::{Col, ColRef, Mat, MatRef};
use num_traits::{FromBytes, ToBytes};
use rand::distributions::{Distribution, Uniform};

use crate::consts::THETA_LOG_DIM;

/// Generate a random orthogonal matrix from QR decomposition.
pub fn gen_random_qr_orthogonal(dim: usize) -> Mat<f32> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::<f32>::new(0.0, 1.0);
    let random = Mat::from_fn(dim, dim, |_, _| uniform.sample(&mut rng));
    random.qr().compute_q()
}

/// Generate an identity matrix as a special orthogonal matrix.
///
/// Use this function to debug the logic.
pub fn gen_identity_matrix(dim: usize) -> Mat<f32> {
    Mat::identity(dim, dim)
}

/// Generate a fixed bias vector.
///
/// Use this function to debug the logic.
pub fn gen_fixed_bias(dim: usize) -> Mat<f32> {
    Mat::from_fn(1, dim, |_, _| 0.5)
}

/// Generate a random bias vector.
pub fn gen_random_bias(dim: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::<f32>::new(0.0, 1.0);
    (0..dim).map(|_| uniform.sample(&mut rng)).collect()
}

/// Convert a vector to a dynamic vector.
pub fn matrix1d_from_vec(vec: &[f32]) -> Col<f32> {
    Col::from_fn(vec.len(), |i| vec[i])
}

/// Read the fvecs file and convert it to a matrix.
pub fn matrix_from_fvecs(path: &Path) -> Mat<f32> {
    let vecs = read_vecs::<f32>(path).expect("read vecs error");
    let dim = vecs[0].len();
    let rows = vecs.len();
    Mat::from_fn(rows, dim, |i, j| vecs[i][j])
}

/// Convert the vector to binary format and store in a u64 vector.
#[inline]
pub fn vector_binarize_u64(vec: &ColRef<f32>) -> Vec<u64> {
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
pub fn vector_binarize_one(vec: &ColRef<f32>) -> Col<f32> {
    Col::from_fn(vec.nrows(), |i| if vec[i] > 0.0 { 1.0 } else { -1.0 })
}

/// Interface of `vector_binarize_query`
#[inline]
pub fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
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
#[inline]
pub fn asymmetric_binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    let length = x.len();
    let mut y_slice = y;
    for i in 0..THETA_LOG_DIM as usize {
        res += {
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") {
                    unsafe { crate::simd::binary_dot_product(x, y_slice) << i }
                } else {
                    binary_dot_product(x, y_slice) << i
                }
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
            {
                binary_dot_product(x, y_slice) << i
            }
        };
        y_slice = &y_slice[length..];
    }
    res
}

/// Calculate the L2 squared distance between two vectors.
#[inline]
pub fn l2_squared_distance(lhs: &ColRef<f32>, rhs: &ColRef<f32>) -> f32 {
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

// Get the min/max value of the residual of two vectors.
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

/// Interface of `min_max_residual`: get the min/max value of the residual of two vectors.
#[inline]
pub fn min_max_residual(res: &mut [f32], x: &ColRef<f32>, y: &ColRef<f32>) -> (f32, f32) {
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

/// Interface of `scalar_quantize`: scale vector to u8.
#[inline]
pub fn scalar_quantize(
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
pub fn project(vec: &ColRef<f32>, orthogonal: &MatRef<f32>) -> Col<f32> {
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
pub fn kmeans_nearest_cluster(centroids: &MatRef<f32>, vec: &ColRef<f32>) -> (usize, f32) {
    let mut min_dist = f32::MAX;
    let mut min_label = 0;
    for (j, centroid) in centroids.col_iter().enumerate() {
        let dist = l2_squared_distance(&centroid, vec);
        if dist < min_dist {
            min_dist = dist;
            min_label = j;
        }
    }
    (min_label, min_dist)
}

/// Read the fvces/ivces file.
pub fn read_vecs<T>(path: &Path) -> std::io::Result<Vec<Vec<T>>>
where
    T: Sized + FromBytes<Bytes = [u8; 4]>,
{
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut buf = [0u8; 4];
    let mut count: usize;
    let mut vecs = Vec::new();
    loop {
        count = reader.read(&mut buf)?;
        if count == 0 {
            break;
        }
        let dim = u32::from_le_bytes(buf) as usize;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut buf)?;
            vec.push(T::from_le_bytes(&buf));
        }
        vecs.push(vec);
    }
    Ok(vecs)
}

/// Read the u64 vecs file.
///
/// This cannot be combined with the `read_vecs` function because Rust doesn't support
/// using generic type for array length https://github.com/rust-lang/rust/issues/43408.
pub fn read_u64_vecs(path: &Path) -> std::io::Result<Vec<Vec<u64>>> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut dim_buf = [0u8; 4];
    let mut val_buf = [0u8; 8];
    let mut count: usize;
    let mut vecs = Vec::new();
    loop {
        count = reader.read(&mut dim_buf)?;
        if count == 0 {
            break;
        }
        let dim = u32::from_le_bytes(dim_buf) as usize;
        let mut vec = Vec::with_capacity(dim);
        for _ in 0..dim {
            reader.read_exact(&mut val_buf)?;
            vec.push(u64::from_le_bytes(val_buf));
        }
        vecs.push(vec);
    }
    Ok(vecs)
}

/// Write the fvecs/ivecs file from DMatrix.
pub fn write_matrix<T>(path: &Path, matrix: &MatRef<T>) -> std::io::Result<()>
where
    T: Sized + ToBytes + faer::Entity,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in matrix.row_iter() {
        writer.write_all(&(vec.ncols() as u32).to_le_bytes())?;
        for i in 0..vec.ncols() {
            writer.write_all(T::to_le_bytes(&vec.read(i)).as_ref())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Write the fvecs/ivecs file.
pub fn write_vecs<T>(path: &Path, vecs: &[&Vec<T>]) -> std::io::Result<()>
where
    T: Sized + ToBytes,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in vecs.iter() {
        writer.write_all(&(vec.len() as u32).to_le_bytes())?;
        for v in vec.iter() {
            writer.write_all(T::to_le_bytes(v).as_ref())?;
        }
    }
    writer.flush()?;
    Ok(())
}

/// Calculate the recall.
pub fn calculate_recall(truth: &[i32], res: &[i32], topk: usize) -> f32 {
    assert_eq!(res.len(), topk);
    let mut count = 0;
    for id in res {
        for t in truth.iter().take(topk) {
            if *id == *t {
                count += 1;
                break;
            }
        }
    }
    (count as f32) / (topk as f32)
}
