//! Accelerate with SIMD.

use faer::{ColRef, RowRef};

use crate::consts::THETA_LOG_DIM;

/// Compute the squared Euclidean distance between two vectors.
/// Code refer to https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx")]
pub unsafe fn l2_squared_distance(lhs: &ColRef<f32>, rhs: &ColRef<f32>) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(lhs.nrows(), rhs.nrows());
    let mut lhs_ptr = lhs.as_ptr();
    let mut rhs_ptr = rhs.as_ptr();
    let block_16_num = lhs.nrows() >> 4;
    let rest_num = lhs.nrows() & 0b1111;
    let mut f32x8 = [0.0f32; 8];
    let (mut diff, mut vx, mut vy): (__m256, __m256, __m256);
    let mut sum = _mm256_setzero_ps();

    for _ in 0..block_16_num {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);

        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }

    for _ in 0..rest_num / 8 {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_fmadd_ps(diff, diff, sum);
    }
    _mm256_store_ps(f32x8.as_mut_ptr(), sum);
    let mut res =
        f32x8[0] + f32x8[1] + f32x8[2] + f32x8[3] + f32x8[4] + f32x8[5] + f32x8[6] + f32x8[7];

    for _ in 0..rest_num {
        let residual = *lhs_ptr - *rhs_ptr;
        res += residual * residual;
        lhs_ptr = lhs_ptr.add(1);
        rhs_ptr = rhs_ptr.add(1);
    }
    res
}

/// Convert an [u8] to 4x binary vector stored as u64.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx,avx2")]
pub unsafe fn vector_binarize_query(vec: &[u8], binary: &mut [u64]) {
    use std::arch::x86_64::*;

    let length = vec.len();
    let mut ptr = vec.as_ptr() as *const __m256i;

    for i in (0..length).step_by(32) {
        // since it's not guaranteed that the vec is fully-aligned
        let mut v = _mm256_loadu_si256(ptr);
        ptr = ptr.add(1);
        v = _mm256_slli_epi32(v, 4);
        for j in 0..THETA_LOG_DIM as usize {
            let mask = (_mm256_movemask_epi8(v) as u32) as u64;
            // let shift = if (i / 32) % 2 == 0 { 32 } else { 0 };
            let shift = ((i >> 5) & 1) << 5;
            binary[(3 - j) * (length >> 6) + (i >> 6)] |= mask << shift;
            v = _mm256_slli_epi32(v, 1);
        }
    }
}

/// Compute the min and max value of a vector.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx")]
pub unsafe fn min_max_residual(res: &mut [f32], x: &ColRef<f32>, y: &ColRef<f32>) -> (f32, f32) {
    use std::arch::x86_64::*;

    let mut min_32x8 = _mm256_set1_ps(f32::MAX);
    let mut max_32x8 = _mm256_set1_ps(f32::MIN);
    let mut x_ptr = x.as_ptr();
    let mut y_ptr = y.as_ptr();
    let mut res_ptr = res.as_mut_ptr();
    let mut f32x8 = [0.0f32; 8];
    let mut min = f32::MAX;
    let mut max = f32::MIN;
    let length = res.len();
    let rest = length & 0b111;
    let (mut x256, mut y256, mut res256);

    for _ in 0..(length / 8) {
        x256 = _mm256_loadu_ps(x_ptr);
        y256 = _mm256_loadu_ps(y_ptr);
        res256 = _mm256_sub_ps(x256, y256);
        _mm256_storeu_ps(res_ptr, res256);
        x_ptr = x_ptr.add(8);
        y_ptr = y_ptr.add(8);
        res_ptr = res_ptr.add(8);
        min_32x8 = _mm256_min_ps(min_32x8, res256);
        max_32x8 = _mm256_max_ps(max_32x8, res256);
    }
    _mm256_storeu_ps(f32x8.as_mut_ptr(), min_32x8);
    for &x in f32x8.iter() {
        if x < min {
            min = x;
        }
    }
    _mm256_storeu_ps(f32x8.as_mut_ptr(), max_32x8);
    for &x in f32x8.iter() {
        if x > max {
            max = x;
        }
    }

    for _ in 0..rest {
        *res_ptr = *x_ptr - *y_ptr;
        if *res_ptr < min {
            min = *res_ptr;
        }
        if *res_ptr > max {
            max = *res_ptr;
        }
        res_ptr = res_ptr.add(1);
        x_ptr = x_ptr.add(1);
        y_ptr = y_ptr.add(1);
    }

    (min, max)
}

/// Compute the u8 scalar quantization of a f32 vector.
///
/// This function doesn't need `bias` because it *round* the f32 to u32 instead of *floor*.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx,avx2")]
pub unsafe fn scalar_quantize(
    quantized: &mut [u8],
    vec: &[f32],
    lower_bound: f32,
    multiplier: f32,
) -> u32 {
    use std::arch::x86_64::*;

    let mut quantize_ptr = quantized.as_mut_ptr() as *mut u64;

    let lower = _mm256_set1_ps(lower_bound);
    let scalar = _mm256_set1_ps(multiplier);
    let mut sum256 = _mm256_setzero_si256();
    let mask = _mm256_setr_epi8(
        0, 4, 8, 12, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 4, 8, 12, -1, -1, -1, -1,
        -1, -1, -1, -1, -1, -1, -1, -1,
    );
    let length = vec.len();
    let rest = length & 0b111;
    let mut vec_ptr = vec.as_ptr();
    let mut quantize8xi32;

    for _ in 0..(length / 8) {
        let v = _mm256_loadu_ps(vec_ptr);
        // `_mm256_cvtps_epi32` is *round* instead of *floor*, so we don't need the bias here
        quantize8xi32 = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_sub_ps(v, lower), scalar));
        sum256 = _mm256_add_epi32(sum256, quantize8xi32);
        // extract the lower 8 bits of each 32-bit integer and save them to [0..32] and [128..160]
        let shuffled = _mm256_shuffle_epi8(quantize8xi32, mask);
        quantize_ptr.write(
            (_mm256_extract_epi32(shuffled, 0) as u64)
                | ((_mm256_extract_epi32(shuffled, 4) as u64) << 32),
        );
        quantize_ptr = quantize_ptr.add(1);
        vec_ptr = vec_ptr.add(8);
    }

    // Compute the sum of the quantized values
    // add [4..7] to [0..3]
    let mut combined = _mm256_add_epi32(sum256, _mm256_permute2f128_si256(sum256, sum256, 1));
    // combine [0..3] to [0..1]
    combined = _mm256_hadd_epi32(combined, combined);
    // combine [0..1] to [0]
    combined = _mm256_hadd_epi32(combined, combined);
    let mut sum = _mm256_cvtsi256_si32(combined) as u32;

    for i in 0..rest {
        // this should be safe as it's a scalar quantization
        let q = ((*vec_ptr - lower_bound) * multiplier)
            .round()
            .to_int_unchecked::<u8>();
        quantized[length - rest + i] = q;
        sum += q as u32;
        vec_ptr = vec_ptr.add(1);
    }

    sum
}

/// Compute the dot product of two vectors.
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "fma,avx,avx2")]
pub unsafe fn vector_dot_product(lhs: &RowRef<f32>, rhs: &ColRef<f32>) -> f32 {
    use std::arch::x86_64::*;

    let mut lhs_ptr = lhs.as_ptr();
    let mut rhs_ptr = rhs.as_ptr();
    let length = lhs.ncols();
    let rest = length & 0b111;
    let (mut vx, mut vy): (__m256, __m256);
    let mut accumulate = _mm256_setzero_ps();
    let mut f32x8 = [0.0f32; 8];

    for _ in 0..(length / 16) {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        accumulate = _mm256_fmadd_ps(vx, vy, accumulate);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);

        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        accumulate = _mm256_fmadd_ps(vx, vy, accumulate);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
    }
    for _ in 0..((length & 0b1111) / 8) {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        accumulate = _mm256_fmadd_ps(vx, vy, accumulate);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
    }
    _mm256_storeu_ps(f32x8.as_mut_ptr(), accumulate);
    let mut sum =
        f32x8[0] + f32x8[1] + f32x8[2] + f32x8[3] + f32x8[4] + f32x8[5] + f32x8[6] + f32x8[7];

    for _ in 0..rest {
        sum += *lhs_ptr * *rhs_ptr;
        lhs_ptr = lhs_ptr.add(1);
        rhs_ptr = rhs_ptr.add(1);
    }

    sum
}
