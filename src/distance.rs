//! Compute the distance between two vectors.

use nalgebra::DVectorView;

/// Compute the squared Euclidean distance between two vectors.
/// Code refer to https://github.com/nmslib/hnswlib/blob/master/hnswlib/space_l2.h
///
/// # Safety
///
/// This function is marked unsafe because it requires the AVX intrinsics.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
pub unsafe fn l2_squared_distance_avx2(lhs: &DVectorView<f32>, rhs: &DVectorView<f32>) -> f32 {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    assert_eq!(lhs.len(), rhs.len());
    let mut lhs_ptr = lhs.as_ptr();
    let mut rhs_ptr = rhs.as_ptr();
    let block_16_num = lhs.len() >> 4;
    let rest_num = lhs.len() & 0b1111;
    let mut temp_block = [0.0f32; 8];
    let temp_block_ptr = temp_block.as_mut_ptr();
    let (mut diff, mut vx, mut vy): (__m256, __m256, __m256);
    let mut sum = _mm256_setzero_ps();

    for _ in 0..block_16_num {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));

        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }

    for _ in 0..rest_num / 8 {
        vx = _mm256_loadu_ps(lhs_ptr);
        vy = _mm256_loadu_ps(rhs_ptr);
        lhs_ptr = lhs_ptr.add(8);
        rhs_ptr = rhs_ptr.add(8);
        diff = _mm256_sub_ps(vx, vy);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    _mm256_store_ps(temp_block_ptr, sum);

    let mut res = temp_block[0]
        + temp_block[1]
        + temp_block[2]
        + temp_block[3]
        + temp_block[4]
        + temp_block[5]
        + temp_block[6]
        + temp_block[7];

    for _ in 0..rest_num % 8 {
        let residual = *lhs_ptr - *rhs_ptr;
        res += residual * residual;
        lhs_ptr = lhs_ptr.add(1);
        rhs_ptr = rhs_ptr.add(1);
    }
    res
}
