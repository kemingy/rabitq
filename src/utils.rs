//! Utility functions for the project.

use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

use nalgebra::debug::RandomOrthogonal;
use nalgebra::{DMatrix, DMatrixView, DVector, Dim, Dyn};
use num_traits::{FromBytes, ToBytes};
use rand::{thread_rng, Rng};

/// Generate a random orthogonal matrix.
pub fn gen_random_orthogonal(dim: usize) -> DMatrix<f32> {
    let mut rng = thread_rng();
    let random: RandomOrthogonal<f32, Dyn> =
        RandomOrthogonal::new(Dim::from_usize(dim), || rng.gen());
    random.unwrap()
}

/// Generate a random orthogonal matrix from QR decomposition.
pub fn gen_random_qr_orthogonal(dim: usize) -> DMatrix<f32> {
    let mut rng = thread_rng();
    let random = DMatrix::from_fn(dim, dim, |_, _| rng.gen::<f32>());
    random.qr().q()
}

/// Generate an identity matrix as a special orthogonal matrix.
///
/// Use this function to debug the logic.
pub fn gen_identity_matrix(dim: usize) -> DMatrix<f32> {
    DMatrix::identity(dim, dim)
}

/// Generate a fixed bias vector.
///
/// Use this function to debug the logic.
pub fn gen_fixed_bias(dim: usize) -> DVector<f32> {
    DVector::from_element(dim, 0.5)
}

/// Generate a random bias vector.
pub fn gen_random_bias(dim: usize) -> DVector<f32> {
    let mut rng = thread_rng();
    DVector::from_fn(dim, |_, _| rng.gen())
}

/// Convert a vector to a dynamic vector.
pub fn dvector_from_vec(vec: Vec<f32>) -> DVector<f32> {
    DVector::from_vec(vec)
}

/// Read the fvecs file and convert it to a matrix.
pub fn matrix_from_fvecs(path: &Path) -> DMatrix<f32> {
    let vecs = read_vecs::<f32>(path).expect("read vecs error");
    let dim = vecs[0].len();
    let rows = vecs.len();
    DMatrix::from_row_iterator(rows, dim, vecs.into_iter().flatten())
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
pub fn write_matrix<T>(path: &Path, matrix: &DMatrixView<T>) -> std::io::Result<()>
where
    T: Sized + ToBytes,
{
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for vec in matrix.row_iter() {
        writer.write_all(&(vec.len() as u32).to_le_bytes())?;
        for v in vec.iter() {
            writer.write_all(T::to_le_bytes(v).as_ref())?;
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
    let mut count = 0;
    let length = min(topk, truth.len());
    for t in truth.iter().take(length) {
        for y in res.iter().take(length.min(res.len())) {
            if *t == *y {
                count += 1;
                break;
            }
        }
    }
    (count as f32) / (length as f32)
}
