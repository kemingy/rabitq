use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use nalgebra::debug::RandomOrthogonal;
use nalgebra::{DMatrix, DVector, Dim, Dyn};
use num_traits::FromBytes;
use rand::{thread_rng, Rng};

pub fn gen_random_orthogonal(dim: usize) -> DMatrix<f32> {
    let mut rng = thread_rng();
    let random: RandomOrthogonal<f32, Dyn> =
        RandomOrthogonal::new(Dim::from_usize(dim), || rng.gen());
    random.unwrap()
}

pub fn gen_random_vector(dim: usize) -> DVector<f32> {
    let mut rng = thread_rng();
    DVector::from_fn(dim, |_, _| rng.gen())
}

pub fn dvector_from_vec(vec: Vec<f32>) -> DVector<f32> {
    DVector::from_vec(vec)
}

pub fn matrix_from_fvecs(path: &Path) -> DMatrix<f32> {
    let vecs = read_vecs::<f32>(path).expect("read vecs error");
    let dim = vecs[0].len();
    let rows = vecs.len();
    let matrix = DMatrix::from_row_iterator(rows, dim, vecs.into_iter().flatten());
    matrix
}

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

pub fn calculate_recall(truth: &[i32], res: &[i32], topk: usize) -> f32 {
    let mut count = 0;
    let length = min(topk, truth.len());
    for i in 0..length {
        for j in 0..min(length, res.len()) {
            if res[j] == truth[i] {
                count += 1;
                break;
            }
        }
    }
    (count as f32) / (length as f32)
}
