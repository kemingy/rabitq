use core::f32;
use std::cmp::min;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use argh::FromArgs;
use nalgebra::debug::RandomOrthogonal;
use nalgebra::{DMatrix, DVector, Dim, Dyn, OMatrix};
use num_traits::{FromBytes, ToPrimitive};
use rand::{thread_rng, Rng};
use serde::{Deserialize, Serialize};

const DEFAULT_X_DOT_PRODUCT: f32 = 0.8;
const EPSILON: f32 = 1.9;
const THETA_LOG_DIM: u32 = 4;
const WINDOWS_SIZE: usize = 16;

fn read_vecs<T>(path: &Path) -> std::io::Result<Vec<Vec<T>>>
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

fn gen_random_orthogonal(dim: usize) -> OMatrix<f32, Dyn, Dyn> {
    let mut rng = thread_rng();
    let random: RandomOrthogonal<f32, Dyn> =
        RandomOrthogonal::new(Dim::from_usize(dim), || rng.gen());
    random.unwrap()
}

fn gen_random_vector(dim: usize) -> DVector<f32> {
    let mut rng = thread_rng();
    DVector::from_fn(dim, |_, _| rng.gen())
}

fn matrix_from_fvecs(path: &Path) -> DMatrix<f32> {
    let vecs = read_vecs::<f32>(path).expect("read vecs error");
    let dim = vecs[0].len();
    let rows = vecs.len();
    let matrix = DMatrix::from_row_iterator(rows, dim, vecs.into_iter().flatten());
    matrix
}

fn vector_binarize_u64(vec: &DVector<f32>) -> Vec<u64> {
    let mut binary = vec![0u64; (vec.len() + 63) / 64];
    for (i, &v) in vec.iter().enumerate() {
        if v > 0.0 {
            binary[i / 64] |= 1 << (i % 64);
        }
    }
    binary
}

fn vector_binarize_one(vec: &DVector<f32>) -> DVector<f32> {
    let mut binary = DVector::zeros(vec.len());
    for (i, &v) in vec.iter().enumerate() {
        binary[i] = if v > 0.0 { 1.0 } else { -1.0 };
    }
    binary
}

fn vector_binarize_query(vec: &[u8]) -> Vec<u64> {
    let length = vec.len();
    let mut binary = vec![0u64; length * THETA_LOG_DIM as usize / 64];
    for j in 0..THETA_LOG_DIM as usize {
        for i in 0..length {
            binary[(i + j * length) / 64] |= (((vec[i] >> j) & 1) as u64) << (i % 64);
        }
    }
    binary
}

#[inline]
fn binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    for i in 0..x.len() {
        res += (x[i] & y[i]).count_ones();
    }
    res
}

fn asymmetric_binary_dot_product(x: &[u64], y: &[u64]) -> u32 {
    let mut res = 0;
    let length = x.len();
    let mut y_slice = y;
    for i in 0..THETA_LOG_DIM as usize {
        res += binary_dot_product(x, &y_slice) << i;
        y_slice = &y_slice[length..];
    }
    res
}

#[derive(Debug, Serialize, Deserialize)]
struct RaBitQ {
    dim: usize,
    base: DMatrix<f32>,
    orthogonal: OMatrix<f32, Dyn, Dyn>,
    rand_bias: DVector<f32>,
    centroids: DMatrix<f32>,
    labels: Vec<Vec<u32>>,
    x_binary_vec: Vec<Vec<u64>>,
    x_c_distance_square: Vec<f32>,
    error_bound: Vec<f32>,
    factor_ip: Vec<f32>,
    factor_ppc: Vec<f32>,
}

impl RaBitQ {
    pub fn from_path(base_path: &Path, centroid_path: &Path) -> Self {
        let base = matrix_from_fvecs(base_path);
        let (n, dim) = base.shape();
        let mut centroids = matrix_from_fvecs(centroid_path);
        let k = centroids.shape().0;
        println!("n: {}, dim: {}, k: {}", n, dim, k);
        let orthogonal = gen_random_orthogonal(dim);
        let rand_bias = gen_random_vector(dim);

        // projection
        println!("projection x & c...");
        let x_projected = (&base * &orthogonal).transpose();
        centroids = (centroids * &orthogonal).transpose();
        println!(
            "x_projected shape: {:?}, centroids shape: {:?}",
            x_projected.shape(),
            centroids.shape()
        );

        // kmeans
        println!("preprocessing x...");
        let mut labels = vec![Vec::new(); k];
        let mut x_c_distance_square = vec![0.0; n];
        let mut x_c_distance = vec![0.0; n];
        let mut x_binary_vec: Vec<Vec<u64>> = Vec::with_capacity(n);
        let mut x_signed_vec: Vec<DVector<f32>> = Vec::with_capacity(n);
        let mut x_dot_product = vec![0.0; n];
        for (i, row) in x_projected.column_iter().enumerate() {
            if i % 5000 == 0 {
                println!("\tpreprocessing {}...", i);
            }
            let mut min_dist = f32::MAX;
            let mut min_label = 0;
            for (j, centroid) in centroids.column_iter().enumerate() {
                let dist = (row - centroid).norm();
                if dist < min_dist {
                    min_dist = dist;
                    min_label = j;
                }
            }
            labels[min_label].push(i as u32);
            let x_c_quantized = row - centroids.column(min_label);
            x_c_distance[i] = x_c_quantized.norm();
            x_c_distance_square[i] = x_c_distance[i].powi(2);
            x_binary_vec.push(vector_binarize_u64(&x_c_quantized));
            x_signed_vec.push(vector_binarize_one(&x_c_quantized));
            let norm = x_c_distance[i] * (dim as f32).sqrt();
            x_dot_product[i] = if norm.is_normal() {
                x_c_quantized.dot(&x_signed_vec[i]) / norm
            } else {
                DEFAULT_X_DOT_PRODUCT
            };
        }

        // factors
        println!("computing factors...");
        let mut error_bound = Vec::with_capacity(n);
        let mut factor_ip = Vec::with_capacity(n);
        let mut factor_ppc = Vec::with_capacity(n);
        let error_base = 2.0 * EPSILON / (dim as f32 - 1.0).sqrt();
        let dim_sqrt = (dim as f32).sqrt();
        let one_vec = DVector::from_element(dim, 1.0);
        for i in 0..n {
            let x_c_over_ip = x_c_distance[i] / x_dot_product[i];
            error_bound
                .push(error_base * (x_c_over_ip * x_c_over_ip - x_c_distance_square[i]).sqrt());
            factor_ip.push(-2.0 / dim_sqrt * x_c_over_ip);
            factor_ppc.push(factor_ip[i] * one_vec.dot(&x_signed_vec[i]));
        }

        // sort by labels
        let mut offsets = vec![0; k + 1];
        for i in 0..k {
            offsets[i + 1] = offsets[i] + labels[i].len();
        }

        Self {
            dim,
            base: base.transpose(),
            orthogonal,
            rand_bias,
            labels,
            centroids,
            x_binary_vec,
            x_c_distance_square,
            error_bound,
            factor_ip,
            factor_ppc,
        }
    }

    pub fn query(&self, query_path: &Path, truth_path: &Path, probe: usize, topk: usize) {
        let query = matrix_from_fvecs(query_path);
        let truth = read_vecs::<i32>(truth_path).expect("read truth error");

        // projection
        let start_time = Instant::now();
        let query = query * &self.orthogonal;

        let mut recall = 0.0;
        for (i, q) in query.row_iter().enumerate() {
            recall += calculate_recall(
                &truth[i],
                &self.query_one(&q.transpose(), probe, topk),
                topk,
            );
        }
        println!(
            "recall: {}, QPS: {}",
            recall / query.shape().0 as f32,
            query.shape().0 as f32 / start_time.elapsed().as_secs_f32()
        );
    }

    fn query_one(&self, query: &DVector<f32>, probe: usize, topk: usize) -> Vec<i32> {
        let k = self.centroids.shape().1;
        let mut lists = Vec::with_capacity(k);
        for (i, centroid) in self.centroids.column_iter().enumerate() {
            let dist = (query - centroid).norm();
            lists.push((dist, i));
        }
        lists.select_nth_unstable_by(probe, |a, b| a.0.total_cmp(&b.0));
        lists.truncate(probe);
        let mut rough_distances = Vec::new();
        for &(dist, i) in lists.iter() {
            let residual = query - self.centroids.column(i);
            let lower_bound = residual.min();
            let upper_bound = residual.max();
            let delta = (upper_bound - lower_bound) / ((1 << THETA_LOG_DIM) as f32 - 1.0);
            let one_over_delta = 1.0 / delta;
            let mut scalar_sum = 0u32;
            let mut y_quantized = vec![0u8; self.dim];
            for j in 0..self.dim {
                y_quantized[j] = ((residual[j] - lower_bound) * one_over_delta + self.rand_bias[j])
                    .to_u8()
                    .expect("convert to u8 error");
                scalar_sum += y_quantized[j] as u32;
            }
            let y_binary_vec = vector_binarize_query(&y_quantized);
            for &j in self.labels[i].iter() {
                let k = j as usize;
                rough_distances.push((
                    (self.x_c_distance_square[k]
                        + dist * dist
                        + lower_bound * self.factor_ppc[k]
                        + (2.0
                            * asymmetric_binary_dot_product(&self.x_binary_vec[k], &y_binary_vec)
                                as f32
                            - scalar_sum as f32)
                            * self.factor_ip[k]
                            * delta
                        - self.error_bound[k] * dist),
                    j,
                ));
            }
        }

        self.rerank(query, &rough_distances, topk)
    }

    fn rerank(
        &self,
        query: &DVector<f32>,
        rough_distances: &[(f32, u32)],
        topk: usize,
    ) -> Vec<i32> {
        let mut threshold = f32::MAX;
        let mut res = Vec::with_capacity(topk);
        let mut recent_max_accurate = f32::MIN;
        let mut count = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < threshold {
                let accurate = self.base.column(u as usize).dot(query);
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

        res.select_nth_unstable_by(topk, |a, b| a.0.total_cmp(&b.0));
        res.truncate(topk);
        res.iter().map(|(_, u)| *u).collect()
    }
}

fn calculate_recall(truth: &[i32], res: &[i32], topk: usize) -> f32 {
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

#[derive(FromArgs, Debug)]
/// RaBitQ
struct Args {
    /// base path
    #[argh(option, short = 'b')]
    base: String,
    /// centroids path
    #[argh(option, short = 'c')]
    centroids: String,
    /// query path
    #[argh(option, short = 'q')]
    query: String,
    /// truth path
    #[argh(option, short = 't')]
    truth: String,
    /// probe
    #[argh(option, short = 'p', default = "100")]
    probe: usize,
    /// topk
    #[argh(option, short = 'k', default = "10")]
    topk: usize,
}

fn main() {
    let args: Args = argh::from_env();
    println!("{:?}", args);
    let base_path = Path::new(args.base.as_str());
    let centroids_path = Path::new(args.centroids.as_str());
    let query_path = Path::new(args.query.as_str());
    let truth_path = Path::new(args.truth.as_str());

    let local_path = Path::new("rabitq.json");
    let rabitq: RaBitQ;
    if local_path.is_file() {
        println!("loading from local...");
        let file = File::open(local_path).expect("open json file error");
        rabitq = serde_json::from_reader(file).expect("deserialize error");
    } else {
        println!("training...");
        rabitq = RaBitQ::from_path(&base_path, &centroids_path);
        println!("saving to local...");
        let file = File::create(local_path).expect("create json file error");
        serde_json::to_writer(file, &rabitq).expect("serialize error");
    }

    println!("querying...");
    rabitq.query(query_path, truth_path, args.probe, args.topk);
}
