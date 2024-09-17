use std::path::Path;
use std::time::Instant;

use argh::FromArgs;
use env_logger::Env;
use log::{debug, info};
use rabitq::metrics::METRICS;
use rabitq::utils::{calculate_recall, matrix1d_from_vec, read_vecs};
use rabitq::RaBitQ;

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
    /// saved directory
    #[argh(option, short = 's')]
    saved: String,
    /// heuristic re-rank (maybe faster when topk is large)
    #[argh(switch, short = 'h')]
    heuristic_rank: bool,
}

fn main() {
    let env = Env::default().filter_or("RABITQ_LOG", "debug");
    env_logger::init_from_env(env);

    let args: Args = argh::from_env();
    debug!("{:?}", args);
    let base_path = Path::new(args.base.as_str());
    let centroids_path = Path::new(args.centroids.as_str());
    let query_path = Path::new(args.query.as_str());
    let truth_path = Path::new(args.truth.as_str());

    let rabitq: RaBitQ;
    let local_path = Path::new(args.saved.as_str());
    if local_path.is_dir() {
        debug!("loading from {:?}...", local_path);
        rabitq = RaBitQ::load_from_dir(local_path);
    } else {
        debug!("training...");
        rabitq = RaBitQ::from_path(base_path, centroids_path);
        debug!("saving to local file: {:?}", local_path);
        rabitq.dump_to_dir(local_path);
    }

    let queries = read_vecs::<f32>(query_path).expect("read query error");
    let truth = read_vecs::<i32>(truth_path).expect("read truth error");
    debug!("querying...");
    let mut total_time = 0.0;
    let mut recall = 0.0;
    for (i, query) in queries.iter().enumerate() {
        let query_vec = matrix1d_from_vec(query);
        let start_time = Instant::now();
        let res = rabitq.query(
            &query_vec.as_ref(),
            args.probe,
            args.topk,
            args.heuristic_rank,
        );
        total_time += start_time.elapsed().as_secs_f64();
        let ids: Vec<i32> = res.iter().map(|(_, id)| *id as i32).collect();
        recall += calculate_recall(&truth[i], &ids, args.topk);
    }

    info!(
        "QPS: {}, recall: {}",
        queries.len() as f64 / total_time,
        recall / queries.len() as f32
    );
    info!("Metrics [{}]", METRICS.to_str());
}
