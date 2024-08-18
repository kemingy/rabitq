use argh::FromArgs;

use std::fs::{read, write};
use std::path::Path;
use std::time::Instant;

use rabitq::{calculate_recall, dvector_from_vec, read_vecs, RaBitQ};

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
        rabitq = serde_json::from_slice(&read(&local_path).expect("open json error"))
            .expect("deserialize error");
    } else {
        println!("training...");
        rabitq = RaBitQ::from_path(&base_path, &centroids_path);
        println!("saving to local...");
        write(
            local_path,
            serde_json::to_string(&rabitq).expect("serialize error"),
        )
        .expect("write json error");
    }

    let queries = read_vecs::<f32>(&query_path).expect("read query error");
    let truth = read_vecs::<i32>(&truth_path).expect("read truth error");
    println!("querying...");
    let mut total_time = 0.0;
    let mut recall = 0.0;
    for (i, query) in queries.iter().enumerate() {
        let query_vec = dvector_from_vec(query.clone());
        let start_time = Instant::now();
        let res = rabitq.query_one(&query_vec, args.probe, args.topk);
        // println!("{:?}", res);
        total_time += start_time.elapsed().as_secs_f64();
        recall += calculate_recall(&truth[i], &res, args.topk);
    }

    println!(
        "QPS: {}, recall: {}",
        queries.len() as f64 / total_time,
        recall / queries.len() as f32
    );
}
