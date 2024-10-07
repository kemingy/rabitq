use std::path::Path;
use std::sync::Arc;

use axum::extract::State;
use axum::response::IntoResponse;
use axum::routing::{get, post};
use axum::{Json, Router};
use disk::cache::download_meta_from_s3;
use disk::disk::DiskRaBitQ;
use env_logger::Env;
use log::info;
use rabitq::metrics::METRICS;
use serde::{Deserialize, Serialize};
use tokio::signal::unix::{signal, SignalKind};

mod args;

async fn shutdown_signal() {
    let mut interrupt = signal(SignalKind::interrupt()).unwrap();
    let mut terminate = signal(SignalKind::terminate()).unwrap();
    loop {
        tokio::select! {
            _ = interrupt.recv() => {
                info!("Received interrupt signal");
                break;
            }
            _ = terminate.recv() => {
                info!("Received terminate signal");
                break;
            }
        };
    }
}

async fn health_check() -> impl IntoResponse {
    "Ok"
}

async fn query_vector(State(state): State<Arc<AppState>>, req: Json<Request>) -> impl IntoResponse {
    let res = state
        .engine
        .query(req.query.clone(), req.probe as usize, req.top_k as usize)
        .await;
    let ids = res.iter().map(|(_, id)| *id).collect();
    let scores = res.iter().map(|(score, _)| *score).collect();
    Json(Response { ids, scores })
}

async fn metrics() -> impl IntoResponse {
    METRICS.to_str()
}

#[derive(Debug)]
struct AppState {
    pub engine: DiskRaBitQ,
}

#[derive(Debug, Serialize, Deserialize)]
struct Request {
    pub query: Vec<f32>,
    pub top_k: u32,
    pub probe: u32,
}

#[derive(Debug, Serialize, Deserialize)]
struct Response {
    pub ids: Vec<u32>,
    pub scores: Vec<f32>,
}

#[tokio::main]
async fn main() {
    let env = Env::default().filter_or("RUST_LOG", "info");
    env_logger::init_from_env(env);

    let config: args::Args = argh::from_env();
    let model_path = Path::new(&config.dir);
    download_meta_from_s3(&config.bucket, &config.key, &model_path)
        .await
        .expect("failed to download meta");
    let rabitq =
        DiskRaBitQ::load_from_dir(model_path, config.cache_dir, config.bucket, config.key).await;

    let state = AppState { engine: rabitq };
    let shared_state = Arc::new(state);
    let app = Router::new()
        .route("/", get(health_check))
        .route("/health", get(health_check))
        .route("/metrics", get(metrics))
        .route("/query", post(query_vector))
        .with_state(Arc::clone(&shared_state));
    let addr = format!("0.0.0.0:{}", config.port);
    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    info!("Server listening on {}", &addr);
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}
