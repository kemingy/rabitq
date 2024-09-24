//! Rank the rough estimate distances with accurate distances.

use std::collections::BinaryHeap;

use faer::{Col, ColRef, MatRef};

use crate::cache::CACHED_VECTOR;
use crate::consts::WINDOW_SIZE;
use crate::metrics::METRICS;
use crate::ord32::Ord32;
use crate::utils::l2_squared_distance;

pub enum ReRanker {
    Heap(HeapReRanker),
    Heuristic(HeuristicReRanker),
}

pub fn new_re_ranker(query: &ColRef<f32>, topk: usize, heuristic_rank: bool) -> ReRanker {
    if heuristic_rank {
        ReRanker::Heuristic(HeuristicReRanker::new(query, topk))
    } else {
        ReRanker::Heap(HeapReRanker::new(query, topk))
    }
}

impl ReRanker {
    pub fn rank_batch(
        &mut self,
        rough_distances: &[(f32, u32)],
        base: &MatRef<f32>,
        map_ids: &[u32],
    ) {
        match self {
            ReRanker::Heap(re_ranker) => re_ranker.rank_batch(rough_distances, base, map_ids),
            ReRanker::Heuristic(re_ranker) => re_ranker.rank_batch(rough_distances, base, map_ids),
        }
    }

    pub async fn rank_batch_async(
        &mut self,
        rough_distances: &[(f32, u32)],
        map_ids: &[u32],
        index: u32,
    ) {
        match self {
            ReRanker::Heap(re_ranker) => {
                re_ranker
                    .rank_batch_async(rough_distances, map_ids, index)
                    .await
            }
            ReRanker::Heuristic(re_ranker) => {
                re_ranker
                    .rank_batch_async(rough_distances, map_ids, index)
                    .await
            }
        }
    }

    pub fn get_result(&self) -> Vec<(f32, u32)> {
        match self {
            ReRanker::Heap(re_ranker) => re_ranker.get_result(),
            ReRanker::Heuristic(re_ranker) => re_ranker.get_result(),
        }
    }
}

pub trait ReRankerTrait {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]);
    async fn rank_batch_async(
        &mut self,
        rough_distances: &[(f32, u32)],
        map_ids: &[u32],
        index: u32,
    );
    fn get_result(&self) -> Vec<(f32, u32)>;
}

#[derive(Debug)]
pub struct HeapReRanker {
    threshold: f32,
    topk: usize,
    heap: BinaryHeap<(Ord32, u32)>,
    query: Col<f32>,
}

impl HeapReRanker {
    fn new(query: &ColRef<f32>, topk: usize) -> Self {
        Self {
            threshold: f32::MAX,
            query: query.to_owned(),
            topk,
            heap: BinaryHeap::with_capacity(topk),
        }
    }
}

impl ReRankerTrait for HeapReRanker {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]) {
        let mut precise = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < self.threshold {
                let accurate = l2_squared_distance(&base.col(u as usize), &self.query.as_ref());
                precise += 1;
                if accurate < self.threshold {
                    self.heap.push((accurate.into(), map_ids[u as usize]));
                    if self.heap.len() > self.topk {
                        self.heap.pop();
                    }
                    if self.heap.len() == self.topk {
                        self.threshold = self.heap.peek().unwrap().0.into();
                    }
                }
            }
        }
        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    async fn rank_batch_async(
        &mut self,
        rough_distances: &[(f32, u32)],
        map_ids: &[u32],
        _index: u32,
    ) {
        let mut precise = 0;
        let cache = CACHED_VECTOR.get().expect("cache not initialized");
        // let entry = CACHED_VECTOR
        //     .get()
        //     .expect("cache not initialized")
        //     .get_mat(index)
        //     .await
        //     .expect("failed to get mat");
        // let mat = entry.value().as_ref();
        for (_i, &(rough, u)) in rough_distances.iter().enumerate() {
            if rough < self.threshold {
                // let accurate = l2_squared_distance(&self.query.as_ref(), &mat.col(i));
                let accurate = cache.get_vec_l2_square_distance(u, &self.query.as_ref()).await;
                precise += 1;
                if accurate < self.threshold {
                    self.heap.push((accurate.into(), map_ids[u as usize]));
                    if self.heap.len() > self.topk {
                        self.heap.pop();
                    }
                    if self.heap.len() == self.topk {
                        self.threshold = self.heap.peek().unwrap().0.into();
                    }
                }
            }
        }
        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    fn get_result(&self) -> Vec<(f32, u32)> {
        self.heap.iter().map(|&(a, b)| (a.into(), b)).collect()
    }
}

#[derive(Debug)]
pub struct HeuristicReRanker {
    threshold: f32,
    recent_max_accurate: f32,
    topk: usize,
    array: Vec<(f32, u32)>,
    query: Col<f32>,
    count: usize,
    window_size: usize,
}

impl HeuristicReRanker {
    fn new(query: &ColRef<f32>, topk: usize) -> Self {
        Self {
            threshold: f32::MAX,
            recent_max_accurate: f32::MIN,
            query: query.to_owned(),
            topk,
            array: Vec::with_capacity(topk),
            count: 0,
            window_size: WINDOW_SIZE,
        }
    }
}

impl ReRankerTrait for HeuristicReRanker {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]) {
        let mut precise = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < self.threshold {
                let accurate = l2_squared_distance(&base.col(u as usize), &self.query.as_ref());
                precise += 1;
                if accurate < self.threshold {
                    self.array.push((accurate, map_ids[u as usize]));
                    self.count += 1;
                    self.recent_max_accurate = self.recent_max_accurate.max(accurate);
                    if self.count >= self.window_size {
                        self.threshold = self.recent_max_accurate;
                        self.count = 0;
                        self.recent_max_accurate = f32::MIN;
                    }
                }
            }
        }
        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    async fn rank_batch_async(
        &mut self,
        rough_distances: &[(f32, u32)],
        map_ids: &[u32],
        index: u32,
    ) {
        let mut precise = 0;
        let entry = CACHED_VECTOR
            .get()
            .expect("cache not initialized")
            .get_mat(index)
            .await
            .expect("failed to get mat");
        let mat = entry.value().as_ref();
        for (i, &(rough, u)) in rough_distances.iter().enumerate() {
            if rough < self.threshold {
                let accurate = l2_squared_distance(&self.query.as_ref(), &mat.col(i));
                precise += 1;
                if accurate < self.threshold {
                    self.array.push((accurate, map_ids[u as usize]));
                    self.count += 1;
                    self.recent_max_accurate = self.recent_max_accurate.max(accurate);
                    if self.count >= self.window_size {
                        self.threshold = self.recent_max_accurate;
                        self.count = 0;
                        self.recent_max_accurate = f32::MIN;
                    }
                }
            }
        }
        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    fn get_result(&self) -> Vec<(f32, u32)> {
        let length = self.topk.min(self.array.len());
        let mut res = self.array.clone();
        res.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        res.truncate(length);
        res
    }
}
