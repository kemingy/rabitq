//! Rank the rough estimate distances with accurate distances.

use std::collections::BinaryHeap;

use faer::MatRef;

use crate::consts::WINDOW_SIZE;
use crate::metrics::METRICS;
use crate::ord32::{AlwaysEqual, Ord32};
use crate::utils::l2_squared_distance;

/// ReRanker enum.
pub enum ReRanker<'a> {
    /// ReRanker with heap.
    Heap(HeapReRanker<'a>),
    /// ReRanker with heuristic.
    Heuristic(HeuristicReRanker<'a>),
}

/// Create a new re-ranker.
pub fn new_re_ranker(query: &[f32], topk: usize, heuristic_rank: bool) -> ReRanker {
    if heuristic_rank {
        ReRanker::Heuristic(HeuristicReRanker::new(query, topk))
    } else {
        ReRanker::Heap(HeapReRanker::new(query, topk))
    }
}

impl ReRanker<'_> {
    /// Rank a batch of items.
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

    /// Get the re-ranked result.
    pub fn get_result(&self) -> Vec<(f32, u32)> {
        match self {
            ReRanker::Heap(re_ranker) => re_ranker.get_result(),
            ReRanker::Heuristic(re_ranker) => re_ranker.get_result(),
        }
    }
}

/// Ranker trait.
pub trait ReRankerTrait {
    /// Rank a batch of items.
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]);
    /// Get the re-ranked result.
    fn get_result(&self) -> Vec<(f32, u32)>;
}

/// Rank with heap.
#[derive(Debug)]
pub struct HeapReRanker<'a> {
    threshold: f32,
    topk: usize,
    heap: BinaryHeap<(Ord32, AlwaysEqual<u32>)>,
    query: &'a [f32],
}

impl<'a> HeapReRanker<'a> {
    fn new(query: &'a [f32], topk: usize) -> Self {
        Self {
            threshold: f32::MAX,
            query,
            topk,
            heap: BinaryHeap::with_capacity(topk),
        }
    }
}

impl ReRankerTrait for HeapReRanker<'_> {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]) {
        let mut precise = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < self.threshold {
                let accurate = l2_squared_distance(
                    base.col(u as usize)
                        .try_as_slice()
                        .expect("failed to get base slice"),
                    self.query,
                );
                precise += 1;
                if accurate < self.threshold {
                    self.heap
                        .push((accurate.into(), AlwaysEqual(map_ids[u as usize])));
                    if self.heap.len() > self.topk {
                        self.heap.pop();
                    }
                    if self.heap.len() == self.topk {
                        self.threshold = self.heap.peek().expect("failed to peek heap").0.into();
                    }
                }
            }
        }
        METRICS.add_precise_count(precise);
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    fn get_result(&self) -> Vec<(f32, u32)> {
        self.heap
            .iter()
            .map(|&(a, AlwaysEqual(b))| (a.into(), b))
            .collect()
    }
}

/// Rank in a heuristic way.
#[derive(Debug)]
pub struct HeuristicReRanker<'a> {
    threshold: f32,
    recent_max_accurate: f32,
    topk: usize,
    array: Vec<(f32, u32)>,
    query: &'a [f32],
    count: usize,
    window_size: usize,
}

impl<'a> HeuristicReRanker<'a> {
    fn new(query: &'a [f32], topk: usize) -> Self {
        Self {
            threshold: f32::MAX,
            recent_max_accurate: f32::MIN,
            query,
            topk,
            array: Vec::with_capacity(topk),
            count: 0,
            window_size: WINDOW_SIZE,
        }
    }
}

impl ReRankerTrait for HeuristicReRanker<'_> {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]) {
        let mut precise = 0;
        for &(rough, u) in rough_distances.iter() {
            if rough < self.threshold {
                let accurate = l2_squared_distance(
                    base.col(u as usize)
                        .try_as_slice()
                        .expect("failed to get base slice"),
                    self.query,
                );
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
