//! Rank the rough estimate distances with accurate distances.

use std::collections::BinaryHeap;

use faer::{Col, ColRef, MatRef};

use crate::consts::WINDOW_SIZE;
use crate::metrics::METRICS;
use crate::ord32::Ord32;
use crate::utils::l2_squared_distance;

pub enum ReRanker {
    Heap(HeapReRanker),
    Heuristic(HeuristicReRanker),
    DoubleHeap(DoubleHeapReRanker),
}

pub fn new_re_ranker(query: &ColRef<f32>, topk: usize, heuristic_rank: bool) -> ReRanker {
    if heuristic_rank {
        // ReRanker::Heuristic(HeuristicReRanker::new(query, topk))
        ReRanker::DoubleHeap(DoubleHeapReRanker::new(query, topk))
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
            ReRanker::DoubleHeap(re_ranker) => re_ranker.rank_batch(rough_distances, base, map_ids),
        }
    }

    pub fn get_result(&mut self, base: &MatRef<f32>, map_ids: &[u32]) -> Vec<(f32, u32)> {
        match self {
            ReRanker::Heap(re_ranker) => re_ranker.get_result(),
            ReRanker::Heuristic(re_ranker) => re_ranker.get_result(),
            ReRanker::DoubleHeap(re_ranker) => re_ranker.get_result(base, map_ids),
        }
    }
}

pub trait ReRankerTrait {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]);
    fn get_result(&self) -> Vec<(f32, u32)>;
}

#[derive(Debug)]
pub struct HeapReRanker {
    threshold: f32,
    topk: usize,
    heap: BinaryHeap<(Ord32, AlwaysEqual<u32>)>,
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
                    self.heap
                        .push((accurate.into(), AlwaysEqual(map_ids[u as usize])));
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
        self.heap
            .iter()
            .map(|&(a, AlwaysEqual(b))| (a.into(), b))
            .collect()
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

    fn get_result(&self) -> Vec<(f32, u32)> {
        let length = self.topk.min(self.array.len());
        let mut res = self.array.clone();
        res.select_nth_unstable_by(length - 1, |a, b| a.0.total_cmp(&b.0));
        res.truncate(length);
        res
    }
}

use std::cmp::Reverse;

use crate::ord32::AlwaysEqual;

#[derive(Debug)]
pub struct DoubleHeapReRanker {
    topk: usize,
    heap: BinaryHeap<(Reverse<Ord32>, AlwaysEqual<u32>)>,
    cache: BinaryHeap<(Reverse<Ord32>, AlwaysEqual<u32>)>,
    query: Col<f32>,
}

impl DoubleHeapReRanker {
    fn new(query: &ColRef<f32>, topk: usize) -> Self {
        Self {
            query: query.to_owned(),
            topk,
            heap: BinaryHeap::with_capacity(topk),
            cache: BinaryHeap::with_capacity(topk),
        }
    }
}

impl DoubleHeapReRanker {
    fn rank_batch(&mut self, rough_distances: &[(f32, u32)], base: &MatRef<f32>, map_ids: &[u32]) {
        for &(dist, index) in rough_distances {
            self.heap.push((Reverse(dist.into()), AlwaysEqual(index)));
        }
        METRICS.add_rough_count(rough_distances.len() as u64);
    }

    fn get_result(&mut self, base: &MatRef<f32>, map_ids: &[u32]) -> Vec<(f32, u32)> {
        let mut precise = 0;
        let length = self.topk.min(self.heap.len());
        let mut vec = Vec::with_capacity(length);
        while vec.len() < length {
            while !self.heap.is_empty()
                && self.heap.peek().map(|x| x.0) > self.cache.peek().map(|x| x.0)
            {
                let (_, AlwaysEqual(u)) = self.heap.pop().unwrap();
                let accurate = l2_squared_distance(&base.col(u as usize), &self.query.as_ref());
                precise += 1;
                self.cache
                    .push((Reverse(accurate.into()), AlwaysEqual(map_ids[u as usize])));
            }
            if let Some((Reverse(dist), AlwaysEqual(index))) = self.cache.pop() {
                vec.push((dist.into(), index));
            } else {
                break;
            }
        }
        METRICS.add_precise_count(precise);
        vec
    }
}
