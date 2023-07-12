pub mod schema;
pub mod generators;
use std::{collections::BTreeSet, ops::Range};
use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, thread_rng};
use schema::*;

#[deprecated]
pub fn uniform_sorted_set(range: Range<i32>, cardinality: usize) -> Vec<i32> {
    let rng = &mut thread_rng();

    let density = cardinality as f64 / range.len() as f64;
    if density < 0.01 {
        let dist = Uniform::from(range);

        let mut set: BTreeSet<i32> = BTreeSet::new();
        while set.len() < cardinality {
            set.insert(dist.sample(rng));
        }
        set.iter().copied().collect()
    } else {
        let mut everything: Vec<i32> = range.collect();
        everything.shuffle(rng);

        let mut result = Vec::from(&everything[0..cardinality]);
        result.sort();
        result
    }
}

pub fn dataset_id(info: &TwoSetDatasetInfo) -> String {
    let p = &info.props;
    match info.vary {
        Parameter::Density =>
            format!("den{}-{}_sel{}_size{}_skew{}",
                p.density, info.to, p.selectivity, p.size, p.skew),
        Parameter::Selectivity =>
            format!("sel{}-{}_den{}_size{}_skew{}",
                p.selectivity, info.to, p.density, p.size, p.skew),
        Parameter::Size =>
            format!("size{}-{}_den{}_sel{}_skew{}",
                p.size, info.to, p.density, p.selectivity, p.skew),
        Parameter::Skew =>
            format!("skew{}-{}_den{}_sel{}_size{}",
                p.skew, info.to, p.density, p.selectivity, p.size),
    }
}
