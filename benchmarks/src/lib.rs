pub mod schema;
pub mod generators;
use std::{collections::BTreeSet, ops::Range};
use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, thread_rng};

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
