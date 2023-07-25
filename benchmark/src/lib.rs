pub mod schema;
pub mod generators;
pub mod datafile;

use std::{collections::BTreeSet, ops::{Range, RangeInclusive}, path::PathBuf, iter::StepBy};
use rand::{distributions::Uniform, prelude::Distribution, seq::SliceRandom, thread_rng};
use schema::{TwoSetDatasetInfo, Parameter};

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

pub fn fmt_open_err(e: impl ToString, path: &PathBuf) -> String {
    format!("unable to open {}: {}", path_str(path), e.to_string())
}

pub fn path_str(path: &PathBuf) -> &str {
    path.to_str().unwrap_or("<unknown path>")
}

pub fn xvalues(info: &TwoSetDatasetInfo) -> StepBy<RangeInclusive<u32>> {
    let begin = match info.vary {
        Parameter::Selectivity => info.props.selectivity,
        Parameter::Density => info.props.density,
        Parameter::Size => info.props.size,
        Parameter::Skew => info.props.skew,
    };

    (begin..=info.to).step_by(info.step as usize)
}
