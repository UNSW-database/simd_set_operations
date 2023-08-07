#![feature(portable_simd)]

pub mod schema;
pub mod generators;
pub mod datafile;
pub mod format;
pub mod harness;
pub mod webdocs;

use std::{
    ops::RangeInclusive,
    path::PathBuf,
    iter::StepBy,
    collections::HashMap
};
use schema::{SyntheticDataset, Parameter, IntersectionInfo, AlgorithmVec, DatasetInfo};

pub fn fmt_open_err(e: impl ToString, path: &PathBuf) -> String {
    format!("unable to open {}: {}", path_str(path), e.to_string())
}

pub fn path_str(path: &PathBuf) -> &str {
    path.to_str().unwrap_or("<unknown path>")
}

pub fn xvalues(info: &DatasetInfo) -> StepBy<RangeInclusive<u32>> {
    match &info.dataset_type {
        schema::DatasetType::Synthetic(s) => xvalues_synthetic(s),
        schema::DatasetType::Real(r) => (r.set_count_start..=r.set_count_end).step_by(1),
    }
}

pub fn xvalues_synthetic(info: &SyntheticDataset) -> StepBy<RangeInclusive<u32>> {
    let begin = match info.vary {
        Parameter::Selectivity => info.intersection.selectivity,
        Parameter::Density     => info.intersection.density,
        Parameter::Size        => info.intersection.max_len,
        Parameter::Skew        => info.intersection.skewness_factor,
        Parameter::SetCount    => info.intersection.set_count,
    };

    (begin..=info.to).step_by(info.step as usize)
}

pub fn props_at_x(info: &SyntheticDataset, x: u32) -> IntersectionInfo {
    let mut props = info.intersection.clone();
    let prop = match info.vary {
        Parameter::Selectivity => &mut props.selectivity,
        Parameter::Density     => &mut props.density,
        Parameter::Size        => &mut props.max_len,
        Parameter::Skew        => &mut props.skewness_factor,
        Parameter::SetCount    => &mut props.set_count,
    };
    *prop = x;

    props
}

pub fn get_algorithms<'a>(
    algorithm_sets: &'a HashMap<String, AlgorithmVec>,
    id: &str) -> Result<&'a AlgorithmVec, String>
{
    algorithm_sets.get(id)
        .ok_or_else(|| format!("algorithm set {} not found", id))
}
