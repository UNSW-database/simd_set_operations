pub mod schema;
pub mod generators;
pub mod datafile;

use std::{ops::RangeInclusive, path::PathBuf, iter::StepBy};
use schema::{DatasetInfo, Parameter};

pub fn fmt_open_err(e: impl ToString, path: &PathBuf) -> String {
    format!("unable to open {}: {}", path_str(path), e.to_string())
}

pub fn path_str(path: &PathBuf) -> &str {
    path.to_str().unwrap_or("<unknown path>")
}

pub fn xvalues(info: &DatasetInfo) -> StepBy<RangeInclusive<u32>> {
    let begin = match info.vary {
        Parameter::Selectivity => info.intersection.selectivity,
        Parameter::Density => info.intersection.density,
        Parameter::Size => info.intersection.max_len,
        Parameter::Skew => info.intersection.skewness_factor,
        Parameter::SetCount => info.intersection.set_count,
    };

    (begin..=info.to).step_by(info.step as usize)
}
