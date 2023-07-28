pub mod schema;
pub mod generators;
pub mod datafile;

use std::{ops::RangeInclusive, path::PathBuf, iter::StepBy};
use schema::{TwoSetDatasetInfo, Parameter};

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
