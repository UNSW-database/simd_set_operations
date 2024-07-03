#![feature(portable_simd)]

// pub mod datafile;
// pub mod format;
// pub mod generators;
// pub mod schema;
// pub mod timer;
// pub mod realdata;
pub mod util;
pub mod algorithms;

use std::path::PathBuf;

pub fn fmt_open_err(e: impl ToString, path: &PathBuf) -> String {
    format!("Unable to open {}: {}", path_str(path), e.to_string())
}

pub fn path_str(path: &PathBuf) -> &str {
    path.to_str().unwrap_or("<unknown path>")
}

use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum Datatype {
    U32,
    U64,
    I32,
    I64,
}

impl Datatype {
    pub fn bytes(&self) -> u64 {
        match self {
            Datatype::U32 => 4,
            Datatype::U64 => 8,
            Datatype::I32 => 4,
            Datatype::I64 => 8,
        }
    }

    pub fn max(&self) -> u64 {
        match self {
            Datatype::U32 => u32::MAX as u64,
            Datatype::U64 => u64::MAX,
            Datatype::I32 => i32::MAX as u64,
            Datatype::I64 => i64::MAX as u64,
        }
    }
}

pub type DataSetDescription = Vec<DataBinDescription>;

// Data bin configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataBinDescription {
    pub datatype: Datatype,
    pub max_value: u64,
    pub lengths: DataBinLengthsEnum,
    pub distribution: DataDistribution,
    pub seed: u64,
    pub trials: u64,
    // byte offset in .data file
    pub offset: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum DataBinLengthsEnum {
    Pair(DataBinLengths),
    Sample(Vec<DataBinLengths>),
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataBinLengths {
    pub set_lengths: Vec<u64>,
    pub intersection_length: u64
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataDistribution {
    Uniform {},
}


/*
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
*/
