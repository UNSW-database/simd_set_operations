#![feature(portable_simd)]

pub mod datafile;
pub mod format;
pub mod generators;
pub mod schema;
// pub mod timer;
pub mod realdata;
pub mod util;

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
    pub fn bytes(&self) -> usize {
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

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Distribution {
    Uniform {},
}

// Data bin configuration
#[derive(Serialize, Deserialize, Debug)]
pub struct DataBinConfig {
    pub datatype: Datatype,
    pub max_value: u64,
    pub long_length: usize,
    pub short_length: usize,
    pub intersection_length: usize,
    pub distribution: Distribution,
    pub seed: u64,
    pub trials: u64,
    // byte offset in .data file
    pub offset: usize,
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
