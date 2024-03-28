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
    format!("unable to open {}: {}", path_str(path), e.to_string())
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
    pub long_length: u64,
    pub short_length: u64,
    pub trials: u64,
    pub intersection_length: u64,
    // minimum value is assumed to be 0 always
    pub max_value: u64,
    pub distribution: Distribution,
    // byte offset in .data file
    pub offset: u64,
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
