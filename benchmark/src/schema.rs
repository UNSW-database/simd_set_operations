use std::collections::HashMap;

use serde::{Serialize, Deserialize};

pub type DatasetId = String;
pub type AlgorithmId = String;

// An integer i represents the percentage value i/MAX_PERCENT_F (from 0.0 to 1.0)
pub const PERCENT: u32 = 10000;
pub const PERCENT_F: f64 = PERCENT as f64;

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub experiment: Vec<ExperimentEntry>,
    pub dataset: Vec<DatasetInfo>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExperimentEntry {
    pub name: String,
    pub title: String,
    pub dataset: DatasetId,
    pub algorithms: Vec<AlgorithmId>,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DatasetInfo {
    pub name: DatasetId,
    pub vary: Parameter,
    pub to: u32,
    pub step: u32,
    pub set_count: usize,
    pub gen_count: usize,
    #[serde(flatten)]
    pub props: SetInfo,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SetInfo {
    pub density: u32,
    pub selectivity: u32,
    pub max_len: u32,
    pub skewness_factor: u32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(rename_all = "lowercase")]
pub enum Parameter {
    Density,
    Selectivity,
    Size,
    Skew,
}

pub type SetPair = (Vec<i32>, Vec<i32>);

#[derive(Serialize, Deserialize, Debug)]
pub struct Results {
    pub experiments: Vec<ExperimentEntry>,
    pub datasets: HashMap<DatasetId, DatasetResults>,
}

pub type AlgorithmResults = HashMap<AlgorithmId, Vec<ResultRun>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetResults {
    pub info: DatasetInfo,
    pub algos: AlgorithmResults
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResultRun {
    pub x: u32,
    // Nanoseconds
    pub times: Vec<u64>,
}
