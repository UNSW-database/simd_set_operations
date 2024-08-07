use std::collections::HashMap;

use serde::{Serialize, Deserialize};

pub type DatasetId = String;
pub type AlgorithmId = String;
pub type AlgorithmVec = Vec<AlgorithmId>;

// An integer i represents the percentage value i/MAX_PERCENT_F (from 0.0 to 1.0)
pub const PERCENT: u32 = 1000;
pub const PERCENT_F: f64 = PERCENT as f64;

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub experiment: Vec<ExperimentEntry>,
    pub dataset: Vec<DatasetInfo>,
    pub algorithm_sets: HashMap<String, AlgorithmVec>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ExperimentEntry {
    pub name: String,
    pub dataset: DatasetId,
    #[serde(flatten)]
    pub algorithms: Algorithms,
    pub relative_to: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(rename_all = "snake_case")]
pub enum Algorithms {
    Algorithms(Vec<String>),
    AlgorithmSet(String),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct DatasetInfo {
    pub name: String,
    #[serde(flatten)]
    pub dataset_type: DatasetType,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "snake_case", tag = "type")]
pub enum DatasetType {
    Synthetic(SyntheticDataset),
    Real(RealDataset),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SyntheticDataset {
    pub vary: Parameter,
    pub to: u32,
    pub step: u32,
    pub gen_count: usize,
    #[serde(flatten)]
    pub intersection: IntersectionInfo,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct IntersectionInfo {
    pub set_count: u32,
    pub density: u32,
    pub selectivity: u32,
    pub max_len: u32,
    pub skewness_factor: u32,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
pub enum Parameter {
    Density,
    Selectivity,
    Size,
    Skew,
    SetCount,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct RealDataset {
    pub source: String,
    pub gen_count: usize,
    pub set_count_start: u32,
    pub set_count_end: u32,
}

pub type SetPair = (Vec<i32>, Vec<i32>);

#[derive(Serialize, Deserialize, Debug)]
pub struct Results {
    pub experiments: Vec<ExperimentEntry>,
    pub datasets: HashMap<DatasetId, DatasetResults>,
    pub algorithm_sets: HashMap<String, AlgorithmVec>,
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
    pub l1d: CacheRun,
    pub l1i: CacheRun,
    pub ll: CacheRun,
    pub branches: Option<Vec<u64>>,
    pub branch_misses: Option<Vec<u64>>,
    pub cpu_stalled_front: Option<Vec<u64>>,
    pub cpu_stalled_back: Option<Vec<u64>>,
    pub instructions: Option<Vec<u64>>,
    pub cpu_cycles: Option<Vec<u64>>,
    pub cpu_cycles_ref: Option<Vec<u64>>,

    pub bytes: Vec<u64>,
}

// Store columnar in JSON
#[derive(Serialize, Deserialize, Debug, Default)]
pub struct CacheRun {
    pub rd_access: Option<Vec<u64>>,
    pub rd_miss: Option<Vec<u64>>,
    pub wr_access: Option<Vec<u64>>,
    pub wr_miss: Option<Vec<u64>>,
}

