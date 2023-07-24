use std::collections::HashMap;

use serde::{Serialize, Deserialize};

pub type DatasetId = String;
pub type AlgorithmId = String;

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub experiment: Vec<ExperimentEntry>,
    pub dataset: Vec<DatasetInfo>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExperimentEntry {
    pub name: String,
    pub dataset: DatasetId,
    pub algorithms: Vec<AlgorithmId>,
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum DatasetInfo {
    #[serde(alias = "2set")]
    TwoSet(TwoSetDatasetInfo),
    KSet(KSetDatasetInfo),
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct TwoSetDatasetInfo {
    pub name: DatasetId,
    pub vary: Parameter,
    pub to: u32,
    pub step: u32,
    pub count: usize,
    #[serde(flatten)]
    pub props: SetInfo,
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
pub struct SetInfo {
    pub density: u32,
    pub selectivity: u32,
    pub size: u32,
    pub skew: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KSetDatasetInfo {}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone)]
#[serde(rename_all = "lowercase")]
pub enum Distribution {
    Uniform
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
    pub datasets: HashMap<DatasetId, DatasetResults>,
}

pub type AlgorithmResults = HashMap<AlgorithmId, Vec<ResultRun>>;

#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetResults {
    pub info: TwoSetDatasetInfo,
    pub algos: AlgorithmResults
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResultRun {
    pub x: u32,
    // Nanoseconds
    pub times: Vec<u64>,
}
