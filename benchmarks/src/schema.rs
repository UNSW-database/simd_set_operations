use std::collections::HashMap;

use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub experiment: Vec<ExperimentEntry>,
    pub dataset: Vec<DatasetInfo>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ExperimentEntry {
    pub name: String,
    pub dataset: String,
    pub algorithms: Vec<String>,
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
    pub name: String,
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
    datasets: HashMap<String, ResultDataset>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResultDataset {
    info: TwoSetDatasetInfo,
    algos: HashMap<String, Vec<ResultRun>>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct ResultRun {
    x: u32,
    // Nanoseconds
    times: Vec<u64>,
}
