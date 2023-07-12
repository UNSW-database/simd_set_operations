use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    pub dataset: Vec<DatasetInfo>
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

#[derive(Serialize, Deserialize, Debug)]
pub struct TwoSetFile {
    pub info: TwoSetDatasetInfo,
    pub xvalues: Vec<TwoSetInput>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TwoSetInput {
    pub x: u32,
    pub pairs: Vec<(Vec<i32>, Vec<i32>)>,
}
