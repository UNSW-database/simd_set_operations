use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct Experiment {
    dataset: Vec<DatasetInfo>
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum DatasetInfo {
    #[serde(alias = "2set")]
    TwoSet(TwoSetDatasetInfo),
    KSet(KSetDatasetInfo),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TwoSetDatasetInfo {
    vary: Parameter,
    to: u32,
    count: usize,
    #[serde(flatten)]
    props: SetInfo,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SetInfo {
    pub density: u32,
    pub selectivity: u32,
    pub skew: u32,
    pub size: u32,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct KSetDatasetInfo {}

#[derive(Serialize, Deserialize, Debug, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum Distribution {
    Uniform
}

#[derive(Serialize, Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Parameter {
    Selectivity,
    Density,
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
    pub x: i32,
    pub pairs: Vec<(Vec<i32>, Vec<i32>)>,
}
