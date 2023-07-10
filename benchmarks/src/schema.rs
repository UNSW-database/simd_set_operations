use serde::Deserialize;

#[derive(Deserialize, Debug)]
pub struct Experiments {
    dataset: Vec<Dataset>
}

#[derive(Deserialize, Debug)]
#[serde(tag = "type")]
#[serde(rename_all = "lowercase")]
pub enum Dataset {
    #[serde(alias = "2set")]
    TwoSet(TwoSetDataset),
    KSet(KSetDataset),
}

#[derive(Deserialize, Debug)]
pub struct TwoSetDataset {
    distribution: Distribution,
    vary: Vary,
    to: u32,
    density: u32,
    skew: u32,
    selectivity: u32,
    size: usize,
    count: usize,
}

#[derive(Deserialize, Debug)]
pub struct KSetDataset {}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Distribution {
    Uniform
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "lowercase")]
pub enum Vary {
    Selectivity,
    Density,
    Size,
    Skew,
}
