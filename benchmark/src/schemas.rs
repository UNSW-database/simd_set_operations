pub mod dataset {
    use serde::{Serialize, Deserialize};
    use crate::{Datatype, DataDistribution};
    use std::collections::BTreeMap;

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct DatasetDescription {
        pub seed        : u64,
        pub kset        : bool,
        pub byte_length : u64,
        pub databins    : Vec<DatabinDescription>,
        pub parameters  : DatabinParameters,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum DatabinParameters {
        Pair(PairParams),
        Sample(SampleParams),
    }

    type IndexMap = BTreeMap<String, Vec<u64>>;

    #[derive(Serialize, Deserialize, Default, Debug, Clone)]
    pub struct PairParams {
        pub skew         : IndexMap,
        pub density      : IndexMap,
        pub datatype     : IndexMap,
        pub selectivity  : IndexMap,
        pub max_set_size : IndexMap,
        pub distribution : IndexMap, 
    }

    #[derive(Serialize, Deserialize, Default, Debug, Clone)]
    pub struct SampleParams {
        pub skew                : IndexMap,
        pub density             : IndexMap,
        pub datatype            : IndexMap,
        pub selectivity         : IndexMap,
        pub max_set_size        : IndexMap,
        pub data_distribution   : IndexMap, 
        pub query_size          : IndexMap,
        pub query_distribution  : IndexMap,
        pub corpus_size         : IndexMap,
        pub corpus_distribution : IndexMap,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct DatabinDescription {
        pub datatype     : Datatype,
        pub max_value    : u64,
        pub distribution : DataDistribution,
        // RNG seed used for data generation
        pub seed         : u64,
        // byte offset and length in .data file
        pub byte_offset  : u64,
        pub byte_length  : u64,
        pub trials       : Vec<TrialDescription>,
    }

    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct TrialDescription {
        pub set_lengths         : Vec<u64>,
        pub intersection_length : u64,
        pub byte_offset         : u64,
        pub byte_length         : u64,
    }
}

pub mod experiment {
    use serde::{Serialize, Deserialize};
    use std::collections::HashMap;

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Config {
        pub algorithm_set : HashMap<String, AlgorithmSet>,
        pub experiment    : HashMap<String, ExperimentConfig>,
    }

    #[derive(Serialize, Deserialize, Debug, Default)]
    #[serde(default)]
    pub struct AlgorithmSet {
        pub twoset            : Vec<String>,
        pub twoset_to_kset    : Vec<String>,
        pub fsearch           : Vec<String>,
        pub fsearch_to_twoset : Vec<String>,
        pub fsearch_to_kset   : Vec<String>,
        pub dummy             : Vec<usize>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct ExperimentConfig {
        pub repeats        : u64,
        pub cache_warmups  : u64,
        pub algorithm_sets : Vec<String>,
        pub reference      : String,
        pub rng_seed       : u64,
    }
}

pub mod results {
    use serde::{Serialize, Deserialize};

    #[derive(Serialize, Deserialize, Debug)]
    pub struct ExperimentResult {
        pub experiment : String,            // experiment name
        pub algorithms : Vec<String>,       // list of algorithm names in recording order
        pub repeats    : Vec<RepeatResult>, // results for each repeat in chronological order
        pub note       : String,            // note for this experiment run
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct RepeatResult {
        pub databins : Vec<DatabinResult>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct DatabinResult {
        pub trials : Vec<TrialResult>,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct TrialResult {
        pub order           : Vec<u64>,
        pub cycles          : Vec<u64>,
        pub ll_cache_misses : Vec<u64>,
        pub branch_misses   : Vec<u64>,
    }
}

pub mod generator {
    use serde::{Serialize, Deserialize};
    use crate::{Datatype, DataDistribution, QueryDistribution, CorpusDistribution};

    #[derive(Serialize, Deserialize, Debug)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Config {
        Pair(Pair),
        Sample(Sample),
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Pair {
        pub datatype     : VecParamOpt<Datatype>,
        pub max_set_size : NumParamOpt<u64>,
        pub skew         : NumParamOpt<f64>,
        pub selectivity  : NumParamOpt<f64>,
        pub density      : NumParamOpt<f64>,
        pub distribution : VecParamOpt<DataDistribution>,
        pub trials       : u64,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Sample {
        pub datatype     : VecParamOpt<Datatype>,
        pub trials       : u64,
        pub distribution : VecParamOpt<DataDistribution>,
        pub query        : Query,
        pub corpus       : Corpus,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Query {
        pub size         : NumParamOpt<u64>,
        pub distribution : VecParamOpt<QueryDistribution>,
        pub selectivity  : NumParamOpt<f64>,
        pub samples      : u64,
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct Corpus {
        pub size         : NumParamOpt<u64>,
        pub distribution : VecParamOpt<CorpusDistribution>,
        pub max_set_size : NumParamOpt<u64>,
        pub skew         : NumParamOpt<f64>,
        pub density      : NumParamOpt<f64>,
    }
   
    pub type VecParamOpt<T> = OptParameter<T, Vec<T>>;
    pub type NumParamOpt<T> = OptParameter<T, NumericalParameter>;

    #[derive(Serialize, Deserialize, Debug)]
    #[serde(untagged)]
    pub enum OptParameter<T, U> {
        Fixed(T),
        Varying(U),
    }

    #[derive(Serialize, Deserialize, Debug)]
    pub struct NumericalParameter {
        pub from : f64,
        pub to   : f64,
        #[serde(flatten)]
        pub step : StepType,
        pub mode : StepMode,
    }

    #[derive(Serialize, Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum StepType {
        Step(f64),
        Steps(u64),
    }

    #[derive(Serialize, Deserialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum StepMode {
        Linear,
        Log,
    }
}
