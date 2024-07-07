use benchmark::{
    fmt_open_err, path_str, util::sample_distribution_unique, DataBinDescription, DataBinLengths, DataBinLengthsEnum, DataDistribution, DataSetDescription, Datatype
};
use clap::Parser;
use colored::*;
use std::{
    fs::{self, File},
    iter,
    path::PathBuf,
};

use rand::{Rng, SeedableRng};
use serde::Deserialize;
use zipf::ZipfDistribution;

// Schema for TOML configuration file
#[derive(Deserialize, Debug)]
#[serde(tag = "type", rename_all = "snake_case")]
enum Config {
    Pair(Pair),
    Sample(Sample),
}

#[derive(Deserialize, Debug)]
struct Pair {
    datatype: VecParamOpt<Datatype>,
    fixed_size: FixedSize,
    skew: NumParamOpt<f64>,
    selectivity: NumParamOpt<f64>,
    density: NumParamOpt<f64>,
    distribution: VecParamOpt<DataDistribution>,
    trials: NumParamOpt<u64>,
}

#[derive(Deserialize, Debug)]
struct Sample {
    datatype: VecParamOpt<Datatype>,
    trials: NumParamOpt<u64>,
    distribution: VecParamOpt<DataDistribution>,
    query: Query,
    corpus: Corpus,
}

#[derive(Deserialize, Debug)]
struct Query {
    size: NumParamOpt<u64>,
    distribution: VecParamOpt<QueryDistribution>,
    selectivity: NumParamOpt<f64>,
    samples: NumParamOpt<u64>,
}

#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
enum QueryDistribution {
    Zipf {},
}

#[derive(Deserialize, Debug)]
struct Corpus {
    size: NumParamOpt<u64>,
    distribution: VecParamOpt<CorpusDistribution>,
    fixed_size: FixedSize,
    skew: NumParamOpt<f64>,
    density: NumParamOpt<f64>,
}

#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CorpusDistribution {
    Zipf {},
}

#[derive(Deserialize, Debug)]
struct FixedSize {
    fixed: FixedSizeRef,
    size: NumParamOpt<u64>,
}

#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum FixedSizeRef {
    Longest,
    Shortest,
}

type VecParamOpt<T> = OptParameter<T, Vec<T>>;
type NumParamOpt<T> = OptParameter<T, NumericalParameter>;

#[derive(Deserialize, Debug)]
#[serde(untagged)]
enum OptParameter<T, U> {
    Fixed(T),
    Varying(U),
}

#[derive(Deserialize, Debug)]
struct NumericalParameter {
    from: f64,
    to: f64,
    #[serde(flatten)]
    step: StepType,
    mode: StepMode,
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum StepType {
    Step(f64),
    Steps(u64),
}

#[derive(Deserialize, Debug)]
#[serde(rename_all = "snake_case")]
enum StepMode {
    Linear,
    Log,
}

// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    config: PathBuf,
    #[arg(long, default_value = "datasets/")]
    outdir: PathBuf,
    #[arg(long)]
    seed: Option<u64>,
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = generate(&cli) {
        println!("{}", err.red().bold());
    } else {
        println!("{}", "DONE".green().bold());
    }
}

fn generate(cli: &Cli) -> Result<(), String> {
    // Set up seed generation
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(match cli.seed {
        Some(seed) => seed,
        None => rand::random(),
    });

    // Read dataset configuration
    let config_string =
        fs::read_to_string(&cli.config).map_err(|e| fmt_open_err(e, &cli.config))?;
    let config: Config = toml::from_str(&config_string)
        .map_err(|e| format!("Invalid toml file {}: {}", path_str(&cli.config), e))?;

    let mut dataset_description: DataSetDescription = Vec::new();
    let mut offset = 0u64;

    match config {
        Config::Pair(pair) => {
            validate_param_u64(&pair.fixed_size.size, "fixed_size.size", 1)?;
            validate_param_u64(&pair.trials, "trials", 1)?;
            validate_param_normalized(&pair.selectivity, "selectivity")?;
            validate_param_normalized(&pair.skew, "skew")?;
            validate_param_normalized(&pair.density, "density")?;

            for datatype in pair.datatype.param_range() {
                for fixed_size in pair.fixed_size.size.param_range() {
                    for skew in pair.skew.param_range() {
                        for selectivity in pair.selectivity.param_range() {
                            for density in pair.density.param_range() {
                                for distribution in pair.distribution.param_range() {
                                    for trials in pair.trials.param_range() {
                                        let seed: u64 = rng.gen();
                                        dataset_description.push(statistics_to_description_2set(
                                            datatype,
                                            fixed_size,
                                            pair.fixed_size.fixed,
                                            trials,
                                            selectivity,
                                            skew,
                                            density,
                                            distribution,
                                            seed,
                                            &mut offset,
                                        )?);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        Config::Sample(sample) => {
            validate_param_u64(&sample.trials, "trials", 1)?;
            validate_param_u64(&sample.query.size, "query.size", 2)?;
            validate_param_u64(&sample.query.samples, "query.samples", 1)?;
            validate_param_u64(&sample.corpus.size, "corpus.size", 2)?;
            validate_param_u64(&sample.corpus.fixed_size.size, "corpus.fixed_size.size", 1)?;
            validate_param_normalized(&sample.query.selectivity, "query.selectivity")?;
            validate_param_normalized(&sample.corpus.skew, "corpus.skew")?;
            validate_param_normalized(&sample.corpus.density, "corpus.density")?;

            for datatype in sample.datatype.param_range() {
                for trials in sample.trials.param_range() {
                    for data_distribution in sample.distribution.param_range() {
                        for query_size in sample.query.size.param_range() {
                            for query_distribution in sample.query.distribution.param_range() {
                                for query_selectivity in sample.query.selectivity.param_range() {
                                    for samples in sample.query.samples.param_range() {
                                        for corpus_size in sample.corpus.size.param_range() {
                                            for corpus_distribution in
                                                sample.corpus.distribution.param_range()
                                            {
                                                for fixed_size in
                                                    sample.corpus.fixed_size.size.param_range()
                                                {
                                                    for corpus_skew in
                                                        sample.corpus.skew.param_range()
                                                    {
                                                        for corpus_density in
                                                            sample.corpus.density.param_range()
                                                        {
                                                            let seed: u64 = rng.gen();
                                                            dataset_description.push(
                                                                statistics_to_description_kset(
                                                                    datatype,
                                                                    fixed_size,
                                                                    sample.corpus.fixed_size.fixed,
                                                                    trials,
                                                                    query_selectivity,
                                                                    corpus_skew,
                                                                    corpus_density,
                                                                    data_distribution,
                                                                    seed,
                                                                    &mut offset,
                                                                    query_size,
                                                                    corpus_size,
                                                                    corpus_distribution,
                                                                    query_distribution,
                                                                    samples,
                                                                    &mut rng,
                                                                )?,
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Write dataset description
    let desc_path = (&cli.outdir).join(&cli.config.with_extension("json"));
    let desc_file = File::create(&desc_path).map_err(|e| {
        format!(
            "Failed to open file {}:\n{}",
            path_str(&desc_path),
            e.to_string()
        )
    })?;
    serde_json::to_writer(desc_file, &dataset_description).map_err(|e| e.to_string())?;

    Ok(())
}

fn validate_param_u64(param: &NumParamOpt<u64>, name: &str, min: u64) -> Result<(), String> {
    match param {
        OptParameter::Fixed(v) => {
            if *v < min {
                return Err(format!(
                    "Invalid paramater ({}): value must be greater than {}.",
                    name, min,
                ));
            }
        }
        OptParameter::Varying(p) => {
            if p.from < min as f64 {
                return Err(format!(
                    "Invalid paramater ({}): 'from' must be greater than {}.",
                    name, min,
                ));
            }
            if p.from >= p.to {
                return Err(format!(
                    "Invalid paramater ({}): 'from' must be less than 'to'.",
                    name
                ));
            }
            match p.step {
                StepType::Steps(steps) => {
                    if steps <= 0 {
                        return Err(format!(
                            "Invalid parameter ({}): 'steps' must be greater than 0.",
                            name
                        ));
                    }
                }
                StepType::Step(step) => {
                    if step <= 0.0 {
                        return Err(format!(
                            "Invalid paramater ({}): 'step' must be greater than zero.",
                            name
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

fn validate_param_normalized(param: &NumParamOpt<f64>, name: &str) -> Result<(), String> {
    match param {
        OptParameter::Fixed(v) => {
            if *v < 0.0 || *v > 1.0 {
                return Err(format!(
                    "Invalid paramater ({}): value must be in the range [0, 1].",
                    name
                ));
            }
        }
        OptParameter::Varying(p) => {
            if p.from < 0.0 {
                return Err(format!(
                    "Invalid paramater ({}): 'from' must be greater than or equal to 0.",
                    name
                ));
            }
            if p.to > 1.0 {
                return Err(format!(
                    "Invalid paramater ({}): 'to' must be less than or equal to 1.",
                    name
                ));
            }
            if p.from >= p.to {
                return Err(format!(
                    "Invalid paramater ({}): 'from' must be less than 'to'.",
                    name
                ));
            }
            if let StepType::Steps(steps) = p.step {
                if steps == 0 {
                    return Err(format!(
                        "Invalid parameter ({}): 'steps' must be greater than 0.",
                        name
                    ));
                }
            }
            if let StepType::Step(step) = p.step {
                if step <= 0.0 {
                    return Err(format!(
                        "Invalid paramater ({}): 'step' must be greater than zero.",
                        name
                    ));
                }
            }
        }
    };

    Ok(())
}

// Param range trait is used to glue paramater ranging together over several types
trait ParamRange<T> {
    fn param_range(&self) -> Vec<T>;
}

impl ParamRange<f64> for NumParamOpt<f64> {
    fn param_range(&self) -> Vec<f64> {
        param_range_opt(self, param_range_num)
    }
}

impl ParamRange<u64> for NumParamOpt<u64> {
    fn param_range(&self) -> Vec<u64> {
        param_range_opt(self, |p| {
            param_range_num(p)
                .iter()
                .map(|v| unsafe { v.round().to_int_unchecked() })
                .collect()
        })
    }
}

impl<T: Clone + Copy> ParamRange<T> for VecParamOpt<T> {
    fn param_range(&self) -> Vec<T> {
        param_range_opt(self, |v| v.to_vec())
    }
}

fn param_range_opt<T: Clone + Copy, U>(
    opt_param: &OptParameter<T, U>,
    f: fn(&U) -> Vec<T>,
) -> Vec<T> {
    match opt_param {
        OptParameter::Fixed(t) => vec![*t],
        OptParameter::Varying(u) => f(u),
    }
}

fn param_range_num(p: &NumericalParameter) -> Vec<f64> {
    let diff = p.to - p.from;
    let ratio = p.to / p.from;

    let steps = match p.step {
        StepType::Steps(steps) => steps,
        StepType::Step(step) => match p.mode {
            StepMode::Linear => (diff / step).round() as u64 + 1,
            StepMode::Log => ratio.log(step).round() as u64 + 1,
        },
    };

    let step = match p.mode {
        StepMode::Linear => diff / (steps - 1) as f64,
        StepMode::Log => ratio.powf(1.0 / (steps - 1) as f64),
    };

    (0..steps)
        .map(|i| match p.mode {
            StepMode::Linear => p.from + i as f64 * step,
            StepMode::Log => p.from * step.powf(i as f64),
        })
        .collect()
}

fn statistics_to_description_2set(
    datatype: Datatype,
    fixed_size: u64,
    fixed: FixedSizeRef,
    trials: u64,
    selectivity: f64,
    skew: f64,
    density: f64,
    distribution: DataDistribution,
    seed: u64,
    byte_offset: &mut u64,
) -> Result<DataBinDescription, String> {
    let (long_length, short_length) = match fixed {
        FixedSizeRef::Longest => (fixed_size, (fixed_size as f64 * skew).round() as u64),
        FixedSizeRef::Shortest => ((fixed_size as f64 / skew).round() as u64, fixed_size),
    };
    let set_lengths = vec![long_length, short_length];
    let intersection_length = (short_length as f64 * selectivity) as u64;
    let lengths = DataBinLengthsEnum::Pair(DataBinLengths {
        set_lengths,
        intersection_length,
    });

    let max_value = (long_length as f64 / density).min(datatype.max() as f64) as u64;

    let byte_length = (long_length + short_length + intersection_length) * datatype.bytes() * trials;

    let bin = DataBinDescription {
        datatype,
        max_value,
        lengths,
        trials,
        distribution,
        seed,
        byte_offset: *byte_offset,
        byte_length,
    };

    *byte_offset += (long_length + short_length + intersection_length) * datatype.bytes() * trials;

    Ok(bin)
}

fn statistics_to_description_kset(
    datatype: Datatype,
    fixed_size: u64,
    fixed: FixedSizeRef,
    trials: u64,
    query_selectivity: f64,
    corpus_skew: f64,
    corpus_density: f64,
    data_distribution: DataDistribution,
    seed: u64,
    offset: &mut u64,
    query_size: u64,
    corpus_size: u64,
    corpus_distribution: CorpusDistribution,
    query_distribution: QueryDistribution,
    samples: u64,
    rng: &mut impl Rng,
) -> Result<DataBinDescription, String> {
    // Sanity check to avoid infinite loop when sampling distribution
    if corpus_size < query_size {
        return Err(format!(
            "Corpus size ({}) must be at least as large as query size ({}).",
            corpus_size, query_size
        ));
    }

    let samples_usize: usize = samples
        .try_into()
        .map_err(|_| format!("Could not convert samples ({}) to usize.", samples))?;
    let query_size_usize: usize = query_size
        .try_into()
        .map_err(|_| format!("Could not convert query_size ({}) to usize.", query_size))?;
    let corpus_size_usize: usize = corpus_size
        .try_into()
        .map_err(|_| format!("Could not convert corpus_size ({}) to usize.", corpus_size))?;

    let query_size_usize_successor = query_size_usize.checked_add(1usize).ok_or(format!(
        "Could not increment query_size_usize ({})",
        query_size_usize
    ))?;

    let longest_length_in_corpus = match fixed {
        FixedSizeRef::Longest => fixed_size,
        FixedSizeRef::Shortest => (fixed_size as f64 / corpus_skew).round() as u64,
    };

    let raw_lengths: Vec<DataBinLengths> = if corpus_skew == 1.0 {
        iter::repeat(DataBinLengths {
            set_lengths: iter::repeat(longest_length_in_corpus)
                .take(query_size_usize)
                .collect(),
            intersection_length: (longest_length_in_corpus as f64 * query_selectivity).round()
                as u64,
        })
        .take(samples_usize)
        .collect()
    } else {
        match query_distribution {
            QueryDistribution::Zipf {} => {
                let zipf_exponent = -f64::log(corpus_skew, corpus_size as f64);
                let length_dist = match ZipfDistribution::new(corpus_size_usize, zipf_exponent) {
                        Ok(x) => Ok(x),
                        Err(()) => Err(format!(
                            "Failed to create discrete Zipf distribution with (num_elements = {}, exponent = {}).", 
                            corpus_size_usize, zipf_exponent
                        )),
                    }?;

                iter::repeat_with(|| {
                    let mut set_index: Vec<usize> = if query_size == corpus_size {
                        (1..query_size_usize_successor).collect()
                    } else {
                        sample_distribution_unique(query_size_usize, &length_dist, rng)
                    };
                    set_index.sort_unstable();

                    match corpus_distribution {
                        CorpusDistribution::Zipf {} => {
                            let set_lengths: Vec<u64> = set_index
                                .into_iter()
                                .map(|i| {
                                    (longest_length_in_corpus as f64
                                        * (i as f64).powf(-zipf_exponent))
                                    .round() as u64
                                })
                                .collect();

                            let shortest_length = *set_lengths.last().unwrap();
                            let intersection_length =
                                (shortest_length as f64 * query_selectivity) as u64;

                            DataBinLengths {
                                set_lengths,
                                intersection_length,
                            }
                        }
                    }
                })
                .take(samples_usize)
                .collect()
            }
        }
    };

    let byte_length = {
        let mut delta = 0u64;
        for databin_lengths in &raw_lengths {
            let total_set_length = databin_lengths
                .set_lengths
                .iter()
                .fold(0u64, |acc, x| acc + *x);
            let total_length = total_set_length + databin_lengths.intersection_length;
            let total_bytes = total_length * datatype.bytes() * trials;
            delta += total_bytes;
        }
        delta
    };

    let max_value = (longest_length_in_corpus as f64 / corpus_density).round() as u64;
    if max_value > datatype.max() {
        return Err(format!(
            "The maximum value ({}) is too large for the datatype ({:?}).",
            max_value, datatype
        ));
    }

    let lengths = DataBinLengthsEnum::Sample(raw_lengths);

    let bin = DataBinDescription {
        datatype,
        max_value,
        lengths,
        trials,
        distribution: data_distribution,
        seed,
        byte_offset: *offset,
        byte_length,
    };

    *offset += byte_length;

    Ok(bin)
}
