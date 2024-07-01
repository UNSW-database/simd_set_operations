use benchmark::{
    fmt_open_err, path_str, util::sample_distribution_unique, DataBinConfig, DataBinLengths,
    DataDescription, DataDistribution, Datatype,
};
use clap::Parser;
use colored::*;
use std::{
    collections::HashMap,
    fs::{self, File},
    iter,
    path::PathBuf,
};

use rand::{Rng, SeedableRng};
use serde::Deserialize;
use zipf::ZipfDistribution;

// Schema for TOML configuration file
#[derive(Deserialize, Debug)]
struct Config {
    dataset: HashMap<String, Dataset>,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    datatype: VecParamOpt<Datatype>,
    max_length: NumParamOpt<usize>,
    trials: NumParamOpt<usize>,
    selectivity: NumParamOpt<f64>,
    skew: NumParamOpt<f64>,
    density: NumParamOpt<f64>,
    distribution: VecParamOpt<DataDistribution>,
    #[serde(flatten)]
    kset_info: Option<KSetInfo>,
}

#[derive(Deserialize, Debug)]
struct KSetInfo {
    sets: NumParamOpt<u64>,
    corpus: Corpus,
}

#[derive(Deserialize, Debug)]
struct Corpus {
    size: NumParamOpt<u64>,
    distribution: VecParamOpt<CorpusDistribution>,
    samples: NumParamOpt<usize>,
}

#[derive(Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
enum CorpusDistribution {
    Zipf {},
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
    println!(
        "{}: {} (\"{}\")",
        "READING".green().bold(),
        "config file",
        path_str(&cli.config)
    );

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

    for (name, dataset) in &config.dataset {
        println!(
            "{}: {} (\"{}\")",
            "GENERATING".green().bold(),
            "dataset description",
            name
        );

        // Validate dataset configuration
        validate_param_unsigned(&dataset.max_length, "max_length")?;
        validate_param_unsigned(&dataset.trials, "trials")?;
        validate_param_normalized(&dataset.selectivity, "selectivity")?;
        validate_param_normalized(&dataset.skew, "skew")?;
        validate_param_normalized(&dataset.density, "density")?;
        if let Some(kset_info) = &dataset.kset_info {
            validate_param_unsigned(&kset_info.sets, "sets")?;
            validate_param_unsigned(&kset_info.corpus.size, "corpus_size")?;
            validate_param_unsigned(&kset_info.corpus.samples, "samples")?;
        }

        // Generate data bin configurations
        let data_bins: DataDescription = {
            let mut ret = Vec::<DataBinConfig>::new();

            // Could be done with `iproduct!` or chained `.flat_map`, but they both add un-needed complexity to
            // something that would ideally be solved with a relatively simple macro.
            let mut offset = 0usize;
            for datatype in dataset.datatype.param_range() {
                for distribution in dataset.distribution.param_range() {
                    for max_length in dataset.max_length.param_range() {
                        for trials in dataset.trials.param_range() {
                            for density in dataset.density.param_range() {
                                for skew in dataset.skew.param_range() {
                                    for selectivity in dataset.selectivity.param_range() {
                                        // Split between 2-set and k-set logic
                                        if let Some(kset_info) = &dataset.kset_info {
                                            for sets in kset_info.sets.param_range() {
                                                for corpus_size in
                                                    kset_info.corpus.size.param_range()
                                                {
                                                    for corpus_dist in
                                                        kset_info.corpus.distribution.param_range()
                                                    {
                                                        for length_samples in
                                                            kset_info.corpus.samples.param_range()
                                                        {
                                                            let seed: u64 = rng.gen();
                                                            ret.push(
                                                                statistics_to_description_kset(
                                                                    datatype,
                                                                    max_length,
                                                                    trials,
                                                                    selectivity,
                                                                    skew,
                                                                    density,
                                                                    distribution,
                                                                    seed,
                                                                    &mut offset,
                                                                    sets,
                                                                    corpus_size,
                                                                    corpus_dist,
                                                                    length_samples,
                                                                    &mut rng,
                                                                )?,
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            let seed: u64 = rng.gen();
                                            ret.push(statistics_to_description_2set(
                                                datatype,
                                                max_length,
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

            ret
        };

        // Write dataset description
        let desc_path = (&cli.outdir).join(name.to_string() + ".json");
        let desc_file = File::create(&desc_path).map_err(|e| {
            format!(
                "Failed to open file {}:\n{}",
                path_str(&desc_path),
                e.to_string()
            )
        })?;
        serde_json::to_writer(desc_file, &data_bins).map_err(|e| e.to_string())?;
    }

    Ok(())
}

fn validate_param_unsigned<T>(param: &NumParamOpt<T>, name: &str) -> Result<(), String>
where
    T: From<u8> + PartialEq,
{
    match param {
        OptParameter::Fixed(v) => {
            if *v == 0.into() {
                return Err(format!(
                    "Invalid paramater ({}): value must be greater than 0.",
                    name
                ));
            }
        }
        OptParameter::Varying(p) => {
            if p.from == 0.0 {
                return Err(format!(
                    "Invalid paramater ({}): 'from' must be greater than 0.",
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
                if step == 0.0 {
                    return Err(format!(
                        "Invalid paramater ({}): 'step' must be greater than zero.",
                        name
                    ));
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

impl ParamRange<usize> for NumParamOpt<usize> {
    fn param_range(&self) -> Vec<usize> {
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
    max_length: usize,
    trials: usize,
    selectivity: f64,
    skew: f64,
    density: f64,
    distribution: DataDistribution,
    seed: u64,
    offset: &mut usize,
) -> Result<DataBinConfig, String> {
    let long_length = max_length;
    let short_length = (max_length as f64 * skew) as usize;
    let set_lengths = vec![long_length, short_length];
    let intersection_length = (short_length as f64 * selectivity) as usize;
    let lengths = vec![ DataBinLengths { set_lengths, intersection_length } ];

    let max_value = (long_length as f64 / density).min(datatype.max() as f64) as u64;

    let bin = DataBinConfig {
        datatype,
        max_value,
        lengths,
        trials,
        distribution,
        seed,
        offset: *offset,
    };

    *offset += (long_length + short_length + intersection_length) * datatype.bytes() * trials;

    Ok(bin)
}

fn statistics_to_description_kset(
    datatype: Datatype,
    max_length: usize,
    trials: usize,
    selectivity: f64,
    skew: f64,
    density: f64,
    distribution: DataDistribution,
    seed: u64,
    offset: &mut usize,
    set_count: u64,
    corpus_size: u64,
    corpus_dist: CorpusDistribution,
    length_samples: usize,
    rng: &mut impl Rng,
) -> Result<DataBinConfig, String> {
    // Sanity check to avoid infinite loop when sampling distribution
    if set_count > corpus_size {
        return Err(format!(
            "Corpus size ({}) must be larger than set count ({}).",
            corpus_size, set_count
        ));
    }

    let lengths: Vec<DataBinLengths> = if skew == 1.0 {
        iter::repeat(DataBinLengths {
            set_lengths: iter::repeat(max_length).take(set_count as usize).collect(),
            intersection_length: (max_length as f64 * selectivity) as usize,
        })
        .take(length_samples)
        .collect()
    } else {
        match corpus_dist {
            CorpusDistribution::Zipf {} => {
                let zipf_exponent = -f64::log(skew, corpus_size as f64);
                let length_dist = match ZipfDistribution::new(corpus_size as usize, zipf_exponent) {
                        Ok(x) => Ok(x),
                        Err(()) => Err(format!(
                            "Failed to create discrete Zipf distribution with (num_elements = {}, exponent = {}).", 
                            corpus_size as usize, zipf_exponent
                        )),
                    }?;
                iter::repeat_with(|| {
                    let mut set_index: Vec<usize> = if set_count == corpus_size {
                        (1..=(set_count as usize)).collect()
                    } else {
                        sample_distribution_unique(set_count as usize, &length_dist, rng)
                    };
                    set_index.sort_unstable();
                    let set_lengths: Vec<usize> = set_index
                        .into_iter()
                        .map(|i| {
                            (max_length as f64 * (1.0 / (i as f64).powf(zipf_exponent))).round()
                                as usize
                        })
                        .collect();

                    let shortest_length = *set_lengths.last().unwrap();
                    let intersection_length = (shortest_length as f64 * selectivity) as usize;

                    DataBinLengths {
                        set_lengths,
                        intersection_length,
                    }
                })
                .take(length_samples)
                .collect()
            }
        }
    };

    let offset_delta = {
        let mut delta = 0usize;
        for databin_lengths in &lengths {
            let total_set_length = databin_lengths.set_lengths.iter().fold(0usize, |acc, x| acc + *x);
            let total_length = total_set_length + databin_lengths.intersection_length;
            let total_bytes = total_length * datatype.bytes() * trials;
            delta += total_bytes;
        }
        delta
    };

    let max_value = (max_length as f64 / density).min(datatype.max() as f64) as u64;

    let bin = DataBinConfig {
        datatype,
        max_value,
        lengths,
        trials,
        distribution,
        seed,
        offset: *offset,
    };

    *offset += offset_delta;

    Ok(bin)
}
