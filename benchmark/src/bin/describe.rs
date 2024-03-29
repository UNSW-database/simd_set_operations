use benchmark::{fmt_open_err, path_str, DataBinConfig, Datatype, Distribution};
use clap::Parser;
use colored::*;
use std::{
    collections::HashMap,
    fs::{self, File},
    path::PathBuf,
};

use rand::{Rng, SeedableRng};
use serde::Deserialize;

// Schema for TOML configuration file
#[derive(Deserialize, Debug)]
struct Config {
    pub dataset: HashMap<String, Dataset>,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    pub datatype: VecParamOpt<Datatype>,
    pub max_length: NumParamOpt<usize>,
    pub trials: NumParamOpt<u64>,
    pub selectivity: NumParamOpt<f64>,
    pub skew: NumParamOpt<f64>,
    pub density: NumParamOpt<f64>,
    pub distribution: VecParamOpt<Distribution>,
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

        // Generate data bin configurations
        let data_bins: Vec<DataBinConfig> = {
            let mut ret = Vec::<DataBinConfig>::new();

            let mut offset = 0usize;
            for datatype in dataset.datatype.param_range() {
                for distribution in dataset.distribution.param_range() {
                    for max_length in dataset.max_length.param_range() {
                        for trials in dataset.trials.param_range() {
                            for density in dataset.density.param_range() {
                                for skew in dataset.skew.param_range() {
                                    for selectivity in dataset.selectivity.param_range() {
                                        let seed: u64 = rng.gen();
                                        ret.push(statistics_to_description(
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
    T: From<u8> + PartialEq
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
            param_range_num(p).iter().map(|v| unsafe {
                v.round().to_int_unchecked()
            }).collect()
        })
    }
}

impl ParamRange<usize> for NumParamOpt<usize> {
    fn param_range(&self) -> Vec<usize> {
        param_range_opt(self, |p| {
            param_range_num(p).iter().map(|v| unsafe {
                v.round().to_int_unchecked()
            }).collect()
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

fn statistics_to_description(
    datatype: Datatype,
    max_length: usize,
    trials: u64,
    selectivity: f64,
    skew: f64,
    density: f64,
    distribution: Distribution,
    seed: u64,
    offset: &mut usize,
) -> Result<DataBinConfig, String> {
    let long_length = max_length;
    let short_length = (max_length as f64 * skew) as usize;
    let intersection_length = (short_length as f64 * selectivity) as usize;

    let max_value = (long_length as f64 / density).min(datatype.max() as f64) as u64;

    let bin = DataBinConfig {
        datatype,
        max_value,
        long_length,
        short_length,
        trials,
        intersection_length,
        distribution,
        seed,
        offset: *offset,
    };

    // Sanity check sizes
    let minimum_cardinality = (long_length - intersection_length) + short_length;
    if (max_value as usize) < minimum_cardinality {
        return Err(format!(
            "Density ({}) too high for given selectivity ({}).\n{:#?}",
            density, selectivity, bin
        ));
    }

    *offset += (long_length + short_length + intersection_length) * datatype.bytes(); 

    Ok(bin)
}
