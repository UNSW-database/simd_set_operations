use benchmark::{fmt_open_err, path_str, DataBinConfig, Datatype};
use clap::Parser;
use colored::*;
use std::{
    collections::HashMap,
    fs::{self, File},
    path::PathBuf,
};

use serde::Deserialize;

// Schema for TOML configuration file
#[derive(Deserialize, Debug)]
struct Config {
    pub dataset: HashMap<String, Dataset>,
}

#[derive(Deserialize, Debug)]
struct Dataset {
    pub datatype: Datatype,
    pub max_length: u64,
    pub trials: u64,
    pub selectivity: Parameter,
    pub skew: Parameter,
    pub density: Parameter,
}

#[derive(Deserialize, Debug)]
struct Parameter {
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
        validate_param(&dataset.selectivity, "selectivity")?;
        validate_param(&dataset.skew, "skew")?;
        validate_param(&dataset.density, "density")?;

        // Generate data bin configurations
        let data_bins: Vec<DataBinConfig> = {
            let mut ret = Vec::<DataBinConfig>::new();

            let mut offset = 0u64;
            for density in param_range(&dataset.density) {
                for skew in param_range(&dataset.skew) {
                    for selectivity in param_range(&dataset.selectivity) {
                        let long_length = dataset.max_length;
                        let short_length = (dataset.max_length as f64 * skew) as u64;
                        let intersection_length = (short_length as f64 * selectivity) as u64;

                        let reciprocal_density = if density == 0.0 {
                            u64::MAX
                        } else {
                            (1.0 / density) as u64
                        };
                        let max_value = if u64::MAX / long_length < reciprocal_density {
                            u64::MAX
                        } else {
                            long_length * reciprocal_density
                        };

                        let bin = DataBinConfig {
                            datatype: dataset.datatype,
                            long_length,
                            short_length,
                            trials: dataset.trials,
                            intersection_length,
                            max_value,
                            offset,
                        };

                        // Sanity check sizes
                        let minimum_cardinality = long_length + short_length - intersection_length;
                        if max_value < minimum_cardinality {
                            return Err(format!(
                                "Density ({}) too high for given selectivity ({}).\n{:#?}",
                                density, selectivity, bin
                            ));
                        }

                        ret.push(bin);
                        offset += (long_length + short_length + intersection_length)
                            * dataset.datatype.bytes();
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

fn validate_param(param: &Parameter, name: &str) -> Result<(), String> {
    if param.from < 0.0 {
        return Err(format!(
            "Invalid paramater ({}): 'from' must be greater than or equal to 0.",
            name
        ));
    }
    if param.to > 1.0 {
        return Err(format!(
            "Invalid paramater ({}): 'to' must be less than or equal to 1.",
            name
        ));
    }
    if param.from > param.to {
        return Err(format!(
            "Invalid paramater ({}): 'from' must be less than or equal to 'to'.",
            name
        ));
    }
    if let StepType::Steps(steps) = param.step {
        if steps == 0 {
            return Err(format!(
                "Invalid parameter ({}): 'steps' must be greater than 0.",
                name
            ));
        }
    }
    if let StepType::Step(step) = param.step {
        if step <= 0.0 && param.from != param.to {
            return Err(format!(
                "Invalid paramater ({}): 'step' must be greater than zero if 'from' is not equal to 'to'.", 
                name
            ));
        }
    }

    Ok(())
}

fn param_range(param: &Parameter) -> Vec<f64> {
    let diff = (param.to - param.from) as f64;
    let ratio = param.to as f64 / param.from as f64;

    let step = match param.step {
        StepType::Step(step) => step,
        StepType::Steps(steps) => match param.mode {
            StepMode::Linear => diff / (steps as f64 - 1.0),
            StepMode::Log => ratio.powf(1.0 / (steps as f64)),
        },
    };

    let step_count = match param.step {
        StepType::Steps(steps) => steps,
        StepType::Step(step) => match param.mode {
            StepMode::Linear => diff / step,
            StepMode::Log => ratio.log(step),
        }
        .round() as u64,
    };

    let mut ret: Vec<f64> = match param.mode {
        StepMode::Linear => (0..step_count)
            .map(|i| param.from + i as f64 * step)
            .collect(),
        StepMode::Log => (0..step_count)
            .map(|i| param.from * step.powf(i as f64))
            .collect(),
    };

    // Ensure max value isn't nonsensical; one step should be guaranteed by parameter validation
    *ret.last_mut().unwrap() = ret.last().unwrap().clamp(0.0, 1.0);
    ret
}
