use benchmark::{
    algorithms::{
        algorithm_fn_from_algorithm_i32, algorithm_fn_from_algorithm_i64,
        algorithm_fn_from_algorithm_u32, algorithm_fn_from_algorithm_u64, Algorithm, AlgorithmFn,
        ALGORITHMS,
    },
    fmt_open_err, path_str, read_databin, read_dataset_description, tsc, DataBin,
    DataSetDescription, Datatype, Trial,
};
use clap::Parser;
use colored::*;
use paste::paste;
use setops::intersect::TwoSetAlgorithmFnGeneric;
use std::{
    collections::HashMap,
    fs::{self, File},
    iter,
    path::PathBuf,
    time::{self, SystemTime},
};

mod experiment_schema {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize, Debug)]
    pub struct Config {
        pub algorithm_set: HashMap<String, AlgorithmSet>,
        pub experiment: HashMap<String, ExperimentConfig>,
    }

    #[derive(Deserialize, Debug, Default)]
    #[serde(default)]
    pub struct AlgorithmSet {
        pub twoset: Vec<String>,
        pub twoset_to_kset: Vec<String>,
        pub fsearch: Vec<String>,
        pub fsearch_to_twoset: Vec<String>,
        pub fsearch_to_kset: Vec<String>,
    }

    #[derive(Deserialize, Debug)]
    pub struct ExperimentConfig {
        pub count_only: bool,
        pub cache: Cache,
        pub runs: u64,
        pub algorithm_sets: Vec<String>,
    }

    #[derive(Deserialize, Debug)]
    #[serde(tag = "type", rename_all = "snake_case")]
    pub enum Cache {
        Warm { runs: u64 },
        Cold {},
    }
}

mod results_schema {
    use serde::Serialize;
    use std::collections::HashMap;

    #[derive(Serialize, Debug)]
    pub struct Results {
        pub tsc_frequency: u64,
        pub tsc_overhead: u64,
        pub experiment_results: Vec<ExperimentResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct ExperimentResult {
        pub experiment_name: String,
        pub databin_results: Vec<DataBinResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct DataBinResult {
        pub databin_index: usize,
        pub results: DataBinResultType,
    }

    pub type AlgorithmResult<T> = HashMap<String, T>;

    #[derive(Serialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum DataBinResultType {
        Pair(AlgorithmResult<Vec<TrialResult>>),
        Sample(AlgorithmResult<Vec<SampleResult>>),
    }

    #[derive(Serialize, Debug)]
    pub struct SampleResult {
        pub trials: Vec<TrialResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct TrialResult {
        pub times: Vec<Time>,
    }

    #[derive(Serialize, Debug, Default)]
    pub struct Time {
        pub pre_freq: u64,
        pub counts: u64,
        pub post_freq: u64,
    }
}

struct Experiment<'config, 'name> {
    r_name: &'config str,
    count_only: bool,
    r_cache: &'config experiment_schema::Cache,
    runs: u64,
    algorithms_r: Vec<(&'name String, &'static Algorithm)>,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    description: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    experiment: Option<String>,
}

fn main() {
    if cfg!(debug_assertions) {
        println!("{}", "WARNING: running in debug mode.".yellow().bold());
    }

    let cli = Cli::parse();

    if let Err(e) = bench(&cli) {
        let msg = format!("ERROR: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn bench(cli: &Cli) -> Result<(), String> {
    // Read and parse experiment configuration file
    let config: experiment_schema::Config = {
        let config_string =
            fs::read_to_string(&cli.config).map_err(|e| fmt_open_err(e, &cli.config))?;
        toml::from_str(&config_string).map_err(|e| {
            format!(
                "Invalid experiment config file {}: {}",
                path_str(&cli.config),
                e
            )
        })?
    };

    let dataset_description = read_dataset_description(&cli.description)?;

    // Open data file for later reading
    let mut data_file: File = {
        let data_file_path = cli.description.with_extension("data");
        File::open(&data_file_path).map_err(|e| fmt_open_err(e, &data_file_path))?
    };

    // Convert algorithm sets to map of set name to vec of actual algorithms
    let algorithm_sets_r = config
        .algorithm_set
        .iter()
        .map(|(set_name, set)| {
            let twoset_to_kset_names: Vec<_> = set
                .twoset_to_kset
                .iter()
                .flat_map(|outer_name| {
                    set.twoset
                        .iter()
                        .map(move |inner_name| format!("{}_{}", outer_name, inner_name))
                })
                .collect();
            let names_iter = set.twoset.iter().chain(twoset_to_kset_names.iter());
            let algorithms = names_iter
                .map(|name| {
                    let algorithm = ALGORITHMS
                        .get(&name)
                        .ok_or(format!("There is no algorithm named {}.", name))?;
                    Ok((name.to_owned(), algorithm))
                })
                .collect::<Result<Vec<(String, &Algorithm)>, String>>()?;

            Ok((set_name, algorithms))
        })
        .collect::<Result<HashMap<&String, Vec<(String, &Algorithm)>>, String>>()?;

    // Filter the experiment configs based on the command line option
    let experiment_configs_r: HashMap<&str, &experiment_schema::ExperimentConfig> = {
        if let Some(experiment_name) = &cli.experiment {
            let value = config.experiment.get(experiment_name).ok_or(format!(
                "Experiment ({}) not found in configuration.",
                experiment_name
            ))?;
            iter::once((experiment_name.as_str(), value)).collect()
        } else {
            config
                .experiment
                .iter()
                .map(|(name, ec)| (name.as_str(), ec))
                .collect()
        }
    };

    // Translate experiment configs into experiment structs for benchmarking use
    let experiments: Vec<Experiment> = experiment_configs_r
        .into_iter()
        .map(|(name, ec)| {
            let algorithm_vecs = ec
                .algorithm_sets
                .iter()
                .map(|set_name| {
                    algorithm_sets_r
                        .get(set_name)
                        .ok_or(format!("Could not find algorithm set: {}", set_name))
                })
                .collect::<Result<Vec<&Vec<(String, &Algorithm)>>, String>>()?;

            let algorithms = algorithm_vecs
                .into_iter()
                .flatten()
                .map(|(name, algo)| (name, *algo))
                .collect();

            Ok(Experiment {
                r_name: name,
                count_only: ec.count_only,
                r_cache: &ec.cache,
                runs: ec.runs,
                algorithms_r: algorithms,
            })
        })
        .collect::<Result<Vec<Experiment>, String>>()?;

    // Run the benchmarks
    let results = run_benchmarks(&dataset_description, &mut data_file, &experiments)?;

    // Write results
    let time = SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
    let results_path = cli
        .description
        .with_extension(format!("results.{}.json", time.as_secs()));
    let results_file = File::create(&results_path).map_err(|e| fmt_open_err(e, &results_path))?;

    serde_json::to_writer(results_file, &results)
        .map_err(|e| format!("Failed to write {}: {}", path_str(&results_path), e))?;

    Ok(())
}

fn run_benchmarks<'name>(
    r_dataset_description: &DataSetDescription,
    mr_data_file: &mut File,
    r_experiments: &Vec<Experiment>,
) -> Result<results_schema::Results, String> {
    let tsc_frequency = tsc::estimate_frequency();
    let tsc_overhead = tsc::measure_overhead();

    let experiment_results = r_experiments
        .iter()
        .map(|r_experiment| {
            let warmup_runs = match r_experiment.r_cache {
                experiment_schema::Cache::Warm { runs } => *runs,
                experiment_schema::Cache::Cold {} => 0,
            };

            let databin_results = r_dataset_description
                .iter()
                .enumerate()
                .map(|(databin_index, r_databin_description)| {
                    macro_rules! datatype_dispatch {
                        ($datatype:ident) => {
                            {
                                // We filter out algorithms that do not exist for the given type and return/count status
                                let algorithms: Vec<_> = r_experiment.algorithms_r.iter().filter_map(|&(r_name, r_algorithm)| {
                                    paste! {
                                        [<algorithm_fn_from_algorithm_ $datatype>](r_algorithm, r_experiment.count_only).map(|algo| (r_name, algo))
                                    }
                                }).collect();

                                let databin = read_databin::<$datatype, { std::mem::size_of::<$datatype>() }>(
                                    r_databin_description,
                                    mr_data_file,
                                )?;

                                benchmark_datatype::<$datatype>(
                                    &databin,
                                    warmup_runs,
                                    r_experiment.runs,
                                    &algorithms,
                                )
                            }
                        }
                    }

                    let databin_result_enum = match r_databin_description.datatype {
                        Datatype::U32 => datatype_dispatch!(u32),
                        Datatype::I32 => datatype_dispatch!(i32),
                        Datatype::I64 => datatype_dispatch!(u64),
                        Datatype::U64 => datatype_dispatch!(i64),
                    }?;

                    Ok(results_schema::DataBinResult {
                        databin_index,
                        results: databin_result_enum,
                    })
                }.map_err(|e: String| format!("Databin {}: {}", databin_index, e)))
                .collect::<Result<Vec<results_schema::DataBinResult>, String>>()?;

            Ok(results_schema::ExperimentResult {
                experiment_name: r_experiment.r_name.to_owned(),
                databin_results,
            })
        }.map_err(|e: String| format!("Experiment ({}): {}", r_experiment.r_name, e)))
        .collect::<Result<Vec<results_schema::ExperimentResult>, String>>()?;

    Ok(results_schema::Results {
        tsc_frequency,
        tsc_overhead,
        experiment_results,
    })
}

fn benchmark_datatype<T>(
    r_data: &DataBin<T>,
    warmup_runs: u64,
    runs: u64,
    r_algorithms: &Vec<(&String, AlgorithmFn<T>)>,
) -> Result<results_schema::DataBinResultType, String>
where
    T: Ord + Copy,
{
    match r_data {
        DataBin::Pair(r_trials) => {
            let mut results = Vec::<_>::with_capacity(r_algorithms.len());
            for (rr_name, r_algorithm) in r_algorithms {
                let result = r_trials
                    .iter()
                    .map(|r_trial| benchmark_trial::<T>(r_trial, warmup_runs, runs, r_algorithm))
                    .collect();
                results.push(((*rr_name).to_owned(), result));
            }
            let map: HashMap<_, _> = results.into_iter().collect();
            Ok(results_schema::DataBinResultType::Pair(map))
        }
        DataBin::Sample(r_samples) => {
            let mut results = Vec::<_>::with_capacity(r_algorithms.len());
            for (rr_name, r_algorithm) in r_algorithms {
                let result = r_samples
                    .iter()
                    .map(|r_sample| results_schema::SampleResult {
                        trials: r_sample
                            .iter()
                            .map(|r_trial| {
                                benchmark_trial::<T>(r_trial, warmup_runs, runs, r_algorithm)
                            })
                            .collect(),
                    })
                    .collect();
                results.push(((*rr_name).to_owned(), result));
            }
            let map: HashMap<_, _> = results.into_iter().collect();
            Ok(results_schema::DataBinResultType::Sample(map))
        }
    }
}

fn benchmark_trial<T: Ord + Copy>(
    r_trial: &Trial<T>,
    warmup_runs: u64,
    runs: u64,
    r_algorithm: &AlgorithmFn<T>,
) -> results_schema::TrialResult {
    let total_runs = warmup_runs + runs;
}

fn benchmark_2set<T: Ord + Copy>(
    sets_r: (&[T], &[T]),
    mr_out: &mut [T],
    r_algorithm: &TwoSetAlgorithmFnGeneric<T>,
    mr_time: &mut results_schema::Time,
) -> usize {
    let mut freq_buf = [0u64; 3];

    let pre_freq = tsc::;
    let start = tsc::start();

    let size = r_algorithm(sets_r, mr_out);

    let end = tsc::end();
    let post_freq = tsc::estimate_frequency();

    let counts = end - start;
    *mr_time = results_schema::Time {
        pre_freq,
        counts,
        post_freq,
    };

    size
}
