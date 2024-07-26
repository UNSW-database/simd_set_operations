use benchmark::{
    algorithms::{Algorithm, AlgorithmFn, AlgorithmType, ALGORITHMS},
    fmt_open_err, path_str, read_databin, read_dataset_description, tsc,
    util::slice_equal,
    DataBin, DataBinDescription, DataSetDescription, Datatype, Trial,
};
use clap::Parser;
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use setops::intersect::{KSetAlgorithmBufFnGeneric, TwoSetAlgorithmFnGeneric};
use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    iter,
    path::PathBuf,
    sync::LazyLock,
    time::{self, Instant, SystemTime},
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
        pub repeats_per_databin: usize,
        pub runs_per_trial: usize,
        pub algorithm_sets: Vec<String>,
    }
}

mod results_schema {
    use benchmark::tsc::TSCCharacteristics;
    use serde::Serialize;

    #[derive(Serialize, Debug)]
    pub struct Results {
        pub tsc_characteristics: TSCCharacteristics,
        pub reference_cycles: u64,
        pub experiment_results: Vec<ExperimentResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct ExperimentResult {
        pub experiment_name: String,
        pub algorithm_results: Vec<AlgorithmResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct AlgorithmResult {
        pub algorithm_name: String,
        pub repeat_results: Vec<RepeatResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct RepeatResult {
        pub databin_results: Vec<Option<DataBinResult>>,
    }

    #[derive(Serialize, Debug)]
    pub struct DataBinResult {
        pub databin_index: usize,
        pub results: DataBinResultType,
    }

    #[derive(Serialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum DataBinResultType {
        Pair(Vec<TrialResult>),
        Sample(Vec<SampleResult>),
    }

    #[derive(Serialize, Debug)]
    pub struct SampleResult {
        pub trials: Vec<TrialResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct TrialResult {
        pub pre: FrequencyMeasurement,
        pub deltas: Vec<u64>,
        pub post: FrequencyMeasurement,
    }

    #[derive(Serialize, Debug)]
    pub struct FrequencyMeasurement {
        pub td: u128, // time delta
        pub cc: u64,  // cycles counts
    }
}

struct Experiment<'config, 'name> {
    r_name: &'config str,
    count_only: bool,
    runs_per_trial: usize,
    repeats_per_databin: usize,
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

const REFERENCE_CYCLES: u64 = 10_000;
const REFERENCE_TRIALS: usize = 3;

static START_INSTANT: LazyLock<Instant> = LazyLock::new(|| Instant::now());

fn main() {
    LazyLock::<Instant>::force(&START_INSTANT);

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
                runs_per_trial: ec.runs_per_trial,
                repeats_per_databin: ec.repeats_per_databin,
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

    print!("Writing results... ");
    let _ = std::io::stdout().flush();
    serde_json::to_writer(results_file, &results).map_err(|e| {
        println!();
        format!("Failed to write {}: {}", path_str(&results_path), e)
    })?;
    println!("DONE");

    Ok(())
}

fn run_benchmarks<'name>(
    r_dataset_description: &DataSetDescription,
    mr_data_file: &mut File,
    r_experiments: &Vec<Experiment>,
) -> Result<results_schema::Results, String> {
    let tsc_characteristics = tsc::characterise();

    // Iteration order:
    // 1. Experiment
    // 2. Algorithm
    // 3. Repeats
    // 4. Databin
    // 5. Sample (if present)
    // 6. Trial
    // 7. Run

    fn tick(bars: &[&ProgressBar]) {
        for bar in bars {
            bar.tick();
        }
    }

    fn update(bar: &ProgressBar, length: usize) {
        bar.reset();
        bar.set_length(length as u64);
    }

    let multi_progress = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{prefix:10} [{elapsed_precise}] {wide_bar} {pos:>5}/{len:5} {msg:40}",
    )
    .unwrap();
    let create_bar_m = |prefix| {
        multi_progress.add({
            let bar = ProgressBar::hidden();
            bar.set_style(style.clone());
            bar.set_prefix(prefix);
            bar
        })
    };

    let experiment_bar = create_bar_m("Experiment");
    let algorithm_bar = create_bar_m("Algorithm");
    let repeat_bar = create_bar_m("Repeat");
    let databin_bar = create_bar_m("Databin");

    update(&experiment_bar, r_experiments.len());
    let mut experiment_results =
        Vec::<results_schema::ExperimentResult>::with_capacity(r_experiments.len());
    for r_experiment in r_experiments {
        experiment_bar.set_message(r_experiment.r_name.to_owned());

        update(&algorithm_bar, r_experiment.algorithms_r.len());
        let mut algorithm_results =
            Vec::<results_schema::AlgorithmResult>::with_capacity(r_experiment.algorithms_r.len());
        for &(r_name, r_algorithm) in &r_experiment.algorithms_r {
            algorithm_bar.set_message(r_name.to_owned());

            update(&repeat_bar, r_experiment.repeats_per_databin);
            let mut repeat_results = Vec::<results_schema::RepeatResult>::with_capacity(
                r_experiment.repeats_per_databin,
            );
            for repeat in 0..r_experiment.repeats_per_databin {
                repeat_bar.tick();

                update(&databin_bar, r_dataset_description.len());
                let mut databin_results =
                    Vec::<Option<results_schema::DataBinResult>>::with_capacity(
                        r_dataset_description.len(),
                    );
                for (databin_index, r_databin_description) in
                    r_dataset_description.iter().enumerate()
                {
                    tick(&[&experiment_bar, &algorithm_bar, &repeat_bar, &databin_bar]);

                    let results_opt = datatype_dispatch(
                        r_algorithm,
                        r_experiment,
                        r_databin_description,
                        mr_data_file,
                    )
                    .map_err(|e| {
                        format!(
                            "Experiment \"{}\": Algorithm \"{}\": Repeat {}: Databin {}: {}",
                            r_experiment.r_name, r_name, repeat, databin_index, e,
                        )
                    })?;

                    let results = results_opt.map(|results| results_schema::DataBinResult {
                        databin_index,
                        results,
                    });

                    databin_results.push(results);

                    databin_bar.inc(1);
                }
                repeat_results.push(results_schema::RepeatResult { databin_results });

                repeat_bar.inc(1);
            }
            algorithm_results.push(results_schema::AlgorithmResult {
                algorithm_name: r_name.to_owned(),
                repeat_results,
            });

            algorithm_bar.inc(1);
        }
        experiment_results.push(results_schema::ExperimentResult {
            experiment_name: r_experiment.r_name.to_owned(),
            algorithm_results,
        });

        experiment_bar.inc(1);
    }

    for r_bar in &[experiment_bar, algorithm_bar, repeat_bar, databin_bar] {
        r_bar.finish();
    }

    Ok(results_schema::Results {
        tsc_characteristics,
        reference_cycles: REFERENCE_CYCLES,
        experiment_results,
    })
}

fn datatype_dispatch(
    r_algorithm: &Algorithm,
    r_experiment: &Experiment,
    r_databin_description: &DataBinDescription,
    mr_data_file: &mut File,
) -> Result<Option<results_schema::DataBinResultType>, String> {
    macro_rules! datatype_dispatch {
        ($datatype:ident) => {{
            let algorithm_fn_opt =
                $datatype::algorithm_fn_from_algorithm(r_algorithm, r_experiment.count_only);
            if let Some(algorithm_fn) = algorithm_fn_opt {
                if algorithm_fn.is_valid(r_databin_description.lengths.set_count()) {
                    let databin = read_databin::<$datatype, { std::mem::size_of::<$datatype>() }>(
                        r_databin_description,
                        mr_data_file,
                    )?;
                    Some(benchmark_databin::<$datatype>(
                        &databin,
                        r_experiment.runs_per_trial,
                        &algorithm_fn,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        }};
    }

    Ok(match r_databin_description.datatype {
        Datatype::U32 => datatype_dispatch!(u32),
        Datatype::I32 => datatype_dispatch!(i32),
        Datatype::I64 => datatype_dispatch!(u64),
        Datatype::U64 => datatype_dispatch!(i64),
    })
}

fn benchmark_databin<T: Ord + Copy + Default>(
    r_data: &DataBin<T>,
    runs_per_trial: usize,
    r_algorithm_fn: &AlgorithmFn<T>,
) -> Result<results_schema::DataBinResultType, String> {
    match r_data {
        DataBin::Pair(r_trials) => {
            let mut trial_results =
                Vec::<results_schema::TrialResult>::with_capacity(r_trials.len());
            for r_trial in r_trials {
                let trial_result = benchmark_trial(r_trial, runs_per_trial, r_algorithm_fn)?;
                trial_results.push(trial_result);
            }
            Ok(results_schema::DataBinResultType::Pair(trial_results))
        }
        DataBin::Sample(r_samples) => {
            let mut sample_results =
                Vec::<results_schema::SampleResult>::with_capacity(r_samples.len());
            for r_trials in r_samples {
                let mut trial_results =
                    Vec::<results_schema::TrialResult>::with_capacity(r_trials.len());
                for r_trial in r_trials {
                    let trial_result = benchmark_trial(r_trial, runs_per_trial, r_algorithm_fn)?;
                    trial_results.push(trial_result);
                }
                sample_results.push(results_schema::SampleResult {
                    trials: trial_results,
                });
            }
            Ok(results_schema::DataBinResultType::Sample(sample_results))
        }
    }
}

fn benchmark_trial<T: Ord + Copy + Default>(
    r_trial: &Trial<T>,
    runs_per_trial: usize,
    r_algorithm_fn: &AlgorithmFn<T>,
) -> Result<results_schema::TrialResult, String> {
    // Get sets in formats required for algorithms
    let (intersection, sets) = r_trial.split_last().unwrap();
    let sets_r_2set = (sets[0].as_slice(), sets[1].as_slice());
    let sets_r_kset: Vec<_> = sets.iter().map(|rv| rv.as_slice()).collect();

    let intersection_size = intersection.len();

    // Pre-initialised vectors for output values
    let mut runs_counter_deltas = vec![0u64; runs_per_trial];
    let mut outs = vec![vec![T::default(); intersection_size]; runs_per_trial];
    let mut buf = vec![T::default(); intersection_size];
    let count_out_iter = std::iter::zip(&mut runs_counter_deltas, &mut outs);

    // Do runs
    let pre_time_delta = Instant::now().duration_since(*START_INSTANT).as_micros();
    let pre_cycles_counter_delta = tsc::measure_cycles::<REFERENCE_CYCLES, REFERENCE_TRIALS>();
    match r_algorithm_fn {
        AlgorithmFn::TwoSet(r_algorithm_fn_2set) => {
            for (count, out) in count_out_iter {
                let intersection_size =
                    benchmark_2set(sets_r_2set, r_algorithm_fn_2set, out.as_mut_slice(), count);
                out.truncate(intersection_size);
            }
        }
        AlgorithmFn::KSetBuf(r_algorithm_fn_kset_buf) => {
            for (count, out) in count_out_iter {
                let intersection_size = benchmark_kset_buf(
                    sets_r_kset.as_slice(),
                    r_algorithm_fn_kset_buf,
                    out.as_mut_slice(),
                    buf.as_mut_slice(),
                    count,
                );
                out.truncate(intersection_size);
            }
        }
    }
    let post_cycles_counter_delta = tsc::measure_cycles::<REFERENCE_CYCLES, REFERENCE_TRIALS>();
    let post_time_delta = Instant::now().duration_since(*START_INSTANT).as_micros();

    // Check for intersection correctness
    for (index, out) in outs.iter().enumerate() {
        if !slice_equal(intersection, out.as_slice()) {
            return Err(format!(
                "Run {}: output differs from expected intersection.",
                index
            ));
        }
    }

    Ok(results_schema::TrialResult {
        pre: results_schema::FrequencyMeasurement {
            td: pre_time_delta,
            cc: pre_cycles_counter_delta,
        },
        deltas: runs_counter_deltas,
        post: results_schema::FrequencyMeasurement {
            td: post_time_delta,
            cc: post_cycles_counter_delta,
        },
    })
}

fn benchmark_2set<T: Ord + Copy>(
    sets_r: (&[T], &[T]),
    r_algorithm: &TwoSetAlgorithmFnGeneric<T>,
    mr_out: &mut [T],
    mr_counts: &mut u64,
) -> usize {
    let start = tsc::start();
    let size = r_algorithm(sets_r, mr_out);
    let end = tsc::end();

    *mr_counts = end - start;

    size
}

fn benchmark_kset_buf<T: Ord + Copy>(
    sets_r: &[&[T]],
    r_algorithm: &KSetAlgorithmBufFnGeneric<T>,
    mr_out: &mut [T],
    mr_buf: &mut [T],
    mr_counts: &mut u64,
) -> usize {
    let start = tsc::start();
    let size = r_algorithm(sets_r, mr_out, mr_buf);
    let end = tsc::end();

    *mr_counts = end - start;

    size
}
