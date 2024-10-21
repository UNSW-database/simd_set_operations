use benchmark::{
    algorithms::{Algorithm, AlgorithmFn, AlgorithmType, ALGORITHMS},
    fmt_open_err,
    path_str,
    read_databin,
    read_dataset_description,
    tsc::{self, TSCCharacteristics},
    util::slice_equal,
    DataBin,
    DataBinDescription,
    DataSetDescription,
    Datatype,
    Sample,
    Trial,
};

use std::{
    collections::HashMap,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    time::{self, Instant, SystemTime},
    arch::asm,
};

use clap::Parser;
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use perf_event::{Builder, Group, Counter, events::{Hardware, Software}};

mod experiment_schema {
    use serde::Deserialize;
    use std::collections::HashMap;

    #[derive(Deserialize, Debug)]
    pub struct Config {
        pub algorithm_set : HashMap<String, AlgorithmSet>,
        pub experiment    : HashMap<String, ExperimentConfig>,
    }

    #[derive(Deserialize, Debug, Default)]
    #[serde(default)]
    pub struct AlgorithmSet {
        pub twoset            : Vec<String>,
        pub twoset_to_kset    : Vec<String>,
        pub fsearch           : Vec<String>,
        pub fsearch_to_twoset : Vec<String>,
        pub fsearch_to_kset   : Vec<String>,
        pub dummy             : Vec<usize>,
    }

    #[derive(Deserialize, Debug)]
    pub struct ExperimentConfig {
        pub count_only          : bool,
        pub repeats_per_databin : usize,
        pub runs_per_trial      : usize,
        pub algorithm_sets      : Vec<String>,
    }
}

mod results_schema {
    use benchmark::tsc::TSCCharacteristics;
    use serde::Serialize;

    #[derive(Serialize, Debug)]
    pub struct Results {
        pub tsc_characteristics : TSCCharacteristics,
        pub reference_cycles    : u64,
        pub experiment_results  : Vec<ExperimentResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct ExperimentResult {
        pub experiment_name   : String,
        pub algorithm_results : Vec<AlgorithmResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct AlgorithmResult {
        pub algorithm_name : String,
        pub repeat_results : Vec<RepeatResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct RepeatResult {
        pub databin_results : Vec<Option<DataBinResult>>,
    }

    #[derive(Serialize, Debug)]
    pub struct DataBinResult {
        pub databin_index : usize,
        pub results       : DataBinResultType,
    }

    #[derive(Serialize, Debug)]
    #[serde(rename_all = "snake_case")]
    pub enum DataBinResultType {
        Pair   ( Vec<TrialResult> ),
        Sample ( Vec<SampleResult> ),
    }

    #[derive(Serialize, Debug)]
    pub struct SampleResult {
        pub trials : Vec<TrialResult>,
    }

    #[derive(Serialize, Debug)]
    pub struct TrialResult {
        pub pre_freq                : FrequencyMeasurement,
        pub cycles                  : Vec<u64>,
        pub cache_misses            : Vec<u64>,
        pub branch_misses           : Vec<u64>,
        // pub stalled_cycles_frontend : Vec<u64>,
        // pub stalled_cycles_backend  : Vec<u64>,
        pub page_faults             : Vec<u64>,
        pub context_switches        : Vec<u64>,
        pub cpu_migrations          : Vec<u64>,
        pub post_freq               : FrequencyMeasurement,
    }

    #[derive(Serialize, Debug)]
    pub struct FrequencyMeasurement {
        pub td : u128, // time delta
        pub cc : u64,  // cycles counts
    }
}

struct Experiment<'config, 'name> {
    r_name              : &'config str,
    count_only          : bool,
    runs_per_trial      : usize,
    repeats_per_databin : usize,
    algorithms_r        : Vec<(&'name str, &'name Algorithm)>,
}

// Store frequency in expected counts
struct FrequencyLimits {
    count_min : u64,
    count_max : u64,
    overhead  : u64,
}

struct PMC {
    group                   : Group,
    cycles                  : Counter,
    cache_misses            : Counter,
    branch_misses           : Counter,
    // stalled_cycles_frontend : Counter,
    // stalled_cycles_backend  : Counter,
    pub page_faults         : Counter,
    pub context_switches    : Counter,
    pub cpu_migrations      : Counter,
}

const NS_F64: f64 = 1_000_000_000.0;

const REFERENCE_CYCLES: u64 = 10_000;
const REFERENCE_TRIALS: usize = 3;

const MAX_FREQ_GHZ: f64 = (u64::MAX / NS_F64 as u64) as f64;


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    description: PathBuf,
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    experiment: Option<String>,
    #[arg(long, default_value_t = 0f64, help = "Minimum CPU frequency (GHz) at which runs are accepted.")]
    freq_min: f64,
    #[arg(long, default_value_t = MAX_FREQ_GHZ, help = "Maximum CPU frequency (GHz) at which runs are accepted.")]
    freq_max: f64,
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
    let config = {
        let config_string = match fs::read_to_string(&cli.config) {
            Ok(v) => v,
            Err(e) => return Err(fmt_open_err(e, &cli.config)),
        };
        let config: experiment_schema::Config = match toml::from_str(&config_string) {
            Ok(v) => v,
            Err(e) => return Err(format!("Invalid experiment config file {}: {}", path_str(&cli.config), e)),
        };
        config
    };

    // Read and parse dataset descriptions
    let dataset_description = match read_dataset_description(&cli.description) {
        Ok(v) => v,
        Err(e) => return Err(e),
    };

    // Open data file for later reading
    let mut data_file: File = {
        let data_file_path = cli.description.with_extension("data");
        match File::open(&data_file_path) {
            Ok(v) => v,
            Err(e) => return Err(fmt_open_err(e, &data_file_path)),
        }
    };

    // Filter the experiment configs based on the command line option
    let experiment_configs = {
        let mut experiment_configs = HashMap::<&str, &experiment_schema::ExperimentConfig>::new();
        if let Some(r_experiment_name) = &cli.experiment {
            match config.experiment.get(r_experiment_name) {
                Some(r_experiment) => experiment_configs.insert(r_experiment_name.as_str(), r_experiment),
                None => return Err(format!("Experiment ({}) not found in configuration.", r_experiment_name)),
            };
        } else {
            for (r_name, r_experiment) in &config.experiment {
                experiment_configs.insert(r_name.as_str(), r_experiment);
            }
        }
        experiment_configs
    };

    // Convert algorithm set config to a map from set name to a vec of algorithm names and implementations
    let algorithm_sets = {
        let mut algorithm_sets = HashMap::<&str, Vec<(String, Algorithm)>>::with_capacity(config.algorithm_set.len());

        fn lookup(name: String) -> Result<(String, Algorithm), String> {
            return match ALGORITHMS.get(&name) {
                Some(r_algorithm) => Ok((name, *r_algorithm)),
                None => Err(format!("There is no algorithm named {}.", name)),
            };
        }

        for (r_set_name, r_set) in &config.algorithm_set {
            let mut algorithms = Vec::<(String, Algorithm)>::new();
            for r_name in &r_set.twoset {
                let pair = lookup(r_name.to_owned())?;
                algorithms.push(pair);
            }
            for r_outer_name in &r_set.twoset_to_kset {
                for r_inner_name in &r_set.twoset {
                    let pair = lookup(format!("{}_{}", r_outer_name, r_inner_name))?;
                    algorithms.push(pair);
                }
            }
            for &count in &r_set.dummy {
                algorithms.push((format!("dummy_{}", count), Algorithm::ConstantTimeDummy(count)));
            }
            algorithm_sets.insert(r_set_name.as_str(), algorithms);
        }

        algorithm_sets
    };

    // Translate experiment configs into experiment structs for benchmarking use
    let experiments = {
        let mut experiments = Vec::<Experiment>::with_capacity(experiment_configs.len());
        for (&r_name, &r_config) in &experiment_configs {
            let mut algorithms_r = Vec::<(&str, &Algorithm)>::new();
            for r_set_name in &r_config.algorithm_sets {
                match algorithm_sets.get(r_set_name.as_str()) {
                    Some(r_set) => {
                        for r_algorithm in r_set {
                            algorithms_r.push((r_algorithm.0.as_str(), &r_algorithm.1));
                        }
                    },
                    None => return Err(format!("Could not find algorithm set: {}", r_set_name)),
                };
            }
            experiments.push(Experiment {
                r_name: r_name,
                count_only: r_config.count_only,
                runs_per_trial: r_config.runs_per_trial,
                repeats_per_databin: r_config.repeats_per_databin,
                algorithms_r: algorithms_r,
            });
        }
        experiments
    };

    // Timing stuff
    let tsc_characteristics = tsc::characterise();
    let freq_limits = {
        let numerator = tsc_characteristics.frequency * REFERENCE_CYCLES;
        let count_min = numerator / (cli.freq_max * NS_F64).round() as u64;
        let count_max = if cli.freq_min == 0f64 {
            u64::MAX
        } else {
            numerator / (cli.freq_max * NS_F64).round() as u64
        };
        FrequencyLimits {
            count_min,
            count_max,
            overhead: tsc_characteristics.overhead,
        }
    };

    let mut pmc = {
        let mut group = match Group::new() {
            Ok(group) => group,
            Err(e) => return Err(format!("Failed to create PMC group: {e}")),
        };
        let cycles = match group.add(&Builder::new(Hardware::CPU_CYCLES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC cycle counter: {e}")),
        };
        let cache_misses = match group.add(&Builder::new(Hardware::CACHE_MISSES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC cache miss counter: {e}")),
        };
        let branch_misses = match group.add(&Builder::new(Hardware::BRANCH_MISSES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC branch miss counter: {e}")),
        };
        /*
        let stalled_cycles_frontend = match group.add(&Builder::new(Hardware::STALLED_CYCLES_FRONTEND)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC frontend stalled cycles counter: {e}")),
        };
        let stalled_cycles_backend = match group.add(&Builder::new(Hardware::STALLED_CYCLES_BACKEND)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC backend stalled cycles counter: {e}")),
        };
        */
        let page_faults = match group.add(&Builder::new(Software::PAGE_FAULTS)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC page fault counter: {e}")),
        };
        let context_switches = match group.add(&Builder::new(Software::CONTEXT_SWITCHES).include_kernel()) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC context switch counter: {e}")),
        };
        let cpu_migrations = match group.add(&Builder::new(Software::CPU_MIGRATIONS)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC cpu migration counter: {e}")),
        };
        PMC {
            group,
            cycles,
            cache_misses,
            branch_misses,
            // stalled_cycles_frontend,
            // stalled_cycles_backend,
            page_faults,
            context_switches,
            cpu_migrations,
        }
    };

    let start_instant = Instant::now();

    // Run the benchmarks
    let results = run_benchmarks(
        &dataset_description,
        &mut data_file,
        &experiments,
        tsc_characteristics,
        &freq_limits,
        &start_instant,
        &mut pmc,
    )?;

    // Write results
    let time = SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
    let results_path = cli.description.with_extension(format!("results.{}.json", time.as_secs()));
    let results_file = match File::create(&results_path) {
        Ok(v) => v,
        Err(e) => return Err(fmt_open_err(e, &results_path)),
    };

    print!("Writing results... ");
    let _ = std::io::stdout().flush();
    if let Err(e) = serde_json::to_writer(results_file, &results) {
        println!();
        return Err(format!("Failed to write {}: {}", path_str(&results_path), e));
    };
    println!("DONE");

    Ok(())
}

fn run_benchmarks(
    r_dataset_description: &    DataSetDescription,
    mr_data_file:          &mut File,
    r_experiments:         &    Vec<Experiment>,
    tsc_characteristics:        TSCCharacteristics,
    r_freq_limits:         &    FrequencyLimits,
    r_start_instant:       &    Instant,
    mr_pmc:                &mut PMC,
) -> Result<results_schema::Results, String> {
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
    let mut experiment_results = Vec::<results_schema::ExperimentResult>::with_capacity(r_experiments.len());
    for r_experiment in r_experiments {
        experiment_bar.set_message(r_experiment.r_name.to_owned());
        update(&algorithm_bar, r_experiment.algorithms_r.len());
        let mut algorithm_results = Vec::<results_schema::AlgorithmResult>::with_capacity(r_experiment.algorithms_r.len());
        for &(r_name, r_algorithm) in &r_experiment.algorithms_r {
            algorithm_bar.set_message(r_name.to_owned());
            update(&repeat_bar, r_experiment.repeats_per_databin);
            let mut repeat_results = Vec::<results_schema::RepeatResult>::with_capacity(r_experiment.repeats_per_databin,);
            for repeat in 0..r_experiment.repeats_per_databin {
                repeat_bar.tick();
                update(&databin_bar, r_dataset_description.len());
                let mut databin_results = Vec::<Option<results_schema::DataBinResult>>::with_capacity(r_dataset_description.len());
                for (databin_index, r_databin_description) in r_dataset_description.iter().enumerate() {
                    tick(&[&experiment_bar, &algorithm_bar, &repeat_bar, &databin_bar]);

                    let results_result = datatype_dispatch(
                        r_algorithm,
                        r_experiment,
                        r_databin_description,
                        mr_data_file,
                        r_freq_limits,
                        r_start_instant,
                        mr_pmc,
                    );

                    let results_opt = match results_result {
                        Ok(opt) => opt,
                        Err(e) => return Err(format!(
                            "Experiment \"{}\": Algorithm \"{}\": Repeat {}: Databin {}: {}",
                            r_experiment.r_name, r_name, repeat, databin_index, e,
                        )),
                    };

                    let results_entry_opt = match results_opt {
                        Some(results) => Some(results_schema::DataBinResult {databin_index, results}),
                        None => None,
                    };

                    databin_results.push(results_entry_opt);

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

    // Set all of the progress bars to their finished state.
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
    r_algorithm           : &    Algorithm,
    r_experiment          : &    Experiment,
    r_databin_description : &    DataBinDescription,
    mr_data_file          : &mut File,
    r_freq_limits         : &    FrequencyLimits,
    r_start_instant       : &    Instant,
    mr_pmc                : &mut PMC,
) -> Result<Option<results_schema::DataBinResultType>, String> {
    macro_rules! datatype_dispatch {
        ($datatype:ident) => {{
            let algorithm_fn_opt = $datatype::algorithm_fn_from_algorithm(r_algorithm, r_experiment.count_only);
            if let Some(algorithm_fn) = algorithm_fn_opt {
                // Stop if we're trying to use k-set data with a 2-set algorithm
                if algorithm_fn.is_valid(r_databin_description.lengths.set_count()) {
                    let databin = read_databin::<$datatype, { std::mem::size_of::<$datatype>() }>(
                        r_databin_description,
                        mr_data_file,
                    )?;
                    Some(benchmark_databin::<$datatype>(
                        &databin,
                        r_experiment.runs_per_trial,
                        &algorithm_fn,
                        r_freq_limits,
                        r_experiment.count_only,
                        r_start_instant,
                        mr_pmc,
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
    r_data          : &    DataBin<T>,
    runs_per_trial  :      usize,
    r_algorithm_fn  : &    AlgorithmFn<T>,
    r_freq_limits   : &    FrequencyLimits,
    count_only      :      bool,
    r_start_instant : &    Instant,
    mr_pmc          : &mut PMC
) -> Result<results_schema::DataBinResultType, String> {
    match r_data {
        DataBin::Pair(r_trials) => {
            let trial_results = benchmark_sample(
                r_trials,
                runs_per_trial,
                r_algorithm_fn,
                r_freq_limits,
                count_only,
                r_start_instant,
                mr_pmc,
            )?;
            Ok(results_schema::DataBinResultType::Pair(trial_results))
        }
        DataBin::Sample(r_samples) => {
            let mut sample_results = Vec::<results_schema::SampleResult>::with_capacity(r_samples.len());
            for r_trials in r_samples {
                let trial_results = benchmark_sample(
                    r_trials,
                    runs_per_trial,
                    r_algorithm_fn,
                    r_freq_limits,
                    count_only,
                    r_start_instant,
                    mr_pmc,

                )?;
                sample_results.push(results_schema::SampleResult {
                    trials: trial_results,
                });
            }
            Ok(results_schema::DataBinResultType::Sample(sample_results))
        }
    }
}

fn benchmark_sample<T: Ord + Copy + Default>(
    r_trials        : &    Sample<T>,
    runs_per_trial  :      usize,
    r_algorithm_fn  : &    AlgorithmFn<T>,
    r_freq_limits   : &    FrequencyLimits,
    count_only      :      bool,
    r_start_instant : &    Instant,
    mr_pmc          : &mut PMC,
) -> Result<Vec<results_schema::TrialResult>, String> {
    let mut trial_results = Vec::<results_schema::TrialResult>::with_capacity(r_trials.len());
    for r_trial in r_trials {
        loop {
            let trial_result = benchmark_trial(
                r_trial,
                runs_per_trial,
                r_algorithm_fn,
                count_only,
                r_freq_limits,
                r_start_instant,
                mr_pmc,
            )?;

            // Check post measurement CPU frequency; stop looping if within bounds
            /*
            let post_cc = trial_result.post_freq.cc - r_freq_limits.overhead;
            if post_cc >= r_freq_limits.count_min && post_cc <= r_freq_limits.count_max {
                trial_results.push(trial_result);
                break;
            }
            */
            trial_results.push(trial_result);
            break;
        }
    }
    Ok(trial_results)
}

fn benchmark_trial<T: Ord + Copy + Default>(
    r_trial         : &    Trial<T>,
    runs_per_trial  :      usize,
    r_algorithm_fn  : &    AlgorithmFn<T>,
    count_only      :      bool,
    r_freq_limits   : &    FrequencyLimits,
    r_start_instant : &    Instant,
    mr_pmc          : &mut PMC,
) -> Result<results_schema::TrialResult, String> {
    let mut check_output = !count_only;

    // Get sets in formats required for algorithms
    let (intersection, sets) = r_trial.split_last().unwrap();
    let sets_r_2set = (sets[0].as_slice(), sets[1].as_slice());
    let sets_r_kset: Vec<_> = sets.iter().map(|rv| rv.as_slice()).collect();

    // We assume algorithms are correct and select the max intersection size
    // accordingly. An incorrect algorithm using unsafe code could write
    // outside these bounds. In this case because of k-set intersection the
    // max could be as big as the largest set, which should be the first.
    let max_intersection_size = sets[0].len();

    // Pre-initialised vectors for output values
    let mut cycles = vec![0u64; runs_per_trial];
    let mut cache_misses = vec![0u64; runs_per_trial];
    let mut branch_misses = vec![0u64; runs_per_trial];
    // let mut stalled_cycles_frontend = vec![0u64; runs_per_trial];
    // let mut stalled_cycles_backend = vec![0u64; runs_per_trial];
    let mut outs = vec![vec![T::default(); max_intersection_size]; runs_per_trial];
    let mut buf = vec![T::default(); max_intersection_size];
    let mut page_faults      = vec![0u64; runs_per_trial];
    let mut context_switches = vec![0u64; runs_per_trial];
    let mut cpu_migrations   = vec![0u64; runs_per_trial];

    // Disable output checking for the dummy algorithm
    if let AlgorithmFn::ConstantTimeDummy(_) = r_algorithm_fn {
        check_output = false;
    }

    // Measure CPU freq in loop until it's within the specified bound
    let pre_freq = {
        let td = Instant::now().duration_since(*r_start_instant).as_micros();
        let mut cc: u64;
        loop {
            cc = tsc::measure_cycles::<REFERENCE_CYCLES, REFERENCE_TRIALS>();
            let true_cc = cc - r_freq_limits.overhead;
            if true_cc >= r_freq_limits.count_min && true_cc <= r_freq_limits.count_max {
                break;
            }
        }
        results_schema::FrequencyMeasurement {td, cc}
    };

    // Do runs
    for i in 0..runs_per_trial {
        let mr_out = &mut outs[i];

        // Reset PMC counters
        if let Err(e) = mr_pmc.group.reset() {
            return Err(format!("Failed to reset counter group: {e}"));
        }

        // Take measurement. I'm assuming that having the group enable/disable
        // outside of the match statemnet has negligible overhead, but it
        // should be examined as a potential source of error for measurements
        // on small datasets.
        let e_enable = mr_pmc.group.enable();
        let intersection_size = match r_algorithm_fn {
            AlgorithmFn::TwoSet(r_algorithm_fn_2set) =>
                r_algorithm_fn_2set(sets_r_2set, mr_out),
            AlgorithmFn::KSetBuf(r_algorithm_fn_kset_buf) =>
                r_algorithm_fn_kset_buf(sets_r_kset.as_slice(), mr_out, buf.as_mut_slice()),
            AlgorithmFn::ConstantTimeDummy(r_dummy_counts) =>
                dummy_algo(*r_dummy_counts),
        };
        let e_disable = mr_pmc.group.disable();

        // We truncate the mr_out slice for the intersection correctness
        // checking step that runs later.
        mr_out.truncate(intersection_size);

        // Delayed handling of group enable/disable errors to reduce
        // potential overhead within the measurement section
        if let Err(e) = e_enable {
            return Err(format!("Failed to enable PMC counters: {e}"))
        }

        if let Err(e) = e_disable {
            return Err(format!("Failed to disable PMC counters: {e}"))
        }

        // Actually read the counters and store values
        let counters = match mr_pmc.group.read() {
            Ok(counters) => counters,
            Err(e) => return Err(format!("Failed to read PMC counters: {e}")),
        };

        cycles[i]                  = counters[&mr_pmc.cycles];
        cache_misses[i]            = counters[&mr_pmc.cache_misses];
        branch_misses[i]           = counters[&mr_pmc.branch_misses];
        // stalled_cycles_frontend[i] = counters[&mr_pmc.stalled_cycles_frontend];
        // stalled_cycles_backend[i]  = counters[&mr_pmc.stalled_cycles_backend];
        page_faults[i]      = counters[&mr_pmc.page_faults];
        context_switches[i] = counters[&mr_pmc.context_switches];
        cpu_migrations[i]   = counters[&mr_pmc.cpu_migrations];
    }

    // Post trial CPU frequency measurement
    let post_freq = {
        let cc = tsc::measure_cycles::<REFERENCE_CYCLES, REFERENCE_TRIALS>();
        let td = Instant::now().duration_since(*r_start_instant).as_micros();
        results_schema::FrequencyMeasurement {td, cc}
    };

    // Check for intersection correctness. We delay this to after the entire
    // trial has completed as it could affect caching and microarchitectural
    // state.
    if check_output {
        for (index, out) in outs.iter().enumerate() {
            if !slice_equal(intersection, out.as_slice()) {
                return Err(format!(
                    "Run {}: output differs from expected intersection.",
                    index
                ));
            }
        }
    }

    Ok(results_schema::TrialResult {
        pre_freq,
        cycles,
        cache_misses,
        branch_misses,
        // stalled_cycles_frontend,
        // stalled_cycles_backend,
        page_faults,
        context_switches,
        cpu_migrations,
        post_freq,
    })
}

// This will run to within a handful of cycles of dummy_counts on most
// architectures, though there are some recent intel architectures where it
// may run twice as fast. This doesn't matter hugely as long as it runs
// consistently.
#[inline(always)]
fn dummy_algo(dummy_counts: usize) -> usize {
    unsafe {
        asm!(
            "2:",
            "sub {val}, 1",
            "jne 2b",
            val = in(reg) dummy_counts,
        )
    }
    return 0;
}
