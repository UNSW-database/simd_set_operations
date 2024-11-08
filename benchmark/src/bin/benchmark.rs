use benchmark::{
    algorithms::{Algorithm, IntersectionAlgorithmLookup, constant_time_dummy},
    schemas::{self, results::*, dataset::*},
    util::{slice_equal, EqStatus},
    Datatype, sets_from_trial_bytes,
};

use std::{
    collections::HashSet,
    fs::{self, File},
    io::Write,
    path::PathBuf,
    hint::black_box,
    time,
};

use rand::{
    seq::SliceRandom,
    Rng, SeedableRng,
};

use clap::Parser;
use colored::*;
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use perf_event::{Builder, Group, Counter, events::Hardware};

#[derive(Default)]
struct Algorithms {
    u32 : Vec<Algorithm<u32>>,
    i32 : Vec<Algorithm<i32>>,
    u64 : Vec<Algorithm<u64>>,
    i64 : Vec<Algorithm<i64>>,
}

struct PMC {
    group           : Group,
    cycles          : Counter,
    ll_cache_misses : Counter,
    branch_misses   : Counter,
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, help = "Experiment configuration TOML file.")]
    config: PathBuf,
    #[arg(short, long, help = "Dataset description JSON file.")]
    dataset_description: PathBuf,
    #[arg(short, long, help = "Experiment from the config file to run.")]
    experiment: String,
    #[arg(short, long, help = "Note about this specific experimental run")]
    note: String,
    #[arg(long, default_value_t = false, help = "Check the correctness of the algorithms (may interfere with benchmarking results).")]
    check_correctness: bool,
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
            Ok(v)  => v,
            Err(e) => return Err(format!("Failed to read config file {}: {}", cli.config.display(), e)),
        };
        let config: schemas::experiment::Config = match toml::from_str(&config_string) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Invalid experiment config file {}: {}", cli.config.display(), e)),
        };
        config
    };

    // Read and parse dataset description file
    let dataset_description = {
        let description_string = match fs::read_to_string(&cli.dataset_description) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Failed to read dataset description file {}: {}", cli.dataset_description.display(), e)),
        };
        let dataset_description: schemas::dataset::DatasetDescription = match serde_json::from_str(&description_string) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Invalid dataset description file {}: {}", cli.dataset_description.display(), e)),
        };
        dataset_description
    };

    let r_experiment_name = cli.experiment.as_str();

    // Grab the selected experiment config from the list of experiment configs
    let r_experiment_config = match config.experiment.get(r_experiment_name) {
        Some(rv) => rv,
        None     => return Err(format!("Experiment {} not found in configuration.", &cli.experiment)),
    };

    // Setup RNG
    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(r_experiment_config.rng_seed);

    // Convert algorithm set config to unique lists of algorithm names and functions
    let (algorithm_names, algorithm_funcs) = {
        // First collate unique sets of algorithm names and dummy algorithm counts
        let mut algorithm_names = HashSet::<String>::new();
        let mut dummy_counts = HashSet::<usize>::new();

        for r_algorithm_set_name in &r_experiment_config.algorithm_sets {
            // Get the algorithm set spec
            let r_set = match config.algorithm_set.get(r_algorithm_set_name) {
                Some(rv) => rv,
                None     => return Err(format!(
                    "Experiment {} specifies algorithm set {} which does not exist.", 
                    r_experiment_name, r_algorithm_set_name
                )),
            };

            // Only include pure 2-set algorithms if not kset
            if !dataset_description.kset {
                for r_name in &r_set.twoset {
                    algorithm_names.insert(r_name.to_string());
                }
            }

            // 2-set to k-set composition
            for r_outer_name in &r_set.twoset_to_kset {
                for r_inner_name in &r_set.twoset {
                    let name = format!("{}_{}", r_outer_name, r_inner_name);
                    algorithm_names.insert(name);
                }
            }

            // Dummy algorithms
            for r_count in &r_set.dummy {
                dummy_counts.insert(*r_count);
            }
        }

        // Convert sets to vecs and sort for consistent ordering
        let mut algorithm_names_vec: Vec<_> = algorithm_names.into_iter().collect();
        let mut dummy_counts_vec: Vec<_> = dummy_counts.into_iter().collect();
        algorithm_names_vec.sort();
        dummy_counts_vec.sort();

        // Insert reference algorithm
        let mut reference = vec![r_experiment_config.reference.clone()];
        reference.extend(algorithm_names_vec);
        algorithm_names_vec = reference;

        // Get all of the algorithms required for all of the datatypes present in the dataset
        let mut algorithm_funcs = Algorithms::default();
        let datatype_params = match &dataset_description.parameters {
            DatabinParameters::Pair(pair) => &pair.datatype,
            DatabinParameters::Sample(sample) => &sample.datatype,
        };
        for r_datatype in datatype_params.keys() {
            match r_datatype.as_str() {
                "U32" => algorithm_funcs.u32 = u32::algorithms_from_names(&algorithm_names_vec)?,
                "I32" => algorithm_funcs.i32 = i32::algorithms_from_names(&algorithm_names_vec)?,
                "U64" => algorithm_funcs.u64 = u64::algorithms_from_names(&algorithm_names_vec)?,
                "I64" => algorithm_funcs.i64 = i64::algorithms_from_names(&algorithm_names_vec)?,
                _     => return Err(format!("Unknown datatype {r_datatype}.")),
            }
        }

        // Extend the algorithm names and functions with the dummy algorithms
        for dummy_count in dummy_counts_vec {
            algorithm_names_vec.push(format!("dummy_{dummy_count}"));
            algorithm_funcs.u32.push(Algorithm::<u32>::ConstantTimeDummy(dummy_count));
            algorithm_funcs.i32.push(Algorithm::<i32>::ConstantTimeDummy(dummy_count));
            algorithm_funcs.u64.push(Algorithm::<u64>::ConstantTimeDummy(dummy_count));
            algorithm_funcs.i64.push(Algorithm::<i64>::ConstantTimeDummy(dummy_count));
        }

        (algorithm_names_vec, algorithm_funcs)
    };

    // Prepare our performance counters
    let mut pmc = {
        let mut group = match Group::new() {
            Ok(group) => group,
            Err(e) => return Err(format!("Failed to create PMC group: {e}")),
        };
        let cycles = match group.add(&Builder::new(Hardware::CPU_CYCLES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC cycle counter: {e}")),
        };
        let ll_cache_misses = match group.add(&Builder::new(Hardware::CACHE_MISSES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC last-level cache miss counter: {e}")),
        };
        let branch_misses = match group.add(&Builder::new(Hardware::BRANCH_MISSES)) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to create PMC branch miss counter: {e}")),
        };
        PMC {
            group,
            cycles,
            ll_cache_misses,
            branch_misses,
        }
    };

    // Read the entire datafile into a byte vector
    let data: Vec<u8> = {
        let data_file_path = cli.dataset_description.with_extension("data");
        let data = match fs::read(&data_file_path) {
            Ok(v) => v,
            Err(e) => return Err(format!("Failed to read data file {}: {}", data_file_path.display(), e)),
        };
        data
    };

    // Extract any other needed data
    let repeats = r_experiment_config.repeats;
    let cache_warmups = r_experiment_config.cache_warmups;
    let check_correctness = cli.check_correctness;
    let r_note = cli.note.as_str();

    // Run the benchmarks
    let results = run_benchmarks(
             repeats,
             cache_warmups,
             check_correctness,
        &    dataset_description,
             r_experiment_name,
             algorithm_names,
        &    algorithm_funcs,
        &mut pmc,
        &mut rng,
        &    data,
             r_note,
    )?;

    // Create results file
    let time = time::SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
    let results_path = cli.dataset_description.with_extension(format!("results.{}.json", time.as_secs()));
    let results_file = match File::create(&results_path) {
        Ok(v) => v,
        Err(e) => return Err(format!("Failed to create results file {}: {}", results_path.display(), e)),
    };

    // Write results
    print!("Writing results... ");
    let _ = std::io::stdout().flush();
    if let Err(e) = serde_json::to_writer(results_file, &results) {
        println!();
        return Err(format!("Failed to write {}: {}", results_path.display(), e));
    };
    println!("DONE");

    Ok(())
}

fn run_benchmarks(
    repeats               :      u64,
    cache_warmups         :      u64,
    check_correctness     :      bool,
    r_dataset_description : &    DatasetDescription,
    r_experiment_name     : &    str,
    algorithm_names       :      Vec<String>,
    r_algorithm_funcs     : &    Algorithms,
    mr_pmc                : &mut PMC,
    mr_rng                : &mut impl Rng,
    r_data                : &    [u8],
    r_note                : &    str,
) -> Result<ExperimentResult, String> {
    // Iteration order:
    // 1. Repeat
    // 2. Databin
    // 3. Trial
    // 4. Algorithms (Randomized)

    let r_databins = &r_dataset_description.databins;

    // Set up multi-progress bar
    let multi_progress = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{prefix:10} [{elapsed_precise}] {wide_bar} {pos:>5}/{len:5} {msg:40}",
    ).unwrap();

    let create_bar_m = |prefix| {
        multi_progress.add({
            let bar = ProgressBar::hidden();
            bar.set_style(style.clone());
            bar.set_prefix(prefix);
            bar
        })
    };

    // Create bars in the multi-progress bar
    let repeat_bar = create_bar_m("Repeat");
    repeat_bar.reset();
    repeat_bar.set_length(repeats);

    let databin_bar = create_bar_m("Databin");
    databin_bar.reset();
    databin_bar.set_length(r_databins.len() as u64);

    let trial_bar = create_bar_m("Trial");
    // Trial bar length is dynamic so we don't deal with it here

    let mut repeat_results = Vec::<RepeatResult>::with_capacity(repeats as usize);
    for repeat_index in 0..repeats {

        databin_bar.reset();
        let mut databin_results = Vec::<DatabinResult>::with_capacity(r_databins.len());
        for databin_index in 0..r_databins.len() {

            let r_databin = &r_databins[databin_index];
            let trial_count = r_databin.trials.len();
            let databin_byte_start = r_databin.byte_offset as usize;

            trial_bar.reset();
            trial_bar.set_length(trial_count as u64);
            let mut trial_results = Vec::<TrialResult>::with_capacity(trial_count);
            for trial_index in 0..trial_count {

                // Tick on the innermost loop to update all of the times in sync
                repeat_bar.tick();
                databin_bar.tick();
                trial_bar.tick();

                let r_trial = &r_databin.trials[trial_index];
                let datatype = r_databin.datatype;

                let trial_byte_start = databin_byte_start + r_trial.byte_offset as usize;
                let trial_byte_end = trial_byte_start + r_trial.byte_length as usize;

                let r_trial_data = &r_data[trial_byte_start..trial_byte_end];

                let trial_result_result = benchmark_trial(
                    cache_warmups,
                    check_correctness,
                    datatype,
                    r_trial,
                    r_trial_data,
                    r_algorithm_funcs,
                    algorithm_names.as_slice(),
                    mr_pmc, 
                    mr_rng,
                );

                let trial_result = match trial_result_result {
                    Ok(v) => v,
                    Err(e) => return Err(format!(
                        "Repeat #{}: Databin #{}: Trial #{}: {}",
                        repeat_index + 1, databin_index + 1, trial_index + 1, e,
                    )),
                };

                trial_results.push(trial_result);
                trial_bar.inc(1);
            }
            databin_results.push(DatabinResult { trials: trial_results });
            databin_bar.inc(1);
        }
        repeat_results.push(RepeatResult { databins: databin_results });
        repeat_bar.inc(1);
    }

    // Set all of the progress bars to their finished state.
    repeat_bar.finish();
    databin_bar.finish();
    trial_bar.finish();

    return Ok(ExperimentResult {
        experiment : r_experiment_name.to_string(),
        algorithms : algorithm_names,
        repeats    : repeat_results,
        note       : r_note.to_string(),
    });
}

fn benchmark_trial(
    cache_warmups     :      u64,
    check_correctness :      bool,
    datatype          :      Datatype,
    r_trial           : &    TrialDescription,
    r_trial_data      : &    [u8],
    r_algorithm_funcs : &    Algorithms,
    r_algorithm_names : &    [impl AsRef<str>],
    mr_pmc            : &mut PMC,
    mr_rng            : &mut impl Rng,
) -> Result<TrialResult, String> {
    macro_rules! bti {
        ($type:ident) => {
            benchmark_trial_inner(
                cache_warmups, 
                check_correctness, 
                r_trial, 
                r_trial_data, 
                r_algorithm_funcs.$type.as_slice(), 
                r_algorithm_names, 
                mr_pmc, 
                mr_rng
            )
        };
    }
    return match datatype {
        Datatype::U32 => bti!(u32),
        Datatype::I32 => bti!(i32),
        Datatype::U64 => bti!(u64),
        Datatype::I64 => bti!(i64),
    };
}

fn benchmark_trial_inner<T: Ord + Copy + Default>(
    cache_warmups     :      u64,
    check_correctness :      bool,
    r_trial           : &    TrialDescription,
    r_trial_data      : &    [u8],
    r_algorithm_funcs : &    [Algorithm<T>],
    r_algorithm_names : &    [impl AsRef<str>],
    mr_pmc            : &mut PMC,
    mr_rng            : &mut impl Rng,
) -> Result<TrialResult, String> {
    // Extract set slices from r_trial_data
    let (sets, r_intersection) = sets_from_trial_bytes(r_trial_data, r_trial);

    // Get sets in formats required for algorithms
    let r_sets_r_kset = sets.as_slice();
    let sets_r_2set = (sets[0], sets[1]);

    // We assume algorithms are correct and select the max intersection size
    // accordingly. An incorrect algorithm using unsafe code could write
    // outside these bounds. In this case because of k-set intersection the
    // max could be as big as the largest set.
    let max_intersection_size = {
        let mut max_intersection_size = 0;
        for r_set in &sets {
            if r_set.len() > max_intersection_size {
                max_intersection_size = r_set.len();
            }
        } 
        max_intersection_size
    };

    let run_count = r_algorithm_funcs.len();

    // Pre-initialised vectors for measured counters
    let mut cycles          = vec![0u64; run_count];
    let mut ll_cache_misses = vec![0u64; run_count];
    let mut branch_misses   = vec![0u64; run_count];

    // Pre initialized vectors for output and buffer
    let mut out = vec![T::default(); max_intersection_size];
    let mut buf = vec![T::default(); max_intersection_size];

    // Randomise algorithm function index
    let algorithm_indices = {
        let mut algorithm_indices: Vec<u64> = (0..r_algorithm_funcs.len() as u64).collect();
        algorithm_indices.shuffle(mr_rng);
        algorithm_indices
    };

    // Cache warmup including output and buffer vectors
    for _ in 0..cache_warmups {
        for r_set in &sets {
            // Hopefully sufficient black boxing will prevent this from being elided
            black_box((&mut out[..r_set.len()]).copy_from_slice(black_box(r_set)));
            black_box((&mut buf[..r_set.len()]).copy_from_slice(black_box(r_set)));
        }
    };

    // Do runs
    for r_index in &algorithm_indices {
        let index = *r_index;
        let r_algorithm = &r_algorithm_funcs[index as usize];

        // Reset PMC counters
        if let Err(e) = mr_pmc.group.reset() {
            return Err(format!("Failed to reset counter group: {e}"));
        }

        // Take measurement. I'm assuming that having the group enable/disable
        // outside of the match statemnet has negligible overhead, but it
        // should be examined as a potential source of error for measurements
        // on small datasets.
        let e_enable = mr_pmc.group.enable();
        let intersection_size = match r_algorithm {
            Algorithm::TwoSet(r_algorithm_fn_2set) =>
                black_box(r_algorithm_fn_2set(
                    black_box(sets_r_2set), 
                    black_box(&mut out)
                )),
            Algorithm::KSetBuf(r_algorithm_fn_kset_buf) => 
                black_box(r_algorithm_fn_kset_buf(
                    black_box(r_sets_r_kset), 
                    black_box(&mut out), 
                    black_box(&mut buf)
                )),
            Algorithm::ConstantTimeDummy(r_dummy_counts) => {
                black_box(constant_time_dummy(black_box(*r_dummy_counts)));
                0
            },
        };
        let e_disable = mr_pmc.group.disable();

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

        cycles[index as usize]          = counters[&mr_pmc.cycles];
        ll_cache_misses[index as usize] = counters[&mr_pmc.ll_cache_misses];
        branch_misses[index as usize]   = counters[&mr_pmc.branch_misses];

        if check_correctness && r_algorithm.has_output() {
            let r_name = r_algorithm_names[index as usize].as_ref();
            let r_out_slice = &out[0..intersection_size];
            match slice_equal(r_intersection, r_out_slice) {
                EqStatus::Equal => (),
                EqStatus::DifferentLengths => return Err(format!(
                    "Algorithm {r_name}: output differs in length from expected intersection."
                )),
                EqStatus::DifferentAt(i) => return Err(format!(
                    "Algorithm {r_name}: output differs in value at index {i}.",
                )),
            };
        }
    }

    Ok(TrialResult {
        order: algorithm_indices,
        cycles,
        ll_cache_misses,
        branch_misses,
    })
    
}
