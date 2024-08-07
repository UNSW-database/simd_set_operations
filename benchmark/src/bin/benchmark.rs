use std::{
    fs::{self, File},
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::Duration,
};
use benchmark::{
    fmt_open_err, path_str, get_algorithms,
    schema::*, datafile,
    timer::{
        Timer,
        harness::Harness,
        perf::PerfCounters,
    },
};
use clap::Parser;
use colored::*;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml", long)]
    experiment: PathBuf,
    #[arg(default_value = "datasets/", long)]
    datasets: PathBuf,
    #[arg(default_value = "results.json", long)]
    out: PathBuf,
    // Ignore --bench provided by cargo.
    #[arg(long, action)]
    bench: bool,
    experiments: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cfg!(debug_assertions) {
        println!("{}", "warning: running in debug mode".yellow().bold());
    }

    if let Err(e) = bench_from_files(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn bench_from_files(cli: &Cli) -> Result<(), String> {
    let experiment_toml = fs::read_to_string(&cli.experiment)
        .map_err(|e| fmt_open_err(e, &cli.experiment))?;

    let experiment: Experiment = toml::from_str(&experiment_toml)
        .map_err(|e| format!(
            "invalid toml file {}: {}",
            path_str(&cli.experiment), e
        ))?;

    let dataset_algos = gen_dataset_to_algos_map(cli, &experiment)?;
        
    if dataset_algos.len() == 0 {
        return Err("no algorithm matches found".to_string());
    }

    let results = run_experiments(cli, experiment, dataset_algos)?;
    
    write_results(results, &cli.out)?;

    Ok(())
}

type AlgorithmSet = HashSet<String>;
/// Map each dataset to algorithms which need to be run on it.
/// This saves us from running multiple dataset/algorithm pairs twice
/// if present in multiple experiments.
fn gen_dataset_to_algos_map(cli: &Cli, experiment: &Experiment)
    -> Result<HashMap<DatasetId, AlgorithmSet>, String>
{
    let mut dataset_algos: HashMap<String, AlgorithmSet> = HashMap::new();
    for e in &experiment.experiment {
        if cli.experiments.len() == 0 || cli.experiments.contains(&e.name) {

            let algorithms =
                get_algorithms(&experiment.algorithm_sets, &e.algorithms)?;

            dataset_algos
                .entry(e.dataset.clone())
                .or_default()
                .extend(algorithms.clone());
        }
    }
    Ok(dataset_algos)
}

fn run_experiments(
    cli: &Cli,
    experiment: Experiment,
    dataset_algos: HashMap<DatasetId, AlgorithmSet>)
    -> Result<Results, String>
{
    let mut results =
        HashMap::<DatasetId, DatasetResults>::new();

    let mut counters = PerfCounters::new();
    counters.summarise();

    for dataset in &experiment.dataset {
        if let Some(algos) = dataset_algos.get(&dataset.name) {
            let dataset_results = DatasetResults{
                info: dataset.clone(),
                algos: run_dataset_benchmarks(cli, &dataset, algos, &mut counters)?,
            };
            results.insert(dataset.name.clone(), dataset_results);
        }
    }

    let experiments = if cli.experiments.len() > 0 {
        experiment.experiment
            .into_iter()
            .filter(|e| cli.experiments.contains(&e.name))
            .collect()
    } else {
        experiment.experiment
    };

    Ok(Results{
        experiments: experiments,
        datasets: results,
        algorithm_sets: experiment.algorithm_sets,
    })
}

fn run_dataset_benchmarks(
    cli: &Cli,
    info: &DatasetInfo,
    algos: &HashSet<String>,
    counters: &mut PerfCounters) -> Result<AlgorithmResults, String>
{
    println!("{}", &info.name.green().bold());

    let dataset_dir = PathBuf::from(&cli.datasets)
        .join(&info.name);

    let mut algorithm_results: AlgorithmResults =
        algos.iter().map(|a| (a.clone(), Vec::new())).collect();

    for x in benchmark::xvalues(info) {
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());
        let xdir = dataset_dir.join(x.to_string());

        for (name, runs) in &mut algorithm_results {
            println!("  {}", name);

            let pairs: Result<Vec<PathBuf>, String> = fs::read_dir(&xdir)
                .map_err(|e| fmt_open_err(e, &xdir))?
                .map(|s| s
                    .map_err(|e| format!(
                        "unable to open directory entry in {}: {}",
                        path_str(&xdir), e.to_string()
                    ))
                    .map(|s| s.path())
                )
                .collect();

            let pairs = pairs?;

            if let Some(timer) = Timer::new(name) {
                let run = time_algorithm_on_x(x, timer, pairs, counters)?;
                runs.push(run);
            }
            else {
                println!("{}", format!("  unknown algorithm {}", name).yellow());
            }
        }
    }
    Ok(algorithm_results)
}

fn time_algorithm_on_x(
    x: u32,
    timer: Timer,
    datafile_paths: Vec<PathBuf>,
    counters: &mut PerfCounters)
    -> Result<ResultRun, String>
{
    let mut result = counters.new_result_run(x);

    for datafile_path in &datafile_paths {
        let datafile = File::open(datafile_path)
            .map_err(|e| fmt_open_err(e, datafile_path))?;

        let sets = datafile::from_reader(datafile)
            .map_err(|e| format!(
                "invalid datafile {}: {}",
                path_str(datafile_path),
                e.to_string())
            )?;

        const TARGET_WARMUP: Duration = Duration::from_millis(1000);
        let warmup = TARGET_WARMUP.div_f32(datafile_paths.len() as f32);

        let mut harness = Harness::new(warmup, counters);
        let run_result = timer.run(&mut harness, &sets);

        match run_result {
            Ok(run) => {
                let perf = &run.perf;

                result.times.push(run.time.as_nanos() as u64);
                if let Some(v) = &mut result.l1d.rd_access { v.push(perf.l1d.rd_access.unwrap()); }
                if let Some(v) = &mut result.l1d.rd_miss { v.push(perf.l1d.rd_miss.unwrap()); }
                if let Some(v) = &mut result.l1d.wr_access { v.push(perf.l1d.wr_access.unwrap()); }
                if let Some(v) = &mut result.l1d.wr_miss { v.push(perf.l1d.wr_miss.unwrap()); }

                if let Some(v) = &mut result.l1i.rd_access { v.push(perf.l1i.rd_access.unwrap()); }
                if let Some(v) = &mut result.l1i.rd_miss { v.push(perf.l1i.rd_miss.unwrap()); }
                if let Some(v) = &mut result.l1i.wr_access { v.push(perf.l1i.wr_access.unwrap()); }
                if let Some(v) = &mut result.l1i.wr_miss { v.push(perf.l1i.wr_miss.unwrap()); }

                if let Some(v) = &mut result.ll.rd_access { v.push(perf.ll.rd_access.unwrap()); }
                if let Some(v) = &mut result.ll.rd_miss { v.push(perf.ll.rd_miss.unwrap()); }
                if let Some(v) = &mut result.ll.wr_access { v.push(perf.ll.wr_access.unwrap()); }
                if let Some(v) = &mut result.ll.wr_miss { v.push(perf.ll.wr_miss.unwrap()); }

                if let Some(v) = &mut result.branches { v.push(perf.branches.unwrap()); }
                if let Some(v) = &mut result.branch_misses { v.push(perf.branch_misses.unwrap()); }

                if let Some(v) = &mut result.cpu_stalled_front { v.push(perf.cpu_stalled_front.unwrap()); }
                if let Some(v) = &mut result.cpu_stalled_back { v.push(perf.cpu_stalled_back.unwrap()); }
                if let Some(v) = &mut result.instructions { v.push(perf.instructions.unwrap()); }
                if let Some(v) = &mut result.cpu_cycles { v.push(perf.cpu_cycles.unwrap()); }
                if let Some(v) = &mut result.cpu_cycles_ref { v.push(perf.cpu_cycles_ref.unwrap()); }

                result.bytes.push(run.bytes as u64);
            },
            Err(e) => {
                println!("warn: {}", e);
                break;
            },
        }
    }

    Ok(result)
}

fn write_results(results: Results, path: &PathBuf) -> Result<(), String> {
    let results_file = File::options()
        .write(true).create(true).truncate(true)
        .open(path)
        .map_err(|e| fmt_open_err(e, path))?;

    serde_json::to_writer(results_file, &results)
        .map_err(|e| format!(
            "failed to write {}: {}",
            path_str(path), e.to_string()
        ))?;

    Ok(())
}
