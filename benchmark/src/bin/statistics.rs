use std::{
    fs::File,
    path::PathBuf,
    time::{self, SystemTime},
};

use benchmark::{
    fmt_open_err, path_str,
    tsc,
};
use clap::Parser;
use colored::*;
use serde::Serialize;

type Ensemble = Vec<u64>;

#[derive(Serialize, Debug)]
struct Results {
    data: Vec<Ensemble>,
    tsc_freq: u64,
    tsc_overhead: u64,
}


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    trials: usize,
    #[arg(long)]
    ensembles: usize,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_stats(cli) {
        let msg = format!("ERROR: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn run_stats(cli: Cli) -> Result<(), String> {
    if !tsc::is_valid() {
        return Err("CPU does not support invariant Time Stamp Counter (TSC).".to_owned());
    }

    let tsc_frequency = tsc::estimate_frequency();
    let tsc_overhead = tsc::measure_overhead();

    // warmup
    for _ in 0..(3 * cli.trials) {
        tsc::measure_cpu_frequency(tsc_frequency, tsc_overhead);
    }

    // measurement
    let mut data = Vec::<Ensemble>::with_capacity(cli.ensembles);
    for _ in 0..cli.ensembles {
        let mut ensemble = Ensemble::with_capacity(cli.trials);
        for _ in 0..cli.trials {
            ensemble.push(tsc::measure_cpu_frequency(tsc_frequency, tsc_overhead))
        }
        data.push(ensemble);
    }

    let results = Results {
        data,
        tsc_freq: tsc_frequency,
        tsc_overhead,
    };

    // write
    let time = SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
    let results_path = PathBuf::from(format!("statistics.{}.json", time.as_secs()));
    let results_file = File::create(&results_path).map_err(|e| fmt_open_err(e, &results_path))?;

    serde_json::to_writer(results_file, &results)
        .map_err(|e| format!("Failed to write {}: {}", path_str(&results_path), e))?;

    Ok(())
}
