use std::{
    arch::asm,
    fs::File,
    path::PathBuf,
    time::{self, SystemTime},
};

use benchmark::{
    fmt_open_err, path_str,
    rdtscp::{end, estimate_tsc_freq, start},
};
use clap::Parser;
use colored::*;
use serde::Serialize;

type Ensemble = Vec<u64>;

#[derive(Serialize, Debug)]
struct Results {
    data: Vec<Ensemble>,
    tsc_freq: u64,
}


#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    cycles: usize,
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
    let tsc_freq = estimate_tsc_freq();

    // warmup
    for _ in 0..(3 * cli.trials) {
        trial(cli.cycles);
    }

    // measurement
    let mut data = Vec::<Ensemble>::with_capacity(cli.ensembles);
    for _ in 0..cli.ensembles {
        let mut ensemble = Ensemble::with_capacity(cli.trials);
        for _ in 0..cli.trials {
            ensemble.push(trial(cli.cycles))
        }
        data.push(ensemble);
    }

    let results = Results {
        data,
        tsc_freq,
    };

    // write
    let time = SystemTime::now().duration_since(time::UNIX_EPOCH).unwrap();
    let results_path = PathBuf::from(format!("statistics.{}.json", time.as_secs()));
    let results_file = File::create(&results_path).map_err(|e| fmt_open_err(e, &results_path))?;

    serde_json::to_writer(results_file, &results)
        .map_err(|e| format!("Failed to write {}: {}", path_str(&results_path), e))?;

    Ok(())
}

fn trial(cycles: usize) -> u64 {
    let start = start();

    let mut sum: u64 = 0;
    for _ in 0..cycles {
        sum = inc(sum);
    }

    let end = end();
    end - start
}

#[inline(always)]
pub fn inc(mut num: u64) -> u64 {
    unsafe {
        asm!(
            "add {val}, 1",
            val = inout(reg) num,
        )
    }
    num
}