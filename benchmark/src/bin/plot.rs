use clap::Parser;
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml")]
    experiment: PathBuf,
    #[arg(default_value = "results.json")]
    out: PathBuf,
}

fn main() {
    // Load results
    // Load experiment.toml
}
