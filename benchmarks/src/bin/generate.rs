use benchmarks::schema::*;
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml")]
    experiment: std::path::PathBuf,
    #[arg(default_value = "datasets/")]
    datasets: std::path::PathBuf,
}

fn main() {
    let args = Cli::parse();
    if let Err(err) = generate(&args) {
        println!("{}", err);
    }
}

fn generate(cli: &Cli) -> Result<(), String> {
    let experiment_toml = std::fs::read_to_string(&cli.experiment)
        .map_err(|e| e.to_string())?;
    let experiments: Experiment = toml::from_str(&experiment_toml)
        .map_err(|e| e.to_string())?;

    dbg!(experiments);

    Ok(())
}
