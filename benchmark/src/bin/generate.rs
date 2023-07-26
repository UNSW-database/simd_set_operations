use benchmark::{schema::*, datafile, path_str, fmt_open_err};
use clap::Parser;
use colored::*;
use std::{path::PathBuf, fs, io::{self, Write}};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml")]
    experiment: PathBuf,
    #[arg(default_value = "datasets/")]
    datasets: PathBuf,
    #[arg(long, action)]
    clean: bool,
}

fn main() {
    let cli = Cli::parse();

    let result = if cli.clean {
        cli.clean().map_err(|e| e.to_string())
    }
    else {
        cli.generate()
    };

    if let Err(err) = result {
        println!("{}", err.red());
    }
    else {
        println!("{}", "Done".green().bold());
    }
}

impl Cli {
    fn clean(&self) -> io::Result<()> {
        let _ = fs::remove_dir_all(self.datasets.join("2set"));
        Ok(())
    }

    fn generate(&self) -> Result<(), String> {
        let experiment_toml = fs::read_to_string(&self.experiment)
            .map_err(|e| fmt_open_err(e, &self.experiment))?;

        let experiments: Experiment = toml::from_str(&experiment_toml)
            .map_err(|e| format!(
                "invalid toml file {}: {}",
                path_str(&self.experiment), e
            ))?;

        for dataset in &experiments.dataset {
            match dataset {
                DatasetInfo::TwoSet(info) => generate_twoset(&self.datasets, info)?,
                DatasetInfo::KSet(_) => todo!(),
            }
        }
        Ok(())
    }
}

fn generate_twoset(datasets: &PathBuf, info: &TwoSetDatasetInfo) -> Result<(), String> {
    // Create directories
    let twoset = datasets.join("2set");
    fs::create_dir_all(&twoset).map_err(|e| e.to_string())?;

    let dataset_path = twoset.join(&info.name);
    let info_path = twoset.join(info.name.clone() + ".json");

    // Check info file
    if let Ok(info_file) = fs::File::open(&info_path) {
        let existing_info: TwoSetDatasetInfo =
            serde_json::from_reader(info_file)
            .map_err(|e| e.to_string())?;

        if existing_info == *info {
            println!("{} {}", "Skipping".bold(), info.name);
            return Ok(());
        }
        else {
            println!("{} {}", "Rebuilding".green().bold(), info.name);
        }
    }
    else {
        println!("{} {}", "Building".green().bold(), info.name);
    }

    build_twoset(info, dataset_path)?;

    // Write new info file
    let info_file = fs::File::options()
        .write(true)
        .truncate(true)
        .create(true)
        .open(&info_path)
        .map_err(|e| format!(
            "failed to open file {}:\n{}",
            info_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        ))?;

    serde_json::to_writer(info_file, info)
        .map_err(|e| e.to_string())?;

    Ok(())
}

fn build_twoset(
    info: &TwoSetDatasetInfo,
    path: PathBuf) -> Result<(), String>
{
    let _ = fs::remove_dir_all(&path);

    for x in benchmark::xvalues(info) {
        let label = format!("[x: {:4}] ", x);
        print!("{}", label.bold());

        let xdir = path.join(x.to_string());
        fs::create_dir_all(&xdir)
            .map_err(|e| format!(
                "failed to create directory {}:\n{}",
                xdir.to_str().unwrap_or("<unknown>"),
                e.to_string()
            ))?;

        for i in 0..info.count {
            let (set_a, set_b) = build_twoset_pair(info, x, i);
            let pair_path = xdir.join(i.to_string());

            let dataset_file = fs::File::options()
                .write(true)
                .truncate(true)
                .create(true)
                .open(&pair_path)
                .map_err(|e| format!(
                    "failed to open file {}:\n{}",
                    pair_path.to_str().unwrap_or("<unknown>"),
                    e.to_string()
                ))?;

            datafile::to_writer(dataset_file, &[set_a, set_b])
                .map_err(|e| e.to_string())?;
        }
        println!();
    }
    Ok(())
}

fn build_twoset_pair(info: &TwoSetDatasetInfo, x: u32, i: usize) -> SetPair {
    print!("{} ", i);
    let _ = io::stdout().flush();
    let mut props = info.props.clone();
    let prop = match info.vary {
        Parameter::Selectivity => &mut props.selectivity,
        Parameter::Density     => &mut props.density,
        Parameter::Size        => &mut props.size,
        Parameter::Skew        => &mut props.skew,
    };
    *prop = x;
    benchmark::generators::gen_twoset(&props)
}
