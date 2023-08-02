use benchmark::{schema::*, datafile::{self, DatafileSet}, path_str, fmt_open_err, generators, format::{format_xlabel, format_x}};
use clap::Parser;
use colored::*;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use rayon::prelude::*;
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
        println!("{}", err.red().bold());
    }
    else {
        println!("{}", "Done".green().bold());
    }
}

impl Cli {
    fn clean(&self) -> io::Result<()> {
        let _ = fs::remove_dir_all(&self.datasets);
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
            maybe_generate_dataset(&self.datasets, dataset)?;
        }
        Ok(())
    }
}

fn maybe_generate_dataset(datasets: &PathBuf, info: &DatasetInfo) -> Result<(), String> {
    let dataset_path = datasets.join(&info.name);
    let info_path = datasets.join(info.name.clone() + ".json");

    // Check info file
    if let Ok(info_file) = fs::File::open(&info_path) {
        let existing_info: DatasetInfo =
            serde_json::from_reader(info_file)
            .map_err(|e| format!(
                "invalid json file {}: {}",
                path_str(&info_path), e.to_string()
            ))?;

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

    generate_dataset(info, dataset_path)?;

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

fn generate_dataset(
    info: &DatasetInfo,
    path: PathBuf) -> Result<(), String>
{
    let _ = fs::remove_dir_all(&path);

    for x in benchmark::xvalues(info) {

        let xdir = path.join(x.to_string());
        fs::create_dir_all(&xdir)
            .map_err(|e| format!(
                "failed to create directory {}:\n{}",
                xdir.to_str().unwrap_or("<unknown>"),
                e.to_string()
            ))?;

        let label = format!(
            "{}: {} ",
            format_xlabel(info.vary),
            format_x(x, &info)
        );
        let style = ProgressStyle::with_template(&(label + "[{bar}] {pos}/{len}"))
            .map_err(|e| e.to_string())?
            .progress_chars("=> ");
        
        let errors: Vec<String> = (0..info.gen_count)
            .into_par_iter()
            .progress_with_style(style)
            .map(|i| generate_datafile(info, &xdir, x, i))
            .map(|r| r.err())
            .flatten()
            .collect();

        if errors.len() > 0 {
            return Err(format!(
                "{} (and {} more errors)",
                errors[0],
                errors.len() - 1
            ));
        }
    }
    Ok(())
}

fn generate_datafile(
    info: &DatasetInfo,
    xdir: &PathBuf,
    x: u32,
    i: usize) -> Result<(), String>
{
    let props = benchmark::props_at_x(info, x);
    let sets = generate_intersection(info, &props);

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

    datafile::to_writer(dataset_file, &sets)
        .map_err(|e| e.to_string())?;
    
    Ok(())
}

fn generate_intersection(info: &DatasetInfo, props: &IntersectionInfo) -> Vec<DatafileSet> {
    let _ = io::stdout().flush();

    if info.intersection.set_count == 2 {
        let (set_a, set_b) = generators::gen_twoset(props);
        vec![set_a, set_b]
    }
    else {
        generators::gen_kset(props)
    }
}
