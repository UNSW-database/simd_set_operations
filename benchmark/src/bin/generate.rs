use benchmark::{schema::*, datafile::{self, DatafileSet}, path_str, fmt_open_err, generators, format::{format_xlabel, format_x}, webdocs::generate_webdocs_dataset};
use clap::Parser;
use colored::*;
use indicatif::{ProgressStyle, MultiProgress, ProgressBar, ParallelProgressIterator};
use rayon::prelude::*;
use std::{path::PathBuf, fs, io};

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

fn maybe_generate_dataset(datasets: &PathBuf, info: &DatasetInfo)
    -> Result<(), String>
{
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

    match &info.dataset_type {
        DatasetType::Synthetic(s) => generate_synthetic_dataset(s, &dataset_path)?,
        DatasetType::Real(r)      => generate_webdocs_dataset(r, &dataset_path)?,
    }

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

fn generate_synthetic_dataset(info: &SyntheticDataset, path: &PathBuf)
    -> Result<(), String>
{
    let _ = fs::remove_dir_all(&path);
    let xvalues: Vec<u32> = benchmark::xvalues_synthetic(info).collect();

    let multi_progress = MultiProgress::new();

    let main_style =
        ProgressStyle::with_template("  Dispatched for {pos}/{len} x-values")
            .map_err(|e| e.to_string())?;

    let main_bar = ProgressBar::new(xvalues.len() as u64)
        .with_style(main_style);

    let main_bar = multi_progress.add(main_bar);

    let gen_errors: Vec<String> = xvalues
        .into_par_iter()
        .progress_with(main_bar)
        .map(move |x| generate_synthetic_for_x(x, &multi_progress, &path, &info))
        .map(|r| r.err())
        .flatten()
        .collect();

    if gen_errors.len() > 0 {
        Err(format!(
            "{} (and {} more errors)",
            gen_errors[0],
            gen_errors.len() - 1
        ))
    }
    else {
        Ok(())
    }
}

fn generate_synthetic_for_x(
    x: u32,
    multi_progress: &MultiProgress,
    path: &PathBuf,
    info: &SyntheticDataset) -> Result<(), String>
{
    let xdir = path.join(x.to_string());
    fs::create_dir_all(&xdir)
        .map_err(|e| format!(
            "failed to create directory {}:\n{}",
            xdir.to_str().unwrap_or("<unknown>"),
            e.to_string()
        ))?;

    let label = format!(
        "    {}: {:10} ",
        format_xlabel(info.vary),
        format_x(x, &info)
    );
    let style = ProgressStyle::with_template(&(label + "[{bar}] {pos}/{len}"))
        .map_err(|e| e.to_string())?
        .progress_chars("##-");

    let bar = ProgressBar::new(info.gen_count as u64)
        .with_style(style);
    let bar = multi_progress.add(bar);

    let errors: Vec<String> = (0..info.gen_count)
        .into_par_iter()
        .progress_with(bar)
        .map(|i| generate_synthetic_datafile(info, &xdir, x, i))
        .map(|r| r.err())
        .flatten()
        .collect();

    if errors.len() > 0 {
        Err(format!(
            "{} (and {} more errors)",
            errors[0],
            errors.len() - 1
        ))
    }
    else {
        Ok(())
    }
}

fn generate_synthetic_datafile(
    info: &SyntheticDataset,
    xdir: &PathBuf,
    x: u32,
    i: usize) -> Result<(), String>
{
    let props = benchmark::props_at_x(info, x);
    let sets = generate_synthetic_intersection(info, &props);

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

fn generate_synthetic_intersection(info: &SyntheticDataset, props: &IntersectionInfo)
    -> Vec<DatafileSet>
{
    if info.intersection.set_count == 2 {
        let (set_a, set_b) = generators::gen_twoset(props);
        vec![set_a, set_b]
    }
    else {
        generators::gen_kset(props)
    }
}
