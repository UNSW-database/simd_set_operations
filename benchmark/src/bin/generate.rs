use benchmark::{
    datafile::{self, DatafileSet},
    fmt_open_err,
    format::{format_x, format_xlabel},
    generators, path_str,
    realdata::generate_real_dataset,
    schema::*,
};
use clap::Parser;
use colored::*;
use indicatif::{MultiProgress, ParallelProgressIterator, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use serde::Serialize;
use std::{
    fs::{self, File},
    io,
    path::PathBuf,
    sync::{Arc, Mutex},
};

#[derive(Default, Serialize)]
struct DatasetMetadata {
    #[serde(skip_serializing_if = "Option::is_none")]
    low_byte_entropy: Option<f64>,
}

#[derive(Default)]
struct DatasetStatsCollector {
    low_byte: Mutex<LowByteHistogram>,
}

struct LowByteHistogram {
    counts: [u64; 256],
    total: u64,
}

impl Default for LowByteHistogram {
    fn default() -> Self {
        Self {
            counts: [0; 256],
            total: 0,
        }
    }
}

impl DatasetStatsCollector {
    fn observe(&self, sets: &[DatafileSet]) {
        if sets.is_empty() {
            return;
        }
        let mut guard = self.low_byte.lock().unwrap();
        for set in sets {
            guard.add_slice(set);
        }
    }

    fn finish(&self) -> DatasetMetadata {
        let guard = self.low_byte.lock().unwrap();
        DatasetMetadata {
            low_byte_entropy: guard.entropy(),
        }
    }
}

impl LowByteHistogram {
    fn add_slice(&mut self, data: &[i32]) {
        for value in data {
            let low = (*value as u32 & 0xFF) as usize;
            self.counts[low] += 1;
            self.total += 1;
        }
    }

    fn entropy(&self) -> Option<f64> {
        if self.total == 0 {
            return None;
        }
        let total = self.total as f64;
        let mut entropy = 0.0f64;
        for &count in &self.counts {
            if count == 0 {
                continue;
            }
            let p = count as f64 / total;
            entropy -= p * p.log2();
        }
        Some(entropy)
    }
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, default_value = "experiment.toml")]
    experiment: PathBuf,
    #[arg(long, default_value = "datasets/")]
    datasets: PathBuf,
    #[arg(long, action)]
    clean: bool,
}

fn main() {
    let cli = Cli::parse();

    let result = if cli.clean {
        cli.clean().map_err(|e| e.to_string())
    } else {
        cli.generate()
    };

    if let Err(err) = result {
        println!("{}", err.red().bold());
    } else {
        println!("{}", "Done".green().bold());
    }
}

impl Cli {
    fn clean(&self) -> io::Result<()> {
        let _ = fs::remove_dir_all(&self.datasets);
        Ok(())
    }

    fn generate(&self) -> Result<(), String> {
        let experiment_toml =
            fs::read_to_string(&self.experiment).map_err(|e| fmt_open_err(e, &self.experiment))?;

        let experiments: Experiment = toml::from_str(&experiment_toml)
            .map_err(|e| format!("invalid toml file {}: {}", path_str(&self.experiment), e))?;

        for dataset in &experiments.dataset {
            maybe_generate_dataset(&self.datasets, dataset)?;
        }
        Ok(())
    }
}

fn write_dataset_metadata(path: &PathBuf, metadata: &DatasetMetadata) -> Result<(), String> {
    let file = File::create(path).map_err(|e| fmt_open_err(e, path))?;
    serde_json::to_writer(file, metadata)
        .map_err(|e| format!("failed to write {}: {}", path_str(path), e))
}

fn maybe_generate_dataset(datasets: &PathBuf, info: &DatasetInfo) -> Result<(), String> {
    let dataset_path = datasets.join(&info.name);
    let info_path = datasets.join(info.name.clone() + ".json");

    // Check info file
    if let Ok(info_file) = File::open(&info_path) {
        let existing_info: DatasetInfo = serde_json::from_reader(info_file).map_err(|e| {
            format!(
                "invalid json file {}: {}",
                path_str(&info_path),
                e.to_string()
            )
        })?;

        if existing_info == *info {
            println!("{} {}", "Skipping".bold(), info.name);
            return Ok(());
        } else {
            println!("{} {}", "Rebuilding".green().bold(), info.name);
        }
    } else {
        println!("{} {}", "Building".green().bold(), info.name);
    }

    let metadata = match &info.dataset_type {
        DatasetType::Synthetic(s) => generate_synthetic_dataset(s, &dataset_path)?,
        DatasetType::Real(r) => {
            generate_real_dataset(r, datasets, &dataset_path)?;
            DatasetMetadata::default()
        }
    };

    // Write new info file
    let info_file = File::create(&info_path).map_err(|e| {
        format!(
            "failed to open file {}:\n{}",
            info_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        )
    })?;

    serde_json::to_writer(info_file, info).map_err(|e| e.to_string())?;

    let meta_path = datasets.join(info.name.clone() + ".stats.json");
    write_dataset_metadata(&meta_path, &metadata)?;

    Ok(())
}

fn generate_synthetic_dataset(
    info: &SyntheticDataset,
    path: &PathBuf,
) -> Result<DatasetMetadata, String> {
    let _ = fs::remove_dir_all(&path);
    let xvalues: Vec<u32> = benchmark::xvalues_synthetic(info).collect();

    let multi_progress = MultiProgress::new();

    let main_style = ProgressStyle::with_template("  Dispatched for {pos}/{len} x-values")
        .map_err(|e| e.to_string())?;

    let main_bar = ProgressBar::new(xvalues.len() as u64).with_style(main_style);

    let main_bar = multi_progress.add(main_bar);

    let stats = Arc::new(DatasetStatsCollector::default());
    let gen_errors: Vec<String> = xvalues
        .into_par_iter()
        .progress_with(main_bar)
        .map(|x| generate_synthetic_for_x(x, &multi_progress, &path, &info, Arc::clone(&stats)))
        .map(|r| r.err())
        .flatten()
        .collect();

    if gen_errors.len() > 0 {
        Err(format!(
            "{} (and {} more errors)",
            gen_errors[0],
            gen_errors.len() - 1
        ))
    } else {
        Ok(stats.finish())
    }
}

fn generate_synthetic_for_x(
    x: u32,
    multi_progress: &MultiProgress,
    path: &PathBuf,
    info: &SyntheticDataset,
    stats: Arc<DatasetStatsCollector>,
) -> Result<(), String> {
    let xdir = path.join(x.to_string());
    fs::create_dir_all(&xdir).map_err(|e| {
        format!(
            "failed to create directory {}:\n{}",
            xdir.to_str().unwrap_or("<unknown>"),
            e.to_string()
        )
    })?;

    let label = format!(
        "    {}: {:10} ",
        format_xlabel(info.vary),
        format_x(x, &info)
    );
    let style = ProgressStyle::with_template(&(label + "[{bar}] {pos}/{len}"))
        .map_err(|e| e.to_string())?
        .progress_chars("##-");

    let bar = ProgressBar::new(info.gen_count as u64).with_style(style);
    let bar = multi_progress.add(bar);

    let props = benchmark::props_at_x(info, x);

    let errors: Vec<String> = (0..info.gen_count)
        .into_par_iter()
        .progress_with(bar)
        .map(|i| generate_synthetic_datafile(&props, &xdir, i, Arc::clone(&stats)))
        .map(|r| r.err())
        .flatten()
        .collect();

    if errors.len() > 0 {
        Err(format!(
            "{} (and {} more errors)",
            errors[0],
            errors.len() - 1
        ))
    } else {
        Ok(())
    }
}

fn generate_synthetic_datafile(
    props: &IntersectionInfo,
    xdir: &PathBuf,
    i: usize,
    stats: Arc<DatasetStatsCollector>,
) -> Result<(), String> {
    let sets = generate_synthetic_intersection(&props);
    stats.observe(&sets);

    let pair_path = xdir.join(i.to_string());

    let dataset_file = File::create(&pair_path).map_err(|e| {
        format!(
            "failed to open file {}:\n{}",
            pair_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        )
    })?;

    datafile::to_writer(dataset_file, &sets).map_err(|e| e.to_string())?;

    Ok(())
}

fn generate_synthetic_intersection(props: &IntersectionInfo) -> Vec<DatafileSet> {
    if props.set_count == 2 {
        let (set_a, set_b) = generators::gen_twoset(props);
        vec![set_a, set_b]
    } else {
        generators::gen_kset(props)
    }
}
