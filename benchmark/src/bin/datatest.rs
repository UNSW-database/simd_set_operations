use std::{path::PathBuf, fs::{self, File}, io::Write};

use benchmark::{fmt_open_err, path_str, schema::*, datafile};
use clap::Parser;
use colored::Colorize;
use setops::intersect::{run_svs_generic, self};

const ERROR_MARGIN: f64 = 0.000001;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "datasets/", long)]
    datasets: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_datatest(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn run_datatest(cli: &Cli) -> Result<(), String> {
    let dir = fs::read_dir(&cli.datasets)
        .map_err(|e| fmt_open_err(e, &cli.datasets))?;

    for entry in dir {
        let entry = entry.map_err(|e| format!(
            "failed to get directory entry in {}: {}",
            path_str(&cli.datasets), e.to_string()
        ))?;

        let file_type = entry.file_type()
            .map_err(|e| format!(
                "failed to determine file's type in {}: {}",
                path_str(&cli.datasets), e.to_string()
            ))?;
        
        if file_type.is_dir() {
            let path = entry.path();
            let info = dataset_info(cli, &path)?;
            verify_dataset(&info, &path)?;
        }
    }

    Ok(())
}

fn dataset_info(cli: &Cli, dataset: &PathBuf) -> Result<DatasetInfo, String> {
    let dataset_name: String = dataset.file_name()
        .ok_or_else(|| "failed to get dataset name".to_string())?
        .to_str()
        .ok_or_else(|| "failed to convert dataset name to string".to_string())?
        .into();

    println!("{}", &dataset_name);

    let info_filename = dataset_name.clone() + ".json";
    let info_path = cli.datasets.join(&info_filename);

    let info_file = fs::File::open(&info_path)
        .map_err(|e| fmt_open_err(e, &info_path))?;

    let dataset_info: DatasetInfo =
        serde_json::from_reader(info_file)
        .map_err(|e| format!(
            "invalid json file {}: {}",
            path_str(&info_path), e.to_string()
        ))?;

    Ok(dataset_info)
}

fn verify_dataset(info: &DatasetInfo, dir: &PathBuf) -> Result<(), String> {

    dbg!(info);
    
    for x in benchmark::xvalues(info) {
        // later: look at throughput?
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());

        let xdir = dir.join(x.to_string());
        let pairs = fs::read_dir(&xdir)
            .map_err(|e| fmt_open_err(e, &xdir))?;

        let mut count = 0;

        for (i, pair_path) in pairs.enumerate() {

            let pair_path = pair_path
                .map_err(|e| format!(
                    "unable to open directory entry in {}: {}",
                    path_str(&xdir), e.to_string()
                ))?;

            let datafile_path = pair_path.path();
            let datafile = File::open(&datafile_path)
                .map_err(|e| fmt_open_err(e, &datafile_path))?;

            let sets = datafile::from_reader(datafile)
                .map_err(|e| format!(
                    "invalid datafile {}: {}",
                    path_str(&datafile_path),
                    e.to_string())
                )?;

            print!("{} ", i);
            let _ = std::io::stdout().flush();
            verify_datafile(&sets, &benchmark::props_at_x(&info, x));

            count += 1;
        }
        println!();

        if count != info.gen_count {
            error(&format!(
                "gen_count is {} but only got {} sets",
                info.gen_count, count
            ));
        }
    }
    Ok(())
}

fn verify_datafile(sets: &[Vec<i32>], info: &IntersectionInfo) {
    verify_set_count(sets, info);
    verify_sizes(sets, info);
    verify_density(sets, info);
    verify_selectivity(sets, info);
}

fn verify_set_count(sets: &[Vec<i32>], info: &IntersectionInfo) {
    if sets.len() != info.set_count as usize {
        error(&format!(
            "set_count is {} but only got {} sets",
            info.set_count, sets.len()
        ));
    }
}

fn verify_sizes(sets: &[Vec<i32>], info: &IntersectionInfo) {
    let largest = sets.last().unwrap().len();

    let max_len = 1 << info.max_len;

    if largest != max_len as usize {
        error(&format!(
            "expected largest set size {} but got {}",
            info.max_len, largest
        ));
    }

    for (i, set) in sets.iter().rev().enumerate() {
        let skewness_f = info.skewness_factor as f64 / PERCENT_F;
        let expect_factor = ((i+1) as f64).powf(skewness_f);

        let actual_factor = largest as f64 / set.len() as f64;

        if (expect_factor - actual_factor) > 0.1 {
            warn(&format!(
                "expected a skewness_factor of {} but got {} ({}th largest set)",
                expect_factor, actual_factor, i, 
            ));
        }
    }
}

// The closer the density is to 1.0, the lower the error should be
fn verify_density(sets: &[Vec<i32>], info: &IntersectionInfo) {
    let max_len = 1 << info.max_len;

    let expected_density = info.density as f64 / PERCENT_F;
    let expected_max = (max_len as f64 / expected_density) as i32;

    let actual_max = *sets.iter()
        .map(|s| s.iter().max().unwrap())
        .max().unwrap();

    if actual_max > expected_max {
        error(&format!(
            "expected max {} smaller than actual {} (expected density {})",
            expected_max, actual_max, expected_density
        ));
    }

    let diff = (expected_max - actual_max).abs();
    if diff > expected_max / 3 {
        warn(&format!(
            "expected max {} but got {} (expected density {})",
            expected_max, actual_max, expected_density
        ));
    }
}

// The closer the density is to 1.0, the lower the error should be
fn verify_selectivity(sets: &[Vec<i32>], info: &IntersectionInfo) {
    let smallest_len = sets.iter()
        .map(|s| s.len())
        .max().unwrap();

    let result_len =
        run_svs_generic(&sets, intersect::branchless_merge).len();

    let actual_selectivity = result_len as f64 / smallest_len as f64;
    let min_selectivity = info.selectivity as f64 / PERCENT_F;

    let message = format!(
        "expected selectivity {}, got {:.06}",
        min_selectivity, actual_selectivity);
    
    let diff = (actual_selectivity - min_selectivity).abs();

    if actual_selectivity < min_selectivity - ERROR_MARGIN * 2.0 {
        error(&message);
    }
    else if sets.len() == 2 && diff > ERROR_MARGIN {
        warn(&message);
    }
    else if diff > 0.1 {
        warn(&message);
    }
    //else {
    //    println!("{}", message);
    //}
}

fn error(text: &str) {
    println!("{}", text.red().bold());
}

fn warn(text: &str) {
    println!("{}", text.yellow().bold());
}
