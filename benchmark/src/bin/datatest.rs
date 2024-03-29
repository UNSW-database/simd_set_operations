use std::{path::PathBuf, fs::{self, File}, io::Write};

use benchmark::{fmt_open_err, path_str, schema::*, datafile};
use clap::Parser;
use colored::Colorize;
use setops::intersect::{run_svs, self};

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "datasets/", long)]
    datasets: PathBuf,
    tests: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_datatest(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn run_datatest(cli: &Cli) -> Result<(), String> {

    if cli.tests.len() == 0 {
        return Err("please specify one or more datasets".to_string());
    }

    for dataset_name in &cli.tests {

        let dataset_path = cli.datasets.join(dataset_name);

        let info = dataset_info(cli, &dataset_name)?;
        verify_dataset(&info, &dataset_path)?;
    }

    Ok(())
}

fn dataset_info(cli: &Cli, dataset_name: &str) -> Result<DatasetInfo, String> {

    println!("{}", &dataset_name);

    let info_filename = dataset_name.to_string() + ".json";
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

            match &info.dataset_type {
                DatasetType::Synthetic(s) =>
                    verify_synthetic(&sets, &benchmark::props_at_x(s, x)),
                DatasetType::Real(_) => 
                    verify_real(&sets, x),
            }
        }
        println!();
    }
    Ok(())
}

fn verify_synthetic(sets: &[Vec<i32>], info: &IntersectionInfo) {

    print!("\n{}", "sizes: ".bold());
    for set in sets {
        print!("{}, ", set.len());
    }
    println!();

    verify_set_count(sets, info.set_count as usize);
    verify_sizes(sets, info);
    verify_density(sets, info);
    verify_selectivity(sets, info.selectivity);
    verify_sorted(sets);
}

fn verify_set_count(sets: &[Vec<i32>], set_count: usize) {
    if sets.len() != set_count as usize {
        error(&format!(
            "set_count is {} but only got {} sets",
            set_count, sets.len()
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

    let lengths_ascending =
        sets.windows(2).all(|s| s[0].len() <= s[1].len());
    if !lengths_ascending {
        error(&format!(
            "expected ascending lengths, got {}",
            sets.iter().map(|s| s.len().to_string())
                .fold(String::new(), |s, arg| s + &arg + ", ")
        ));
    }

    for (i, set) in sets.iter().rev().enumerate() {
        let skewness_f = info.skewness_factor as f64 / PERCENT_F;
        let expect_factor = ((i+1) as f64).powf(skewness_f);

        println!("i: {}, len: {}, skewness: {}, expect: {}",
            i, set.len(), skewness_f, expect_factor);

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
fn verify_selectivity(sets: &[Vec<i32>], selectivity: u32) {
    let smallest_len = sets.iter()
        .map(|s| s.len())
        .min().unwrap();

    assert!(sets[0].len() == smallest_len);

    let result_len =
        run_svs(&sets, intersect::branchless_merge).len();

    let selectivity = selectivity as f64 / PERCENT_F;
    let target_len = (smallest_len as f64 * selectivity) as usize;

    let message = format!(
        "expected result len {}, got {} (sel {:.06})",
        target_len, result_len, selectivity);
    
    if result_len < target_len {
        error(&message);
    }
    else if result_len > target_len {
        if sets.len() == 2 {
            error(&message);
        }
        else {
            warn(&message);
        }
    }
}

fn verify_sorted(sets: &[Vec<i32>]) {
    for (i, set) in sets.iter().enumerate() {
        let sorted = set.windows(2).all(|s| s[0] < s[1]);
        if !sorted {
            error(&format!("set {} not sorted", i));
        }
    }
}

fn verify_real(sets: &[Vec<i32>], set_count: u32) {

    print!("\n{}", "sizes: ".bold());
    for set in sets {
        print!("{}, ", set.len());
    }
    println!();

    verify_set_count(sets, set_count as usize);
    verify_sorted(sets);

    trace_selectivity(sets);
    trace_density(sets);
}

fn trace_selectivity(sets: &[Vec<i32>]) {
    let smallest_len = sets.iter()
        .map(|s| s.len())
        .min().unwrap();

    assert!(sets[0].len() == smallest_len);

    let result_len =
        run_svs(&sets, intersect::branchless_merge).len();

    let selectivity = result_len as f64 / smallest_len as f64;
    println!("selectivity: {:.4}", selectivity);
}

fn trace_density(sets: &[Vec<i32>]) {
    let max_len = sets.iter()
        .map(|s| s.len())
        .max().unwrap();

    let max_value = *sets.iter()
        .map(|s| s.iter().max().unwrap())
        .max().unwrap();

    let density = max_len as f64 / max_value as f64;
    println!("max density: {:.4}", density);
}

fn error(text: &str) {
    println!("{}", text.red().bold());
}

fn warn(text: &str) {
    println!("{}", text.yellow().bold());
}
