use std::{path::PathBuf, fs::{self, File}};

use benchmark::{fmt_open_err, path_str, schema::*, datafile};
use clap::Parser;
use colored::Colorize;

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
    
    for x in benchmark::xvalues(info) {
        // later: look at throughput?
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());

        let xdir = dir.join(x.to_string());
        let pairs = fs::read_dir(&xdir)
            .map_err(|e| fmt_open_err(e, &xdir))?;

        let mut times: Vec<u64> = Vec::new();

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

            println!("{}", i);
            // TODO
        }
    }

    Ok(())
}

fn verify_datafile(sets: Vec<Vec<i32>>) {

}
