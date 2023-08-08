use std::{
    path::PathBuf,
    fs::{File, self},
    io::{BufReader, BufRead}
};
use rand::{thread_rng, seq::SliceRandom};
use crate::{
    schema::*,
    datafile::{DatafileSet, self},
    fmt_open_err, path_str
};

const WEBDOCS_FILE: &str = "webdocs.dat";
const WEBDOCS_DATAFILE: &str = "webdocs.cache";

pub fn generate_webdocs_dataset(
    info: &RealDataset,
    root: &PathBuf,
    dataset_path: &PathBuf) -> Result<(), String>
{
    let webdocs_encoded_path = root.join(WEBDOCS_DATAFILE);

    let sets =
        if let Ok(webdocs_encoded) = File::open(&webdocs_encoded_path) {
            println!("Using cache");
            datafile::from_reader(webdocs_encoded)
                .map_err(|e| format!(
                    "unable to parse {}: {}",
                    path_str(&webdocs_encoded_path), e.to_string()
                ))?
        }
        else {
            println!("Cache not found, building...");
            parse_and_cache_webdocs(root, &webdocs_encoded_path)?
        };

    println!("Building intersections...");

    let _ = fs::remove_dir_all(&dataset_path);
    for count in info.set_count_start..=info.set_count_end {
        println!("  set count: {}", count);

        let xdir = dataset_path.join(count.to_string());
        fs::create_dir_all(&xdir)
            .map_err(|e| format!(
                "failed to create directory {}:\n{}",
                xdir.to_str().unwrap_or("<unknown>"),
                e.to_string()
            ))?;

        for i in 0..info.gen_count {
            generate_webdocs_intersection(&sets, &xdir, count as usize, i)?;
        }
    }

    Ok(())
}

fn parse_and_cache_webdocs(root: &PathBuf, webdocs_encoded_path: &PathBuf)
    -> Result<Vec<DatafileSet>, String>
{
    let webdocs_path = root.join(WEBDOCS_FILE);
    let webdocs_file = File::open(&webdocs_path)
        .map_err(|e|
            fmt_open_err(e, &webdocs_path) +
            ", did you run ./scripts/fetch_webdocs.bash ?"
        )?;

    let sets = parse_webdocs(webdocs_file)?;

    println!("Writing cache...");

    let webdocs_datafile = File::create(webdocs_encoded_path)
        .map_err(|e| format!(
            "unable to write webdocs datafile: {}",
            e.to_string()
        ))?;

    datafile::to_writer(webdocs_datafile, &sets)
        .map_err(|e| format!(
            "unable to parse webdocs datafile: {}", e.to_string()
        ))?;

    Ok(sets)
}

fn parse_webdocs(webdocs: File) -> Result<Vec<DatafileSet>, String> {
    let reader = BufReader::new(webdocs);

    reader
        .lines()
        .map(|line| parse_line(
            line.map_err(|e| format!("unable to read line: {}", e.to_string()))?
        ))
        .collect()
}

fn parse_line(line: String) -> Result<DatafileSet, String> {
    line
        .split_ascii_whitespace()
        .map(|number| number.parse::<i32>()
            .map_err(|e| format!("unable to parse integer: {}", e.to_string()))
        )
        .collect()
}

fn generate_webdocs_intersection(
    all_sets: &Vec<DatafileSet>,
    xdir: &PathBuf,
    set_count: usize,
    i: usize) -> Result<(), String>
{
    let rng = &mut thread_rng();

    let mut sets: Vec<&DatafileSet> = all_sets
        .choose_multiple(rng, set_count)
        .collect();

    sets.sort_by_key(|&s| s.len());
    
    let pair_path = xdir.join(i.to_string());

    let dataset_file = File::create(&pair_path)
        .map_err(|e| format!(
            "failed to open file {}:\n{}",
            pair_path.to_str().unwrap_or("<unknown>"),
            e.to_string()
        ))?;

    datafile::to_writer(dataset_file, &sets)
        .map_err(|e| e.to_string())?;
    
    Ok(())
}
