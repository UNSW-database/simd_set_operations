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

const TEXT_FILE_EXT: &str = ".dat";
const CACHE_EXT: &str = ".cache";

pub fn generate_real_dataset(
    info: &RealDataset,
    root: &PathBuf,
    dataset_path: &PathBuf) -> Result<(), String>
{
    let sets = load_sets(root, &info.source)?;

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
            generate_real_intersection(&sets, &xdir, count as usize, i)?;
        }
    }

    Ok(())
}

pub fn load_sets(root: &PathBuf, source: &str) -> Result<Vec<Vec<i32>>, String> {
    let cache_path = root.join(source.to_string() + CACHE_EXT);

    let sets = if let Ok(cache) = File::open(&cache_path) {
        println!("Using cache");
        datafile::from_reader(cache)
            .map_err(|e| format!(
                "unable to parse {}: {}",
                path_str(&cache_path), e.to_string()
            ))?
    }
    else {
        println!("Cache not found, building...");
        parse_and_cache_webdocs(root, source, &cache_path)?
    };

    Ok(sets)
}

fn parse_and_cache_webdocs(root: &PathBuf, source: &str, cache_path: &PathBuf)
    -> Result<Vec<DatafileSet>, String>
{
    let text_path = root.join(source.to_string() + TEXT_FILE_EXT);
    let text_file = File::open(&text_path)
        .map_err(|e|
            fmt_open_err(e, &text_path) +
            ", did you run ./scripts/fetch_*.bash ?"
        )?;

    let sets = parse_text(text_file)?;

    println!("Writing cache...");

    let cache = File::create(cache_path)
        .map_err(|e| format!(
            "unable to write datafile: {}",
            e.to_string()
        ))?;

    datafile::to_writer(cache, &sets)
        .map_err(|e| format!(
            "unable to parse datafile: {}", e.to_string()
        ))?;

    Ok(sets)
}

fn parse_text(text: File) -> Result<Vec<DatafileSet>, String> {
    let reader = BufReader::new(text);

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

fn generate_real_intersection(
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
