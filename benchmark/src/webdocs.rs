use std::{path::PathBuf, fs::File, io::{BufReader, Read, self, BufRead}};

use flate2::bufread::GzDecoder;

use crate::{schema::*, datafile::DatafileSet, fmt_open_err};

const WEBDOCS_FILE: &str = "webdocs.dat";

type LinePositions = Vec<usize>;

pub fn generate_webdocs_dataset(info: &RealDataset, root: &PathBuf, dataset_path: &PathBuf)
    -> Result<(), String>
{
    let webdocs = open_webdocs(root)?;

    let reader = BufReader::new(webdocs);
    let maybe_sets: Result<Vec<DatafileSet>, String> = reader.lines()
        .map(|line| parse_line(
            line.map_err(|e| format!("unable to read line: {}", e.to_string()))?
        ))
        .collect();

    let sets = maybe_sets?;



    Ok(())
}

fn parse_line(line: String) -> Result<DatafileSet, String> {
    line
        .split_ascii_whitespace()
        .map(|number| number.parse::<i32>()
            .map_err(|e| format!("unable to parse integer: {}", e.to_string()))
        )
        .collect()
}

fn open_webdocs(datasets: &PathBuf) -> Result<File, String> {
    let path = datasets.join(WEBDOCS_FILE);
    File::open(&path)
        .map_err(|e| fmt_open_err(e, &path) + ", did you run fetch_webdocs.bash?")
}

fn generate_webdocs_intersection(sets: Vec<DatafileSet>, set_count: usize) -> Result<(), String> {


    todo!()
}

fn read_webdocs_intersection(
    set_count: usize,
    line_positions: &LinePositions,
    webdocs: ()) -> Result<Vec<DatafileSet>, String>
{
    (0..set_count)
        .map(|_| random_webdocs_set(&line_positions, webdocs))
        .collect()
}

fn random_webdocs_set(line_positions: &LinePositions, webdocs: ())
    -> Result<DatafileSet, String>
{


    todo!()
}
