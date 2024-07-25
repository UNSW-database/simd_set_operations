#![feature(portable_simd)]
#![feature(array_chunks)]
#![feature(iter_map_windows)]

// pub mod datafile;
// pub mod format;
// pub mod generators;
// pub mod schema;
// pub mod timer;
// pub mod realdata;
pub mod algorithms;
pub mod tsc;
pub mod util;

use std::{
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    iter,
    path::PathBuf,
};

pub fn fmt_open_err(e: impl ToString, path: &PathBuf) -> String {
    format!("Unable to open {}: {}", path_str(path), e.to_string())
}

pub fn path_str(path: &PathBuf) -> &str {
    path.to_str().unwrap_or("<unknown path>")
}

pub type Set<T> = Vec<T>;
pub type Trial<T> = Vec<Set<T>>;
pub type Sample<T> = Vec<Trial<T>>;
pub type DataBinPair<T> = Vec<Trial<T>>;
pub type DataBinSample<T> = Vec<Sample<T>>;

use serde::{Deserialize, Serialize};
use util::{bytes_to_vec, to_usize, Byteable};

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
pub enum Datatype {
    U32,
    U64,
    I32,
    I64,
}

impl Datatype {
    pub fn bytes(&self) -> u64 {
        match self {
            Datatype::U32 => 4,
            Datatype::U64 => 8,
            Datatype::I32 => 4,
            Datatype::I64 => 8,
        }
    }

    pub fn max(&self) -> u64 {
        match self {
            Datatype::U32 => u32::MAX as u64,
            Datatype::U64 => u64::MAX,
            Datatype::I32 => i32::MAX as u64,
            Datatype::I64 => i64::MAX as u64,
        }
    }
}

pub type DataSetDescription = Vec<DataBinDescription>;

// Data bin configuration
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataBinDescription {
    pub datatype: Datatype,
    pub max_value: u64,
    pub lengths: DataBinLengthsEnum,
    pub distribution: DataDistribution,
    pub seed: u64,
    pub trials: u64,
    // byte offset and length in .data file
    pub byte_offset: u64,
    pub byte_length: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum DataBinLengthsEnum {
    Pair(DataBinLengths),
    Sample(Vec<DataBinLengths>),
}

impl DataBinLengthsEnum {
    pub fn set_count(&self) -> usize {
        match self {
            DataBinLengthsEnum::Pair(lengths) => lengths.set_count(),
            DataBinLengthsEnum::Sample(lengths_vec) => lengths_vec[0].set_count(),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct DataBinLengths {
    pub set_lengths: Vec<u64>,
    pub intersection_length: u64,
}

impl<'a> IntoIterator for &'a DataBinLengths {
    type Item = &'a u64;
    type IntoIter = std::iter::Chain<std::slice::Iter<'a, u64>, std::iter::Once<&'a u64>>;

    fn into_iter(self) -> Self::IntoIter {
        self.set_lengths
            .iter()
            .chain(std::iter::once(&self.intersection_length))
    }
}

impl DataBinLengths {
    pub fn set_count(&self) -> usize {
        self.set_lengths.len()
    }
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataDistribution {
    Uniform {},
}

//
// === FILE READING ===
//

pub fn read_dataset_description(path: &PathBuf) -> Result<DataSetDescription, String> {
    let desc_string = fs::read_to_string(path).map_err(|e| fmt_open_err(e, path))?;
    serde_json::from_str(&desc_string)
        .map_err(|e| format!("Invalid JSON file {}: {}", path_str(path), e))
}

pub enum DataBin<T> {
    Pair(DataBinPair<T>),
    Sample(DataBinSample<T>),
}

pub fn read_databin<T: Byteable<N>, const N: usize>(
    r_description: &DataBinDescription,
    mr_file: &mut File,
) -> Result<DataBin<T>, String> {
    let byte_offset = r_description.byte_offset;
    let byte_count = to_usize(r_description.byte_length, "byte_count")?;
    let trial_count = to_usize(r_description.trials, "trial_count")?;

    Ok(match &r_description.lengths {
        DataBinLengthsEnum::Pair(r_lengths) => {
            let data = read_databin_pair::<T, N>(
                r_lengths,
                byte_offset,
                byte_count,
                trial_count,
                mr_file,
            )?;
            DataBin::Pair(data)
        },
        DataBinLengthsEnum::Sample(r_lengths_vec) => {
            let data = read_databin_sample::<T, N>(
                r_lengths_vec,
                byte_offset,
                byte_count,
                trial_count,
                mr_file,
            )?;
            DataBin::Sample(data)
        }
    })
}

pub fn read_databin_pair<T: Byteable<N>, const N: usize>(
    lengths: &DataBinLengths,
    byte_offset: u64,
    byte_count: usize,
    trial_count: usize,
    file: &mut File,
) -> Result<DataBinPair<T>, String> {
    file.seek(SeekFrom::Start(byte_offset))
        .map_err(|e| format!("Failed to seek in data file: {}", e))?;

    let bytes = read_bytes(file, byte_count)?;
    let mut values = bytes_to_vec::<N, T>(&bytes);
    extract_trials(lengths, trial_count, &mut values)
}

pub fn read_databin_sample<T: Byteable<N>, const N: usize>(
    lengths_vec: &Vec<DataBinLengths>,
    byte_offset: u64,
    byte_count: usize,
    trial_count: usize,
    file: &mut File,
) -> Result<DataBinSample<T>, String> {
    file.seek(SeekFrom::Start(byte_offset))
        .map_err(|e| format!("Failed to seek in data file: {}", e))?;

    let bytes = read_bytes(file, byte_count)?;
    let mut values = bytes_to_vec::<N, T>(&bytes);

    let samples: DataBinSample<T> = lengths_vec
        .iter()
        .map(|lengths| extract_trials(lengths, trial_count, &mut values))
        .collect::<Result<DataBinSample<T>, String>>()?;

    Ok(samples)
}

fn read_bytes(file: &mut File, byte_count: usize) -> Result<Vec<u8>, String> {
    let mut byte_buf: Vec<u8> = iter::repeat(0).take(byte_count).collect();
    file.read_exact(&mut byte_buf)
        .map_err(|e| format!("Failed to read bytes: {}", e))?;
    Ok(byte_buf)
}

fn extract_trials<T>(
    lengths: &DataBinLengths,
    trial_count: usize,
    values: &mut Vec<T>,
) -> Result<Sample<T>, String> {
    Ok(iter::repeat_with(|| {
        lengths
            .into_iter()
            .map(|length_u64_r| {
                let length = to_usize(*length_u64_r, "set_length")?;
                Ok::<Vec<T>, String>(values.drain(0..length).collect::<Set<T>>())
            })
            .collect::<Result<Trial<T>, String>>()
    })
    .take(trial_count)
    .collect::<Result<Sample<T>, String>>()?)
}
