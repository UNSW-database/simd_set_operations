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
pub mod schemas;

use serde::{Deserialize, Serialize};

pub type Set<T> = Vec<T>;
pub type Trial<T> = Vec<Set<T>>;
pub type Sample<T> = Vec<Trial<T>>;
pub type DataBinPair<T> = Vec<Trial<T>>;
pub type DataBinSample<T> = Vec<Sample<T>>;

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

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DataDistribution {
    Uniform {},
}

#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryDistribution {
    Zipf {},
}


#[derive(Serialize, Deserialize, Debug, PartialEq, Clone, Copy)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CorpusDistribution {
    Zipf {},
}

use std::fmt;

impl fmt::Display for Datatype {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for DataDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for QueryDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl fmt::Display for CorpusDistribution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

pub fn sets_from_trial_bytes<'a, 'b, T>(r_bytes: &'a [u8], r_trial: &'b schemas::dataset::TrialDescription) -> (Vec<&'a [T]>, &'a [T]) {
    let bytes = std::mem::size_of::<T>();
    let mut sets = Vec::<&[T]>::with_capacity(r_trial.set_lengths.len());
    let mut byte_offset = 0;
    for r_length in &r_trial.set_lengths {
        let ptr = &r_bytes[byte_offset] as *const u8 as *const T;
        let slice = unsafe { std::slice::from_raw_parts(ptr, *r_length as usize) };
        sets.push(slice);
        byte_offset += *r_length as usize * bytes;
    }
    let intersection_ptr = if r_trial.intersection_length != 0 {
        &r_bytes[byte_offset] as *const u8 as *const T
    } else {
        std::ptr::NonNull::<T>::dangling().as_ptr()
    };
    let r_intersection = unsafe { 
        std::slice::from_raw_parts(intersection_ptr, r_trial.intersection_length as usize) 
    };
    (sets, r_intersection)
}

