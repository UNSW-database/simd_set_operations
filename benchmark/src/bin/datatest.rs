#![feature(trait_alias)]

use std::{
    fmt::Display,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::PathBuf,
    sync::{Arc, Mutex},
};

use benchmark::{
    Datatype, sets_from_trial_bytes,
    algorithms::intersect,
    util::{is_ascending},
    schemas::dataset::*,
};
use clap::Parser;
use colored::Colorize;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use rayon::prelude::*;

trait Verifyable = Default + Copy + Display + PartialEq + Ord + TryFrom<u64>;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(short, long, help = "Dataset description JSON file.")]
    dataset_description: PathBuf,
    #[arg(short, long, help = "Generate only the single databin of the given index.")]
    single: Option<usize>,
    #[arg(long, default_value_t = false)]
    single_threaded: bool,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_datatest(&cli) {
        println!("{}: {e}", "ERROR".red().bold());
    }
}

fn run_datatest(cli: &Cli) -> Result<(), String> {
    print!("Checking dataset description... ");
    // Read and parse dataset description file
    let dataset_description = {
        let description_string = match fs::read_to_string(&cli.dataset_description) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Failed to read dataset description file {}: {}", cli.dataset_description.display(), e)),
        };
        let dataset_description: DatasetDescription = match serde_json::from_str(&description_string) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Invalid dataset description file {}: {}", cli.dataset_description.display(), e)),
        };
        dataset_description
    };

    verify_dataset_description(&dataset_description)?;
    println!("{}", "DONE".green().bold());

    print!("Checking dataset data... ");
    let data_path = if let Some(index) = cli.single {
        cli.dataset_description.with_extension(format!("{index}.data"))
    } else {
        cli.dataset_description.with_extension("data")
    };

    let mut data_file = match File::open(&data_path) {
        Ok(v)  => v,
        Err(e) => return Err(format!("Failed to open data file {} for reading: {e}", data_path.display())),
    };
    let parallel_data_file = Arc::new(Mutex::new(&mut data_file));

    let databin_closure = |(index, r_databin_description)| {
        match test_databin(r_databin_description, parallel_data_file.clone()) {
            Ok(()) => Ok(()),
            Err(e) => Err(format!("Data bin #{}: {e}", index + 1)),
        }
    };

    let databin_count = dataset_description.databins.len() as u64;
    if cfg!(debug_assertions) {
        dataset_description.databins
            .iter()
            .enumerate()
            .progress_count(databin_count)
            .try_for_each(databin_closure)?;
    } else {
        dataset_description.databins
            .par_iter()
            .enumerate()
            .progress_count(databin_count)
            .try_for_each(databin_closure)?;
    }
    println!("{}", "DONE".green().bold());

    Ok(())
}

fn verify_dataset_description(r_dataset_description: &DatasetDescription) -> Result<(), String> {
    // Verify:
    // - all byte_lengths and byte_offsets agree
    // - no kset trials are present in non-kset marked datasets
    // - listed datatypes match present datatypes

    let mut kset = false;
    let mut datatypes = Vec::<String>::new();

    let mut byte_offset = 0u64;
    for (databin_index, r_databin) in r_dataset_description.databins.iter().enumerate() { 
    (|| {
        if byte_offset != r_databin.byte_offset {
            return Err(format!("Calculated ({}) and given ({}) databin byte offset disagree.", byte_offset, r_databin.byte_offset));
        }
        byte_offset += r_databin.byte_length;

        let mut trial_byte_offset = 0u64;
        for (trial_index, r_trial) in r_databin.trials.iter().enumerate() {
        (|| {
            if trial_byte_offset != r_trial.byte_offset {
                return Err(format!("Calculated ({}) and given ({}) trial byte offset disagree.", trial_byte_offset, r_trial.byte_offset));
            }
            trial_byte_offset += r_trial.byte_length;
            if r_trial.set_lengths.len() != 2 {
                kset = true;
            }

            // Check that the trial set count is at least 2
            if r_trial.set_lengths.len() < 2 {
                return Err(format!("Must have at least 2 sets."));
            }

            // Check that the sets are in non-decreasing length order
            for r_pair in r_trial.set_lengths.as_slice().windows(2) {
                if r_pair[0] < r_pair[1] {
                    return Err(format!("Set lengths are not in non-decreasing order."));
                }
            }

            // Check that the trial byte length matches the set lengths
            {
                let mut total_length = r_trial.intersection_length;
                for &set_length in &r_trial.set_lengths {
                    total_length += set_length;
                }
                let byte_length = total_length * r_databin.datatype.bytes();
                if byte_length != r_trial.byte_length {
                    return Err(format!("Calculated ({}) and expected ({}) trial byte length differ.", byte_length, r_trial.byte_length));
                }
            }

            // Check that the intersection length is <= smallest set length
            if *r_trial.set_lengths.last().unwrap() < r_trial.intersection_length {
                return Err(format!("Intersection length is larger than smallest set length."));
            }

            // Warn on high selectivity
            if *r_trial.set_lengths.first().unwrap() as f64 / r_databin.max_value as f64 > 0.1 {
                println!("{}: Databin #{}: Trial #{}: Densities approaching 0.1 and greater will take a very long time to generate.", "WARNING".yellow().bold(), databin_index + 1, trial_index + 1);
            }

            Ok(())
        })().map_err(|e| format!("Trial #{}: {e}", trial_index + 1))?;
        }

        if trial_byte_offset != r_databin.byte_length {
            return Err(format!("Calculated ({}) and given ({}) databin byte length disagree.", byte_offset, r_dataset_description.byte_length));
        }

        let datatype_s = r_databin.datatype.to_string();
        if !datatypes.contains(&datatype_s) {
            datatypes.push(datatype_s);
        }

        if r_databin.datatype.max() < r_databin.max_value {
            return Err(format!("Max value ({}) too large for datatype ({:#?}).", r_databin.max_value, r_databin.datatype));
        }

        Ok(())
    })().map_err(|e| format!("Databin #{}: {e}", databin_index + 1))?;
    }

    if byte_offset != r_dataset_description.byte_length {
        return Err(format!("Calculated ({}) and given ({}) dataset byte length disagree.", byte_offset, r_dataset_description.byte_length));
    }

    if kset == true && r_dataset_description.kset == false {
        return Err(format!("K-set trials found in dataset marked as not k-set."));
    }

    let datatype_params = match &r_dataset_description.parameters {
        DatabinParameters::Pair(pair) => &pair.datatype,
        DatabinParameters::Sample(sample) => &sample.datatype,
    };

    for r_datatype in datatype_params.keys() {
        if !datatypes.contains(r_datatype) {
            return Err(format!("Expected datatype {:#?} was not found in dataset.", r_datatype));
        }
    }

    for r_datatype in &datatypes {
        if !datatype_params.contains_key(r_datatype) {
            return Err(format!("Found unexpected datatype {:#?} in dataset.", r_datatype))
        }
    }    

    return Ok(());
}

fn test_databin(
    r_databin_description : & DatabinDescription,
    parallel_data_file    :   Arc<Mutex<&mut File>>,
) -> Result<(), String> {
    let databin: Vec<u8> = {
        let byte_offset = r_databin_description.byte_offset;
        let byte_length = r_databin_description.byte_length as usize;

        let mut databin = vec![0u8; byte_length];

        let mut locked_data_file = parallel_data_file.lock().unwrap();
        if let Err(_) = locked_data_file.seek(SeekFrom::Start(byte_offset)) {
            return Err(format!("Failed to seek to {byte_offset} in data file."));
        }
        if let Err(e) = locked_data_file.read_exact(databin.as_mut_slice()) {
            return Err(format!("Failed reading from data file: {}", e.to_string()));
        }

        databin
    };

    return match r_databin_description.datatype {
        Datatype::U32 => test_databin_typed::<u32>(r_databin_description, databin),
        Datatype::I32 => test_databin_typed::<u32>(r_databin_description, databin),
        Datatype::U64 => test_databin_typed::<u32>(r_databin_description, databin),
        Datatype::I64 => test_databin_typed::<u32>(r_databin_description, databin),
    };
}

fn test_databin_typed<T: Verifyable>(
    r_databin_description: & DatabinDescription,
    databin              :   Vec<u8>,
) -> Result<(), String> {
    let max_value: T = match r_databin_description.max_value.try_into() {
        Ok(v)  => v,
        Err(_) => return Err(format!("Could not convert max_value ({}) to {:#?}.", r_databin_description.max_value, r_databin_description.datatype)),
    };

    for (trial_index, r_trial) in r_databin_description.trials.iter().enumerate() {
    (|| {
        // Extract sets and intersection
        let r_trial_data = {
            let trial_byte_offset = r_trial.byte_offset as usize;
            let trial_byte_offset_end = trial_byte_offset + r_trial.byte_length as usize;
            &databin[trial_byte_offset..trial_byte_offset_end]
        };

        let (sets, r_intersection) = sets_from_trial_bytes(r_trial_data, r_trial);

        // Within sets check that:
        // - values are in ascending order
        // - values are unique
        // - max value is <= given max value
        let set_check = |r_set| {
            if !is_ascending(r_set) {
                return Err(format!("Values are not in ascending order."));
            }
            if let Some(&set_max) = r_set.last() {
                if set_max > max_value {
                    return Err(format!("Largest value ({}) is greater than maximum ({}).", set_max, max_value));
                }
            }
            return Ok(());
        };
        for (set_index, r_set) in sets.iter().enumerate() {
            set_check(r_set).map_err(|e| format!("Set #{}: {e}", set_index + 1))?;
        }
        set_check(r_intersection).map_err(|e| format!("Intersection: {e}"))?;

        // Between sets:
        // - Verify intersection size
        // - Verify intersection correctness

        let mut intersection = vec![T::default(); sets[0].len()];
        let mut buffer = intersection.as_slice().to_vec();
        let size = intersect(sets.as_slice(), &mut intersection, &mut buffer);

        if size != r_trial.intersection_length as usize {
            return Err(format!("Expected intersection size of {} but found {}.", r_trial.intersection_length, size));
        }

        intersection.truncate(size);
        for i in 0..size {
            if intersection[i] != r_intersection[i] {
                return Err(format!("Calculated and given intersection differ."));
            }
        }

        return Ok(());
    })().map_err(|e| format!("Trial #{}: {e}", trial_index + 1))?;
    }

    return Ok(());
}
