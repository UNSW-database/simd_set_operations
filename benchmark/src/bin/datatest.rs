#![feature(trait_alias)]

use std::{
    borrow::BorrowMut,
    fmt::Display,
    fs::File,
    iter,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use benchmark::{
    algorithms::get_kset_buf,
    fmt_open_err, read_databin_pair, read_databin_sample, read_dataset_description,
    util::{is_ascending, to_u64, to_usize, Byteable},
    DataBinDescription, DataBinLengthsEnum, DataBinPair, Datatype,
};
use clap::Parser;
use colored::Colorize;
use indicatif::{ParallelProgressIterator, ProgressIterator};
use rayon::prelude::*;
use setops::intersect::KSetAlgorithmBufFnGeneric;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    description: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    if let Err(e) = run_datatest(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn run_datatest(cli: &Cli) -> Result<(), String> {
    let dataset_description = read_dataset_description(&cli.description)?;

    let bin_path = cli.description.with_extension("data");
    let mut bin_file = File::open(&bin_path).map_err(|e| fmt_open_err(e, &bin_path))?;
    let parallel_bin_file = Arc::new(Mutex::new(&mut bin_file));

    let algo = get_kset_buf("svs_zipper_branch_loop_optimized");

    let databin_closure = |(db_num, data_bin_description): (usize, &DataBinDescription)| {
        match data_bin_description.datatype {
            Datatype::U32 => datatype_dispatch::<u32, { std::mem::size_of::<u32>() }>(
                data_bin_description,
                parallel_bin_file.clone(),
                algo.out.u32.unwrap(),
            ),
            Datatype::I32 => datatype_dispatch::<i32, { std::mem::size_of::<i32>() }>(
                data_bin_description,
                parallel_bin_file.clone(),
                algo.out.i32.unwrap(),
            ),
            Datatype::U64 => datatype_dispatch::<u64, { std::mem::size_of::<u64>() }>(
                data_bin_description,
                parallel_bin_file.clone(),
                algo.out.u64.unwrap(),
            ),
            Datatype::I64 => datatype_dispatch::<i64, { std::mem::size_of::<i64>() }>(
                data_bin_description,
                parallel_bin_file.clone(),
                algo.out.i64.unwrap(),
            ),
        }
        .map_err(|e| format!("Data bin #{}: {}", db_num + 1, e))
    };

    let databin_count = to_u64(dataset_description.len(), "databin_count")?;
    if cfg!(debug_assertions) {
        dataset_description
            .iter()
            .enumerate()
            .progress_count(databin_count)
            .try_for_each(databin_closure)?;
    } else {
        dataset_description
            .par_iter()
            .enumerate()
            .progress_count(databin_count)
            .try_for_each(databin_closure)?;
    }

    Ok(())
}

fn datatype_dispatch<T, const N: usize>(
    data_bin_description: &DataBinDescription,
    parallel_bin_file: Arc<Mutex<&mut File>>,
    algo: KSetAlgorithmBufFnGeneric<T>,
) -> Result<(), String>
where
    T: Byteable<N> + Verifyable + TryFrom<u64>,
{
    let byte_offset = data_bin_description.byte_offset;
    let byte_count = to_usize(data_bin_description.byte_length, "byte_count")?;
    let trial_count = to_usize(data_bin_description.trials, "trial_count")?;
    let max_value: T = data_bin_description.max_value.try_into().or(Err(format!(
        "Could not convert max_value ({}) to datatype.",
        data_bin_description.max_value
    )))?;

    match &data_bin_description.lengths {
        DataBinLengthsEnum::Pair(lengths) => {
            let data_bin = {
                let mut bin_file = parallel_bin_file.lock().unwrap();
                read_databin_pair::<T, N>(
                    lengths,
                    byte_offset,
                    byte_count,
                    trial_count,
                    bin_file.borrow_mut(),
                )
            }?;
            verify(&data_bin, max_value, algo)?;
        }
        DataBinLengthsEnum::Sample(lengths) => {
            let data_bin = {
                let mut bin_file = parallel_bin_file.lock().unwrap();
                read_databin_sample::<T, N>(
                    lengths,
                    byte_offset,
                    byte_count,
                    trial_count,
                    bin_file.borrow_mut(),
                )
            }?;
            for (sample_num, sample) in data_bin.iter().enumerate() {
                verify(sample, max_value, algo)
                    .map_err(|e| format!("Sample #{}: {}", sample_num + 1, e))?;
            }
        }
    }

    Ok(())
}

trait Verifyable = Default + Copy + Display + PartialEq + PartialOrd;

fn verify<T: Verifyable>(
    trials: &DataBinPair<T>,
    max_value: T,
    algo: KSetAlgorithmBufFnGeneric<T>,
) -> Result<(), String> {
    for (trial_num, trial) in trials.iter().enumerate() {
        (|| {
            // Between sets:
            // - Verify intersection size
            // - Verify intersection correctness
            if trial.len() < 3 {
                return Err(format!(
                    "Trial has a set count of {}, but it must be at least 3.",
                    trial.len()
                ));
            }

            for (set_num, set) in trial.iter().enumerate() {
                (|| {
                    // Within sets:
                    // - Verify ascending valuse
                    // - Verify max is not too large
                    if !is_ascending(&set) {
                        return Err("Values are not ascending.".to_owned());
                    }
                    if let Some(&set_max) = set.last() {
                        if set_max > max_value {
                            return Err(format!(
                                "Largest value ({}) is greater than maximum ({}).",
                                set_max, max_value
                            ));
                        }
                    }
                    Ok(())
                })()
                .map_err(|e| format!("Set #{}: {}", set_num + 1, e))?;
            }

            let mut intersection: Vec<T> =
                iter::repeat(T::default()).take(trial[0].len()).collect();
            let mut buffer = intersection.as_slice().to_vec();
            let sets: Vec<&[T]> = trial[0..trial.len() - 1]
                .iter()
                .map(|v| v.as_slice())
                .collect();
            let size = algo(&sets, &mut intersection, &mut buffer);

            let expected_intersection = trial.last().unwrap();

            if size != expected_intersection.len() {
                return Err(format!(
                    "Expected intersection size of {} but found {}.",
                    expected_intersection.len(),
                    size
                ));
            }

            intersection.truncate(size);
            let same = iter::zip(&intersection, expected_intersection).all(|(a, b)| *a == *b);
            if !same {
                return Err("Found and given intersection differ.".to_owned());
            }
            Ok(())
        })()
        .map_err(|e| format!("Trial #{}: {}", trial_num + 1, e))?;
    }

    Ok(())
}
