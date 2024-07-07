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
    util::{to_u64, to_usize, Byteable},
    DataBinDescription, DataBinLengthsEnum, DataBinPair, Datatype,
};
use clap::Parser;
use colored::Colorize;
use indicatif::ParallelProgressIterator;
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

    let databin_count = to_u64(dataset_description.len(), "databin_count")?;
    dataset_description
        .par_iter()
        .enumerate()
        .progress_count(databin_count)
        .try_for_each(|(db_num, data_bin_description)| {
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
            .map_err(|e| format!("Data bin #{}: {}", db_num, e))
        })?;

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
                read_databin_pair::<N, T>(
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
                read_databin_sample::<N, T>(
                    lengths,
                    byte_offset,
                    byte_count,
                    trial_count,
                    bin_file.borrow_mut(),
                )
            }?;
            for (sample_num, sample) in data_bin.iter().enumerate() {
                verify(sample, max_value, algo)
                    .map_err(|e| format!("Sample #{}: {}", sample_num, e))?;
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
        // Between sets:
        // - Verify intersection size
        // - Verify intersection correctness
        if trial.len() < 3 {
            return Err(format!(
                "Trial #{} had set count of {}, but it must be at least 3.",
                trial_num,
                trial.len()
            ));
        }

        for (set_num, set) in trial.iter().enumerate() {
            // Within sets:
            // - Verify ascending valuse
            // - Verify max is not too large
            let is_ascending = set.windows(2).all(|w| {
                let (lhs, rhs) = unsafe { (w.get_unchecked(0), w.get_unchecked(1)) };
                lhs < rhs
            });
            if !is_ascending {
                return Err(format!(
                    "Set #{} in trial #{} is not ascending.",
                    set_num, trial_num
                ));
            }
            if let Some(&set_max) = set.last() {
                if set_max > max_value {
                    return Err(format!(
                        "Largest value ({}) in set #{} in trial #{} is greater than maximum ({}).",
                        set_max, set_num, trial_num, max_value
                    ));
                }
            }
        }

        let mut intersection: Vec<T> = iter::repeat(T::default()).take(trial[0].len()).collect();
        let mut buffer = intersection.as_slice().to_vec();
        let sets: Vec<&[T]> = trial[0..trial.len() - 1]
            .iter()
            .map(|v| v.as_slice())
            .collect();
        let size = algo(&sets, &mut intersection, &mut buffer);

        let expected_intersection = trial.last().unwrap();

        if size != expected_intersection.len() {
            return Err(format!(
                "Expected intersection size of {} in trial #{}, found {}.",
                trial[2].len(),
                trial_num,
                size
            ));
        }

        intersection.truncate(size);
        let same = iter::zip(&intersection, expected_intersection).all(|(a, b)| *a == *b);
        if !same {
            return Err(format!(
                "Found and given intersection differ for pair trial #{}.",
                trial_num
            ));
        }
    }

    Ok(())
}
