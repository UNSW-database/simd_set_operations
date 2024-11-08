#![feature(trait_alias)]

use benchmark::{
    Datatype, DataDistribution,
    schemas::dataset::*,
};
use clap::Parser;
use colored::*;
use indicatif::{ProgressIterator, ParallelProgressIterator};
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    seq::SliceRandom,
    Rng, SeedableRng,
};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator, IndexedParallelIterator};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    hash::Hash,
    io::{Seek, SeekFrom, Write},
    path::PathBuf,
    sync::{Arc, Mutex},
};

trait Generatable = SampleUniform + TryFrom<u64> + From<u8> + Eq + Hash + Ord + Clone + Copy + Sized;

// CLI arguments
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

    if let Err(err) = main_inner(&cli) {
        println!("{}: {}", "ERROR".red().bold(), err);
    } else {
        println!("{}", "DONE".green().bold());
    }
}

fn main_inner(cli: &Cli) -> Result<(), String> {
    println!("{}: dataset ({})", "GENERATING".green().bold(), cli.dataset_description.display());

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

    let output_path = if let Some(index) = cli.single {
        cli.dataset_description.with_extension(format!("{index}.data"))
    } else {
        cli.dataset_description.with_extension("data")
    };

    let mut output_file = match File::create(&output_path) {
        Ok(v)  => v,
        Err(e) => return Err(format!("Failed to open data file {} for writing: {e}", output_path.display())),
    };
    let parallel_output_file = Arc::new(Mutex::new(&mut output_file));

    let databin_count = dataset_description.databins.len();
    let kset = dataset_description.kset;

    // Handle dispatch of databin generation and output to generic function handling the specified datatype
    if let Some(index) = cli.single {
        if index >= databin_count {
            return Err(format!("Single databin selection index ({index}) outside valid range [0, {databin_count})."));
        }
        println!("{} / {}", index + 1, databin_count);
        generate_databin(&dataset_description.databins[index], parallel_output_file, kset)?;
    } else {
        let closure = |(index, r_databin_description)| {
            let num = index + 1;
            match generate_databin(r_databin_description, parallel_output_file.clone(), kset) {
                Ok(_)  => Ok(()),
                Err(e) => return Err(format!("Failed to generate databin {num}: {e}")),
            }
        };
        if cli.single_threaded {
            dataset_description.databins
                .iter()
                .enumerate()
                .progress_count(databin_count as u64)
                .try_for_each(closure)?;
        } else {
            dataset_description.databins
                .par_iter()
                .enumerate()
                .progress_count(databin_count as u64)
                .try_for_each(closure)?;
        }
    }

    Ok(())
}

fn generate_databin(
    r_databin_description : & DatabinDescription,
    parallel_output_file  :   Arc<Mutex<&mut File>>,
    kset                  :   bool,
) -> Result<(), String> {
    macro_rules! generate_typed {
        ($type:ident) => {
            generate_int_databins::<$type>(r_databin_description, parallel_output_file, kset)
        };
    }

    return match r_databin_description.datatype {
        Datatype::U32 => generate_typed!(u32),
        Datatype::I32 => generate_typed!(i32),
        Datatype::U64 => generate_typed!(u64),
        Datatype::I64 => generate_typed!(i64),
    };
}

fn generate_int_databins<T: Generatable>(
    r_databin_description : & DatabinDescription,
    parallel_output_file  :   Arc<Mutex<&mut File>>,
    kset                  :   bool,
) -> Result<(), String> {
    let datatype = r_databin_description.datatype;
    let distribution_type = r_databin_description.distribution;
    let byte_offset = r_databin_description.byte_offset;
    let max_value = r_databin_description.max_value;

    //
    // === Generation setup
    //
    let start_value: T = 0.into();
    let end_value: T = match max_value.try_into() {
        Ok(v)  => v,
        Err(_) => return Err(format!("max_value ({max_value}) too large for datatype ({:?}).", datatype)),
    };

    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(r_databin_description.seed);
    let distribution = match distribution_type {
        DataDistribution::Uniform {} => Uniform::<T>::new_inclusive(start_value, end_value),
    };

    //
    // === Generate
    //
    let databin = if kset {
        gen_kset::<T>(r_databin_description, &mut rng, distribution)
    } else {
        gen_2set::<T>(r_databin_description, &mut rng, distribution)
    }?;

    //
    // === Write out
    //
    let mut locked_output_file = parallel_output_file.lock().unwrap();
    if let Err(_) = locked_output_file.seek(SeekFrom::Start(byte_offset)) {
        return Err(format!("Failed to seek to {byte_offset} in output file."));
    }
    if let Err(e) = locked_output_file.write_all(&databin) {
        return Err(format!("Failed writing: {}", e.to_string()));
    }

    Ok(())
}

fn gen_2set<T: Generatable>(
    r_databin_description : &    DatabinDescription,
    mr_rng                : &mut impl Rng,
    distribution          :      impl Distribution<T>,
) -> Result<Vec<u8>, String> {
    let bytes = r_databin_description.datatype.bytes() as usize;

    let mut databin = vec![0u8; r_databin_description.byte_length as usize];
    let mut value_set = HashSet::<T>::new();
    let mut byte_offset = 0usize;

    for r_trial in &r_databin_description.trials {
        let long_length         = r_trial.set_lengths[0] as usize;
        let short_length        = r_trial.set_lengths[1] as usize;
        let intersection_length = r_trial.intersection_length as usize;
        let value_length        = long_length + short_length - intersection_length;
        let total_length        = long_length + short_length + intersection_length;
        let byte_length         = total_length * bytes;

        if byte_length != r_trial.byte_length as usize {
            return Err(format!("Trial byte length ({byte_length}) differs from expected ({}).", r_trial.byte_length));
        }
        if byte_offset != r_trial.byte_offset as usize {
            return Err(format!("Trial byte offset ({byte_offset}) differs from expected ({}).", r_trial.byte_offset));
        }

        let slice = {
            let byte_offset_end = byte_offset + byte_length;
            let byte_slice = &mut databin[byte_offset..byte_offset_end];
            let ptr = &mut byte_slice[0] as *mut u8 as *mut T;
            unsafe { std::slice::from_raw_parts_mut(ptr, total_length) }
        };
        byte_offset += byte_length;

        // For each trial we will generate all of the values for both sets, 
        // unsorted, in a single array. 
        // NOTE: this will get very slow as the density approaches 1, but 
        // I have not been able to think up or find a better approcah that 
        // doesn't interfere with the ability to generate to a distribution.
        value_set.clear();
        while value_set.len() != total_length {
            let value = distribution.sample(mr_rng);
            if value_set.insert(value) {
                slice[value_set.len() - 1] = value;
            }
        }

        // We currently have {long}{short \ intersection} in slice, so we append
        // intersection twice to get {long}{short}{intersection}
        {
            let (s0, s1) = slice.split_at_mut(intersection_length);
            let (_s10, s11) = s1.split_at_mut(value_length - intersection_length);
            s11[0..intersection_length].copy_from_slice(s0);
            s11[intersection_length..].copy_from_slice(s0);
        }

        // Finally we sort the values in {long}, {short}, and {intersection} in
        // ascending order
        {
            let (long_slice, rs) = slice.split_at_mut(long_length);
            let (short_slice, intersection_slice) = rs.split_at_mut(short_length);
            long_slice.sort_unstable();
            short_slice.sort_unstable();
            intersection_slice.sort_unstable();
        }
    }

    Ok(databin)
}

fn gen_kset<T: Generatable>(
    r_databin_description : &    DatabinDescription,
    mr_rng                : &mut impl Rng,
    distribution          :      impl Distribution<T>,
) -> Result<Vec<u8>, String> {
    let bytes = r_databin_description.datatype.bytes() as usize;
    let max_value = r_databin_description.max_value as usize;

    let mut databin = vec![0u8; r_databin_description.byte_length as usize];
    let mut value_vec = Vec::<T>::new();
    let mut value_set = HashSet::<T>::new();
    let mut count_map = HashMap::<T, usize>::new();
    let mut byte_offset = 0usize;

    for (trial_index, r_trial) in r_databin_description.trials.iter().enumerate() {
    (||{
        let set_count = r_trial.set_lengths.len();

        let longest_length      = r_trial.set_lengths[0] as usize;
        let shortest_length     = *r_trial.set_lengths.last().unwrap() as usize;
        let intersection_length = r_trial.intersection_length as usize;

        let final_proportion = intersection_length as f64 / shortest_length as f64;
        let pair_proportion = final_proportion.powf(1.0 / set_count as f64);

        // Calculate intersectable length (NB: not _intersection_ length)
        let ilen = |set_length: u64| (set_length as f64 * pair_proportion).round() as usize;

        let mut non_intersectable_size = 0usize;
        let mut total_length = intersection_length;
        for &set_length in &r_trial.set_lengths {
            non_intersectable_size += set_length as usize - ilen(set_length);
            total_length += set_length as usize;
        }
        let value_length = non_intersectable_size + longest_length;
        let byte_length = total_length * bytes;

        if value_length > max_value {
            return Err(format!("max_value ({max_value}) lower than value required for generation ({value_length}). Please decrease density."));
        }
        if byte_length != r_trial.byte_length  as usize {
            return Err(format!("Byte length ({byte_length}) differs from expected ({}).", r_trial.byte_length));
        }
        if byte_offset != r_trial.byte_offset as usize {
            return Err(format!("Byte offset ({byte_offset}) differs from expected ({}).", r_trial.byte_offset));
        }

        // Get the T slice where we will be writing all of the sets
        let slice = {
            let byte_offset_end = byte_offset + byte_length;
            let byte_slice = &mut databin[byte_offset..byte_offset_end];
            let ptr = &mut byte_slice[0] as *mut u8 as *mut T;
            unsafe { std::slice::from_raw_parts_mut(ptr, total_length) }
        };
        byte_offset += byte_length;

        // Generate values
        value_set.clear();
        value_vec.clear();
        while value_set.len() != value_length {
            let value = distribution.sample(mr_rng);
            if value_set.insert(value) {
                value_vec.push(value);
            }
        }

        let intersection_base = &value_vec[..longest_length];

        // Hashmap to hold the set count for values that could be intersected
        // but shouldn't be in the final intersection
        count_map.clear();
        for r_value in intersection_base {
            count_map.insert(*r_value, 0);
        }

        // Copy from the base intersection to fill each set and then shuffle all
        // of the values in each set that aren't in the final intersection
        let mut set_offset = 0usize;
        for &set_length in &r_trial.set_lengths {
            let set = &mut slice[set_offset..set_offset + set_length as usize];
            set.copy_from_slice(&intersection_base[0..set_length as usize]);
            (&mut set[intersection_length..]).shuffle(mr_rng);

            // Update frequency counts of values not in the final intersection
            for r_value in &set[intersection_length..ilen(set_length)] {
                *count_map.get_mut(r_value).unwrap() += 1;
            }

            set_offset += set_length as usize;
        }

        // Count the number of intersections that we have to fix up
        let mut excess_intersections = 0usize;
        for &value in count_map.values() {
            if value == set_count {
                excess_intersections += 1;
            }
        }

        // Perform swaps to remove intersectable values that should not be in the final intersection
        let mut set_offset = 0usize;
        for &set_length in &r_trial.set_lengths {
            let set = &mut slice[set_offset..set_offset + set_length as usize];
            set_offset += set_length as usize;

            // Split the set into the bit that isn't in the final intersection 
            // but can be intersected, and the bit that is thrown away
            let (lower, outer) = set.split_at_mut(ilen(set_length));
            let inter = &mut lower[intersection_length..];

            let mut ii = 0usize;
            let mut oi = 0usize;
            while excess_intersections != 0 {
                let mut iv: T = 0.into();
                let mut ov: T = 0.into();
                // Find an intersection that shouldn't be
                while ii != inter.len() {
                    iv = inter[ii];
                    if count_map[&iv] == set_count {
                        break
                    }
                    ii += 1;
                }
                // Find a non-intersection that it can be swapped for
                while oi != outer.len() {
                    ov = outer[oi];
                    if count_map[&ov] != set_count - 1 {
                        break;
                    } 
                    oi += 1;
                }

                if ii == inter.len() || oi == outer.len() {
                    break;
                }

                // Swap and update counts
                inter[ii] = ov;
                outer[oi] = iv;
                *count_map.get_mut(&iv).unwrap() -= 1;
                *count_map.get_mut(&ov).unwrap() += 1;
                excess_intersections -= 1;
            }
        }

        // Double check that we haven't violated selectivity
        if excess_intersections != 0 {
            return Err("Could not generate k-set of the given selectivity given its other parameters.".to_owned());
        }

        { // Insert non-intersecting portions of the sets and sort them
            let mut remainder = &value_vec[longest_length..];
            let mut set_offset = 0usize;
            for &set_length in &r_trial.set_lengths {
                let set = &mut slice[set_offset..set_offset + set_length as usize];
                set_offset += set_length as usize;

                let intersectable_length = ilen(set_length);
                let remainder_length = set_length as usize - intersectable_length;
                let remainder_slice = &remainder[0..remainder_length];
                remainder = &remainder[remainder_length..];
                (&mut set[intersectable_length..]).copy_from_slice(remainder_slice);

                set.sort_unstable();
            }

            // Copy and sort intersection
            let intersection = &mut slice[set_offset..set_offset + intersection_length];
            intersection.copy_from_slice(&value_vec[..intersection_length]);
            intersection.sort_unstable();
        }

        return Ok(());
    })().map_err(|e| format!("Trial #{}: {e}", trial_index + 1))?;
    }

    Ok(databin)
}
