#![feature(step_trait)]
#![feature(trait_alias)]

use benchmark::{
    fmt_open_err, path_str,
    util::{random_subset, sample_distribution_unique, to_u64, to_usize, vec_to_bytes, Byteable},
    DataBinDescription, DataBinLengths, DataBinLengthsEnum, DataDistribution, Datatype,
};
use clap::Parser;
use colored::*;
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    seq::SliceRandom,
    Rng, SeedableRng,
};
use std::{
    cell::Cell,
    collections::HashMap,
    fs::{self, File},
    hash::Hash,
    io::Write,
    iter::{self, zip, Step},
    mem::swap,
    ops::Range,
    path::PathBuf,
};

type Set<T> = Vec<T>;
type Trial<T> = Vec<Set<T>>;
type DataBinPair<T> = Vec<Trial<T>>;
type Sample<T> = Vec<Trial<T>>;
type DataBinSample<T> = Vec<Sample<T>>;

trait Generatable = SampleUniform + TryFrom<u64> + From<u8> + Eq + Hash + Ord + Clone + Copy + Step;

trait Writeable<const N: usize> = Byteable<N>;

// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    description: PathBuf,
    #[arg(long, default_value = "datasets/")]
    outdir: PathBuf,
    #[arg(long)]
    single: Option<usize>,
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = main_inner(&cli) {
        println!("{}", err.red().bold());
    } else {
        println!("{}", "DONE".green().bold());
    }
}

fn main_inner(cli: &Cli) -> Result<(), String> {
    println!(
        "{}: dataset ({})",
        "GENERATING".green().bold(),
        path_str(&cli.description)
    );

    let dataset_description: Vec<DataBinDescription> = {
        let desc_string =
            fs::read_to_string(&cli.description).map_err(|e| fmt_open_err(e, &cli.description))?;
        serde_json::from_str(&desc_string)
            .map_err(|e| format!("Invalid JSON file {}: {}", path_str(&cli.description), e))?
    };

    let bin_out_path = cli.description.with_extension("data");
    let mut bin_out_file =
        File::create(&bin_out_path).map_err(|e| fmt_open_err(e, &bin_out_path))?;

    // Handle dispatch of databin generation and output to generic function handling the specified datatype
    if let Some(single) = cli.single {
        let length = dataset_description.len();
        if single > length || single == 0 {
            return Err(format!(
                "Single databin selection index ({}) outside valid range ([1, {}])", 
                single, length,
            ));
        }
        println!("{} / {}", single, length);
        datatype_dispatch(&dataset_description[single - 1], &mut bin_out_file)?;
    } else {
        let mut count = 1;
        for data_bin_description in &dataset_description {
            println!("{} / {}", count, dataset_description.len());
            datatype_dispatch(data_bin_description, &mut bin_out_file)?;
            count += 1;
        }
    }

    Ok(())
}

fn datatype_dispatch(
    data_bin_description: &DataBinDescription,
    bin_out_file: &mut File,
) -> Result<(), String> {
    match data_bin_description.datatype {
        Datatype::U32 => generate_and_write_ints::<u32, { std::mem::size_of::<u32>() }>(
            &data_bin_description,
            bin_out_file,
        )?,
        Datatype::I32 => generate_and_write_ints::<i32, { std::mem::size_of::<i32>() }>(
            &data_bin_description,
            bin_out_file,
        )?,
        Datatype::U64 => generate_and_write_ints::<u64, { std::mem::size_of::<u64>() }>(
            &data_bin_description,
            bin_out_file,
        )?,
        Datatype::I64 => generate_and_write_ints::<i64, { std::mem::size_of::<i64>() }>(
            &data_bin_description,
            bin_out_file,
        )?,
    };
    Ok(())
}

fn generate_and_write_ints<T, const N: usize>(
    data_bin_description: &DataBinDescription,
    out_file: &mut File,
) -> Result<(), String>
where
    T: Generatable + Writeable<N>,
{
    // Ensure that we can increment max_value by 1
    if data_bin_description.max_value == u64::MAX {
        return Err(format!(
            "max_value ({}) too large.",
            data_bin_description.max_value
        ));
    }

    let value_range = {
        let end: T = match (data_bin_description.max_value + 1).try_into() {
            Ok(val) => val,
            Err(_) => {
                return Err(format!(
                    "max_value ({}) too large for datatype ({:?}).",
                    data_bin_description.max_value, data_bin_description.datatype
                ))
            }
        };

        0.into()..end
    };

    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(data_bin_description.seed);
    let distribution = make_distribution(value_range.clone(), data_bin_description.distribution);

    let trials_usize: usize = data_bin_description.trials.try_into().or(Err(format!(
        "Could not convert trials ({}) to usize.",
        data_bin_description.trials,
    )))?;

    match &data_bin_description.lengths {
        DataBinLengthsEnum::Pair(lengths) => {
            let data_bin_pairs = gen_pair::<T>(
                data_bin_description,
                lengths,
                value_range,
                &mut rng,
                distribution,
                trials_usize,
            )?;
            write_pairs::<T, N>(&data_bin_pairs, out_file)?;
        }
        DataBinLengthsEnum::Sample(lengths_vec) => {
            let data_bin_samples = gen_samples::<T>(
                data_bin_description,
                lengths_vec,
                value_range,
                &mut rng,
                distribution,
                trials_usize,
            )?;
            write_samples::<T, N>(&data_bin_samples, out_file)?;
        }
    }

    Ok(())
}

fn make_distribution<T: Generatable>(
    value_range: Range<T>,
    distribution: DataDistribution,
) -> impl Distribution<T> {
    match distribution {
        DataDistribution::Uniform {} => Uniform::from(value_range),
    }
}

fn gen_pair<T: Generatable>(
    data_bin_description: &DataBinDescription,
    lengths: &DataBinLengths,
    value_range: Range<T>,
    rng: &mut impl Rng,
    distribution: impl Distribution<T>,
    trials_usize: usize,
) -> Result<DataBinPair<T>, String> {
    let long_length = to_usize(lengths.set_lengths[0], "long_length")?;
    let short_length = to_usize(lengths.set_lengths[1], "short_length")?;
    let intersection_length = to_usize(lengths.intersection_length, "intersection_length")?;

    // For each trial we will generate all of the values for both sets, unsorted, in a single array
    let total_length = long_length + short_length - intersection_length;

    let values_vec = generate_values_vec(
        total_length,
        data_bin_description.max_value,
        &value_range,
        rng,
        &distribution,
        trials_usize,
    )?;

    // For each trial we split the array into short and long with the given intersection size,
    // plus the intersection itself
    let mut data_bin = DataBinPair::<T>::with_capacity(trials_usize);
    for values in values_vec {
        let mut intersection = Vec::<T>::from(&values[0..intersection_length]);
        intersection.sort_unstable();

        let mut long = Vec::<T>::from(&values[0..long_length]);
        long.sort_unstable();

        let mut short = Vec::<T>::with_capacity(short_length);
        short.extend(&intersection);
        short.extend(&values[long_length..]);
        short.sort_unstable();

        data_bin.push(vec![long, short, intersection]);
    }

    Ok(data_bin)
}

fn gen_samples<T: Generatable>(
    data_bin_description: &DataBinDescription,
    lengths_vec: &Vec<DataBinLengths>,
    value_range: Range<T>,
    rng: &mut impl Rng,
    distribution: impl Distribution<T>,
    trials_usize: usize,
) -> Result<DataBinSample<T>, String> {
    let sample_count = lengths_vec.len();
    let mut databin: DataBinSample<T> = Vec::with_capacity(sample_count);

    for lengths in lengths_vec {
        let query_size = lengths.set_lengths.len();

        let set_lengths: Vec<usize> = lengths
            .set_lengths
            .iter()
            .map(|l| to_usize(*l, "set_length"))
            .collect::<Result<Vec<_>, String>>()?;

        let longest_length = set_lengths[0];
        let shortest_length = set_lengths[query_size - 1];
        let intersection_length = to_usize(lengths.intersection_length, "intersection_length")?;

        let (intersectable_lengths, non_intersectable_size) = {
            let mut intersectable_lengths = Vec::<usize>::with_capacity(query_size);
            let mut non_intersectable_size = 0usize;

            let proportion = {
                let final_proportion = intersection_length as f64 / shortest_length as f64;
                final_proportion.powf(1.0 / query_size as f64)
            };

            for set_length in &set_lengths {
                let intersectable_length = (*set_length as f64 * proportion).round() as usize;
                intersectable_lengths.push(intersectable_length);
                non_intersectable_size += set_length - intersectable_length;
            }

            (intersectable_lengths, non_intersectable_size)
        };

        let total_length = non_intersectable_size + longest_length;

        let max_value = to_usize(data_bin_description.max_value, "max_value")?;
        if total_length > max_value {
            return Err(format!(
                "max_value ({}) lower than value required for generation ({}). Please decrease density.", 
                max_value, total_length
            ));
        }

        let values_vec = generate_values_vec(
            total_length,
            data_bin_description.max_value,
            &value_range,
            rng,
            &distribution,
            trials_usize,
        )?;

        let mut sample: Sample<T> = Vec::with_capacity(trials_usize);
        for values in values_vec {
            let mut trial = Trial::<T>::with_capacity(query_size + 1);

            let intersection_base = &values[..longest_length];

            // Hashmap to hold the set count for values that could be intersected but shouldn't be in the final intersection
            let counts: HashMap<T, Cell<usize>> = intersection_base[intersection_length..]
                .iter()
                .map(|n| (*n, Cell::from(0usize)))
                .collect();

            // Create the initial states of the sets from the intersectable subset of values
            for (length, intersectable_length) in zip(&set_lengths, &intersectable_lengths) {
                let mut set = Vec::<T>::from(&intersection_base[..*length]);
                // We shuffle only the intersectable section outside of the final intersection
                (&mut set[intersection_length..]).shuffle(rng);

                // Update frequency counts of values in the intersectable portion that are not in the final intersection
                for value in &set[intersection_length..*intersectable_length] {
                    let count = counts.get(value).unwrap();
                    count.set(count.get() + 1);
                }

                trial.push(set);
            }

            // Count the number of intersections that we have to fix up
            let mut excess_intersections =
                counts.iter().fold(
                    0usize,
                    |acc, (_, c)| if c.get() == query_size { acc + 1 } else { acc },
                );

            // Perform swaps to remove intersectable values that should not be in the final intersection
            'outer: for (set, intersectable_length) in zip(trial.iter_mut(), &intersectable_lengths)
            {
                let (inter, outer) =
                    (&mut set[intersection_length..]).split_at_mut(*intersectable_length - intersection_length);

                let mut ii = 0usize;
                let mut oi = 0usize;

                'inner: loop {
                    // Early exit if all excess intersections have been dealt with
                    if excess_intersections == 0 {
                        break 'outer;
                    }

                    // Mutable references to counts
                    let mut ic;
                    let mut oc;

                    // Find an intersection that shouldn't be
                    loop {
                        if ii == inter.len() {
                            break 'inner;
                        }
                        ic = counts.get(&inter[ii]).unwrap();
                        if ic.get() != query_size {
                            ii += 1;
                        } else {
                            break;
                        }
                    }

                    // Find a non-intersection that it can be swapped for
                    loop {
                        if oi == outer.len() {
                            break 'inner;
                        }
                        oc = counts.get(&outer[oi]).unwrap();
                        if oc.get() == query_size - 1 {
                            oi += 1;
                        } else {
                            break;
                        }
                    }

                    // Swap and update counts
                    swap(inter.get_mut(ii).unwrap(), outer.get_mut(oi).unwrap());
                    ic.set(ic.get() - 1);
                    oc.set(oc.get() + 1);
                    excess_intersections -= 1;
                }
            }

            // Double check that we haven't violated selectivity
            if excess_intersections != 0 {
                return Err(
                    "Could not generate k-set of the given selectivity given its other parameters."
                        .to_owned(),
                );
            }

            {
                // Insert non-intersecting portion of the sets
                let mut remainder = &values[longest_length..];
                for (set, (length, intersectable_length)) in
                    zip(trial.iter_mut(), zip(&set_lengths, &intersectable_lengths))
                {
                    set.truncate(*intersectable_length);
                    let remainder_length = length - intersectable_length;
                    set.extend(&remainder[..remainder_length]);
                    remainder = &remainder[remainder_length..];
                }
            }

            {
                // Insert the intersection set into trial
                let intersection = Vec::<T>::from(&intersection_base[0..intersection_length]);
                trial.push(intersection);
            }

            // Sort all of the sets
            for set in trial.iter_mut() {
                set.sort_unstable();
            }

            sample.push(trial);
        }
        databin.push(sample);
    }

    Ok(databin)
}

fn generate_values_vec<T>(
    total_length: usize,
    max_value: u64,
    value_range: &Range<T>,
    rng: &mut impl Rng,
    distribution: &impl Distribution<T>,
    trials_usize: usize,
) -> Result<Vec<Vec<T>>, String>
where
    T: Step + Eq + Hash + Copy,
{
    let total_length_u64 = to_u64(total_length, "total_length")?;

    Ok(if is_dense(total_length_u64, max_value) {
        iter::repeat_with(|| random_subset(value_range.clone(), total_length, rng))
            .take(trials_usize)
            .collect()
    } else {
        iter::repeat_with(|| sample_distribution_unique(total_length, &distribution, rng))
            .take(trials_usize)
            .collect()
    })
}

fn is_dense(total_length: u64, max_value: u64) -> bool {
    // values between 2 and 10 seem to have about the same performance (on the data I was testing at least)
    // keeping it lower to minimise the potential for very large arrays
    const DENSE_RATIO: u64 = 2;
    total_length > (max_value / DENSE_RATIO)
}

fn write_samples<T, const N: usize>(
    samples: &DataBinSample<T>,
    out_file: &mut File,
) -> Result<(), String>
where
    T: Writeable<N>,
{
    for sample in samples {
        write_pairs(sample, out_file)?;
    }

    Ok(())
}

fn write_pairs<T, const N: usize>(
    trials: &DataBinPair<T>,
    out_file: &mut File,
) -> Result<(), String>
where
    T: Writeable<N>,
{
    for trial in trials {
        for set in trial {
            match out_file.write_all(&vec_to_bytes(&set)) {
                Ok(()) => (),
                Err(e) => return Err(format!("Failed writing databin: {}", e.to_string())),
            };
        }
    }

    Ok(())
}
