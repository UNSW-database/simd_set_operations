#![feature(step_trait)]

use benchmark::{
    fmt_open_err, path_str,
    util::{vec_to_bytes, Byteable},
    DataBinConfig, Datatype,
};
use clap::Parser;
use colored::*;
use rand::{
    distributions::{uniform::SampleUniform, Distribution, Uniform},
    SeedableRng,
};
use std::{
    collections::HashSet, fs::{self, File}, hash::Hash, io::Write, iter::Step, path::PathBuf
};

// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long)]
    description: PathBuf,
    #[arg(long, default_value = "datasets/")]
    outdir: PathBuf,
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = generate(&cli) {
        println!("{}", err.red().bold());
    } else {
        println!("{}", "DONE".green().bold());
    }
}

fn generate(cli: &Cli) -> Result<(), String> {
    println!(
        "{}: dataset ({})",
        "GENERATING".green().bold(),
        path_str(&cli.description)
    );

    // Read dataset description
    let dataset: Vec<DataBinConfig> = {
        let desc_string =
            fs::read_to_string(&cli.description).map_err(|e| fmt_open_err(e, &cli.description))?;
        serde_json::from_str(&desc_string)
            .map_err(|e| format!("Invalid JSON file {}: {}", path_str(&cli.description), e))?
    };

    // Create dataset binary output file
    let out_path = cli.description.with_extension("data");
    let mut out_file = File::create(&out_path).map_err(|e| fmt_open_err(e, &out_path))?;

    let mut count = 1;
    for data_bin in &dataset {
        println!("{} / {}", count, dataset.len());
        match data_bin.datatype {
            Datatype::U32 => gen_bin::<u32, {std::mem::size_of::<u32>()}>(data_bin, &mut out_file)?,
            Datatype::U64 => gen_bin::<u64, {std::mem::size_of::<u64>()}>(data_bin, &mut out_file)?,
            Datatype::I32 => gen_bin::<i32, {std::mem::size_of::<i32>()}>(data_bin, &mut out_file)?,
            Datatype::I64 => gen_bin::<i64, {std::mem::size_of::<i64>()}>(data_bin, &mut out_file)?,
        }
        count += 1;
    }

    Ok(())
}

fn gen_bin<T, const N: usize>(data_bin: &DataBinConfig, out_file: &mut File) -> Result<(), String>
where
    T: SampleUniform + TryFrom<u64> + From<u8> + Eq + Hash + Ord + Clone + Copy + Byteable<N> + Step,
{
    let max_value_t: T = match data_bin.max_value.try_into() {
        Ok(val) => val,
        Err(_) => {
            return Err(format!(
                "max_value ({}) too large for datatype ({:?}).",
                data_bin.max_value, data_bin.datatype
            ))
        }
    };
    let range_t = 0.into() ..= max_value_t;

    let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(data_bin.seed);
    let distribution: Uniform<T> = Uniform::from(range_t.clone());

    for _trial in 0..data_bin.trials {
        // Generate all of the values for both sets, unsorted, in a single array
        let total_length: usize =
            data_bin.long_length + data_bin.short_length - data_bin.intersection_length;

        let values: Vec<T> = {
            let dense = total_length > (data_bin.max_value as usize / 2);

            let mut set = HashSet::<T>::with_capacity(if dense {
                data_bin.max_value as usize + 1
            } else {
                total_length
            });

            if dense {
                set.extend(range_t.clone());
                while set.len() != total_length {
                    set.remove(&distribution.sample(&mut rng));
                }
            } else {
                while set.len() != total_length {
                    set.insert(distribution.sample(&mut rng));
                }
            }

            set.into_iter().collect()
        };

        // Split the array into short and long with the given intersection size, plus the intersection itself
        let mut intersection = Vec::<T>::from(&values[0..data_bin.intersection_length]);
        intersection.sort_unstable();

        let mut long = Vec::<T>::from(&values[0..data_bin.long_length]);
        long.sort_unstable();

        let mut short = Vec::<T>::from(intersection);
        short.extend(&values[data_bin.long_length..total_length]);
        short.sort_unstable();

        // Write data out to file
        match out_file.write_all(&vec_to_bytes(&long)) {
            Ok(()) => (),
            Err(e) => return Err(format!("Failed writing databin: {}", e.to_string())),
        };
    }

    Ok(())
}
