use benchmark::{
    util::sample_distribution_unique, 
    schemas::{generator::*, dataset::*},
    Datatype, DataDistribution, CorpusDistribution, QueryDistribution,
};

use std::{fs::{self, File}, path::PathBuf};
use clap::Parser;
use colored::*;
use rand::{Rng, SeedableRng};
use zipf::ZipfDistribution;

// CLI arguments
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(long, short)]
    config: PathBuf,
    #[arg(long, short, default_value = "datasets/")]
    outdir: PathBuf,
    #[arg(long, short)]
    seed: Option<u64>,
}

fn main() {
    let cli = Cli::parse();

    if let Err(err) = describe(&cli) {
        println!("{}", err.red().bold());
    } else {
        println!("{}", "DONE".green().bold());
    }
}

fn describe(cli: &Cli) -> Result<(), String> {
    // Set up seed generation
    let seed = match cli.seed {
        Some(seed) => seed,
        None       => rand::random(),
    };
    let mut rng = rand_chacha::ChaChaRng::seed_from_u64(seed);

    // Read dataset configuration
    let config: Config = {
        let config_string = match fs::read_to_string(&cli.config) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Failed to read config file {}: {}", cli.config.display(), e)),
        };
        let config = match toml::from_str(&config_string) {
            Ok(v)  => v,
            Err(e) => return Err(format!("Invalid config file {}: {}", cli.config.display(), e)),
        };
        config
    };

    let mut kset = false;
    let mut offset = 0u64;
    let mut databins = Vec::<DatabinDescription>::new();

    let parameters = match &config {
        Config::Pair(pair) => {
            let mut parameters = PairParams::default();

            validate_param_u64(&pair.max_set_size, "max_set_size", 1)?;
            validate_param_normalized(&pair.selectivity, "selectivity")?;
            validate_param_normalized(&pair.skew, "skew")?;
            validate_param_normalized(&pair.density, "density")?;

            if pair.trials == 0 {
                return Err(format!("Trial count must be greater than 0."));
            }

            let mut index = 0u64;
            for datatype     in pair.datatype.param_range()        {
            for max_set_size in pair.max_set_size.param_range()    {
            for skew         in pair.skew.param_range()            {
            for selectivity  in pair.selectivity.param_range()     {
            for density      in pair.density.param_range()         {
            for distribution in pair.distribution.param_range()    {
                parameters.skew         .entry(skew.to_string())         .or_insert_with(|| vec![]).push(index);
                parameters.density      .entry(density.to_string())      .or_insert_with(|| vec![]).push(index);
                parameters.datatype     .entry(datatype.to_string())     .or_insert_with(|| vec![]).push(index);
                parameters.selectivity  .entry(selectivity.to_string())  .or_insert_with(|| vec![]).push(index);
                parameters.max_set_size .entry(max_set_size.to_string()) .or_insert_with(|| vec![]).push(index);
                parameters.distribution .entry(distribution.to_string()) .or_insert_with(|| vec![]).push(index);
                index += 1;
                let seed: u64 = rng.gen();
                let databin_description = parameters_to_description_2set(
                    datatype,
                    max_set_size,
                    pair.trials,
                    selectivity,
                    skew,
                    density,
                    distribution,
                    seed,
                    &mut offset,
                )?;
                databins.push(databin_description)
            }}}}}}

            DatabinParameters::Pair(parameters)
        },
        Config::Sample(sample) => {
            let mut parameters = SampleParams::default();

            validate_param_u64(&sample.query.size, "query.size", 2)?;
            validate_param_u64(&sample.corpus.size, "corpus.size", 2)?;
            validate_param_u64(&sample.corpus.max_set_size, "corpus.max_set_size", 1)?;
            validate_param_normalized(&sample.query.selectivity, "query.selectivity")?;
            validate_param_normalized(&sample.corpus.skew, "corpus.skew")?;
            validate_param_normalized(&sample.corpus.density, "corpus.density")?;

            if sample.trials == 0 || sample.query.samples == 0 {
                return Err(format!("Trial and sample count must be greater than 0."));
            }

            let mut index = 0u64;
            for datatype            in sample.datatype.param_range()               {
            for data_distribution   in sample.distribution.param_range()           {
            for query_size          in sample.query.size.param_range()             {
            for query_distribution  in sample.query.distribution.param_range()     {
            for query_selectivity   in sample.query.selectivity.param_range()      {
            for corpus_size         in sample.corpus.size.param_range()            {
            for corpus_distribution in sample.corpus.distribution.param_range()    {
            for max_set_size        in sample.corpus.max_set_size.param_range()    {
            for corpus_skew         in sample.corpus.skew.param_range()            {
            for corpus_density      in sample.corpus.density.param_range()         {
                parameters.skew                .entry(corpus_skew.to_string())         .or_insert_with(|| vec![]).push(index);
                parameters.density             .entry(corpus_density.to_string())      .or_insert_with(|| vec![]).push(index);
                parameters.datatype            .entry(datatype.to_string())            .or_insert_with(|| vec![]).push(index);
                parameters.selectivity         .entry(query_selectivity.to_string())   .or_insert_with(|| vec![]).push(index);
                parameters.max_set_size        .entry(max_set_size.to_string())        .or_insert_with(|| vec![]).push(index);
                parameters.data_distribution   .entry(data_distribution.to_string())   .or_insert_with(|| vec![]).push(index);
                parameters.query_size          .entry(query_size.to_string())          .or_insert_with(|| vec![]).push(index);
                parameters.query_distribution  .entry(query_distribution.to_string())  .or_insert_with(|| vec![]).push(index);
                parameters.corpus_size         .entry(corpus_size.to_string())         .or_insert_with(|| vec![]).push(index);
                parameters.corpus_distribution .entry(corpus_distribution.to_string()) .or_insert_with(|| vec![]).push(index);
                index += 1;
                if query_size > 2 {
                    kset = true;
                }
                let seed: u64 = rng.gen();
                let databin_description = parameters_to_description_kset(
                    datatype,
                    max_set_size,
                    sample.trials,
                    query_selectivity,
                    corpus_skew,
                    corpus_density,
                    data_distribution,
                    seed,
                    &mut offset,
                    query_size,
                    corpus_size,
                    corpus_distribution,
                    query_distribution,
                    sample.query.samples,
                    &mut rng,
                )?;
                databins.push(databin_description);
            }}}}}}}}}}

            DatabinParameters::Sample(parameters)
        },
    };

    let dataset_description = DatasetDescription {
        seed,
        kset,
        byte_length: offset,
        databins,
        parameters,
    };

    // Write dataset description
    let filename = PathBuf::from(&cli.config.file_name().unwrap()).with_extension("json");
    let desc_path = (&cli.outdir).join(filename);
    let desc_file = match File::create(&desc_path) {
        Ok(v)  => v,
        Err(e) => return Err(format!("Failed to open output file {}: {}", desc_path.display(), e)),
    };
    if let Err(e) = serde_json::to_writer(desc_file, &dataset_description) {
        return Err(format!("Failed to write to output file {}: {}", desc_path.display(), e));
    }

    Ok(())
}

fn validate_param_u64(param: &NumParamOpt<u64>, name: &str, min: u64) -> Result<(), String> {
    match param {
        OptParameter::Fixed(v) => if *v < min {
            return Err(format!("Invalid paramater ({name}): value must be greater than {min}."));
        },
        OptParameter::Varying(p) => {
            if p.from < min as f64 {
                return Err(format!("Invalid paramater ({name}): 'from' must be greater than {min}."));
            }
            if p.from >= p.to {
                return Err(format!("Invalid paramater ({name}): 'from' must be less than 'to'."));
            }
            match p.step {
                StepType::Steps(steps) => if steps <= 0 {
                    return Err(format!("Invalid parameter ({name}): 'steps' must be greater than 0."));
                },
                StepType::Step(step) => if step <= 0.0 {
                    return Err(format!("Invalid paramater ({name}): 'step' must be greater than zero."));
                },
            };
        },
    };

    return Ok(());
}

fn validate_param_normalized(param: &NumParamOpt<f64>, name: &str) -> Result<(), String> {
    match param {
        OptParameter::Fixed(v) => if *v < 0.0 || *v > 1.0 {
            return Err(format!("Invalid paramater ({name}): value must be in the range [0, 1]."));
        },
        OptParameter::Varying(p) => {
            if p.from < 0.0 {
                return Err(format!("Invalid paramater ({name}): 'from' must be greater than or equal to 0."));
            }
            if p.to > 1.0 {
                return Err(format!("Invalid paramater ({name}): 'to' must be less than or equal to 1."));
            }
            if p.from >= p.to {
                return Err(format!("Invalid paramater ({name}): 'from' must be less than 'to'."));
            }
            match p.step {
                StepType::Steps(steps) => if steps == 0 {
                    return Err(format!("Invalid parameter ({name}): 'steps' must be greater than 0."));
                },
                StepType::Step(step) => if step <= 0.0 {
                    return Err(format!("Invalid paramater ({name}): 'step' must be greater than zero."));
                },
            };
        },
    };

    Ok(())
}

// Param range trait is used to glue paramater ranging together over several types
// Iterators might be better but returning impl Iterator seems to do some type erasure
trait ParamRange<T> {
    fn param_range(&self) -> Vec<T>;
}

impl<T: Clone + Copy, U: ParamRange<T>> ParamRange<T> for OptParameter<T, U> {
    fn param_range(&self) -> Vec<T> {
        return match self {
            OptParameter::Fixed(t)   => vec![*t],
            OptParameter::Varying(u) => u.param_range(),
        };
    }
}

impl<T: Clone + Copy> ParamRange<T> for Vec<T> {
    fn param_range(&self) -> Vec<T> {
        // I don't like that we have to copy the whole vector here
        return self.to_vec();
    }
}

impl ParamRange<f64> for NumericalParameter {
    fn param_range(&self) -> Vec<f64> {
        let p = self;
        let diff = p.to - p.from;
        let ratio = p.to / p.from;

        let (steps, step) = match p.step {
            StepType::Steps(steps) => {
                let step = match p.mode {
                    StepMode::Linear => diff / (steps - 1) as f64,
                    StepMode::Log => ratio.powf(1.0 / (steps - 1) as f64),
                };
                (steps, step)
            },
            StepType::Step(step) => {
                let steps = match p.mode {
                    StepMode::Linear => (diff / step).round() as u64 + 1,
                    StepMode::Log => ratio.log(step).round() as u64 + 1,
                };
                (steps, step)
            },
        };

        let mut values = Vec::<f64>::with_capacity(steps as usize);
        for i in 0..steps {
            let value = match p.mode {
                StepMode::Linear => p.from + (i as f64 / (steps - 1) as f64) * diff,
                StepMode::Log => p.from * step.powf(i as f64),
            };
            values.push(value);
        }
        return values;
    }
}

impl ParamRange<u64> for NumericalParameter {
    fn param_range(&self) -> Vec<u64> {
        // I don't like the uneeded allocation for f64_values
        let f64_values: Vec<f64> = self.param_range();
        let mut u64_values = Vec::<u64>::with_capacity(f64_values.len());
        for f64_value in f64_values {
            let u64_value = unsafe { f64_value.round().to_int_unchecked() };
            u64_values.push(u64_value);
        }
        return u64_values;
    }
}

fn parameters_to_description_2set(
    datatype     : Datatype,
    max_set_size : u64,
    trials       : u64,
    selectivity  : f64,
    skew         : f64,
    density      : f64,
    distribution : DataDistribution,
    seed         : u64,
    mr_offset    : &mut u64,
) -> Result<DatabinDescription, String> {
    let long_length = max_set_size;
    let short_length = (long_length as f64 * skew).round() as u64;
    let set_lengths = vec![long_length, short_length];
    let intersection_length = (short_length as f64 * selectivity).round() as u64;
    let max_value = (long_length as f64 / density).min(datatype.max() as f64).round() as u64;
    let byte_length = (long_length + short_length + intersection_length) * datatype.bytes();

    let mut trials_vec = Vec::<TrialDescription>::with_capacity(trials as usize);
    for i in 0..trials {
        let byte_offset = i * byte_length;
        trials_vec.push(TrialDescription {
            set_lengths: set_lengths.clone(),
            intersection_length,
            byte_offset,
            byte_length,
        });
    }

    let total_byte_length = byte_length * trials;

    let bin = DatabinDescription {
        datatype,
        max_value,
        distribution,
        seed,
        byte_offset: *mr_offset,
        byte_length: total_byte_length,
        trials: trials_vec,
    };

    *mr_offset += total_byte_length;

    Ok(bin)
}

fn parameters_to_description_kset(
    datatype            : Datatype,
    max_set_size        : u64,
    trials              : u64,
    query_selectivity   : f64,
    corpus_skew         : f64,
    corpus_density      : f64,
    data_distribution   : DataDistribution,
    seed                : u64,
    mr_offset           : &mut u64,
    query_size          : u64,
    corpus_size         : u64,
    corpus_distribution : CorpusDistribution,
    query_distribution  : QueryDistribution,
    samples             : u64,
    mr_rng              : &mut impl Rng,
) -> Result<DatabinDescription, String> {
    // Sanity check to avoid infinite loop when sampling distribution
    if corpus_size < query_size {
        return Err(format!("Corpus size ({corpus_size}) must be at least as large as query size ({query_size})."));
    }

    let longest_length_in_corpus = max_set_size;
    let total_trials = trials * samples;

    let mut trials_vec = Vec::<TrialDescription>::with_capacity(total_trials as usize);
    let mut byte_offset = 0u64;
    if corpus_skew == 1.0 {
        let set_lengths: Vec<_> = vec![longest_length_in_corpus; query_size as usize];
        let intersection_length = (longest_length_in_corpus as f64 * query_selectivity).round() as u64;
        let byte_length = (query_size * set_lengths[0] + intersection_length) * datatype.bytes();

        for _ in 0..total_trials {
            let trial = TrialDescription {
                set_lengths: set_lengths.clone(),
                intersection_length,
                byte_offset,
                byte_length,
            };
            trials_vec.push(trial);
            byte_offset += byte_length;
        }
    } else {
        match query_distribution {
            QueryDistribution::Zipf {} => {
                let zipf_exponent = -f64::log(corpus_skew, corpus_size as f64);
                let length_dist = match ZipfDistribution::new(corpus_size as usize, zipf_exponent) {
                    Ok(v)   => v,
                    Err(()) => return Err(format!(
                        "Failed to create discrete Zipf distribution with (num_elements = {corpus_size}, exponent = {zipf_exponent})."
                    )),
                };

                for _ in 0..samples {
                    // NOTE: Not 0-indexed 
                    let set_indices = {
                        let mut set_indices: Vec<usize>;
                        if query_size == corpus_size {
                            set_indices = (1..query_size as usize + 1).collect()
                        } else {
                            set_indices = sample_distribution_unique(query_size as usize, &length_dist, mr_rng)
                        }
                        set_indices.sort_unstable();
                        set_indices
                    };

                    let (set_lengths, intersection_length, byte_length) = match corpus_distribution {
                        CorpusDistribution::Zipf {} => {
                            let mut set_lengths = Vec::<u64>::with_capacity(set_indices.len());
                            let mut total_length = 0u64;
                            for index in set_indices {
                                let length = (longest_length_in_corpus as f64 * (index as f64).powf(-zipf_exponent)).round() as u64;
                                set_lengths.push(length);
                                total_length += length;
                            }

                            let shortest_length = *set_lengths.last().unwrap();
                            let intersection_length = (shortest_length as f64 * query_selectivity).round() as u64;

                            total_length += intersection_length;
                            let byte_length = total_length * datatype.bytes();

                            (set_lengths, intersection_length, byte_length)
                        },
                    };

                    for _ in 0..trials {
                        let trial = TrialDescription {
                            set_lengths: set_lengths.clone(),
                            intersection_length,
                            byte_offset,
                            byte_length,
                        };
                        trials_vec.push(trial);
                        byte_offset += byte_length;
                    }
                }
            },
        };
    }

    let byte_length = byte_offset;
    let max_value = (longest_length_in_corpus as f64 / corpus_density).round() as u64;
    if max_value > datatype.max() {
        return Err(format!(
            "The maximum value ({}) is too large for the datatype ({:?}).",
            max_value, datatype
        ));
    }

    let bin = DatabinDescription {
        datatype,
        max_value,
        distribution: data_distribution,
        seed,
        byte_offset: *mr_offset,
        byte_length,
        trials: trials_vec,
    };

    *mr_offset += byte_length;

    Ok(bin)
}
