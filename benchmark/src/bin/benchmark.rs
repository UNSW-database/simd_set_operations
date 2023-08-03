use std::{
    fs::{self, File},
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::Duration, io::Write,
};
use setops::{
    intersect::{self, Intersect2, IntersectK, fesia::HashScale},
    visitor::VecWriter, bsr::Intersect2Bsr,
};
use colored::*;

use benchmark::{
    fmt_open_err, path_str,
    schema::*, datafile::{self, DatafileSet}, harness,
};
use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "experiment.toml", long)]
    experiment: PathBuf,
    #[arg(default_value = "datasets/", long)]
    datasets: PathBuf,
    #[arg(default_value = "results.json", long)]
    out: PathBuf,
    #[arg(default_value = "8", long)]
    warmup_rounds: u32,
    // Ignore --bench provided by cargo.
    #[arg(long, action)]
    bench: bool,
    experiments: Vec<String>,
}

fn main() {
    let cli = Cli::parse();

    if cfg!(debug_assertions) {
        println!("{}", "warning: running in debug mode".yellow().bold());
    }

    if let Err(e) = bench_from_files(&cli) {
        let msg = format!("error: {}", e);
        println!("{}", msg.red().bold());
    }
}

fn bench_from_files(cli: &Cli) -> Result<(), String> {
    let experiment_toml = fs::read_to_string(&cli.experiment)
        .map_err(|e| fmt_open_err(e, &cli.experiment))?;

    let experiment: Experiment = toml::from_str(&experiment_toml)
        .map_err(|e| format!(
            "invalid toml file {}: {}",
            path_str(&cli.experiment), e
        ))?;

    let dataset_algos =
        gen_dataset_to_algos_map(cli, &experiment.experiment);
        
    if dataset_algos.len() == 0 {
        return Err("no algorithm matches found".to_string());
    }

    let results = run_experiments(cli, experiment, dataset_algos)?;
    
    write_results(results, &cli.out)?;

    Ok(())
}

type AlgorithmSet = HashSet<String>;
/// Map datasets to algorithms which need to be run on said dataset.
/// This saves us from running multiple dataset/algorithm pairs twice
/// if present in multiple experiments.
fn gen_dataset_to_algos_map(cli: &Cli, experiments: &Vec<ExperimentEntry>)
    -> HashMap<DatasetId, AlgorithmSet>
{
    let mut dataset_algos: HashMap<String, AlgorithmSet> = HashMap::new();
    for e in experiments {
        if cli.experiments.len() == 0 || cli.experiments.contains(&e.name) {
            dataset_algos
                .entry(e.dataset.clone())
                .or_default()
                .extend(e.algorithms.clone());
        }
    }
    dataset_algos
}

fn run_experiments(
    cli: &Cli,
    experiment: Experiment,
    dataset_algos: HashMap<DatasetId, AlgorithmSet>)
    -> Result<Results, String>
{
    let mut results =
        HashMap::<DatasetId, DatasetResults>::new();

    for dataset in &experiment.dataset {
        if let Some(algos) = dataset_algos.get(&dataset.name) {
            let dataset_results = DatasetResults{
                info: dataset.clone(),
                algos: run_bench(cli, &dataset, algos)?,
            };
            results.insert(dataset.name.clone(), dataset_results);
        }
    }

    let experiments = if cli.experiments.len() > 0 {
        experiment.experiment
            .into_iter()
            .filter(|e| cli.experiments.contains(&e.name))
            .collect()
    } else {
        experiment.experiment
    };

    Ok(Results{
        experiments: experiments,
        datasets: results,
    })
}

fn run_bench(
    cli: &Cli,
    info: &DatasetInfo,
    algos: &HashSet<String>) -> Result<AlgorithmResults, String>
{
    println!("{}", &info.name.green().bold());

    let dataset_dir = PathBuf::from(&cli.datasets)
        .join(&info.name);

    let mut algorithm_results: AlgorithmResults =
        algos.iter().map(|a| (a.clone(), Vec::new())).collect();

    for x in benchmark::xvalues(info) {
        // later: look at throughput?
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());
        let xdir = dataset_dir.join(x.to_string());

        for (name, runs) in &mut algorithm_results {
            print!("{: <20}", name);

            let algorithm = get_algorithm(name)
                .ok_or_else(|| format!("unknown algorithm {}", name))?;

            let pairs = fs::read_dir(&xdir)
                .map_err(|e| fmt_open_err(e, &xdir))?;

            let mut times: Vec<u64> = Vec::new();

            for (i, pair_path) in pairs.enumerate() {
                print!("{} ", i);
                let _ = std::io::stdout().flush();

                let pair_path = pair_path
                    .map_err(|e| format!(
                        "unable to open directory entry in {}: {}",
                        path_str(&xdir), e.to_string()
                    ))?;

                let datafile_path = pair_path.path();
                let datafile = File::open(&datafile_path)
                    .map_err(|e| fmt_open_err(e, &datafile_path))?;

                let sets = datafile::from_reader(datafile)
                    .map_err(|e| format!(
                        "invalid datafile {}: {}",
                        path_str(&datafile_path),
                        e.to_string())
                    )?;

                let duration = time_algorithm(cli, &sets, &algorithm)?;
                times.push(duration.as_nanos() as u64);
            }

            runs.push(ResultRun{x, times});
            println!();
        }
    }
    Ok(algorithm_results)
}

fn time_algorithm(
    cli: &Cli,
    sets: &[DatafileSet],
    algorithm: &Algorithm) -> Result<Duration, String>
{
    use Algorithm::*;
    use FesiaBlockBits::*;
    use FesiaSimdType::*;
    use setops::intersect::fesia::*;

    match *algorithm {
        TwoSet(intersect) =>
            if sets.len() == 2 {
                harness::time_twoset(cli.warmup_rounds, &sets[0], &sets[1], intersect)
            }
            else if sets.len() > 2 {
                harness::time_svs(cli.warmup_rounds, sets, intersect)
            }
            else {
                Err(format!("twoset: cannot intersect {} sets", sets.len()))
            },
        TwoSetBsr(intersect) => 
            if sets.len() == 2 {
                harness::time_bsr(cli.warmup_rounds, &sets[0], &sets[1], intersect)
            }
            else {
                Err(format!("BSR: cannot intersect {} sets", sets.len()))
            },
        KSet(intersect) => harness::time_kset(cli.warmup_rounds, sets, intersect),
        Roaring(intersect, intersect_svs) =>
            if sets.len() == 2 {
                Ok(intersect(cli.warmup_rounds, &sets[0], &sets[1]))
            }
            else if sets.len() > 2 {
                Ok(intersect_svs(cli.warmup_rounds, &sets))
            }
            else {
                Err(format!("croaring: cannot intersect {} sets", sets.len()))
            },
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        Fesia(B8,  Sse,    h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i8,  u16, 16, VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        Fesia(B16, Sse,    h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i16, u8,  8,  VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        Fesia(B32, Sse,    h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i32, u8,  4,  VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        Fesia(B8,  Avx2,   h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i8,  u32, 32, VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        Fesia(B16, Avx2,   h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i16, u16, 16, VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        Fesia(B32, Avx2,   h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i32, u8,  8,  VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        Fesia(B8,  Avx512, h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i8,  u64, 64, VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        Fesia(B16, Avx512, h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i16, u32, 32, VecWriter<i32>>),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        Fesia(B32, Avx512, h) => harness::time_fesia(cli.warmup_rounds, sets, h, fesia::<MixHash, i32, u16, 16, VecWriter<i32>>),
    }
}

fn write_results(results: Results, path: &PathBuf) -> Result<(), String> {
    let results_file = File::options()
        .write(true).create(true).truncate(true)
        .open(path)
        .map_err(|e| fmt_open_err(e, path))?;

    serde_json::to_writer(results_file, &results)
        .map_err(|e| format!(
            "failed to write {}: {}",
            path_str(path), e.to_string()
        ))?;

    Ok(())
}

enum Algorithm {
    TwoSet(Intersect2<[i32], VecWriter<i32>>),
    TwoSetBsr(Intersect2Bsr),
    KSet(IntersectK<DatafileSet, VecWriter<i32>>),
    Fesia(FesiaBlockBits, FesiaSimdType, HashScale),
    Roaring(
        fn(warmup_rounds: u32, set_a: &[i32], set_b: &[i32]) -> Duration,
        fn(warmup_rounds: u32, sets: &[DatafileSet]) -> Duration
    ),
}
enum FesiaBlockBits {
    B8, B16, B32
}
enum FesiaSimdType {
    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    Sse,
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    Avx2,
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    Avx512
}

fn get_algorithm(name: &str) -> Option<Algorithm> {
    use Algorithm::*;
    try_parse_twoset(name).map(|i| TwoSet(i))
        .or_else(|| try_parse_bsr(name).map(|i| TwoSetBsr(i)))
        .or_else(|| try_parse_kset(name).map(|i| KSet(i)))
        .or_else(|| try_parse_roaring(name))
        .or_else(|| try_parse_fesia(name))
}

fn try_parse_twoset(name: &str) -> Option<Intersect2<[i32], VecWriter<i32>>> {
    match name {
        "naive_merge"      => Some(intersect::naive_merge),
        "branchless_merge" => Some(intersect::branchless_merge),
        "bmiss_scalar_3x"  => Some(intersect::bmiss_scalar_3x),
        "bmiss_scalar_4x"  => Some(intersect::bmiss_scalar_4x),
        "galloping"        => Some(intersect::galloping),
        "baezayates"       => Some(intersect::baezayates),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse"    => Some(intersect::shuffling_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse"    => Some(intersect::broadcast_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse"        => Some(intersect::bmiss),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse_sttni"  => Some(intersect::bmiss_sttni),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter"          => Some(intersect::qfilter),
        #[cfg(all(feature = "simd"))]
        "galloping_sse"    => Some(intersect::galloping_sse),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2"   => Some(intersect::shuffling_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2"   => Some(intersect::broadcast_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2"   => Some(intersect::galloping_avx2),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512"       => Some(intersect::shuffling_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "vp2intersect_emulation" => Some(intersect::vp2intersect_emulation),
        #[cfg(all(feature = "simd", target_feature = "avx512cd"))]
        "conflict_intersect"     => Some(intersect::conflict_intersect),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512"       => Some(intersect::galloping_avx512),
        _ => None,
    }
}

fn try_parse_bsr(name: &str) -> Option<Intersect2Bsr> {
    match name {
        "branchless_merge_bsr" => Some(intersect::branchless_merge_bsr),
        "galloping_bsr"        => Some(intersect::galloping_bsr),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_bsr"    => Some(intersect::shuffling_sse_bsr),
        // #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        // "broadcast_sse"    => Some(intersect::broadcast_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_bsr"          => Some(intersect::qfilter_bsr),
        #[cfg(all(feature = "simd"))]
        "galloping_sse_bsr"    => Some(intersect::galloping_sse_bsr),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2_bsr"   => Some(intersect::shuffling_avx2_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2_bsr"   => Some(intersect::galloping_avx2_bsr),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512"       => Some(intersect::shuffling_avx512_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512_bsr"       => Some(intersect::galloping_avx512_bsr),
        _ => None,
    }
}

fn try_parse_kset(name: &str) -> Option<IntersectK<DatafileSet, VecWriter<i32>>> {
    match name {
        "adaptive" => Some(intersect::adaptive),
        "small_adaptive" => Some(intersect::small_adaptive),
        _ => None,
    }
}

fn try_parse_roaring(name: &str) -> Option<Algorithm> {
    use Algorithm::*;
    match name {
        "croaring" => Some(Roaring(
            harness::time_croaring_2set,
            harness::time_croaring_svs)),
        "roaringrs" => Some(Roaring(
            harness::time_roaringrs_2set,
            harness::time_roaringrs_svs)),
        _ => None,
    }
}

fn try_parse_fesia(name: &str) -> Option<Algorithm> {
    use FesiaBlockBits::*;
    use FesiaSimdType::*;
    use Algorithm::*;

    let last_underscore = name.rfind("_")?;

    let hash_scale = &name[last_underscore+1..];
    if hash_scale.is_empty() {
        return None;
    }

    let hash_scale: HashScale = hash_scale.parse().ok()?;
    if hash_scale <= 0.0 {
        return None;
    }

    let prefix = &name[..last_underscore];
    match prefix {
        // FESIA
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia8_sse"   => Some(Fesia(B8,  Sse, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia16_sse"  => Some(Fesia(B16, Sse, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia32_sse"  => Some(Fesia(B32, Sse, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia8_avx2"  => Some(Fesia(B8, Avx2, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia16_avx2" => Some(Fesia(B16, Avx2, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia32_avx2" => Some(Fesia(B32, Avx2, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia8_avx512"  => Some(Fesia(B8, Avx512, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia16_avx512" => Some(Fesia(B16, Avx512, hash_scale)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia32_avx512" => Some(Fesia(B32, Avx512, hash_scale)),
        _ => None,
    }
}
