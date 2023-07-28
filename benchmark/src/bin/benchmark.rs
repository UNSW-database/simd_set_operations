#![feature(portable_simd)]
use std::{
    fs::{self, File},
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::{Duration, Instant}, io::Write, simd::*, ops::BitAnd,
};
use core::fmt::Debug;
use setops::{
    intersect::{self, Intersect2, fesia::{Fesia, IntegerHash, SetWithHashScale}},
    visitor::VecWriter,
};
use colored::*;

//use roaring::{RoaringBitmap, MultiOps};
//#[cfg(feature = "simd")]
//use setops::intersect::fesia::*;
//type KSetAlg = (&'static str, IntersectK<Vec<i32>, VecWriter<i32>>);

use benchmark::{
    fmt_open_err, path_str,
    schema::*, datafile,
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

type AlgorithmSet = HashSet<String>;
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

    let results =
        run_experiments(cli, experiment, dataset_algos)?;
    
    write_results(results, &cli.out)?;

    Ok(())
}

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
        match dataset {
            DatasetInfo::TwoSet(d) => {
                if let Some(algos) = dataset_algos.get(&d.name) {
                    let dataset_results = DatasetResults{
                        info: d.clone(),
                        algos: run_twoset_bench(cli, &d, algos)?,
                    };
                    results.insert(d.name.clone(), dataset_results);
                }
            },
            DatasetInfo::KSet(_) => todo!(),
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

fn run_twoset_bench(
    cli: &Cli,
    info: &TwoSetDatasetInfo,
    algos: &HashSet<String>) -> Result<AlgorithmResults, String>
{
    println!("{}", &info.name.green().bold());

    let dataset_dir = PathBuf::from(&cli.datasets)
        .join("2set")
        .join(&info.name);

    let mut algorithm_results: AlgorithmResults =
        algos.iter().map(|a| (a.clone(), Vec::new())).collect();

    for x in benchmark::xvalues(info) {
        // later: look at throughput?
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());

        for (name, runs) in &mut algorithm_results {
            print!("{: <20}", name);

            let algorithm = get_algorithm(name)
                .ok_or_else(|| format!("unknown algorithm {}", name))?;

            let xdir = dataset_dir.join(x.to_string());
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

                if sets.len() != 2 {
                    return Err(format!("expected 2 sets, got {}", sets.len()));
                }

                let duration = time_algorithm(cli, &sets[0], &sets[1], &algorithm);
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
    set_a: &[i32],
    set_b: &[i32],
    algorithm: &Algorithm) -> Duration
{
    use Algorithm::*;
    use FesiaBlockBits::*;
    use FesiaSimdType::*;
    use setops::intersect::fesia::*;

    match *algorithm {
        TwoSet(intersect)   => time_twoset(cli, set_a, set_b, intersect),
        Fesia(B8,  Sse)     => time_fesia(cli, set_a, set_b, fesia::<MixHash, i8,  u16, 16, VecWriter<i32>>),
        Fesia(B16, Sse)     => time_fesia(cli, set_a, set_b, fesia::<MixHash, i16, u8,  8,  VecWriter<i32>>),
        Fesia(B32, Sse)     => time_fesia(cli, set_a, set_b, fesia::<MixHash, i32, u8,  4,  VecWriter<i32>>),
        Fesia(B8,  Avx2)    => time_fesia(cli, set_a, set_b, fesia::<MixHash, i8,  u32, 32, VecWriter<i32>>),
        Fesia(B16, Avx2)    => time_fesia(cli, set_a, set_b, fesia::<MixHash, i16, u16, 16, VecWriter<i32>>),
        Fesia(B32, Avx2)    => time_fesia(cli, set_a, set_b, fesia::<MixHash, i32, u8,  8,  VecWriter<i32>>),
        Fesia(B8,  Avx512)  => time_fesia(cli, set_a, set_b, fesia::<MixHash, i8,  u64, 64, VecWriter<i32>>),
        Fesia(B16, Avx512)  => time_fesia(cli, set_a, set_b, fesia::<MixHash, i16, u32, 32, VecWriter<i32>>),
        Fesia(B32, Avx512)  => time_fesia(cli, set_a, set_b, fesia::<MixHash, i32, u16, 16, VecWriter<i32>>),
    }
}

fn time_twoset(
    cli: &Cli,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> Duration
{
    let capacity = set_a.len().min(set_b.len());
    // Warmup
    for _ in 0..cli.warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        std::hint::black_box(intersect(set_a, set_b, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    std::hint::black_box(intersect(set_a, set_b, &mut writer));
    start.elapsed()
}

fn time_fesia<H, S, M, const LANES: usize>(
    cli: &Cli,
    set_a: &[i32],
    set_b: &[i32],
    intersect: fn(&Fesia<H, S, M, LANES>, &Fesia<H, S, M, LANES>, &mut VecWriter<i32>),
) -> Duration
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    let capacity = set_a.len().min(set_b.len());

    // TODO: allow hash scale to be changed.
    let set_a = Fesia::<H, S, M, LANES>::from_sorted(set_a, 2);
    let set_b = Fesia::<H, S, M, LANES>::from_sorted(set_b, 2);

    // Warmup
    for _ in 0..cli.warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        std::hint::black_box(intersect(&set_a, &set_b, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    std::hint::black_box(intersect(&set_a, &set_b, &mut writer));
    start.elapsed()
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
    Fesia(FesiaBlockBits, FesiaSimdType),
}
enum FesiaBlockBits {
    B8, B16, B32
}
enum FesiaSimdType {
    Sse, Avx2, Avx512
}

fn get_algorithm(name: &str) -> Option<Algorithm> {
    use FesiaBlockBits::*;
    use FesiaSimdType::*;
    use Algorithm::*;
    match name {
        "naive_merge"      => Some(TwoSet(intersect::naive_merge)),
        "branchless_merge" => Some(TwoSet(intersect::branchless_merge)),
        "bmiss_scalar_3x"  => Some(TwoSet(intersect::bmiss_scalar_3x)),
        "bmiss_scalar_4x"  => Some(TwoSet(intersect::bmiss_scalar_4x)),
        "galloping"        => Some(TwoSet(intersect::galloping)),
        "baezayates"       => Some(TwoSet(intersect::baezayates)),
        "baezayates_opt"   => Some(TwoSet(intersect::baezayates_opt)),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse"    => Some(TwoSet(intersect::shuffling_sse)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse"    => Some(TwoSet(intersect::broadcast_sse)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse"        => Some(TwoSet(intersect::bmiss)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse_sttni"  => Some(TwoSet(intersect::bmiss_sttni)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter"          => Some(TwoSet(intersect::qfilter)),
        #[cfg(all(feature = "simd"))]
        "galloping_sse"    => Some(TwoSet(intersect::galloping_sse)),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2"   => Some(TwoSet(intersect::shuffling_avx2)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2"   => Some(TwoSet(intersect::broadcast_avx2)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2"   => Some(TwoSet(intersect::galloping_avx2)),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512"       => Some(TwoSet(intersect::shuffling_avx512)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "vp2intersect_emulation" => Some(TwoSet(intersect::vp2intersect_emulation)),
        #[cfg(all(feature = "simd", target_feature = "avx512cd"))]
        "conflict_intersect"     => Some(TwoSet(intersect::conflict_intersect)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512"       => Some(TwoSet(intersect::galloping_avx512)),
        // FESIA
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia8_sse"   => Some(Fesia(B8,  Sse)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia16_sse"  => Some(Fesia(B16, Sse)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "fesia32_sse"  => Some(Fesia(B32, Sse)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia8_avx2"  => Some(Fesia(B8, Avx2)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia16_avx2" => Some(Fesia(B16, Avx2)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "fesia32_avx2" => Some(Fesia(B32, Avx2)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia8_avx512"  => Some(Fesia(B8, Avx512)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia16_avx512" => Some(Fesia(B16, Avx512)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "fesia32_avx512" => Some(Fesia(B32, Avx512)),
        _ => None,
    }
}


// 2-set:
// array 2set (done)

// roaring
//   || {
//       (RoaringBitmap::from_sorted(&left), RoaringBitmap::from_sorted(&right))
//   },
//   |(mut set_a, set_b)| set_a &= set_b,

// fesia
//    run_custom_2set::<Fesia8Sse<8>>(b, intersect::fesia::fesia, size, generator)
//    run_fesia_2set(b, intersect::fesia::fesia_sse_shuffling, size, generator)
//    run_custom_2set(b, intersect::hash_set_intersect, size, generator)
//    run_custom_2set(b, intersect::btree_set_intersect, size, generator)


// K-SET:
// svs two set
// k-set (adaptive)
// roaring
//        b.iter_batched(
//            || Vec::from_iter(
//                generator().iter().map(|s| RoaringBitmap::from_sorted(&s))
//            ),
//            |sets| sets.intersection(),
//            criterion::BatchSize::LargeInput,
//        );
//    }
// later: fesia


//const TWOSET_ARRAY_SCALAR: [&'static str; 6] = [
//    "naive_merge",
//    "branchless_merge",
//    "bmiss_scalar_3x",
//    "bmiss_scalar_4x",
//    "galloping",
//    "baezayates",
//];
//
//#[cfg(all(feature = "simd", target_feature = "ssse3"))]
//const TWOSET_ARRAY_SSE: [&'static str; 6] = [
//    "shuffling_sse",
//    "broadcast_sse",
//    "bmiss_sse",
//    "bmiss_sse_sttni",
//    "qfilter",
//    "galloping_sse",
//];
//#[cfg(all(feature = "simd", target_feature = "avx2"))]
//const TWOSET_ARRAY_AVX2: [&'static str; 2] = [
//    "shuffling_avx2",
//    "galloping_avx2",
//];
//#[cfg(all(feature = "simd", target_feature = "avx512f"))]
//const TWOSET_ARRAY_AVX512: [&'static str; 4] = [
//    "shuffling_avx512",
//    "vp2intersect_emulation",
//    "conflict_intersect",
//    "galloping_avx512",
//];
//#[cfg(not(target_feature = "ssse3"))]
//const TWOSET_ARRAY_SSE: [&'static str; 0] = [];
//#[cfg(not(target_feature = "avx2"))]
//const TWOSET_ARRAY_AVX2: [&'static str; 0] = [];
//#[cfg(not(target_feature = "avx512f"))]
//const TWOSET_ARRAY_AVX512: [&'static str; 0] = [];
//
//const KSET_ARRAY_SCALAR: [KSetAlg; 3] = [
//    ("adaptive", intersect::adaptive),
//    ("small_adaptive", intersect::small_adaptive),
//    ("small_adaptive_sorted", intersect::small_adaptive_sorted),
//];
