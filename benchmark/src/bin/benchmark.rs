#![feature(portable_simd)]
use std::{
    fs::{self, File},
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::{Duration, Instant}, io::Write,
};
use setops::{
    intersect::{self, Intersect2},
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
    #[arg(default_value = "experiment.toml")]
    experiment: PathBuf,
    #[arg(default_value = "datasets/")]
    datasets: PathBuf,
    #[arg(default_value = "results.json")]
    out: PathBuf,
    #[arg(default_value = "8")]
    warmup_rounds: u32,
    // Ignore --bench provided by cargo.
    #[arg(long, action)]
    bench: bool,
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
        gen_dataset_to_algos_map(experiment.experiment);

    let results =
        run_experiments(cli, experiment.dataset, dataset_algos)?;
    
    write_results(results, &cli.out)?;

    Ok(())
}

/// Map datasets to algorithms which need to be run on said dataset.
/// This saves us from running multiple dataset/algorithm pairs twice
/// if present in multiple experiments.
fn gen_dataset_to_algos_map(experiments: Vec<ExperimentEntry>)
    -> HashMap<DatasetId, AlgorithmSet>
{
    let mut dataset_algos: HashMap<String, AlgorithmSet> = HashMap::new();
    for e in experiments {
        dataset_algos
            .entry(e.dataset)
            .or_default()
            .extend(e.algorithms);
    }
    dataset_algos
}

fn run_experiments(
    cli: &Cli,
    datasets: Vec<DatasetInfo>,
    mut dataset_algos: HashMap<DatasetId, AlgorithmSet>)
    -> Result<Results, String>
{
    let mut results =
        HashMap::<DatasetId, DatasetResults>::new();

    for dataset in datasets {
        match dataset {
            DatasetInfo::TwoSet(d) => {
                if let Some(algos) = dataset_algos.get(&d.name) {
                    let dataset_results = DatasetResults{
                        info: d.clone(),
                        algos: run_twoset_bench(cli, &d, algos)?,
                    };
                    
                    dataset_algos.remove(&d.name);
                    results.insert(d.name, dataset_results);
                }
            },
            DatasetInfo::KSet(_) => todo!(),
        }
    }
    assert!(dataset_algos.len() == 0);
    Ok(Results{ datasets: results })
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

            let algo = get_2set_algorithm(name)
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

                let duration = time_twoset(cli, &sets[0], &sets[1], algo);
                times.push(duration.as_nanos() as u64);
            }

            runs.push(ResultRun{x, times});
            println!();
        }
    }
    Ok(algorithm_results)
}

fn time_twoset(
    cli: &Cli,
    set_a: &[i32],
    set_b: &[i32],
    algo: Intersect2<[i32], VecWriter<i32>>) -> Duration
{
    let capacity = set_a.len().min(set_b.len());
    // Warmup
    for _ in 0..cli.warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        std::hint::black_box(algo(set_a, set_b, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    std::hint::black_box(algo(set_a, set_b, &mut writer));
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

fn get_2set_algorithm(name: &str) -> Option<Intersect2<[i32], VecWriter<i32>>> {
    match name {
        "naive_merge"      => Some(intersect::naive_merge),
        "branchless_merge" => Some(intersect::branchless_merge),
        "bmiss_scalar_3x"  => Some(intersect::bmiss_scalar_3x),
        "bmiss_scalar_4x"  => Some(intersect::bmiss_scalar_4x),
        "galloping"        => Some(intersect::galloping),
        "baezayates"       => Some(intersect::baezayates),
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
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "galloping_sse"    => Some(intersect::galloping_sse),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2"   => Some(intersect::shuffling_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2"   => Some(intersect::galloping_avx2),
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

//fn run_fesia_2set<H, S, const LANES: usize, const HASH_SCALE: usize>(
//    b: &mut Bencher,
//    intersect: fn(&Fesia<H, S, LANES, HASH_SCALE>, &Fesia<H, S, LANES, HASH_SCALE>, &mut VecWriter<i32>),
//    output_len: usize,
//    generator: impl Fn() -> (Vec<i32>, Vec<i32>))
//where
//    H: IntegerHash,
//    S: SimdElement + MaskElement,
//    LaneCount<LANES>: SupportedLaneCount,
//    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
//    Mask<S, LANES>: ToBitMask<BitMask=u8>,
//{
//    use intersect::{Fesia, MixHash};
//    b.iter_batched(
//        || {
//            let (left, right) = generator();
//            (
//                Fesia::<MixHash, 4>::from_sorted(&left),
//                Fesia::<MixHash, 4>::from_sorted(&right),
//                VecWriter::with_capacity(output_len)
//            )
//        },
//        |(set_a, set_b, mut writer)| intersect(set_a.as_view(), set_b.as_view(), &mut writer),
//        criterion::BatchSize::LargeInput,
//    );
//}

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
