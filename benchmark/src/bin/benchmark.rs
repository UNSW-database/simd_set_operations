use std::{
    fs::{self, File},
    collections::{HashMap, HashSet},
    path::PathBuf,
    time::Duration,
};
use html_builder::Html5;
use setops::{
    intersect::{self, Intersect2, IntersectK, fesia::HashScale},
    visitor::{VecWriter, Visitor, SimdVisitor4, SimdVisitor8, SimdVisitor16, Counter}, bsr::{Intersect2Bsr, BsrVec},
};
use colored::*;

use benchmark::{
    fmt_open_err, path_str,
    schema::*, datafile::{self, DatafileSet}, harness::{self, FesiaIntersect, HarnessVisitor, DurationResult}, get_algorithms,
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
    // Ignore --bench provided by cargo.
    #[arg(long, action)]
    bench: bool,
    #[arg(long, action)]
    count_only: bool,
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

    let dataset_algos = gen_dataset_to_algos_map(cli, &experiment)?;
        
    if dataset_algos.len() == 0 {
        return Err("no algorithm matches found".to_string());
    }

    let results = run_experiments(cli, experiment, dataset_algos)?;
    
    write_results(results, &cli.out)?;

    Ok(())
}

type AlgorithmSet = HashSet<String>;
/// Map each dataset to algorithms which need to be run on it.
/// This saves us from running multiple dataset/algorithm pairs twice
/// if present in multiple experiments.
fn gen_dataset_to_algos_map(cli: &Cli, experiment: &Experiment)
    -> Result<HashMap<DatasetId, AlgorithmSet>, String>
{
    let mut dataset_algos: HashMap<String, AlgorithmSet> = HashMap::new();
    for e in &experiment.experiment {
        if cli.experiments.len() == 0 || cli.experiments.contains(&e.name) {

            let algorithms =
                get_algorithms(&experiment.algorithm_sets, &e.algorithm_set)?;

            dataset_algos
                .entry(e.dataset.clone())
                .or_default()
                .extend(algorithms.clone());
        }
    }
    Ok(dataset_algos)
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
        algorithm_sets: experiment.algorithm_sets,
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
        let xlabel = format!("[x: {:4}]", x);
        println!("{}", xlabel.bold());
        let xdir = dataset_dir.join(x.to_string());

        for (name, runs) in &mut algorithm_results {
            println!("  {}", name);

            let pairs: Result<Vec<PathBuf>, String> = fs::read_dir(&xdir)
                .map_err(|e| fmt_open_err(e, &xdir))?
                .map(|s| s
                    .map_err(|e| format!(
                        "unable to open directory entry in {}: {}",
                        path_str(&xdir), e.to_string()
                    ))
                    .map(|s| s.path())
                )
                .collect();

            let pairs = pairs?;

            let timer = if cli.count_only {
                get_algorithm::<Counter>(name, cli.count_only)
                    .ok_or_else(|| format!("unknown algorithm {}", name))
            }
            else {
                get_algorithm::<VecWriter<i32>>(name, cli.count_only)
                    .ok_or_else(|| format!("unknown algorithm {}", name))
            }?;

            let run = run_algorithm_on_x(x, timer, pairs)?;
            runs.push(run);
        }
    }
    Ok(algorithm_results)
}

fn run_algorithm_on_x(x: u32, timer: Timer, datafile_paths: Vec<PathBuf>)
    -> Result<ResultRun, String>
{
    let mut times: Vec<u64> = Vec::new();

    for datafile_path in &datafile_paths {
        let datafile = File::open(datafile_path)
            .map_err(|e| fmt_open_err(e, datafile_path))?;

        let sets = datafile::from_reader(datafile)
            .map_err(|e| format!(
                "invalid datafile {}: {}",
                path_str(datafile_path),
                e.to_string())
            )?;

        const TARGET_WARMUP: Duration = Duration::from_millis(1000);
        let warmup = TARGET_WARMUP.div_f32(datafile_paths.len() as f32);

        let duration = timer.run(warmup, &sets)?;
        times.push(duration.as_nanos() as u64);
    }

    Ok(ResultRun{x, times})
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

type TwosetTimer = Box<dyn Fn(Duration, &[i32], &[i32]) -> DurationResult>;
type KsetTimer = Box<dyn Fn(Duration, &[DatafileSet]) -> DurationResult>;

struct Timer {
    twoset: Option<TwosetTimer>,
    kset: Option<KsetTimer>,
}

impl Timer {
    fn run(&self, warmup: Duration, sets: &[DatafileSet]) -> DurationResult {
        if sets.len() == 2 {
            if let Some(twoset) = &self.twoset {
                twoset(warmup, &sets[0], &sets[1])
            }
            else if let Some(kset) = &self.kset {
                kset(warmup, sets)
            }
            else {
                Err("2-set or k-set intersection not supported".to_string())
            }
        }
        else {
            if let Some(kset) = &self.kset {
                kset(warmup, sets)
            }
            else {
                Err("k-set intersection not supported".to_string())
            }
        }
    }
}

fn get_algorithm<V>(name: &str, count_only: bool) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + HarnessVisitor + TwosetTimingSpec<V>
{
    try_parse_twoset::<V>(name)
        .or_else(|| try_parse_bsr(name))
        .or_else(|| try_parse_kset::<V>(name))
        .or_else(|| try_parse_roaring(name, count_only))
        .or_else(|| try_parse_fesia::<V>(name))
}

fn try_parse_twoset<V>(name: &str) -> Option<Timer> 
where
    V: Visitor<i32> + SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + HarnessVisitor + TwosetTimingSpec<V>
{
    let maybe_intersect: Option<Intersect2<[i32], V>> = match name {
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

    };
    maybe_intersect.map(|intersect| V::twoset_timer(intersect))
}

trait TwosetTimingSpec<V> {
    fn twoset_timer(i: Intersect2<[i32], V>) -> Timer;
}

impl TwosetTimingSpec<VecWriter<i32>> for VecWriter<i32> {
    fn twoset_timer(i: Intersect2<[i32], VecWriter<i32>>) -> Timer {
        Timer {
            twoset: Some(Box::new(|warmup, a, b| harness::time_twoset(warmup, a, b, i))),
            kset: Some(Box::new(|warmup, sets| harness::time_svs::<VecWriter<i32>>(warmup, sets, i))),
        }
    }
}

impl TwosetTimingSpec<Counter> for Counter {
    fn twoset_timer(i: Intersect2<[i32], Counter>) -> Timer {
        Timer {
            twoset: Some(Box::new(|warmup, a, b| harness::time_twoset(warmup, a, b, i))),
            kset: None,
        }
    }
}

fn try_parse_bsr(name: &str) -> Option<Timer> {
    let maybe_intersect: Option<Intersect2Bsr> = match name {
        "branchless_merge_bsr" => Some(intersect::branchless_merge_bsr),
        "galloping_bsr"        => Some(intersect::galloping_bsr),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_bsr"    => Some(intersect::shuffling_sse_bsr),
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
        "shuffling_avx512_bsr"       => Some(intersect::shuffling_avx512_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512_bsr"       => Some(intersect::galloping_avx512_bsr),
        _ => None,
    };
    maybe_intersect.map(|intersect: Intersect2Bsr| Timer {
        twoset: Some(Box::new(move |warmup, a, b| harness::time_bsr(warmup, a, b, intersect))),
        kset: None,
    })
}

fn try_parse_kset<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + HarnessVisitor
{
    let maybe_intersect: Option<IntersectK<DatafileSet, V>> = match name {
        "adaptive"              => Some(intersect::adaptive),
        "small_adaptive"        => Some(intersect::small_adaptive),
        "small_adaptive_sorted" => Some(intersect::small_adaptive_sorted),
        _ => None,
    };
    maybe_intersect.map(|intersect| Timer {
        twoset: None,
        kset: Some(Box::new(move |warmup, sets| harness::time_kset(warmup, sets, intersect))),
    })
}

fn try_parse_roaring(name: &str, count_only: bool) -> Option<Timer> { 
    match name {
        "croaring" => Some(Timer {
            twoset:
                Some(Box::new(move |warmup, a, b| Ok(harness::time_croaring_2set(warmup, a, b, count_only)))),
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_croaring_svs(warmup, sets))))
                },
            }),
        "roaringrs" => Some(Timer {
            twoset:
                if count_only { None } else {
                    Some(Box::new(|warmup, a, b| Ok(harness::time_roaringrs_2set(warmup, a, b))))
                },
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_roaringrs_svs(warmup, sets))))
                },
            }),
        _ => None,
    }
}

fn try_parse_fesia<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + HarnessVisitor
{
    use FesiaIntersect::*;
    use intersect::fesia::*;

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

    const FESIA_HASH: &str = "fesia_hash";
    const FESIA_SHUFFLING: &str = "fesia_shuffling";
    const FESIA: &str = "fesia";

    let (intersect, rest) =
        if prefix.len() >= FESIA_HASH.len() && &prefix[..FESIA_HASH.len()] == FESIA_HASH {
            (Skewed, &prefix[FESIA_HASH.len()..])
        }
        else if prefix.len() >= FESIA_SHUFFLING.len() && &prefix[..FESIA_SHUFFLING.len()] == FESIA_SHUFFLING {
            (SimilarSizeShuffling, &prefix[FESIA_SHUFFLING.len()..])
        }
        else if prefix.len() >= FESIA.len() && &prefix[..FESIA.len()] == FESIA {
            (SimilarSize, &prefix[FESIA.len()..])
        }
        else {
            return None;
        };

    let maybe_twoset_timer: Option<TwosetTimer> =
    match rest {
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "8_sse" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i8, u16, 16, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "16_sse" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i16, u8, 8, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "32_sse" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i32, u8, 4, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "8_avx2" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i8, u32, 32, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "16_avx2" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i16, u16, 16, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "32_avx2" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i32, u8, 8, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f", intersect))]
        "8_avx512" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i8, u64, 64, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f", intersect))]
        "16_avx512" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i16, u32, 32, V>(warmup, a, b, hash_scale, intersect)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f", intersect))]
        "32_avx512" => Some(Box::new(move |warmup, a, b|
            harness::time_fesia::<MixHash, i32, u16, 16, V>(warmup, a, b, hash_scale, intersect)
        )),
        _ => None,
    };

    maybe_twoset_timer.map(|twoset_timer| Timer {
        twoset: Some(twoset_timer),
        kset: None,
    })
}
