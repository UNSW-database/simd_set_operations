#![feature(portable_simd)]
use std::{fs::{self, File}, collections::{HashMap, HashSet}, path::PathBuf, time::{Duration, Instant}, os::unix::raw::time_t};
use criterion::{
    criterion_group, criterion_main, Bencher, BenchmarkId, Criterion,
    BenchmarkGroup, measurement::WallTime, PlotConfiguration, AxisScale, BatchSize, SamplingMode
};
use roaring::{RoaringBitmap, MultiOps};
use setops::{
    intersect::{self, Intersect2, IntersectK},
    visitor::VecWriter,
    Set,
};
#[cfg(feature = "simd")]
use setops::intersect::fesia::*;
use benchmarks::schema::*;

const SAMPLE_SIZE: usize = 10;

type KSetAlg = (&'static str, IntersectK<Vec<i32>, VecWriter<i32>>);

const TWOSET_ARRAY_SCALAR: [&'static str; 6] = [
    "naive_merge",
    "branchless_merge",
    "bmiss_scalar_3x",
    "bmiss_scalar_4x",
    "galloping",
    "baezayates",
];

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
const TWOSET_ARRAY_SSE: [&'static str; 6] = [
    "shuffling_sse",
    "broadcast_sse",
    "bmiss_sse",
    "bmiss_sse_sttni",
    "qfilter",
    "galloping_sse",
];
#[cfg(all(feature = "simd", target_feature = "avx2"))]
const TWOSET_ARRAY_AVX2: [&'static str; 2] = [
    "shuffling_avx2",
    "galloping_avx2",
];
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
const TWOSET_ARRAY_AVX512: [&'static str; 4] = [
    "shuffling_avx512",
    "vp2intersect_emulation",
    "conflict_intersect",
    "galloping_avx512",
];
#[cfg(not(target_feature = "ssse3"))]
const TWOSET_ARRAY_SSE: [&'static str; 0] = [];
#[cfg(not(target_feature = "avx2"))]
const TWOSET_ARRAY_AVX2: [&'static str; 0] = [];
#[cfg(not(target_feature = "avx512f"))]
const TWOSET_ARRAY_AVX512: [&'static str; 0] = [];

const KSET_ARRAY_SCALAR: [KSetAlg; 3] = [
    ("adaptive", intersect::adaptive),
    ("small_adaptive", intersect::small_adaptive),
    ("small_adaptive_sorted", intersect::small_adaptive_sorted),
];

//criterion_group!(benches, bench_from_files);
//criterion_main!(benches);

fn main() {
    bench_from_files();
}

fn bench_from_files() {
    //let mut group = c.benchmark_group("intersect_kset_same_size");
    //group.sample_size(SAMPLE_SIZE);

    //println!("{}", std::env::current_dir().unwrap().to_str().unwrap());

    let experiment_toml = fs::read_to_string("../experiment.toml")
        .unwrap();
    let experiment: Experiment = toml::from_str(&experiment_toml)
        .unwrap();

    let mut datasets: HashMap<String, HashSet<String>> = HashMap::new();
    // Map datasets to algorithms.
    for e in experiment.experiment {
        datasets.entry(e.dataset).or_default().extend(e.algorithms);
    }

    dbg!(&experiment.dataset);
    dbg!(&datasets);

    for dataset in &experiment.dataset {
        match dataset {
            DatasetInfo::TwoSet(d) =>
                if let Some(algos) = datasets.get(&d.name) {
                    run_twoset_bench(d, algos);
                    datasets.remove(&d.name);
                }
            DatasetInfo::KSet(_) => todo!(),
        }
    }
    assert!(datasets.len() == 0);
}

fn run_twoset_bench(
    info: &TwoSetDatasetInfo,
    algos: &HashSet<String>) -> ResultDataset
{
    let dataset_dir = PathBuf::from("../datasets/2set").join(&info.name);

    let scale = match info.vary {
        Parameter::Density => AxisScale::Linear,
        Parameter::Selectivity => AxisScale::Linear,
        Parameter::Size => AxisScale::Logarithmic,
        Parameter::Skew => AxisScale::Logarithmic,
    };

    let xdirs = fs::read_dir(dataset_dir).unwrap();

    let mut result_dataset = ResultDataset {
        info: info.clone(),
        algos: HashMap::new(),
    };
    //let mut results: Vec<ResultRun> = Vec::new();

    for xdir in xdirs {
        // later: look at throughput?
        let xdir = xdir.unwrap();
        //assert!(xdir.unwrap().len() <= SAMPLE_SIZE as u64);

        let x: u32 = xdir
            .file_name().to_str().unwrap()
            .parse().unwrap();

        for name in algos {
            let xid = format_x(x, info.vary);
            //println!("\n\n\n");
            let algo = get_2set_algorithm(name).unwrap();

            let mut times: Vec<u64> = Vec::new();

            let datafiles = fs::read_dir(xdir.path()).unwrap();
            for datafile in datafiles {
                let datafile = datafile.unwrap();
                let reader = File::open(datafile.path()).unwrap();

                let duration = time_twoset(reader, algo);

                times.push(duration.as_nanos() as u64);
            }
        }
    }
    results
}

fn time_twoset(dataset: File, algo: Intersect2<[i32], VecWriter<i32>>) -> Duration {
    let pair: SetPair = ciborium::from_reader(dataset).unwrap();

    let capacity = pair.0.len().min(pair.1.len());
    // Warmup
    for _ in 0..10 {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        std::hint::black_box(algo(&pair.0, &pair.1, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
    let start = Instant::now();
    std::hint::black_box(algo(&pair.0, &pair.1, &mut writer));
    start.elapsed()
}


fn format_x(x: u32, vary: Parameter) -> String {
    match vary {
        Parameter::Density | Parameter::Selectivity =>
            format!("{:.2}", x as f64 / 1000.0),
        Parameter::Size => format_size(x),
        Parameter::Skew => format!("1:{}", 1 << (x - 1)),
    }
}

fn format_size(size: u32) -> String {
    match size {
        0..=9   => format!("{size}"),
        10..=19 => format!("{}KiB", 1 << (size - 10)),
        20..=29 => format!("{}MiB", 1 << (size - 20)),
        30..=39 => format!("{}GiB", 1 << (size - 30)),
        _ => size.to_string(),
    }
}

fn bench_2set_same_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_2set_same_size");
    group.sample_size(SAMPLE_SIZE);
    group.plot_config(
        PlotConfiguration::default().summary_scale(AxisScale::Logarithmic)
    );

    const K: usize = 1000;
    const SIZES: [usize; 8] = [
        K, 4 * K, 16 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K,
    ];
    bench_2set(group, SIZES.iter().map(|&size| (
        size,
        size,
        move || (
            benchmarks::uniform_sorted_set(0..size as i32 * 100, size),
            benchmarks::uniform_sorted_set(0..size as i32 * 100, size)
        )
    )))
}

fn bench_2set_skewed(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_2set_skewed");
    group.sample_size(SAMPLE_SIZE);

    const SMALL_SIZE: usize = 1024;
    const SKEWS: [usize; 9] = [
        1, 2, 4, 16, 64, 128, 256, 512, 1024
    ];
    bench_2set(group, SKEWS.iter().map(|&skew| (
        SMALL_SIZE,
        skew,
        move || (
            benchmarks::uniform_sorted_set(0..i32::MAX/2, SMALL_SIZE),
            benchmarks::uniform_sorted_set(0..i32::MAX/2, SMALL_SIZE * skew)
        )
    )))
}

fn bench_2set<Gs, G, P>(
    mut group: BenchmarkGroup<'_, WallTime>,
    generators: Gs)
where
    G: Fn() -> (Vec<i32>, Vec<i32>) + Copy,
    P: std::fmt::Display,
    Gs: IntoIterator<Item=(usize, P, G)>,
{
    let mut array_algs: Vec<&'static str> = TWOSET_ARRAY_SCALAR.into();
    array_algs.extend(TWOSET_ARRAY_SSE);
    array_algs.extend(TWOSET_ARRAY_AVX2);
    array_algs.extend(TWOSET_ARRAY_AVX512);

    for (min_length, id, generator) in generators {

        for &name in &array_algs {
            group.bench_with_input(BenchmarkId::new(name, &id), &min_length,
                |b, &size| run_array_2set(b, get_2set_algorithm(name).unwrap(), size, generator)
            );
        }

        //group.bench_with_input(BenchmarkId::new("roaring", &id), &min_length,
        //    |b, &_size| {
        //        b.iter_batched(
        //            || {
        //                let (left, right) = generator();
        //                (RoaringBitmap::from_sorted(&left), RoaringBitmap::from_sorted(&right))
        //            },
        //            |(mut set_a, set_b)| set_a &= set_b,
        //            criterion::BatchSize::LargeInput
        //        )
        //    });
        group.bench_with_input(BenchmarkId::new("fesia_sse (8N,8)", &id), &min_length,
            |b, &size| run_custom_2set::<Fesia8Sse<8>>(b, intersect::fesia::fesia, size, generator)
        );
        //group.bench_with_input(BenchmarkId::new("fesia_sse_shuffling", &id), &min_length,
        //    |b, &size| run_fesia_2set(b, intersect::fesia::fesia_sse_shuffling, size, generator)
        //);
        //group.bench_with_input(BenchmarkId::new("hash_set", &id), &min_length,
        //    |b, &size| run_custom_2set(b, intersect::hash_set_intersect, size, generator)
        //);
        //group.bench_with_input(BenchmarkId::new("btree_set", &id), &min_length,
        //    |b, &size| run_custom_2set(b, intersect::btree_set_intersect, size, generator)
        //);
    }
}

fn bench_kset_same_size(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_kset_same_size");
    group.sample_size(SAMPLE_SIZE);

    let mut array_algs: Vec<&'static str> = TWOSET_ARRAY_SCALAR.into();
    if cfg!(feature = "simd") {
        array_algs.extend(TWOSET_ARRAY_SSE);
    }

    const SIZE: usize = 1024 * 1000;

    for set_count in 3..=8 {
        let generator = ||
            Vec::from_iter((0..set_count).map(
                |_| benchmarks::uniform_sorted_set(0..i32::MAX, SIZE)
            ));

        for &name in &array_algs {
            let id = "svs_".to_string() + name;
            group.bench_with_input(BenchmarkId::new(id, set_count), &set_count,
                |b, &_count| run_svs_kset(b, get_2set_algorithm(name).unwrap(), SIZE, generator)
            );
        }
        for (name, intersect) in KSET_ARRAY_SCALAR {
            group.bench_with_input(BenchmarkId::new(name, set_count), &set_count,
                |b, &_count| run_array_kset(b, intersect, SIZE, generator)
            );
        }
        group.bench_with_input(BenchmarkId::new("roaring", set_count), &set_count,
            |b, &_count| {
                b.iter_batched(
                    || Vec::from_iter(
                        generator().iter().map(|s| RoaringBitmap::from_sorted(&s))
                    ),
                    |sets| sets.intersection(),
                    criterion::BatchSize::LargeInput,
                );
            }
        );
    }
}


fn run_array_2set(
    b: &mut Bencher,
    intersect: Intersect2<[i32], VecWriter<i32>>,
    output_len: usize,
    generator: impl Fn() -> (Vec<i32>, Vec<i32>))
{
    b.iter_batched(
        || {
            let (left, right) = generator();
            (left, right, VecWriter::with_capacity(output_len))
        },
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::LargeInput,
    );
}

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

fn run_custom_2set<S>(
    b: &mut Bencher,
    intersect: Intersect2<S, VecWriter<i32>>,
    output_len: usize,
    generator: impl Fn() -> (Vec<i32>, Vec<i32>))
where
    S: Set<i32>
{
    b.iter_batched(
        || {
            let (left, right) = generator();
            (S::from_sorted(&left), S::from_sorted(&right), VecWriter::with_capacity(output_len))
        },
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::LargeInput,
    );
}

fn run_array_kset(
    b: &mut Bencher,
    intersect: IntersectK<Vec<i32>, VecWriter<i32>>,
    set_size: usize,
    generator: impl Fn() -> Vec<Vec<i32>>)
{
    b.iter_batched(
        || (
            generator(),
            VecWriter::with_capacity(set_size)
        ),
        |(sets, mut writer)| intersect(sets.as_slice(), &mut writer),
        criterion::BatchSize::LargeInput,
    );
}

fn run_svs_kset(
    b: &mut Bencher,
    intersect: Intersect2<[i32], VecWriter<i32>>,
    set_size: usize,
    generator: impl Fn() -> Vec<Vec<i32>>)
{
    b.iter_batched(
        || (
            generator(),
            VecWriter::with_capacity(set_size),
            VecWriter::with_capacity(set_size),
        ),
        |(sets, mut left, mut right)| {
            let _ = intersect::svs_generic(
                sets.as_slice(), &mut left, &mut right, intersect);
        },
        criterion::BatchSize::LargeInput,
    );
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
