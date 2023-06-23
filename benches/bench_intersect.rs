mod benchlib;

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, BenchmarkGroup, measurement::WallTime, PlotConfiguration, AxisScale};
use roaring::{RoaringBitmap, MultiOps};
use setops::{
    intersect::{self, Intersect2, IntersectK},
    visitor::VecWriter,
    Set,
};

const SAMPLE_SIZE: usize = 16;

type TwoSetAlg = (&'static str, Intersect2<[i32], VecWriter<i32>>);
type KSetAlg = (&'static str, IntersectK<Vec<i32>, VecWriter<i32>>);

const TWOSET_ARRAY_SCALAR: [TwoSetAlg; 6] = [
    ("naive_merge", intersect::naive_merge),
    ("branchless_merge", intersect::branchless_merge),
    ("bmiss_scalar_3x", intersect::bmiss_scalar_3x),
    ("bmiss_scalar_4x", intersect::bmiss_scalar_4x),
    ("galloping", intersect::galloping),
    ("baezayates", intersect::baezayates),
];

#[cfg(feature = "simd")]
const TWOSET_ARRAY_VECTOR: [TwoSetAlg; 6] = [
    ("simd_shuffling_sse", intersect::simd_shuffling),
    ("simd_shuffling_avx2", intersect::simd_shuffling_avx2),
    ("bmiss_sse", intersect::bmiss),
    ("bmiss_sse_sttni", intersect::bmiss_sttni),
    ("simd_galloping", intersect::simd_galloping),
    ("simd_galloping_avx2", intersect::simd_galloping_8x),
    //("simd_galloping_avx512", intersect::simd_galloping_16x),
];

const KSET_ARRAY_SCALAR: [KSetAlg; 3] = [
    ("adaptive", intersect::adaptive),
    ("small_adaptive", intersect::small_adaptive),
    ("small_adaptive_sorted", intersect::small_adaptive_sorted),
];


criterion_group!(benches,
    bench_2set_same_size,
    bench_2set_skewed,
    bench_kset_same_size
);
criterion_main!(benches);

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
            benchlib::uniform_sorted_set(0..i32::MAX/2, size),
            benchlib::uniform_sorted_set(0..i32::MAX/2, size)
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
            benchlib::uniform_sorted_set(0..i32::MAX/2, SMALL_SIZE),
            benchlib::uniform_sorted_set(0..i32::MAX/2, SMALL_SIZE * skew)
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
    let mut array_algs: Vec<TwoSetAlg> = TWOSET_ARRAY_SCALAR.into();
    if cfg!(feature = "simd") {
        array_algs.extend(TWOSET_ARRAY_VECTOR);
    }

    for (min_length, id, generator) in generators {

        for &(name, intersect) in &array_algs {
            group.bench_with_input(BenchmarkId::new(name, &id), &min_length,
                |b, &size| run_array_2set(b, intersect, size, generator)
            );
        }

        group.bench_with_input(BenchmarkId::new("roaring", &id), &min_length,
            |b, &_size| {
                b.iter_batched(
                    || {
                        let (left, right) = generator();
                        (RoaringBitmap::from_sorted(&left), RoaringBitmap::from_sorted(&right))
                    },
                    |(mut set_a, set_b)| set_a &= set_b,
                    criterion::BatchSize::LargeInput
                )
            });
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

    let mut array_algs: Vec<TwoSetAlg> = TWOSET_ARRAY_SCALAR.into();
    if cfg!(feature = "simd") {
        array_algs.extend(TWOSET_ARRAY_VECTOR);
    }

    const SIZE: usize = 1024 * 1000;

    for set_count in 3..=8 {
        let generator = ||
            Vec::from_iter((0..set_count).map(
                |_| benchlib::uniform_sorted_set(0..i32::MAX, SIZE)
            ));

        for &(name, intersect) in &array_algs {
            let id = "svs_".to_string() + name;
            group.bench_with_input(BenchmarkId::new(id, set_count), &set_count,
                |b, &_count| run_svs_kset(b, intersect, SIZE, generator)
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
