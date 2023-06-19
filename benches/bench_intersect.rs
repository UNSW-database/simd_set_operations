mod benchlib;

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use setops::{
    intersect::{self, Intersect2, IntersectK},
    visitor::VecWriter,
    Set,
};

fn array_2set(
    b: &mut Bencher,
    intersect: Intersect2<[i32], VecWriter<i32>>,
    size: usize)
{
    b.iter_batched(
        || {
            (
                benchlib::uniform_sorted_set(0..i32::MAX, size),
                benchlib::uniform_sorted_set(0..i32::MAX, size),
                VecWriter::<i32>::with_capacity(size),
            )
        },
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn custom_2set<S>(
    b: &mut Bencher,
    intersect: Intersect2<S, VecWriter<i32>>,
    size: usize)
where
    S: Set<i32>
{
    let gen_custom_set = || S::from_sorted(
        &benchlib::uniform_sorted_set(0..i32::MAX, size)
    );

    b.iter_batched(
        || (
            gen_custom_set(),
            gen_custom_set(),
            VecWriter::with_capacity(size),
        ),
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn array_kset(
    b: &mut Bencher,
    intersect: IntersectK<Vec<i32>, VecWriter<i32>>,
    set_size: usize,
    set_count: usize)
{
    b.iter_batched(
        || (
            Vec::from_iter((0..set_count).map(
                |_| benchlib::uniform_sorted_set(0..i32::MAX, set_size)
            )),
            VecWriter::with_capacity(set_size)
        ),
        |(sets, mut writer)| intersect(sets.as_slice(), &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn svs_kset(
    b: &mut Bencher,
    intersect: Intersect2<[i32], VecWriter<i32>>,
    set_size: usize,
    set_count: usize)
{
    b.iter_batched(
        || (
            Vec::from_iter((0..set_count).map(
                |_| benchlib::uniform_sorted_set(0..i32::MAX, set_size)
            )),
            VecWriter::with_capacity(set_size),
            VecWriter::with_capacity(set_size),
        ),
        |(sets, mut left, mut right)| {
            let _ = intersect::svs_generic(
                sets.as_slice(), &mut left, &mut right, intersect);
        },
        criterion::BatchSize::SmallInput,
    );
}

fn bench_2set(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_2set");
    group.sample_size(25);

    type Alg = (&'static str, Intersect2<[i32], VecWriter<i32>>);

    let scalar_array_algorithms: [Alg; 4] = [
        ("naive_merge", intersect::naive_merge),
        ("branchless_merge", intersect::branchless_merge),
        ("galloping", intersect::galloping),
        ("baezayates", intersect::baezayates),
    ];
    let mut all_array_algorithms: Vec<Alg> = scalar_array_algorithms.into();

    if cfg!(feature = "simd") {
        all_array_algorithms.push(("simd_shuffling", intersect::simd_shuffling));
        all_array_algorithms.push(("simd_galloping", intersect::simd_shuffling));
    }

    const K: usize = 1000;
    const SIZES: [usize; 8] = [
        K, 4 * K, 16 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K,
    ];

    for size in SIZES {
        for &(name, intersect) in &all_array_algorithms {
            group.bench_with_input(BenchmarkId::new(name, size), &size,
                |b, &size| array_2set(b, intersect, size)
            );
        }

        group.bench_with_input(BenchmarkId::new("hash_set", size), &size,
            |b, &size| custom_2set(b, intersect::hash_set_intersect, size)
        );
        group.bench_with_input(BenchmarkId::new("btree_set", size), &size,
            |b, &size| custom_2set(b, intersect::btree_set_intersect, size)
        );
    }
}

fn bench_kset(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect_kset");
    group.sample_size(25);

    let kset_algorithms: [(&str, IntersectK<Vec<i32>, VecWriter<i32>>); 3] = [
        ("adaptive", intersect::adaptive),
        ("small_adaptive", intersect::small_adaptive),
        ("small_adaptive_sorted", intersect::small_adaptive_sorted),
    ];
    let pairwise_algorithms: [(&str, Intersect2<[i32], VecWriter<i32>>); 4] = [
        ("naive_merge", intersect::naive_merge),
        ("branchless_merge", intersect::branchless_merge),
        ("baezayates", intersect::baezayates),
        ("galloping", intersect::galloping),
    ];

    const K: usize = 1000;
    const SIZES: [usize; 8] = [
        K, 4 * K, 16 * K, 64 * K, 128 * K, 256 * K, 512 * K, 1024 * K,
    ];

    for size in SIZES {
        for (name, intersect) in pairwise_algorithms {
            let id = "svs_".to_string() + name;
            group.bench_with_input(BenchmarkId::new(id, size), &size,
                |b, &size| svs_kset(b, intersect, size, 3)
            );
        }
        for (name, intersect) in kset_algorithms {
            group.bench_with_input(BenchmarkId::new(name, size), &size,
                |b, &size| array_kset(b, intersect, size, 3)
            );
        }
    }
}


criterion_group!(benches, bench_2set, bench_kset);
criterion_main!(benches);
