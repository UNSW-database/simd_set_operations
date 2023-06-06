mod benchlib;

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use setops::{
    intersect::{self, Intersect2, IntersectK},
    visitor::VecWriter,
    Set,
};

fn array_2set(
    b: &mut Bencher,
    intersect: Intersect2<[u32], VecWriter<u32>>,
    size: usize)
{
    b.iter_batched(
        || {
            (
                benchlib::uniform_sorted_set(0..u32::MAX, size),
                benchlib::uniform_sorted_set(0..u32::MAX, size),
                VecWriter::<u32>::with_capacity(size),
            )
        },
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn custom_2set<S>(
    b: &mut Bencher,
    intersect: Intersect2<S, VecWriter<u32>>,
    size: usize)
where
    S: Set<u32>
{
    let gen_custom_set = || S::from_sorted(
        &benchlib::uniform_sorted_set(0..u32::MAX, size)
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
    intersect: IntersectK<Vec<u32>, VecWriter<u32>>,
    set_size: usize,
    set_count: usize)
{
    b.iter_batched(
        || (
            Vec::from_iter(std::iter::repeat(set_count).map(
                |_| benchlib::uniform_sorted_set(0..u32::MAX, set_size)
            )),
            VecWriter::with_capacity(set_size)
        ),
        |(sets, mut writer)| intersect(sets.as_slice(), &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn bench_2set(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect");
    group.sample_size(25);

    let sorted_array_algorithms: [(&str, Intersect2<[u32], VecWriter<u32>>); 3] = [
        ("naive_merge", intersect::naive_merge),
        ("branchless_merge", intersect::branchless_merge),
        ("baezayates", intersect::baezayates),
    ];

    const K: usize = 1000;
    const SIZES: [usize; 8] = [
        K,
        4 * K,
        16 * K,
        64 * K,
        128 * K,
        256 * K,
        512 * K,
        1024 * K,
    ];

    for size in SIZES {
        for (name, intersect) in sorted_array_algorithms {
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

criterion_group!(benches, bench_2set);
criterion_main!(benches);
