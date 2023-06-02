mod benchlib;

use criterion::{criterion_group, criterion_main, Bencher, BenchmarkId, Criterion};
use setops::{
    intersect::{self, Intersect2},
    visitor::VecWriter,
    CustomSet,
};

fn intersect_benchmark<In, VT>(
    b: &mut Bencher,
    intersect: Intersect2<In, VecWriter<VT>>,
    size: usize,
) where
    In: CustomSet<u32>,
{
    b.iter_batched(
        || {
            let set_a = benchlib::uniform_sorted_set(0..u32::MAX, size);
            let set_b = benchlib::uniform_sorted_set(0..u32::MAX, size);
            (
                In::from_sorted(&set_a),
                In::from_sorted(&set_b),
                VecWriter::with_capacity(size),
            )
        },
        |(set_a, set_b, mut writer)| intersect(&set_a, &set_b, &mut writer),
        criterion::BatchSize::SmallInput,
    );
}

fn bench_intersection(c: &mut Criterion) {
    let mut group = c.benchmark_group("intersect");
    group.sample_size(25);

    let sorted_array_algorithms: [(&str, Intersect2<[u32], VecWriter<u32>>); 2] = [
        ("naive_merge", intersect::naive_merge),
        ("branchless_merge", intersect::branchless_merge),
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
            group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &size| {
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
            });
        }

        group.bench_with_input(BenchmarkId::new("hash_set", size), &size, |b, &size| {
            intersect_benchmark(b, intersect::hash_set_intersect, size);
        });
        group.bench_with_input(BenchmarkId::new("btree_set", size), &size, |b, &size| {
            intersect_benchmark(b, intersect::btree_set_intersect, size);
        });
    }
}

criterion_group!(benches, bench_intersection);
criterion_main!(benches);
