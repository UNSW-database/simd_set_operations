#![feature(portable_simd)]
use std::{simd::{*, cmp::*}, ops::BitAnd, path::PathBuf};

use benchmark::{util, realdata};
use rand::{thread_rng, distributions::Uniform, Rng};
use setops::{
    intersect::{
        self, Intersect2, IntersectK,
        run_2set, run_2set_bsr, run_kset, run_svs,
        fesia::*,
    },
    visitor::VecWriter,
    bsr::{Intersect2Bsr, BsrVec},
    Set,
};

use clap::Parser;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[arg(default_value = "datasets/", long)]
    datasets: PathBuf,
    #[arg(default_value = "10000", long)]
    test_count: u32,
}

type TwoSetAlgorithm = (Intersect2<[i32], VecWriter<i32>>, &'static str);
type TwoSetBsrAlgorithm = (Intersect2Bsr, &'static str);

fn main() {
    let cli = Cli::parse();

    let real_datasets = [
        "webdocs",
        "twitter",
        "as-skitter",
        "census1881",
        "census-income",
    ];

    for real_dataset in real_datasets {
        if let Err(s) = test_on_dataset(&cli, real_dataset) {
            eprintln!("error: {}", s);
        };
    }
}

fn test_on_dataset(cli: &Cli, real_dataset: &str) -> Result<(), String> {
    let all_sets = realdata::load_sets(&cli.datasets, real_dataset)?;

    let min_len = all_sets.iter().map(|s| s.len()).min().unwrap();
    let max_len = all_sets.iter().map(|s| s.len()).max().unwrap();

    let total_len: usize = all_sets.iter().map(|s| s.len()).sum();
    let avg_len = total_len as f64 / all_sets.len() as f64;

    println!("{}: set lengths: avg {:.2}, min {}, max {}",
        real_dataset, avg_len, min_len, max_len);

    let mut twoset_array_algorithms: Vec<TwoSetAlgorithm> = TWOSET.into();
    twoset_array_algorithms.extend_from_slice(&TWOSET_SSE);
    twoset_array_algorithms.extend_from_slice(&TWOSET_AVX2);
    twoset_array_algorithms.extend_from_slice(&TWOSET_AVX512);

    let mut twoset_bsr_algorithms: Vec<TwoSetBsrAlgorithm> = TWOSET_BSR.into();
    twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_SSE);
    twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_AVX2);
    twoset_bsr_algorithms.extend_from_slice(&TWOSET_BSR_AVX512);

    println!("2-set:");
    run_twoset_tests(&all_sets, cli.test_count, &twoset_array_algorithms, test_twoset_array);
    run_twoset_tests(&all_sets, cli.test_count, &twoset_bsr_algorithms,   test_twoset_bsr);

    run_twoset_test(&all_sets, cli.test_count, "croaring",  |a, b| test_croaring_2set(a, b));
    // run_twoset_test(&all_sets, cli.test_count, "roaringrs", |a, b| test_roaringrs_2set(a, b));

    println!("k-set:");
    run_kset_tests(&all_sets, cli.test_count, &twoset_array_algorithms, |sets, f| test_svs(sets, f));
    run_kset_test(&all_sets, cli.test_count,
        "adaptive", |sets| test_kset(sets, intersect::adaptive));
    run_kset_test(&all_sets, cli.test_count,
        "baezayates_k", |sets| test_kset(sets, intersect::baezayates_k));
    run_kset_test(&all_sets, cli.test_count,
        "small_adaptive", |sets| test_kset(sets, intersect::small_adaptive));
    run_kset_test(&all_sets, cli.test_count,
        "small_adaptive_sorted", |sets| test_kset(sets, intersect::small_adaptive_sorted));

    run_kset_test(&all_sets, cli.test_count, "croaring_svs", |sets| test_croaring_svs(sets));
    // run_kset_test(&all_sets, cli.test_count, "roaringrs_svs", |sets| test_roaringrs_svs(sets));

    println!("fesia:");
    run_fesia_tests(&all_sets, cli.test_count);

    Ok(())
}

fn run_twoset_tests<F: Copy>(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32,
    algorithms: &[(F, &str)],
    test: fn(&[i32], &[i32], F) -> bool)
{
    for (intersect, name) in algorithms {
        run_twoset_test(
            all_sets, test_count, name,
            |a, b| test(a, b, *intersect)
        );
    }
}

fn run_twoset_test(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32,
    name: &str,
    test: impl Fn(&[i32], &[i32]) -> bool)
{
    let rng = &mut thread_rng();
    let index_distr = Uniform::from(0..all_sets.len());

    print!("{:24}", name);

    let mut set_indices: [usize; 2] = [
        rng.sample(index_distr),
        rng.sample(index_distr),
    ];

    set_indices.sort_by_key(|&s| all_sets[s].len());

    let sets: [&Vec<i32>; 2] = [
        &all_sets[set_indices[0]],
        &all_sets[set_indices[1]],
    ];

    for _ in 0..test_count {
        if !test(sets[0], sets[1]) {
            println!("FAIL");

            println!("left: set #{} of len {}:\n{:?}\n",
                set_indices[0], sets[0].len(), sets[0]);
            println!("right: set #{} of len {}:\n{:?}\n",
                set_indices[1], sets[1].len(), sets[1]);

            return;
        }
    }
    println!("pass");
}

fn run_kset_tests<F: Copy>(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32,
    algorithms: &[(F, &str)],
    test: fn(&[&Vec<i32>], F) -> bool)
{
    for (intersect, name) in algorithms {
        run_kset_test(
            all_sets, test_count, name,
            |a| test(a, *intersect)
        );
    }
}

fn run_kset_test(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32,
    name: &str,
    test: impl Fn(&[&Vec<i32>]) -> bool)
{
    let rng = &mut thread_rng();
    let index_distr = Uniform::from(0..all_sets.len());

    print!("{:24}", name);

    let set_count = rng.sample(Uniform::from(2..=8));

    let mut set_indices: Vec<usize> = (0..set_count).into_iter()
        .map(|_| rng.sample(index_distr))
        .collect();

    set_indices.sort_by_key(|&s| all_sets[s].len());

    let sets: Vec<&Vec<i32>> = set_indices.iter()
        .map(|&i| &all_sets[i])
        .collect();

    for _ in 0..test_count {
        if !test(&sets) {
            println!("FAIL");

            let sets_iter = set_indices.iter().zip(sets.iter()).enumerate();
            for (i, (set_index, set)) in sets_iter {
                println!("[{}] set #{} of len {}:\n{:?}\n",
                    i, set_index, set.len(), set);
            }
            return;
        }
    }
    println!("pass");
}

fn run_fesia_tests(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32)
{
    let hash_scale = 0.01;

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    run_fesia_test::<MixHash, i8, 16>(all_sets, test_count, "fesia8_sse", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    run_fesia_test::<MixHash, i16, 8>(all_sets, test_count, "fesia16_sse", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    run_fesia_test::<MixHash, i32, 4>(all_sets, test_count, "fesia32_sse", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    run_fesia_test::<MixHash, i8, 32>(all_sets, test_count, "fesia8_avx2", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    run_fesia_test::<MixHash, i16, 16>(all_sets, test_count, "fesia16_avx2", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    run_fesia_test::<MixHash, i32, 8>(all_sets, test_count, "fesia32_avx2", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    run_fesia_test::<MixHash, i8,  64>(all_sets, test_count, "fesia8_avx512", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    run_fesia_test::<MixHash, i16, 32>(all_sets, test_count, "fesia16_avx512", hash_scale);
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    run_fesia_test::<MixHash, i32, 16>(all_sets, test_count, "fesia32_avx512", hash_scale);
}

pub fn run_fesia_test<H, S, const LANES: usize>(
    all_sets: &Vec<Vec<i32>>,
    test_count: u32,
    name: &str,
    hash_scale: HashScale)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
{
    run_twoset_test(all_sets, test_count, name,
        |a, b| test_fesia::<MixHash, i32, u16, 16>(a, b, hash_scale));
}

fn test_twoset_array(
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> bool
{
    let actual = run_2set(set_a, set_b, intersect);
    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    actual == expected
}

fn test_twoset_bsr(
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2Bsr) -> bool
{
    let bsr_a = BsrVec::from_sorted(util::slice_i32_to_u32(set_a));
    let bsr_b = BsrVec::from_sorted(util::slice_i32_to_u32(set_b));

    let actual   = run_2set_bsr(bsr_a.bsr_ref(), bsr_b.bsr_ref(), intersect);
    let expected = run_2set_bsr(bsr_a.bsr_ref(), bsr_b.bsr_ref(),
        intersect::branchless_merge_bsr);

    actual == expected
}

fn test_kset<S: AsRef<[i32]>>(
    sets: &[S],
    intersect: IntersectK<S, VecWriter<i32>>) -> bool
{
    let actual = run_kset(sets, intersect);
    let expected = run_svs(sets, intersect::naive_merge);

    actual == expected
}

fn test_svs<S: AsRef<[i32]>>(
    sets: &[S],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> bool
{
    let actual = run_svs(sets, intersect);
    let expected = run_svs(sets, intersect::naive_merge);

    actual == expected
}

fn test_croaring_2set(set_a: &[i32], set_b: &[i32]) -> bool {
    use croaring::Bitmap;

    let mut victim = Bitmap::of(util::slice_i32_to_u32(&set_a));
    let mut other = Bitmap::of(util::slice_i32_to_u32(&set_b));
    victim.run_optimize();
    other.run_optimize();

    victim.and_inplace(&other);

    let actual: Vec<u32> = victim.to_vec();
    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    util::slice_u32_to_i32(&actual) == expected
}

fn test_croaring_svs<S: AsRef<[i32]>>(sets: &[S]) -> bool {
    use croaring::Bitmap;
    assert!(sets.len() >= 2);

    let mut victim = Bitmap::of(util::slice_i32_to_u32(sets[0].as_ref()));
    victim.run_optimize();

    let rest: Vec<Bitmap> = (&sets[1..]).iter()
        .map(|s| {
            let mut bitmap = Bitmap::of(util::slice_i32_to_u32(s.as_ref()));
            bitmap.run_optimize();
            bitmap
        }).collect();
        
    for bitmap in rest {
        victim.and_inplace(&bitmap);
    }

    let actual: Vec<u32> = victim.to_vec();
    let expected = run_svs(sets, intersect::naive_merge);

    util::slice_u32_to_i32(&actual) == expected
}

// fn test_roaringrs_2set(set_a: &[i32], set_b: &[i32]) -> bool {
//     use roaring::RoaringBitmap;

//     let iter_a = set_a.iter().map(|&i| i as u32);
//     let iter_b = set_b.iter().map(|&i| i as u32);

//     let mut victim = RoaringBitmap::from_sorted_iter(iter_a).unwrap();
//     let other = RoaringBitmap::from_sorted_iter(iter_b).unwrap();

//         victim &= &other;

//     let actual: Vec<i32> = victim.into_iter().map(|i| i as i32).collect();
//     let expected = run_2set(set_a, set_b, intersect::naive_merge);

//     actual == expected
// }

// pub fn test_roaringrs_svs<S: AsRef<[i32]>>(sets: &[S]) -> bool {
//     use roaring::RoaringBitmap;
//     assert!(sets.len() > 2);

//     let mut victim = RoaringBitmap::from_sorted_iter(
//         sets[0].as_ref().iter().map(|&i| i as u32)).unwrap();

//     let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
//         .map(|s|
//             RoaringBitmap::from_sorted_iter(s.as_ref().iter().map(|&i| i as u32)).unwrap()
//         ).collect();

//     for bitmap in &rest {
//         victim &= bitmap;
//     }

//     let actual: Vec<i32> = victim.into_iter().map(|i| i as i32).collect();
//     let expected = run_svs(sets, intersect::naive_merge);

//     actual == expected  
// }

pub fn test_fesia<H, S, M, const LANES: usize>(
    set_a: &[i32],
    set_b: &[i32],
    hash_scale: HashScale) -> bool
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
{
    let fesia_a = Fesia::<H, S, LANES>::from_sorted(set_a, hash_scale);
    let fesia_b = Fesia::<H, S, LANES>::from_sorted(set_b, hash_scale);

    let mut writer = VecWriter::new();

    fesia_a.intersect::<VecWriter<i32>, SegmentIntersectSse>(&fesia_b, &mut writer);

    let mut actual: Vec<i32> = writer.into();
    actual.sort();

    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    actual == expected
}

const TWOSET: [TwoSetAlgorithm; 6] = [
    (intersect::naive_merge, "naive_merge"),
    (intersect::branchless_merge, "branchless_merge"),
    (intersect::galloping, "galloping"),
    (intersect::bmiss_scalar_3x, "bmiss_scalar_3x"),
    (intersect::bmiss_scalar_4x, "bmiss_scalar_4x"),
    (intersect::baezayates, "baezayates"),
];

const TWOSET_SSE: [TwoSetAlgorithm; 6] = [
    (intersect::shuffling_sse, "shuffling_sse"),
    (intersect::broadcast_sse, "broadcast_sse"),
    (intersect::galloping_sse, "galloping_sse"),
    (intersect::bmiss, "bmiss"),
    (intersect::bmiss_sttni, "bmiss_sttni"),
    (intersect::qfilter, "qfilter"),
];

const TWOSET_AVX2: [TwoSetAlgorithm; 3] = [
    (intersect::shuffling_avx2, "shuffling_avx2"),
    (intersect::broadcast_avx2, "broadcast_avx2"),
    (intersect::galloping_avx2, "galloping_avx2"),
];

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
const TWOSET_AVX512: [TwoSetAlgorithm; 5] = [
    (intersect::shuffling_avx512, "shuffling_avx512"),
    (intersect::broadcast_avx512, "broadcast_avx512"),
    (intersect::galloping_avx512, "galloping_avx512"),
    (intersect::vp2intersect_emulation, "vp2intersect_emulation"),
    (intersect::conflict_intersect, "conflict_intersect"),
];
#[cfg(not(all(feature = "simd", target_feature = "avx512f")))]
const TWOSET_AVX512: [(Intersect2<[i32], VecWriter<i32>>, &'static str); 0] = [];

const TWOSET_BSR: [TwoSetBsrAlgorithm; 1] = [
    (intersect::branchless_merge_bsr, "branchless_merge_bsr"),
];

const TWOSET_BSR_SSE: [TwoSetBsrAlgorithm; 4] = [
    (intersect::shuffling_sse_bsr, "shuffling_sse_bsr"),
    (intersect::broadcast_sse_bsr, "broadcast_sse_bsr"),
    (intersect::galloping_sse_bsr, "galloping_sse_bsr"),
    (intersect::qfilter_bsr, "qfilter_bsr"),
];

const TWOSET_BSR_AVX2: [TwoSetBsrAlgorithm; 3] = [
    (intersect::shuffling_avx2_bsr, "shuffling_avx2_bsr"),
    (intersect::broadcast_avx2_bsr, "broadcast_avx2_bsr"),
    (intersect::galloping_avx2_bsr, "galloping_avx2_bsr"),
];

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
const TWOSET_BSR_AVX512: [TwoSetBsrAlgorithm; 3] = [
    (intersect::shuffling_avx512_bsr, "shuffling_avx512_bsr"),
    (intersect::broadcast_avx512_bsr, "broadcast_avx512_bsr"),
    (intersect::galloping_avx512_bsr, "galloping_avx512_bsr"),
];
#[cfg(not(all(feature = "simd", target_feature = "avx512f")))]
const TWOSET_BSR_AVX512: [TwoSetBsrAlgorithm; 0] = [];
