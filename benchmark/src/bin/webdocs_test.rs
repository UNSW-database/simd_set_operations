#![feature(portable_simd)]
use std::{simd::*, ops::BitAnd, path::PathBuf};

use benchmark::{util, datafile::DatafileSet, webdocs};
use rand::{thread_rng, seq::SliceRandom};
use setops::{
    intersect::{
        self, Intersect2, IntersectK,
        run_2set, run_2set_bsr, run_kset, run_svs,
        fesia::*
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
    #[arg(default_value = "100", long)]
    test_count: u32,
}

const TWOSET: [(Intersect2<[i32], VecWriter<i32>>, &'static str); 5] = [
    (intersect::naive_merge, "naive_merge"),
    (intersect::branchless_merge, "branchless_merge"),
    (intersect::bmiss_scalar_3x, "bmiss_scalar_3x"),
    (intersect::bmiss_scalar_4x, "bmiss_scalar_3x"),
    (intersect::baezayates, "baezayates"),
];

fn main() {
    let cli = Cli::parse();

    if let Err(s) = test_on_webdocs(cli) {
        eprintln!("error: {}", s);
    };
}

fn test_on_webdocs(cli: Cli) -> Result<(), String> {
    let sets = webdocs::load_sets(&cli.datasets)?;
    run_twoset_tests(&sets, cli.test_count);

    Ok(())
}

fn run_twoset_tests(all_sets: &Vec<Vec<i32>>, test_count: u32) {
    let rng = &mut thread_rng();

    for (intersect, name) in TWOSET {
        print!("{}: ", name);

        let mut sets: [&DatafileSet; 2] = [
            all_sets.choose(rng).unwrap(),
            all_sets.choose(rng).unwrap(),
        ];

        sets.sort_by_key(|&s| s.len());

        for _ in 0..test_count {
            if !test_twoset(sets[0], sets[1], intersect) {
                println!("FAIL");
                println!("set_a: {:?}", sets[0]);
                println!("set_b: {:?}", sets[1]);
                return;
            }
        }
        println!("pass");
    }
}

fn test_twoset(
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> bool
{
    let actual = run_2set(set_a, set_b, intersect);
    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    actual == expected
}

fn test_bsr(
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

fn test_kset(
    sets: &[DatafileSet],
    intersect: IntersectK<DatafileSet, VecWriter<i32>>) -> bool
{
    let actual = run_kset(sets, intersect);
    let expected = run_svs(sets, intersect::naive_merge);

    actual == expected
}

fn test_svs(
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> bool
{
    let actual = run_svs(sets, intersect);
    let expected = run_svs(sets, intersect::naive_merge);

    actual == expected
}

fn time_croaring_2set(set_a: &[i32], set_b: &[i32]) -> bool {
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

fn test_croaring_svs(sets: &[DatafileSet]) -> bool {
    use croaring::Bitmap;
    assert!(sets.len() > 2);

    let mut victim = Bitmap::of(util::slice_i32_to_u32(&sets[0]));
    victim.run_optimize();

    let rest: Vec<Bitmap> = (&sets[1..]).iter()
        .map(|s| {
            let mut bitmap = Bitmap::of(util::slice_i32_to_u32(&s));
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

fn test_roaringrs_2set(set_a: &[i32], set_b: &[i32]) -> bool {
    use roaring::RoaringBitmap;

    let iter_a = set_a.iter().map(|&i| i as u32);
    let iter_b = set_b.iter().map(|&i| i as u32);

    let mut victim = RoaringBitmap::from_sorted_iter(iter_a).unwrap();
    let other = RoaringBitmap::from_sorted_iter(iter_b).unwrap();

    victim &= &other;

    let actual: Vec<i32> = victim.into_iter().map(|i| i as i32).collect();
    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    actual == expected
}

pub fn time_roaringrs_svs(sets: &[DatafileSet]) -> bool {
    use roaring::RoaringBitmap;
    assert!(sets.len() > 2);

    let mut victim = RoaringBitmap::from_sorted_iter(
        sets[0].iter().map(|&i| i as u32)).unwrap();

    let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
        .map(|s|
            RoaringBitmap::from_sorted_iter(s.iter().map(|&i| i as u32)).unwrap()
        ).collect();

    for bitmap in &rest {
        victim &= bitmap;
    }

    let actual: Vec<i32> = victim.into_iter().map(|i| i as i32).collect();
    let expected = run_svs(sets, intersect::naive_merge);

    actual == expected  
}

pub fn test_fesia<H, S, M, const LANES: usize>(
    set_a: &[i32],
    set_b: &[i32],
    hash_scale: HashScale) -> bool
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    let fesia_a = Fesia::<H, S, M, LANES>::from_sorted(set_a, hash_scale);
    let fesia_b = Fesia::<H, S, M, LANES>::from_sorted(set_b, hash_scale);

    let mut writer = VecWriter::new();

    fesia_intersect(&fesia_a, &fesia_b, &mut writer);

    let mut actual: Vec<i32> = writer.into();
    actual.sort();

    let expected = run_2set(set_a, set_b, intersect::naive_merge);

    actual == expected
}
