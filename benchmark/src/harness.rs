use std::{
    time::{Duration, Instant},
    hint, simd::*, ops::BitAnd,
};
use setops::{
    intersect::{Intersect2, IntersectK, fesia::*, self},
    visitor::VecWriter,
    bsr::{BsrVec, Intersect2Bsr},
    Set,
};
use crate::datafile::DatafileSet;

pub fn time_twoset(
    warmup_rounds: u32,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> Result<Duration, String>
{
    let capacity = set_a.len().min(set_b.len());
    // Warmup
    for _ in 0..warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        hint::black_box(intersect(set_a, set_b, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    hint::black_box(intersect(set_a, set_b, &mut writer));
    let elapsed = start.elapsed();

    ensure_no_realloc(capacity, writer)?;
    Ok(elapsed)
}

fn ensure_no_realloc(target: usize, writer: VecWriter<i32>) -> Result<(), String> {
    let vec: Vec<i32> = writer.into();
    if vec.len() > target {
        Err("unexpected VecWriter resize".to_string())
    }
    else {
        Ok(())
    }
}

pub fn time_bsr(
    warmup_rounds: u32,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2Bsr)
    -> Result<Duration, String>
{
    let bsr_a = BsrVec::from_sorted(slice_i32_to_u32(set_a));
    let bsr_b = BsrVec::from_sorted(slice_i32_to_u32(set_b));

    let capacity = bsr_a.len().min(bsr_b.len());

    // Warmup
    for _ in 0..warmup_rounds {
        let mut writer = BsrVec::with_capacities(capacity);
        hint::black_box(intersect(bsr_a.bsr_ref(), bsr_b.bsr_ref(), &mut writer));
    }

    let mut writer = BsrVec::with_capacities(capacity);

    let start = Instant::now();
    hint::black_box(intersect(bsr_a.bsr_ref(), bsr_b.bsr_ref(), &mut writer));
    let elapsed = start.elapsed();

    ensure_no_realloc_bsr(capacity, writer)?;
    Ok(elapsed)
}

pub fn ensure_no_realloc_bsr(target: usize, writer: BsrVec) -> Result<(), String> {
    if writer.len() > target {
        Err("unexpected VecWriter resize".to_string())
    }
    else {
        Ok(())
    }
}

pub fn time_kset(
    warmup_rounds: u32,
    sets: &[DatafileSet],
    intersect: IntersectK<DatafileSet, VecWriter<i32>>) -> Result<Duration, String>
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    // Warmup
    for _ in 0..warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        hint::black_box(intersect(sets, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    hint::black_box(intersect(sets, &mut writer));
    let elapsed = start.elapsed();

    ensure_no_realloc(capacity, writer)?;
    Ok(elapsed)
}

pub fn time_svs(
    warmup_rounds: u32,
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> Result<Duration, String>
{
    // Note: max() required here
    let capacity = sets.iter().map(|s| s.len()).max()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    // Warmup
    for _ in 0..warmup_rounds {
        let mut left: VecWriter<i32> = VecWriter::with_capacity(capacity);
        let mut right: VecWriter<i32> = VecWriter::with_capacity(capacity);
        hint::black_box(intersect::svs_generic(sets, &mut left, &mut right, intersect));
    }

    let mut left: VecWriter<i32> = VecWriter::with_capacity(capacity);
    let mut right: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    hint::black_box(intersect::svs_generic(sets, &mut left, &mut right, intersect));
    let elapsed = start.elapsed();

    ensure_no_realloc(capacity, left)?;
    ensure_no_realloc(capacity, right)?;
    Ok(elapsed)
}

pub fn time_croaring_2set(warmup_rounds: u32, set_a: &[i32], set_b: &[i32])
    -> Duration
{
    use croaring::Bitmap;
    // Warmup
    for _ in 0..warmup_rounds {
        let mut bitmap_a = Bitmap::of(slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(slice_i32_to_u32(&set_b));
        bitmap_a.run_optimize();
        bitmap_b.run_optimize();

        hint::black_box(bitmap_a.and_inplace(&bitmap_b));
    }

    let mut bitmap_a = Bitmap::of(slice_i32_to_u32(&set_a));
    let mut bitmap_b = Bitmap::of(slice_i32_to_u32(&set_b));
    bitmap_a.run_optimize();
    bitmap_b.run_optimize();

    let start = Instant::now();
    hint::black_box(bitmap_a.and_inplace(&bitmap_b));
    start.elapsed()
}

pub fn time_croaring_svs(warmup_rounds: u32, sets: &[DatafileSet])
    -> Duration
{
    use croaring::Bitmap;
    assert!(sets.len() > 2);

    // Warmup
    for _ in 0..warmup_rounds {
        let mut victim = Bitmap::of(slice_i32_to_u32(&sets[0]));
        victim.run_optimize();

        let rest: Vec<Bitmap> = (&sets[1..]).iter()
            .map(|s| {
                let mut bitmap = Bitmap::of(slice_i32_to_u32(&s));
                bitmap.run_optimize();
                bitmap
            }).collect();

        hint::black_box(croaring_intersect_svs(&mut victim, &rest));
    }

    let mut victim = Bitmap::of(slice_i32_to_u32(&sets[0]));
    let rest: Vec<Bitmap> = (&sets[1..]).iter()
        .map(|s| {
            let mut bitmap = Bitmap::of(slice_i32_to_u32(&s));
            bitmap.run_optimize();
            bitmap
        }).collect();

    let start = Instant::now();
    hint::black_box(croaring_intersect_svs(&mut victim, &rest));
    start.elapsed()
}

fn croaring_intersect_svs(victim: &mut croaring::Bitmap, rest: &[croaring::Bitmap]) {
    for bitmap in rest {
        victim.and_inplace(bitmap);
    }
}

pub fn time_roaringrs_2set(warmup_rounds: u32, set_a: &[i32], set_b: &[i32])
    -> Duration
{
    use roaring::RoaringBitmap;
    // Warmup
    for _ in 0..warmup_rounds {
        let mut victim = RoaringBitmap::from_sorted_iter(set_a.iter().map(|&i| i as u32)).unwrap();
        let other = RoaringBitmap::from_sorted_iter(set_b.iter().map(|&i| i as u32)).unwrap();
        hint::black_box(victim &= other);
    }

    let mut victim = RoaringBitmap::from_sorted_iter(set_a.iter().map(|&i| i as u32)).unwrap();
    let other = RoaringBitmap::from_sorted_iter(set_b.iter().map(|&i| i as u32)).unwrap();

    let start = Instant::now();
    hint::black_box(victim &= other);
    start.elapsed()
}

pub fn time_roaringrs_svs(warmup_rounds: u32, sets: &[DatafileSet])
    -> Duration
{
    use roaring::RoaringBitmap;
    assert!(sets.len() > 2);

    // Warmup
    for _ in 0..warmup_rounds {
        let mut victim = RoaringBitmap::from_sorted_iter(
            sets[0].iter().map(|&i| i as u32)).unwrap();

        let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
            .map(|s|
                RoaringBitmap::from_sorted_iter(s.iter().map(|&i| i as u32)).unwrap()
            ).collect();

        hint::black_box(roaringrs_intersect_svs(&mut victim, &rest));
    }

    let mut victim = RoaringBitmap::from_sorted_iter(
        sets[0].iter().map(|&i| i as u32)).unwrap();

    let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
        .map(|s|
            RoaringBitmap::from_sorted_iter(s.iter().map(|&i| i as u32)).unwrap()
        ).collect();

    let start = Instant::now();
    hint::black_box(roaringrs_intersect_svs(&mut victim, &rest));
    start.elapsed()
}

fn roaringrs_intersect_svs(victim: &mut roaring::RoaringBitmap, rest: &[roaring::RoaringBitmap]) {
    for bitmap in rest {
        *victim &= bitmap;
    }
}

pub fn time_fesia<H, S, M, const LANES: usize>(
    warmup_rounds: u32,
    sets: &[DatafileSet],
    hash_scale: HashScale,
    intersect: fn(&Fesia<H, S, M, LANES>, &Fesia<H, S, M, LANES>, &mut VecWriter<i32>))
    -> Result<Duration, String>
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    let (set_a, set_b) = ensure_twoset(sets)?;

    let capacity = set_a.len().min(set_b.len());

    let set_a = Fesia::<H, S, M, LANES>::from_sorted(set_a, hash_scale);
    let set_b = Fesia::<H, S, M, LANES>::from_sorted(set_b, hash_scale);

    // Warmup
    for _ in 0..warmup_rounds {
        let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);
        hint::black_box(intersect(&set_a, &set_b, &mut writer));
    }

    let mut writer: VecWriter<i32> = VecWriter::with_capacity(capacity);

    let start = Instant::now();
    hint::black_box(intersect(&set_a, &set_b, &mut writer));
    let elapsed = start.elapsed();

    Ok(elapsed)
}

fn ensure_twoset(sets: &[DatafileSet]) -> Result<(&DatafileSet, &DatafileSet), String> {
    if sets.len() != 2 {
        return Err(format!("expected 2 sets, got {}", sets.len()));
    }
    return Ok((&sets[0], &sets[1]))
}

fn slice_i32_to_u32(slice_i32: &[i32]) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            slice_i32.as_ptr() as *const u32, slice_i32.len()
        )
    }
}
