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

fn time<D>(
    warmup: Duration,
    prepare: impl Fn() -> D,
    run: impl Fn(&mut D)) -> (Duration, D)
{
    let warmup_start = Instant::now();
    while warmup_start.elapsed() < warmup {
        let mut data = prepare();
        hint::black_box(run(&mut data));
    }

    let mut data = prepare();

    let start = Instant::now();
    hint::black_box(run(&mut data));
    let elapsed = start.elapsed();

    (elapsed, data)
}

pub fn time_twoset(
    warmup: Duration,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> Result<Duration, String>
{
    let capacity = set_a.len().min(set_b.len());

    let prepare = || VecWriter::with_capacity(capacity);
    let run = |writer: &mut _| intersect(set_a, set_b, writer);

    let (elapsed, writer) = time(warmup, prepare, run);

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
    warmup: Duration,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2Bsr)
    -> Result<Duration, String>
{
    let bsr_a = BsrVec::from_sorted(slice_i32_to_u32(set_a));
    let bsr_b = BsrVec::from_sorted(slice_i32_to_u32(set_b));

    let capacity = bsr_a.len().min(bsr_b.len());

    let prepare = || BsrVec::with_capacities(capacity);
    let run = |writer: &mut _| intersect(bsr_a.bsr_ref(), bsr_b.bsr_ref(), writer);

    let (elapsed, writer) = time(warmup, prepare, run);

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
    warmup: Duration,
    sets: &[DatafileSet],
    intersect: IntersectK<DatafileSet, VecWriter<i32>>) -> Result<Duration, String>
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || VecWriter::with_capacity(capacity);
    let run = |writer: &mut _| intersect(sets, writer);

    let (elapsed, writer) = time(warmup, prepare, run);

    ensure_no_realloc(capacity, writer)?;
    Ok(elapsed)
}

pub fn time_svs(
    warmup: Duration,
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], VecWriter<i32>>) -> Result<Duration, String>
{
    // Note: max() required here
    let capacity = sets.iter().map(|s| s.len()).max()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || (
        VecWriter::with_capacity(capacity),
        VecWriter::with_capacity(capacity)
    );
    let run = |(left, right): &mut _| {
        intersect::svs_generic(sets, left, right, intersect);
    };

    let (elapsed, (left, right)) = time(warmup, prepare, run);

    ensure_no_realloc(capacity, left)?;
    ensure_no_realloc(capacity, right)?;
    Ok(elapsed)
}

pub fn time_croaring_2set(warmup: Duration, set_a: &[i32], set_b: &[i32])
    -> Duration
{
    use croaring::Bitmap;

    let prepare = || {
        let mut bitmap_a = Bitmap::of(slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(slice_i32_to_u32(&set_b));
        bitmap_a.run_optimize();
        bitmap_b.run_optimize();
        (bitmap_a, bitmap_b)
    };
    let run = |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
        bitmap_a.and_inplace(&bitmap_b);
    };

    let (elapsed, _) = time(warmup, prepare, run);
    elapsed
}

pub fn time_croaring_svs(warmup: Duration, sets: &[DatafileSet])
    -> Duration
{
    use croaring::Bitmap;
    assert!(sets.len() > 2);

    let prepare = || {
        let mut victim = Bitmap::of(slice_i32_to_u32(&sets[0]));
        victim.run_optimize();

        let rest: Vec<Bitmap> = (&sets[1..]).iter()
            .map(|s| {
                let mut bitmap = Bitmap::of(slice_i32_to_u32(&s));
                bitmap.run_optimize();
                bitmap
            }).collect();

        (victim, rest)
    };
    let run = |(victim, rest): &mut (Bitmap, Vec<Bitmap>)| {
        for bitmap in rest {
            victim.and_inplace(bitmap);
        }
    };

    let (elapsed, _) = time(warmup, prepare, run);
    elapsed
}

pub fn time_roaringrs_2set(warmup: Duration, set_a: &[i32], set_b: &[i32])
    -> Duration
{
    use roaring::RoaringBitmap;

    let prepare = || {
        let iter_a = set_a.iter().map(|&i| i as u32);
        let iter_b = set_b.iter().map(|&i| i as u32);

        let victim = RoaringBitmap::from_sorted_iter(iter_a).unwrap();
        let other = RoaringBitmap::from_sorted_iter(iter_b).unwrap();

        (victim, other)
    };
    let run = |(victim, other): &mut _| {
        *victim &= &*other
    };

    let (elapsed, _) = time(warmup, prepare, run);
    elapsed
}

pub fn time_roaringrs_svs(warmup: Duration, sets: &[DatafileSet])
    -> Duration
{
    use roaring::RoaringBitmap;
    assert!(sets.len() > 2);

    let prepare = || {
        let victim = RoaringBitmap::from_sorted_iter(
            sets[0].iter().map(|&i| i as u32)).unwrap();

        let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
            .map(|s|
                RoaringBitmap::from_sorted_iter(s.iter().map(|&i| i as u32)).unwrap()
            ).collect();

        (victim, rest)
    };
    let run = |(victim, rest): &mut _| {
        for bitmap in &*rest {
            *victim &= bitmap;
        }
    };

    let (elapsed, _) = time(warmup, prepare, run);
    elapsed
}

pub fn time_fesia<H, S, M, const LANES: usize>(
    warmup: Duration, 
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

    let prepare = || {
        VecWriter::with_capacity(capacity)
    };
    let run = |writer: &mut _| {
        intersect(&set_a, &set_b, writer);
    };

    let (elapsed, _) = time(warmup, prepare, run);
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
