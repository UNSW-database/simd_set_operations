use std::{
    time::{Duration, Instant},
    hint, simd::*, ops::BitAnd,
};
use setops::{
    intersect::{Intersect2, IntersectK, fesia::*, self},
    visitor::{
        Visitor, SimdVisitor4, SimdVisitor8, SimdVisitor16,
        UnsafeWriter, UnsafeBsrWriter, Counter
    },
    bsr::{BsrVec, BsrRef},
    Set,
};
use crate::{datafile::DatafileSet, util};

pub type TimeResult = Result<RunTime, String>;

pub type UnsafeIntersectBsr = for<'a> fn(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut UnsafeBsrWriter);

pub struct Harness {
    warmup: Duration,
    method: TimeMethod,
}

pub enum TimeMethod {
    Seconds,
    Cycles,
}

pub enum RunTime {
    Duration(Duration),
    Cycles(u64),
}

impl Harness {
    pub fn new(warmup: Duration, method: TimeMethod) -> Self {
        Self { warmup, method }
    }

    pub fn time<D>(
        &self,
        prepare: impl Fn() -> D,
        run: impl Fn(&mut D)) -> (RunTime, D)
    {
        let warmup_start = Instant::now();
        while warmup_start.elapsed() < self.warmup {
            let mut data = prepare();
            hint::black_box(run(&mut data));
        }

        match self.method {
            TimeMethod::Seconds => self.time_seconds(prepare, run),
            TimeMethod::Cycles  => self.time_cycles(prepare, run),
        }
    }

    fn time_seconds<D>(
        &self,
        prepare: impl Fn() -> D,
        run: impl Fn(&mut D)) -> (RunTime, D)
    {
        let mut data = prepare();

        let start = Instant::now();
        hint::black_box(run(&mut data));
        let elapsed = start.elapsed();

        (RunTime::Duration(elapsed), data)
    }

    fn time_cycles<D>(
        &self,
        prepare: impl Fn() -> D,
        run: impl Fn(&mut D)) -> (RunTime, D)
    {
        use std::arch::x86_64::_rdtsc;

        let mut data = prepare();

        let before = unsafe { _rdtsc() };
        hint::black_box(run(&mut data));
        let after = unsafe { _rdtsc() };

        (RunTime::Cycles(after - before), data)
    }
}

pub trait HarnessVisitor {
    fn with_capacity(cardinality: usize) -> Self;
}

impl<T> HarnessVisitor for UnsafeWriter<T> {
    fn with_capacity(cardinality: usize) -> Self {
        UnsafeWriter::with_capacity(cardinality)
    }
}

impl HarnessVisitor for Counter {
    fn with_capacity(_cardinality: usize) -> Self {
        Counter::new()
    }
}

pub fn time_twoset<V>(
    harness: &Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], V>) -> RunTime
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = set_a.len().min(set_b.len());

    let prepare = || V::with_capacity(capacity);
    let run = |writer: &mut _| intersect(set_a, set_b, writer);

    let (elapsed, _writer) = harness.time(prepare, run);

    elapsed
}

pub fn time_bsr(
    harness: &Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: UnsafeIntersectBsr) -> RunTime
{
    let bsr_a = BsrVec::from_sorted(util::slice_i32_to_u32(set_a));
    let bsr_b = BsrVec::from_sorted(util::slice_i32_to_u32(set_b));

    let capacity = bsr_a.len().min(bsr_b.len());

    let prepare = || UnsafeBsrWriter::with_capacities(capacity);
    let run = |writer: &mut _| intersect(bsr_a.bsr_ref(), bsr_b.bsr_ref(), writer);

    let (elapsed, _writer) = harness.time(prepare, run);

    elapsed
}

pub fn time_kset<V>(
    harness: &Harness,
    sets: &[DatafileSet],
    intersect: IntersectK<DatafileSet, V>) -> TimeResult
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || V::with_capacity(capacity);
    let run = |writer: &mut _| intersect(sets, writer);

    let (elapsed, _writer) = harness.time(prepare, run);

    Ok(elapsed)
}

pub fn time_svs<V>(
    harness: &Harness,
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], UnsafeWriter<i32>>) -> TimeResult
{
    // Note: max() required here
    let capacity = sets.iter().map(|s| s.len()).max()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || (
        UnsafeWriter::with_capacity(capacity),
        UnsafeWriter::with_capacity(capacity)
    );
    let run = |(left, right): &mut _| {
        intersect::svs_generic(sets, left, right, intersect);
    };

    let (elapsed, _) = harness.time(prepare, run);

    Ok(elapsed)
}

pub fn time_croaring_2set(
    harness: &Harness,
    set_a: &[i32],
    set_b: &[i32],
    count_only: bool) -> RunTime
{
    use croaring::Bitmap;

    let prepare = || {
        let mut bitmap_a = Bitmap::of(util::slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(util::slice_i32_to_u32(&set_b));
        bitmap_a.run_optimize();
        bitmap_b.run_optimize();
        (bitmap_a, bitmap_b)
    };
    let run = if count_only {
        |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
            bitmap_a.and_inplace(&bitmap_b);
        }
    } else {
        |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
            bitmap_a.and_cardinality(&bitmap_b);
        }
    };

    let (elapsed, _) = harness.time(prepare, run);
    elapsed
}

pub fn time_croaring_svs(harness: &Harness, sets: &[DatafileSet])
    -> RunTime
{
    use croaring::Bitmap;
    assert!(sets.len() > 2);

    let prepare = || {
        let mut victim = Bitmap::of(util::slice_i32_to_u32(&sets[0]));
        victim.run_optimize();

        let rest: Vec<Bitmap> = (&sets[1..]).iter()
            .map(|s| {
                let mut bitmap = Bitmap::of(util::slice_i32_to_u32(&s));
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

    let (elapsed, _) = harness.time(prepare, run);
    elapsed
}

pub fn time_roaringrs_2set(harness: &Harness, set_a: &[i32], set_b: &[i32])
    -> RunTime
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

    let (elapsed, _) = harness.time(prepare, run);
    elapsed
}

pub fn time_roaringrs_svs(harness: &Harness, sets: &[DatafileSet])
    -> RunTime
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

    let (elapsed, _) = harness.time(prepare, run);
    elapsed
}

pub fn time_fesia<H, S, M, const LANES: usize, V>(
    harness: &Harness,
    set_a: &[i32],
    set_b: &[i32],
    hash_scale: HashScale,
    intersect_method: FesiaTwoSetMethod,
    simd_type: SimdType)
    -> TimeResult
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = set_a.len().min(set_b.len());

    let set_a: Fesia<H, S, M, LANES> = Fesia::from_sorted(set_a, hash_scale);
    let set_b: Fesia<H, S, M, LANES> = Fesia::from_sorted(set_b, hash_scale);

    let prepare = || V::with_capacity(capacity);

    use FesiaTwoSetMethod::*;
    use SimdType::*;

    let (elapsed, _) = match (intersect_method, simd_type) {
        #[cfg(target_feature = "ssse3")]
        (SimilarSize, Sse) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectSse>(&set_b, writer);
            harness.time(prepare, run)
        }
        #[cfg(target_feature = "avx2")]
        (SimilarSize, Avx2) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectAvx2>(&set_b, writer);
            harness.time(prepare, run)
        }
        #[cfg(target_feature = "avx512f")]
        (SimilarSize, Avx512) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectAvx512>(&set_b, writer);
            harness.time(prepare, run)
        }
        #[allow(unreachable_patterns)]
        (SimilarSize, width) =>
            return Err(format!("fesia SimilarSize does not support {:?}", width)),
        #[cfg(target_feature = "ssse3")]
        (SimilarSizeShuffling, Sse) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectShufflingSse>(&set_b, writer);
            harness.time(prepare, run)
        },
        #[cfg(target_feature = "avx2")]
        (SimilarSizeShuffling, Avx2) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectShufflingAvx2>(&set_b, writer);
            harness.time(prepare, run)
        },
        #[cfg(target_feature = "avx512f")]
        (SimilarSizeShuffling, Avx512) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectShufflingAvx512>(&set_b, writer);
            harness.time(prepare, run)
        },
        #[allow(unreachable_patterns)]
        (SimilarSizeShuffling, width) => 
            return Err(format!("fesia SimilarSizeShuffling does not support {:?}", width)),
        (Skewed, _) =>
            harness.time(prepare, |writer: &mut _| set_a.hash_intersect(&set_b, writer)),
    };

    Ok(elapsed)
}

pub fn time_fesia_kset<H, S, M, const LANES: usize, V>(
    harness: &Harness,
    sets: &[DatafileSet],
    hash_scale: HashScale,
    intersect_method: FesiaKSetMethod)
    -> TimeResult
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let fesia_sets: Vec<Fesia<H, S, M, LANES>> = sets.iter()
        .map(|s| Fesia::from_sorted(s, hash_scale))
        .collect();

    let prepare = || V::with_capacity(capacity);

    use FesiaKSetMethod::*;

    let (elapsed, _) = match intersect_method {
        SimilarSize => harness.time(prepare,
            |writer: &mut _| Fesia::<H, S, M, LANES>::intersect_k(&fesia_sets, writer)),
    };

    Ok(elapsed)
}
