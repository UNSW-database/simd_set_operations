use std::{
    time::{Duration, Instant},
    hint, simd::{*, cmp::*}, ops::BitAnd,
};
use setops::{
    bsr::{BsrRef, BsrVec}, intersect::{self, fesia::*, Intersect2, Intersect2C, IntersectK}, visitor::*, Set
};
use crate::{datafile::DatafileSet, util, timer::perf::*};

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use setops::visitor::UnsafeCompressWriter;

pub type RunResult = Result<Run, String>;
pub type IntersectBsr<V> = for<'a> fn(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V);

pub struct Run {
    pub time: Duration,
    pub perf: PerfResults,
    pub bytes: u64,
}


pub struct Harness<'a> {
    warmup: u32,
    counters: &'a mut PerfCounters,
}

impl<'a> Harness<'a> {
    pub fn new(warmup: u32, counters: &'a mut PerfCounters) -> Self {
        Self { warmup, counters }
    }

    pub fn time<D>(
        &mut self,
        prepare: impl Fn() -> D,
        run: impl Fn(&mut D),
        bytes_read: u64) -> (Run, D)
    {
        for _ in 0..self.warmup {
            let mut data = prepare();
            hint::black_box(run(&mut data));
        }

        let mut data = prepare();

        self.counters.enable();

        let start = Instant::now();
        hint::black_box(run(&mut data));
        let elapsed = start.elapsed();

        self.counters.disable();

        let run_result = Run {
            time: elapsed,
            perf: self.counters.results(),
            bytes: bytes_read,
        };

        (run_result, data)
    }
}

pub trait HarnessVisitor {
    fn with_capacity(cardinality: usize) -> Self;
}

impl<T> HarnessVisitor for UnsafeLookupWriter<T> {
    fn with_capacity(cardinality: usize) -> Self {
        UnsafeLookupWriter::with_capacity(cardinality)
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> HarnessVisitor for UnsafeCompressWriter<T> {
    fn with_capacity(cardinality: usize) -> Self {
        UnsafeCompressWriter::with_capacity(cardinality)
    }
}

impl HarnessVisitor for Counter {
    fn with_capacity(_cardinality: usize) -> Self {
        Counter::new()
    }
}

impl HarnessVisitor for UnsafeLookupBsrWriter {
    fn with_capacity(cardinality: usize) -> Self {
        UnsafeLookupBsrWriter::with_capacities(cardinality)
    }
}

impl HarnessVisitor for UnsafeCompressBsrWriter {
    fn with_capacity(cardinality: usize) -> Self {
        UnsafeCompressBsrWriter::with_capacities(cardinality)
    }
}

pub fn time_twoset<V>(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2<[i32], V>) -> Run
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = set_a.len().min(set_b.len());

    let prepare = || V::with_capacity(capacity);
    let run = |writer: &mut _| intersect(set_a, set_b, writer);

    let bytes = (set_a.len() + set_b.len()) as u64 * std::mem::size_of::<i32>() as u64;
    let (elapsed, _writer) = harness.time(prepare, run, bytes);

    elapsed
}

pub fn time_twoset_c(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: Intersect2C<[i32]>) -> Run
{
    let capacity = set_a.len().min(set_b.len());

    let prepare = || vec![0;capacity];
    let run = |result: &mut Vec<i32>| _ = intersect(set_a, set_b, result.as_mut_slice());

    let bytes = (set_a.len() + set_b.len()) as u64 * std::mem::size_of::<i32>() as u64;
    let (elapsed, _writer) = harness.time(prepare, run, bytes);

    elapsed
}

pub fn time_bsr<V>(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: IntersectBsr<V>) -> Run
where
    V: BsrVisitor + SimdBsrVisitor4 + SimdBsrVisitor8 + SimdBsrVisitor16 + HarnessVisitor
{
    let bsr_a = BsrVec::from_sorted(util::slice_i32_to_u32(set_a));
    let bsr_b = BsrVec::from_sorted(util::slice_i32_to_u32(set_b));

    let capacity = bsr_a.len().min(bsr_b.len());

    let prepare = || V::with_capacity(capacity);
    let run = |writer: &mut _| intersect(bsr_a.bsr_ref(), bsr_b.bsr_ref(), writer);

    let bytes = (bsr_a.size_bytes() + bsr_b.size_bytes()) as u64;
    let (elapsed, _writer) = harness.time(prepare, run, bytes);

    elapsed
}

pub fn time_kset<V>(
    harness: &mut Harness,
    sets: &[DatafileSet],
    intersect: IntersectK<DatafileSet, V>) -> RunResult
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || V::with_capacity(capacity);
    let run = |writer: &mut _| intersect(sets, writer);

    let bytes = sets.iter().map(|s| s.len()).sum::<usize>() as u64 * std::mem::size_of::<i32>() as u64;
    let (elapsed, _writer) = harness.time(prepare, run, bytes);

    Ok(elapsed)
}

pub fn time_svs<V>(
    harness: &mut Harness,
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], V>) -> RunResult
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor,
    V: Clearable + AsRef<[i32]>,
{
    // Note: max() required here
    let capacity = sets.iter().map(|s| s.len()).max()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || (
        V::with_capacity(capacity),
        V::with_capacity(capacity)
    );
    let run = |(left, right): &mut _| {
        intersect::svs_generic(sets, left, right, intersect);
    };

    let bytes = sets.iter().map(|s| s.len()).sum::<usize>() as u64 * std::mem::size_of::<i32>() as u64;
    let (elapsed, _) = harness.time(prepare, run, bytes);

    Ok(elapsed)
}

pub fn time_svs_c(
    harness: &mut Harness,
    sets: &[DatafileSet],
    intersect: Intersect2C<[i32]>) -> RunResult
{
    // Note: max() required here
    let capacity = sets.iter().map(|s| s.len()).max()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let prepare = || (
        vec![0 as i32;capacity],
        vec![0 as i32;capacity]
    );
    let run = |(ref mut left, ref mut right): &mut (Vec<i32>, Vec<i32>)| {
        intersect::svs_generic_c(sets, left, right, intersect);
    };

    let bytes = sets.iter().map(|s| s.len()).sum::<usize>() as u64 * std::mem::size_of::<i32>() as u64;
    let (elapsed, _) = harness.time(prepare, run, bytes);

    Ok(elapsed)
}

fn croaring_bytes(bitmap: &croaring::Bitmap) -> u64{
    let stats = bitmap.statistics();
    return (stats.n_bytes_array_containers +
            stats.n_bytes_bitset_containers +
            stats.n_bytes_run_containers) as u64;
}

#[derive(Debug, Clone, Copy)]
pub enum CRoaringType {
    CountOnly,
    Inplace,
    Normal,
}

pub fn time_croaring_2set(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    mode: CRoaringType,
    optimise: bool) -> Run
{
    use croaring::Bitmap;

    let bytes = {
        let mut bitmap_a = Bitmap::of(util::slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(util::slice_i32_to_u32(&set_b));
        
        if optimise {
            bitmap_a.run_optimize();
            bitmap_b.run_optimize();
        }

        croaring_bytes(&bitmap_a) + croaring_bytes(&bitmap_b)
    };

    let prepare = || {
        let mut bitmap_a = Bitmap::of(util::slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(util::slice_i32_to_u32(&set_b));
        if optimise {
            bitmap_a.run_optimize();
            bitmap_b.run_optimize();
        }
        (bitmap_a, bitmap_b)
    };
    let run = match mode {
        CRoaringType::CountOnly => |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
            bitmap_a.and_cardinality(&bitmap_b);
        },
        CRoaringType::Inplace => |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
            bitmap_a.and_inplace(&bitmap_b);
        },
        CRoaringType::Normal => |(bitmap_a, bitmap_b): &mut (Bitmap, Bitmap)| {
            bitmap_a.and(&bitmap_b);
        },
    };

    let (elapsed, _) = harness.time(prepare, run, bytes);
    elapsed
}

pub fn time_croaring_svs(harness: &mut Harness, sets: &[DatafileSet], optimise: bool)
    -> Run
{
    use croaring::Bitmap;
    assert!(sets.len() > 2);

    let bytes = {
        sets.iter()
            .map(|s| {
                let mut bitmap = Bitmap::of(util::slice_i32_to_u32(&s));
                if optimise {
                    bitmap.run_optimize();
                }
                croaring_bytes(&bitmap)
            }).sum::<u64>()
    };

    let prepare = || {
        let mut victim = Bitmap::of(util::slice_i32_to_u32(&sets[0]));
        victim.run_optimize();

        let rest: Vec<Bitmap> = (&sets[1..]).iter()
            .map(|s| {
                let mut bitmap = Bitmap::of(util::slice_i32_to_u32(&s));
                if optimise {
                    bitmap.run_optimize();
                }
                bitmap
            }).collect();

        (victim, rest)
    };
    let run = |(victim, rest): &mut (Bitmap, Vec<Bitmap>)| {
        for bitmap in rest {
            victim.and_inplace(bitmap);
        }
    };

    let (elapsed, _) = harness.time(prepare, run, bytes);
    elapsed
}

// pub fn time_roaringrs_2set(harness: &Harness, set_a: &[i32], set_b: &[i32])
//     -> RunTime
// {
//     use roaring::RoaringBitmap;

//     let prepare = || {
//         let iter_a = set_a.iter().map(|&i| i as u32);
//         let iter_b = set_b.iter().map(|&i| i as u32);

//         let victim = RoaringBitmap::from_sorted_iter(iter_a).unwrap();
//         let other = RoaringBitmap::from_sorted_iter(iter_b).unwrap();

//         (victim, other)
//     };
//     let run = |(victim, other): &mut _| {
//         *victim &= &*other
//     };

//     let (elapsed, _) = harness.time(prepare, run);
//     elapsed
// }

// pub fn time_roaringrs_svs(harness: &Harness, sets: &[DatafileSet])
//     -> RunTime
// {
//     use roaring::RoaringBitmap;
//     assert!(sets.len() > 2);

//     let prepare = || {
//         let victim = RoaringBitmap::from_sorted_iter(
//             sets[0].iter().map(|&i| i as u32)).unwrap();

//         let rest: Vec<RoaringBitmap> = (&sets[1..]).iter()
//             .map(|s|
//                 RoaringBitmap::from_sorted_iter(s.iter().map(|&i| i as u32)).unwrap()
//             ).collect();

//         (victim, rest)
//     };
//     let run = |(victim, rest): &mut _| {
//         for bitmap in &*rest {
//             *victim &= bitmap;
//         }
//     };

//     let (elapsed, _) = harness.time(prepare, run);
//     elapsed
// }

pub fn time_fesia<H, S, const LANES: usize, V>(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    hash_scale: HashScale,
    intersect_method: FesiaTwoSetMethod,
    simd_type: SimdType)
    -> RunResult
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = set_a.len().min(set_b.len());
    assert!(set_a.len() <= set_b.len());

    let set_a: Fesia<H, S, LANES> = Fesia::from_sorted(set_a, hash_scale);
    let set_b: Fesia<H, S, LANES> = Fesia::from_sorted(set_b, hash_scale);

    let prepare = || V::with_capacity(capacity);

    use FesiaTwoSetMethod::*;
    use SimdType::*;

    let (elapsed, _) = match (intersect_method, simd_type) {
        #[cfg(target_feature = "ssse3")]
        (SimilarSize, Sse) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectSse>(&set_b, writer);
            harness.time(prepare, run, 0)
        }
        #[cfg(target_feature = "avx2")]
        (SimilarSize, Avx2) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectAvx2>(&set_b, writer);
            harness.time(prepare, run, 0)
        }
        #[cfg(target_feature = "avx512f")]
        (SimilarSize, Avx512) => {
            let run = |writer: &mut _| set_a.intersect::<V, SegmentIntersectAvx512>(&set_b, writer);
            harness.time(prepare, run, 0)
        }
        #[allow(unreachable_patterns)]
        (SimilarSize, width) =>
            return Err(format!("fesia SimilarSize does not support {:?}", width)),
        (Skewed, _) =>
            harness.time(prepare, |writer: &mut _| set_a.hash_intersect(&set_b, writer), 0),
    };

    Ok(elapsed)
}

pub fn time_fesia_kset<H, S, const LANES: usize, V>(
    harness: &mut Harness,
    sets: &[DatafileSet],
    hash_scale: HashScale,
    intersect_method: FesiaKSetMethod)
    -> RunResult
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    let capacity = sets.iter().map(|s| s.len()).min()
        .ok_or_else(|| "cannot intersect 0 sets".to_string())?;

    let fesia_sets: Vec<Fesia<H, S, LANES>> = sets.iter()
        .map(|s| Fesia::from_sorted(s, hash_scale))
        .collect();

    let prepare = || V::with_capacity(capacity);

    use FesiaKSetMethod::*;

    let (elapsed, _) = match intersect_method {
        SimilarSize => harness.time(prepare,
            |writer: &mut _| Fesia::<H, S, LANES>::intersect_k(&fesia_sets, writer), 0),
    };

    Ok(elapsed)
}
