use std::{
    time::{Duration, Instant},
    hint, simd::{*, cmp::*}, ops::BitAnd,
};
use setops::{
    intersect::{Intersect2, Intersect2C, IntersectK, fesia::*, self},
    visitor::{
        Visitor, SimdVisitor4, SimdVisitor8, SimdVisitor16,
        UnsafeWriter, UnsafeBsrWriter, Counter
    },
    bsr::{BsrVec, BsrRef},
    Set,
};
use crate::{datafile::DatafileSet, util, schema};
use perf_event;

pub type RunResult = Result<Run, String>;
pub type UnsafeIntersectBsr = for<'a> fn(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut UnsafeBsrWriter);

pub struct Run {
    pub time: Duration,
    pub perf: PerfResults,
}

#[derive(Debug)]
pub struct PerfResults {
    pub l1d: CacheResult,
    pub l1i: CacheResult,
    pub ll: CacheResult,

    pub branches: Option<u64>,
    pub branch_misses: Option<u64>,

    pub cpu_stalled_front: Option<u64>,
    pub cpu_stalled_back: Option<u64>,
    pub cpu_cycles: Option<u64>,
    pub cpu_cycles_ref: Option<u64>,
}

#[derive(Debug)]
pub struct CacheResult {
    pub rd_access: Option<u64>,
    pub rd_miss: Option<u64>,
    pub wr_access: Option<u64>,
    pub wr_miss: Option<u64>,
}

pub struct PerfCounters {
    group: perf_event::Group,
    l1d: CacheCounters,
    l1i: CacheCounters,
    ll: CacheCounters,
    branches: Option<perf_event::Counter>,
    branch_misses: Option<perf_event::Counter>,
    cpu_stalled_front: Option<perf_event::Counter>,
    cpu_stalled_back: Option<perf_event::Counter>,
    cpu_cycles: Option<perf_event::Counter>,
    cpu_cycles_ref: Option<perf_event::Counter>,
}

pub struct CacheCounters {
    pub rd_access: Option<perf_event::Counter>,
    pub rd_miss: Option<perf_event::Counter>,
    pub wr_access: Option<perf_event::Counter>,
    pub wr_miss: Option<perf_event::Counter>,
}

impl PerfCounters {
    pub fn new() -> Self {
        use perf_event::{*, events::*, events::Hardware};
        let mut group = Group::new().unwrap();
        
        let l1d = Self::cache_group(CacheId::L1D, &mut group);
        let l1i = Self::cache_group(CacheId::L1I, &mut group);
        // let ll = Self::cache_group(CacheId::LL, &mut group);
        let branches = group.add(&Builder::new(Hardware::BRANCH_INSTRUCTIONS)).ok();
        let branch_misses = group.add(&Builder::new(Hardware::BRANCH_MISSES)).ok();
        // let cpu_stalled_front = group.add(&Builder::new(Hardware::STALLED_CYCLES_FRONTEND)).ok();
        // let cpu_stalled_back = group.add(&Builder::new(Hardware::STALLED_CYCLES_BACKEND)).ok();
        let cpu_cycles = group.add(&Builder::new(Hardware::CPU_CYCLES)).ok();
        let cpu_cycles_ref = group.add(&Builder::new(Hardware::REF_CPU_CYCLES)).ok();

        // let lld = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        // let l1i = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        let ll = CacheCounters{ rd_access: None, rd_miss: None, wr_access: None, wr_miss: None};
        // let branches = None;
        // let branch_misses = None;
        let cpu_stalled_front = None;
        let cpu_stalled_back = None;
        // let cpu_cycles = None;
        // let cpu_cycles_ref = None;
        Self {
            group, l1d, l1i, ll, branches, branch_misses,
            cpu_stalled_front, cpu_stalled_back, cpu_cycles, cpu_cycles_ref
        }
    }

    pub fn summarise(&self) {
        use colored::Colorize;
        let convert = |c: &Option<perf_event::Counter>|
            c.as_ref().map_or("disabled".yellow(), |_| "enabled".green());

        println!("=== CPU Performance Counters ===");

        println!("l1d.rd_access: {}", convert(&self.l1d.rd_access));
        println!("l1d.rd_miss: {}", convert(&self.l1d.rd_miss));
        println!("l1d.wr_access: {}", convert(&self.l1d.wr_access));
        println!("l1d.wr_miss: {}", convert(&self.l1d.wr_miss));

        println!("l1i.rd_access: {}", convert(&self.l1i.rd_access));
        println!("l1i.rd_miss: {}", convert(&self.l1i.rd_miss));
        println!("l1i.wr_access: {}", convert(&self.l1i.wr_access));
        println!("l1i.wr_miss: {}", convert(&self.l1i.wr_miss));

        println!("ll.rd_access: {}", convert(&self.ll.rd_access));
        println!("ll.rd_miss: {}", convert(&self.ll.rd_miss));
        println!("ll.wr_access: {}", convert(&self.ll.wr_access));
        println!("ll.wr_miss: {}", convert(&self.ll.wr_miss));

        println!("branches: {}", convert(&self.branches));
        println!("branch_misses: {}", convert(&self.branch_misses));

        println!("cpu_stalled_front: {}", convert(&self.cpu_stalled_front));
        println!("cpu_stalled_back: {}", convert(&self.cpu_stalled_back));
        println!("cpu_cycles: {}", convert(&self.cpu_cycles));
        println!("cpu_cycles_ref: {}", convert(&self.cpu_cycles_ref));

        println!("================================");
    }

    pub fn enable(&mut self) {
        self.group.enable().expect("Failed to enable group");
    }

    pub fn disable(&mut self) {
        self.group.disable().expect("Failed to disable group");
    }

    pub fn results(&mut self) -> PerfResults {
        let counts = self.group.read().unwrap();
        PerfResults {
            l1d: Self::cache_results(&self.l1d, &counts),
            l1i: Self::cache_results(&self.l1i, &counts),
            ll: Self::cache_results(&self.ll, &counts),
            branches: self.branches.as_ref().map(|c| counts[c]),
            branch_misses: self.branch_misses.as_ref().map(|c| counts[c]),
            cpu_stalled_front: self.cpu_stalled_front.as_ref().map(|c| counts[c]),
            cpu_stalled_back: self.cpu_stalled_back.as_ref().map(|c| counts[c]),
            cpu_cycles: self.cpu_cycles.as_ref().map(|c| counts[c]),
            cpu_cycles_ref: self.cpu_cycles_ref.as_ref().map(|c| counts[c]),
        }
    }

    pub fn new_result_run(&self, x: u32) -> schema::ResultRun {
        schema::ResultRun {
            x: x,
            times: Vec::default(),
            l1d: Self::new_cache_run(&self.l1d),
            l1i: Self::new_cache_run(&self.l1i),
            ll: Self::new_cache_run(&self.ll),
            branches: self.branches.as_ref().map(|_| Vec::new()),
            branch_misses: self.branch_misses.as_ref().map(|_| Vec::new()),
            cpu_stalled_front: self.cpu_stalled_front.as_ref().map(|_| Vec::new()),
            cpu_stalled_back: self.cpu_stalled_back.as_ref().map(|_| Vec::new()),
            cpu_cycles: self.cpu_cycles.as_ref().map(|_| Vec::new()),
            cpu_cycles_ref: self.cpu_cycles_ref.as_ref().map(|_| Vec::new()),
        }
    }


    fn cache_group(which: perf_event::events::CacheId, group: &mut perf_event::Group) -> CacheCounters {
        use perf_event::{*, events::*};
        CacheCounters {
            rd_access: group.add(&Builder::new(Cache{ which: which, operation: CacheOp::READ, result: CacheResult::ACCESS })).ok(),
            rd_miss:   group.add(&Builder::new(Cache{ which: which, operation: CacheOp::READ, result: CacheResult::MISS })).ok(),
            wr_access: group.add(&Builder::new(Cache{ which: which, operation: CacheOp::WRITE, result: CacheResult::ACCESS })).ok(),
            wr_miss:   group.add(&Builder::new(Cache{ which: which, operation: CacheOp::WRITE, result: CacheResult::MISS })).ok(),
        }
    }

    fn cache_results(counters: &CacheCounters, counts: &perf_event::GroupData) -> CacheResult {
        CacheResult {
            rd_access: counters.rd_access.as_ref().map(|c| counts[c]),
            rd_miss: counters.rd_miss.as_ref().map(|c| counts[c]),
            wr_access: counters.wr_access.as_ref().map(|c| counts[c]),
            wr_miss: counters.wr_miss.as_ref().map(|c| counts[c]),
        }
    }

    fn new_cache_run(counters: &CacheCounters) -> schema::CacheRun {
        schema::CacheRun {
            rd_access: counters.rd_access.as_ref().map(|_| Vec::default()),
            rd_miss: counters.rd_miss.as_ref().map(|_| Vec::default()),
            wr_access: counters.wr_access.as_ref().map(|_| Vec::default()),
            wr_miss: counters.wr_miss.as_ref().map(|_| Vec::default()),
        }
    }
}

pub struct Harness<'a> {
    warmup: Duration,
    counters: &'a mut PerfCounters,
}

impl<'a> Harness<'a> {
    pub fn new(warmup: Duration, counters: &'a mut PerfCounters) -> Self {
        Self { warmup, counters }
    }

    pub fn time<D>(
        &mut self,
        prepare: impl Fn() -> D,
        run: impl Fn(&mut D)) -> (Run, D)
    {
        let warmup_start = Instant::now();
        while warmup_start.elapsed() < self.warmup {
            let mut data = prepare();
            hint::black_box(run(&mut data));
        }

        let mut data = prepare();

        self.counters.group.reset().unwrap();
        self.counters.enable();

        let start = Instant::now();
        hint::black_box(run(&mut data));
        let elapsed = start.elapsed();

        self.counters.disable();

        let run_result = Run {
            time: elapsed,
            perf: self.counters.results(),
        };

        (run_result, data)
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

    let (elapsed, _writer) = harness.time(prepare, run);

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

    let (elapsed, _writer) = harness.time(prepare, run);

    elapsed
}

pub fn time_bsr(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    intersect: UnsafeIntersectBsr) -> Run
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

    let (elapsed, _writer) = harness.time(prepare, run);

    Ok(elapsed)
}

pub fn time_svs<V>(
    harness: &mut Harness,
    sets: &[DatafileSet],
    intersect: Intersect2<[i32], UnsafeWriter<i32>>) -> RunResult
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

    let (elapsed, _) = harness.time(prepare, run);

    Ok(elapsed)
}

pub fn time_croaring_2set(
    harness: &mut Harness,
    set_a: &[i32],
    set_b: &[i32],
    count_only: bool,
    optimise: bool) -> Run
{
    use croaring::Bitmap;

    let prepare = || {
        let mut bitmap_a = Bitmap::of(util::slice_i32_to_u32(&set_a));
        let mut bitmap_b = Bitmap::of(util::slice_i32_to_u32(&set_b));
        if optimise {
            bitmap_a.run_optimize();
            bitmap_b.run_optimize();
        }
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

pub fn time_croaring_svs(harness: &mut Harness, sets: &[DatafileSet], optimise: bool)
    -> Run
{
    use croaring::Bitmap;
    assert!(sets.len() > 2);

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

    let (elapsed, _) = harness.time(prepare, run);
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
        (Skewed, _) =>
            harness.time(prepare, |writer: &mut _| set_a.hash_intersect(&set_b, writer)),
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
            |writer: &mut _| Fesia::<H, S, LANES>::intersect_k(&fesia_sets, writer)),
    };

    Ok(elapsed)
}
