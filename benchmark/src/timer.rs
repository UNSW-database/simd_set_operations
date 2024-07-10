pub mod harness;
pub mod perf;

use std::{simd::{*, cmp::*}, ops::BitAnd};

use setops::{
    intersect::{
        self, fesia::{FesiaKSetMethod, FesiaTwoSetMethod, HashScale, IntegerHash, SimdType}, Intersect2, Intersect2C, IntersectK
    },
    visitor::*,
};

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use setops::visitor::UnsafeCompressWriter;

use crate::{datafile::DatafileSet, timer::harness::time_fesia_kset};
use harness::{Harness, HarnessVisitor, RunResult, IntersectBsr};

type TwosetTimer = Box<dyn Fn(&mut Harness, &[i32], &[i32]) -> RunResult>;
type KsetTimer = Box<dyn Fn(&mut Harness, &[DatafileSet]) -> RunResult>;

pub struct Timer {
    twoset: Option<TwosetTimer>,
    kset: Option<KsetTimer>,
}

impl Timer {
    pub fn new(name: &str) -> Option<Self>
    {
        try_parse_with_visitor(name)
            .or_else(|| try_parse_twoset_c(name))
            .or_else(|| try_parse_bsr(name))
            .or_else(|| try_parse_roaring(name))
    }

    pub fn run(&self, harness: &mut Harness, sets: &[DatafileSet]) -> RunResult {
        if sets.len() == 2 {
            if let Some(twoset) = &self.twoset {
                twoset(harness, &sets[0], &sets[1])
            }
            else if let Some(kset) = &self.kset {
                kset(harness, sets)
            }
            else {
                Err("intersection not supported".to_string())
            }
        }
        else {
            if let Some(kset) = &self.kset {
                kset(harness, sets)
            }
            else {
                Err("k-set intersection not supported".to_string())
            }
        }
    }
}

fn try_parse_with_visitor(name: &str) -> Option<Timer> {
    const COMP: &str = "_comp";
    const COUNT: &str = "_count";
    const LUT: &str = "_lut";

    if name.ends_with(COMP) {
        #[cfg(all(feature = "simd", target_feature = "avx512f"))] {
            let name = &name[..name.len() - COMP.len()];
            type V = UnsafeCompressWriter<i32>;
            try_parse_twoset_simd_with_visitor::<V>(name)
                .or_else(|| try_parse_kset_with_visitor::<V>(name))
                .or_else(|| try_parse_fesia_hash_with_visitor::<V>(name))
                .or_else(|| try_parse_fesia_with_visitor::<V>(name))
        }
        #[cfg(all(feature = "simd", not(target_feature = "avx512f")))] {
            None
        }
    }
    else if name.ends_with(COUNT) {
        let name = &name[..name.len() - COUNT.len()];
        type V = Counter;
        try_parse_twoset_simd_with_visitor::<V>(name)
            .or_else(|| try_parse_kset_with_visitor::<V>(name))
            .or_else(|| try_parse_fesia_hash_with_visitor::<V>(name))
            .or_else(|| try_parse_fesia_with_visitor::<V>(name))
    }
    else if name.ends_with(LUT) {
        let name = &name[..name.len() - LUT.len()];
        type V = UnsafeLookupWriter<i32>;
        try_parse_twoset_simd_with_visitor::<V>(name)
            .or_else(|| try_parse_kset_with_visitor::<V>(name))
            .or_else(|| try_parse_fesia_hash_with_visitor::<V>(name))
            .or_else(|| try_parse_fesia_with_visitor::<V>(name))
    }
    else {
        None
    }
}

fn try_parse_bsr(name: &str) -> Option<Timer> {
    const COMP: &str = "_comp";
    const COUNT: &str = "_count";
    const LUT: &str = "_lut";

    if name.ends_with(COMP) {
        #[cfg(all(feature = "simd", target_feature = "avx512f"))] {
            try_parse_bsr_with_visitor::<UnsafeCompressBsrWriter>(&name[..name.len() - COMP.len()])
        }
        #[cfg(all(feature = "simd", not(target_feature = "avx512f")))] {
            None
        }
    }
    else if name.ends_with(COUNT) {
        try_parse_bsr_with_visitor::<Counter>(&name[..name.len() - COUNT.len()])
    }
    else if name.ends_with(LUT) {
        try_parse_bsr_with_visitor::<UnsafeLookupBsrWriter>(&name[..name.len() - LUT.len()])
    }
    else {
        None
    }
}

fn try_parse_twoset_simd_with_visitor<V>(name: &str) -> Option<Timer> 
where
    V: Visitor<i32> + HarnessVisitor + TwosetTimingSpec<V>,
    V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + 'static
{
    let maybe_intersect: Option<Intersect2<[i32], V>> = match name {
        "naive_merge"      => Some(intersect::naive_merge),
        "branchless_merge" => Some(intersect::branchless_merge),
        "bmiss_scalar_3x"  => Some(intersect::bmiss_scalar_3x),
        "bmiss_scalar_4x"  => Some(intersect::bmiss_scalar_4x),
        "galloping"        => Some(intersect::galloping),
        "binary_search"    => Some(intersect::binary_search_intersect),
        "baezayates"       => Some(intersect::baezayates),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse"    => Some(intersect::shuffling_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse"    => Some(intersect::broadcast_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss"        => Some(intersect::bmiss),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sttni"  => Some(intersect::bmiss_sttni),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter"          => Some(intersect::qfilter),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_v1"          => Some(intersect::qfilter_v1),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "lbk_v1x4_sse"    => Some(intersect::lbk_v1x4_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "lbk_v1x8_sse"    => Some(intersect::lbk_v1x8_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "lbk_v3_sse"    => Some(intersect::lbk_v3_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "galloping_sse"    => Some(intersect::galloping_sse),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2"   => Some(intersect::shuffling_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2"   => Some(intersect::broadcast_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "lbk_v1x8_avx2"   => Some(intersect::lbk_v1x8_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "lbk_v1x16_avx2"   => Some(intersect::lbk_v1x16_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "lbk_v3_avx2"   => Some(intersect::lbk_v3_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2"   => Some(intersect::galloping_avx2),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512"       => Some(intersect::shuffling_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "broadcast_avx512"       => Some(intersect::broadcast_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "vp2intersect_emulation" => Some(intersect::vp2intersect_emulation),
        #[cfg(all(feature = "simd", target_feature = "avx512cd"))]
        "conflict_intersect"     => Some(intersect::conflict_intersect),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "lbk_v1x16_avx512"       => Some(intersect::lbk_v1x16_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "lbk_v1x32_avx512"       => Some(intersect::lbk_v1x32_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "lbk_v3_avx512"       => Some(intersect::lbk_v3_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512"       => Some(intersect::galloping_avx512),
        // Branch
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_br"    => Some(intersect::shuffling_sse_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse_br"    => Some(intersect::broadcast_sse_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_br"        => Some(intersect::bmiss_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sttni_br"  => Some(intersect::bmiss_sttni_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_br"          => Some(intersect::qfilter_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_v1_br"       => Some(intersect::qfilter_v1_branch),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2_br"   => Some(intersect::shuffling_avx2_branch),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2_br"   => Some(intersect::broadcast_avx2_branch),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512_br"       => Some(intersect::shuffling_avx512_branch),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "broadcast_avx512_br"       => Some(intersect::broadcast_avx512_branch),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "vp2intersect_emulation_br" => Some(intersect::vp2intersect_emulation_branch),
        #[cfg(all(feature = "simd", target_feature = "avx512cd"))]
        "conflict_intersect_br"     => Some(intersect::conflict_intersect_branch),
        _ => None,
    };
    maybe_intersect.map(|intersect| V::twoset_timer(intersect))
}

fn try_parse_twoset_c(name: &str) -> Option<Timer> {
    let maybe_intersect: Option<Intersect2C<[i32]>> = match name {
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_c"    => Some(intersect::qfilter_c),
        _ => None,
    };
    maybe_intersect.map(|i| 
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_twoset_c(warmup, a, b, i)))),
            kset: Some(Box::new(
                move |warmup, sets| harness::time_svs_c(warmup, sets, i))),
        })
}

pub trait TwosetTimingSpec<V> {
    fn twoset_timer(i: Intersect2<[i32], V>) -> Timer;
}

impl TwosetTimingSpec<UnsafeLookupWriter<i32>> for UnsafeLookupWriter<i32> {
    fn twoset_timer(i: Intersect2<[i32], UnsafeLookupWriter<i32>>) -> Timer {
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_twoset(warmup, a, b, i)))),
            kset: Some(Box::new(
                move |warmup, sets| harness::time_svs::<UnsafeLookupWriter<i32>>(warmup, sets, i))),
        }
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl TwosetTimingSpec<UnsafeCompressWriter<i32>> for UnsafeCompressWriter<i32> {
    fn twoset_timer(i: Intersect2<[i32], UnsafeCompressWriter<i32>>) -> Timer {
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_twoset(warmup, a, b, i)))),
            kset: Some(Box::new(
                move |warmup, sets| harness::time_svs::<UnsafeCompressWriter<i32>>(warmup, sets, i))),
        }
    }
}

impl TwosetTimingSpec<Counter> for Counter {
    fn twoset_timer(i: Intersect2<[i32], Counter>) -> Timer {
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_twoset(warmup, a, b, i)))),
            kset: None,
        }
    }
}

fn try_parse_bsr_with_visitor<V>(name: &str) -> Option<Timer> 
where
    V: BsrVisitor + HarnessVisitor,
    V: SimdBsrVisitor4 + SimdBsrVisitor8 + SimdBsrVisitor16 + 'static
{
    let maybe_intersect: Option<IntersectBsr<V>> = match name {
        "branchless_merge_bsr" => Some(intersect::branchless_merge_bsr),
        "galloping_bsr"        => Some(intersect::galloping_bsr),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_bsr"    => Some(intersect::shuffling_sse_bsr),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse_bsr"    => Some(intersect::broadcast_sse_bsr),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_bsr"          => Some(intersect::qfilter_bsr),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "galloping_sse_bsr"    => Some(intersect::galloping_sse_bsr),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2_bsr"   => Some(intersect::shuffling_avx2_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2_bsr"   => Some(intersect::broadcast_avx2_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2_bsr"   => Some(intersect::galloping_avx2_bsr),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512_bsr"       => Some(intersect::shuffling_avx512_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "broadcast_avx512_bsr"       => Some(intersect::broadcast_avx512_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512_bsr"       => Some(intersect::galloping_avx512_bsr),
        // Branch
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_bsr_br"    => Some(intersect::shuffling_sse_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse_bsr_br"    => Some(intersect::broadcast_sse_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_bsr_br"          => Some(intersect::qfilter_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2_bsr_br"   => Some(intersect::shuffling_avx2_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2_bsr_br"   => Some(intersect::broadcast_avx2_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512_bsr_br"       => Some(intersect::shuffling_avx512_bsr_branch),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "broadcast_avx512_bsr_br"       => Some(intersect::broadcast_avx512_bsr_branch),
        _ => None,
    };
    maybe_intersect.map(|intersect: IntersectBsr<V>| Timer {
        twoset: Some(Box::new(move |warmup, a, b| Ok(harness::time_bsr(warmup, a, b, intersect)))),
        kset: None,
    })
}

fn try_parse_kset_with_visitor<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + HarnessVisitor + TwosetTimingSpec<V>,
    V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + 'static
{
    let maybe_intersect: Option<IntersectK<DatafileSet, V>> = match name {
        "baezayates_k"          => Some(intersect::baezayates_k),
        "small_adaptive"        => Some(intersect::small_adaptive),
        "small_adaptive_sorted" => Some(intersect::small_adaptive_sorted),
        _ => None,
    };
    maybe_intersect.map(|intersect| Timer {
        twoset: None,
        kset: Some(Box::new(move |warmup, sets| harness::time_kset(warmup, sets, intersect))),
    })
}

fn try_parse_roaring(name: &str) -> Option<Timer> { 

    let count_only = name.ends_with("_count");
    let name = if count_only { &name[..name.len() - "_count".len()] } else { name };

    match name {
        "croaring_opt" => Some(Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_croaring_2set(warmup, a, b, count_only, true)))),
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_croaring_svs(warmup, sets, true))))
                },
            }),
        "croaring" => Some(Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_croaring_2set(warmup, a, b, count_only, false)))),
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_croaring_svs(warmup, sets, false))))
                },
            }),
        // "roaringrs" => Some(Timer {
        //     twoset:
        //         if count_only { None } else {
        //             Some(Box::new(|warmup, a, b| Ok(harness::time_roaringrs_2set(warmup, a, b))))
        //         },
        //     kset:
        //         if count_only { None } else {
        //             Some(Box::new(|warmup, sets| Ok(harness::time_roaringrs_svs(warmup, sets))))
        //         },
        //     }),
        _ => None,
    }
}

fn try_parse_fesia_with_visitor<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    use intersect::fesia::*;

    let last_underscore = name.rfind("_")?;

    let hash_scale = &name[last_underscore+1..];
    if hash_scale.is_empty() {
        return None;
    }

    let hash_scale: HashScale = hash_scale.parse().ok()?;
    if hash_scale <= 0.0 {
        return None;
    }

    let prefix = &name[..last_underscore];

    const FESIA: &str = "fesia";

    use FesiaTwoSetMethod::*;
    let (intersect, rest) =
        if prefix.starts_with(FESIA) {
            (SimilarSize, &prefix[FESIA.len()..])
        }
        else {
            return None;
        };
    
    use SimdType::*;
    let simd_type =
        if rest.ends_with("sse") { Sse }
        else if rest.ends_with("avx2") { Avx2 }
        else if rest.ends_with("avx512") { Avx512 }
        else { return None; };

    let maybe_timer: Option<Timer> =
    match rest {
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "8_sse" =>
            Some(gen_fesia_timer::<MixHash, i8, 16, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "16_sse" =>
            Some(gen_fesia_timer::<MixHash, i16, 8, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "32_sse" =>
            Some(gen_fesia_timer::<MixHash, i32, 4, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "8_avx2" =>
            Some(gen_fesia_timer::<MixHash, i8, 32, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "16_avx2" =>
            Some(gen_fesia_timer::<MixHash, i16, 16, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "32_avx2" =>
            Some(gen_fesia_timer::<MixHash, i32, 8, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "8_avx512" =>
            Some(gen_fesia_timer::<MixHash, i8, 64, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "16_avx512" =>
            Some(gen_fesia_timer::<MixHash, i16, 32, V>(hash_scale, intersect, simd_type)),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "32_avx512" =>
            Some(gen_fesia_timer::<MixHash, i32, 16, V>(hash_scale, intersect, simd_type)),
        _ => None,
    };

    maybe_timer
}

fn try_parse_fesia_hash_with_visitor<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    use intersect::fesia::*;

    let last_underscore = name.rfind("_")?;

    let hash_scale = &name[last_underscore+1..];
    if hash_scale.is_empty() {
        return None;
    }

    let hash_scale: HashScale = hash_scale.parse().ok()?;
    if hash_scale <= 0.0 {
        return None;
    }

    let prefix = &name[..last_underscore];

    const FESIA_HASH: &str = "fesia_hash";

    use FesiaTwoSetMethod::*;
    let (intersect, rest) =
        if prefix.starts_with(FESIA_HASH) {
            (Skewed, &prefix[FESIA_HASH.len()..])
        }
        else {
            return None;
        };

    use SimdType::*;
    let maybe_timer: Option<Timer> =
    match rest {
        "8" => Some(gen_fesia_timer::<MixHash, i8, 16, V>(hash_scale, intersect, Sse)),
        "16" => Some(gen_fesia_timer::<MixHash, i16, 8, V>(hash_scale, intersect, Sse)),
        "32" => Some(gen_fesia_timer::<MixHash, i32, 4, V>(hash_scale, intersect, Sse)),
        _ => None,
    };

    maybe_timer
}

fn gen_fesia_timer<H, S, const LANES: usize, V>(
    hash_scale: HashScale,
    intersect_method: FesiaTwoSetMethod,
    simd_type: SimdType)
    -> Timer
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    V: Visitor<i32> + SimdVisitor4 + SimdVisitor8 + SimdVisitor16 + HarnessVisitor
{
    // TODO: k-set skewed intersect
    let intersect_kset = FesiaKSetMethod::SimilarSize;

    use harness::time_fesia;

    Timer {
        twoset: Some(Box::new(move |warmup, a, b|
            time_fesia::<H, S, LANES, V>(warmup, a, b, hash_scale, intersect_method, simd_type))),
        kset: Some(Box::new(move |warmup, sets|
            time_fesia_kset::<H, S, LANES, V>(warmup, sets, hash_scale, intersect_kset)))
    }
}

