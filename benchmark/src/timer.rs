pub mod harness;

use setops::{
    intersect::{self, Intersect2, IntersectK},
    visitor::{
        VecWriter, Visitor, Counter,
        SimdVisitor4, SimdVisitor8, SimdVisitor16
    },
    bsr::Intersect2Bsr,
};
use crate::datafile::DatafileSet;
use harness::{Harness, HarnessVisitor, TimeResult};

type TwosetTimer = Box<dyn Fn(&Harness, &[i32], &[i32]) -> TimeResult>;
type KsetTimer = Box<dyn Fn(&Harness, &[DatafileSet]) -> TimeResult>;

pub struct Timer {
    twoset: Option<TwosetTimer>,
    kset: Option<KsetTimer>,
}

impl Timer {
    pub fn new(name: &str, count_only: bool) -> Option<Self>
    {
        if count_only {
            Self::make::<Counter>(name, count_only)
        }
        else {
            Self::make::<VecWriter<i32>>(name, count_only)
        }
    }

    fn make<V>(name: &str, count_only: bool) -> Option<Self>
    where
        V: Visitor<i32> + HarnessVisitor + TwosetTimingSpec<V>,
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + 'static
    {
        try_parse_twoset::<V>(name)
            .or_else(|| try_parse_bsr(name))
            .or_else(|| try_parse_kset::<V>(name))
            .or_else(|| try_parse_roaring(name, count_only))
            .or_else(|| try_parse_fesia::<V>(name))
    }

    pub fn run(&self, harness: &Harness, sets: &[DatafileSet]) -> TimeResult {
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

fn try_parse_twoset<V>(name: &str) -> Option<Timer> 
where
    V: Visitor<i32> + HarnessVisitor + TwosetTimingSpec<V>,
    V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + 'static
{
    let maybe_intersect: Option<Intersect2<[i32], V>> = match name {
        "naive_merge"      => Some(intersect::naive_merge),
        "branchless_merge" => Some(intersect::branchless_merge),
        "bmiss_scalar_3x"  => Some(intersect::bmiss_scalar_3x),
        "bmiss_scalar_4x"  => Some(intersect::bmiss_scalar_4x),
        "galloping"        => Some(intersect::galloping),
        "baezayates"       => Some(intersect::baezayates),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse"    => Some(intersect::shuffling_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "broadcast_sse"    => Some(intersect::broadcast_sse),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse"        => Some(intersect::bmiss),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "bmiss_sse_sttni"  => Some(intersect::bmiss_sttni),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter"          => Some(intersect::qfilter),
        #[cfg(all(feature = "simd"))]
        "galloping_sse"    => Some(intersect::galloping_sse),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2"   => Some(intersect::shuffling_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "broadcast_avx2"   => Some(intersect::broadcast_avx2),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2"   => Some(intersect::galloping_avx2),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512"       => Some(intersect::shuffling_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "broadcast_avx512"   => Some(intersect::broadcast_avx512),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "vp2intersect_emulation" => Some(intersect::vp2intersect_emulation),
        #[cfg(all(feature = "simd", target_feature = "avx512cd"))]
        "conflict_intersect"     => Some(intersect::conflict_intersect),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512"       => Some(intersect::galloping_avx512),
        _ => None,

    };
    maybe_intersect.map(|intersect| V::twoset_timer(intersect))
}

pub trait TwosetTimingSpec<V> {
    fn twoset_timer(i: Intersect2<[i32], V>) -> Timer;
}

impl TwosetTimingSpec<VecWriter<i32>> for VecWriter<i32> {
    fn twoset_timer(i: Intersect2<[i32], VecWriter<i32>>) -> Timer {
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| harness::time_twoset(warmup, a, b, i))),
            kset: Some(Box::new(
                move |warmup, sets| harness::time_svs::<VecWriter<i32>>(warmup, sets, i))),
        }
    }
}

impl TwosetTimingSpec<Counter> for Counter {
    fn twoset_timer(i: Intersect2<[i32], Counter>) -> Timer {
        Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| harness::time_twoset(warmup, a, b, i))),
            kset: None,
        }
    }
}

fn try_parse_bsr(name: &str) -> Option<Timer> {
    let maybe_intersect: Option<Intersect2Bsr> = match name {
        "branchless_merge_bsr" => Some(intersect::branchless_merge_bsr),
        "galloping_bsr"        => Some(intersect::galloping_bsr),
        // SSE
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "shuffling_sse_bsr"    => Some(intersect::shuffling_sse_bsr),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "qfilter_bsr"          => Some(intersect::qfilter_bsr),
        #[cfg(all(feature = "simd"))]
        "galloping_sse_bsr"    => Some(intersect::galloping_sse_bsr),
        // AVX2
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "shuffling_avx2_bsr"   => Some(intersect::shuffling_avx2_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "galloping_avx2_bsr"   => Some(intersect::galloping_avx2_bsr),
        // AVX-512
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "shuffling_avx512_bsr"       => Some(intersect::shuffling_avx512_bsr),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "galloping_avx512_bsr"       => Some(intersect::galloping_avx512_bsr),
        _ => None,
    };
    maybe_intersect.map(|intersect: Intersect2Bsr| Timer {
        twoset: Some(Box::new(move |warmup, a, b| harness::time_bsr(warmup, a, b, intersect))),
        kset: None,
    })
}

fn try_parse_kset<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + HarnessVisitor + TwosetTimingSpec<V>,
    V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + 'static
{
    let maybe_intersect: Option<IntersectK<DatafileSet, V>> = match name {
        "adaptive"              => Some(intersect::adaptive),
        "small_adaptive"        => Some(intersect::small_adaptive),
        "small_adaptive_sorted" => Some(intersect::small_adaptive_sorted),
        _ => None,
    };
    maybe_intersect.map(|intersect| Timer {
        twoset: None,
        kset: Some(Box::new(move |warmup, sets| harness::time_kset(warmup, sets, intersect))),
    })
}

fn try_parse_roaring(name: &str, count_only: bool) -> Option<Timer> { 
    match name {
        "croaring" => Some(Timer {
            twoset: Some(Box::new(
                move |warmup, a, b| Ok(harness::time_croaring_2set(warmup, a, b, count_only)))),
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_croaring_svs(warmup, sets))))
                },
            }),
        "roaringrs" => Some(Timer {
            twoset:
                if count_only { None } else {
                    Some(Box::new(|warmup, a, b| Ok(harness::time_roaringrs_2set(warmup, a, b))))
                },
            kset:
                if count_only { None } else {
                    Some(Box::new(|warmup, sets| Ok(harness::time_roaringrs_svs(warmup, sets))))
                },
            }),
        _ => None,
    }
}

fn try_parse_fesia<V>(name: &str) -> Option<Timer>
where
    V: Visitor<i32> + SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32> + HarnessVisitor
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
    const FESIA_SHUFFLING: &str = "fesia_shuffling";
    const FESIA_SPLAT: &str = "fesia_splat";
    const FESIA: &str = "fesia";

    use FesiaIntersectMethod::*;
    let (intersect, rest) =
        if prefix.starts_with(FESIA_HASH) {
            (Skewed, &prefix[FESIA_HASH.len()..])
        }
        else if prefix.starts_with(FESIA_SHUFFLING) {
            (SimilarSizeShuffling, &prefix[FESIA_SHUFFLING.len()..])
        }
        else if prefix.starts_with(FESIA_SPLAT) {
            (SimilarSizeSplat, &prefix[FESIA_SPLAT.len()..])
        }
        else if prefix.starts_with(FESIA) {
            (SimilarSize, &prefix[FESIA.len()..])
        }
        else {
            return None;
        };
    
    use SimdType::*;
    let simd_type =
        if rest.ends_with("sse") {
            Sse
        }
        else if rest.ends_with("avx2") {
            Avx2
        }
        else if rest.ends_with("avx512") {
            Avx512
        }
        else {
            return None;
        };

    use harness::time_fesia;

    let maybe_twoset_timer: Option<TwosetTimer> =
    match rest {
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "8_sse" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i8, u16, 16, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "16_sse" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i16, u8, 8, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "ssse3"))]
        "32_sse" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i32, u8, 4, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "8_avx2" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i8, u32, 32, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "16_avx2" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i16, u16, 16, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx2"))]
        "32_avx2" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i32, u8, 8, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "8_avx512" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i8, u64, 64, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "16_avx512" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i16, u32, 32, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        #[cfg(all(feature = "simd", target_feature = "avx512f"))]
        "32_avx512" => Some(Box::new(move |warmup, a, b|
            time_fesia::<MixHash, i32, u16, 16, V>(warmup, a, b, hash_scale, intersect, simd_type)
        )),
        _ => None,
    };

    maybe_twoset_timer.map(|twoset_timer| Timer {
        twoset: Some(twoset_timer),
        kset: None,
    })
}

