#![feature(portable_simd)]

#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{
    DualIntersectFn, SortedSet, SetCollection,
    properties::prop_intersection_correct,
    SimilarSetPair, SkewedSetPair,
};
use setops::{
    intersect::{self, fesia::*, Intersect2}, bsr::BsrVec, Set,
    visitor::{VecWriter, UnsafeLookupWriter, EnsureVisitor, EnsureVisitorBsr, Counter},
};
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use setops::visitor::UnsafeCompressWriter;

use FesiaTwoSetMethod::*;
use SimdType::*;

quickcheck! {
    fn same_as_naive_merge(
        intersect: DualIntersectFn,
        sets: SimilarSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect.1);

        actual == expected
    }

    fn same_as_naive_merge_skewed(
        intersect: DualIntersectFn,
        sets: SkewedSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect.1);

        actual == expected
    }

    fn galloping_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::galloping);

        actual == expected
    }

    fn binary_search_intersect_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::binary_search_intersect);

        actual == expected
    }

    fn branchless_merge_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual =
            intersect::run_2set_bsr(left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr)
            .to_sorted_set();

        actual == expected
    }

    // K-set
    fn svs_correct(
        intersect: DualIntersectFn,
        sets: SetCollection<i32>) -> bool
    {
        let result = intersect::run_svs(sets.as_slice(), intersect.1);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn small_adaptive_correct(sets: SetCollection<i32>) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::small_adaptive);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn small_adaptive_sorted_correct(sets: SetCollection<i32>) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::small_adaptive_sorted);
        prop_intersection_correct(result, sets.as_slice())
    }

    // SIMD Shuffling
    #[cfg(feature = "simd")]
    fn shuffling_sse_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn shuffling_avx2_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_avx2);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn shuffling_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::shuffling_avx512);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn shuffling_sse_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::shuffling_sse_bsr(left.bsr_ref(), right.bsr_ref(), &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn shuffling_avx2_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());
        intersect::shuffling_avx2_bsr(left.bsr_ref(), right.bsr_ref(), &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn shuffling_avx512_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);
        
        let actual = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::shuffling_avx512_bsr);

        actual == expected
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn broadcast_avx512_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);
        
        let actual = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::broadcast_avx512_bsr);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn broadcast_sse_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::broadcast_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn broadcast_avx2_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::broadcast_avx2);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn broadcast_avx512_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::broadcast_avx512);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(feature = "simd")]
    fn broadcast_sse_u32_correct(set_a: SortedSet<u32>, set_b: SortedSet<u32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::broadcast_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    // LBK
    #[cfg(feature = "simd")]
    fn lbk_v1_sse_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let v1x4 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x4_sse);

        let v1x8 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x8_sse);

        v1x4 == expected && v1x8 == expected
    }

    #[cfg(feature = "simd")]
    fn lbk_v3_sse_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v3_sse);

        actual == expected
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn lbk_v3_avx2_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v3_avx2);

        actual == expected
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn lbk_v3_avx512_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v3_avx512);

        actual == expected
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn lbk_v1_avx2_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let v1x8 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x8_avx2);

        let v1x16 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x16_avx2);

        v1x8 == expected &&
        v1x16 == expected
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn lbk_v1_avx512_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let v1x16 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x16_avx512);

        let v1x32 = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::lbk_v1x32_avx512);

        v1x16 == expected &&
        v1x32 == expected
    }


    // SIMD Galloping
    #[cfg(feature = "simd")]
    fn galloping_sse_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::galloping_sse);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn galloping_avx2_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::galloping_avx2);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn galloping_avx512_correct(sets: SkewedSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::galloping_avx512);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn galloping_sse_bsr_correct(sets: SkewedSetPair<u32>) -> bool {
        let small = BsrVec::from_sorted(sets.small.as_ref());
        let large = BsrVec::from_sorted(sets.large.as_ref());

        let expected = intersect::run_2set_bsr(
            small.bsr_ref(), large.bsr_ref(), intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::galloping_sse_bsr(small.bsr_ref(), large.bsr_ref(), &mut ensurer);
        ensurer.position() == expected.len()
    }

    fn galloping_bsr_correct(sets: SkewedSetPair<u32>) -> bool {
        let small = BsrVec::from_sorted(sets.small.as_ref());
        let large = BsrVec::from_sorted(sets.large.as_ref());

        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual =
            intersect::run_2set_bsr(small.bsr_ref(), large.bsr_ref(), intersect::galloping_bsr)
            .to_sorted_set();

        actual == expected
    }

    // BMiss
    #[cfg(feature = "simd")]
    fn bmiss_scalar_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let x3 = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss_scalar_3x);

        let x4 = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss_scalar_4x);

        x3 == expected && x4 == expected
    }

    #[cfg(feature = "simd")]
    fn bmiss_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn bmiss_sttni_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss_sttni);

        actual == expected
    }

    // QFilter
    #[cfg(feature = "simd")]
    fn qfilter_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::qfilter);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn qfilter_v1_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::qfilter_v1);

        actual == expected
    }

    // fn qfilter_c_correct(sets: SimilarSetPair<i32>) -> bool {
    //     let expected = intersect::run_2set(
    //         sets.0.as_slice(),
    //         sets.1.as_slice(),
    //         intersect::naive_merge);

    //     let actual = intersect::run_2set_c(
    //         sets.0.as_slice(),
    //         sets.1.as_slice(),
    //         intersect::qfilter_c);

    //     actual == expected
    // }

    #[cfg(feature = "simd")]
    fn qfilter_ensure(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let mut ensurer = EnsureVisitor::<i32>::from(expected.as_slice());
        intersect::qfilter(sets.0.as_slice(), sets.1.as_slice(), &mut ensurer);

        ensurer.position() == expected.len()
    }

    #[cfg(feature = "simd")]
    fn qfilter_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            left.bsr_ref(), right.bsr_ref(), intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::qfilter_bsr(left.bsr_ref(), right.bsr_ref(), &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(feature = "simd")]
    fn qfilter_bsr_counter_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let mut expected = Counter::new();
        intersect::naive_merge(sets.0.as_slice(), sets.1.as_slice(), &mut expected);

        let mut actual = Counter::new();
        intersect::qfilter_bsr(left.bsr_ref(), right.bsr_ref(), &mut actual);

        actual.count() == expected.count()
    }

    // Misc AVX-512
    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn vp2intersect_emulation_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let mut ensurer = EnsureVisitor::from(expected.as_slice());

        intersect::vp2intersect_emulation(
            sets.0.as_slice(),
            sets.1.as_slice(),
            &mut ensurer);

        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn conflict_intersect_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let mut ensurer = EnsureVisitor::from(expected.as_slice());

        intersect::conflict_intersect(
            sets.0.as_slice(),
            sets.1.as_slice(),
            &mut ensurer);

        ensurer.position() == expected.len()
    }

    // FESIA
    #[cfg(feature = "simd")]
    fn fesia8_sse_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia8Sse>(set_a, set_b, hash_scale, SimilarSize, Sse)
        })
    }
    #[cfg(feature = "simd")]
    fn fesia8_sse_skewed_correct(sets: SkewedSetPair<i32>) -> bool {
        let small = sets.small.as_slice();
        let large = sets.large.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia8Sse>(small, large, hash_scale, SimilarSize, Sse)
        })
    }
    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    fn fesia16_sse_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia16Sse>(set_a, set_b, hash_scale, SimilarSize, Sse)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    fn fesia32_sse_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia32Sse>(set_a, set_b, hash_scale, SimilarSize, Sse)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn fesia8_avx2_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia8Avx2>(set_a, set_b, hash_scale, SimilarSize, Avx2)
        })
    }
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn fesia16_avx2_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia16Avx2>(set_a, set_b, hash_scale, SimilarSize, Avx2)
        })
    }
    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn fesia32_avx2_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia32Avx2>(set_a, set_b, hash_scale, SimilarSize, Avx2)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn fesia8_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia8Avx512>(set_a, set_b, hash_scale, SimilarSize, Avx512)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn fesia16_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia16Avx512>(set_a, set_b, hash_scale, SimilarSize, Avx512)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn fesia32_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let set_a = sets.0.as_slice();
        let set_b = sets.1.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia32Avx512>(set_a, set_b, hash_scale, SimilarSize, Avx512)
        })
    }

    #[cfg(feature = "simd")]
    fn fesia_hash_correct(sets: SkewedSetPair<i32>) -> bool {
        let small = sets.small.as_slice();
        let large = sets.large.as_slice();
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_correct::<Fesia8Sse>(small, large, hash_scale, Skewed, Sse) &&
            fesia_correct::<Fesia16Sse>(small, large, hash_scale, Skewed, Sse) &&
            fesia_correct::<Fesia32Sse>(small, large, hash_scale, Skewed, Sse)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "ssse3"))]
    fn fesia_kset_sse_correct(sets: SetCollection<i32>) -> bool {
        let mut sets: Vec<SortedSet<i32>> = sets.into();
        sets.sort_by_key(|s| s.as_slice().len());
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_kset_correct::<Fesia8Sse>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia16Sse>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia32Sse>(sets.as_slice(), hash_scale)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn fesia_kset_avx2_correct(sets: SetCollection<i32>) -> bool {
        let mut sets: Vec<SortedSet<i32>> = sets.into();
        sets.sort_by_key(|s| s.as_slice().len());
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_kset_correct::<Fesia8Avx2>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia16Avx2>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia32Avx2>(sets.as_slice(), hash_scale)
        })
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn fesia_kset_avx512_correct(sets: SetCollection<i32>) -> bool {
        let mut sets: Vec<SortedSet<i32>> = sets.into();
        sets.sort_by_key(|s| s.as_slice().len());
        (0..10).map(|h| h as f64 * 2.0).all(|hash_scale| {
            fesia_kset_correct::<Fesia8Avx512>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia16Avx512>(sets.as_slice(), hash_scale) &&
            fesia_kset_correct::<Fesia32Avx512>(sets.as_slice(), hash_scale)
        })
    }

    fn merge_k_correct(sets: SetCollection<i32>) -> bool {
        let mut visitor: VecWriter<i32> = VecWriter::new();
        intersect::fesia::merge_k(sets.as_slice().iter().map(|s| s.as_slice()), &mut visitor);

        prop_intersection_correct(visitor.into(), sets.as_slice())
    }
    // TODO: test FESIA k-set
    // then benchmark

    // Misc
    fn bsr_encode_decode(set: SortedSet<u32>) -> bool {
        set.as_ref() == BsrVec::from_sorted(set.as_ref()).to_sorted_set()
    }

    // Unsafe writer
    #[cfg(feature = "simd")]
    fn unsafe_lookup_writer_sse_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = run_unsafe_lookup_writer(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn unsafe_lookup_writer_avx2_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = run_unsafe_lookup_writer(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_avx2);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn unsafe_lookup_writer_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = run_unsafe_lookup_writer(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = run_unsafe_lookup_writer(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::shuffling_avx512);

        actual == expected
    }

    fn unsafe_compress_writer_sse_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = run_unsafe_compress_writer(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn unsafe_compress_writer_avx2_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = run_unsafe_compress_writer(
            set_a.as_slice(), set_b.as_slice(), intersect::shuffling_avx2);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn unsafe_compress_writer_avx512_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = run_unsafe_compress_writer(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = run_unsafe_compress_writer(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::shuffling_avx512);

        actual == expected
    }
}

fn run_unsafe_lookup_writer<T>(
    set_a: &[T],
    set_b: &[T],
    intersect: Intersect2<[T], UnsafeLookupWriter<T>>) -> Vec<T>
{
    let mut writer: UnsafeLookupWriter<T> = UnsafeLookupWriter::with_capacity(set_a.len().min(set_b.len()));
    intersect(set_a, set_b, &mut writer);
    writer.into()
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
fn run_unsafe_compress_writer<T>(
    set_a: &[T],
    set_b: &[T],
    intersect: Intersect2<[T], UnsafeCompressWriter<T>>) -> Vec<T>
{
    let mut writer: UnsafeCompressWriter<T> = UnsafeCompressWriter::with_capacity(set_a.len().min(set_b.len()));
    intersect(set_a, set_b, &mut writer);
    writer.into()
}

#[cfg(feature = "simd")]
fn fesia_correct<S>(
    set_a: &[i32],
    set_b: &[i32],
    hash_scale: HashScale,
    intersect_method: FesiaTwoSetMethod,
    simd_type: SimdType) -> bool
where
    S: SetWithHashScale + FesiaIntersect
{
    let expected = intersect::run_2set(
        set_a, set_b, intersect::naive_merge);

    let set1 = S::from_sorted(set_a, hash_scale);
    let set2 = S::from_sorted(set_b, hash_scale);
    let mut visitor: VecWriter<i32> = VecWriter::new();

    match (intersect_method, simd_type) {
        #[cfg(target_feature = "ssse3")]
        (SimilarSize, Sse) => {
            set1.intersect::<VecWriter<i32>, SegmentIntersectSse>(&set2, &mut visitor);
        }
        #[cfg(target_feature = "avx2")]
        (SimilarSize, Avx2) => {
            set1.intersect::<VecWriter<i32>, SegmentIntersectAvx2>(&set2, &mut visitor);
        }
        #[cfg(target_feature = "avx512f")]
        (SimilarSize, Avx512) => {
            set1.intersect::<VecWriter<i32>, SegmentIntersectAvx512>(&set2, &mut visitor);
        }
        #[allow(unreachable_patterns)]
        (SimilarSize, _) =>
            panic!("fesia SimilarSize does not yet support avx512"),
        (Skewed, _) =>
            set1.hash_intersect(&set2, &mut visitor),
    };

    let mut actual: Vec<i32> = visitor.into();
    actual.sort();
    actual == expected
}

#[cfg(feature = "simd")]
fn fesia_kset_correct<S>(
    sets: &[SortedSet<i32>],
    hash_scale: HashScale) -> bool
where
    S: SetWithHashScale + FesiaIntersect + AsRef<S>
{

    let expected = intersect::run_svs(sets, intersect::naive_merge);

    let fesia_sets: Vec<S> = sets.iter().map(|s| S::from_sorted(s.as_slice(), hash_scale)).collect();

    let mut visitor: VecWriter<i32> = VecWriter::new();

    S::intersect_k(fesia_sets.as_slice(), &mut visitor);

    let mut actual: Vec<i32> = visitor.into();
    actual.sort();
    actual == expected
}
