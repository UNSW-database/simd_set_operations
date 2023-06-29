#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{
    DualIntersectFn, SortedSet, SetCollection,
    properties::prop_intersection_correct,
    SimilarSetPair, SkewedSetPair,
};

use setops::{
    intersect, bsr::BsrVec, Set,
    visitor::{EnsureVisitor, EnsureVisitorBsr, Counter}
};


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

    fn svs_correct(
        intersect: DualIntersectFn,
        sets: SetCollection<i32>) -> bool
    {
        let result = intersect::run_svs_generic(sets.as_slice(), intersect.1);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn adaptive_correct(sets: SetCollection<i32>) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::adaptive);
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

    #[cfg(feature = "simd")]
    fn simd_shuffling_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::simd_shuffling);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(feature = "simd")]
    fn simd_shuffling_avx2_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::simd_shuffling_avx2);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }

    #[cfg(feature = "simd")]
    fn simd_galloping_correct(sets: SkewedSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::simd_galloping);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn simd_galloping_8x_correct(sets: SkewedSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::simd_galloping_8x);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn simd_galloping_16x_correct(sets: SkewedSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::simd_galloping_16x);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn bmiss_scalar_correct(sets: SimilarSetPair<i32>) -> bool
    {
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
    fn bmiss_correct(sets: SimilarSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss);

        let actual_sttni = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss_sttni);

        actual == expected && actual_sttni == expected
    }

    fn bsr_encode_decode(set: SortedSet<u32>) -> bool {
        set.as_ref() == BsrVec::from_sorted(set.as_ref()).to_sorted_set()
    }

    fn merge_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual =
            intersect::run_2set_bsr(&left, &right, intersect::merge_bsr)
            .to_sorted_set();

        actual == expected
    }

    fn galloping_bsr_correct(sets: SkewedSetPair<u32>) -> bool {
        let small = BsrVec::from_sorted(sets.small.as_ref());
        let large = BsrVec::from_sorted(sets.large.as_ref());

        let expected = intersect::run_2set(
            sets.small.as_slice(),
            sets.large.as_slice(),
            intersect::naive_merge);

        let actual =
            intersect::run_2set_bsr(&small, &large, intersect::galloping_bsr)
            .to_sorted_set();

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn qfilter_correct(sets: SimilarSetPair<i32>) -> bool
    {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::qfilter);

        let actual_v1 = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::qfilter_v1);

        actual == expected && actual_v1 == expected
    }
    
    #[cfg(feature = "simd")]
    fn simd_ensurer_works(sets: SimilarSetPair<i32>) -> bool
    {
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
            &left, &right, intersect::merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::qfilter_bsr(&left, &right, &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(feature = "simd")]
    fn qfilter_bsr_counter_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let mut expected = Counter::new();
        intersect::naive_merge(sets.0.as_slice(), sets.1.as_slice(), &mut expected);

        let mut actual = Counter::new();
        intersect::qfilter_bsr(&left, &right, &mut actual);

        actual.count() == expected.count()
    }

    #[cfg(feature = "simd")]
    fn simd_galloping_bsr_correct(sets: SkewedSetPair<u32>) -> bool {
        let small = BsrVec::from_sorted(sets.small.as_ref());
        let large = BsrVec::from_sorted(sets.large.as_ref());

        let expected = intersect::run_2set_bsr(
            &small, &large, intersect::merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::simd_galloping_bsr(&small, &large, &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(feature = "simd")]
    fn simd_shuffling_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            &left, &right, intersect::merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::simd_shuffling_bsr(&left, &right, &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn simd_shuffling_avx512_naive_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let mut ensurer = EnsureVisitor::from(expected.as_slice());

        intersect::simd_shuffling_avx512_naive(
            sets.0.as_slice(), 
            sets.1.as_slice(),
            &mut ensurer);

        ensurer.position() == expected.len()
    }

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
}
