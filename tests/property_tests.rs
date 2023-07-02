#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{
    DualIntersectFn, SortedSet, SetCollection,
    properties::prop_intersection_correct,
    SimilarSetPair, SkewedSetPair,
};
use setops::{
    intersect::{self, Fesia, MixHash}, bsr::BsrVec, Set,
    visitor::{VecWriter, EnsureVisitor, EnsureVisitorBsr, Counter},
    
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

    fn branchless_merge_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let actual =
            intersect::run_2set_bsr(&left, &right, intersect::branchless_merge_bsr)
            .to_sorted_set();

        actual == expected
    }

    // K-set
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
            &left, &right, intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::shuffling_sse_bsr(&left, &right, &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx2"))]
    fn shuffling_avx2_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            &left, &right, intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());
        intersect::shuffling_avx2_bsr(&left, &right, &mut ensurer);
        ensurer.position() == expected.len()
    }

    #[cfg(all(feature = "simd", target_feature = "avx512f"))]
    fn shuffling_avx512_bsr_correct(sets: SimilarSetPair<u32>) -> bool {
        let left = BsrVec::from_sorted(sets.0.as_ref());
        let right = BsrVec::from_sorted(sets.1.as_ref());

        let expected = intersect::run_2set_bsr(
            &left, &right, intersect::branchless_merge_bsr);
        
        let actual = intersect::run_2set_bsr(
            &left, &right, intersect::shuffling_avx512_bsr);

        actual == expected
    }

    #[cfg(feature = "simd")]
    fn broadcast_sse_correct(set_a: SortedSet<i32>, set_b: SortedSet<i32>) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::broadcast_sse);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
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
            &small, &large, intersect::branchless_merge_bsr);

        let mut ensurer = EnsureVisitorBsr::from(expected.bsr_ref());

        intersect::galloping_sse_bsr(&small, &large, &mut ensurer);
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
            intersect::run_2set_bsr(&small, &large, intersect::galloping_bsr)
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

        let actual_sttni = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::bmiss_sttni);

        actual == expected && actual_sttni == expected
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

        let actual_v1 = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::qfilter_v1);

        actual == expected && actual_v1 == expected
    }

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
            &left, &right, intersect::branchless_merge_bsr);

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
    fn fesia_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let fesia0: Fesia<MixHash, 4> = Fesia::from_sorted(sets.0.as_slice());
        let fesia1: Fesia<MixHash, 4> = Fesia::from_sorted(sets.1.as_slice());
        let mut visitor: VecWriter<i32> = VecWriter::new();

        intersect::fesia_sse(
            fesia0.as_view(),
            fesia1.as_view(),
            &mut visitor);

        let mut actual: Vec<i32> = visitor.into();
        actual.sort();
        actual == expected
    }

    #[cfg(feature = "simd")]
    fn fesia_shuffling_correct(sets: SimilarSetPair<i32>) -> bool {
        let expected = intersect::run_2set(
            sets.0.as_slice(),
            sets.1.as_slice(),
            intersect::naive_merge);

        let fesia0: Fesia<MixHash, 4> = Fesia::from_sorted(sets.0.as_slice());
        let fesia1: Fesia<MixHash, 4> = Fesia::from_sorted(sets.1.as_slice());
        let mut visitor: VecWriter<i32> = VecWriter::new();

        intersect::fesia_sse_shuffling(
            fesia0.as_view(),
            fesia1.as_view(),
            &mut visitor);

        let mut actual: Vec<i32> = visitor.into();
        actual.sort();
        actual == expected
    }

    // Misc
    fn bsr_encode_decode(set: SortedSet<u32>) -> bool {
        set.as_ref() == BsrVec::from_sorted(set.as_ref()).to_sorted_set()
    }
}
