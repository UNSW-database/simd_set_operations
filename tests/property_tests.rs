#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{
    DualIntersectFn, SortedSet, SetCollection,
    properties::prop_intersection_correct
};

use setops::intersect;


quickcheck! {
    fn same_as_naive_merge(
        intersect: DualIntersectFn,
        set_a: SortedSet,
        set_b: SortedSet) -> bool
    {
        let expected = intersect::run_2set(
            set_a.as_slice(),
            set_b.as_slice(),
            intersect::naive_merge);

        let actual = intersect::run_2set(
            set_a.as_slice(),
            set_b.as_slice(),
            intersect.intersect);

        actual == expected
    }

    fn svs_correct(
        intersect: DualIntersectFn,
        sets: SetCollection) -> bool
    {
        let result = intersect::run_svs_generic(sets.as_slice(), intersect.intersect);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn adaptive_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::adaptive);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn small_adaptive_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::small_adaptive);
        prop_intersection_correct(result, sets.as_slice())
    }

    fn small_adaptive_sorted_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.as_slice(), intersect::small_adaptive_sorted);
        prop_intersection_correct(result, sets.as_slice())
    }

    #[cfg(feature = "simd")]
    fn simd_shuffling_correct(set_a: SortedSet, set_b: SortedSet) -> bool {
        let result = intersect::run_2set(
            set_a.as_slice(), set_b.as_slice(), intersect::simd_shuffling);
        prop_intersection_correct(result, &[set_a.as_slice(), set_b.as_slice()])
    }
}
