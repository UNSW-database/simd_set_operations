#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{
    DualIntersectFn, SortedSet, SetCollection,
    properties::prop_intersection_correct
};

use setops::{
    intersect,
    visitor::VecWriter,
};


quickcheck! {
    fn same_as_naive_merge(
        intersect: DualIntersectFn,
        set_a: SortedSet,
        set_b: SortedSet) -> bool
    {
        let result_len =
            usize::min(set_a.cardinality(), set_b.cardinality());

        let mut writers: [VecWriter<u32>; 2] = [
            VecWriter::with_capacity(result_len),
            VecWriter::with_capacity(result_len),
        ];

        intersect::naive_merge(set_a.as_slice(), set_b.as_slice(), &mut writers[0]);
        (intersect.intersect)(set_a.as_slice(), set_b.as_slice(), &mut writers[1]);

        let outputs: [Vec<u32>; 2] = writers.map(Into::<Vec<u32>>::into);

        outputs[0] == outputs[1]
    }

    fn svs_correct(
        intersect: DualIntersectFn,
        sets: SetCollection) -> bool
    {
        let result = intersect::run_svs_generic(sets.sets(), intersect.intersect);
        prop_intersection_correct(result, sets)
    }

    fn adaptive_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.sets(), intersect::adaptive);
        prop_intersection_correct(result, sets)
    }

    fn small_adaptive_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.sets(), intersect::small_adaptive);
        prop_intersection_correct(result, sets)
    }

    fn small_adaptive_sorted_correct(sets: SetCollection) -> bool {
        let result = intersect::run_kset(sets.sets(), intersect::small_adaptive_sorted);
        prop_intersection_correct(result, sets)
    }
}
