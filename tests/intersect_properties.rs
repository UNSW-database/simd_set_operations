#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{DualIntersectFn, SortedSet, SetCollection};

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

    fn svs_strictly_increasing(
        intersect: DualIntersectFn,
        sets: SetCollection) -> bool
    {
        let result = intersect::run_svs_generic(sets.sets(), intersect.intersect);

        result.windows(2).all(|w| w[0] < w[1])
    }

    fn svs_result_items_in_every_input(
        intersect: DualIntersectFn,
        sets: SetCollection) -> bool
    {
        let result = intersect::run_svs_generic(sets.sets(), intersect.intersect);

        result.iter().all(|result_item| {
            sets.sets().iter().all(|input_set| {
                input_set.as_slice().contains(&result_item)
            })
        })
    }

    fn svs_result_contains_all_common_items(
        intersect: DualIntersectFn,
        sets: SetCollection) -> bool
    {
        let result = intersect::run_svs_generic(sets.sets(), intersect.intersect);

        let raw_sets = sets.into_inner();

        for item in raw_sets[0].as_slice() {
            if raw_sets.iter().skip(1).all(|set|
                set.as_slice().contains(&item)
            ) {
                if !result.contains(&item) {
                    return false;
                }
            }
        }
        true
    }
}
