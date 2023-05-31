#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{DualIntersectFn, SortedSet};

use setops::{
    intersect,
    visitor::{VecWriter, Visitor},
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
            VecWriter::preallocate(result_len),
            VecWriter::preallocate(result_len),
        ];

        let len_naive = intersect::naive_merge(
            set_a.as_slice(), set_b.as_slice(), &mut writers[0]);
        let len_other = (intersect.intersect)(
            set_a.as_slice(), set_b.as_slice(), &mut writers[1]);

        let outputs: [Vec<u32>; 2] = writers.map(Into::<Vec<u32>>::into);

        outputs[0][0..len_naive] == outputs[1][0..len_other]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gallop() {
        let mut writer = VecWriter::preallocate(4);

        intersect::galloping(
            [1,2,3,4].as_slice(),
            [1,2,3,4,5].as_slice(),
            &mut writer);
    }
}
