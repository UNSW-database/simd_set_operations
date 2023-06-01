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

        len_naive == len_other && outputs[0] == outputs[1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intersect1() {
        test_intersect(&[1,2,3,4], &[1,2,3,4,5], &[1,2,3,4]);
    }

    #[test]
    fn test_intersect2() {
        test_intersect(&[0,4,5,8], &[1,2,3,6], &[]);
    }

    #[test]
    fn test_intersect3() {
        test_intersect(&[1,4,5], &[1,4,5], &[1,4,5]);
    }

    #[test]
    fn test_intersect4() {
        test_intersect(&[10,42],
            &[1,2,3,4,5,6,7,8,9,10,22,25,28,39,42,43,47,49], &[10,42]);
    }

    #[test]
    fn test_intersect5() {
        test_intersect(&[
            1, 14551737, 308423503, 417273473, 731394076, 764331843, 816111760,
            1689942455, 1761460264, 1836814004, 1854053547, 2082830231,
            2295305143, 2318016244, 2404898535, 2523638113, 2539394728,
            2662474414, 2840961257, 2882274791, 3026977316, 3271982067,
            3316245172, 3483020463, 3635510430, 3699103118, 3997987522,
            4004771302, 4022120072, 4217692208, 4294967295,
        ],
        &[0, 1, 4294967295,], &[1,4294967295]);
    }

    fn test_intersect(a: &[u32], b: &[u32], out: &[u32]) {
        let mut writer = VecWriter::preallocate(out.len());

        intersect::baezayates(a, b, &mut writer);

        let result: Vec<u32> = writer.into();

        assert!(result == out);
    }
}
