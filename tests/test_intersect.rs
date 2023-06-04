#[macro_use(quickcheck)]
extern crate quickcheck;
mod testlib;
use testlib::{DualIntersectFnVec, SortedSet, SetCollection};

use setops::{
    intersect,
    visitor::{VecWriter, SliceWriter},
};

use crate::testlib::DualIntersectFnSlice;

quickcheck! {
    fn same_as_naive_merge(
        intersect: DualIntersectFnVec,
        set_a: SortedSet,
        set_b: SortedSet) -> bool
    {
        let result_len =
            usize::min(set_a.cardinality(), set_b.cardinality());

        let mut writers: [VecWriter<u32>; 2] = [
            VecWriter::with_capacity(result_len),
            VecWriter::with_capacity(result_len),
        ];

        let len_naive = intersect::naive_merge(
            set_a.as_slice(), set_b.as_slice(), &mut writers[0]);
        let len_other = (intersect.intersect)(
            set_a.as_slice(), set_b.as_slice(), &mut writers[1]);

        let outputs: [Vec<u32>; 2] = writers.map(Into::<Vec<u32>>::into);

        len_naive == len_other && outputs[0] == outputs[1]
    }

    fn svs_strictly_increasing(
        intersect: DualIntersectFnSlice,
        sets: SetCollection) -> bool
    {
        let result = run_svs(&sets, intersect.intersect);

        result.windows(2).all(|w| w[0] < w[1])
    }

    fn svs_result_items_in_every_input(
        intersect: DualIntersectFnSlice,
        sets: SetCollection) -> bool
    {
        let result = run_svs(&sets, intersect.intersect);

        result.iter().all(|result_item| {
            sets.sets().iter().all(|input_set| {
                input_set.as_slice().contains(&result_item)
            })
        })
    }
    
    fn svs_result_contains_all_common_items(
        intersect: DualIntersectFnSlice,
        sets: SetCollection) -> bool
    {
        let result = run_svs(&sets, intersect.intersect);

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

fn run_svs(
    sets: &SetCollection,
    intersect: fn(&[u32], &[u32], &mut SliceWriter<u32>) -> usize) -> Vec<u32>
{
    let result_len = sets.sets().iter()
        .map(|set| set.cardinality()).max().unwrap();

    let mut out0: Vec<u32> = vec![0; result_len];
    let mut out1: Vec<u32> = vec![0; result_len];
    
    let (count, index) = intersect::svs_generic(
        &sets.sets(), &mut out0, &mut out1, intersect);
    
    match index {
        0 => { out0.truncate(count); out0 },
        1 => { out1.truncate(count); out1 },
        _ => panic!("Invalid out index!"),
    }
}

// Sanity check
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
        let mut writer = VecWriter::with_capacity(out.len());

        intersect::baezayates(a, b, &mut writer);

        let result: Vec<u32> = writer.into();

        assert!(result == out);
    }
}
