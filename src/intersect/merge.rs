use std::marker;

use super::SortedIntersect2;

pub struct NaiveMerge<T> {
    _m: marker::PhantomData<T>,
}

impl<T> SortedIntersect2<T> for NaiveMerge<T>
where
    T: Ord + Copy,
{
    fn intersect(set_a: &[T], set_b: &[T], result: &mut [T]) -> usize {
        let mut idx_a: usize = 0;
        let mut idx_b: usize = 0;
        let mut count : usize = 0;

        while idx_a < set_a.len() && idx_b < set_b.len() {
            let value_a = set_a[idx_a];
            let value_b = set_b[idx_b];
            if value_a < value_b {
                idx_a += 1;
            }
            else if value_b < value_a {
                idx_b += 1;
            }
            else {
                result[count] = value_a;
                count += 1;
                idx_a += 1;
                idx_b += 1;
            }
        }
        count
    }
}
