/// Adaptive set intersection algorithms.

use crate::{intersect::search::galloping_intersect_inplace, visitor::Visitor};

use super::search;


/// "Small vs. Small" adaptive set intersection algorithm.
/// Assumes input sets are ordered from smallest to largest.
pub fn svs<T>(sets: &[&[T]], out: &mut [T]) -> usize
where
    T: Ord + Copy,
{
    assert!(sets.len() >= 2);

    let mut count = 0;

    // Copies smallest set into (temporary) output set.
    // Is there a better way to do this?
    out[..sets[0].len()].clone_from_slice(sets[0]);

    for set in sets.iter().skip(1) {
        count = galloping_intersect_inplace(&mut out[..count], set);
    }
    count
}

pub fn svs_inplace<T>(sets: &mut [&mut [T]]) -> usize
where
    T: Ord + Copy,
{
    assert!(sets.len() >= 2);

    let mut count = 0;
    let mut iter = sets.iter_mut();

    let first = unsafe { iter.next().unwrap_unchecked() };

    for set in iter {
        count = galloping_intersect_inplace(&mut first[0..count], set);
    }
    count
}


pub fn baezayates<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V) -> usize
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mid_index = small_set.len() / 2;
    let mid_value = small_set[mid_index];

    let search_index = search::binary_search(large_set, mid_value, 0, large_set.len(), );


}

