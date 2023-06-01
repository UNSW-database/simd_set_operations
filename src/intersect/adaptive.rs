/// Adaptive set intersection algorithms.

use crate::{
    intersect::search::{galloping_inplace, binary_search},
    visitor::Visitor,
};


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
        count = galloping_inplace(&mut out[..count], set);
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
        count = galloping_inplace(&mut first[0..count], set);
    }
    count
}

/// Recursively intersects the two sets.
// Baeza-Yates, R., & Salinger, A. (2010, April). Fast Intersection Algorithms
// for Sorted Sequences. In Algorithms and Applications (pp. 45-61).
pub fn baezayates<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V) -> usize
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    if small_set.len() == 0 || large_set.len() == 0 {
        return 0;
    }

    if small_set.len() > large_set.len() {
        return baezayates(large_set, small_set, visitor);
    }

    let small_partition = small_set.len() / 2;
    let target = small_set[small_partition];
    let mut count = 0;

    let large_partition = binary_search(large_set, target, 0, large_set.len()-1);

    count += baezayates(&small_set[..small_partition],
                        &large_set[..large_partition], visitor);

    if large_partition >= large_set.len() {
        return count;
    }

    if large_set[large_partition] == target {
        visitor.visit(target);
        count += 1;
    }

    count + baezayates(&small_set[small_partition+1..],
                       &large_set[large_partition..], visitor)
}
