/// Adaptive set intersection algorithms.

use crate::{
    intersect::search::binary_search,
    visitor::Visitor,
};

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
