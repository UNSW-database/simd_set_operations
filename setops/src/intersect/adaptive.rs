/// Adaptive set intersection algorithms.
// Some of these implementations are inspired by works by Daniel Lemire:
// https://github.com/lemire/SIMDCompressionAndIntersection
// https://github.com/lemire/SIMDIntersections

use std::fmt::{Display, Debug};

use smallvec::{SmallVec, smallvec};

use crate::{
    intersect::galloping::binary_search,
    visitor::Visitor,
};

/// Recursively intersects the two sets.
/// Baeza-Yates, R., & Salinger, A. (2010, April). Fast Intersection Algorithms
/// for Sorted Sequences. In Algorithms and Applications (pp. 45-61).
pub fn baezayates<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    if small_set.is_empty() || large_set.is_empty() {
        return;
    }

    if small_set.len() > large_set.len() {
        return baezayates(large_set, small_set, visitor);
    }

    let small_partition = small_set.len() / 2;
    let target = small_set[small_partition];

    let large_partition = binary_search(large_set, target, 0, large_set.len() as isize - 1);

    baezayates(&small_set[..small_partition],
               &large_set[..large_partition], visitor);

    if large_partition >= large_set.len() {
        return;
    }

    if large_set[large_partition] == target {
        visitor.visit(target);
    }

    baezayates(&small_set[small_partition+1..],
               &large_set[large_partition..], visitor)
}

// Experimental extension of above algorithm into k sets. Very slow.
pub fn baezayates_k<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    debug_assert!(sets.len() >= 2);

    for set in sets {
        if set.as_ref().is_empty() {
            return;
        }
    }

    let smallest = sets[0].as_ref();

    let small_partition = smallest.len() / 2;
    let target = smallest[small_partition];

    let mut lowers: SmallVec<[&[T]; 8]> = SmallVec::new();
    let mut uppers: SmallVec<[&[T]; 8]> = SmallVec::new();

    lowers.push(&smallest[..small_partition]);
    uppers.push(&smallest[small_partition+1..]);

    let mut match_count = 0;

    for large_set in &sets[1..] {
        let large_set = large_set.as_ref();
        let large_partition = binary_search(large_set, target, 0, large_set.len() as isize - 1);

        if large_partition >= large_set.len() {
            return;
        }
        if large_set[large_partition] == target {
            match_count += 1;
        }

        lowers.push(&large_set[..large_partition]);
        uppers.push(&large_set[large_partition..]);
    }

    if match_count == sets.len() - 1 {
        visitor.visit(target);
    }

    baezayates_k(&lowers, visitor);
    baezayates_k(&uppers, visitor);
}

/// Demaine, E. D., López-Ortiz, A., & Ian Munro, J. (2001). Experiments on
/// adaptive set intersections for text retrieval systems. In Algorithm
/// Engineering and Experimentation: Third International Workshop, ALENEX 2001
/// Washington, DC, USA, January 5–6, 2001 Revised Papers 3 (pp. 91-104).
/// Springer Berlin Heidelberg.
pub fn small_adaptive<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    assert!(sets.len() >= 2);
    debug_assert!(
        sets.iter().all(|set| set.as_ref().windows(2).all(|w| w[0] < w[1]))
    );

    // TODO: check if this optimisation is meaningful
    let mut positions_vec: SmallVec<[usize; 8]> = smallvec![0; sets.len()];
    let positions = &mut positions_vec[..];

    'outer: for &element in sets[0].as_ref() {

        let other_sets = sets.iter().map(|s| s.as_ref()).enumerate().skip(1);
        for (i, set) in other_sets {

            let base = positions[i];
            let mut offset = 1;

            while base + offset < set.len() && set[base + offset] <= element {
                offset *= 2;
            }

            let lo = base as isize;
            let hi = (set.len() as isize - 1).min((base + offset) as isize);

            let new_base = binary_search(set, element, lo, hi);

            positions[i] = new_base;

            if new_base >= set.len() || set[new_base] != element {
                continue 'outer;
            }
        }
        visitor.visit(element);
    }
}

// Experiment: sort sets each iteration. Result: always slower than standard Small Adaptive.
pub fn small_adaptive_sorted<T, S, V>(given_sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    assert!(given_sets.len() >= 2);
    debug_assert!(
        given_sets.iter().all(|set| set.as_ref().windows(2).all(|w| w[0] < w[1]))
    );

    let mut sets_vec: SmallVec<[&[T]; 8]> = SmallVec::from_iter(
        given_sets.iter().map(|s| s.as_ref())
    );
    let sets = &mut sets_vec[..];

    'outer: loop {
        sets.sort_by_key(|a| a.len());

        let (first, other_sets) = sets.split_at_mut(1);

        let primary_set = &mut first[0];
        if primary_set.is_empty() {
            break;
        }

        let element = primary_set[0];

        for set in other_sets {

            let mut offset = 1;

            while offset < set.len() && set[offset] <= element {
                offset *= 2;
            }

            let lo = 0;
            let hi = (set.len() as isize - 1).min(offset as isize);

            let new_base = binary_search(set, element, lo, hi);

            if new_base >= set.len() {
                break 'outer;
            }

            *set = &set[new_base..];

            if set[0] != element {
                // Not found, start again with next element.
                *primary_set = &primary_set[1..];
                continue 'outer;
            }
        }
        *primary_set = &primary_set[1..];
        visitor.visit(element);
    }
}
