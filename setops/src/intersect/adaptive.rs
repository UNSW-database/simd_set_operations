/// Adaptive set intersection algorithms.
// Some of these implementations are inspired by works by Daniel Lemire:
// https://github.com/lemire/SIMDCompressionAndIntersection
// https://github.com/lemire/SIMDIntersections

use std::fmt::{Display, Debug};

use smallvec::{SmallVec, smallvec};

use crate::{
    intersect::{galloping::binary_search, branchless_merge},
    visitor::Visitor,
};

/// Recursively intersects the two sets.
// Baeza-Yates, R., & Salinger, A. (2010, April). Fast Intersection Algorithms
// for Sorted Sequences. In Algorithms and Applications (pp. 45-61).
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

    let large_partition = binary_search(large_set, target, 0, large_set.len()-1);

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

/// Recursively intersects the two sets.
// Baeza-Yates, R., & Salinger, A. (2010, April). Fast Intersection Algorithms
// for Sorted Sequences. In Algorithms and Applications (pp. 45-61).
pub fn baezayates_opt<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    if small_set.is_empty() || large_set.is_empty() {
        return;
    }

    const LARGE_MAX: usize = 512;
    if large_set.len() < LARGE_MAX {
        return branchless_merge(small_set, large_set, visitor);
    }

    if small_set.len() > large_set.len() {
        return baezayates(large_set, small_set, visitor);
    }

    let small_partition = small_set.len() / 2;
    let target = small_set[small_partition];

    let large_partition = binary_search(large_set, target, 0, large_set.len()-1);

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

pub fn adaptive<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy + Display + Debug,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    assert!(sets.len() >= 2);
    debug_assert!(
        sets.iter().all(|set| set.as_ref().windows(2).all(|w| w[0] < w[1]))
    );

    if sets.iter().any(|set| set.as_ref().is_empty()) {
        return;
    }

    // TODO: check if this optimisation is meaningful
    let mut positions_vec: SmallVec<[usize; 8]> = smallvec![0; sets.len()];
    let positions = &mut positions_vec[..];

    let mut elim_set_idx = 0;
    let mut elim_value = sets[0].as_ref()[0];
    let mut curr_set_idx = 1;
    let mut gallop_size = 1;

    loop {
        let elim_set = sets[elim_set_idx].as_ref();
        let curr_set = sets[curr_set_idx].as_ref();
        let curr_position = positions[curr_set_idx];

        if curr_set[curr_position + gallop_size] >= elim_value {
            let search_result = binary_search(
                curr_set, elim_value, curr_position, curr_position + gallop_size);

            positions[curr_set_idx] = search_result;
            if curr_set[search_result] == elim_value {
                // Found
                positions[curr_set_idx] += 1;
                curr_set_idx = (curr_set_idx + 1) % sets.len();

                if curr_set_idx == elim_set_idx {
                    // Found last occurrence
                    visitor.visit(elim_value);

                    if positions[elim_set_idx] == elim_set.len() - 1 {
                        break;
                    }

                    // Choose next eliminator
                    positions[elim_set_idx] += 1;
                    elim_value = elim_set[positions[elim_set_idx]];

                    curr_set_idx = (curr_set_idx + 1) % sets.len();
                }
            }
            else {
                // Not found
                elim_value = curr_set[search_result];
                positions[elim_set_idx] += 1;
                elim_set_idx = curr_set_idx;

                curr_set_idx = (curr_set_idx + 1) % sets.len();
            }

            // Set gallop_size to 0 to compare the
            // last element of the current set.
            let curr_set_len = sets[curr_set_idx].as_ref().len();
            match (positions[curr_set_idx] + 1).cmp(&curr_set_len) {
                std::cmp::Ordering::Less    => gallop_size = 1,
                std::cmp::Ordering::Equal   => gallop_size = 0,
                std::cmp::Ordering::Greater => break,
            };
            continue;
        }
        else if curr_set[curr_set.len()-1] < elim_value {
            break;
        }

        // Update gallop size
        gallop_size = if positions[curr_set_idx] + gallop_size*2 < curr_set.len() {
            gallop_size * 2
        }
        else {
            curr_set.len() - positions[curr_set_idx] - 1
        }
    }
}

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

            let lo = base;
            let hi = (set.len() - 1).min(base + offset);

            let new_base = binary_search(set, element, lo, hi);

            positions[i] = new_base;

            if new_base >= set.len() || set[new_base] != element {
                continue 'outer;
            }
        }
        visitor.visit(element);
    }
}

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
            let hi = (set.len() - 1).min(offset);

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
