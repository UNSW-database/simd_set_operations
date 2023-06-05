/// Adaptive set intersection algorithms.
// Some of these implementations are inspired by works by Daniel Lemire:
// https://github.com/lemire/SIMDCompressionAndIntersection
// https://github.com/lemire/SIMDIntersections

use smallvec::{SmallVec, smallvec};

use crate::{
    intersect::search::binary_search,
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
    if small_set.len() == 0 || large_set.len() == 0 {
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

pub fn adaptive<T, S, V>(sets: &[S], visitor: &mut V)
where
    T: Ord + Copy,
    S: AsRef<[T]>,
    V: Visitor<T>,
{
    assert!(sets.len() >= 2);

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
                    // Update eliminator
                    visitor.visit(elim_value);

                    if search_result == curr_set.len() - 1 {
                        break;
                    }
                    elim_set_idx += 1;
                    elim_value = elim_set[elim_set_idx];

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
            if positions[curr_set_idx] + 1 < curr_set_len {
                gallop_size = 1;
            }
            else if positions[curr_set_idx] + 1 == curr_set_len {
                gallop_size = 0;
            }
            else {
                break;
            }
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
