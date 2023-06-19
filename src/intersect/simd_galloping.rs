#![cfg(feature = "simd")]

use std::simd::*;

use crate::{visitor::{VecWriter, Visitor}, intersect, instructions::load};

const LANES: usize = 4;
const SEARCH_SIZE: usize = 4;
const COMPARE_SIZE: usize = 8;

const BOUND_VEC: usize = SEARCH_SIZE * COMPARE_SIZE;
const BOUND: usize = BOUND_VEC * LANES;

/// SIMD Galloping algorithm by D. Lemire et al.
/// 
/// Extends the classical galloping algorithm by performing comparisons of
/// blocks of 8 4xi32 registers, placing results in bitmasks Q1, Q2, Q3, Q4
/// where each Q is the result of a pairwise comparison between two SIMD
/// vectors. The galloping stage bounds in leaps of 4x8 SIMD registers = 32x4
/// integers, then performs a mini binary search to narrow it down to a block of
/// 8 registers.
#[inline(never)]
pub fn simd_galloping<'a, V>(mut small: &'a[i32], mut large: &'a[i32], visitor: &mut V)
where
    V: Visitor<i32>,
{
    if small.len() > large.len() {
        return simd_galloping(large, small, visitor);
    }

    if large.len() < BOUND {
        return intersect::branchless_merge(small, large, visitor);
    }

    while !small.is_empty() && large.len() >= BOUND {
        let target = small[0];

        let upper_bound = if large[BOUND - 1] >= target {
            0
        }
        else {
            let mut offset = 1;
            while (offset + 1) * BOUND - 1 < large.len()
                && large[(offset + 1) * BOUND - 1] < target
            {
                offset *= 2;
            }
            offset
        };

        let lo = upper_bound / 2;
        let hi = (large.len() / BOUND - 1).min(upper_bound);

        let target_block = binary_search_wide(target, large, lo, hi);

        // Check if block actually might contain target.
        if large[(target_block + 1) * BOUND - 1] < target {
            // If not, shrink large.
            large = &large[(target_block + 1) * BOUND..];

            debug_assert!(large.len() < BOUND);
            // Swap small and large if small is big enough.
            if small.len() >= BOUND {
                (small, large) = (&large[..], &small[..]);
                continue;
            }
            else {
                break;
            }
        }

        debug_assert!(target_block == 0 || large[target_block * BOUND - 1] <= target);
        debug_assert!(large[(target_block+1) * BOUND - 1] >= target);

        large = &large[target_block * BOUND..];
        debug_assert!(large.len() >= BOUND);

        // Check if target appears in range large[0..128]
        let search_offset: usize =
        if large[LANES * 16 - 1] >= target {
            if large[LANES * 8 - 1] < target { 8 } else { 0 }
        }
        else {
            if large[LANES * 24 - 1] < target { 24 } else { 16 }
        };

        let target_vec = i32x4::splat(target);
        let qs = [
            target_vec.simd_eq(load(&large[LANES * (search_offset + 0)..])) |
            target_vec.simd_eq(load(&large[LANES * (search_offset + 1)..])),
            target_vec.simd_eq(load(&large[LANES * (search_offset + 2)..])) |
            target_vec.simd_eq(load(&large[LANES * (search_offset + 3)..])),
            target_vec.simd_eq(load(&large[LANES * (search_offset + 4)..])) |
            target_vec.simd_eq(load(&large[LANES * (search_offset + 5)..])),
            target_vec.simd_eq(load(&large[LANES * (search_offset + 6)..])) |
            target_vec.simd_eq(load(&large[LANES * (search_offset + 7)..]))
        ];

        let result = (qs[0] | qs[1]) | (qs[2] | qs[3]);
        if result.any() {
            visitor.visit(target);
        }

        small = &small[1..];
    }

    debug_assert!(small.is_empty() || large.len() < BOUND);
    intersect::branchless_merge(small, large, visitor)
}

pub fn binary_search_wide(target: i32, large: &[i32], low: usize, high: usize) -> usize {
    let mut lo = low as isize;
    let mut hi = high as isize;

    // Trying to find the block index such that the last element in the block
    // is greater than or equal to the target value
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if large[(mid as usize + 1) * BOUND - 1] < target {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    assert!(lo == hi);
    lo as usize
}

pub fn simd_galloping_mono(small: &[i32], large: &[i32], visitor: &mut VecWriter<i32>) {
    simd_galloping(small, large, visitor);
}
