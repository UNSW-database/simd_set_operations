#![cfg(feature = "simd")]

use std::simd::*;

use crate::{visitor::{SimdVisitor, VecWriter}, intersect::branchless_merge, instructions::load};



/// SIMD Galloping algorithm by D. Lemire et al.
/// 
/// Extends the classical galloping algorithm by performing comparisons of
/// blocks of 8 4xi32 registers, placing results in bitmasks Q1, Q2, Q3, Q4
/// where each Q is the result of a pairwise comparison between two SIMD
/// vectors. The galloping stage bounds in leaps of 4x8 SIMD registers = 32x4
/// integers, then performs a mini binary search to narrow it down to a block of
/// 8 registers.
#[inline(never)]
pub fn simd_galloping<V>(mut small: &[i32], mut large: &[i32], visitor: &mut V)
where
    V: SimdVisitor<i32, 4>,
{
    assert!(small.len() <= large.len());
    const LANES: usize = 4;
    const SEARCH_SIZE: usize = 4;
    const COMPARE_SIZE: usize = 8;

    const BOUND_VEC: usize = SEARCH_SIZE * COMPARE_SIZE;
    const BOUND: usize = BOUND_VEC * LANES;

    if large.len() < BOUND {
        return branchless_merge(small, large, visitor);
    }

    while !small.is_empty() && large.len() >= BOUND {
        let target = small[0];
        let target_vec = i32x4::splat(target);

        let mut offset = 1;

        while (offset + 1) * BOUND <= large.len()
            && large[(offset + 1) * BOUND - 1] < target
        {
            offset *= 2;
        }

        let mut lo = offset / 2;
        let mut hi = (large.len() / BOUND).min(offset);

        // TODO: handle case where lo ~= hi and there isn't enough room.

        while lo + 1 != hi {
            let mid = lo + (hi - lo) / 2;
            if large[mid * BOUND - 1] < target {
                lo = mid;
            }
            else {
                hi = mid;
            }
        }
        assert!(large[hi * BOUND] <= target && large[hi * BOUND - 1] <= target);

        large = &large[hi * BOUND..];
        assert!(large.len() >= BOUND);

        // Check if target appears in range large[0..128]
        let search_offset: usize =
        if large[LANES * 16 - 1] >= target {
            if large[LANES * 8 - 1] < target { 8 } else { 0 }
        }
        else {
            if large[LANES * 24 - 1] < target { 24 } else { 16 }
        };

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

    assert!(large.len() < BOUND);
    branchless_merge(small, large, visitor)
}

pub fn simd_galloping_mono(small: &[i32], large: &[i32], visitor: &mut VecWriter<i32>) {
    simd_galloping(small, large, visitor);
}
