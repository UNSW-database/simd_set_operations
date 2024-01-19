#![cfg(feature = "simd")]
/// SIMD Galloping algorithm by D. Lemire et al.
///
/// Extends the classical galloping algorithm by performing comparisons of
/// blocks of 8 4xi32 registers, placing results in bitmasks Q1, Q2, Q3, Q4
/// where each Q is the result of a pairwise comparison between two SIMD
/// vectors. The galloping stage bounds in leaps of 4x8 SIMD registers = 32x4
/// integers, then performs a mini binary search to narrow it down to a block of
/// 8 registers.

use std::simd::*;
use std::simd::cmp::*;

use crate::{visitor::{Visitor, BsrVisitor}, intersect, instructions::load_unsafe, bsr::BsrRef};

const NUM_LANES_IN_BOUND: usize = 32;

/// 4 lane version used to intersect with 128-bit vectors, e.g., i32x4.
pub fn galloping_sse<T, V>(small: &[T], large: &[T], visitor: &mut V)
where
    T: SimdElement + MaskElement + Ord + Default,
    Simd<T, 4>: SimdPartialEq<Mask=Mask<T, 4>>,
    V: Visitor<T>,
{
    simd_galloping_impl::<T, V, 4>(small, large, visitor)
}

/// 8 lane version used to intersect with 256-bit vectors, e.g., i32x8.
pub fn galloping_avx2<T, V>(small: &[T], large: &[T], visitor: &mut V)
where
    T: SimdElement + MaskElement + Ord + Default,
    Simd<T, 8>: SimdPartialEq<Mask=Mask<T, 8>>,
    V: Visitor<T>,
{
    simd_galloping_impl::<T, V, 8>(small, large, visitor)
}

/// 16 lane version used to intersect with 512-bit vectors, e.g., i32x16.
/// Only faster if native 512-bit vectors are supported.
pub fn galloping_avx512<T, V>(small: &[T], large: &[T], visitor: &mut V)
where
    T: SimdElement + MaskElement + Ord + Default,
    Simd<T, 16>: SimdPartialEq<Mask=Mask<T, 16>>,
    V: Visitor<T>,
{
    simd_galloping_impl::<T, V, 16>(small, large, visitor)
}

fn simd_galloping_impl<'a, T, V, const LANES: usize>(
    mut small: &'a[T],
    mut large: &'a[T],
    visitor: &mut V)
where
    T: SimdElement + MaskElement + Ord + Default,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SimdPartialEq<Mask=Mask<T, LANES>>,
    V: Visitor<T>,
{
    if small.len() > large.len() {
        (small, large) = (large, small);
    }

    let bound = Simd::<T, LANES>::from_array([T::default(); LANES]).len() * NUM_LANES_IN_BOUND;

    while !small.is_empty() && large.len() >= bound {
        let target = small[0];

        let target_block = gallop_wide(target, large, bound);

        // Check if block actually contains target.
        if large[(target_block + 1) * bound - 1] < target {
            // If not, shrink large.
            large = &large[(target_block + 1) * bound..];

            debug_assert!(large.len() < bound);
            // Swap small and large if small is big enough.
            if small.len() >= bound {
                (small, large) = (large, small);
                continue;
            }
            else {
                break;
            }
        }

        debug_assert!(target_block == 0 || large[target_block * bound - 1] < target);
        debug_assert!(large[(target_block+1) * bound - 1] >= target);

        large = &large[target_block * bound..];
        debug_assert!(large.len() >= bound);

        let inner_offset: usize = reduce_search_bound(target, large, bound);

        let result = block_compare::<T, LANES>(target, inner_offset, large);

        if result.any() {
            visitor.visit(target);
        }
        small = &small[1..];
    }

    debug_assert!(small.is_empty() || large.len() < bound);
    intersect::branchless_merge(small, large, visitor)
}

pub fn galloping_sse_bsr<'a, V>(
    small: BsrRef<'a>,
    large: BsrRef<'a>,
    visitor: &mut V)
where
    V: BsrVisitor,
{
    simd_galloping_bsr_impl::<V, 4, u8>(small, large, visitor)
}

pub fn galloping_avx2_bsr<'a, V>(
    small: BsrRef<'a>,
    large: BsrRef<'a>,
    visitor: &mut V)
where
    V: BsrVisitor,
{
    simd_galloping_bsr_impl::<V, 8, u8>(small, large, visitor)
}

pub fn galloping_avx512_bsr<'a, V>(
    small: BsrRef<'a>,
    large: BsrRef<'a>,
    visitor: &mut V)
where
    V: BsrVisitor,
{
    simd_galloping_bsr_impl::<V, 16, u16>(small, large, visitor)
}

pub fn simd_galloping_bsr_impl<'a, V, const LANES: usize, B>(
    mut small: BsrRef<'a>,
    mut large: BsrRef<'a>,
    visitor: &mut V)
where
    V: BsrVisitor,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<u32, LANES>: SimdPartialEq<Mask=Mask<i32, LANES>>,
{
    if small.len() > large.len() {
        (small, large) = (large, small);
    }

    let bound = Simd::<i32, LANES>::from_array([0; LANES]).len();

    while !small.is_empty() && large.len() >= bound {
        let target_base = small.bases[0];
        let target_state = small.states[0];

        let found_block = gallop_wide(target_base, large.bases, bound);

        // Check if block actually contains target.
        if large.bases[(found_block + 1) * bound - 1] < target_base {
            // If not, shrink large.
            large = large.advanced_by((found_block + 1) * bound);

            debug_assert!(large.len() < bound);
            // Swap small and large if small is big enough.
            if small.len() >= bound {
                (small, large) = (large, small);
                continue;
            }
            else {
                break;
            }
        }

        debug_assert!(found_block == 0 || large.bases[found_block * bound - 1] < target_base);
        debug_assert!(large.bases[(found_block+1) * bound - 1] >= target_base);

        large = large.advanced_by(found_block * bound);
        debug_assert!(large.len() >= bound);

        let target_vec = Simd::<u32, LANES>::splat(target_base);
        let cmp_mask = target_vec.simd_eq(unsafe { load_unsafe(large.bases.as_ptr()) });
        if cmp_mask.any() {
            let p = cmp_mask.to_bitmask().trailing_zeros();
            let result_state = target_state & large.states[p as usize];
            if result_state != 0 {
                visitor.visit_bsr(target_base, result_state);
            }
        }
        small = small.advanced_by(1);
    }

    debug_assert!(small.is_empty() || large.len() < bound);
    intersect::branchless_merge_bsr(small, large, visitor)
}

fn gallop_wide<T>(target: T, large: &[T], bound: usize) -> usize
where
    T: Ord
{
    let upper_bound = if large[bound - 1] >= target {
        0
    }
    else {
        let mut offset = 1;
        while (offset + 1) * bound - 1 < large.len()
            && large[(offset + 1) * bound - 1] < target
        {
            offset *= 2;
        }
        offset
    };

    let lo = upper_bound / 2;
    let hi = (large.len() / bound - 1).min(upper_bound);

    binary_search_wide(target, large, lo, hi, bound)
}

fn binary_search_wide<T>(
    target: T,
    large: &[T],
    low: usize,
    high: usize,
    bound: usize) -> usize
where
    T: Ord
{
    let mut lo = low as isize;
    let mut hi = high as isize;

    // Trying to find the block index such that the last element in the block
    // is greater than or equal to the target value
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if large[(mid as usize + 1) * bound - 1] < target {
            lo = mid + 1;
        }
        else {
            hi = mid;
        }
    }
    debug_assert!(lo == hi);
    lo as usize
}

fn reduce_search_bound<T>(target: T, large: &[T], bound: usize) -> usize
where
    T: Ord,
{
    if large[bound / 2 - 1] >= target {
        if large[bound / 4 - 1] < target {
            NUM_LANES_IN_BOUND / 4
        }
        else {
            0
        }
    }
    else if large[bound * 3 / 4 - 1] < target {
        NUM_LANES_IN_BOUND * 3 / 4
    }
    else {
        NUM_LANES_IN_BOUND / 2
    }
}

#[inline]
fn block_compare<T, const LANES: usize>(
    target: T,
    inner_offset: usize,
    large: &[T]) -> Mask<T, LANES>
where
    T: SimdElement + MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SimdPartialEq<Mask=Mask<T, LANES>>,
{
    let target_vec = Simd::<T, LANES>::splat(target);
    let qs = [
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset    ))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 1))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 2))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 3))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 4))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 5))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 6))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.as_ptr().add(LANES * (inner_offset + 7))) })
    ];
    (qs[0] | qs[1]) | (qs[2] | qs[3])
}
