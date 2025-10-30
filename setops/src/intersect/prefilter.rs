#![cfg(feature = "simd")]

use std::simd::cmp::SimdPartialEq;
use std::simd::*;

use crate::instructions::load_unsafe;

/// Performs low-byte prefiltering and immediately refines matches using the same loaded lanes.
#[inline(always)]
pub unsafe fn probe_low_byte_and_compare<const LANES: usize>(
    target: i32,
    ptr: *const i32,
    segments: usize,
) -> Mask<i32, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
    Simd<i32, LANES>: SimdPartialEq<Mask = Mask<i32, LANES>>,
{
    let target_vec = Simd::<i32, LANES>::splat(target);
    let low_mask = Simd::<i32, LANES>::splat(0xFF);
    let target_low = Simd::<i32, LANES>::splat(((target as u32) & 0xFF) as i32);

    let mut combined = Mask::<i32, LANES>::splat(false);
    let mut offset = 0usize;
    while offset < segments {
        let lanes = load_unsafe::<i32, LANES>(ptr.add(offset * LANES));
        let low_eq = (lanes & low_mask).simd_eq(target_low);
        if low_eq.any() {
            combined |= target_vec.simd_eq(lanes) & low_eq;
        }
        offset += 1;
    }

    combined
}
