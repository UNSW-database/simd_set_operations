#![cfg(feature = "simd")]

use std::simd::cmp::SimdPartialEq;
use std::simd::*;

use crate::instructions::load_unsafe;

/// Prefilter trait that allows fast rejection before full intersection.
pub trait SimdPrefilter {
    /// Returns true if the SIMD block pointed to by `ptr` potentially contains `target`.
    unsafe fn should_probe<const LANES: usize>(target: i32, ptr: *const i32) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount;
}

/// Matches on the least significant byte of each element.
pub struct LowBytePrefilter;

impl SimdPrefilter for LowBytePrefilter {
    #[inline(always)]
    unsafe fn should_probe<const LANES: usize>(target: i32, ptr: *const i32) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let vec = load_unsafe::<i32, LANES>(ptr);
        let masked = vec.to_array().map(|value| (value as u32) & 0xFF);
        let vec_u = Simd::<u32, LANES>::from_array(masked);
        let target_low = Simd::<u32, LANES>::splat((target as u32) & 0xFF);
        vec_u.simd_eq(target_low).any()
    }
}

/// Matches on the most significant 16 bits of each element.
pub struct HighWordPrefilter;

impl SimdPrefilter for HighWordPrefilter {
    #[inline(always)]
    unsafe fn should_probe<const LANES: usize>(target: i32, ptr: *const i32) -> bool
    where
        LaneCount<LANES>: SupportedLaneCount,
    {
        let vec = load_unsafe::<i32, LANES>(ptr);
        let masked = vec.to_array().map(|value| ((value as u32) >> 16) & 0xFFFF);
        let vec_u = Simd::<u32, LANES>::from_array(masked);
        let target_high = Simd::<u32, LANES>::splat(((target as u32) >> 16) & 0xFFFF);
        vec_u.simd_eq(target_high).any()
    }
}

/// Returns true if any of the `segments` SIMD blocks match according to `P`.
#[inline(always)]
pub unsafe fn any_prefilter_match<P: SimdPrefilter, const LANES: usize>(
    target: i32,
    ptr: *const i32,
    segments: usize,
) -> bool
where
    LaneCount<LANES>: SupportedLaneCount,
{
    let mut offset = 0usize;
    while offset < segments {
        if P::should_probe::<LANES>(target, ptr.add(offset * LANES)) {
            return true;
        }
        offset += 1;
    }
    false
}
