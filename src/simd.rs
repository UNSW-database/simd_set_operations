// Inspired by roaring-rs

#![cfg(feature = "simd")]

use core::simd::{
    simd_swizzle, LaneCount, Mask, Simd, SimdElement, SimdPartialEq,
    SimdPartialOrd, SupportedLaneCount, ToBitMask,
};

#[inline]
pub fn load<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(src.len());
    unsafe { load_unchecked(src) }
}

#[inline]
pub unsafe fn load_unchecked<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::ptr::read_unaligned(src as *const _ as *const Simd<U, LANES>) }
}
