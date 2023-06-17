#![cfg(feature = "simd")]

// Taken from roaring-rs
// Licensed under either of
//    Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
//    MIT license (https://opensource.org/licenses/MIT)

use core::simd::*;

#[inline]
pub fn load<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(src.len() >= LANES);
    unsafe { load_unchecked(src) }
}

#[inline]
pub unsafe fn load_unchecked<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::ptr::read_unaligned(src as *const _ as *const Simd<T, LANES>) }
}


#[inline]
pub fn store<T, const LANES: usize>(v: Simd<T, LANES>, out: &mut [T])
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(out.len() >= LANES);
    unsafe {
        store_unchecked(v, out);
    }
}

#[inline]
unsafe fn store_unchecked<T, const LANES: usize>(v: Simd<T, LANES>, out: &mut [T])
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::ptr::write_unaligned(out as *mut _ as *mut Simd<T, LANES>, v) }
}

pub const SWIZZLE_TO_FRONT4: [[i32; 4]; 16] = gen_swizzle_to_front();
pub const SWIZZLE_TO_FRONT8: [[i32; 8]; 256] = gen_swizzle_to_front();

const fn gen_swizzle_to_front<const LANES: usize, const COUNT: usize>() -> [[i32; LANES]; COUNT] {
    assert!(COUNT == 2usize.pow(LANES as u32));

    let mut result = [[0; LANES]; COUNT];

    let mut n: usize = 0;
    while n < COUNT {
        result[n] = swizzle_to_front_value(n);
        n += 1;
    }
    result
}

const fn swizzle_to_front_value<const SIZE: usize>(n: usize) -> [i32; SIZE] {
    let mut result = [0; SIZE];
    let mut x = n;
    let mut i = 0;
    while x > 0 {
        let lsb = x.trailing_zeros() as i32;
        result[i] = lsb;
        x ^= 1 << lsb;
        i += 1;
    }
    result
}
