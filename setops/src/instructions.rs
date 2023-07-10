#![cfg(feature = "simd")]

// Many functions taken from roaring-rs
// Licensed under either of
//    Apache License, Version 2.0 (https://www.apache.org/licenses/LICENSE-2.0)
//    MIT license (https://opensource.org/licenses/MIT)

use core::simd::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;


#[inline]
pub fn load<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(src.len() >= LANES);
    unsafe { load_slice_unchecked(src) }
}

#[inline]
pub unsafe fn load_slice_unchecked<T, const LANES: usize>(src: &[T]) -> Simd<T, LANES>
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    unsafe { std::ptr::read_unaligned(src as *const _ as *const Simd<T, LANES>) }
}

#[inline]
pub unsafe fn load_unsafe<T, const LANES: usize>(src: *const T) -> Simd<T, LANES>
where
    T: SimdElement,
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

#[inline]
#[cfg(target_feature = "ssse3")]
pub fn shuffle_epi8<P, Q>(a: P, b: Q) -> P
where
    P: Into<__m128i> + From<__m128i>,
    Q: Into<__m128i>,
{
    unsafe{ _mm_shuffle_epi8(a.into(), b.into() )}.into()
}

#[inline]
#[cfg(target_feature = "ssse3")]
pub fn permutevar8x32_epi32<P, Q>(a: P, b: Q) -> P
where
    P: Into<__m256i> + From<__m256i>,
    Q: Into<__m256i>,
{
    unsafe { _mm256_permutevar8x32_epi32(a.into(), b.into()) }.into()
}

pub const SWIZZLE_TO_FRONT4: [[i32; 4]; 16] = gen_swizzle_to_front();
pub const SWIZZLE_TO_FRONT8: [[i32; 8]; 256] = gen_swizzle_to_front();
pub const VEC_SHUFFLE_MASK4: [u8x16; 16] = gen_vec_shuffle();
pub const VEC_SHUFFLE_MASK8: [i32x8; 256] = prepare_shuffling_dictionary_avx();

#[inline]
pub fn convert<P, Q>(a: P) -> Q
where
    __m128i: From<P> + Into<Q>,
{
    __m128i::from(a).into()
}

// For BMiss. From https://github.com/pkumod/GraphSetIntersection.
pub const BYTE_CHECK_GROUP_A: [[usize; 16]; 4] = [
    [0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12],
    [1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13],
    [2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14],
    [3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15],
];
pub const BYTE_CHECK_GROUP_B: [[usize; 16]; 4] = [
    [0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12],
    [1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13],
    [2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14],
    [3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15],
];


pub const BYTE_CHECK_GROUP_A_VEC: [u8x16; 4] = [
    u8x16::from_array([0, 0, 0, 0, 4, 4, 4, 4, 8, 8, 8, 8, 12, 12, 12, 12]),
    u8x16::from_array([1, 1, 1, 1, 5, 5, 5, 5, 9, 9, 9, 9, 13, 13, 13, 13]),
    u8x16::from_array([2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14, 14, 14]),
    u8x16::from_array([3, 3, 3, 3, 7, 7, 7, 7, 11, 11, 11, 11, 15, 15, 15, 15]),
];
pub const BYTE_CHECK_GROUP_B_VEC: [u8x16; 4] = [
    u8x16::from_array([0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12]),
    u8x16::from_array([1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13]),
    u8x16::from_array([2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14]),
    u8x16::from_array([3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15]),
];

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

const fn gen_vec_shuffle() -> [u8x16; 16] {
    let mut result = [u8x16::from_array([0; 16]); 16];

    let mut i = 0;
    while i < 16 {
        let mut shuffle_mask = [0u8; 16];

        let mut counter = 0;
        let mut b: u8 = 0;
        while b < 4 {
            if get_bit(i, b) != 0 {
                shuffle_mask[counter] = 4*b;
                shuffle_mask[counter+1] = 4*b + 1;
                shuffle_mask[counter+2] = 4*b + 2;
                shuffle_mask[counter+3] = 4*b + 3;
                counter += 4;
            }
            b += 1;
        }
        result[i as usize] = u8x16::from_array(shuffle_mask);
        i += 1;
    }

    result
}

const fn get_bit(value: i32, position: u8) -> i32 {
    (value & (1 << position)) >> position
}


// Source: tetzank
// https://github.com/tetzank/SIMDSetOperations
const fn prepare_shuffling_dictionary_avx() -> [i32x8; 256] {
    let mut result = [i32x8::from_array([0; 8]); 256];

    let mut i = 0;
    while i < 256 {
        let mut shuffle_mask = [0i32; 8];

        let mut count = 0;
        let mut rest: i32 = 7;
        let mut b = 0;
        while b < 8 {
            if i & (1 << b) != 0 {
                // n index at pos p - move nth element to pos p
                shuffle_mask[count] = b; // move all set bits to beginning
                count += 1;
            } else {
                shuffle_mask[rest as usize] = b; // move rest at the end
                rest -= 1;
            }

            b += 1;
        }
        result[i] = i32x8::from_array(shuffle_mask);
        i += 1;
    }
    result
}

#[inline]
#[cold]
pub fn cold() {}

#[inline]
pub fn likely(b: bool) -> bool {
    if !b { cold() }
    b
}

#[inline]
pub fn unlikely(b: bool) -> bool {
    if b { cold() }
    b
}
