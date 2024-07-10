use crate::{bsr::{BsrVec, BsrRef}, instructions};
#[cfg(feature = "simd")]
use {
    std::simd::*,
    crate::util::slice_i32_to_u32
};
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
use crate::instructions::{ VEC_SHUFFLE_MASK4, shuffle_epi8 };

#[cfg(all(feature = "simd", target_feature = "avx2"))]
use crate::instructions::{VEC_SHUFFLE_MASK8, permutevar8x32_epi32};

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
use crate::instructions::{VEC_SHUFFLE_MASK16, permutevar_avx512};

/// Used to receive set intersection results in a generic way. Inspired by
/// roaring-rs.
pub trait Visitor<T> {
    fn visit(&mut self, value: T);
}

pub trait Clearable {
    fn clear(&mut self);
}

/// Counts intersection size without storing result.
pub struct Counter {
    count: usize,
}

impl<T> Visitor<T> for Counter {
    fn visit(&mut self, _value: T) {
        self.count += 1;
    }
}

impl Counter {
    pub fn new() -> Self {
        Self { count: 0 }
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

impl Default for Counter {
    fn default() -> Self {
        Self::new()
    }
}

/// Stores intersection result in a vector.
pub struct VecWriter<T> {
    items: Vec<T>,
}

impl<T> VecWriter<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub fn with_capacity(cardinality: usize) -> Self {
        Self {
            items: Vec::with_capacity(cardinality),
        }
    }
}

impl<T> AsRef<[T]> for VecWriter<T> {
    fn as_ref(&self) -> &[T] {
        &self.items
    }
}

impl<T> From<VecWriter<T>> for Vec<T> {
    fn from(value: VecWriter<T>) -> Self {
        value.items
    }
}

impl<T> Default for VecWriter<T> {
    fn default() -> Self {
        Self { items: Vec::default() }
    }
}

impl<T> Visitor<T> for VecWriter<T> {
    fn visit(&mut self, value: T) {
        self.items.push(value);
    }
}

impl<T> Clearable for VecWriter<T> {
    fn clear(&mut self) {
        self.items.clear();
    }
}

/// Writes intersection result to provided array slice.
pub struct SliceWriter<'a, T> {
    data: &'a mut[T],
    position: usize,
}

impl<'a, T> SliceWriter<'a, T> {
    pub fn position(&self) -> usize {
        self.position
    }
}

impl<'a, T> From<&'a mut[T]> for SliceWriter<'a, T> {
    fn from(data: &'a mut[T]) -> Self {
        Self {
            data,
            position: 0,
        }
    }
}

impl<'a, T> Visitor<T> for SliceWriter<'a, T> {
    fn visit(&mut self, value: T) {
        self.data[self.position] = value;
        self.position += 1;
    }
}

impl<'a, T> Clearable for SliceWriter<'a, T> {
    fn clear(&mut self) {
        self.position = 0;
    }
}

/*-------- SIMD --------*/
/// Allows visiting of multiple elements
#[cfg(feature = "simd")]
pub trait SimdVisitor4 : Visitor<i32> {
    fn visit_vector4(&mut self, value: i32x4, mask: u64);
}
pub trait SimdVisitor8: Visitor<i32> {
    fn visit_vector8(&mut self, value: i32x8, mask: u64);
}
pub trait SimdVisitor16: Visitor<i32> {
    fn visit_vector16(&mut self, value: i32x16, mask: u64);
}

#[cfg(feature = "simd")]
impl SimdVisitor4 for Counter {
    fn visit_vector4(&mut self, _value: i32x4, mask: u64) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(feature = "simd")]
impl SimdVisitor8 for Counter {
    fn visit_vector8(&mut self, _value: i32x8, mask: u64) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(feature = "simd")]
impl SimdVisitor16 for Counter {
    fn visit_vector16(&mut self, _value: i32x16, mask: u64) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor4 for VecWriter<i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        extend_i32vec_x4(&mut self.items, value, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor8 for VecWriter<i32> {
    #[cfg(target_feature = "avx2")]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        extend_i32vec_x8(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let arr = value.as_array();
        let masks = [
            mask       & 0xF,
            mask >> 4  & 0xF,
        ];

        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]), masks[0]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..]), masks[1]);
    }
}
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor16 for VecWriter<i32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        extend_i32vec_x16(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let left = mask & 0xFF;
        let right = (mask >> 8) & 0xFF;

        extend_i32vec_x8(&mut self.items, i32x8::from_slice(&arr[..8]), left);
        extend_i32vec_x8(&mut self.items, i32x8::from_slice(&arr[8..]), right);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let masks = [
            (mask       & 0xF) as u8,
            (mask >> 4  & 0xF) as u8,
            (mask >> 8  & 0xF) as u8,
            (mask >> 12 & 0xF) as u8,
        ];

        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]),   masks[0]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..8]),  masks[1]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[8..12]), masks[2]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[12..]),  masks[3]);
    }
}

impl Visitor<i32> for VecWriter<u32> {
    fn visit(&mut self, value: i32) {
        self.items.push(value as u32);
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor4 for VecWriter<u32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        extend_u32vec_x4(&mut self.items, value, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor8 for VecWriter<u32> {
    #[cfg(target_feature = "avx2")]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        extend_u32vec_x8(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let arr = value.as_array();
        let masks = [
            mask       & 0xF,
            mask >> 4  & 0xF,
        ];

        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]), masks[0]);
        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..]), masks[1]);
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor16 for VecWriter<u32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        extend_u32vec_x16(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let left = mask & 0xFF;
        let right = (mask >> 8) & 0xFF;

        extend_u32vec_x8(&mut self.items, i32x8::from_slice(&arr[..8]), left);
        extend_u32vec_x8(&mut self.items, i32x8::from_slice(&arr[8..]), right);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let masks = [
            (mask       & 0xF) as u8,
            (mask >> 4  & 0xF) as u8,
            (mask >> 8  & 0xF) as u8,
            (mask >> 12 & 0xF) as u8,
        ];

        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]),   masks[0]);
        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..8]),  masks[1]);
        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[8..12]), masks[2]);
        extend_u32vec_x4(&mut self.items, i32x4::from_slice(&arr[12..]),  masks[3]);
    }
}


// SLICE WRITER
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl<'a> SimdVisitor4 for SliceWriter<'a, i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        extend_i32slice_x4(&mut self.data, &mut self.position, value, mask);
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl<'a> SimdVisitor8 for SliceWriter<'a, i32> {
    #[cfg(target_feature = "avx2")]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let shuffled = permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);
        instructions::store(shuffled, &mut self.data[self.position..]);

        self.position += mask.count_ones() as usize;
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let arr = value.as_array();
        let masks = [
            mask       & 0xF,
            mask >> 4  & 0xF,
        ];

        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[..4]), masks[0]);
        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[4..]), masks[1]);
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl<'a> SimdVisitor16 for SliceWriter<'a, i32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        extend_i32slice_x16(&mut self.data, &mut self.position, value, mask);
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let left = mask & 0xFF;
        let right = (mask >> 8) & 0xFF;

        extend_i32slice_x8(&mut self.data, &mut self.position, i32x8::from_slice(&arr[..8]), left);
        extend_i32slice_x8(&mut self.data, &mut self.position, i32x8::from_slice(&arr[8..]), right);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let masks = [
            (mask       & 0xF) as u8,
            (mask >> 4  & 0xF) as u8,
            (mask >> 8  & 0xF) as u8,
            (mask >> 12 & 0xF) as u8,
        ];

        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[..4]),   masks[0]);
        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[4..8]),  masks[1]);
        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[8..12]), masks[2]);
        extend_i32slice_x4(&mut self.data, &mut self.position, i32x4::from_slice(&arr[12..]),  masks[3]);
    }
}

/// Allows visiting of single entries in Base and State Representation
pub trait BsrVisitor {
    fn visit_bsr(&mut self, base: u32, state: u32);
}

/// Allows visiting of multiple entries in Base and State Representation
#[cfg(feature = "simd")]
pub trait SimdBsrVisitor4 : BsrVisitor {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u64);
}
pub trait SimdBsrVisitor8 : BsrVisitor {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u64);
}
pub trait SimdBsrVisitor16 : BsrVisitor {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u64);
}

impl BsrVisitor for BsrVec {
    fn visit_bsr(&mut self, base: u32, state: u32) {
        self.append(base, state)
    }
}

impl BsrVisitor for Counter {
    fn visit_bsr(&mut self, _base: u32, state: u32) {
        self.count += state.count_ones() as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdBsrVisitor4 for BsrVec {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u64) {
        extend_u32vec_x4(&mut self.bases, base, mask);
        extend_u32vec_x4(&mut self.states, state, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl SimdBsrVisitor8 for BsrVec {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u64) {
        extend_u32vec_x8(&mut self.bases, base, mask);
        extend_u32vec_x8(&mut self.states, state, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor16 for BsrVec {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u64) {
        extend_u32vec_x16(&mut self.bases, base, mask);
        extend_u32vec_x16(&mut self.states, state, mask);
    }
}

// These should be vectorised with AVX-512's VPOPCNT
#[cfg(feature = "simd")]
impl SimdBsrVisitor4 for Counter {
    fn visit_bsr_vector4(&mut self, _base: i32x4, state: i32x4, mask: u64) {
        let masked_state = mask32x4::from_bitmask(mask).to_int() & state;
        let s = masked_state.as_array();
        let count =
            s[0].count_ones() + s[1].count_ones() +
            s[2].count_ones() + s[3].count_ones();
        self.count += count as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl SimdBsrVisitor8 for Counter {
    fn visit_bsr_vector8(&mut self, _base: i32x8, state: i32x8, mask: u64) {
        let masked_state = mask32x8::from_bitmask(mask).to_int() & state;
        let s = masked_state.as_array();
        let count =
            s[0].count_ones() + s[1].count_ones() +
            s[2].count_ones() + s[3].count_ones() +
            s[4].count_ones() + s[5].count_ones() +
            s[6].count_ones() + s[7].count_ones();
        self.count += count as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor16 for Counter {
    fn visit_bsr_vector16(&mut self, _base: i32x16, state: i32x16, mask: u64) {
        let masked_state = mask32x16::from_bitmask(mask).to_int() & state;
        let s = masked_state.as_array();
        let count =
            s[0].count_ones() + s[1].count_ones() +
            s[2].count_ones() + s[3].count_ones() +
            s[4].count_ones() + s[5].count_ones() +
            s[6].count_ones() + s[7].count_ones() +
            s[8].count_ones() + s[9].count_ones() +
            s[10].count_ones() + s[11].count_ones() +
            s[12].count_ones() + s[13].count_ones() +
            s[14].count_ones() + s[15].count_ones();
        self.count += count as usize;
    }
}


/// Ensures all visits match expected output.
/// Used for testing algorithm correctness.
pub struct EnsureVisitor<'a, T>
where
    T: PartialEq,
{
    expected: &'a[T],
    position: usize,
}

impl<'a, T> EnsureVisitor<'a, T>
where
    T: PartialEq,
{
    pub fn position(&self) -> usize {
        self.position
    }
}

impl<'a, T> From<&'a[T]> for EnsureVisitor<'a, T>
where
    T: PartialEq,
{
    fn from(expected: &'a[T]) -> Self {
        Self {
            expected,
            position: 0,
        }
    }
}

impl<'a, T> Visitor<T> for EnsureVisitor<'a, T>
where
    T: PartialEq + std::fmt::Debug,
{
    fn visit(&mut self, value: T) {
        assert_eq!(value, self.expected[self.position]);
        self.position += 1;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl<'a> SimdVisitor4 for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);

        let count = mask.count_ones() as usize;
        assert_eq!(&shuffled[..count],
            &self.expected[self.position..self.position+count]);

        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl<'a> SimdVisitor8 for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let shuffled =
            permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

        let count = mask.count_ones() as usize;
        assert_eq!(&shuffled[..count],
            &self.expected[self.position..self.position+count]);

        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<'a> SimdVisitor16 for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let actual: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(),
            mask as u16,
            value.into(),
        )}.into();

        let count = mask.count_ones() as usize;

        assert_eq!(&actual.to_array()[..count],
            &self.expected[self.position..self.position+count]);

        self.position += count;
    }
}

pub struct EnsureVisitorBsr<'a> {
    expected: BsrRef<'a>,
    position: usize,
}

impl<'a> EnsureVisitorBsr<'a> {
    pub fn position(&self) -> usize {
        self.position
    }
}

impl<'a> From<BsrRef<'a>> for EnsureVisitorBsr<'a> {
    fn from(expected: BsrRef<'a>) -> Self {
        Self {
            expected,
            position: 0,
        }
    }
}

impl<'a> BsrVisitor for EnsureVisitorBsr<'a> {
    fn visit_bsr(&mut self, base: u32, state: u32) {
        let expected = (
            self.expected.bases[self.position],
            self.expected.states[self.position]
        );
        assert_eq!((base, state), expected);
        self.position += 1;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl<'a> SimdBsrVisitor4 for EnsureVisitorBsr<'a> {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u64) {
        let base_s = shuffle_epi8(base, VEC_SHUFFLE_MASK4[mask as usize]);
        let state_s = shuffle_epi8(state, VEC_SHUFFLE_MASK4[mask as usize]);
        let count = mask.count_ones() as usize;

        let expected = (
            &self.expected.bases[self.position..self.position+count],
            &self.expected.states[self.position..self.position+count],
        );
        let actual = (
            slice_i32_to_u32(&base_s[..count]),
            slice_i32_to_u32(&state_s[..count]),
        );
        assert_eq!(actual, expected);
        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl<'a> SimdBsrVisitor8 for EnsureVisitorBsr<'a> {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u64) {
        let base_s =
            permutevar8x32_epi32(base, VEC_SHUFFLE_MASK8[mask as usize]);
        let state_s =
            permutevar8x32_epi32(state, VEC_SHUFFLE_MASK8[mask as usize]);

        let count = mask.count_ones() as usize;
        let expected = (
            &self.expected.bases[self.position..self.position+count],
            &self.expected.states[self.position..self.position+count],
        );
        let actual = (
            slice_i32_to_u32(&base_s[..count]),
            slice_i32_to_u32(&state_s[..count]),
        );
        assert_eq!(actual, expected);
        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl<'a> SimdBsrVisitor16 for EnsureVisitorBsr<'a> {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let actual_base: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(), mask as u16, base.into(),
        )}.into();
        let actual_state: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(), mask as u16, state.into(),
        )}.into();

        let count = mask.count_ones() as usize;
        let expected = (
            &self.expected.bases[self.position..self.position+count],
            &self.expected.states[self.position..self.position+count],
        );
        let actual = (
            slice_i32_to_u32(&actual_base[..count]),
            slice_i32_to_u32(&actual_state[..count]),
        );
        assert_eq!(actual, expected);
        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline]
fn extend_i32vec_x4(items: &mut Vec<i32>, value: i32x4, mask: u64) {
    let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
    extend_vec(items, &shuffled.as_array()[..], shuffled.len(), mask);
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline]
fn extend_u32vec_x4(items: &mut Vec<u32>, value: i32x4, mask: u64) {
    let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
    extend_vec(
        items, slice_i32_to_u32(&shuffled.as_array()[..]),
        shuffled.len(), mask);
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline]
fn extend_i32slice_x4(data: &mut [i32], position: &mut usize, value: i32x4, mask: u64) {
    let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
    instructions::store(shuffled, &mut data[*position..]);
    *position += mask.count_ones() as usize;
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline]
fn extend_i32vec_x8(items: &mut Vec<i32>, value: i32x8, mask: u64) {
    let shuffled = permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

    extend_vec(items, &shuffled.as_array()[..], shuffled.len(), mask);
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline]
#[allow(dead_code)]
fn extend_i32slice_x8(data: &mut [i32], position: &mut usize, value: i32x8, mask: u64) {
    let shuffled = permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);
    instructions::store(shuffled, &mut data[*position..]);
    *position += mask.count_ones() as usize;
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[inline]
fn extend_i32vec_x16(items: &mut Vec<i32>, value: i32x16, mask: u64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    items.reserve(items.len() + 16);
    unsafe {
        _mm512_mask_compressstoreu_epi32(
            items.as_mut_ptr().add(items.len()) as *mut u8,
            mask as u16,
            value.into(),
        );
        items.set_len(items.len() + mask.count_ones() as usize);
    };
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[inline]
fn extend_i32slice_x16(data: &mut [i32], position: &mut usize, value: i32x16, mask: u64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    unsafe {
        _mm512_mask_compressstoreu_epi32(
            data.as_mut_ptr().add(*position) as *mut u8,
            mask as u16,
            value.into(),
        );
    }
    *position += mask.count_ones() as usize;
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[inline]
fn extend_u32vec_x16(items: &mut Vec<u32>, value: i32x16, mask: u64) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    items.reserve(items.len() + 16);
    unsafe {
        _mm512_mask_compressstoreu_epi32(
            items.as_mut_ptr().add(items.len()) as *mut u8,
            mask as u16,
            value.into(),
        );
        items.set_len(items.len() + mask.count_ones() as usize);
    };
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline]
fn extend_u32vec_x8(items: &mut Vec<u32>, value: i32x8, mask: u64) {

    let shuffled =
        permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

    extend_vec(
        items, slice_i32_to_u32(&shuffled.as_array()[..]),
        shuffled.len(), mask);
}

#[cfg(feature = "simd")]
#[inline]
fn extend_vec<T>(items: &mut Vec<T>, shuffled: &[T], lanes: usize, mask: u64)
where
    T: Clone
{
    items.extend_from_slice(shuffled);
    // Truncate the masked out values
    items.truncate(items.len() - (lanes - mask.count_ones() as usize));
}

// Unsafe writers: only for benchmarking!
// Always assumes the vec aleady has enough space.
pub struct UnsafeLookupWriter<T> {
    items: Vec<T>,
}

impl<T> UnsafeLookupWriter<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub fn with_capacity(cardinality: usize) -> Self {
        Self {
            // For a final set size of x, we need to round up to nearest 16
            // to ensure we don't write past buffer with SIMD vector.
            // To be extra safe, we just add 16.
            // This is ok as UnsafeWriter is just for benchmarking.
            items: Vec::with_capacity(cardinality + 16),
        }
    }
}

impl<T> AsRef<[T]> for UnsafeLookupWriter<T> {
    fn as_ref(&self) -> &[T] {
        &self.items
    }
}

impl<T> From<UnsafeLookupWriter<T>> for Vec<T> {
    fn from(value: UnsafeLookupWriter<T>) -> Self {
        value.items
    }
}

impl<T> Default for UnsafeLookupWriter<T> {
    fn default() -> Self {
        Self { items: Vec::default() }
    }
}

impl<T> Visitor<T> for UnsafeLookupWriter<T> {
    fn visit(&mut self, value: T) {
        unsafe {
            *self.items.as_mut_ptr().add(self.items.len()) = value;
            self.items.set_len(self.items.len() + 1);
        }
    }
}

impl<T> Clearable for UnsafeLookupWriter<T> {
    fn clear(&mut self) {
        self.items.clear();
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor4 for UnsafeLookupWriter<i32> {
    #[inline]
    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx512f")))]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled, mask, &mut self.items) };
    }

    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm_mask_compressstoreu_epi32(
                self.items.as_mut_ptr().add(self.items.len()) as *mut u8,
                mask as u8,
                value.into(),
            );
            self.items.set_len(self.items.len() + mask.count_ones() as usize);
        };
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor8 for UnsafeLookupWriter<i32> {
    #[cfg(all(target_feature = "avx2"))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let shuffled = permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled, mask, &mut self.items) };
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        let arr = value.as_array();
        let masks = [
            mask       & 0xF,
            mask >> 4  & 0xF,
        ];

        let shuffled1 = shuffle_epi8(i32x4::from_slice(&arr[..4]), VEC_SHUFFLE_MASK4[masks[0] as usize]);
        let shuffled2 = shuffle_epi8(i32x4::from_slice(&arr[4..]), VEC_SHUFFLE_MASK4[masks[1] as usize]);

        unsafe { unsafe_vec_extend(shuffled1, masks[0], &mut self.items) };
        unsafe { unsafe_vec_extend(shuffled2, masks[1], &mut self.items) };
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor16 for UnsafeLookupWriter<i32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let shuffled = permutevar_avx512(value, VEC_SHUFFLE_MASK16[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled, mask, &mut self.items) };
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let left = mask & 0xFF;
        let right = (mask >> 8) & 0xFF;

        let shuffled1 = permutevar8x32_epi32(i32x8::from_slice(&arr[..8]), VEC_SHUFFLE_MASK8[left as usize]);
        let shuffled2 = permutevar8x32_epi32(i32x8::from_slice(&arr[8..]), VEC_SHUFFLE_MASK8[right as usize]);

        unsafe { unsafe_vec_extend(shuffled1, left,  &mut self.items) };
        unsafe { unsafe_vec_extend(shuffled2, right, &mut self.items) };
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        let arr = value.as_array();
        let masks = [
            (mask       & 0xF) as u8,
            (mask >> 4  & 0xF) as u8,
            (mask >> 8  & 0xF) as u8,
            (mask >> 12 & 0xF) as u8,
        ];

        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]),   masks[0]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..8]),  masks[1]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[8..12]), masks[2]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[12..]),  masks[3]);

        let shuffled = [
            shuffle_epi8(i32x4::from_slice(&arr[..4]),  VEC_SHUFFLE_MASK4[masks[0] as usize]),
            shuffle_epi8(i32x4::from_slice(&arr[4..8]), VEC_SHUFFLE_MASK4[masks[1] as usize]),
            shuffle_epi8(i32x4::from_slice(&arr[8..12]), VEC_SHUFFLE_MASK4[masks[1] as usize]),
            shuffle_epi8(i32x4::from_slice(&arr[12..]), VEC_SHUFFLE_MASK4[masks[1] as usize]),
        ];

        unsafe { unsafe_vec_extend(shuffled[0], masks[0], &mut self.items) };
        unsafe { unsafe_vec_extend(shuffled[1], masks[1], &mut self.items) };
        unsafe { unsafe_vec_extend(shuffled[2], masks[2], &mut self.items) };
        unsafe { unsafe_vec_extend(shuffled[3], masks[3], &mut self.items) };
    }
}


// Unsafe writers: only for benchmarking!
// Always assumes the vec aleady has enough space.
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub struct UnsafeCompressWriter<T> {
    items: Vec<T>,
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> UnsafeCompressWriter<T> {
    pub fn new() -> Self {
        Self {
            items: Vec::new(),
        }
    }

    pub fn with_capacity(cardinality: usize) -> Self {
        Self {
            // For a final set size of x, we need to round up to nearest 16
            // to ensure we don't write past buffer with SIMD vector.
            // To be extra safe, we just add 16.
            // This is ok as UnsafeWriter is just for benchmarking.
            items: Vec::with_capacity(cardinality + 16),
        }
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> AsRef<[T]> for UnsafeCompressWriter<T> {
    fn as_ref(&self) -> &[T] {
        &self.items
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> From<UnsafeCompressWriter<T>> for Vec<T> {
    fn from(value: UnsafeCompressWriter<T>) -> Self {
        value.items
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> Default for UnsafeCompressWriter<T> {
    fn default() -> Self {
        Self { items: Vec::default() }
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> Visitor<T> for UnsafeCompressWriter<T> {
    fn visit(&mut self, value: T) {
        unsafe {
            *self.items.as_mut_ptr().add(self.items.len()) = value;
            self.items.set_len(self.items.len() + 1);
        }
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<T> Clearable for UnsafeCompressWriter<T> {
    fn clear(&mut self) {
        self.items.clear();
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdVisitor4 for UnsafeCompressWriter<i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm_mask_compressstoreu_epi32(
                self.items.as_mut_ptr().add(self.items.len()) as *mut u8,
                mask as u8,
                value.into(),
            );
            self.items.set_len(self.items.len() + mask.count_ones() as usize);
        };
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdVisitor8 for UnsafeCompressWriter<i32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm256_mask_compressstoreu_epi32(
                self.items.as_mut_ptr().add(self.items.len()) as *mut u8,
                mask as u8,
                value.into(),
            );
            self.items.set_len(self.items.len() + mask.count_ones() as usize);
        };
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdVisitor16 for UnsafeCompressWriter<i32> {
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm512_mask_compressstoreu_epi32(
                self.items.as_mut_ptr().add(self.items.len()) as *mut u8,
                mask as u16,
                value.into(),
            );
            self.items.set_len(self.items.len() + mask.count_ones() as usize);
        };
    }
}

unsafe fn unsafe_vec_extend<T, V, const LANES: usize>(
    value: Simd<T, LANES>,
    mask: u64,
    items: &mut Vec<V>)
where
    T: SimdElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
{
    debug_assert!(std::mem::size_of::<T>() == std::mem::size_of::<V>());
    debug_assert!(items.len() + LANES <= items.capacity());

    let write_ptr = items.as_mut_ptr().add(items.len())
        as *mut _ as *mut Simd<T, LANES>;
    write_ptr.write_unaligned(value);
    items.set_len(items.len() + mask.count_ones() as usize);
}

pub struct UnsafeLookupBsrWriter(BsrVec);

impl UnsafeLookupBsrWriter {
    pub fn new() -> Self {
        Self (BsrVec::new())
    }

    pub fn with_capacities(s: usize) -> Self {
        Self (BsrVec::with_capacities(s + 16))
    }
}

impl Into<BsrVec> for UnsafeLookupBsrWriter {
    fn into(self) -> BsrVec {
        self.0
    }
}

impl BsrVisitor for UnsafeLookupBsrWriter {
    fn visit_bsr(&mut self, base: u32, state: u32) {
        unsafe {
            *self.0.bases.as_mut_ptr().add(self.0.bases.len()) = base;
            self.0.bases.set_len(self.0.bases.len() + 1);
            *self.0.states.as_mut_ptr().add(self.0.states.len()) = state;
            self.0.states.set_len(self.0.states.len() + 1);
        }
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdBsrVisitor4 for UnsafeLookupBsrWriter {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u64) {
        let shuffled_base = shuffle_epi8(base, VEC_SHUFFLE_MASK4[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_base, mask, &mut self.0.bases) };

        let shuffled_state = shuffle_epi8(state, VEC_SHUFFLE_MASK4[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_state, mask, &mut self.0.states) };
    }
}
#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl SimdBsrVisitor8 for UnsafeLookupBsrWriter {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u64) {
        let shuffled_base = permutevar8x32_epi32(base, VEC_SHUFFLE_MASK8[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_base, mask, &mut self.0.bases) };

        let shuffled_state = permutevar8x32_epi32(state, VEC_SHUFFLE_MASK8[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_state, mask, &mut self.0.states) };
    }
}
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor16 for UnsafeLookupBsrWriter {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u64) {
        let shuffled_base = permutevar_avx512(base, VEC_SHUFFLE_MASK16[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_base, mask, &mut self.0.bases) };

        let shuffled_state = permutevar_avx512(state, VEC_SHUFFLE_MASK16[mask as usize]);
        unsafe { unsafe_vec_extend(shuffled_state, mask, &mut self.0.states) };
    }
}

impl<'a> From<&'a UnsafeLookupBsrWriter> for BsrRef<'a> {
    fn from(vec: &'a UnsafeLookupBsrWriter) -> Self {
        Self {
            bases: &vec.0.bases,
            states: &vec.0.states,
        }
    }
}

pub struct UnsafeCompressBsrWriter(BsrVec);

impl UnsafeCompressBsrWriter {
    pub fn new() -> Self {
        Self (BsrVec::new())
    }

    pub fn with_capacities(s: usize) -> Self {
        Self (BsrVec::with_capacities(s + 16))
    }
}

impl Into<BsrVec> for UnsafeCompressBsrWriter {
    fn into(self) -> BsrVec {
        self.0
    }
}

impl BsrVisitor for UnsafeCompressBsrWriter {
    fn visit_bsr(&mut self, base: u32, state: u32) {
        unsafe {
            *self.0.bases.as_mut_ptr().add(self.0.bases.len()) = base;
            self.0.bases.set_len(self.0.bases.len() + 1);
            *self.0.states.as_mut_ptr().add(self.0.states.len()) = state;
            self.0.states.set_len(self.0.states.len() + 1);
        }
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor4 for UnsafeCompressBsrWriter {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm_mask_compressstoreu_epi32(
                self.0.bases.as_mut_ptr().add(self.0.bases.len()) as *mut u8,
                mask as u8,
                base.into(),
            );
            self.0.bases.set_len(self.0.bases.len() + mask.count_ones() as usize);
        };
        unsafe {
            _mm_mask_compressstoreu_epi32(
                self.0.states.as_mut_ptr().add(self.0.states.len()) as *mut u8,
                mask as u8,
                state.into(),
            );
            self.0.states.set_len(self.0.states.len() + mask.count_ones() as usize);
        };
    }
}
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor8 for UnsafeCompressBsrWriter {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm256_mask_compressstoreu_epi32(
                self.0.bases.as_mut_ptr().add(self.0.bases.len()) as *mut u8,
                mask as u8,
                base.into(),
            );
            self.0.bases.set_len(self.0.bases.len() + mask.count_ones() as usize);
        };
        unsafe {
            _mm256_mask_compressstoreu_epi32(
                self.0.states.as_mut_ptr().add(self.0.states.len()) as *mut u8,
                mask as u8,
                state.into(),
            );
            self.0.states.set_len(self.0.states.len() + mask.count_ones() as usize);
        };
    }
}
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor16 for UnsafeCompressBsrWriter {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u64) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        unsafe {
            _mm512_mask_compressstoreu_epi32(
                self.0.bases.as_mut_ptr().add(self.0.bases.len()) as *mut u8,
                mask as u16,
                base.into(),
            );
            self.0.bases.set_len(self.0.bases.len() + mask.count_ones() as usize);
        };
        unsafe {
            _mm512_mask_compressstoreu_epi32(
                self.0.states.as_mut_ptr().add(self.0.states.len()) as *mut u8,
                mask as u16,
                state.into(),
            );
            self.0.states.set_len(self.0.states.len() + mask.count_ones() as usize);
        };
    }
}

impl<'a> From<&'a UnsafeCompressBsrWriter> for BsrRef<'a> {
    fn from(vec: &'a UnsafeCompressBsrWriter) -> Self {
        Self {
            bases: &vec.0.bases,
            states: &vec.0.states,
        }
    }
}
