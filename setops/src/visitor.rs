use crate::bsr::{BsrVec, BsrRef};
#[cfg(feature = "simd")]
use {
    std::simd::*,
    crate::{
        util::slice_i32_to_u32,
        instructions::{
            VEC_SHUFFLE_MASK4,
            shuffle_epi8,
        }
    }
};
#[cfg(all(feature = "simd", target_feature="avx2"))]
use crate::instructions::{VEC_SHUFFLE_MASK8, permutevar8x32_epi32};

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
pub trait SimdVisitor4<T: SimdElement> : Visitor<T> {
    fn visit_vector4(&mut self, value: Simd<T, 4>, mask: u8);
}
pub trait SimdVisitor8<T: SimdElement>: Visitor<T> {
    fn visit_vector8(&mut self, value: Simd<T, 8>, mask: u8);
}
pub trait SimdVisitor16<T: SimdElement>: Visitor<T> {
    fn visit_vector16(&mut self, value: Simd<T, 16>, mask: u16);
}

#[cfg(feature = "simd")]
impl<T> SimdVisitor4<T> for Counter
where
    T: SimdElement,
{
    fn visit_vector4(&mut self, _value: Simd<T, 4>, mask: u8) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(feature = "simd")]
impl<T> SimdVisitor8<T> for Counter
where
    T: SimdElement,
{
    fn visit_vector8(&mut self, _value: Simd<T, 8>, mask: u8) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(feature = "simd")]
impl<T> SimdVisitor16<T> for Counter
where
    T: SimdElement,
{
    fn visit_vector16(&mut self, _value: Simd<T, 16>, mask: u16) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor4<i32> for VecWriter<i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u8) {
        extend_i32vec_x4(&mut self.items, value, mask);
    }
}
#[cfg(feature = "simd")]
impl SimdVisitor8<i32> for VecWriter<i32> {
    #[cfg(target_feature = "avx2")]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u8) {
        extend_i32vec_x8(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u8) {
        let arr = value.as_array();
        let masks = [
            mask       & 0xF,
            mask >> 4  & 0xF,
        ];

        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[..4]), masks[0]);
        extend_i32vec_x4(&mut self.items, i32x4::from_slice(&arr[4..]), masks[1]);
    }
}
#[cfg(feature = "simd")]
impl SimdVisitor16<i32> for VecWriter<i32> {
    #[cfg(target_feature = "avx512f")]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u16) {
        extend_i32vec_x16(&mut self.items, value, mask);
    }

    #[cfg(all(target_feature = "avx2", not(target_feature = "avx512f")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u16) {
        let arr = value.as_array();
        let left = (mask & 0xFF) as u8;
        let right = ((mask >> 8) & 0xFF) as u8;

        extend_i32vec_x8(&mut self.items, i32x8::from_slice(&arr[..8]), left);
        extend_i32vec_x8(&mut self.items, i32x8::from_slice(&arr[8..]), right);
    }

    #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u16) {
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

/// Allows visiting of single entries in Base and State Representation
pub trait BsrVisitor {
    fn visit_bsr(&mut self, base: u32, state: u32);
}

/// Allows visiting of multiple entries in Base and State Representation
#[cfg(feature = "simd")]
pub trait SimdBsrVisitor4 : BsrVisitor {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u8);
}
pub trait SimdBsrVisitor8 : BsrVisitor {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u8);
}
pub trait SimdBsrVisitor16 : BsrVisitor {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u16);
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

#[cfg(feature = "simd")]
impl SimdBsrVisitor4 for BsrVec {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u8) {
        extend_u32vec_x4(&mut self.bases, base, mask);
        extend_u32vec_x4(&mut self.states, state, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl SimdBsrVisitor8 for BsrVec {
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u8) {
        extend_u32vec_x8(&mut self.bases, base, mask);
        extend_u32vec_x8(&mut self.states, state, mask);
    }
}
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl SimdBsrVisitor16 for BsrVec {
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u16) {
        extend_u32vec_x16(&mut self.bases, base, mask);
        extend_u32vec_x16(&mut self.states, state, mask);
    }
}

#[cfg(feature = "simd")]
impl SimdBsrVisitor4 for Counter {
    fn visit_bsr_vector4(&mut self, _base: i32x4, state: i32x4, mask: u8) {
        let masked_state = mask32x4::from_bitmask(mask).to_int() & state;
        let s = masked_state.as_array();

        let count =
            s[0].count_ones() + s[1].count_ones() +
            s[2].count_ones() + s[3].count_ones();
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
impl<'a> SimdVisitor4<i32> for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector4(&mut self, value: i32x4, mask: u8) {
        let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);

        let count = mask.count_ones() as usize;
        assert_eq!(&shuffled[..count],
            &self.expected[self.position..self.position+count]);

        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl<'a> SimdVisitor8<i32> for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector8(&mut self, value: i32x8, mask: u8) {
        let shuffled =
            permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

        let count = mask.count_ones() as usize;
        assert_eq!(&shuffled[..count],
            &self.expected[self.position..self.position+count]);

        self.position += count;
    }
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
impl<'a> SimdVisitor16<i32> for EnsureVisitor<'a, i32> {
    #[inline]
    fn visit_vector16(&mut self, value: i32x16, mask: u16) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let actual: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(),
            mask,
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

#[cfg(feature = "simd")]
impl<'a> SimdBsrVisitor4 for EnsureVisitorBsr<'a> {
    fn visit_bsr_vector4(&mut self, base: i32x4, state: i32x4, mask: u8) {
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
    fn visit_bsr_vector8(&mut self, base: i32x8, state: i32x8, mask: u8) {
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
    fn visit_bsr_vector16(&mut self, base: i32x16, state: i32x16, mask: u16) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let actual_base: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(), mask, base.into(),
        )}.into();
        let actual_state: i32x16 = unsafe { _mm512_mask_compress_epi32(
            i32x16::from_array([0;16]).into(), mask, state.into(),
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
fn extend_i32vec_x4(items: &mut Vec<i32>, value: i32x4, mask: u8) {
    let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
    extend_vec(items, &shuffled.as_array()[..], shuffled.lanes(), mask);
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline]
fn extend_u32vec_x4(items: &mut Vec<u32>, value: i32x4, mask: u8) {
    let shuffled = shuffle_epi8(value, VEC_SHUFFLE_MASK4[mask as usize]);
    extend_vec(
        items, slice_i32_to_u32(&shuffled.as_array()[..]),
        shuffled.lanes(), mask);
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline]
fn extend_i32vec_x8(items: &mut Vec<i32>, value: i32x8, mask: u8) {
    let shuffled =
        permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

    extend_vec(items, &shuffled.as_array()[..], shuffled.lanes(), mask);
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[inline]
fn extend_i32vec_x16(items: &mut Vec<i32>, value: i32x16, mask: u16) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    items.reserve(items.len() + 16);
    unsafe {
        _mm512_mask_compressstoreu_epi32(
            items.as_mut_ptr().add(items.len()) as *mut u8,
            mask,
            value.into(),
        );
        items.set_len(items.len() + mask.count_ones() as usize);
    };
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
#[inline]
fn extend_u32vec_x16(items: &mut Vec<u32>, value: i32x16, mask: u16) {
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    items.reserve(items.len() + 16);
    unsafe {
        _mm512_mask_compressstoreu_epi32(
            items.as_mut_ptr().add(items.len()) as *mut u8,
            mask,
            value.into(),
        );
        items.set_len(items.len() + mask.count_ones() as usize);
    };
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline]
fn extend_u32vec_x8(items: &mut Vec<u32>, value: i32x8, mask: u8) {

    let shuffled =
        permutevar8x32_epi32(value, VEC_SHUFFLE_MASK8[mask as usize]);

    extend_vec(
        items, slice_i32_to_u32(&shuffled.as_array()[..]),
        shuffled.lanes(), mask);
}

#[cfg(feature = "simd")]
#[inline]
fn extend_vec<T>(items: &mut Vec<T>, shuffled: &[T], lanes: usize, mask: u8)
where
    T: Clone
{
    items.extend_from_slice(shuffled);
    // Truncate the masked out values
    items.truncate(items.len() - (lanes - mask.count_ones() as usize));
}
