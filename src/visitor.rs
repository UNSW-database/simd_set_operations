
#[cfg(feature = "simd")]
use {
    std::simd::{Simd, SimdElement, SupportedLaneCount, LaneCount},
    crate::instructions::{VEC_SHUFFLE_MASK4,VEC_SHUFFLE_MASK8}
};

/// Used to receive set intersection results in a generic way. Inspired by
/// roaring-rs.
pub trait Visitor<T> {
    fn visit(&mut self, value: T);
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

    fn clear(&mut self) {
        self.count = 0;
    }
}

impl Counter {
    pub fn count(&self) -> usize {
        self.count
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

    fn clear(&mut self) {
        self.position = 0;
    }
}

// SIMD //
#[cfg(feature = "simd")]
pub trait SimdVisitor<T, const LANES: usize> : Visitor<T>
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn visit_vector(&mut self, value: Simd<T, LANES>, mask: u8);
}

#[cfg(feature = "simd")]
impl<T, const LANES: usize> SimdVisitor<T, LANES> for Counter
where
    T: SimdElement,
    LaneCount<LANES>: SupportedLaneCount,
{
    fn visit_vector(&mut self, _value: Simd<T, LANES>, mask: u8) {
        self.count += mask.count_ones() as usize;
    }
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
impl SimdVisitor<i32, 4> for VecWriter<i32>
{
    #[inline]
    fn visit_vector(&mut self, value: core::simd::i32x4, mask: u8) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let result: core::simd::i32x4 = unsafe {
            _mm_shuffle_epi8(value.into(), VEC_SHUFFLE_MASK4[mask as usize].into())
        }.into();

        self.items.extend_from_slice(&result.as_array()[..]);
        // next truncate the masked out values
        self.items.truncate(self.items.len() - (result.lanes() - mask.count_ones() as usize));
    }
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
impl SimdVisitor<i32, 8> for VecWriter<i32>
{
    #[inline]
    fn visit_vector(&mut self, value: core::simd::i32x8, mask: u8) {
        #[cfg(target_arch = "x86")]
        use std::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use std::arch::x86_64::*;

        let result: core::simd::i32x8 = unsafe {
            _mm256_permutevar8x32_epi32(value.into(), VEC_SHUFFLE_MASK8[mask as usize].into())
        }.into();

        self.items.extend_from_slice(&result.as_array()[..]);
        // next truncate the masked out values
        self.items.truncate(self.items.len() - (result.lanes() - mask.count_ones() as usize));
    }
}
