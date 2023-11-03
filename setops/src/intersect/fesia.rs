#![cfg(feature = "simd")]
/// Implementation of the FESIA set intersection algorithm from below paper.
/// Zhang, J., Lu, Y., Spampinato, D. G., & Franchetti, F. (2020, April). Fesia:
/// A fast and simd-efficient set intersection approach on modern cpus. In 2020
/// IEEE 36th International Conference on Data Engineering (ICDE) (pp.
/// 1465-1476). IEEE.

mod kernels_sse;
mod kernels_avx2;
mod kernels_avx512;

use std::{
    marker::PhantomData,
    num::Wrapping,
    simd::*,
    ops::BitAnd,
};
use smallvec::SmallVec;

use crate::{
    intersect,
    visitor::{SimdVisitor4, Visitor, SimdVisitor8, SimdVisitor16},
    instructions::load_unsafe,
};

// Use a power of 2 output space as this allows reducing the hash without skewing
const MIN_HASH_SIZE: usize = 16 * i32::BITS as usize; 

pub type Fesia8Sse     = Fesia<MixHash, i8,  u16, 16>;
pub type Fesia16Sse    = Fesia<MixHash, i16, u8,  8 >;
pub type Fesia32Sse    = Fesia<MixHash, i32, u8,  4 >;
pub type Fesia8Avx2    = Fesia<MixHash, i8,  u32, 32>;
pub type Fesia16Avx2   = Fesia<MixHash, i16, u16, 16>;
pub type Fesia32Avx2   = Fesia<MixHash, i32, u8,  8 >;
pub type Fesia8Avx512  = Fesia<MixHash, i8,  u64, 64>;
pub type Fesia16Avx512 = Fesia<MixHash, i16, u32, 32>;
pub type Fesia32Avx512 = Fesia<MixHash, i32, u16, 16>;

pub type HashScale = f64;

pub trait SetWithHashScale {
    fn from_sorted(sorted: &[i32], hash_scale: HashScale) -> Self;
}

pub trait FesiaIntersect {
    fn intersect<V, I>(&self, other: &Self, visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16,
        I: SegmentIntersect;

    fn hash_intersect(&self, other: &Self, visitor: &mut impl Visitor<i32>);

    fn intersect_k<S: AsRef<Self>>(sets: &[S], visitor: &mut impl Visitor<i32>);
}

#[derive(Clone, Copy, PartialEq)]
pub enum FesiaTwoSetMethod {
    SimilarSize,
    Skewed,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum FesiaKSetMethod {
    SimilarSize,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum SimdType {
    Sse,
    Avx2,
    Avx512,
}

pub struct Fesia<H, S, M, const LANES: usize>
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    bitmap: Vec<u8>,
    sizes: Vec<i32>,
    offsets: Vec<i32>,
    reordered_set: Vec<i32>,
    hash_size: usize,
    hash_t: PhantomData<H>,
    segment_t: PhantomData<S>,
}

impl<H, S, M, const LANES: usize> Fesia<H, S, M, LANES> 
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    pub fn segment_count(&self) -> usize {
        self.offsets.len()
    }

    pub fn debug_print(&self) {
        let iter = self.offsets.iter().zip(self.sizes.iter()).enumerate();
        for (i, (&offset, &size)) in iter {
            if size > 0 {
                println!("<{i}, {offset}> {:08x?}",
                    &self.reordered_set[offset as usize..(offset+size) as usize]);
            }
            else {
                print!("[] ");
            }
        }
    }

    pub fn to_sorted_set(&self) -> Vec<i32> {
        let mut result = self.reordered_set.clone();
        result.sort();
        result
    }

    fn fesia_intersect_block<V, I>(
        &self, other: &Self,
        base_segment: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16,
        I: SegmentIntersect,
    {
        debug_assert!(self.segment_count() <= other.segment_count());
        debug_assert!(base_segment <= other.segment_count() - self.segment_count());

        // Ensure we do not overflow into next block.
        let large_last_segment = base_segment + self.segment_count() - 1;
        let large_reordered_max = unsafe {
            *other.offsets.get_unchecked(large_last_segment) +
            *other.sizes.get_unchecked(large_last_segment)
         } as usize;

        let mut small_offset = 0;
        while small_offset < self.segment_count() {
            let large_offset = base_segment + small_offset;

            let pos_a = unsafe { (self.bitmap.as_ptr() as *const S).add(small_offset) };
            let pos_b = unsafe { (other.bitmap.as_ptr() as *const S).add(large_offset) };
            let v_a: Simd<S, LANES> = unsafe{ load_unsafe(pos_a) };
            let v_b: Simd<S, LANES> = unsafe{ load_unsafe(pos_b) };

            let and_result = v_a & v_b;
            let and_mask = and_result.simd_ne(Mask::<S, LANES>::from_array([false; LANES]).to_int());
            let mut mask = and_mask.to_bitmask();

            while !mask.is_zero() {
                let bit_offset = mask.trailing_zeros() as usize;
                mask = mask & (mask.sub(M::one()));

                let offset_a = *unsafe{ self.offsets.get_unchecked(small_offset + bit_offset) } as usize;
                let offset_b = *unsafe{ other.offsets.get_unchecked(large_offset + bit_offset) } as usize;
                let size_a = *unsafe{ self.sizes.get_unchecked(small_offset + bit_offset) } as usize;
                let size_b = *unsafe { other.sizes.get_unchecked(large_offset + bit_offset) } as usize;

                I::intersect(
                    unsafe{ self.reordered_set.get_unchecked(offset_a..) },
                    unsafe { other.reordered_set.get_unchecked(offset_b..large_reordered_max) },
                    size_a,
                    size_b,
                    visitor);
            }

            small_offset += LANES;
        }
    }
}

impl<H, S, M, const LANES: usize> FesiaIntersect for Fesia<H, S, M, LANES>
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    fn intersect<V, I>(&self, other: &Self, visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16,
        I: SegmentIntersect,
    {
        if self.segment_count() > other.segment_count() {
            return other.intersect::<V, I>(self, visitor);
        }
        debug_assert!(other.segment_count() % self.segment_count() == 0);

        for block in 0..other.segment_count() / self.segment_count() {
            let base = block * self.segment_count();
            self.fesia_intersect_block::<V, I>(other, base, visitor);
        }
    }

    fn hash_intersect(
        &self,
        other: &Self,
        visitor: &mut impl Visitor<i32>)
    {
        if self.reordered_set.len() > other.reordered_set.len() {
            return other.hash_intersect(self, visitor);
        }
        debug_assert!(other.hash_size % self.hash_size == 0);
        debug_assert!(other.segment_count() % self.segment_count() == 0);

        let segment_bits: usize = std::mem::size_of::<S>() * u8::BITS as usize;

        for &item in &self.reordered_set {
            let hash = masked_hash::<H>(item, other.hash_size);
            let segment_index = hash as usize / segment_bits;
            
            let offset = unsafe { *other.offsets.get_unchecked(segment_index) } as usize;
            let size = unsafe { *other.sizes.get_unchecked(segment_index) } as usize;
            
            let others = unsafe { other.reordered_set.get_unchecked(offset..offset+size) };
            for &other in others {
                if item == other {
                    visitor.visit(item);
                    break;
                }
            }
        }
    }

    fn intersect_k<F: AsRef<Self>>(sets: &[F], visitor: &mut impl Visitor<i32>) {
        debug_assert!(sets.windows(2).all(|s|
            s[1].as_ref().segment_count() >= s[0].as_ref().segment_count()
        ));
        debug_assert!(sets.windows(2).all(|s|
            s[1].as_ref().segment_count()  % s[0].as_ref().segment_count() == 0
        ));
        debug_assert!(sets.len() > 0);
        let last = sets.last().unwrap().as_ref();

        let mut last_offset = 0;

        while last_offset < last.segment_count() {
            let last_bitmap_pos = unsafe { (last.bitmap.as_ptr() as *const S).add(last_offset) };
            let mut and_result: Simd<S, LANES> = unsafe { load_unsafe(last_bitmap_pos) };

            for set in unsafe { sets.get_unchecked(..sets.len() - 1) } {
                let set = set.as_ref();
                // TODO: change this to segment_bits and use shift
                let set_offset = last_offset % set.segment_count();
                
                let set_bitmap_pos = unsafe { (set.bitmap.as_ptr() as *const S).add(set_offset) };
                let set_bitvec: Simd<S, LANES> = unsafe{ load_unsafe(set_bitmap_pos) };

                and_result &= set_bitvec;
            }

            let and_mask = and_result.simd_ne(Mask::<S, LANES>::from_array([false; LANES]).to_int());
            let mut mask = and_mask.to_bitmask();

            while !mask.is_zero() {
                let bit_offset = mask.trailing_zeros() as usize;
                mask = mask & (mask.sub(M::one()));

                merge_k(sets.iter().map(|set| {
                    let set = set.as_ref();
                    // TODO: change to bit shift
                    let segment_index = last_offset % set.segment_count();

                    let offset = unsafe { *set.offsets.get_unchecked(segment_index + bit_offset) } as usize;
                    let size = unsafe { *set.sizes.get_unchecked(segment_index + bit_offset) } as usize;

                    unsafe { set.reordered_set.get_unchecked(offset..offset+size) }
                }), visitor);
            }

            last_offset += LANES;
        }
    }
}

impl<H, S, M, const LANES: usize> AsRef<Fesia<H, S, M, LANES>> for Fesia<H, S, M, LANES>
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    fn as_ref(&self) -> &Fesia<H, S, M, LANES> {
        &self
    }
}

impl<H, S, M, const LANES: usize> SetWithHashScale for Fesia<H, S, M, LANES>
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    /// The authors propose a hash_scale of sqrt(w) is optimal where w is the
    /// SIMD width.
    fn from_sorted(sorted: &[i32], hash_scale: HashScale) -> Self {
        let segment_bits: usize = std::mem::size_of::<S>() * u8::BITS as usize;

        let hash_size = ((sorted.len() as f64 * hash_scale) as usize)
            .next_power_of_two()
            .max(MIN_HASH_SIZE);
        let segment_count = hash_size / segment_bits;
        let bitmap_len = hash_size / u8::BITS as usize;

        let mut bitmap: Vec<u8> = vec![0; bitmap_len];
        let mut sizes: Vec<i32> = vec![0; segment_count];

        let mut segments: Vec<SmallVec<[i32; 8]>> = vec![SmallVec::new(); segment_count];
        let mut offsets: Vec<i32> = Vec::with_capacity(segment_count);
        let mut reordered_set: Vec<i32> = Vec::with_capacity(sorted.len());

        for &item in sorted {
            let hash = masked_hash::<H>(item, hash_size);
            let segment_index = hash as usize / segment_bits;
            sizes[segment_index] += 1;
            segments[segment_index].push(item);

            let bitmap_index = hash as usize / u8::BITS as usize;
            bitmap[bitmap_index] |= 1 << (hash % u8::BITS as i32);
        }

        // let avg_segment_size =
        //     segments.iter().map(|s| s.len()).sum::<usize>() as f64 / segments.len() as f64;
        // let min_segment_size = segments.iter().map(|s| s.len()).min().unwrap();
        // let max_segment_size = segments.iter().map(|s| s.len()).max().unwrap();

        // let bitmap_density =
        //     bitmap.iter().map(|b| b.count_ones()).sum::<u32>() as f64
        //     / (bitmap.len() as u32 * u8::BITS) as f64;

        // println!("min {} avg {} max {} bitmap density {}",
        //     min_segment_size, avg_segment_size, max_segment_size,
        //     bitmap_density
        // );

        for segment in segments {
            // print!("{} ", segment.len());
            // println!("\n");
            offsets.push(reordered_set.len() as i32);
            reordered_set.extend_from_slice(&segment);
        }

        Self {
            bitmap,
            sizes,
            offsets,
            reordered_set,
            hash_size,
            hash_t: PhantomData,
            segment_t: PhantomData,
        }
    }
}

pub trait SegmentIntersect
{
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16;
}

pub struct SegmentIntersectSse;
impl SegmentIntersect for SegmentIntersectSse {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16
    {
        const MAX_KERNEL: usize = 7;
        const OVERFLOW: usize = 8;
        // Each kernel function may intersect up to set_a[..8], set_b[..8] even if
        // the reordered segment contains less than 8 elements. This won't lead to
        // false-positives as all elements in successive segments must hash to a
        // different value.
        if size_a > MAX_KERNEL || size_b > MAX_KERNEL ||
            set_a.len() < OVERFLOW || set_b.len() < OVERFLOW
        {
            return intersect::branchless_merge(
                unsafe { set_a.get_unchecked(..size_a) },
                unsafe { set_b.get_unchecked(..size_b) },
                visitor);
        }

        let left = set_a.as_ptr();
        let right = set_b.as_ptr();

        let ctrl = (size_a << 3) | size_b;
        match ctrl {
            0o11 => unsafe { kernels_sse::sse_1x4(left, right, visitor) }
            0o12 => unsafe { kernels_sse::sse_1x4(left, right, visitor) }
            0o13 => unsafe { kernels_sse::sse_1x4(left, right, visitor) }
            0o14 => unsafe { kernels_sse::sse_1x4(left, right, visitor) }
            0o15 => unsafe { kernels_sse::sse_1x8(left, right, visitor) }
            0o16 => unsafe { kernels_sse::sse_1x8(left, right, visitor) }
            0o17 => unsafe { kernels_sse::sse_1x8(left, right, visitor) }
            0o21 => unsafe { kernels_sse::sse_1x4(right, left, visitor) }
            0o22 => unsafe { kernels_sse::sse_2x4(left, right, visitor) }
            0o23 => unsafe { kernels_sse::sse_2x4(left, right, visitor) }
            0o24 => unsafe { kernels_sse::sse_2x4(left, right, visitor) }
            0o25 => unsafe { kernels_sse::sse_2x8(left, right, visitor) }
            0o26 => unsafe { kernels_sse::sse_2x8(left, right, visitor) }
            0o27 => unsafe { kernels_sse::sse_2x8(left, right, visitor) }
            0o31 => unsafe { kernels_sse::sse_1x4(right, left, visitor) }
            0o32 => unsafe { kernels_sse::sse_2x4(right, left, visitor) }
            0o33 => unsafe { kernels_sse::sse_3x4(left, right, visitor) }
            0o34 => unsafe { kernels_sse::sse_3x4(left, right, visitor) }
            0o35 => unsafe { kernels_sse::sse_3x8(left, right, visitor) }
            0o36 => unsafe { kernels_sse::sse_3x8(left, right, visitor) }
            0o37 => unsafe { kernels_sse::sse_3x8(left, right, visitor) }
            0o41 => unsafe { kernels_sse::sse_1x4(right, left, visitor) }
            0o42 => unsafe { kernels_sse::sse_2x4(right, left, visitor) }
            0o43 => unsafe { kernels_sse::sse_3x4(right, left, visitor) }
            0o44 => unsafe { kernels_sse::sse_4x4(left, right, visitor) }
            0o45 => unsafe { kernels_sse::sse_4x8(left, right, visitor) }
            0o46 => unsafe { kernels_sse::sse_4x8(left, right, visitor) }
            0o47 => unsafe { kernels_sse::sse_4x8(left, right, visitor) }
            0o51 => unsafe { kernels_sse::sse_1x8(right, left, visitor) }
            0o52 => unsafe { kernels_sse::sse_2x8(right, left, visitor) }
            0o53 => unsafe { kernels_sse::sse_3x8(right, left, visitor) }
            0o54 => unsafe { kernels_sse::sse_4x8(right, left, visitor) }
            0o55 => unsafe { kernels_sse::sse_5x8(left, right, visitor) }
            0o56 => unsafe { kernels_sse::sse_5x8(left, right, visitor) }
            0o57 => unsafe { kernels_sse::sse_5x8(left, right, visitor) }
            0o61 => unsafe { kernels_sse::sse_1x8(right, left, visitor) }
            0o62 => unsafe { kernels_sse::sse_2x8(right, left, visitor) }
            0o63 => unsafe { kernels_sse::sse_3x8(right, left, visitor) }
            0o64 => unsafe { kernels_sse::sse_4x8(right, left, visitor) }
            0o65 => unsafe { kernels_sse::sse_5x8(right, left, visitor) }
            0o66 => unsafe { kernels_sse::sse_6x8(left, right, visitor) }
            0o67 => unsafe { kernels_sse::sse_6x8(left, right, visitor) }
            0o71 => unsafe { kernels_sse::sse_1x8(right, left, visitor) }
            0o72 => unsafe { kernels_sse::sse_2x8(right, left, visitor) }
            0o73 => unsafe { kernels_sse::sse_3x8(right, left, visitor) }
            0o74 => unsafe { kernels_sse::sse_4x8(right, left, visitor) }
            0o75 => unsafe { kernels_sse::sse_5x8(right, left, visitor) }
            0o76 => unsafe { kernels_sse::sse_6x8(right, left, visitor) }
            0o77 => unsafe { kernels_sse::sse_7x8(left, right, visitor) }
            _ => panic!("Invalid kernel {:02o}", ctrl),
        }
    }
}

#[cfg(target_feature = "avx2")]
pub struct SegmentIntersectAvx2;
#[cfg(target_feature = "avx2")]
impl SegmentIntersect for SegmentIntersectAvx2 {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16
    {
        const MAX_KERNEL: usize = 15;
        const OVERFLOW: usize = 16;
        // Each kernel function may intersect up to set_a[..16], set_b[..16] even if
        // the reordered segment contains less than 8 elements. This won't lead to
        // false-positives as all elements in successive segments must hash to a
        // different value.
        if size_a > MAX_KERNEL || size_b > MAX_KERNEL ||
            set_a.len() < OVERFLOW || set_b.len() < OVERFLOW
        {
            return intersect::branchless_merge(
                unsafe { set_a.get_unchecked(..size_a) },
                unsafe { set_b.get_unchecked(..size_b) },
                visitor);
        }

        let left = set_a.as_ptr();
        let right = set_b.as_ptr();

        let ctrl = (size_a << 4) | size_b;
        match ctrl {
            0x11 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x12 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x13 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x14 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x15 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x16 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x17 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x18 => unsafe { kernels_avx2::avx2_1x8(left, right, visitor) }
            0x19 => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1a => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1b => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1c => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1d => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1e => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x1f => unsafe { kernels_avx2::avx2_1x16(left, right, visitor) }
            0x21 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x22 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x23 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x24 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x25 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x26 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x27 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x28 => unsafe { kernels_avx2::avx2_2x8(left, right, visitor) }
            0x29 => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2a => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2b => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2c => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2d => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2e => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x2f => unsafe { kernels_avx2::avx2_2x16(left, right, visitor) }
            0x31 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x32 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x33 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x34 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x35 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x36 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x37 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x38 => unsafe { kernels_avx2::avx2_3x8(left, right, visitor) }
            0x39 => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3a => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3b => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3c => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3d => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3e => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x3f => unsafe { kernels_avx2::avx2_3x16(left, right, visitor) }
            0x41 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x42 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x43 => unsafe { kernels_avx2::avx2_3x8(right, left, visitor) }
            0x44 => unsafe { kernels_avx2::avx2_4x8(left, right, visitor) }
            0x45 => unsafe { kernels_avx2::avx2_4x8(left, right, visitor) }
            0x46 => unsafe { kernels_avx2::avx2_4x8(left, right, visitor) }
            0x47 => unsafe { kernels_avx2::avx2_4x8(left, right, visitor) }
            0x48 => unsafe { kernels_avx2::avx2_4x8(left, right, visitor) }
            0x49 => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4a => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4b => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4c => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4d => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4e => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x4f => unsafe { kernels_avx2::avx2_4x16(left, right, visitor) }
            0x51 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x52 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x53 => unsafe { kernels_avx2::avx2_3x8(right, left, visitor) }
            0x54 => unsafe { kernels_avx2::avx2_4x8(right, left, visitor) }
            0x55 => unsafe { kernels_avx2::avx2_5x8(left, right, visitor) }
            0x56 => unsafe { kernels_avx2::avx2_5x8(left, right, visitor) }
            0x57 => unsafe { kernels_avx2::avx2_5x8(left, right, visitor) }
            0x58 => unsafe { kernels_avx2::avx2_5x8(left, right, visitor) }
            0x59 => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5a => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5b => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5c => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5d => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5e => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x5f => unsafe { kernels_avx2::avx2_5x16(left, right, visitor) }
            0x61 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x62 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x63 => unsafe { kernels_avx2::avx2_3x8(right, left, visitor) }
            0x64 => unsafe { kernels_avx2::avx2_4x8(right, left, visitor) }
            0x65 => unsafe { kernels_avx2::avx2_5x8(right, left, visitor) }
            0x66 => unsafe { kernels_avx2::avx2_6x8(left, right, visitor) }
            0x67 => unsafe { kernels_avx2::avx2_6x8(left, right, visitor) }
            0x68 => unsafe { kernels_avx2::avx2_6x8(left, right, visitor) }
            0x69 => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6a => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6b => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6c => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6d => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6e => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x6f => unsafe { kernels_avx2::avx2_6x16(left, right, visitor) }
            0x71 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x72 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x73 => unsafe { kernels_avx2::avx2_3x8(right, left, visitor) }
            0x74 => unsafe { kernels_avx2::avx2_4x8(right, left, visitor) }
            0x75 => unsafe { kernels_avx2::avx2_5x8(right, left, visitor) }
            0x76 => unsafe { kernels_avx2::avx2_6x8(right, left, visitor) }
            0x77 => unsafe { kernels_avx2::avx2_7x8(left, right, visitor) }
            0x78 => unsafe { kernels_avx2::avx2_7x8(left, right, visitor) }
            0x79 => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7a => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7b => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7c => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7d => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7e => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x7f => unsafe { kernels_avx2::avx2_7x16(left, right, visitor) }
            0x81 => unsafe { kernels_avx2::avx2_1x8(right, left, visitor) }
            0x82 => unsafe { kernels_avx2::avx2_2x8(right, left, visitor) }
            0x83 => unsafe { kernels_avx2::avx2_3x8(right, left, visitor) }
            0x84 => unsafe { kernels_avx2::avx2_4x8(right, left, visitor) }
            0x85 => unsafe { kernels_avx2::avx2_5x8(right, left, visitor) }
            0x86 => unsafe { kernels_avx2::avx2_6x8(right, left, visitor) }
            0x87 => unsafe { kernels_avx2::avx2_7x8(right, left, visitor) }
            0x88 => unsafe { kernels_avx2::avx2_8x8(left, right, visitor) }
            0x89 => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8a => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8b => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8c => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8d => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8e => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x8f => unsafe { kernels_avx2::avx2_8x16(left, right, visitor) }
            0x91 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0x92 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0x93 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0x94 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0x95 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0x96 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0x97 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0x98 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0x99 => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9a => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9b => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9c => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9d => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9e => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0x9f => unsafe { kernels_avx2::avx2_9x16(left, right, visitor) }
            0xa1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xa2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xa3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xa4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xa5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xa6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xa7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xa8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xa9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xaa => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xab => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xac => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xad => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xae => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xaf => unsafe { kernels_avx2::avx2_10x16(left, right, visitor) }
            0xb1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xb2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xb3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xb4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xb5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xb6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xb7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xb8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xb9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xba => unsafe { kernels_avx2::avx2_10x16(right, left, visitor) }
            0xbb => unsafe { kernels_avx2::avx2_11x16(left, right, visitor) }
            0xbc => unsafe { kernels_avx2::avx2_11x16(left, right, visitor) }
            0xbd => unsafe { kernels_avx2::avx2_11x16(left, right, visitor) }
            0xbe => unsafe { kernels_avx2::avx2_11x16(left, right, visitor) }
            0xbf => unsafe { kernels_avx2::avx2_11x16(left, right, visitor) }
            0xc1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xc2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xc3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xc4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xc5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xc6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xc7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xc8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xc9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xca => unsafe { kernels_avx2::avx2_10x16(right, left, visitor) }
            0xcb => unsafe { kernels_avx2::avx2_11x16(right, left, visitor) }
            0xcc => unsafe { kernels_avx2::avx2_12x16(left, right, visitor) }
            0xcd => unsafe { kernels_avx2::avx2_12x16(left, right, visitor) }
            0xce => unsafe { kernels_avx2::avx2_12x16(left, right, visitor) }
            0xcf => unsafe { kernels_avx2::avx2_12x16(left, right, visitor) }
            0xd1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xd2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xd3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xd4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xd5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xd6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xd7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xd8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xd9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xda => unsafe { kernels_avx2::avx2_10x16(right, left, visitor) }
            0xdb => unsafe { kernels_avx2::avx2_11x16(right, left, visitor) }
            0xdc => unsafe { kernels_avx2::avx2_12x16(right, left, visitor) }
            0xdd => unsafe { kernels_avx2::avx2_13x16(left, right, visitor) }
            0xde => unsafe { kernels_avx2::avx2_13x16(left, right, visitor) }
            0xdf => unsafe { kernels_avx2::avx2_13x16(left, right, visitor) }
            0xe1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xe2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xe3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xe4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xe5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xe6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xe7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xe8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xe9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xea => unsafe { kernels_avx2::avx2_10x16(right, left, visitor) }
            0xeb => unsafe { kernels_avx2::avx2_11x16(right, left, visitor) }
            0xec => unsafe { kernels_avx2::avx2_12x16(right, left, visitor) }
            0xed => unsafe { kernels_avx2::avx2_13x16(right, left, visitor) }
            0xee => unsafe { kernels_avx2::avx2_14x16(left, right, visitor) }
            0xef => unsafe { kernels_avx2::avx2_14x16(left, right, visitor) }
            0xf1 => unsafe { kernels_avx2::avx2_1x16(right, left, visitor) }
            0xf2 => unsafe { kernels_avx2::avx2_2x16(right, left, visitor) }
            0xf3 => unsafe { kernels_avx2::avx2_3x16(right, left, visitor) }
            0xf4 => unsafe { kernels_avx2::avx2_4x16(right, left, visitor) }
            0xf5 => unsafe { kernels_avx2::avx2_5x16(right, left, visitor) }
            0xf6 => unsafe { kernels_avx2::avx2_6x16(right, left, visitor) }
            0xf7 => unsafe { kernels_avx2::avx2_7x16(right, left, visitor) }
            0xf8 => unsafe { kernels_avx2::avx2_8x16(right, left, visitor) }
            0xf9 => unsafe { kernels_avx2::avx2_9x16(right, left, visitor) }
            0xfa => unsafe { kernels_avx2::avx2_10x16(right, left, visitor) }
            0xfb => unsafe { kernels_avx2::avx2_11x16(right, left, visitor) }
            0xfc => unsafe { kernels_avx2::avx2_12x16(right, left, visitor) }
            0xfd => unsafe { kernels_avx2::avx2_13x16(right, left, visitor) }
            0xfe => unsafe { kernels_avx2::avx2_14x16(right, left, visitor) }
            0xff => unsafe { kernels_avx2::avx2_15x16(left, right, visitor) }
            _ => panic!("Invalid kernel {:02o}", ctrl),
        }
    }
}

#[cfg(target_feature = "avx512f")]
pub struct SegmentIntersectAvx512;
#[cfg(target_feature = "avx512f")]
impl SegmentIntersect for SegmentIntersectAvx512 {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4 + SimdVisitor8 + SimdVisitor16
    {
        const MAX_KERNEL: usize = 31;
        const OVERFLOW: usize = 32;
        // Each kernel function may intersect up to set_a[..16], set_b[..16] even if
        // the reordered segment contains less than 8 elements. This won't lead to
        // false-positives as all elements in successive segments must hash to a
        // different value.
        if size_a > MAX_KERNEL || size_b > MAX_KERNEL ||
            set_a.len() < OVERFLOW || set_b.len() < OVERFLOW
        {
            return intersect::branchless_merge(
                unsafe { set_a.get_unchecked(..size_a) },
                unsafe { set_b.get_unchecked(..size_b) },
                visitor);
        }

        let left = set_a.as_ptr();
        let right = set_b.as_ptr();

        let ctrl = (size_a << 5) | size_b;
        match ctrl {
            33 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            34 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            35 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            36 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            37 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            38 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            39 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            40 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            41 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            42 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            43 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            44 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            45 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            46 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            47 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            48 => unsafe { kernels_avx512::avx512_1x16(left, right, visitor) }
            49 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            50 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            51 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            52 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            53 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            54 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            55 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            56 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            57 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            58 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            59 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            60 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            61 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            62 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            63 => unsafe { kernels_avx512::avx512_1x32(left, right, visitor) }
            65 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            66 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            67 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            68 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            69 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            70 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            71 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            72 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            73 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            74 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            75 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            76 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            77 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            78 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            79 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            80 => unsafe { kernels_avx512::avx512_2x16(left, right, visitor) }
            81 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            82 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            83 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            84 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            85 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            86 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            87 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            88 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            89 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            90 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            91 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            92 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            93 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            94 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            95 => unsafe { kernels_avx512::avx512_2x32(left, right, visitor) }
            97 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            98 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            99 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            100 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            101 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            102 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            103 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            104 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            105 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            106 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            107 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            108 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            109 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            110 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            111 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            112 => unsafe { kernels_avx512::avx512_3x16(left, right, visitor) }
            113 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            114 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            115 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            116 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            117 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            118 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            119 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            120 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            121 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            122 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            123 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            124 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            125 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            126 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            127 => unsafe { kernels_avx512::avx512_3x32(left, right, visitor) }
            129 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            130 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            131 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            132 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            133 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            134 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            135 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            136 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            137 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            138 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            139 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            140 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            141 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            142 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            143 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            144 => unsafe { kernels_avx512::avx512_4x16(left, right, visitor) }
            145 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            146 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            147 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            148 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            149 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            150 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            151 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            152 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            153 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            154 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            155 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            156 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            157 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            158 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            159 => unsafe { kernels_avx512::avx512_4x32(left, right, visitor) }
            161 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            162 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            163 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            164 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            165 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            166 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            167 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            168 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            169 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            170 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            171 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            172 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            173 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            174 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            175 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            176 => unsafe { kernels_avx512::avx512_5x16(left, right, visitor) }
            177 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            178 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            179 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            180 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            181 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            182 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            183 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            184 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            185 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            186 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            187 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            188 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            189 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            190 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            191 => unsafe { kernels_avx512::avx512_5x32(left, right, visitor) }
            193 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            194 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            195 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            196 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            197 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            198 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            199 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            200 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            201 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            202 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            203 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            204 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            205 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            206 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            207 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            208 => unsafe { kernels_avx512::avx512_6x16(left, right, visitor) }
            209 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            210 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            211 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            212 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            213 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            214 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            215 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            216 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            217 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            218 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            219 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            220 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            221 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            222 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            223 => unsafe { kernels_avx512::avx512_6x32(left, right, visitor) }
            225 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            226 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            227 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            228 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            229 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            230 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            231 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            232 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            233 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            234 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            235 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            236 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            237 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            238 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            239 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            240 => unsafe { kernels_avx512::avx512_7x16(left, right, visitor) }
            241 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            242 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            243 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            244 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            245 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            246 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            247 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            248 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            249 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            250 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            251 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            252 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            253 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            254 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            255 => unsafe { kernels_avx512::avx512_7x32(left, right, visitor) }
            257 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            258 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            259 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            260 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            261 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            262 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            263 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            264 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            265 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            266 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            267 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            268 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            269 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            270 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            271 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            272 => unsafe { kernels_avx512::avx512_8x16(left, right, visitor) }
            273 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            274 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            275 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            276 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            277 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            278 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            279 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            280 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            281 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            282 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            283 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            284 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            285 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            286 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            287 => unsafe { kernels_avx512::avx512_8x32(left, right, visitor) }
            289 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            290 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            291 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            292 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            293 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            294 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            295 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            296 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            297 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            298 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            299 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            300 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            301 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            302 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            303 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            304 => unsafe { kernels_avx512::avx512_9x16(left, right, visitor) }
            305 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            306 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            307 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            308 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            309 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            310 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            311 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            312 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            313 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            314 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            315 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            316 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            317 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            318 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            319 => unsafe { kernels_avx512::avx512_9x32(left, right, visitor) }
            321 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            322 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            323 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            324 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            325 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            326 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            327 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            328 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            329 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            330 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            331 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            332 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            333 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            334 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            335 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            336 => unsafe { kernels_avx512::avx512_10x16(left, right, visitor) }
            337 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            338 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            339 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            340 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            341 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            342 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            343 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            344 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            345 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            346 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            347 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            348 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            349 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            350 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            351 => unsafe { kernels_avx512::avx512_10x32(left, right, visitor) }
            353 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            354 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            355 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            356 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            357 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            358 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            359 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            360 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            361 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            362 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            363 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            364 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            365 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            366 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            367 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            368 => unsafe { kernels_avx512::avx512_11x16(left, right, visitor) }
            369 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            370 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            371 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            372 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            373 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            374 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            375 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            376 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            377 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            378 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            379 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            380 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            381 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            382 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            383 => unsafe { kernels_avx512::avx512_11x32(left, right, visitor) }
            385 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            386 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            387 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            388 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            389 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            390 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            391 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            392 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            393 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            394 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            395 => unsafe { kernels_avx512::avx512_11x16(right, left, visitor) }
            396 => unsafe { kernels_avx512::avx512_12x16(left, right, visitor) }
            397 => unsafe { kernels_avx512::avx512_12x16(left, right, visitor) }
            398 => unsafe { kernels_avx512::avx512_12x16(left, right, visitor) }
            399 => unsafe { kernels_avx512::avx512_12x16(left, right, visitor) }
            400 => unsafe { kernels_avx512::avx512_12x16(left, right, visitor) }
            401 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            402 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            403 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            404 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            405 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            406 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            407 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            408 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            409 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            410 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            411 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            412 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            413 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            414 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            415 => unsafe { kernels_avx512::avx512_12x32(left, right, visitor) }
            417 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            418 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            419 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            420 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            421 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            422 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            423 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            424 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            425 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            426 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            427 => unsafe { kernels_avx512::avx512_11x16(right, left, visitor) }
            428 => unsafe { kernels_avx512::avx512_12x16(right, left, visitor) }
            429 => unsafe { kernels_avx512::avx512_13x16(left, right, visitor) }
            430 => unsafe { kernels_avx512::avx512_13x16(left, right, visitor) }
            431 => unsafe { kernels_avx512::avx512_13x16(left, right, visitor) }
            432 => unsafe { kernels_avx512::avx512_13x16(left, right, visitor) }
            433 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            434 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            435 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            436 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            437 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            438 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            439 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            440 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            441 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            442 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            443 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            444 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            445 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            446 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            447 => unsafe { kernels_avx512::avx512_13x32(left, right, visitor) }
            449 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            450 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            451 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            452 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            453 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            454 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            455 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            456 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            457 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            458 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            459 => unsafe { kernels_avx512::avx512_11x16(right, left, visitor) }
            460 => unsafe { kernels_avx512::avx512_12x16(right, left, visitor) }
            461 => unsafe { kernels_avx512::avx512_13x16(right, left, visitor) }
            462 => unsafe { kernels_avx512::avx512_14x16(left, right, visitor) }
            463 => unsafe { kernels_avx512::avx512_14x16(left, right, visitor) }
            464 => unsafe { kernels_avx512::avx512_14x16(left, right, visitor) }
            465 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            466 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            467 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            468 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            469 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            470 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            471 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            472 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            473 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            474 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            475 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            476 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            477 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            478 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            479 => unsafe { kernels_avx512::avx512_14x32(left, right, visitor) }
            481 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            482 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            483 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            484 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            485 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            486 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            487 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            488 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            489 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            490 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            491 => unsafe { kernels_avx512::avx512_11x16(right, left, visitor) }
            492 => unsafe { kernels_avx512::avx512_12x16(right, left, visitor) }
            493 => unsafe { kernels_avx512::avx512_13x16(right, left, visitor) }
            494 => unsafe { kernels_avx512::avx512_14x16(right, left, visitor) }
            495 => unsafe { kernels_avx512::avx512_15x16(left, right, visitor) }
            496 => unsafe { kernels_avx512::avx512_15x16(left, right, visitor) }
            497 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            498 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            499 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            500 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            501 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            502 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            503 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            504 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            505 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            506 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            507 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            508 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            509 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            510 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            511 => unsafe { kernels_avx512::avx512_15x32(left, right, visitor) }
            513 => unsafe { kernels_avx512::avx512_1x16(right, left, visitor) }
            514 => unsafe { kernels_avx512::avx512_2x16(right, left, visitor) }
            515 => unsafe { kernels_avx512::avx512_3x16(right, left, visitor) }
            516 => unsafe { kernels_avx512::avx512_4x16(right, left, visitor) }
            517 => unsafe { kernels_avx512::avx512_5x16(right, left, visitor) }
            518 => unsafe { kernels_avx512::avx512_6x16(right, left, visitor) }
            519 => unsafe { kernels_avx512::avx512_7x16(right, left, visitor) }
            520 => unsafe { kernels_avx512::avx512_8x16(right, left, visitor) }
            521 => unsafe { kernels_avx512::avx512_9x16(right, left, visitor) }
            522 => unsafe { kernels_avx512::avx512_10x16(right, left, visitor) }
            523 => unsafe { kernels_avx512::avx512_11x16(right, left, visitor) }
            524 => unsafe { kernels_avx512::avx512_12x16(right, left, visitor) }
            525 => unsafe { kernels_avx512::avx512_13x16(right, left, visitor) }
            526 => unsafe { kernels_avx512::avx512_14x16(right, left, visitor) }
            527 => unsafe { kernels_avx512::avx512_15x16(right, left, visitor) }
            528 => unsafe { kernels_avx512::avx512_16x16(left, right, visitor) }
            529 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            530 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            531 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            532 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            533 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            534 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            535 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            536 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            537 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            538 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            539 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            540 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            541 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            542 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            543 => unsafe { kernels_avx512::avx512_16x32(left, right, visitor) }
            545 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            546 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            547 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            548 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            549 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            550 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            551 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            552 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            553 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            554 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            555 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            556 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            557 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            558 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            559 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            560 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            561 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            562 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            563 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            564 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            565 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            566 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            567 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            568 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            569 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            570 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            571 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            572 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            573 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            574 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            575 => unsafe { kernels_avx512::avx512_17x32(left, right, visitor) }
            577 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            578 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            579 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            580 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            581 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            582 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            583 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            584 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            585 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            586 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            587 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            588 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            589 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            590 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            591 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            592 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            593 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            594 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            595 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            596 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            597 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            598 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            599 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            600 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            601 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            602 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            603 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            604 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            605 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            606 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            607 => unsafe { kernels_avx512::avx512_18x32(left, right, visitor) }
            609 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            610 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            611 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            612 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            613 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            614 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            615 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            616 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            617 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            618 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            619 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            620 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            621 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            622 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            623 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            624 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            625 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            626 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            627 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            628 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            629 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            630 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            631 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            632 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            633 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            634 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            635 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            636 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            637 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            638 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            639 => unsafe { kernels_avx512::avx512_19x32(left, right, visitor) }
            641 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            642 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            643 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            644 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            645 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            646 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            647 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            648 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            649 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            650 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            651 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            652 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            653 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            654 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            655 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            656 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            657 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            658 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            659 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            660 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            661 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            662 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            663 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            664 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            665 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            666 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            667 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            668 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            669 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            670 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            671 => unsafe { kernels_avx512::avx512_20x32(left, right, visitor) }
            673 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            674 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            675 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            676 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            677 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            678 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            679 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            680 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            681 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            682 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            683 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            684 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            685 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            686 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            687 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            688 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            689 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            690 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            691 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            692 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            693 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            694 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            695 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            696 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            697 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            698 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            699 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            700 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            701 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            702 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            703 => unsafe { kernels_avx512::avx512_21x32(left, right, visitor) }
            705 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            706 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            707 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            708 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            709 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            710 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            711 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            712 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            713 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            714 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            715 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            716 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            717 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            718 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            719 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            720 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            721 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            722 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            723 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            724 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            725 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            726 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            727 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            728 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            729 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            730 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            731 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            732 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            733 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            734 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            735 => unsafe { kernels_avx512::avx512_22x32(left, right, visitor) }
            737 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            738 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            739 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            740 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            741 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            742 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            743 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            744 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            745 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            746 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            747 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            748 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            749 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            750 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            751 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            752 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            753 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            754 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            755 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            756 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            757 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            758 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            759 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            760 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            761 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            762 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            763 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            764 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            765 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            766 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            767 => unsafe { kernels_avx512::avx512_23x32(left, right, visitor) }
            769 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            770 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            771 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            772 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            773 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            774 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            775 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            776 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            777 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            778 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            779 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            780 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            781 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            782 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            783 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            784 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            785 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            786 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            787 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            788 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            789 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            790 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            791 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            792 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            793 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            794 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            795 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            796 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            797 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            798 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            799 => unsafe { kernels_avx512::avx512_24x32(left, right, visitor) }
            801 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            802 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            803 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            804 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            805 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            806 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            807 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            808 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            809 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            810 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            811 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            812 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            813 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            814 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            815 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            816 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            817 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            818 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            819 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            820 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            821 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            822 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            823 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            824 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            825 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            826 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            827 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            828 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            829 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            830 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            831 => unsafe { kernels_avx512::avx512_25x32(left, right, visitor) }
            833 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            834 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            835 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            836 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            837 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            838 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            839 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            840 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            841 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            842 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            843 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            844 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            845 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            846 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            847 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            848 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            849 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            850 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            851 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            852 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            853 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            854 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            855 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            856 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            857 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            858 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            859 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            860 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            861 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            862 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            863 => unsafe { kernels_avx512::avx512_26x32(left, right, visitor) }
            865 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            866 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            867 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            868 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            869 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            870 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            871 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            872 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            873 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            874 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            875 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            876 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            877 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            878 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            879 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            880 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            881 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            882 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            883 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            884 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            885 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            886 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            887 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            888 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            889 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            890 => unsafe { kernels_avx512::avx512_26x32(right, left, visitor) }
            891 => unsafe { kernels_avx512::avx512_27x32(left, right, visitor) }
            892 => unsafe { kernels_avx512::avx512_27x32(left, right, visitor) }
            893 => unsafe { kernels_avx512::avx512_27x32(left, right, visitor) }
            894 => unsafe { kernels_avx512::avx512_27x32(left, right, visitor) }
            895 => unsafe { kernels_avx512::avx512_27x32(left, right, visitor) }
            897 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            898 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            899 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            900 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            901 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            902 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            903 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            904 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            905 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            906 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            907 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            908 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            909 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            910 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            911 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            912 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            913 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            914 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            915 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            916 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            917 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            918 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            919 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            920 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            921 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            922 => unsafe { kernels_avx512::avx512_26x32(right, left, visitor) }
            923 => unsafe { kernels_avx512::avx512_27x32(right, left, visitor) }
            924 => unsafe { kernels_avx512::avx512_28x32(left, right, visitor) }
            925 => unsafe { kernels_avx512::avx512_28x32(left, right, visitor) }
            926 => unsafe { kernels_avx512::avx512_28x32(left, right, visitor) }
            927 => unsafe { kernels_avx512::avx512_28x32(left, right, visitor) }
            929 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            930 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            931 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            932 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            933 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            934 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            935 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            936 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            937 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            938 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            939 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            940 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            941 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            942 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            943 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            944 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            945 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            946 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            947 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            948 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            949 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            950 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            951 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            952 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            953 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            954 => unsafe { kernels_avx512::avx512_26x32(right, left, visitor) }
            955 => unsafe { kernels_avx512::avx512_27x32(right, left, visitor) }
            956 => unsafe { kernels_avx512::avx512_28x32(right, left, visitor) }
            957 => unsafe { kernels_avx512::avx512_29x32(left, right, visitor) }
            958 => unsafe { kernels_avx512::avx512_29x32(left, right, visitor) }
            959 => unsafe { kernels_avx512::avx512_29x32(left, right, visitor) }
            961 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            962 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            963 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            964 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            965 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            966 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            967 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            968 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            969 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            970 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            971 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            972 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            973 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            974 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            975 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            976 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            977 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            978 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            979 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            980 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            981 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            982 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            983 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            984 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            985 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            986 => unsafe { kernels_avx512::avx512_26x32(right, left, visitor) }
            987 => unsafe { kernels_avx512::avx512_27x32(right, left, visitor) }
            988 => unsafe { kernels_avx512::avx512_28x32(right, left, visitor) }
            989 => unsafe { kernels_avx512::avx512_29x32(right, left, visitor) }
            990 => unsafe { kernels_avx512::avx512_30x32(left, right, visitor) }
            991 => unsafe { kernels_avx512::avx512_30x32(left, right, visitor) }
            993 => unsafe { kernels_avx512::avx512_1x32(right, left, visitor) }
            994 => unsafe { kernels_avx512::avx512_2x32(right, left, visitor) }
            995 => unsafe { kernels_avx512::avx512_3x32(right, left, visitor) }
            996 => unsafe { kernels_avx512::avx512_4x32(right, left, visitor) }
            997 => unsafe { kernels_avx512::avx512_5x32(right, left, visitor) }
            998 => unsafe { kernels_avx512::avx512_6x32(right, left, visitor) }
            999 => unsafe { kernels_avx512::avx512_7x32(right, left, visitor) }
            1000 => unsafe { kernels_avx512::avx512_8x32(right, left, visitor) }
            1001 => unsafe { kernels_avx512::avx512_9x32(right, left, visitor) }
            1002 => unsafe { kernels_avx512::avx512_10x32(right, left, visitor) }
            1003 => unsafe { kernels_avx512::avx512_11x32(right, left, visitor) }
            1004 => unsafe { kernels_avx512::avx512_12x32(right, left, visitor) }
            1005 => unsafe { kernels_avx512::avx512_13x32(right, left, visitor) }
            1006 => unsafe { kernels_avx512::avx512_14x32(right, left, visitor) }
            1007 => unsafe { kernels_avx512::avx512_15x32(right, left, visitor) }
            1008 => unsafe { kernels_avx512::avx512_16x32(right, left, visitor) }
            1009 => unsafe { kernels_avx512::avx512_17x32(right, left, visitor) }
            1010 => unsafe { kernels_avx512::avx512_18x32(right, left, visitor) }
            1011 => unsafe { kernels_avx512::avx512_19x32(right, left, visitor) }
            1012 => unsafe { kernels_avx512::avx512_20x32(right, left, visitor) }
            1013 => unsafe { kernels_avx512::avx512_21x32(right, left, visitor) }
            1014 => unsafe { kernels_avx512::avx512_22x32(right, left, visitor) }
            1015 => unsafe { kernels_avx512::avx512_23x32(right, left, visitor) }
            1016 => unsafe { kernels_avx512::avx512_24x32(right, left, visitor) }
            1017 => unsafe { kernels_avx512::avx512_25x32(right, left, visitor) }
            1018 => unsafe { kernels_avx512::avx512_26x32(right, left, visitor) }
            1019 => unsafe { kernels_avx512::avx512_27x32(right, left, visitor) }
            1020 => unsafe { kernels_avx512::avx512_28x32(right, left, visitor) }
            1021 => unsafe { kernels_avx512::avx512_29x32(right, left, visitor) }
            1022 => unsafe { kernels_avx512::avx512_30x32(right, left, visitor) }
            1023 => unsafe { kernels_avx512::avx512_31x32(left, right, visitor) }
            _ => panic!("Invalid kernel {:02}", ctrl),
        }
    }
}

fn masked_hash<H: IntegerHash>(item: i32, segment_count: usize) -> i32 {
    debug_assert!(segment_count.count_ones() == 1);
    H::hash(item) & (segment_count as i32 - 1)
}


pub trait IntegerHash {
    fn hash(item: i32) -> i32;
}

pub struct IdentityHash;
impl IntegerHash for IdentityHash {
    fn hash(item: i32) -> i32 {
        item
    }
}

pub struct MixHash;
impl IntegerHash for MixHash {
    // https://gist.github.com/badboy/6267743
    fn hash(item: i32) -> i32 {
        let mut key = Wrapping(item as i32);
        key = !key + (key << 15); // key = (key << 15) - key - 1;
        key = key ^ (key >> 12);
        key = key + (key << 2);
        key = key ^ (key >> 4);
        key = key * Wrapping(2057); // key = (key + (key << 3)) + (key << 11);
        key = key ^ (key >> 16);
        key.0 as i32
    }
}

/// Similar to `small_adaptive` but uses linear search instead of galloping.
pub fn merge_k<'a, T, V, I>(sets: I, visitor: &mut V)
where
    T: Ord + Copy + 'a,
    V: Visitor<T>,
    I: Iterator<Item=&'a [T]>,
{
    let mut set_spans: SmallVec<[&[T]; 8]> = sets.collect();

    set_spans.sort_unstable_by_key(|s| s.len());

    'target_loop:
    for &target in set_spans[0] {
        'set_loop:
        for set in &mut set_spans[1..] {
            for (i, &item) in set.iter().enumerate() {
                if target < item {
                    // `target` not found
                    *set = &set[i..];
                    continue 'target_loop;
                }
                else if item == target {
                    // `target` in current set, keep going.
                    *set = &set[i+1..];
                    continue 'set_loop;
                }
            }
            return;
        }
        visitor.visit(target);
    }
}

// Used with cargo-show-asm to verify correct instructions are being used.
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline(never)]
pub fn test8_sse(
    left: &Fesia<MixHash, i8, u16, 16>,
    right: &Fesia<MixHash, i8, u16, 16>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline(never)]
pub fn test16_sse(
    left: &Fesia<MixHash, i16, u8, 8>,
    right: &Fesia<MixHash, i16, u8, 8>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
#[inline(never)]
pub fn test32_sse(
    left: &Fesia<MixHash, i32, u8, 4>,
    right: &Fesia<MixHash, i32, u8, 4>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline(never)]
pub fn test8_avx2(
    left: &Fesia<MixHash, i8, u32, 32>,
    right: &Fesia<MixHash, i8, u32, 32>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline(never)]
pub fn test16_avx2(
    left: &Fesia<MixHash, i16, u16, 16>,
    right: &Fesia<MixHash, i16, u16, 16>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx2"))]
#[inline(never)]
pub fn test32_avx2(
    left: &Fesia<MixHash, i32, u8, 8>,
    right: &Fesia<MixHash, i32, u8, 8>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn test8_avx512(
    left: &Fesia<MixHash, i8, u64, 64>,
    right: &Fesia<MixHash, i8, u64, 64>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn test16_avx512(
    left: &Fesia<MixHash, i16, u32, 32>,
    right: &Fesia<MixHash, i16, u32, 32>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn test32_avx512(
    left: &Fesia<MixHash, i32, u16, 16>,
    right: &Fesia<MixHash, i32, u16, 16>,
    visitor: &mut crate::visitor::VecWriter<i32>)
{
    left.intersect::<crate::visitor::VecWriter<i32>, SegmentIntersectSse>(right, visitor);
}
