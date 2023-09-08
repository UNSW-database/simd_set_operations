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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>,
        I: SegmentIntersect;

    fn hash_intersect(&self, other: &Self, visitor: &mut impl Visitor<i32>);
}

#[derive(Clone, Copy, PartialEq)]
pub enum FesiaIntersectMethod {
    SimilarSize,
    SimilarSizeShuffling,
    SimilarSizeSplat,
    SimilarSizeTable,
    Skewed,
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>,
        I: SegmentIntersect,
    {
        debug_assert!(self.segment_count() <= other.segment_count());
        debug_assert!(base_segment <= other.segment_count() - self.segment_count());

        // Ensure we do not overflow into next block.
        let large_last_segment = base_segment + self.segment_count() - 1;
        let large_reordered_max = (
            other.offsets[large_last_segment] +
            other.sizes[large_last_segment]
        ) as usize;

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

                let offset_a = self.offsets[small_offset + bit_offset] as usize;
                let offset_b = other.offsets[large_offset + bit_offset] as usize;
                let size_a = self.sizes[small_offset + bit_offset] as usize;
                let size_b = other.sizes[large_offset + bit_offset] as usize;

                I::intersect(
                    &self.reordered_set[offset_a..],
                    &other.reordered_set[offset_b..large_reordered_max],
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>,
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

        // TODO: check loop order
        for block in 0..other.segment_count() / self.segment_count() {
            let base = block * self.segment_count();

            for &item in &self.reordered_set {
                let hash = masked_hash::<H>(item, self.hash_size);
                let segment_index = base + (hash as usize / segment_bits);
                
                let offset = other.offsets[segment_index] as usize;
                let size = other.sizes[segment_index] as usize;
                
                // TODO: compare with vector comparison
                for &other in &other.reordered_set[offset..offset+size] {
                    if item == other {
                        visitor.visit(item);
                        break;
                    }
                }
            }
        }
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

        let mut segments: Vec<Vec<i32>> = vec![Vec::new(); segment_count];
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>;
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
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
            return intersect::branchless_merge(&set_a[..size_a], &set_b[..size_b], visitor);
        }

        let (small_ptr, small_size, large_ptr, large_size) =
            if size_a <= size_b {
                (set_a.as_ptr(), size_a, set_b.as_ptr(), size_b)
            }
            else {
                (set_b.as_ptr(), size_b, set_a.as_ptr(), size_a)
            };

        let ctrl = (small_size << 3) | large_size;
        match ctrl {
            0o11..=0o14 => unsafe { kernels_sse::sse_1x4(small_ptr, large_ptr, visitor) },
            0o15..=0o17 => unsafe { kernels_sse::sse_1x8(small_ptr, large_ptr, visitor) },
            0o22..=0o24 => unsafe { kernels_sse::sse_2x4(small_ptr, large_ptr, visitor) },
            0o25..=0o27 => unsafe { kernels_sse::sse_2x8(small_ptr, large_ptr, visitor) },
            0o33..=0o34 => unsafe { kernels_sse::sse_3x4(small_ptr, large_ptr, visitor) },
            0o35..=0o37 => unsafe { kernels_sse::sse_3x8(small_ptr, large_ptr, visitor) },
            0o44        => unsafe { kernels_sse::sse_4x4(small_ptr, large_ptr, visitor) },
            0o45..=0o47 => unsafe { kernels_sse::sse_4x8(small_ptr, large_ptr, visitor) },
            0o55..=0o57 => unsafe { kernels_sse::sse_5x8(small_ptr, large_ptr, visitor) },
            0o66..=0o67 => unsafe { kernels_sse::sse_6x8(small_ptr, large_ptr, visitor) },
            0o77        => unsafe { kernels_sse::sse_7x8(small_ptr, large_ptr, visitor) },
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
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
            return intersect::branchless_merge(&set_a[..size_a], &set_b[..size_b], visitor);
        }

        let (small_ptr, small_size, large_ptr, large_size) =
            if size_a <= size_b {
                (set_a.as_ptr(), size_a, set_b.as_ptr(), size_b)
            }
            else {
                (set_b.as_ptr(), size_b, set_a.as_ptr(), size_a)
            };

        let ctrl = (small_size << 4) | large_size;
        match ctrl {
            0x11..=0x18 => unsafe { kernels_avx2::avx2_1x8(small_ptr, large_ptr, visitor) },
            0x21..=0x28 => unsafe { kernels_avx2::avx2_2x8(small_ptr, large_ptr, visitor) },
            0x31..=0x38 => unsafe { kernels_avx2::avx2_3x8(small_ptr, large_ptr, visitor) },
            0x41..=0x48 => unsafe { kernels_avx2::avx2_4x8(small_ptr, large_ptr, visitor) },
            0x51..=0x58 => unsafe { kernels_avx2::avx2_5x8(small_ptr, large_ptr, visitor) },
            0x61..=0x68 => unsafe { kernels_avx2::avx2_6x8(small_ptr, large_ptr, visitor) },
            0x71..=0x78 => unsafe { kernels_avx2::avx2_7x8(small_ptr, large_ptr, visitor) },
            0x81..=0x88 => unsafe { kernels_avx2::avx2_8x8(small_ptr, large_ptr, visitor) },
            0x19..=0x1F => unsafe { kernels_avx2::avx2_1x16(small_ptr, large_ptr, visitor) },
            0x29..=0x2F => unsafe { kernels_avx2::avx2_2x16(small_ptr, large_ptr, visitor) },
            0x39..=0x3F => unsafe { kernels_avx2::avx2_3x16(small_ptr, large_ptr, visitor) },
            0x49..=0x4F => unsafe { kernels_avx2::avx2_4x16(small_ptr, large_ptr, visitor) },
            0x59..=0x5F => unsafe { kernels_avx2::avx2_5x16(small_ptr, large_ptr, visitor) },
            0x69..=0x6F => unsafe { kernels_avx2::avx2_6x16(small_ptr, large_ptr, visitor) },
            0x79..=0x7F => unsafe { kernels_avx2::avx2_7x16(small_ptr, large_ptr, visitor) },
            0x89..=0x8F => unsafe { kernels_avx2::avx2_8x16(small_ptr, large_ptr, visitor) },
            0x99..=0x9F => unsafe { kernels_avx2::avx2_9x16(small_ptr, large_ptr, visitor) },
            0xA9..=0xAF => unsafe { kernels_avx2::avx2_10x16(small_ptr, large_ptr, visitor) },
            0xB9..=0xBF => unsafe { kernels_avx2::avx2_11x16(small_ptr, large_ptr, visitor) },
            0xC9..=0xCF => unsafe { kernels_avx2::avx2_12x16(small_ptr, large_ptr, visitor) },
            0xD9..=0xDF => unsafe { kernels_avx2::avx2_13x16(small_ptr, large_ptr, visitor) },
            0xE9..=0xEF => unsafe { kernels_avx2::avx2_14x16(small_ptr, large_ptr, visitor) },
            0xF9..=0xFF => unsafe { kernels_avx2::avx2_15x16(small_ptr, large_ptr, visitor) },
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
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
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
            return intersect::branchless_merge(&set_a[..size_a], &set_b[..size_b], visitor);
        }

        let (small_ptr, small_size, large_ptr, large_size) =
            if size_a <= size_b {
                (set_a.as_ptr(), size_a, set_b.as_ptr(), size_b)
            }
            else {
                (set_b.as_ptr(), size_b, set_a.as_ptr(), size_a)
            };

        let ctrl = (small_size << 5) | large_size;
        match ctrl {
            33..=48   => unsafe { kernels_avx512::avx512_1x16(small_ptr, large_ptr, visitor) }
            65..=80   => unsafe { kernels_avx512::avx512_2x16(small_ptr, large_ptr, visitor) }
            97..=112  => unsafe { kernels_avx512::avx512_3x16(small_ptr, large_ptr, visitor) }
            129..=144 => unsafe { kernels_avx512::avx512_4x16(small_ptr, large_ptr, visitor) }
            161..=176 => unsafe { kernels_avx512::avx512_5x16(small_ptr, large_ptr, visitor) }
            193..=208 => unsafe { kernels_avx512::avx512_6x16(small_ptr, large_ptr, visitor) }
            225..=240 => unsafe { kernels_avx512::avx512_7x16(small_ptr, large_ptr, visitor) }
            257..=272 => unsafe { kernels_avx512::avx512_8x16(small_ptr, large_ptr, visitor) }
            289..=304 => unsafe { kernels_avx512::avx512_9x16(small_ptr, large_ptr, visitor) }
            321..=336 => unsafe { kernels_avx512::avx512_10x16(small_ptr, large_ptr, visitor) }
            353..=368 => unsafe { kernels_avx512::avx512_11x16(small_ptr, large_ptr, visitor) }
            385..=400 => unsafe { kernels_avx512::avx512_12x16(small_ptr, large_ptr, visitor) }
            417..=432 => unsafe { kernels_avx512::avx512_13x16(small_ptr, large_ptr, visitor) }
            449..=464 => unsafe { kernels_avx512::avx512_14x16(small_ptr, large_ptr, visitor) }
            481..=496 => unsafe { kernels_avx512::avx512_15x16(small_ptr, large_ptr, visitor) }
            49..=63   => unsafe { kernels_avx512::avx512_1x32(small_ptr, large_ptr, visitor) }
            81..=95   => unsafe { kernels_avx512::avx512_2x32(small_ptr, large_ptr, visitor) }
            113..=127 => unsafe { kernels_avx512::avx512_3x32(small_ptr, large_ptr, visitor) }
            145..=159 => unsafe { kernels_avx512::avx512_4x32(small_ptr, large_ptr, visitor) }
            177..=191 => unsafe { kernels_avx512::avx512_5x32(small_ptr, large_ptr, visitor) }
            209..=223 => unsafe { kernels_avx512::avx512_6x32(small_ptr, large_ptr, visitor) }
            241..=255 => unsafe { kernels_avx512::avx512_7x32(small_ptr, large_ptr, visitor) }
            273..=287 => unsafe { kernels_avx512::avx512_8x32(small_ptr, large_ptr, visitor) }
            305..=319 => unsafe { kernels_avx512::avx512_9x32(small_ptr, large_ptr, visitor) }
            337..=351 => unsafe { kernels_avx512::avx512_10x32(small_ptr, large_ptr, visitor) }
            369..=383 => unsafe { kernels_avx512::avx512_11x32(small_ptr, large_ptr, visitor) }
            401..=415 => unsafe { kernels_avx512::avx512_12x32(small_ptr, large_ptr, visitor) }
            433..=447 => unsafe { kernels_avx512::avx512_13x32(small_ptr, large_ptr, visitor) }
            465..=479 => unsafe { kernels_avx512::avx512_14x32(small_ptr, large_ptr, visitor) }
            497..=511 => unsafe { kernels_avx512::avx512_15x32(small_ptr, large_ptr, visitor) }
            529..=543 => unsafe { kernels_avx512::avx512_16x32(small_ptr, large_ptr, visitor) }
            561..=575 => unsafe { kernels_avx512::avx512_17x32(small_ptr, large_ptr, visitor) }
            593..=607 => unsafe { kernels_avx512::avx512_18x32(small_ptr, large_ptr, visitor) }
            625..=639 => unsafe { kernels_avx512::avx512_19x32(small_ptr, large_ptr, visitor) }
            657..=671 => unsafe { kernels_avx512::avx512_20x32(small_ptr, large_ptr, visitor) }
            689..=703 => unsafe { kernels_avx512::avx512_21x32(small_ptr, large_ptr, visitor) }
            721..=735 => unsafe { kernels_avx512::avx512_22x32(small_ptr, large_ptr, visitor) }
            753..=767 => unsafe { kernels_avx512::avx512_23x32(small_ptr, large_ptr, visitor) }
            785..=799 => unsafe { kernels_avx512::avx512_24x32(small_ptr, large_ptr, visitor) }
            817..=831 => unsafe { kernels_avx512::avx512_25x32(small_ptr, large_ptr, visitor) }
            849..=863 => unsafe { kernels_avx512::avx512_26x32(small_ptr, large_ptr, visitor) }
            881..=895 => unsafe { kernels_avx512::avx512_27x32(small_ptr, large_ptr, visitor) }
            913..=927 => unsafe { kernels_avx512::avx512_28x32(small_ptr, large_ptr, visitor) }
            945..=959 => unsafe { kernels_avx512::avx512_29x32(small_ptr, large_ptr, visitor) }
            977..=991 => unsafe { kernels_avx512::avx512_30x32(small_ptr, large_ptr, visitor) }
            1009..=1023 => unsafe { kernels_avx512::avx512_31x32(small_ptr, large_ptr, visitor) }
            _ => panic!("Invalid kernel {:02o}", ctrl),
        }
    }
}

pub struct SegmentIntersectSplatSse;
impl SegmentIntersect for SegmentIntersectSplatSse {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
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
            return intersect::branchless_merge(&set_a[..size_a], &set_b[..size_b], visitor);
        }

        let (small_ptr, small_size, large_ptr, large_size) =
            if size_a <= size_b {
                (set_a.as_ptr(), size_a, set_b.as_ptr(), size_b)
            }
            else {
                (set_b.as_ptr(), size_b, set_a.as_ptr(), size_a)
            };

        let ctrl = (small_size << 3) | large_size;
        match ctrl {
            0o11 => unsafe { kernels_sse::sse_1x4(small_ptr, large_ptr, visitor) },
            0o12 => unsafe { kernels_sse::sse_1x4(small_ptr, large_ptr, visitor) },
            0o13 => unsafe { kernels_sse::sse_1x4(small_ptr, large_ptr, visitor) },
            0o14 => unsafe { kernels_sse::sse_1x4(small_ptr, large_ptr, visitor) },

            0o15 => unsafe { kernels_sse::sse_1x8(small_ptr, large_ptr, visitor) },
            0o16 => unsafe { kernels_sse::sse_1x8(small_ptr, large_ptr, visitor) },
            0o17 => unsafe { kernels_sse::sse_1x8(small_ptr, large_ptr, visitor) },

            0o22 => unsafe { kernels_sse::sse_2x4(small_ptr, large_ptr, visitor) },
            0o23 => unsafe { kernels_sse::sse_2x4(small_ptr, large_ptr, visitor) },
            0o24 => unsafe { kernels_sse::sse_2x4(small_ptr, large_ptr, visitor) },

            0o25 => unsafe { kernels_sse::sse_2x8(small_ptr, large_ptr, visitor) },
            0o26 => unsafe { kernels_sse::sse_2x8(small_ptr, large_ptr, visitor) },
            0o27 => unsafe { kernels_sse::sse_2x8(small_ptr, large_ptr, visitor) },

            0o33 => unsafe { kernels_sse::sse_3x4(small_ptr, large_ptr, visitor) },
            0o34 => unsafe { kernels_sse::sse_3x4(small_ptr, large_ptr, visitor) },

            0o35 => unsafe { kernels_sse::sse_3x8(small_ptr, large_ptr, visitor) },
            0o36 => unsafe { kernels_sse::sse_3x8(small_ptr, large_ptr, visitor) },
            0o37 => unsafe { kernels_sse::sse_3x8(small_ptr, large_ptr, visitor) },

            0o44 => unsafe { kernels_sse::sse_4x4(small_ptr, large_ptr, visitor) },

            0o45 => unsafe { kernels_sse::sse_4x8(small_ptr, large_ptr, visitor) },
            0o46 => unsafe { kernels_sse::sse_4x8(small_ptr, large_ptr, visitor) },
            0o47 => unsafe { kernels_sse::sse_4x8(small_ptr, large_ptr, visitor) },

            0o55 => unsafe { kernels_sse::sse_5x8(small_ptr, large_ptr, visitor) },
            0o56 => unsafe { kernels_sse::sse_5x8(small_ptr, large_ptr, visitor) },
            0o57 => unsafe { kernels_sse::sse_5x8(small_ptr, large_ptr, visitor) },

            0o66 => unsafe { kernels_sse::sse_6x8(small_ptr, large_ptr, visitor) },
            0o67 => unsafe { kernels_sse::sse_6x8(small_ptr, large_ptr, visitor) },

            0o77        => unsafe { kernels_sse::sse_7x8(small_ptr, large_ptr, visitor) },
            _ => panic!("Invalid kernel {:02o}", ctrl),
        }
    }
}

pub struct SegmentIntersectTableSse;
impl SegmentIntersect for SegmentIntersectTableSse {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
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
            return intersect::branchless_merge(&set_a[..size_a], &set_b[..size_b], visitor);
        }

        let (small_ptr, small_size, large_ptr, large_size) =
            if size_a <= size_b {
                (set_a.as_ptr(), size_a, set_b.as_ptr(), size_b)
            }
            else {
                (set_b.as_ptr(), size_b, set_a.as_ptr(), size_a)
            };

        let ctrl = (small_size << 3) | large_size;
        unsafe { kernel_table::<V>()[ctrl](small_ptr, large_ptr, visitor) };
    }
}

type SseKernelTable<V> = [unsafe fn(*const i32, *const i32, *mut V); 0o77];

const fn kernel_table<V>() -> SseKernelTable<V>
where
    V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
{
    let mut result: SseKernelTable<V> = [kernels_sse::unknown; 0o77];

    let mut ctrl = 0;
    while ctrl < 0o77 {
        result[ctrl] = match ctrl {
            0o11..=0o14 => kernels_sse::sse_1x4,
            0o15..=0o17 => kernels_sse::sse_1x8,
            0o22..=0o24 => kernels_sse::sse_2x4,
            0o25..=0o27 => kernels_sse::sse_2x8,
            0o33..=0o34 => kernels_sse::sse_3x4,
            0o35..=0o37 => kernels_sse::sse_3x8,
            0o44        => kernels_sse::sse_4x4,
            0o45..=0o47 => kernels_sse::sse_4x8,
            0o55..=0o57 => kernels_sse::sse_5x8,
            0o66..=0o67 => kernels_sse::sse_6x8,
            0o77        => kernels_sse::sse_7x8,
            _ => kernels_sse::unknown,
        };
        ctrl += 1;
    }
    
    result
}

pub struct SegmentIntersectShufflingSse;
impl SegmentIntersect for SegmentIntersectShufflingSse {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
    {
        intersect::shuffling_sse(&set_a[..size_a], &set_b[..size_b], visitor);
    }
}

#[cfg(target_feature = "avx2")]
pub struct SegmentIntersectShufflingAvx2;
#[cfg(target_feature = "avx2")]
impl SegmentIntersect for SegmentIntersectShufflingAvx2 {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
    {
        intersect::shuffling_avx2(&set_a[..size_a], &set_b[..size_b], visitor);
    }
}

#[cfg(target_feature = "avx512f")]
pub struct SegmentIntersectShufflingAvx512;
#[cfg(target_feature = "avx512f")]
impl SegmentIntersect for SegmentIntersectShufflingAvx512 {
    fn intersect<V>(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut V)
    where
        V: SimdVisitor4<i32> + SimdVisitor8<i32> + SimdVisitor16<i32>
    {
        intersect::shuffling_avx512(&set_a[..size_a], &set_b[..size_b], visitor);
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

pub fn segment_comp(
        set_a: &[i32],
        set_b: &[i32],
        size_a: usize,
        size_b: usize,
        visitor: &mut crate::visitor::VecWriter<i32>)
{
    SegmentIntersectSplatSse::intersect(set_a, set_b, size_a, size_b, visitor)
}

