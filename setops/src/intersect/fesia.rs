#![cfg(feature = "simd")]
/// Implementation of the FESIA set intersection algorithm from below paper.
/// Zhang, J., Lu, Y., Spampinato, D. G., & Franchetti, F. (2020, April). Fesia:
/// A fast and simd-efficient set intersection approach on modern cpus. In 2020
/// IEEE 36th International Conference on Data Engineering (ICDE) (pp.
/// 1465-1476). IEEE.

mod kernels;

use std::{
    marker::PhantomData,
    num::Wrapping,
    simd::*,
    ops::BitAnd,
};
use crate::{
    intersect,
    visitor::{SimdVisitor4, Visitor},
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

        for segment in segments  {
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

pub fn fesia_intersect<H, S, M, const LANES: usize, V>(
    left: &Fesia<H, S, M, LANES>,
    right: &Fesia<H, S, M, LANES>,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    if left.segment_count() > right.segment_count() {
        return fesia_intersect(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_intersect_block(left, right, base, visitor);
    }
}

fn fesia_intersect_block<H, S, M, const LANES: usize, V>(
    small: &Fesia<H, S, M, LANES>,
    large: &Fesia<H, S, M, LANES>,
    base_segment: usize,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    debug_assert!(small.segment_count() <= large.segment_count());
    debug_assert!(base_segment <= large.segment_count() - small.segment_count());

    // Ensure we do not overflow into next block.
    let large_last_segment = base_segment + small.segment_count() - 1;
    let large_reordered_max =
        large.offsets[large_last_segment] +
        large.sizes[large_last_segment];

    let mut small_offset = 0;
    while small_offset < small.segment_count() {
        let large_offset = base_segment + small_offset;

        let pos_a = unsafe { (small.bitmap.as_ptr() as *const S).add(small_offset) };
        let pos_b = unsafe { (large.bitmap.as_ptr() as *const S).add(large_offset) };
        let v_a: Simd<S, LANES> = unsafe{ load_unsafe(pos_a) };
        let v_b: Simd<S, LANES> = unsafe{ load_unsafe(pos_b) };

        let and_result = v_a & v_b;
        let and_mask = and_result.simd_ne(Mask::<S, LANES>::from_array([false; LANES]).to_int());
        let mut mask = and_mask.to_bitmask();

        while !mask.is_zero() {
            let bit_offset = mask.trailing_zeros() as usize;
            mask = mask & (mask.sub(M::one()));

            let offset_a = small.offsets[small_offset + bit_offset] as usize;
            let offset_b = large.offsets[large_offset + bit_offset] as usize;
            let size_a = small.sizes[small_offset + bit_offset] as usize;
            let size_b = large.sizes[large_offset + bit_offset] as usize;

            segment_intersect_sse(
                &small.reordered_set[offset_a..],
                &large.reordered_set[offset_b..large_reordered_max as usize],
                size_a,
                size_b,
                visitor);
        }

        small_offset += LANES;
    }
}

pub fn fesia_intersect_shuffling<H, S, M, const LANES: usize, V>(
    left: &Fesia<H, S, M, LANES>,
    right: &Fesia<H, S, M, LANES>,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    if left.segment_count() > right.segment_count() {
        return fesia_intersect_shuffling(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_intersect_block_shuffling(left, right, base, visitor);
    }
}

fn fesia_intersect_block_shuffling<H, S, M, const LANES: usize, V>(
    small: &Fesia<H, S, M, LANES>,
    large: &Fesia<H, S, M, LANES>,
    base_segment: usize,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    debug_assert!(small.segment_count() <= large.segment_count());
    debug_assert!(base_segment <= large.segment_count() - small.segment_count());

    let mut small_offset = 0;
    while small_offset < small.segment_count() {
        let large_offset = base_segment + small_offset;

        let pos_a = unsafe { (small.bitmap.as_ptr() as *const S).add(small_offset) };
        let pos_b = unsafe { (large.bitmap.as_ptr() as *const S).add(large_offset) };
        let v_a: Simd<S, LANES> = unsafe{ load_unsafe(pos_a) };
        let v_b: Simd<S, LANES> = unsafe{ load_unsafe(pos_b) };

        let and_result = v_a & v_b;
        let and_mask = and_result.simd_ne(Mask::<S, LANES>::from_array([false; LANES]).to_int());
        let mut mask = and_mask.to_bitmask();

        while !mask.is_zero() {
            let bit_offset = mask.trailing_zeros() as usize;
            mask = mask & (mask.sub(M::one()));

            let offset_a = small.offsets[small_offset + bit_offset] as usize;
            let offset_b = large.offsets[large_offset + bit_offset] as usize;
            let size_a = small.sizes[small_offset + bit_offset] as usize;
            let size_b = large.sizes[large_offset + bit_offset] as usize;

            intersect::shuffling_sse(
                &small.reordered_set[offset_a..offset_a + size_a],
                &large.reordered_set[offset_b..offset_b + size_b],
                visitor);
        }

        small_offset += LANES;
    }
}

fn segment_intersect_sse<V>(
    set_a: &[i32],
    set_b: &[i32],
    size_a: usize,
    size_b: usize,
    visitor: &mut V)
where
    V: SimdVisitor4<i32>,
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
        0o11..=0o14 => unsafe { kernels::sse_1x4(small_ptr, large_ptr, visitor) },
        0o15..=0o17 => unsafe { kernels::sse_1x8(small_ptr, large_ptr, visitor) },
        0o22..=0o24 => unsafe { kernels::sse_2x4(small_ptr, large_ptr, visitor) },
        0o25..=0o27 => unsafe { kernels::sse_2x8(small_ptr, large_ptr, visitor) },
        0o33..=0o34 => unsafe { kernels::sse_3x4(small_ptr, large_ptr, visitor) },
        0o35..=0o37 => unsafe { kernels::sse_3x8(small_ptr, large_ptr, visitor) },
        0o44        => unsafe { kernels::sse_4x4(small_ptr, large_ptr, visitor) },
        0o45..=0o47 => unsafe { kernels::sse_4x8(small_ptr, large_ptr, visitor) },
        0o55..=0o57 => unsafe { kernels::sse_5x8(small_ptr, large_ptr, visitor) },
        0o66..=0o67 => unsafe { kernels::sse_6x8(small_ptr, large_ptr, visitor) },
        0o77        => unsafe { kernels::sse_7x8(small_ptr, large_ptr, visitor) },
        _ => panic!("Invalid kernel {:02o}", ctrl),
    }
}

fn masked_hash<H: IntegerHash>(item: i32, segment_count: usize) -> i32 {
    debug_assert!(segment_count.count_ones() == 1);
    H::hash(item) & (segment_count as i32 - 1)
}

pub fn fesia_hash_intersect<H, S, M, const LANES: usize>(
    small: &Fesia<H, S, M, LANES>,
    large: &Fesia<H, S, M, LANES>,
    visitor: &mut impl Visitor<i32>)
where
    H: IntegerHash,
    S: SimdElement + MaskElement,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    debug_assert!(small.reordered_set.len() <= large.reordered_set.len());
    debug_assert!(large.hash_size % small.hash_size == 0);
    debug_assert!(large.segment_count() % small.segment_count() == 0);

    let segment_bits: usize = std::mem::size_of::<S>() * u8::BITS as usize;

    // TODO: check loop order
    for block in 0..large.segment_count() / small.segment_count() {
        let base = block * small.segment_count();

        for &item in &small.reordered_set {
            let hash = masked_hash::<H>(item, small.hash_size);
            let segment_index = base + (hash as usize / segment_bits);
            
            let offset = large.offsets[segment_index] as usize;
            let size = large.sizes[segment_index] as usize;
            
            // TODO: compare with vector comparison
            for &other in &large.reordered_set[offset..offset+size] {
                if item == other {
                    visitor.visit(item);
                    break;
                }
            }
        }
    }
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
