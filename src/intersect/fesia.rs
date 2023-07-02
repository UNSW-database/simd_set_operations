#![cfg(feature = "simd")]
/// Implementation of the FESIA set intersection algorithm from below paper.
/// Zhang, J., Lu, Y., Spampinato, D. G., & Franchetti, F. (2020, April). Fesia:
/// A fast and simd-efficient set intersection approach on modern cpus. In 2020
/// IEEE 36th International Conference on Data Engineering (ICDE) (pp.
/// 1465-1476). IEEE.

use std::{
    marker::PhantomData,
    num::Wrapping,
    simd::*,
    ops::BitAnd,
    fmt::Debug,
};
use crate::{
    intersect, Set,
    visitor::{SimdVisitor4, Visitor},
    instructions::load_unsafe,
    util::or_4
};

// Use a power of 2 output space as this allows reducing the hash without skewing
const MIN_HASH_SIZE: usize = 16 * i32::BITS as usize; 

pub type Fesia8Sse<const HASH_SCALE: usize>     = Fesia<MixHash, i8,  u16, 16, HASH_SCALE>;
pub type Fesia16Sse<const HASH_SCALE: usize>    = Fesia<MixHash, i16, u8,  8,  HASH_SCALE>;
pub type Fesia32Sse<const HASH_SCALE: usize>    = Fesia<MixHash, i32, u8,  4,  HASH_SCALE>;
pub type Fesia8Avx2<const HASH_SCALE: usize>    = Fesia<MixHash, i8,  u32, 32, HASH_SCALE>;
pub type Fesia16Avx2<const HASH_SCALE: usize>   = Fesia<MixHash, i16, u16, 16, HASH_SCALE>;
pub type Fesia32Avx2<const HASH_SCALE: usize>   = Fesia<MixHash, i32, u8,  8,  HASH_SCALE>;
pub type Fesia8Avx512<const HASH_SCALE: usize>  = Fesia<MixHash, i8,  u64, 64, HASH_SCALE>;
pub type Fesia16Avx512<const HASH_SCALE: usize> = Fesia<MixHash, i16, u32, 32, HASH_SCALE>;
pub type Fesia32Avx512<const HASH_SCALE: usize> = Fesia<MixHash, i32, u16, 16, HASH_SCALE>;

pub struct Fesia<H, S, M, const LANES: usize, const HASH_SCALE: usize>
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    bitmap: Vec<u8>,
    sizes: Vec<i32>,
    offsets: Vec<i32>,
    reordered_set: Vec<i32>,
    hash_t: PhantomData<H>,
    segment_t: PhantomData<S>,
}

impl<H, S, M, const LANES: usize, const HS: usize> Fesia<H, S, M, LANES, HS> 
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
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

impl<H, S, M, const LANES: usize, const HASH_SCALE: usize> Set<i32>
for Fesia<H, S, M, LANES, HASH_SCALE>
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    M: num::PrimInt,
{
    fn from_sorted(sorted: &[i32]) -> Self {
        let segment_bits: usize = std::mem::size_of::<S>() * u8::BITS as usize;

        let hash_size = (sorted.len() * HASH_SCALE)
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
            bitmap: bitmap,
            sizes: sizes,
            offsets: offsets,
            reordered_set: reordered_set,
            hash_t: PhantomData,
            segment_t: PhantomData,
        }
    }
}

#[inline(never)]
pub fn fesia<H, S, M, const LANES: usize, const HASH_SCALE: usize, V>(
    left: &Fesia<H, S, M, LANES, HASH_SCALE>,
    right: &Fesia<H, S, M, LANES, HASH_SCALE>,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    if left.segment_count() > right.segment_count() {
        return fesia(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_block(left, right, base, visitor);
    }
}

fn fesia_block<H, S, M, const LANES: usize, const HASH_SCALE: usize, V>(
    small: &Fesia<H, S, M, LANES, HASH_SCALE>,
    large: &Fesia<H, S, M, LANES, HASH_SCALE>,
    base_segment: usize,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
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

            segment_intersect(
                &small.reordered_set[offset_a..],
                &large.reordered_set[offset_b..large_reordered_max as usize],
                size_a,
                size_b,
                visitor);
        }

        small_offset += LANES;
    }
}

#[inline(never)]
pub fn fesia_shuffling<H, S, M, const LANES: usize, const HASH_SCALE: usize, V>(
    left: &Fesia<H, S, M, LANES, HASH_SCALE>,
    right: &Fesia<H, S, M, LANES, HASH_SCALE>,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<S, LANES>: BitAnd<Output=Simd<S, LANES>> + SimdPartialEq<Mask=Mask<S, LANES>>,
    Mask<S, LANES>: ToBitMask<BitMask=M>,
    V: SimdVisitor4<i32>,
    M: num::PrimInt,
{
    if left.segment_count() > right.segment_count() {
        return fesia_shuffling(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_block_shuffling(left, right, base, visitor);
    }
}

fn fesia_block_shuffling<H, S, M, const LANES: usize, const HASH_SCALE: usize, V>(
    small: &Fesia<H, S, M, LANES, HASH_SCALE>,
    large: &Fesia<H, S, M, LANES, HASH_SCALE>,
    base_segment: usize,
    visitor: &mut V)
where
    H: IntegerHash,
    S: SimdElement + MaskElement + Debug,
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

fn segment_intersect<V>(
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
        0o11..=0o14 => unsafe { kernel1x4(small_ptr, large_ptr, visitor) },
        0o15..=0o17 => unsafe { kernel1x8(small_ptr, large_ptr, visitor) },
        0o22..=0o24 => unsafe { kernel2x4(small_ptr, large_ptr, visitor) },
        0o25..=0o27 => unsafe { kernel2x8(small_ptr, large_ptr, visitor) },
        0o33..=0o34 => unsafe { kernel3x4(small_ptr, large_ptr, visitor) },
        0o35..=0o37 => unsafe { kernel3x8(small_ptr, large_ptr, visitor) },
        0o44        => unsafe { kernel4x4(small_ptr, large_ptr, visitor) },
        0o45..=0o47 => unsafe { kernel4x8(small_ptr, large_ptr, visitor) },
        0o55..=0o57 => unsafe { kernel5x8(small_ptr, large_ptr, visitor) },
        0o66..=0o67 => unsafe { kernel6x8(small_ptr, large_ptr, visitor) },
        0o77        => unsafe { kernel7x8(small_ptr, large_ptr, visitor) },
        _ => panic!("Invalid kernel {:02o}", ctrl),
    }
}

unsafe fn kernel1x4<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a = i32x4::splat(*set_a);
    let v_b: i32x4 = load_unsafe(set_b);
    let mask = v_a.simd_eq(v_b);
    if mask.any() {
        visitor.visit(*set_a);
    }
}
unsafe fn kernel1x8<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a = i32x4::splat(*set_a);
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let mask = v_a.simd_eq(v_b0) | v_a.simd_eq(v_b1);
    if mask.any() {
        visitor.visit(*set_a);
    }
}

unsafe fn kernel2x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
    ];
    let mask = masks[0] | masks[1];
    visitor.visit_vector4(v_b, mask.to_bitmask());
}
unsafe fn kernel3x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
        v_b.simd_eq(i32x4::splat(*set_a.add(2))),
    ];
    let mask = masks[0] | masks[1] | masks[2];
    visitor.visit_vector4(v_b, mask.to_bitmask());
}
unsafe fn kernel4x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
        v_b.simd_eq(i32x4::splat(*set_a.add(2))),
        v_b.simd_eq(i32x4::splat(*set_a.add(3))),
    ];
    let mask = (masks[0] | masks[1]) | (masks[2] | masks[3]);
    visitor.visit_vector4(v_b, mask.to_bitmask());
}

unsafe fn kernel2x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a0 = i32x4::splat(*set_a);
    let v_a1 = i32x4::splat(*set_a.add(1));
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = v_b0.simd_eq(v_a0) | v_b0.simd_eq(v_a1);
    let m_b1 = v_b1.simd_eq(v_a0) | v_b1.simd_eq(v_a1);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
unsafe fn kernel3x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a0 = i32x4::splat(*set_a);
    let v_a1 = i32x4::splat(*set_a.add(1));
    let v_a2 = i32x4::splat(*set_a.add(2));

    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));

    let m_b0 = v_b0.simd_eq(v_a0) | v_b0.simd_eq(v_a1) | v_b0.simd_eq(v_a2);
    let m_b1 = v_b1.simd_eq(v_a0) | v_b1.simd_eq(v_a1) | v_b1.simd_eq(v_a2);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
unsafe fn kernel4x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]),
        v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]),
        v_b0.simd_eq(a[3]),
    ]);
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]),
        v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]),
        v_b1.simd_eq(a[3]),
    ]);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
unsafe fn kernel5x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]),
        v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]),
        v_b0.simd_eq(a[3]),
    ]) | v_b0.simd_eq(a[4]);
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]),
        v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]),
        v_b1.simd_eq(a[3]),
    ]) | v_b1.simd_eq(a[4]);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
unsafe fn kernel6x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
        i32x4::splat(*set_a.add(5)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]), v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]), v_b0.simd_eq(a[3]),
    ]) | (v_b0.simd_eq(a[4]) | v_b0.simd_eq(a[5]));
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]), v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]), v_b1.simd_eq(a[3]),
    ]) | (v_b1.simd_eq(a[4]) | v_b1.simd_eq(a[5]));

    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
unsafe fn kernel7x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
        i32x4::splat(*set_a.add(5)),
        i32x4::splat(*set_a.add(6)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]), v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]), v_b0.simd_eq(a[3]),
    ]) | (v_b0.simd_eq(a[4]) | v_b0.simd_eq(a[5]) | v_b0.simd_eq(a[6]));
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]), v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]), v_b1.simd_eq(a[3]),
    ]) | (v_b1.simd_eq(a[4]) | v_b1.simd_eq(a[5]) | v_b1.simd_eq(a[6]));

    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}

fn masked_hash<H: IntegerHash>(item: i32, segment_count: usize) -> i32 {
    debug_assert!(segment_count.count_ones() == 1);
    H::hash(item) & (segment_count as i32 - 1)
}

pub trait IntegerHash {
    /// Hashes randomly to the range 0..SIZE
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

//pub fn fesia_mono(left: FesiaView, right: FesiaView, visitor: &mut crate::visitor::VecWriter<i32>)
//{
//    fesia_sse(left, right, visitor);
//}
