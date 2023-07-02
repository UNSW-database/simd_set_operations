#![cfg(feature = "simd")]

use std::{
    marker::PhantomData,
    num::Wrapping,
    simd::*,
    ops::Range,
};

use crate::{intersect, Set, visitor::{SimdVisitor4, Visitor}, instructions::load_unsafe, util::or_4};

// Use a power of 2 output space as this allows reducing the hash without skewing
const MIN_SEGMENT_COUNT: usize = 16 * i32::BITS as usize; 


pub struct Fesia<H: IntegerHash, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount
{
    bitmap: Vec<i32>,
    sizes: Vec<i32>,
    offsets: Vec<i32>,
    reordered_set: Vec<i32>,
    hash: PhantomData<H>,
}

#[derive(Clone, Copy)]
pub struct FesiaView<'a> {
    sizes: &'a[i32],
    offsets: &'a[i32],
    bitmap: &'a[i32],
    reordered_set: &'a[i32],
}

impl<H: IntegerHash, const LANES: usize> Fesia<H, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn segment_count(&self) -> usize {
        self.bitmap.len()
    }

    pub fn as_view(&self) -> FesiaView {
        FesiaView {
            sizes: &self.sizes,
            offsets: &self.offsets,
            bitmap: &self.bitmap,
            reordered_set: &self.reordered_set,
        }
    }
}

impl<'a> FesiaView<'a> {
    pub fn segment_count(&self) -> usize {
        self.bitmap.len()
    }

    pub fn subview(&self, range: Range<usize>) -> FesiaView<'a> {
        let reorder_max = self.offsets[range.end-1] + self.sizes[range.end-1];
        Self {
            sizes: &self.sizes[range.clone()],
            offsets: &self.offsets[range.clone()],
            bitmap: &self.bitmap[range],
            reordered_set: &self.reordered_set[..reorder_max as usize],
        }
    }
}

impl<H: IntegerHash, const LANES: usize> Set<i32> for Fesia<H, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from_sorted(sorted: &[i32]) -> Self {
        // From paper: m = n * sqrt(w) where w is SIMD width
        let m = sorted.len() * (LANES as f64).sqrt() as usize;
        let segment_count = m.next_power_of_two().max(MIN_SEGMENT_COUNT);

        let mut bitmap: Vec<i32> = vec![0; segment_count];
        let mut sizes: Vec<i32> = vec![0; segment_count];

        let mut segments: Vec<Vec<i32>> = vec![Vec::new(); segment_count];
        let mut offsets: Vec<i32> = Vec::with_capacity(segment_count);
        let mut reordered_set: Vec<i32> = Vec::with_capacity(sorted.len());

        for &item in sorted {
            let hash = hash::<H>(item, segment_count);
            let index = (hash / i32::BITS as i32) as usize;
            bitmap[index] |= 1 << (hash % i32::BITS as i32);
            sizes[index] += 1;
            segments[index].push(item);
        }

        for segment in segments {
            offsets.push(reordered_set.len() as i32);
            reordered_set.extend_from_slice(&segment);
        }

        Self {
            bitmap: bitmap,
            sizes: sizes,
            offsets: offsets,
            reordered_set: reordered_set,
            hash: PhantomData,
        }
    }
}

#[inline(never)]
pub fn fesia_sse<V>(left: FesiaView, right: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<i32>,
{
    if left.segment_count() > right.segment_count() {
        return fesia_sse(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_block_sse(left, right.subview(base..base+left.segment_count()), visitor)
    }
}

pub fn fesia_sse_shuffling<V>(left: FesiaView, right: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<i32>,
{
    if left.segment_count() > right.segment_count() {
        return fesia_sse(right, left, visitor);
    }
    debug_assert!(right.segment_count() % left.segment_count() == 0);

    for block in 0..right.segment_count() / left.segment_count() {
        let base = block * left.segment_count();
        fesia_block_sse_shuffling(left, right.subview(base..base+left.segment_count()), visitor)
    }
}

fn fesia_block_sse_shuffling<V>(set_a: FesiaView, set_b: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<i32>,
{
    debug_assert!(set_a.segment_count() == set_b.segment_count());

    let mut base_segment = 0;
    while base_segment < set_a.segment_count() {
        let v_a: i32x4 = unsafe{ load_unsafe(set_a.bitmap.as_ptr().add(base_segment)) };
        let v_b: i32x4 = unsafe{ load_unsafe(set_b.bitmap.as_ptr().add(base_segment)) };

        let and_result = v_a & v_b;
        let and_mask = and_result.simd_ne(i32x4::from_array([0;4]));
        let mut mask = and_mask.to_bitmask();
        while mask != 0 {
            let segment = base_segment + mask.trailing_zeros() as usize;
            mask &= mask - 1;

            let offset_a = set_a.offsets[segment] as usize;
            let size_a = set_a.sizes[segment] as usize;
            let offset_b = set_b.offsets[segment] as usize;
            let size_b = set_b.sizes[segment] as usize;
            intersect::shuffling_sse(
                &set_a.reordered_set[offset_a..offset_a+size_a],
                &set_b.reordered_set[offset_b..offset_b+size_b],
                visitor);
        }

        base_segment += i32x4::LANES;
    }
}

fn fesia_block_sse<V>(set_a: FesiaView, set_b: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<i32>,
{
    debug_assert!(set_a.segment_count() == set_b.segment_count());

    let mut base_segment = 0;
    while base_segment < set_a.segment_count() {
        let v_a: i32x4 = unsafe{ load_unsafe(set_a.bitmap.as_ptr().add(base_segment)) };
        let v_b: i32x4 = unsafe{ load_unsafe(set_b.bitmap.as_ptr().add(base_segment)) };

        let and_result = v_a & v_b;
        let and_mask = and_result.simd_ne(i32x4::from_array([0;4]));
        let mut mask = and_mask.to_bitmask();
        while mask != 0 {
            let segment = base_segment + mask.trailing_zeros() as usize;
            mask &= mask - 1;

            let offset_a = set_a.offsets[segment] as usize;
            let size_a = set_a.sizes[segment] as usize;
            let offset_b = set_b.offsets[segment] as usize;
            let size_b = set_b.sizes[segment] as usize;
            segment_intersect(
                &set_a.reordered_set[offset_a..],
                &set_b.reordered_set[offset_b..],
                size_a,
                size_b,
                visitor);
        }

        base_segment += i32x4::LANES;
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

mod tests {
    use super::*;

    #[test]
    fn test_from_sorted() {
        let set = Vec::from_iter((0..1024).map(|i| i * 2));
        let fesia: Fesia<MixHash, 4> = Fesia::from_sorted(&set);

        let mut reordered_sorted = fesia.reordered_set.clone();
        reordered_sorted.sort();
        assert!(set == reordered_sorted);

        assert!(fesia.bitmap.len() == 128/32);
        for bitmap in fesia.bitmap {
            assert!(bitmap == 0b01010101010101010101010101010101);
        }
        for size in fesia.sizes {
            assert!(size == 1024/(128/32));
        }
    }
}

fn hash<H: IntegerHash>(item: i32, segment_count: usize) -> i32 {
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

pub fn fesia_mono(left: FesiaView, right: FesiaView, visitor: &mut crate::visitor::VecWriter<i32>)
{
    fesia_sse(left, right, visitor);
}
