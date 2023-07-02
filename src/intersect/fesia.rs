#![cfg(feature = "simd")]

use std::{
    marker::PhantomData,
    num::Wrapping,
    simd::*,
    ops::Range,
};

use crate::{intersect, Set, visitor::{SimdVisitor4, Visitor}, instructions::load_unsafe};

// Use a power of 2 output space as this allows reducing the hash without skewing
const MIN_SEGMENT_COUNT: usize = 16 * u32::BITS as usize; 


pub struct Fesia<H: IntegerHash, const LANES: usize>
where
    LaneCount<LANES>: SupportedLaneCount
{
    bitmap: Vec<u32>,
    sizes: Vec<u32>,
    offsets: Vec<u32>,
    reordered_set: Vec<u32>,
    hash: PhantomData<H>,
}

#[derive(Clone, Copy)]
pub struct FesiaView<'a> {
    sizes: &'a[u32],
    offsets: &'a[u32],
    bitmap: &'a[u32],
    reordered_set: &'a[u32],
}

impl<H: IntegerHash, const LANES: usize> Fesia<H, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    pub fn segment_count(&self) -> usize {
        self.bitmap.len()
    }

    fn as_view(&self) -> FesiaView {
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
        Self {
            sizes: &self.sizes[range],
            offsets: &self.offsets[range],
            bitmap: &self.bitmap[range],
            reordered_set: &self.reordered_set,
        }
    }
}

impl<H: IntegerHash, const LANES: usize> Set<u32> for Fesia<H, LANES>
where
    LaneCount<LANES>: SupportedLaneCount,
{
    fn from_sorted(sorted: &[u32]) -> Self {
        // From paper: m = n * sqrt(w) where w is SIMD width
        let m = sorted.len() * (LANES as f64).sqrt() as usize;
        let segment_count = m.next_power_of_two().max(MIN_SEGMENT_COUNT);

        let mut bitmap: Vec<u32> = vec![0; segment_count];
        let mut sizes: Vec<u32> = vec![0; segment_count];

        let mut segments: Vec<Vec<u32>> = vec![Vec::new(); segment_count];
        let mut offsets: Vec<u32> = Vec::with_capacity(segment_count);
        let mut reordered_set: Vec<u32> = Vec::with_capacity(sorted.len());

        for &item in sorted {
            let hash = hash::<H>(item, segment_count);
            let index = (hash / u32::BITS) as usize;
            bitmap[index] |= 1 << (hash % u32::BITS);
            sizes[index] += 1;
            segments[index].push(item);
        }

        for segment in segments {
            offsets.push(reordered_set.len() as u32);
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

pub fn fesia_sse<V>(left: FesiaView, right: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<u32>,
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

fn fesia_block_sse<V>(set_a: FesiaView, set_b: FesiaView, visitor: &mut V)
where
    V: SimdVisitor4<u32>,
{
    debug_assert!(set_a.segment_count() == set_b.segment_count());

    let mut base_segment = 0;
    while base_segment < set_a.segment_count() {
        let v_a: u32x4 = unsafe{ load_unsafe(set_a.bitmap.as_ptr().add(base_segment)) };
        let v_b: u32x4 = unsafe{ load_unsafe(set_b.bitmap.as_ptr().add(base_segment)) };

        let and_result = v_a & v_b;
        let and_mask = and_result.simd_ne(u32x4::from_array([0;4]));
        let mut mask = and_mask.to_bitmask();
        while mask != 0 {
            let segment = base_segment + mask.trailing_zeros() as usize;
            mask &= mask - 1;

            let offset_a = set_a.offsets[segment] as usize;
            let size_a = set_a.sizes[segment] as usize;
            let offset_b = set_b.offsets[segment] as usize;
            let size_b = set_b.sizes[segment] as usize;
            segment_intersect(
                &set_a.reordered_set[offset_a..offset_a+size_a],
                &set_b.reordered_set[offset_b..offset_b+size_b],
                visitor);
        }

        base_segment += i32x4::LANES;
    }
}

fn segment_intersect<V>(left: &[u32], right: &[u32], visitor: &mut V)
where
    V: SimdVisitor4<u32>,
{
    if left.len() > 7 || right.len() > 7 {
        return intersect::branchless_merge(left, right, visitor);
    }

    let ctrl = (left.len() << 3) | right.len();
    match ctrl {
        0b001001 => kernel1x1(left[0], right[0], visitor),
        0b001010 => unsafe { kernel1x2(left[0], right.as_ptr(), visitor) },
        0b001011 => unsafe { kernel1x3(left[0], right.as_ptr(), visitor) },
        0b001100 => unsafe { kernel1x4(left[0], right.as_ptr(), visitor) },
        0b001101 => unsafe { kernel1x5(left[0], right.as_ptr(), visitor) },
        0..=2 | _ => panic!("Invalid kernel"),
    }
}

fn kernel1x1<V: Visitor<u32>>(left: u32, right: u32, visitor: &mut V) {
    if left == right {
        visitor.visit(left);
    }
}
unsafe fn kernel1x2<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    if left == *right || left == *right.add(1) {
        visitor.visit(left);
    }
}
unsafe fn kernel1x3<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    if left == *right || left == *right.add(1) || left == *right.add(2) {
        visitor.visit(left);
    }
}
unsafe fn kernel1x4<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    let l = i32x4::splat(left as i32);
    let r: i32x4 = load_unsafe(right as *const i32);
    let mask = l.simd_eq(r);
    if mask.any() {
        visitor.visit(left);
    }
}
unsafe fn kernel1x5<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    let l = i32x4::splat(left as i32);
    let r: i32x4 = load_unsafe(right as *const i32);
    let mask = l.simd_eq(r);
    if mask.any() || left == *right.add(4) {
        visitor.visit(left);
    }
}
unsafe fn kernel1x6<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    let l = i32x4::splat(left as i32);
    let r: i32x4 = load_unsafe(right as *const i32);
    let mask = l.simd_eq(r);
    if mask.any() || left == *right.add(4) || left == *right.add(5) {
        visitor.visit(left);
    }
}
unsafe fn kernel1x7<V: Visitor<u32>>(left: u32, right: *const u32, visitor: &mut V) {
    let l = i32x4::splat(left as i32);
    let r: i32x4 = load_unsafe(right as *const i32);
    let mask = l.simd_eq(r);
    if mask.any() || left == *right.add(4) || left == *right.add(5) || left == *right.add(6) {
        visitor.visit(left);
    }
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

fn hash<H: IntegerHash>(item: u32, segment_count: usize) -> u32 {
    H::hash(item) & (segment_count as u32 - 1)
}

pub trait IntegerHash {
    /// Hashes randomly to the range 0..SIZE
    fn hash(item: u32) -> u32;
}

pub struct IdentityHash;
impl IntegerHash for IdentityHash {
    fn hash(item: u32) -> u32 {
        item
    }
}

pub struct MixHash;
impl IntegerHash for MixHash {
    // https://gist.github.com/badboy/6267743
    fn hash(item: u32) -> u32 {
        let mut key = Wrapping(item);
        key = !key + (key << 15); // key = (key << 15) - key - 1;
        key = key ^ (key >> 12);
        key = key + (key << 2);
        key = key ^ (key >> 4);
        key = key * Wrapping(2057); // key = (key + (key << 3)) + (key << 11);
        key = key ^ (key >> 16);
        key.0
    }
}
