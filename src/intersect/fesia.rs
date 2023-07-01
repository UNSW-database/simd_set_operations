use std::marker::PhantomData;

use crate::{Set, visitor::SimdVisitor4};

// Use a power of 2 output space as this allows reducing the hash without skewing

pub trait FesiaHash {
    // SCALE ensures output size 
    const SCALE: usize;
    const SIZE: usize = 16 * Self::SCALE * u32::BITS as usize;

    /// Hashes randomly to the range 0..SIZE
    fn hash(item: u32) -> u32;
}

/// Naive hash function using modulo
struct ModHash<const SIZE: usize>;
impl<const SCALE: usize> FesiaHash for ModHash<SCALE> {
    const SCALE: usize = SCALE;
    fn hash(item: u32) -> u32 {
        item % Self::SIZE as u32
    }
}

pub struct Fesia<H: FesiaHash> {
    // Each segment represented as a bitmap
    // There should be hash::size()/32 segments.
    bitmap: Vec<u32>,
    sizes: Vec<u32>,
    offsets: Vec<u32>,
    reordered_set: Vec<u32>,
    hash: PhantomData<H>,
}

impl<H: FesiaHash> Set<u32> for Fesia<H> {
    fn from_sorted(sorted: &[u32]) -> Self {
        assert!(H::SIZE % u32::BITS as usize == 0);
        let segment_count = H::SIZE / u32::BITS as usize;

        let mut bitmap: Vec<u32> = vec![0; segment_count];
        let mut sizes: Vec<u32> = vec![0; segment_count];

        let mut segments: Vec<Vec<u32>> = vec![Vec::new(); segment_count];
        let mut offsets: Vec<u32> = Vec::with_capacity(segment_count);
        let mut reordered_set: Vec<u32> = Vec::with_capacity(sorted.len());

        for &item in sorted {
            let hash = H::hash(item);
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

pub fn fesia_sse<H, V>(left: Fesia<H>, right: Fesia<H>, visitor: &mut V)
where
    H: FesiaHash,
    V: SimdVisitor4<u32>,
{
    let segment_count = H::SIZE * u32::BITS as usize;

    for i in 0..segment_count {
        
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_from_sorted() {
        let set = Vec::from_iter((0..1024).map(|i| i * 2));
        let fesia: Fesia<ModHash<128>> = Fesia::from_sorted(&set);

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

/*
 *
 * |A| = 8, |B| = 4
 * 
 * 
 * 
 * 
 * 
 * 
 * 
 */