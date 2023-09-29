/// BSR stands for Base and State Representation, and is an alternate way to
/// store sets which allows fast intersection when sets have high densities.
/// 
/// Shuo Han, Lei Zou, and Jeffrey Xu Yu. 2018. Speeding Up Set Intersections in
/// Graph Algorithms using SIMD Instructions. In Proceedings of the 2018
/// International Conference on Management of Data (SIGMOD '18). Association for
/// Computing Machinery, New York, NY, USA, 1587–1602.
/// https://doi.org/10.1145/3183713.3196924
/// 
/// A significant portion of the implementation is derived from
/// https://github.com/pkumod/GraphSetIntersection (MIT License)

use std::{slice, iter::Zip};
use crate::Set;

pub type Intersect2Bsr = for<'a> fn(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut BsrVec);
pub struct BsrRef<'a> {
    pub bases: &'a[u32],
    pub states: &'a[u32],
}

impl<'a> BsrRef<'a> {
    pub fn len(&self) -> usize {
        debug_assert!(self.bases.len() == self.states.len());
        self.bases.len()
    }

    pub fn is_empty(&self) -> bool {
        debug_assert!(self.bases.is_empty() == self.states.is_empty());
        self.bases.is_empty()
    }

    pub fn advanced_by(self, offset: usize) -> BsrRef<'a> {
        BsrRef::<'a> {
            bases: &self.bases[offset..],
            states: &self.states[offset..],
        }
    }

    pub unsafe fn advanced_by_unchecked(self, offset: usize) -> BsrRef<'a> {
        BsrRef::<'a> {
            bases: self.bases.get_unchecked(offset..),
            states: self.states.get_unchecked(offset..),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BsrVec {
    pub bases: Vec<u32>,
    pub states: Vec<u32>,
}

impl BsrVec {
    pub fn new() -> Self {
        Self {
            bases: Vec::new(),
            states: Vec::new(),
        }
    }

    pub fn with_capacities(s: usize) -> Self {
        Self {
            bases: Vec::with_capacity(s),
            states: Vec::with_capacity(s),
        }
    }

    pub fn append(&mut self, base: u32, state: u32) {
        debug_assert!(state != 0);
        debug_assert!(self.bases.last().map(|b| b < &base).unwrap_or(true));

        self.bases.push(base);
        self.states.push(state);
    }

    pub fn to_sorted_set(&self) -> Vec<u32> {
        let mut result = Vec::new();
        let iter = self.bases.iter().copied().zip(self.states.iter().copied());
        for (base, mut state) in iter {
            let high = base << BSR_SHIFT;
            while state != 0 {
                result.push(high | state.trailing_zeros());
                state &= state - 1;
            }
        }
        result
    }

    pub fn iter(&self) -> Zip<slice::Iter<'_, u32>, slice::Iter<'_, u32>> {
        self.bases.iter().zip(self.states.iter())
    }

    pub fn bsr_ref(&self) -> BsrRef {
        BsrRef {
            bases: &self.bases,
            states: &self.states,
        }
    }

    pub fn len(&self) -> usize {
        debug_assert_eq!(self.bases.len(), self.states.len());
        self.bases.len()
    }

    pub fn is_empty(&self) -> bool {
        debug_assert_eq!(self.bases.is_empty(), self.states.is_empty());
        self.bases.is_empty()
    }
}

impl Default for BsrVec {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> IntoIterator for BsrRef<'a> {
    type Item = (&'a u32, &'a u32);
    type IntoIter = Zip<slice::Iter<'a, u32>, slice::Iter<'a, u32>>;

    fn into_iter(self) -> Self::IntoIter {
        self.bases.iter().zip(self.states.iter())
    }
}

impl<'a> From<&'a BsrVec> for BsrRef<'a> {
    fn from(vec: &'a BsrVec) -> Self {
        Self {
            bases: &vec.bases,
            states: &vec.states,
        }
    }
}

pub const BSR_WIDTH: u32 = u32::BITS;
pub const BSR_SHIFT: u32 = BSR_WIDTH.trailing_zeros();
pub const BSR_MASK: u32 = BSR_WIDTH - 1;

impl Set<u32> for BsrVec {
    fn from_sorted(sorted: &[u32]) -> Self {
        let mut bsr = BsrVec::new();

        let mut it = sorted.iter().copied();
        if let Some(first) = it.next() {
            bsr.bases.push(first >> BSR_SHIFT);
            bsr.states.push(1 << (first & BSR_MASK));
        }

        for item in it {
            let base = item >> BSR_SHIFT;
            let bit = 1 << (item & BSR_MASK);

            if *bsr.bases.last().unwrap() != base {
                bsr.bases.push(base);
                bsr.states.push(bit);
            }
            else {
                *bsr.states.last_mut().unwrap() |= bit;
            }
        }
        bsr
    }
}
