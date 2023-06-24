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

use std::mem::size_of;

use crate::Set;

pub struct BSR<'a> {
    pub base: &'a[u32],
    pub state: &'a[u32],
}

pub struct BSRVec {
    base: Vec<u32>,
    state: Vec<u32>,
}

impl<'a> From<&'a BSRVec> for BSR<'a> {
    fn from(vec: &'a BSRVec) -> Self {
        Self {
            base: &vec.base,
            state: &vec.state,
        }
    }
}

pub const BSR_WIDTH: u32 = (size_of::<u32>() * 8) as u32;
pub const BSR_SHIFT: u32 = BSR_WIDTH.trailing_zeros();
pub const BSR_MASK: u32 = BSR_WIDTH - 1;

impl Set<u32> for BSRVec {
    fn from_sorted(sorted: &[u32]) -> Self {
        let mut bsr = BSRVec::empty();

        let mut it = sorted.iter().copied();
        if let Some(first) = it.next() {
            bsr.base.push(first >> BSR_SHIFT);
            bsr.state.push(1 << (first & BSR_MASK));
        }

        for item in it {
            let base = item >> BSR_SHIFT;
            let bit = 1 << (item & BSR_MASK);

            if *bsr.base.last().unwrap() != base {
                bsr.base.push(base);
                bsr.state.push(bit);
            }
            else {
                *bsr.state.last_mut().unwrap() |= bit;
            }
        }
        bsr
    }
}

impl BSRVec {
    fn empty() -> Self {
        Self {
            base: Vec::new(),
            state: Vec::new(),
        }
    }

    pub fn to_sorted_set(&self) -> Vec<u32> {
        let mut result = Vec::new();
        let iter = self.base.iter().copied().zip(self.state.iter().copied());
        for (base, mut state) in iter {
            let high = base << BSR_SHIFT;
            while state != 0 {
                result.push(high | state.trailing_zeros());
                state &= state - 1;
            }
        }
        result
    }
}
