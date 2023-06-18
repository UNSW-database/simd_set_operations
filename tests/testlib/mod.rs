pub mod properties;

use quickcheck::Arbitrary;
use setops::{
    intersect::{self, Intersect2},
    visitor::VecWriter,
};
use std::fmt;

// Arbitrary Set //
#[derive(Debug, Clone)]
pub struct SortedSet(Vec<i32>);

impl SortedSet {
    pub fn from_unsorted(mut vec: Vec<i32>) -> Self {
        vec.sort_unstable();
        vec.dedup();
        Self(vec)
    }

    pub fn as_slice(&self) -> &[i32] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<i32> {
        self.0
    }
}

impl From<SortedSet> for Vec<i32> {
    fn from(value: SortedSet) -> Self {
        value.into_inner()
    }
}

impl From<Vec<i32>> for SortedSet {
    fn from(value: Vec<i32>) -> Self {
        Self::from_unsorted(value)
    }
}

impl quickcheck::Arbitrary for SortedSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Self::from_unsorted(Vec::<i32>::arbitrary(g))
    }
}

impl AsRef<[i32]> for SortedSet {
    fn as_ref(&self) -> &[i32] {
        &self.0
    }
}

// Arbitrary Intersection Function //
#[derive(Clone)]
pub struct DualIntersectFn(
    &'static str, pub Intersect2<[i32], VecWriter<i32>>
);

impl fmt::Debug for DualIntersectFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl quickcheck::Arbitrary for DualIntersectFn {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        g.choose([
            DualIntersectFn("branchless_merge", intersect::branchless_merge),
            DualIntersectFn("galloping", intersect::galloping),
            DualIntersectFn("baezayates", intersect::baezayates),
            #[cfg(feature = "simd")]
            DualIntersectFn("simd_shuffling", intersect::simd_shuffling),
            //#[cfg(feature = "simd")]
            //DualIntersectFn("simd_galloping", intersect::simd_galloping),
        ].as_slice())
        .unwrap()
        .clone()
    }
}


// Arbitrary Pair of Sets //
#[derive(Debug, Clone)]
pub struct SimilarSetPair(pub SortedSet, pub SortedSet);

impl quickcheck::Arbitrary for SimilarSetPair {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let shared: Vec<i32> = Vec::arbitrary(g);

        let mut left = Vec::arbitrary(g);
        let mut right = Vec::arbitrary(g);
        left.extend(&shared);
        right.extend(&shared);

        SimilarSetPair(left.into(), right.into())
    }
}

#[derive(Debug, Clone)]
pub struct SkewedSetPair {
    pub small: SortedSet,
    pub large: SortedSet,
} 

impl quickcheck::Arbitrary for SkewedSetPair {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let small_size = (usize::arbitrary(g) % 128) + 1;
        let large_size = (usize::arbitrary(g) % 8192) + 128;
        let amount_shared = usize::arbitrary(g) % small_size;

        let shared: Vec<i32> = vec_of_len(amount_shared, g);

        let mut small = vec_of_len(small_size - amount_shared, g);
        let mut large = vec_of_len(large_size - amount_shared, g);
        small.extend(&shared);
        large.extend(&shared);

        SkewedSetPair{
            small: small.into(),
            large: large.into()
        }
    }
}

fn vec_of_len(len: usize, g: &mut quickcheck::Gen) -> Vec<i32> {
    let mut result: Vec<i32> = Vec::with_capacity(len);
    while result.len() < len {
        let add = Vec::arbitrary(g);
        result.extend(&add);
        result.truncate(len);
    }
    result
}



// Arbitrary Collection of Sets //
#[derive(Clone, Debug)]
pub struct SetCollection {
    sets: Vec<SortedSet>,
}

impl SetCollection {
    pub fn as_slice(&self) -> &[SortedSet] {
        self.sets.as_slice()
    }
}

impl quickcheck::Arbitrary for SetCollection {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let set_count = u32::arbitrary(g) % 4 + 2;
        let mut sets: Vec<SortedSet> = Vec::new();
        
        let mutual: Vec<i32> = Vec::arbitrary(g);

        for _ in 0..set_count {
            let mut set = Vec::arbitrary(g);
            set.extend(&mutual);
            sets.push(SortedSet::from_unsorted(set));
        }

        Self { sets }
    }
}

