use setops::{
    intersect::{self, Intersect2},
    visitor::{VecWriter, SliceWriter},
};
use std::fmt;

// Arbitrary Set //
#[derive(Debug, Clone)]
pub struct SortedSet(Vec<u32>);

impl SortedSet {
    pub fn from_unsorted(mut vec: Vec<u32>) -> Self {
        vec.sort_unstable();
        vec.dedup();
        Self(vec)
    }

    pub fn as_slice(&self) -> &[u32] {
        &self.0
    }

    pub fn cardinality(&self) -> usize {
        self.0.len()
    }

    pub fn into_inner(self) -> Vec<u32> {
        self.0
    }
}

impl Into<Vec<u32>> for SortedSet {
    fn into(self) -> Vec<u32> {
        self.into_inner()
    }
}

impl quickcheck::Arbitrary for SortedSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Self::from_unsorted(Vec::<u32>::arbitrary(g))
    }
}

impl AsRef<[u32]> for SortedSet {
    fn as_ref(&self) -> &[u32] {
        &self.0
    }
}

// Arbitrary Intersection Function //
#[derive(Clone)]
pub struct DualIntersectFnVec {
    pub name: String,
    pub intersect: Intersect2<[u32], VecWriter<u32>>,
}

impl DualIntersectFnVec {
    fn new(name: &str, intersect: Intersect2<[u32], VecWriter<u32>>) -> Self {
        Self {
            name: name.into(),
            intersect: intersect,
        }
    }
}

impl fmt::Debug for DualIntersectFnVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl quickcheck::Arbitrary for DualIntersectFnVec {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        g.choose([
            DualIntersectFnVec::new("branchless_merge", intersect::branchless_merge),
            DualIntersectFnVec::new("galloping", intersect::galloping),
            DualIntersectFnVec::new("baezayates", intersect::baezayates)
        ].as_slice())
        .unwrap()
        .clone()
    }
}

// Arbitrary intersect function using SliceWriter
#[derive(Clone)]
pub struct DualIntersectFnSlice {
    pub name: String,
    pub intersect: fn(a: &[u32], b: &[u32], visitor: &mut SliceWriter<u32>) -> usize,
}

impl DualIntersectFnSlice {
    fn new(name: &str, intersect: fn(a: &[u32], b: &[u32], visitor: &mut SliceWriter<u32>) -> usize) -> Self {
        Self {
            name: name.into(),
            intersect: intersect,
        }
    }
}

impl fmt::Debug for DualIntersectFnSlice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl quickcheck::Arbitrary for DualIntersectFnSlice {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        g.choose([
            DualIntersectFnSlice::new("branchless_merge", intersect::branchless_merge_slice),
            DualIntersectFnSlice::new("galloping", intersect::galloping_slice),
            DualIntersectFnSlice::new("baezayates", intersect::baezayates_slice)
        ].as_slice())
        .unwrap()
        .clone()
    }
}


// Arbitrary Collection of Sets //
#[derive(Clone, Debug)]
pub struct SetCollection {
    sets: Vec<SortedSet>,
}

impl SetCollection {
    pub fn sets(&self) -> &Vec<SortedSet> {
        &self.sets
    }

    pub fn into_inner(self) -> Vec<SortedSet> {
        self.sets
    }
}

impl quickcheck::Arbitrary for SetCollection {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let set_count = u32::arbitrary(g) % 4 + 2;
        let mut sets: Vec<SortedSet> = Vec::new();
        
        let mutual: Vec<u32> = Vec::arbitrary(g);

        for _ in 0..set_count {
            let mut set = Vec::arbitrary(g);
            set.extend(&mutual);
            sets.push(SortedSet::from_unsorted(set));
        }

        Self { sets }
    }
}

