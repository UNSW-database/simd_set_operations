use setops::{
    intersect::{self, Intersect2},
    visitor::VecWriter,
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
}

impl Into<Vec<u32>> for SortedSet {
    fn into(self) -> Vec<u32> {
        self.0
    }
}

impl quickcheck::Arbitrary for SortedSet {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Self::from_unsorted(Vec::<u32>::arbitrary(g))
    }
}

// Arbitrary Intersection Function //
#[derive(Clone)]
pub struct IntersectFn {
    pub name: String,
    pub intersect: Intersect2<[u32], VecWriter<u32>>,
}

impl IntersectFn {
    fn new(name: &str, intersect: Intersect2<[u32], VecWriter<u32>>) -> Self {
        Self {
            name: name.into(),
            intersect: intersect,
        }
    }
}

impl fmt::Debug for IntersectFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl quickcheck::Arbitrary for IntersectFn {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        g.choose(
            [IntersectFn::new(
                "branchless_merge",
                intersect::branchless_merge,
            )]
            .as_slice(),
        )
        .unwrap()
        .clone()
    }
}
