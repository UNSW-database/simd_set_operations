pub mod properties;

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

impl Into<Vec<i32>> for SortedSet {
    fn into(self) -> Vec<i32> {
        self.into_inner()
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
pub struct DualIntersectFn {
    pub name: String,
    pub intersect: Intersect2<[i32], VecWriter<i32>>,
}

impl DualIntersectFn {
    fn new(name: &str, intersect: Intersect2<[i32], VecWriter<i32>>) -> Self {
        Self {
            name: name.into(),
            intersect: intersect,
        }
    }
}

impl fmt::Debug for DualIntersectFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

impl quickcheck::Arbitrary for DualIntersectFn {
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        g.choose([
            DualIntersectFn::new("branchless_merge", intersect::branchless_merge),
            DualIntersectFn::new("galloping", intersect::galloping),
            DualIntersectFn::new("baezayates", intersect::baezayates)
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

