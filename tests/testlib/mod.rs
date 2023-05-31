use setops::{
    intersect::{self, Intersect2, IntersectK},
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
pub struct DualIntersectFn {
    pub name: String,
    pub intersect: Intersect2<[u32], VecWriter<u32>>,
}

impl DualIntersectFn {
    fn new(name: &str, intersect: Intersect2<[u32], VecWriter<u32>>) -> Self {
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
        g.choose(
            [DualIntersectFn::new(
                "branchless_merge",
                intersect::branchless_merge,
            ),
            DualIntersectFn::new(
                "galloping",
                intersect::galloping,
            ),
            DualIntersectFn::new(
                "baezayates",
                intersect::baezayates,
            )
            ]
            .as_slice(),
        )
        .unwrap()
        .clone()
    }
}

#[derive(Clone)]
pub struct KIntersectFn {
    pub name: String,
    pub intersect: IntersectK<[u32], VecWriter<u32>>,
}

impl KIntersectFn {
    fn new(name: &str, intersect: IntersectK<[u32], VecWriter<u32>>) -> Self {
        Self {
            name: name.into(),
            intersect: intersect,
        }
    }
}

impl fmt::Debug for KIntersectFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name)
    }
}

//impl quickcheck::Arbitrary for KIntersectFn {
//    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
//        g.choose(
//            [KIntersectFn::new(
//                "svs",
//                intersect::svs,
//            )]
//            .as_slice(),
//        )
//        .unwrap()
//        .clone()
//    }
//}
