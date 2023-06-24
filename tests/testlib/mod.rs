pub mod properties;

use quickcheck::Arbitrary;
use setops::{
    intersect::{self, Intersect2},
    visitor::VecWriter,
};
use std::fmt;

// Arbitrary Set //
#[derive(Debug, Clone)]
pub struct SortedSet<T>(Vec<T>)
where
    T: Ord + Arbitrary + Copy;

impl<T> SortedSet<T>
where
    T: Ord + Arbitrary + Copy
{
    pub fn from_unsorted(mut vec: Vec<T>) -> Self {
        vec.sort_unstable();
        vec.dedup();
        Self(vec)
    }

    pub fn as_slice(&self) -> &[T] {
        &self.0
    }

    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T> From<SortedSet<T>> for Vec<T>
where
    T: Ord + Arbitrary + Copy
{
    fn from(value: SortedSet<T>) -> Self {
        value.into_inner()
    }
}

impl<T> From<Vec<T>> for SortedSet<T>
where
    T: Ord + Arbitrary + Copy
{
    fn from(value: Vec<T>) -> Self {
        Self::from_unsorted(value)
    }
}

impl<T> quickcheck::Arbitrary for SortedSet<T>
where
    T: Ord + Arbitrary + Copy
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        Self::from_unsorted(Vec::<T>::arbitrary(g))
    }
}

impl<T> AsRef<[T]> for SortedSet<T>
where
    T: Ord + Arbitrary + Copy
{
    fn as_ref(&self) -> &[T] {
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
pub struct SimilarSetPair<T>(pub SortedSet<T>, pub SortedSet<T>)
where
    T: Ord + Arbitrary + Copy;


impl<T> quickcheck::Arbitrary for SimilarSetPair<T>
where
    T: Ord + Arbitrary + Copy
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let shared: Vec<T> = Vec::arbitrary(g);

        let mut left = Vec::arbitrary(g);
        let mut right = Vec::arbitrary(g);
        left.extend(&shared);
        right.extend(&shared);

        SimilarSetPair(left.into(), right.into())
    }
}

#[derive(Debug, Clone)]
pub struct SkewedSetPair<T>
where
    T: Ord + Arbitrary + Copy
{
    pub small: SortedSet<T>,
    pub large: SortedSet<T>,
} 

impl<T> quickcheck::Arbitrary for SkewedSetPair<T>
where
    T: Ord + Arbitrary + Copy
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let small_size = (usize::arbitrary(g) % 128) + 1;
        let large_size = (usize::arbitrary(g) % 8192) + 128;
        let amount_shared = usize::arbitrary(g) % small_size;

        let shared: Vec<T> = vec_of_len(amount_shared, g);

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

fn vec_of_len<T: Arbitrary>(len: usize, g: &mut quickcheck::Gen) -> Vec<T> {
    let mut result: Vec<T> = Vec::with_capacity(len);
    while result.len() < len {
        let add = Vec::arbitrary(g);
        result.extend(add);
        result.truncate(len);
    }
    result
}



// Arbitrary Collection of Sets //
#[derive(Clone, Debug)]
pub struct SetCollection<T>
where
    T: Ord + Arbitrary + Copy
{
    sets: Vec<SortedSet<T>>,
}

impl<T> SetCollection<T>
where
    T: Ord + Arbitrary + Copy
{
    pub fn as_slice(&self) -> &[SortedSet<T>] {
        self.sets.as_slice()
    }
}

impl<T> quickcheck::Arbitrary for SetCollection<T>
where
    T: Ord + Arbitrary + Copy
{
    fn arbitrary(g: &mut quickcheck::Gen) -> Self {
        let set_count = u32::arbitrary(g) % 4 + 2;
        let mut sets: Vec<SortedSet<T>> = Vec::new();

        let mutual: Vec<T> = Vec::arbitrary(g);

        for _ in 0..set_count {
            let mut set = Vec::<T>::arbitrary(g);
            set.extend(&mutual);
            sets.push(SortedSet::from_unsorted(set));
        }

        Self { sets }
    }
}

