use crate::{
    intersect::{
        galloping_inplace,
        Intersect2,
    },
    visitor::Visitor,
};


/// "Small vs. Small" adaptive set intersection algorithm.
/// Assumes input sets are ordered from smallest to largest.
pub fn svs<T>(sets: &[&[T]], out: &mut [T]) -> usize
where
    T: Ord + Copy,
{
    assert!(sets.len() >= 2);

    let mut count = 0;

    // Copies smallest set into (temporary) output set.
    // Is there a better way to do this?
    out[..sets[0].len()].clone_from_slice(sets[0]);

    for set in sets.iter().skip(1) {
        count = galloping_inplace(&mut out[..count], set);
    }
    count
}

pub fn svs_inplace<T>(sets: &mut [&mut [T]]) -> usize
where
    T: Ord + Copy,
{
    assert!(sets.len() >= 2);

    let mut count = 0;
    let mut iter = sets.iter_mut();

    let first = unsafe { iter.next().unwrap_unchecked() };

    for set in iter {
        count = galloping_inplace(&mut first[0..count], set);
    }
    count
}



// Opt 1: store mut reference?
// Opt 2: store lambda?
// Opt 3: force inplace alg

pub fn as_svs<T, V>(sets: &[&[T]], out: &mut [T], intersect: Intersect2<T, V>) -> usize
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    assert!(sets.len() >= 2);

    let mut count = 0;

    // TODO: intersect first two sets

    for set in sets.iter().skip(1) {
        //count = intersect(&out[..count], set, visitor);
    }
    count
}
