use crate::{
    intersect, 
    visitor::{Visitor, VecWriter, SliceWriter},
};


/// "Small vs. Small" adaptive set intersection algorithm.
/// Assumes input sets are ordered from smallest to largest.
pub fn svs_galloping<T, S>(sets: &[S], out: &mut [T]) -> usize
where
    T: Ord + Copy,
    S: AsRef<[T]>,
{
    assert!(sets.len() >= 2);

    let mut writer = SliceWriter::from(&mut *out);
    intersect::galloping(sets[0].as_ref(), sets[1].as_ref(), &mut writer);

    let mut count = writer.position();

    for set in sets.iter().skip(2) {
        count = intersect::galloping_inplace(&mut out[..count], set.as_ref());
    }
    count
}

pub fn svs_galloping_inplace<T, S>(sets: &mut [S]) -> usize
where
    T: Ord + Copy,
    S: AsMut<[T]>,
{
    assert!(sets.len() >= 2);

    let mut count = 0;
    let mut iter = sets.iter_mut();

    let first = iter.next().unwrap().as_mut();

    for set in iter {
        count = intersect::galloping_inplace(&mut first[0..count], set.as_mut());
    }
    count
}


/// Extends 2-set intersection algorithms to k-set.
/// Since SIMD algorithms cannot operate in place, to extend them to k sets, we
/// must use two output vectors.
/// Returns (intersection length, final output index)
pub fn svs_generic<'a, T, S, V>(
    sets: &[S],
    mut left: &'a mut V,
    mut right: &'a mut V,
    intersect: fn(&[T], &[T], &mut V)
) -> &'a mut V
where
    T: Ord + Copy,
    S: AsRef<[T]>,
    V: Visitor<T> + AsRef<[T]>,
{
    assert!(sets.len() >= 2);

    intersect(sets[0].as_ref(), sets[1].as_ref(), left);

    for set in sets.iter().skip(2) {
        // Alternate output sets.
        std::mem::swap(&mut left, &mut right);
        left.clear();
        intersect(right.as_ref(), set.as_ref(), &mut left);
    }

    left
}

/// Convenience function which makes calling svs_generic simpler for users and
/// tests. For code requiring zero allocation (like benchmarking), use
/// svs_generic directly. See svs_generic for details.
pub fn run_svs_generic<T, S>(
    sets: &[S],
    intersect: fn(&[T], &[T], &mut VecWriter<T>)) -> Vec<T>
where
    T: Ord + Copy + Default,
    S: AsRef<[T]>,
{
    let mut left: VecWriter<T> = VecWriter::new();
    let mut right: VecWriter<T> = VecWriter::new();

    let result = svs_generic(&sets, &mut left, &mut right, intersect);

    std::mem::take(result).into()
}
