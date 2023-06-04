use crate::{
    intersect::galloping_inplace,
    visitor::{SliceWriter},
};


/// "Small vs. Small" adaptive set intersection algorithm.
/// Assumes input sets are ordered from smallest to largest.
pub fn svs_galloping<T>(sets: &[&[T]], out: &mut [T]) -> usize
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

pub fn svs_galloping_inplace<T>(sets: &mut [&mut [T]]) -> usize
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



/// Extends 2-set intersection algorithms to k-set.
/// Since SIMD algorithms cannot operate in place, to extend them to k sets, we
/// must use an two output vectors.
/// Returns (intersection length, final output index)
pub fn svs_generic<T, S>(
    sets: &[S],
    out0: &mut [T],
    out1: &mut [T],
    intersect: fn(&[T], &[T], &mut SliceWriter<T>) -> usize
) -> (usize, usize)
where
    T: Ord + Copy,
    S: AsRef<[T]>,
{
    assert!(sets.len() >= 2);
    // Output sets must be large enough to hold intermediate intersection values.
    assert!(sets.iter().all(|set| {
        let len = set.as_ref().len();
        out0.len() >= len && out1.len() >= len
    }));

    let mut count = 0;
    let mut out_index = 0;
    
    let mut writer = SliceWriter::from(&mut *out0);
    count = intersect(
        sets[0].as_ref(),
        sets[1].as_ref(), &mut writer);
    

    for set_b in sets.iter().skip(2) {
        // Alternate output sets.
        let (mut writer, set_a) = if out_index == 0 {
            out_index = 1;
            (SliceWriter::from(&mut *out1), &out0[..count])
        }
        else {
            out_index = 0;
            (SliceWriter::from(&mut *out0), &out1[..count])
        };
        count = intersect(&set_a, set_b.as_ref(), &mut writer);
    }

    (count, out_index)
}

/// Convenience function which makes calling svs_generic simpler for users and
/// tests. For code requiring zero allocation (like benchmarking), use
/// svs_generic directly.
pub fn run_svs_generic<T, S>(
    sets: &[S],
    intersect: fn(&[T], &[T], &mut SliceWriter<T>) -> usize) -> Vec<T>
where
    T: Ord + Copy + Default,
    S: AsRef<[T]>,
{
    let result_len = sets.iter()
        .map(|set| set.as_ref().len()).max().unwrap();

    let mut outputs: Vec<T> = vec![T::default(); result_len * 2];
    let (left, right) = outputs.split_at_mut(result_len);
    
    let (count, index) = svs_generic(
        &sets, left, right, intersect);
    
    match index {
        0 => (),
        1 => {
            for i in 0..count {
                outputs[i] = outputs[i + result_len];
            }
        },
        _ => panic!("Invalid out index!"),
    };
    outputs.truncate(count);
    outputs
}
