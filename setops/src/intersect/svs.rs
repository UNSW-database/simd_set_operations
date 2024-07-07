use super::TwoSetAlgorithmFnGeneric;

/// Extends a 2-set intersection algorithm to k-set.
/// 
/// Small vs. Small or SvS is an algorithm for adapting 2-set intersection algorithms into k-set intersection
/// algorithms. Originally described by Demaine et. al in <https://doi.org/10.1007/3-540-44808-X_7> in which the 
/// algorithm used exponential search based 2-set intersection. This implementation generalises that approach to any 
/// 2-set intersection algorithm.
/// 
/// Conforms to [super::TwoSetToKSetBufFnGeneric], see there for more usage details.
/// 
pub fn svs<T: Ord + Copy>(twoset_fn: TwoSetAlgorithmFnGeneric<T>, sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize
{
    // K-Set algorithms require at least 2 sets
    assert!(sets.len() > 1);

    // We select the first buffer in outs as the current output buffer then swap the order per intersection call.
    let mut outs = (out, buf);

    // We choose the starting order such that the last output buffer is `out`.
    if sets.len() % 2 == 1 {
        outs = (outs.1, outs.0);
    }

    // We run the initial intersection separately as its the only one that uses two sets from `sets`.
    let mut count = twoset_fn((sets[0], sets[1]), outs.0);

    // We intersect the remaining sets with the result of the previous intersection(s), swapping the input and output
    // buffer as we go.
    for &set in sets.iter().skip(2) {
        count = twoset_fn((outs.0, set), outs.1);
        outs = (outs.1, outs.0);
    }

    count
}
