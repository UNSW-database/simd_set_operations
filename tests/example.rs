use setops::intersect::{NaiveMerge, SortedIntersect2};

#[test]
fn test_simple() {
    let x = [1,3,6];
    let y = [3,4,5,6];
    let mut res = [0; 3];

    let size: usize = NaiveMerge::intersect(
        x.as_slice(), y.as_slice(), res.as_mut_slice());

    assert_eq!(size, 2);
    assert_eq!(res[0..size], [3,6]);
}
