use setops::{visitor::VecWriter, intersect};


// Sanity check
#[cfg(test)]
#[test]
fn test_2set_intersect1() {
    test_2set_intersect(&[1,2,3,4], &[1,2,3,4,5], &[1,2,3,4]);
}

#[test]
fn test_2set_intersect2() {
    test_2set_intersect(&[0,4,5,8], &[1,2,3,6], &[]);
}

#[test]
fn test_2set_intersect3() {
    test_2set_intersect(&[1,4,5], &[1,4,5], &[1,4,5]);
}

#[test]
fn test_2set_intersect4() {
    test_2set_intersect(&[10,42],
        &[1,2,3,4,5,6,7,8,9,10,22,25,28,39,42,43,47,49], &[10,42]);
}

#[test]
fn test_2set_intersect5() {
    const A: [i32; 10] = [1,3,5,8,9,10,14,15,18,20];
    const B: [i32; 10] = [1,2,3,4,9,10,11,15,20,21];
    const EXP: [i32; 6] = intersect::const_intersect(&A, &B);
    test_2set_intersect(&A, &B, &EXP);
}

fn test_2set_intersect(left: &[i32], right: &[i32], out: &[i32]) {
    let mut writer = VecWriter::with_capacity(out.len());
    intersect::naive_merge(left, right, &mut writer);

    let result: Vec<i32> = writer.into();

    println!("got: {:?}", result);
    println!("expected: {:?}", out);

    assert!(result == out);
}

#[cfg(feature = "simd")]
#[test]
fn test_simd_galloping() {
    const MAX: i32 = 12345;

    let small = vec![1<<12 + 1];
    let large = Vec::from_iter(0..MAX);

    let expected = intersect::run_2set(small.as_slice(), large.as_slice(), intersect::branchless_merge);
    let actual = intersect::run_2set(small.as_slice(), large.as_slice(), intersect::galloping_sse);

    assert!(actual == expected);
}
