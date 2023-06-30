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

#[test]
fn test_adaptive_simple1() {
    let sets = [
        vec![1,3,4],
        vec![3,6,7],
        vec![1,2,3],
    ];
    let expected = vec![3];

    test_adaptive(sets.as_slice(), expected);
}

#[test]
fn test_adaptive_simple2() {
    let sets = [
        vec![3,4,6,7],
        vec![3,6,7,8,9],
        vec![3,4,5,6,7,8,9],
    ];
    let expected = vec![3,6,7];

    test_adaptive(sets.as_slice(), expected);
}

#[test]
fn test_adaptive_skewed_shrink() {
    let sets = [
        vec![12,52,95],
        vec![2,9,12,52,69,95],
        vec![6,12,36,52,70,95,100],
    ];
    let expected = vec![12,52,95];

    test_adaptive(sets.as_slice(), expected);
}

#[test]
fn test_adaptive_skewed() {
    let sets = [
        vec![12,21,52,95],
        vec![2,4,6,7,8,9,10,11,12,14,16,18,20,25,28,29,35,39,46,52,69,72,95],
        vec![1,4,6,8,12,13,14,18,19,21,22,23,24,25,28,31,35,36,51,52,70,80,90,95,100],
    ];
    let expected = vec![12,52,95];

    test_adaptive(sets.as_slice(), expected);
}

fn test_adaptive(sets: &[Vec<i32>], expected: Vec<i32>) {
    assert!(sets.iter().all(|set| set.windows(2).all(|w| w[0] < w[1])));

    let result = intersect::run_kset(&sets, intersect::adaptive);

    assert!(result == expected);
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
