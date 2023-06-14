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
    test_2set_intersect(&[
        1, 14551737, 308423503, 417273473, 731394076, 764331843, 816111760,
        1689942455, 1761460264, 1836814004, 1854053547, 2082830231,
        2295305143, 2318016244, 2404898535, 2523638113, 2539394728,
        2662474414, 2840961257, 2882274791, 3026977316, 3271982067,
        3316245172, 3483020463, 3635510430, 3699103118, 3997987522,
        4004771302, 4022120072, 4217692208, 4294967295,
    ],
    &[0, 1, 4294967295,], &[1,4294967295]);
}

fn test_2set_intersect(left: &[u32], right: &[u32], out: &[u32]) {
    let mut writer = VecWriter::with_capacity(out.len());
    intersect::baezayates(left, right, &mut writer);

    let result: Vec<u32> = writer.into();
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

fn test_adaptive(sets: &[Vec<u32>], expected: Vec<u32>) {
    dbg!(sets);
    dbg!(&expected);

    assert!(sets.iter().all(|set| set.windows(2).all(|w| w[0] < w[1])));

    let mut writer = VecWriter::new();
    intersect::adaptive(&sets, &mut writer);

    let result: Vec<u32> = writer.into();
    dbg!(&result);

    assert!(result == expected);
}

