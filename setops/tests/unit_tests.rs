#![allow(non_snake_case)]
use std::collections::BTreeSet;
use rand::rngs::StdRng;
use rand::SeedableRng;
use setops::{visitor::VecWriter, intersect};
use setops::intersect::{small_adaptive, BroadcastK, Gather, Cuda};
use setops::KSetInput::KSetInput;

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
// fn test_kset_intersect()

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
#[cfg(target_feature = "avx2")]
#[test]
fn test_gather() {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..10 {
        let minSizeOfResult = rng.gen_range(1..100);
        let mut minResultSet : BTreeSet<i32> = std::collections::BTreeSet::new();
        for _ in 0..minSizeOfResult {
            minResultSet.insert(rng.gen_range(0..1000));
        }
        let minResultVec : Vec<i32> =  minResultSet.clone().into_iter().collect();
        let mut setVec = Vec::<Vec::<i32>>::new();
        for i in 0..100 {
            let mut set = BTreeSet::<i32>::new();
            for _ in 0..100 {
                set.insert(rng.gen_range(0..1000));
            }
            for ele in minResultSet.iter() {
                set.insert(*ele);
            }
            let vec: Vec<i32> = set.into_iter().collect();
            setVec.push(vec);
        } 
        let mut writer = VecWriter::<i32>::with_capacity(100usize);
        small_adaptive(&*setVec, &mut writer);
        let expected : Vec<i32> = writer.into();
        let ksetInput = KSetInput::new(&setVec[1..]);
        writer = VecWriter::<i32>::with_capacity(100usize);
        Gather(&setVec[0], &ksetInput, &mut writer);
        let actual : Vec<i32> = writer.into();
        assert_eq!(actual, expected);
    }
}
#[cfg(target_feature = "avx2")]
#[test]
fn test_broadcastK() {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..10 {
        let minSizeOfResult = rng.gen_range(1..100);
        let mut minResultSet : BTreeSet<i32> = std::collections::BTreeSet::new();
        for _ in 0..minSizeOfResult {
            minResultSet.insert(rng.gen_range(0..1000));
        }
        let minResultVec : Vec<i32> =  minResultSet.clone().into_iter().collect();
        let mut setVec = Vec::<Vec::<i32>>::new();
        for i in 0..100 {
            let mut set = BTreeSet::<i32>::new();
            for _ in 0..100 {
                set.insert(rng.gen_range(0..1000));
            }
            for ele in minResultSet.iter() {
                set.insert(*ele);
            }
            let vec: Vec<i32> = set.into_iter().collect();
            setVec.push(vec);
        }
        let mut writer = VecWriter::<i32>::with_capacity(100usize);
        small_adaptive(&*setVec, &mut writer);
        let expected : Vec<i32> = writer.into();
        let ksetInput = KSetInput::new(&setVec[1..]);
        writer = VecWriter::<i32>::with_capacity(100usize);
        BroadcastK(&setVec[0], &ksetInput, &mut writer);
        let actual : Vec<i32> = writer.into();
        assert_eq!(actual, expected);
    }
    
}
#[test]
fn test_CudaBroadcast() {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(0);
    for _ in 0..10 {
        let minSizeOfResult = rng.gen_range(1..100);
        let mut minResultSet : BTreeSet<i32> = std::collections::BTreeSet::new();
        for _ in 0..minSizeOfResult {
            minResultSet.insert(rng.gen_range(0..1000));
        }
        let minResultVec : Vec<i32> =  minResultSet.clone().into_iter().collect();
        let mut setVec = Vec::<Vec::<i32>>::new();
        for i in 0..10000 {
            let mut set = BTreeSet::<i32>::new();
            for _ in 0..100 {
                set.insert(rng.gen_range(0..1000));
            }
            for ele in minResultSet.iter() {
                set.insert(*ele);
            }
            let vec: Vec<i32> = set.into_iter().collect();
            setVec.push(vec);
        }
        let mut writer = VecWriter::<i32>::with_capacity(100usize);
        let kSetInput = KSetInput::new(&setVec);
        Gather(&setVec[0], &kSetInput, &mut writer);
        let expected : Vec<i32> = writer.into();
        writer = VecWriter::<i32>::with_capacity(100usize);
        let (input, ownedData) = Cuda::convert(&kSetInput);
        Cuda::cudaGatherWarp(input, &mut writer);
        let actual : Vec<i32> = writer.into();
        assert_eq!(actual, expected);
    }
}
