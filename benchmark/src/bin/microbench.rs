use benchmark::{tsc::{self, end, start}, util::{large_median, small_median}};
use std::hint::black_box;
use rand::prelude::*;

const N: u64 = 100;
const R: u64 = 10;

fn main() {
    let tscc = tsc::characterise();

    let mut rng = rand::thread_rng();

    let mut values_vec: Vec<Vec<u64>> = std::iter::repeat_with(|| {
        let mut values: Vec<u64> = (0..N).collect();
        values.shuffle(&mut rng);
        values
    }).take(R as usize).collect();

    let small = (test_small(&mut values_vec) - tscc.overhead) / R;
    let large = (test_large(&mut values_vec) - tscc.overhead) / R;
    println!("large[{}], small[{}], diff[{}]", large, small, large as i64 - small as i64);
}

fn test_small(values_vec: &mut Vec<Vec<u64>>) -> u64 {
    let start = start();
    for values in values_vec {
        black_box(small_median(black_box(values.as_mut_slice())));
    }
    let end = end();
    end - start
}

fn test_large(values_vec: &mut Vec<Vec<u64>>) -> u64 {
    let start = start();
    for values in values_vec {
        black_box(large_median(black_box(values.as_mut_slice())));
    }
    let end = end();
    end - start
}