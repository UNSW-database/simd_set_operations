use benchmark::{tsc::{self, end, start}, util::{large_median, random_subset, small_median}};
use std::hint::black_box;
use rand::prelude::*;

const N: u64 = 100;
const R: u64 = 10;

const DATA_SIZE: usize = 1024 * 1024 * 1024 / 8;

fn main() {
    let tscc = tsc::characterise();

    let mut rng = rand::thread_rng();
    let rand_data: Vec<u64> = std::iter::repeat_with(|| rng.gen()).take(DATA_SIZE).collect();
    let rand_indices = random_subset(0..DATA_SIZE, DATA_SIZE, &mut rng);

    let mut values_vec: Vec<Vec<u64>> = std::iter::repeat_with(|| {
        let mut values: Vec<u64> = (0..N).collect();
        values.shuffle(&mut rng);
        values
    }).take(R as usize).collect();

    let small = (test_small(&mut values_vec) - tscc.overhead) / R;
    let large = (test_large(&mut values_vec) - tscc.overhead) / R;
    println!("large[{}], small[{}], diff[{}]", large, small, large as i64 - small as i64);

    let (sum, cc) = test_cache_clear(&rand_data, &rand_indices);
    let time = (cc - tscc.overhead) as f64 / tscc.frequency as f64;
    println!("sum[{}], time[{}]", sum, time);
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

fn test_cache_clear(data: &[u64], indices: &[usize]) -> (u64, u64) {
    let start = start();
    let mut sum: u64 = 0;
    for &index in indices {
        sum = sum.wrapping_add(unsafe{*data.get_unchecked(index)});
    }
    let end = end();
    (sum, end - start)
}