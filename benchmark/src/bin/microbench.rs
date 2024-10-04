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

    let (min, max, td) = test_instant_variance();
    println!("min[{min}], max[{max}], td[{td}]");

    test_perf_counter_overhead();
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

fn test_instant_variance() -> (u64, u64, u64) {
    let si = std::time::Instant::now();
    let mut td = si.duration_since(si).as_secs();
    let mut min = u64::max_value();
    let mut max = u64::min_value();
    for _ in 0..100000 {
        let start = start();
        let now = std::time::Instant::now();
        let end = end();
        let delta = end - start;
        let tdd = now.duration_since(si).as_secs();
        if delta < min {
            min = delta;
        }
        if delta > max {
            max = delta;
        }
        if tdd > td {
            td = tdd;
        }
    }
    (min, max, td)
}

#[cfg(not(target_os = "linux"))]
fn test_perf_counter_overhead() {
    println!("Not on linux. Can't test performance counter overhead.");
}

#[cfg(target_os = "linux")]
fn test_perf_counter_overhead() {
    use perf_event::{Builder, Group, events::Hardware};

    let mut group = match Group::new() {
        Ok(group) => group,
        Err(e) => {
            println!("Failed to create group: {e}");
            return;
        }
    };
    
    let cycles = match group.add(&Builder::new(Hardware::CPU_CYCLES)) {
        Ok(cycles) => cycles,
        Err(e) => {
            println!("Failed to create cycle counter: {e}");
            return;
        }
    };

    let mut sum = 0u64;
    let mut sum2 = 0u64;
    let mut min = u64::MAX;
    let mut max = 0;

    const N: u64 = 1000;

    for _ in 0..N {
        group.enable().unwrap();
        group.disable().unwrap();
        let counts = group.read().unwrap();
        let cycle_count = counts[&cycles];
        sum += cycle_count;
        sum2 += cycle_count * cycle_count;
        if cycle_count < min {
            min = cycle_count;
        }
        if cycle_count > max {
            max = cycle_count;
        }
    }

    let average = sum as f64 / N as f64;
    let sample_variance = (sum2 as f64 - (sum * sum) as f64 / N as f64) / (N - 1) as f64;
    let sample_std_dev = f64::sqrt(sample_variance);
    println!("\nCycle Count Results\nAverage: {average}\nStd. Dev: {sample_std_dev}\nMin: {min}\nMax: {max}");
}

