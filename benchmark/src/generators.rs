use std::collections::HashSet;

use crate::{schema::{SetInfo, PERCENT_F}, datafile::DatafileSet};

use colored::Colorize;
use rand::{distributions::Uniform, thread_rng, Rng, seq::SliceRandom};

const MIN_SET_LENGTH: usize = 100;

struct GenContext {
    pub density: f64,
    pub selectivity: f64,
    pub max_len: usize,
    pub skewness_factor: u32,
}

impl From<&SetInfo> for GenContext {
    fn from(props: &SetInfo) -> Self {
        Self {
            density:     props.density     as f64 / PERCENT_F,
            selectivity: props.selectivity as f64 / PERCENT_F,
            max_len: 1 << props.max_len,
            skewness_factor: props.skewness_factor,
        }
    }
}

pub fn gen_twoset(props: &SetInfo) -> (DatafileSet, DatafileSet) {
    let gen: GenContext = props.into();

    let large_len = gen.max_len;
    let small_len = large_len / get_skew(1, gen.skewness_factor);

    if small_len < MIN_SET_LENGTH {
        warn_set_len(small_len);
    }

    let max_value = (large_len as f64 / gen.density) as i32;

    let (target_shared_count, target_gen_count) =
        get_gen_counts(gen.selectivity, small_len, large_len);

    let (shared_count, gen_count) = if target_gen_count as i32 > max_value {
        let shared_count = small_len + large_len - max_value as usize;
        warn_selectivity(shared_count, small_len, gen.selectivity, gen.density);
        (shared_count, max_value as usize)
    }
    else {
        (target_shared_count, target_gen_count)
    };

    let values = shuffled_set(gen_count, max_value);

    let (shared, unshared) = values.split_at(shared_count);
    let (only_small, only_large) = unshared.split_at(small_len - shared_count);

    let mut small = [shared, only_small].concat();
    let mut large = [shared, only_large].concat();
    small.sort_unstable();
    large.sort_unstable();

    assert!(small.len() == small_len);
    assert!(large.len() == large_len);

    (small, large)
}

fn get_gen_counts(
    selectivity: f64,
    small_len: usize,
    large_len: usize) -> (usize, usize)
{
    let shared_count = (selectivity * small_len as f64) as usize;
    let different_count = small_len + large_len - 2*shared_count;
    let gen_count = shared_count + different_count;
    (shared_count, gen_count)
}

/// Returns a random set of length `result_len` with a domain of 0 to
/// `max_value-1`. Values are uniformly distributed.
fn shuffled_set(
    result_len: usize,
    max_value: i32) -> Vec<i32>
{
    let rng = &mut thread_rng();
    let distribution = uniform_up_to(max_value);

    // if gen_count is <50% of domain
    let low_density = result_len * 2 < max_value as usize;
    if low_density {
        let mut items: Vec<i32> = Vec::new();
        while items.len() < result_len {
            let need = result_len - items.len();
            items.extend(rng.sample_iter(distribution).take(need * 2));
            items.sort_unstable();
            items.dedup();
        }
        items.shuffle(rng);
        items.truncate(result_len);
        items
    }
    else {
        let mut everything: Vec<i32> = (0..max_value).collect();
        everything.shuffle(rng);
        everything.truncate(result_len);
        everything
    }
}

pub fn gen_kset(props: &SetInfo, set_count: usize) -> Vec<DatafileSet> {
    let gen: GenContext = props.into();

    let max_value = (gen.max_len as f64 / gen.density) as i32;

    let min_len = gen.max_len / get_skew(set_count - 1, gen.skewness_factor);
    if min_len < MIN_SET_LENGTH {
        warn_set_len(min_len);
    }

    let shared_count = (gen.selectivity * min_len as f64) as usize;
    let shared = shuffled_set(shared_count, max_value);

    let mut sets = Vec::with_capacity(set_count);

    for set_index in 0..set_count {
        let set_len = gen.max_len / get_skew(set_index, gen.skewness_factor);
        let set = sorted_set_containing(&shared, set_len, max_value);
        sets.push(set);
    }

    assert!(sets.len() == set_count);
    sets
}

/// Same as `shuffed_set` but result is sorted and all elements from `include`
/// must be present.
fn sorted_set_containing(
    include: &[i32],
    result_len: usize,
    max_value: i32) -> Vec<i32>
{
    assert!(result_len >= include.len());

    // if gen_count is <50% of domain
    let low_density = result_len * 2 < max_value as usize;

    if low_density {
        sorted_set_low_density_containing(include, result_len, max_value)
    }
    else {
        sorted_set_high_density_containing(include, result_len, max_value)
    }
}

// TODO: double check and test.
fn sorted_set_low_density_containing(
    include_slice: &[i32],
    result_len: usize,
    max_value: i32) -> Vec<i32>
{
    let rng = &mut thread_rng();
    let distribution = uniform_up_to(max_value);

    let included: HashSet<i32> = include_slice.iter().copied().collect();
    let mut not_included: Vec<i32> = Vec::with_capacity(result_len - include_slice.len());

    let not_included_len = result_len - include_slice.len();
    while not_included.len() < not_included_len {
        let need = result_len - not_included.len();
        not_included.extend(rng
            .sample_iter(distribution)
            .filter(|v| !included.contains(v))
            .take(need * 2));

        not_included.sort_unstable();
        not_included.dedup();
    }
    not_included.shuffle(rng);
    not_included.truncate(not_included_len);

    let mut result = not_included;
    result.extend(include_slice);
    result.sort_unstable();

    assert!(result.len() == result_len);
    result
}

fn sorted_set_high_density_containing(
    include_slice: &[i32],
    result_len: usize,
    max_value: i32) -> Vec<i32>
{
    let rng = &mut thread_rng();

    let included: HashSet<i32> = include_slice.iter().copied().collect();

    let mut not_included: Vec<i32> =
        if include_slice.len() > 0 {
            (0..max_value).filter(|v| !included.contains(v)).collect()
        }
        else {
            (0..max_value).collect()
        };
    not_included.shuffle(rng);
    not_included.truncate(result_len);

    let mut result = not_included;
    result.extend(include_slice);
    result.sort_unstable();

    result
}

/// The skew WRT the largest set is k^f where f is the skewness factor.
/// The size of the kth set is S_1/(k^f)
/// `set_index` is 0-based.
fn get_skew(set_index: usize, skew_factor: u32) -> usize {
    (set_index + 1).pow(skew_factor)
}

fn uniform_up_to(max_value: i32) -> Uniform<i32> {
    Uniform::from(0..max_value)
}

#[cfg(debug_assertions)]
fn warn_selectivity(
    shared_count: usize,
    small_len: usize,
    target_selectivity: f64,
    density: f64)
{
    let actual_selectivity = shared_count as f64 / small_len as f64;
    let warning = format!(
        "\nwarning: target selectivity {:.2} \
        is unachievable with density {:.2}\n(minimum selectivity {:.2})",
        target_selectivity, density, actual_selectivity
    );
    println!("{}", warning.yellow());
}

#[cfg(not(debug_assertions))]
fn warn_selectivity(
    _shared_count: usize,
    _small_len: usize,
    _target_selectivity: f64,
    _density: f64) {}

fn warn_set_len(len: usize) {
    println!("{}", format!(
        "warning: smallest set is of length {}",
        len).yellow());
}

// TODO: also return "real" selectivity for plotting
