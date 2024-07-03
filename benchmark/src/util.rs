use rand::{distributions::Distribution, Rng, seq::SliceRandom};
use std::{collections::HashSet, hash::Hash};

pub fn slice_i32_to_u32(slice_i32: &[i32]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice_i32.as_ptr() as *const u32, slice_i32.len()) }
}

pub fn slice_u32_to_i32(slice_u32: &[u32]) -> &[i32] {
    unsafe { std::slice::from_raw_parts(slice_u32.as_ptr() as *const i32, slice_u32.len()) }
}

// Conversion of integers to arrays of bytes and vectors of integers to vectors of bytes, all native orders
pub trait Byteable<const N: usize> {
    fn to_bytes(&self) -> [u8; N];
}

macro_rules! byteable_int {
    ( $( $t:ty ),* ) => {
        $(
impl Byteable<{std::mem::size_of::<$t>()}> for $t {
    fn to_bytes(&self) -> [u8; std::mem::size_of::<$t>()] {
        self.to_ne_bytes()
    }
}
        )*
    }
}

byteable_int! {u8, u16, u32, u64, i8, i16, i32, i64, usize}

pub fn vec_to_bytes<const N: usize, T: Byteable<N>>(vec: &Vec<T>) -> Vec<u8> {
    vec.iter().flat_map(|i| i.to_bytes()).collect()
}

// Trait that allows you to access the maximum value for a type
pub trait Max {
    fn max() -> Self;
}

macro_rules! max_int {
    ( $( $t:ty ), *) => {
        $(
impl Max for $t {
    fn max() -> Self {
        Self::MAX
    }
}
        )*
    }
}

max_int! {u8, u16, u32, u64, i8, i16, i32, i64, usize}

// Random sampling helpers
pub fn sample_distribution_unique<T: Eq + Hash + Copy>(
    total_length: usize,
    distribution: &impl Distribution<T>,
    rng: &mut impl Rng,
) -> Vec<T> {
    let mut set = HashSet::<T>::with_capacity(total_length);
    let mut vec = Vec::<T>::with_capacity(total_length);

    while vec.len() != total_length {
        let value = distribution.sample(rng);
        if set.insert(value) {
            vec.push(value);
        }
    }

    vec
}

pub fn random_subset<T>(
    value_range: impl IntoIterator<Item = T>,
    total_length: usize,
    rng: &mut impl Rng,
) -> Vec<T> {
    let mut vec: Vec<T> = value_range.into_iter().collect();
    vec.shuffle(rng);
    vec.truncate(total_length);
    vec
}
