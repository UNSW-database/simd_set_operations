use rand::{distributions::Distribution, Rng, seq::SliceRandom};
use std::{collections::HashSet, fmt::Display, hash::Hash};

pub fn slice_i32_to_u32(slice_i32: &[i32]) -> &[u32] {
    unsafe { std::slice::from_raw_parts(slice_i32.as_ptr() as *const u32, slice_i32.len()) }
}

pub fn slice_u32_to_i32(slice_u32: &[u32]) -> &[i32] {
    unsafe { std::slice::from_raw_parts(slice_u32.as_ptr() as *const i32, slice_u32.len()) }
}

// Conversion of integers to arrays of bytes and vectors of integers to vectors of bytes, all native orders
pub trait Byteable<const N: usize> {
    fn to_bytes(&self) -> [u8; N];
    fn from_bytes(bytes: &[u8; N]) -> Self;
}

macro_rules! byteable_int {
    ( $( $t:ty ),* ) => {
        $(
impl Byteable<{std::mem::size_of::<$t>()}> for $t {
    fn to_bytes(&self) -> [u8; std::mem::size_of::<$t>()] {
        self.to_ne_bytes()
    }

    fn from_bytes(bytes: &[u8; std::mem::size_of::<$t>()]) -> Self {
        Self::from_ne_bytes(*bytes)
    }
}
        )*
    }
}

byteable_int! {u8, u16, u32, u64, i8, i16, i32, i64, usize}

pub fn vec_to_bytes<const N: usize, T: Byteable<N>>(vec: &[T]) -> Vec<u8> {
    vec.iter().flat_map(|i| i.to_bytes()).collect()
}

pub fn bytes_to_vec<const N: usize, T: Byteable<N>>(bytes: &[u8]) -> Vec<T> {
    assert!(bytes.len() % N == 0);
    bytes.array_chunks::<N>().map(|c| T::from_bytes(c)).collect()
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

// Checked conversion helpers
pub fn to_usize<T>(value: T, name: &str) -> Result<usize, String> 
where 
    T: Display + Clone + Copy,
    usize: TryFrom<T>,
{
    value.try_into().or(Err(format!("Could not convert {} ({}) to usize.", name, value )))
}

pub fn to_u64<T>(value: T, name: &str) -> Result<u64, String> 
where 
    T: Display + Clone + Copy,
    u64: TryFrom<T>,
{
    value.try_into().or(Err(format!("Could not convert {} ({}) to u64.", name, value )))
}

pub fn is_ascending<T: PartialOrd>(values: &[T]) -> bool {
    values.windows(2).all(|w| {
        let (lhs, rhs) = unsafe { (w.get_unchecked(0), w.get_unchecked(1)) };
        lhs < rhs
    })
}

// Calculate the median of 3 or 3x3 values in a way that the compiler can 
// easily optimise down to cmp and cmov instructions
// See: https://godbolt.org/z/Px314KWE7
pub fn median3_u64(values: &[u64]) -> u64 {
    assert!(values.len() == 3);

    let mut a = values[0];
    let mut b = values[1];
    let mut c = values[2];
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    if b > c {
        std::mem::swap(&mut b, &mut c);
    }
    if a > b {
        std::mem::swap(&mut a, &mut b);
    }
    b
}

// Calculate median without sorting for small slices of values. Performance vs.
// large_median is highly dependent on warm vs. cold, beating cold large_median
// up to about len == 200, but only beating up to len == 20 for warm.
pub fn small_median(values: &[u64]) -> u64 {
    let lt = ((values.len() - 1) / 2) as u64;
    for &val_o in values {
        let mut ltc = 0;
        let mut eqc = 0;
        for &val_i in values {
            if val_i < val_o {
                ltc += 1;
            }
            if val_i == val_o {
                eqc += 1;
            }
        }
        if ltc <= lt && ltc + eqc > lt {
            return val_o;
        }
    }
    std::unreachable!();
}

// Calculate median of a large slice
pub fn large_median(values: &mut [u64]) -> u64 {
    values.sort_unstable();
    values[values.len() / 2]
}
