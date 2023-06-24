mod merge;
mod galloping;
mod svs;
mod adaptive;
mod std_set;
mod simd_shuffling;
mod simd_galloping;
mod bmiss;

pub use merge::*;
pub use galloping::{galloping, galloping_inplace, galloping_bsr};
pub use adaptive::*;
pub use std_set::*;
pub use svs::*;
pub use bmiss::*;

#[cfg(feature = "simd")]
pub use {
    simd_shuffling::*,
    simd_galloping::*,
};

use crate::{visitor::VecWriter, bsr::{BsrVec, BsrRef}};

pub type Intersect2<I, V> = fn(a: &I, b: &I, visitor: &mut V);
pub type IntersectK<S, V> = fn(sets: &[S], visitor: &mut V);

pub fn run_2set<T>(
    set_a: &[T],
    set_b: &[T],
    intersect: Intersect2<[T], VecWriter<T>>) -> Vec<T>
{
    let mut writer: VecWriter<T> = VecWriter::new();
    intersect(set_a, set_b, &mut writer);
    writer.into()
}

pub fn run_kset<T, S>(sets: &[S], intersect: IntersectK<S, VecWriter<T>>) -> Vec<T>
where
    T: Ord + Copy,
    S: AsRef<[T]>,
{
    assert!(sets.len() >= 2);

    let mut writer: VecWriter<T> = VecWriter::new();
    intersect(sets, &mut writer);
    writer.into()
}

pub fn run_2set_bsr<'a, S>(
    set_a: S,
    set_b: S,
    intersect: fn(l: S, r: S, v: &mut BsrVec)) -> BsrVec
where
    S: Into<BsrRef<'a>>,
{
    let mut writer = BsrVec::new();
    intersect(set_a, set_b, &mut writer);
    writer
}
