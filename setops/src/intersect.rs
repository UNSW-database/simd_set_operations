mod merge;
mod galloping;
mod svs;
mod adaptive;
mod std_set;
mod shuffling;
mod broadcast;
mod lbk;
mod simd_galloping;
mod bmiss;
mod qfilter;
mod qfilter_c;
mod avx512;
pub mod fesia;

pub use {
    merge::*,
    galloping::{galloping, binary_search_intersect, galloping_inplace, galloping_bsr},
    adaptive::*,
    std_set::*,
    svs::*,
    bmiss::*,
};

#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub use {
    shuffling::*,
    broadcast::*,
    simd_galloping::*,
    qfilter::*,
    qfilter_c::qfilter_c,
    lbk::*,
};
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub use avx512::*;

use crate::{visitor::VecWriter, bsr::{BsrVec, BsrRef}};

pub type Intersect2<I, V> = fn(a: &I, b: &I, visitor: &mut V);
pub type Intersect2C<I> = fn(a: &I, b: &I, result: &mut I) -> usize;
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

pub fn run_2set_c<T>(
    set_a: &[T],
    set_b: &[T],
    intersect: Intersect2C<[T]>) -> Vec<T>
where
    T: Default + Clone + Copy
{
    let len = set_a.len().min(set_b.len());
    let mut result = vec![T::default();len];

    let result_len = intersect(set_a, set_b, &mut result);

    result.resize(result_len, T::default());
    result
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

pub fn run_2set_bsr<'a>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    intersect: fn(l: BsrRef<'a>, r: BsrRef<'a>, v: &mut BsrVec)) -> BsrVec
{
    let mut writer = BsrVec::new();
    intersect(set_a, set_b, &mut writer);
    writer
}
