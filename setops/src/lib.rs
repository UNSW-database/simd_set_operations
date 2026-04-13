#![feature(portable_simd)]

pub mod bsr;
pub mod instructions;
pub mod intersect;
pub mod stats;
mod util;
pub mod visitor;

pub trait Set<T>
where
    T: Clone,
{
    fn from_sorted(sorted: &[T]) -> Self;
}
