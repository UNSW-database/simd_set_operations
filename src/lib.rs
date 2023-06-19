#![feature(portable_simd)]

pub mod intersect;
pub mod visitor;
pub mod instructions;
pub mod roaring;

pub trait Set<T>
where
    T: Clone
{
    fn from_sorted(sorted: &[T]) -> Self;
}
