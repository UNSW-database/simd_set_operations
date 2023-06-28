#![feature(portable_simd)]
#![feature(stdsimd)]

pub mod intersect;
pub mod visitor;
pub mod instructions;
pub mod roaring;
pub mod bsr;

pub trait Set<T>
where
    T: Clone
{
    fn from_sorted(sorted: &[T]) -> Self;
}
