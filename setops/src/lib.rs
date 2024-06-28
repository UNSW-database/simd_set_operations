#![feature(portable_simd)]
#![cfg_attr(target_os = "linux", feature(stdarch_x86_avx512))]

pub mod intersect;
pub mod visitor;
pub mod instructions;
pub mod bsr;
mod util;

pub trait Set<T>
where
    T: Clone
{
    fn from_sorted(sorted: &[T]) -> Self;
}
