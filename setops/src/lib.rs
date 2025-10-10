#![feature(portable_simd)]
#![cfg_attr(target_os = "linux", feature(stdarch_x86_avx512))]

pub mod bsr;
pub mod instructions;
pub mod intersect;
mod util;
pub mod visitor;

pub trait Set<T>
where
    T: Clone,
{
    fn from_sorted(sorted: &[T]) -> Self;
}
