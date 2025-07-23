#![allow(warnings)]
#![feature(portable_simd)]
#![feature(core_intrinsics)]
#![feature(stmt_expr_attributes)]
#![cfg_attr(all(target_os = "linux", target_arch = "x86_64"), feature(stdarch_x86_avx512))]
#![feature(generic_const_exprs)]


pub mod intersect;
pub mod visitor;
pub mod instructions;
pub mod bsr;
mod util;
pub mod KSetInput;

pub trait Set<T>
where
    T: Clone
{
    fn from_sorted(sorted: &[T]) -> Self;
}
