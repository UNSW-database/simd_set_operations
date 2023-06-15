pub mod intersect;
pub mod visitor;
mod simd;

pub trait Set<T>
where
    T: Clone
{
    fn from_sorted(sorted: &[T]) -> Self;
}
