pub mod intersect;
pub mod visitor;

pub trait CustomSet<T> {
    fn from_sorted(sorted: &[T]) -> Self;
}
