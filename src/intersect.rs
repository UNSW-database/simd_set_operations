mod merge;
pub use merge::NaiveMerge;

pub trait SortedIntersect2<T>
where
    T: Ord + Copy,
{
    fn intersect(set_a: &[T], set_b: &[T], result: &mut [T]) -> usize;
}

pub trait CustomIntersect2<T, In, Out>
where
    T: Ord,
    In: CustomSet<T>,
    Out: CustomSet<T>,
{
    fn intersect(set_a: &In, set_b: &In, result: &mut Out) -> usize;
}

pub trait SortedIntersectK<T> {
    fn intersect(sets: &[&[T]], result: &mut [T]) -> usize;
}

pub trait CustomIntersectK<T, In, Out>
where
    T: Ord,
    In: CustomSet<T>,
    Out: CustomSet<T>,
{
    fn intersect(sets: &In, result: &mut Out) -> usize;
}

pub trait CustomSet<T>
where
    T: Ord,
{
    fn from(set: &[T]) -> Self;
    fn preallocate(cardinality: usize) -> Self;
    fn cardinality(&self) -> usize;
}
