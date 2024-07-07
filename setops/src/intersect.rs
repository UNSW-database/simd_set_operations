pub mod merge;
pub mod svs;
/*mod galloping;
mod adaptive;
mod std_set;
mod shuffling;
mod broadcast;
mod simd_galloping;
mod bmiss;
mod qfilter;
mod avx512;
pub mod fesia;
*/

/// 2-set intersection algorithms that are generic over type `T: Ord + Copy`.
/// 
/// # Generic Paramaters
/// * `T` - The type of the values in the sets.
/// 
/// # Parameters
/// * `sets` - A tuple of the two sets to be intersected.
/// * `out` - The slice to which the intersection will be written.
/// * Returns the size of the intersection.
/// 
/// # Preconditions
/// * `out` is large enough to hold the intersection of the two sets.
/// 
/// # Postconditions
/// * `out` contains the intersection of the given sets.
/// * The return value is the size of the intersection of the given sets.
/// 
/// # Used By
/// * [merge::zipper]
/// * [merge::zipper_branch_optimized]
/// * [merge::zipper_branch_loop_optimized]
/// 
/// # Example
/// ```
/// use setops::intersect::merge::zipper;
/// 
/// let a = vec![1, 2, 3];
/// let b = vec![2, 3, 4];
/// let mut out = vec![0; 3];
/// 
/// let size = zipper::<i32, false>((&a, &b), &mut out);
/// ```
/// 
pub type TwoSetAlgorithmFnGeneric<T> = fn(sets: (&[T], &[T]), out: &mut [T]) -> usize;

/// K-set intersection algorithms that are generic over type `T: Ord + Copy`.
/// 
/// # Generic Parameters
/// * `T` - The type of the values in the sets.
/// 
/// # Parameters
/// * `sets` - A slice of slices of values to be intersected.
/// * `out` - The slice to which the intermediate and the final output intersection will be written.
/// * Returns the size of the output intersection.
/// 
/// # Preconditions
/// * `sets` is ordered from shortest to longest slice and contains more than 2 sets, and the sets are sorted in
/// ascending order.
/// * `out` is large enough to hold the intersection of the smallest two sets.
///
/// # Postconditions
/// * `out` contains the intersection of all of the given sets.
/// * The return value is the size of said intersection.
/// 
pub type KSetAlgorithmFnGeneric<T> = fn(sets: &[&[T]], out: &mut [T]) -> usize;

/// K-set intersection algorithms that are generic over type `T: Ord + Copy` and require an extra buffer for
/// intermediate calculations.
/// 
/// Conforms to [KSetAlgorithmFnGeneric] once `buf` has been bound, see there for more usage details.
/// 
/// # Paramaters
/// * `buf` - A slice to which intermediate intersections will be written.
/// 
/// # Precondition
/// * `buf` is large enough to hold the intersection of the smallest two sets.
/// 
pub type KSetAlgorithmBufFnGeneric<T> = fn(sets: &[&[T]], out: &mut [T], buf: &mut [T]) -> usize;

/// Algorithm that adapts a 2-set intersection algorithm into a k-set intersection algorithm over generic type 
/// `T: Ord + Copy`, requiring an extra buffer for intermediate calculations.
/// 
/// Conforms to [KSetAlgorithmBufFnGeneric] when composed with [TwoSetAlgorithmFnGeneric]. See 
/// [KSetAlgorithmBufFnGeneric] for further usage information.
/// 
/// # Paramaters
/// * `twoset_fn` - A 2-set intersection function that conforms to the standard interface used in this crate.
/// 
/// # Preconditions
/// * `twoset_fn` calculates the intersection of two sets given to it.
/// 
/// # Used By
/// * [svs::svs]
/// 
/// # Example
/// ```
/// use setops::intersect::{merge::zipper, svs::svs};
/// 
/// let a = vec![1, 2, 3];
/// let b = vec![2, 3, 4];
/// let c = vec![3, 4, 5];
/// let sets = vec![a.as_slice(), b.as_slice(), c.as_slice()];
/// 
/// let mut out = vec![0; 3];
/// let mut buf = vec![0; 3];
/// 
/// // Conforms to KSetAlgorithmBufFnGeneric<i32>
/// let intersect_fn = |sets: &[&[i32]], out: &mut [i32], buf: &mut [i32]| svs(zipper::<i32, true>, sets, out, buf);
/// 
/// let size = intersect_fn(&sets, &mut out, &mut buf);
/// ```
/// 
pub type TwoSetToKSetBufFnGeneric<T> = fn(
    twoset_fn: TwoSetAlgorithmFnGeneric<T>,
    sets: &[&[T]],
    out: &mut [T],
    buf: &mut [T],
) -> usize;
