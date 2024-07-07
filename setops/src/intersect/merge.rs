use std::cmp::Ordering;

/// Basic linear intersection of two sorted arrays. 
/// 
/// Zipper intersection algorithm derived from the 'zipper' or 'tape' sorted array merging algorithm described in 
/// <https://doi.org/10.1137/0201004> and 
/// <https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/>. 
/// 
/// Conforms to [super::TwoSetAlgorithmFnGeneric] once `OUT` has been specified, see there for more usage details.
/// 
/// # Generic Parameters
/// * `OUT` - Whether the function should output the intersection to `out`, otherwise it will just calculate the size of
/// the intersection.
/// 
pub fn zipper<T: Ord + Copy, const OUT: bool>(sets: (&[T], &[T]), out: &mut [T]) -> usize {
    let mut idx_0 = 0;
    let mut idx_1 = 0;
    let mut count = 0;

    while idx_0 < sets.0.len() && idx_1 < sets.1.len() {
        let value_0 = sets.0[idx_0];
        let value_1 = sets.1[idx_1];

        match value_0.cmp(&value_1) {
            Ordering::Less => idx_0 += 1,

            Ordering::Greater => idx_1 += 1,

            Ordering::Equal => {
                if OUT {
                    unsafe {
                        *out.get_unchecked_mut(count) = value_0;
                    }
                }
                count += 1;
                idx_0 += 1;
                idx_1 += 1;
            }
        }
    }

    count
}

/// Zipper intersection rearranged for easier branch prediction and branchless index updates. See [zipper] for usage
/// details.
/// 
/// Proposed in <https://doi.org/10.14778/2735508.2735518> by Inoue, Ohara, and Taura.
pub fn zipper_branch_optimized<T: Ord + Copy, const OUT: bool>(sets: (&[T], &[T]), out: &mut [T],) -> usize {
    let mut idx_0 = 0;
    let mut idx_1 = 0;
    let mut count = 0;

    while idx_0 < sets.0.len() && idx_1 < sets.1.len() {
        let value_0 = sets.0[idx_0];
        let value_1 = sets.1[idx_1];

        if value_0 == value_1 {
            if OUT {
                unsafe {
                    *out.get_unchecked_mut(count) = value_0;
                }
            }
            count += 1;
            idx_0 += 1;
            idx_1 += 1;
        } else {
            idx_0 += (value_0 < value_1) as usize;
            idx_1 += (value_1 < value_0) as usize;
        }
    }

    count
}

/// Branch-optimized zipper intersection with simplified loop condition. See [zipper] for usage details.
/// 
/// Reduces the main loop condition from checking two indices against the set lengths to only checking the index of the
/// array with the lowest last value. This works as the index that is incremented is the index to the lowest 
/// value (or both indices if they index the same value), thus the array with the lowest last value is guaranteed to 
/// always be the comparison that terminates the loop.
pub fn zipper_branch_loop_optimized<T: Ord + Copy, const OUT: bool>(sets: (&[T], &[T]), out: &mut [T],) -> usize {
    let (lo, hi) = if sets.0.last().unwrap() <= sets.1.last().unwrap() {
        (sets.0, sets.1)
    } else {
        (sets.1, sets.0)
    };

    let mut idx_lo = 0;
    let mut idx_hi = 0;
    let mut count = 0;

    while idx_lo < lo.len() {
        let vlo = lo[idx_lo];
        let vhi = hi[idx_hi];

        if vlo == vhi {
            if OUT {
                unsafe {
                    *out.get_unchecked_mut(count) = vlo;
                }
            }
            count += 1;
            idx_lo += 1;
            idx_hi += 1;
        } else {
            idx_lo += (vlo < vhi) as usize;
            idx_hi += (vhi < vlo) as usize;
        }
    }

    count
}
