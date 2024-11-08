use std::cmp::Ordering;

//                                   //
// === REFERENCE IMPLEMENTATIONS === //
//                                   //

/// Basic linear intersection of two sorted arrays.
///
/// Zipper intersection algorithm derived from the 'zipper' or 'tape' sorted
/// array merging algorithm described in <https://doi.org/10.1137/0201004> and
/// <https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/>.
///
/// Conforms to [super::TwoSetAlgorithmFnGeneric] once `OUT` has been
/// specified, see there for more usage details.
///
/// # Generic Parameters
/// * `OUT` - Whether the function should output the intersection to `out`,
/// otherwise it will just calculate the size of the intersection.
///
pub fn zipper_ref
    <T: Ord + Copy, const OUT: bool>
    (sets: (&[T], &[T]), out: &mut [T])
-> usize {
    let mut idx_0 = 0;
    let mut idx_1 = 0;
    let mut count = 0;

    while idx_0 < sets.0.len() && idx_1 < sets.1.len() {
        let value_0 = unsafe { *sets.0.get_unchecked(idx_0) };
        let value_1 = unsafe { *sets.1.get_unchecked(idx_1) };

        match value_0.cmp(&value_1) {
            Ordering::Less => idx_0 += 1,
            Ordering::Greater => idx_1 += 1,
            Ordering::Equal => {
                if OUT {
                    unsafe { *out.get_unchecked_mut(count) = value_0; }
                }
                count += 1;
                idx_0 += 1;
                idx_1 += 1;
            }
        }
    }

    count
}

/// Zipper intersection rearranged for easier branch prediction and branchless
/// index updates. See [zipper] for usage details.
///
/// Proposed in <https://doi.org/10.14778/2735508.2735518> by Inoue, Ohara, and
/// Taura.
pub fn zipper_branch_optimized_ref
    <T: Ord + Copy, const OUT: bool>
    (sets: (&[T], &[T]), out: &mut [T])
-> usize {
    let mut idx_0 = 0;
    let mut idx_1 = 0;
    let mut count = 0;

    while idx_0 < sets.0.len() && idx_1 < sets.1.len() {
        let value_0 = unsafe { *sets.0.get_unchecked(idx_0) };
        let value_1 = unsafe { *sets.1.get_unchecked(idx_1) };

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

/// Zipper intersection with simplified loop condition. See [zipper] for usage
/// details.
///
/// Reduces the main loop condition from checking two indices against the set
/// lengths to only checking the index of the array with the lowest last value.
/// This works as the index that is incremented is the index to the lowest
/// value (or both indices if they index the same value), thus the array with
/// the lowest last value is guaranteed to always be the comparison that
/// terminates the loop.
pub fn zipper_loop_optimized_ref
    <T: Ord + Copy , const OUT: bool>
    (sets: (&[T], &[T]), out: &mut [T],)
-> usize {
    if sets.0.len() == 0 || sets.1.len() == 0 {
        return 0;
    }

    let (lo, hi) = if *sets.0.last().unwrap() <= *sets.1.last().unwrap() {
        (sets.0, sets.1)
    } else {
        (sets.1, sets.0)
    };

    let mut idx_lo = 0;
    let mut idx_hi = 0;
    let mut count = 0;

    while idx_lo < lo.len() {
        let vlo = unsafe { *lo.get_unchecked(idx_lo) };
        let vhi = unsafe { *hi.get_unchecked(idx_hi) };

        match vlo.cmp(&vhi) {
            Ordering::Less => idx_lo += 1,
            Ordering::Greater => idx_hi += 1,
            Ordering::Equal => {
                if OUT {
                    unsafe { *out.get_unchecked_mut(count) = vlo; }
                }
                count += 1;
                idx_lo += 1;
                idx_hi += 1;
            }
        }
    }

    count
}

//                                   //
// === OPTIMIZED IMPLEMENTATIONS === //
//                                   //

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
extern "sysv64" {
    fn zipper_u32_asm(
        set0    : *const u32,
        set1    : *const u32,
        set0_len: usize,
        set1_len: usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;

    fn zipper_i32_asm(
        set0    : *const i32,
        set1    : *const i32,
        set0_len: usize,
        set1_len: usize,
        out     : *mut i32,
        out_len : usize,
    ) -> usize;

    fn zipper_branch_optimized_u32_asm(
        set0    : *const u32,
        set1    : *const u32,
        set0_len: usize,
        set1_len: usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;

    fn zipper_branchless_u32_asm(
        set0    : *const u32,
        set1    : *const u32,
        set0_len: usize,
        set1_len: usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;

    fn zipper_bad_branch_u32_asm(
        set0    : *const u32,
        set1    : *const u32,
        set0_len: usize,
        set1_len: usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;

    fn zipper_loop_optimized_u32_asm(
        lo      : *const u32,
        hi      : *const u32,
        lo_len  : usize,
        hi_len  : usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;

    fn zipper_noindex_u32_asm(
        lo      : *const u32,
        hi      : *const u32,
        lo_len  : usize,
        hi_len  : usize,
        out     : *mut u32,
        out_len : usize,
    ) -> usize;
}

pub fn zipper_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    return unsafe {
        zipper_u32_asm(
            sets.0.as_ptr(),
            sets.1.as_ptr(),
            sets.0.len(),
            sets.1.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_i32(sets: (&[i32], &[i32]), out: &mut [i32]) -> usize {
    return unsafe {
        zipper_i32_asm(
            sets.0.as_ptr(),
            sets.1.as_ptr(),
            sets.0.len(),
            sets.1.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_branch_optimized_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    return unsafe {
        zipper_branch_optimized_u32_asm(
            sets.0.as_ptr(),
            sets.1.as_ptr(),
            sets.0.len(),
            sets.1.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_branchless_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    return unsafe {
        zipper_branchless_u32_asm(
            sets.0.as_ptr(),
            sets.1.as_ptr(),
            sets.0.len(),
            sets.1.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_loop_optimized_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    if sets.0.len() == 0 || sets.1.len() == 0 {
        return 0;
    }

    let (lo, hi) = if *sets.0.last().unwrap() <= *sets.1.last().unwrap() {
        (sets.0, sets.1)
    } else {
        (sets.1, sets.0)
    };

    return unsafe {
        zipper_loop_optimized_u32_asm(
            lo.as_ptr(),
            hi.as_ptr(),
            lo.len(),
            hi.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_noindex_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    if sets.0.len() == 0 || sets.1.len() == 0 {
        return 0;
    }

    let (lo, hi) = if *sets.0.last().unwrap() <= *sets.1.last().unwrap() {
        (sets.0, sets.1)
    } else {
        (sets.1, sets.0)
    };

    return unsafe {
        zipper_noindex_u32_asm(
            lo.as_ptr(),
            hi.as_ptr(),
            lo.len(),
            hi.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

pub fn zipper_bad_branch_u32(sets: (&[u32], &[u32]), out: &mut [u32]) -> usize {
    return unsafe {
        zipper_bad_branch_u32_asm(
            sets.0.as_ptr(),
            sets.1.as_ptr(),
            sets.0.len(),
            sets.1.len(),
            out.as_mut_ptr(),
            out.len(),
        )
    };
}

