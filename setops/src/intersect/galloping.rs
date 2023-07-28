use std::cmp::Ordering;

/// Search-based set intersection algorithms.

use crate::{visitor::{Visitor, BsrVisitor}, bsr::BsrRef};

pub fn galloping<T, V>(small: &[T], large: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    if small.is_empty() || large.is_empty() {
        return;
    }

    let mut base = 0;

    for &target in small {

        let mut offset = 1;

        while base + offset < large.len() &&
            large[base + offset] <= target
        {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min((base + offset) as isize);

        base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            visitor.visit(target);
        }
    }
}

pub fn galloping_bsr<'a, S, V>(small_bsr: S, large_bsr: S, visitor: &mut V)
where
    S: Into<BsrRef<'a>>,
    V: BsrVisitor,
{
    let small = small_bsr.into();
    let large = large_bsr.into();

    if small.is_empty() || large.is_empty() {
        return;
    }

    let mut large_idx = 0;

    for (&small_base, &small_state) in small {

        let mut offset = 1;

        while (large_idx + offset) < large.len() &&
            large.bases[large_idx + offset] <= small_base
        {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min((large_idx + offset) as isize);

        large_idx = binary_search(large.bases, small_base, lo, hi);

        if large_idx < large.len() && large.bases[large_idx] == small_base {
            let new_state = small_state & large.states[large_idx];
            if new_state != 0 {
                visitor.visit_bsr(small_base, new_state);
            }
        }
    }
}

pub fn galloping_inplace<T>(small: &mut [T], large: &[T]) -> usize
where
    T: Ord + Copy,
{
    let mut base = 0;
    let mut count = 0;

    for i in 0..small.len() {

        let target = unsafe { *small.get_unchecked(i) };
        let mut offset = 1;

        while base + offset < large.len() &&
            large[base + offset] <= target
        {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min((base + offset) as isize);

        base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            small[count] = target;
            count += 1;
        }
    }

    count
}

pub fn binary_search<T>(
    set: &[T],
    target: T,
    mut lo: isize,
    mut hi: isize) -> usize
where
    T: Ord + Copy,
{
    while lo <= hi {

        let mid = lo + (hi - lo) / 2;
        let actual = set[mid as usize];

        match actual.cmp(&target) {
            Ordering::Less    => lo = mid + 1,
            Ordering::Greater => hi = mid - 1,
            Ordering::Equal   => return mid as usize,
        }
    }

    lo as usize
}
