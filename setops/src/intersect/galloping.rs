use std::cmp::Ordering;

/// Search-based set intersection algorithms.

use crate::{visitor::{Visitor, BsrVisitor}, bsr::BsrRef};

pub fn galloping<T, V>(small: &[T], mut large: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    for &target in small {

        let mut offset = 1;

        while offset < large.len() && large[offset] <= target {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min(offset as isize);

        let base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            visitor.visit(target);
        }
        large = &large[base..];
    }
}

pub fn binary_search_intersect<T, V>(small: &[T], mut large: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    for &target in small {

        let lo: isize = 0;
        let hi: isize = large.len() as isize - 1;

        let base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            visitor.visit(target);
        }
        large = &large[base..];
    }
}

pub fn galloping_bsr<'a, V>(small: BsrRef<'a>, mut large: BsrRef<'a>, visitor: &mut V)
where
    V: BsrVisitor,
{
    for (&small_base, &small_state) in small {

        let mut offset = 1;

        while offset < large.len() && large.bases[offset] <= small_base {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min(offset as isize);

        let large_idx = binary_search(large.bases, small_base, lo, hi);

        if large_idx < large.len() && large.bases[large_idx] == small_base {
            let new_state = small_state & large.states[large_idx];
            if new_state != 0 {
                visitor.visit_bsr(small_base, new_state);
            }
        }
        large = large.advanced_by(large_idx);
    }
}

pub fn galloping_inplace<T>(small: &mut [T], mut large: &[T]) -> usize
where
    T: Ord + Copy,
{
    let mut count = 0;

    for i in 0..small.len() {

        let target = unsafe { *small.get_unchecked(i) };
        let mut offset = 1;

        while offset < large.len() &&
            large[offset] <= target
        {
            offset *= 2;
        }

        let lo: isize = (offset / 2) as isize;
        let hi: isize = (large.len() as isize - 1).min(offset as isize);

        let base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            small[count] = target;
            count += 1;
        }
        large = &large[base..];
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
