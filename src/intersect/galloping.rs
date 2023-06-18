use std::cmp::Ordering;

/// Search-based set intersection algorithms.

use crate::visitor::Visitor;

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

        let lo = offset / 2;
        let hi = (large.len() - 1).min(base + offset);

        base = binary_search(large, target, lo, hi);

        if base < large.len() && large[base] == target {
            visitor.visit(target);
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

        let lo = offset / 2;
        let hi = (large.len() - 1).min(base + offset);

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
    lo: usize,
    hi: usize) -> usize
where
    T: Ord + Copy,
{
    let mut lower = lo as isize;
    let mut upper = hi as isize;

    while lower <= upper {

        let mid = lower + (upper - lower) / 2;
        let actual = set[mid as usize];

        match actual.cmp(&target) {
            Ordering::Less    => lower = mid + 1,
            Ordering::Greater => upper = mid - 1,
            Ordering::Equal   => return mid as usize,
        }
    }

    lower as usize
}
