/// Search-based set intersection algorithms.

use crate::visitor::Visitor;

pub fn galloping<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    if small_set.len() == 0 || large_set.len() == 0 {
        return;
    }

    let mut base = 0;

    for target in small_set.iter().cloned() {

        let mut offset = 1;

        while base + offset < large_set.len() &&
            large_set[base + offset] <= target
        {
            offset *= 2;
        }

        let lo = base;
        let hi = (large_set.len() - 1).min(base + offset);

        base = binary_search(large_set, target, lo, hi);

        if base < large_set.len() && large_set[base] == target {
            visitor.visit(target);
        }
    }
}

pub fn galloping_inplace<T>(small_set: &mut [T], large_set: &[T]) -> usize
where
    T: Ord + Copy,
{
    let mut base = 0;
    let mut count = 0;

    for i in 0..small_set.len() {

        let target = unsafe { *small_set.get_unchecked(i) };
        let mut offset = 1;

        while base + offset < large_set.len() &&
            large_set[base + offset] <= target
        {
            offset *= 2;
        }

        let lo = base;
        let hi = (large_set.len() - 1).min(base + offset);

        base = binary_search(large_set, target, lo, hi);

        if base < large_set.len() && large_set[base] == target {
            small_set[count] = target;
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

        if actual < target {
            lower = mid + 1;
        }
        else if actual > target {
            upper = mid - 1;
        }
        else {
            return mid as usize;
        }
    }

    lower as usize
}
