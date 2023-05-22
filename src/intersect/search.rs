/// Search-based set intersection algorithms.
/// 
use crate::visitor::Visitor;

pub fn galloping_intersect<T, V>(small_set: &[T], large_set: &[T], visitor: &mut V) -> usize
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mut base = 0;
    let mut count = 0;

    for target in small_set.iter().cloned() {

        let mut offset = 1;

        while base + offset < large_set.len() && large_set[base] > target {
            offset *= 2;
        }

        let lo = base;
        let hi = large_set.len().min(base + offset);
        
        base = binary_search(large_set, target, lo, hi, visitor, &mut count);
    }

    count
}

fn binary_search<T, V>(
    set: &[T],
    target: T,
    mut lo: usize,
    mut hi: usize,
    visitor: &mut V,
    count: &mut usize) -> usize
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    while lo <= hi {
        let mid = (hi - lo) / 2;
        let actual = set[mid];

        if actual < target {
            lo = mid + 1;
        }
        else if actual > target {
            hi = mid - 1;
        }
        else {
            visitor.visit(actual);
            *count += 1;
            return mid;
        }
    }

    lo
}

pub fn galloping_intersect_inplace<T>(small_set: &mut [T], large_set: &[T]) -> usize
where
    T: Ord + Copy,
{
    let mut base = 0;
    let mut count = 0;

    for i in 0..small_set.len() {

        let target = unsafe { *small_set.get_unchecked(i) };
        let mut offset = 1;

        while base + offset < large_set.len() && large_set[base] > target {
            offset *= 2;
        }

        let lo = base;
        let hi = large_set.len().min(base + offset);
        
        base = binary_search_inplace(large_set, target, lo, hi, &mut small_set[count], &mut count);
    }

    count
}

fn binary_search_inplace<T>(
    set: &[T],
    target: T,
    mut lo: usize,
    mut hi: usize,
    out: &mut T,
    count: &mut usize) -> usize
where
    T: Ord + Copy,
{
    while lo <= hi {
        let mid = (hi - lo) / 2;
        let actual = set[mid];

        if actual < target {
            lo = mid + 1;
        }
        else if actual > target {
            hi = mid - 1;
        }
        else {
            *out = actual;
            *count += 1;
            return mid;
        }
    }

    lo
}
