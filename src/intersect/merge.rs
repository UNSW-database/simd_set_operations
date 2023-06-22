use std::cmp::Ordering;

use crate::visitor::Visitor;

/// Classical set intersection via merge. Original author unknown.
// Inspired by https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
pub fn naive_merge<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let value_a = set_a[idx_a];
        let value_b = set_b[idx_b];

        match value_a.cmp(&value_b) {
            Ordering::Less =>
                idx_a += 1,

            Ordering::Greater =>
                idx_b += 1,

            Ordering::Equal => {
                visitor.visit(value_a);
                idx_a += 1;
                idx_b += 1;
            },
        }
    }
}

/// Removes hard-to-predict 'less than' branch.
/// From [BMiss](http://www.vldb.org/pvldb/vol8/p293-inoue.pdf) paper.
// Faster Set Intersection with SIMD instructions by Reducing Branch Mispredictions
// H. Inoue, M. Ohara, K. Taura, 2014
pub fn branchless_merge<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let value_a = set_a[idx_a];
        let value_b = set_b[idx_b];

        if value_a == value_b {
            visitor.visit(value_a);
            idx_a += 1;
            idx_b += 1;
        } else {
            idx_a += (value_a < value_b) as usize;
            idx_b += (value_b < value_a) as usize;
        }
    }
}

pub const fn const_intersect<const LEN: usize>(set_a: &[i32], set_b: &[i32]) -> [i32; LEN] {
    let mut idx_a = 0;
    let mut idx_b = 0;
    let mut count = 0;

    let mut result = [0; LEN];

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let value_a = set_a[idx_a];
        let value_b = set_b[idx_b];

        if value_a == value_b {
            result[count] = value_a;
            count += 1;
            idx_a += 1;
            idx_b += 1;
        } else {
            idx_a += (value_a < value_b) as usize;
            idx_b += (value_b < value_a) as usize;
        }
    }

    assert!(count == result.len());
    result
}
