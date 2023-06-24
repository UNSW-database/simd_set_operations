use std::cmp::Ordering;

use crate::{visitor::{Visitor, BsrVisitor}, bsr::BsrRef};

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


pub fn merge_bsr<'a, S, V>(set_a: S, set_b: S, visitor: &mut V)
where
    S: Into<BsrRef<'a>>,
    V: BsrVisitor,
{
    let s_a = set_a.into();
    let s_b = set_b.into();

    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < s_a.len() && idx_b < s_b.len() {
        let base_a = s_a.base[idx_a];
        let base_b = s_b.base[idx_b];
        let state_a = s_a.state[idx_a];
        let state_b = s_b.state[idx_b];

        if base_a == base_b {
            let new_state = state_a & state_b;
            if new_state != 0 {
                visitor.visit_bsr(base_a, new_state);
            }
            idx_a += 1;
            idx_b += 1;
        } else {
            idx_a += (base_a < base_b) as usize;
            idx_b += (base_b < base_a) as usize;
        }
    }
}

pub const fn const_intersect<const LEN: usize>(
    set_a: &[i32],
    set_b: &[i32]) -> [i32; LEN]
{
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
