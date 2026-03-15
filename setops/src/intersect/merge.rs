use std::cmp::Ordering;

use crate::{
    bsr::BsrRef,
    stats,
    visitor::{BsrVisitor, Visitor},
};

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
        stats::record_stage3_scalar_kernel(1);

        match value_a.cmp(&value_b) {
            Ordering::Less => {
                stats::record_stage1_linear_step(true, false);
                idx_a += 1;
            }

            Ordering::Greater => {
                stats::record_stage1_linear_step(false, true);
                idx_b += 1;
            }

            Ordering::Equal => {
                stats::record_stage1_linear_step(true, true);
                visitor.visit(value_a);
                idx_a += 1;
                idx_b += 1;
            }
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
        stats::record_stage3_scalar_kernel(1);

        if value_a == value_b {
            stats::record_stage1_linear_step(true, true);
            visitor.visit(value_a);
            idx_a += 1;
            idx_b += 1;
        } else {
            let advance_a = value_a < value_b;
            let advance_b = value_b < value_a;
            stats::record_stage1_linear_step(advance_a, advance_b);
            idx_a += advance_a as usize;
            idx_b += advance_b as usize;
        }
    }
}

pub fn branchless_merge_bsr<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: BsrVisitor,
{
    let mut idx_a = 0;
    let mut idx_b = 0;

    while idx_a < set_a.len() && idx_b < set_b.len() {
        let base_a = set_a.bases[idx_a];
        let base_b = set_b.bases[idx_b];
        let state_a = set_a.states[idx_a];
        let state_b = set_b.states[idx_b];
        stats::record_stage3_scalar_kernel(1);

        if base_a == base_b {
            let new_state = state_a & state_b;
            stats::record_stage1_linear_step(true, true);
            if new_state != 0 {
                visitor.visit_bsr(base_a, new_state);
            }
            idx_a += 1;
            idx_b += 1;
        } else {
            let advance_a = base_a < base_b;
            let advance_b = base_b < base_a;
            stats::record_stage1_linear_step(advance_a, advance_b);
            idx_a += advance_a as usize;
            idx_b += advance_b as usize;
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
