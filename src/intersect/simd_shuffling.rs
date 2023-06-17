#![cfg(feature = "simd")]

use std::simd::*;

use crate::{
    visitor::{SimdVisitor, VecWriter},
    intersect, instructions::load,
};

/// SIMD Shuffling set intersection algorithm - Ilya Katsov 2012
/// https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
/// Implementation inspired by roaring-rs
#[inline(never)]
pub fn simd_shuffling<V>(set_a: &[i32], set_b: &[i32], visitor: &mut V)
where
    V: SimdVisitor<i32, 4>,
{
    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x4 = load(&set_a[i_a..]);
        let mut v_b: i32x4 = load(&set_b[i_b..]);
        loop {
            let mask = v_a.simd_eq(v_b)
                | v_a.simd_eq(v_b.rotate_lanes_left::<1>())
                | v_a.simd_eq(v_b.rotate_lanes_left::<2>())
                | v_a.simd_eq(v_b.rotate_lanes_left::<3>());

            visitor.visit_vector(v_a, mask.to_bitmask());

            let a_max = set_a[i_a + W - 1];
            let b_max = set_b[i_b + W - 1];
            if a_max <= b_max {
                i_a += W;
                if i_a == st_a {
                    break;
                }
                v_a = load(&set_a[i_a..]);
            }
            if b_max <= a_max {
                i_b += W;
                if i_b == st_b {
                    break;
                }
                v_b = load(&set_b[i_b..]);
            }
        }
    }

    intersect::branchless_merge(&set_a[i_a..], &set_b[i_b..], visitor)
}

pub fn simd_shuffling_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>) {
    simd_shuffling(set_a, set_b, visitor);
}
