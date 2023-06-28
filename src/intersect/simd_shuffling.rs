#![cfg(feature = "simd")]

use std::{
    cmp::Ordering,
    simd::*,
};

use crate::{
    visitor::{SimdVisitor4, SimdVisitor8, SimdVisitor16},
    intersect, instructions::load_unsafe,
};

use crate::{visitor::SimdBsrVisitor4, bsr::BsrRef};

/// SIMD Shuffling set intersection algorithm - Ilya Katsov 2012
/// https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
/// Implementation modified from roaring-rs
#[cfg(target_feature = "ssse3")]
pub fn simd_shuffling<V>(set_a: &[i32], set_b: &[i32], visitor: &mut V)
where
    V: SimdVisitor4<i32>,
{
    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x4 = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
        let mut v_b: i32x4 = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
        loop {
            let masks = [
                v_a.simd_eq(v_b),
                v_a.simd_eq(v_b.rotate_lanes_left::<1>()),
                v_a.simd_eq(v_b.rotate_lanes_left::<2>()),
                v_a.simd_eq(v_b.rotate_lanes_left::<3>()),
            ];
            let mask = (masks[0] | masks[1]) | (masks[2] | masks[3]);

            visitor.visit_vector4(v_a, mask.to_bitmask());

            let a_max = set_a[i_a + W - 1];
            let b_max = set_b[i_b + W - 1];
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
                    v_b = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
                },
            }
        }
    }

    intersect::branchless_merge(&set_a[i_a..], &set_b[i_b..], visitor)
}

#[cfg(target_feature = "avx2")]
pub fn simd_shuffling_avx2<V>(set_a: &[i32], set_b: &[i32], visitor: &mut V)
where
    V: SimdVisitor8<i32>,
{
    const W: usize = 8;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x8 = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
        let mut v_b: i32x8 = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
        loop {
            let layer1 = [
                 v_a.simd_eq(v_b) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<1>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<2>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<3>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<4>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<5>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<6>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<7>()),
            ];
            let layer2 = [
                layer1[0] | layer1[1],
                layer1[2] | layer1[3],
                layer1[4] | layer1[5],
                layer1[6] | layer1[7],
            ];
            let mask = (layer2[0] | layer2[1]) | (layer2[2] | layer2[3]);

            visitor.visit_vector8(v_a, mask.to_bitmask());

            let a_max = set_a[i_a + W - 1];
            let b_max = set_b[i_b + W - 1];
            if a_max <= b_max {
                i_a += W;
                if i_a == st_a {
                    break;
                }
                v_a = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
            }
            if b_max <= a_max {
                i_b += W;
                if i_b == st_b {
                    break;
                }
                v_b = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
            }
        }
    }

    intersect::branchless_merge(&set_a[i_a..], &set_b[i_b..], visitor)
}

#[cfg(target_feature = "ssse3")]
pub fn simd_shuffling_bsr<'a, S, V>(
    a: S,
    b: S,
    visitor: &mut V)
where
    S: Into<BsrRef<'a>>,
    V: SimdBsrVisitor4,
{
    let set_a = a.into();
    let set_b = b.into();

    const W: usize = 4;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut base_a: i32x4 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let mut base_b: i32x4 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let mut state_a: i32x4 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let mut state_b: i32x4 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        loop {
            let base_masks = [
                base_a.simd_eq(base_b),
                base_a.simd_eq(base_b.rotate_lanes_left::<1>()),
                base_a.simd_eq(base_b.rotate_lanes_left::<2>()),
                base_a.simd_eq(base_b.rotate_lanes_left::<3>()),
            ];
            let state_masks = [
                base_masks[0].to_int() & (state_a & state_b),
                base_masks[1].to_int() & (state_a & state_b.rotate_lanes_left::<1>()),
                base_masks[2].to_int() & (state_a & state_b.rotate_lanes_left::<2>()),
                base_masks[3].to_int() & (state_a & state_b.rotate_lanes_left::<3>()),
            ];

            let base_mask = (base_masks[0] | base_masks[1]) | (base_masks[2] | base_masks[3]);
            let state_all = (state_masks[0] | state_masks[1]) | (state_masks[2] | state_masks[3]);
            let state_mask = state_all.simd_ne(i32x4::from_array([0; 4]));

            let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

            visitor.visit_bsr_vector4(base_a, state_all, total_mask);

            let a_max = set_a.bases[i_a + W - 1];
            let b_max = set_b.bases[i_b + W - 1];
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    base_a = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
                    base_b = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
                    state_a = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
                    state_b = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    base_a = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
                    state_a = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    base_b = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
                    state_b = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
                },
            }
        }
    }

    intersect::merge_bsr(set_a.advanced_by(i_a), set_b.advanced_by(i_b), visitor)
}

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn simd_shuffling_avx512_naive<V>(set_a: &[i32], set_b: &[i32], visitor: &mut V)
where
    V: SimdVisitor16<i32>,
{
    const W: usize = 16;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x16 = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
        let mut v_b: i32x16 = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
        loop {
            let layer1 = [
                 v_a.simd_eq(v_b) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<1>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<2>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<3>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<4>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<5>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<6>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<7>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<8>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<9>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<10>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<11>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<12>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<13>()),
                 v_a.simd_eq(v_b.rotate_lanes_left::<14>()) |
                 v_a.simd_eq(v_b.rotate_lanes_left::<15>()),
            ];
            let layer2 = [
                layer1[0] | layer1[1],
                layer1[2] | layer1[3],
                layer1[4] | layer1[5],
                layer1[6] | layer1[7],
                layer1[8] | layer1[9],
                layer1[10] | layer1[11],
                layer1[12] | layer1[13],
                layer1[14] | layer1[15],
            ];
            let layer3 = [
                layer2[0] | layer2[1],
                layer2[2] | layer2[3],
                layer2[4] | layer2[5],
                layer2[6] | layer2[7],
            ];
            let mask = (layer3[0] | layer3[1]) | (layer3[2] | layer3[3]);

            //visitor.visit_vector(v_a, mask.to_bitmask());

            let a_max = set_a[i_a + W - 1];
            let b_max = set_b[i_b + W - 1];
            if a_max <= b_max {
                i_a += W;
                if i_a == st_a {
                    break;
                }
                v_a = unsafe{ load_unsafe(set_a.as_ptr().add(i_a)) };
            }
            if b_max <= a_max {
                i_b += W;
                if i_b == st_b {
                    break;
                }
                v_b = unsafe{ load_unsafe(set_b.as_ptr().add(i_b)) };
            }
        }
    }

    intersect::branchless_merge(&set_a[i_a..], &set_b[i_b..], visitor)
}