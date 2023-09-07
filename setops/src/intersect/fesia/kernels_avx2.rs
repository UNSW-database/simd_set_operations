#![cfg(feature = "simd")]
use std::simd::*;
use crate::{visitor::{Visitor, SimdVisitor8}, instructions::load_unsafe, util};

pub unsafe fn avx2_1x8<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_a = i32x8::splat(*set_a);
    let v_b: i32x8 = load_unsafe(set_b);
    let mask = v_a.simd_eq(v_b);
    if mask.any() {
        (*visitor).visit(*set_a);
    }
}

pub unsafe fn avx2_2x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
    ];
    let mask = masks[0] | masks[1];
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_3x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
    ];
    let mask = masks[0] | masks[1] | masks[2];
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_4x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
    ];
    let mask = util::or_4(masks);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_5x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks_1_to_4 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
    ];
    let mask5 = v_b.simd_eq(i32x8::splat(*set_a.add(4)));

    let mask = util::or_4(masks_1_to_4) | mask5;
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_6x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks_1_to_4 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
    ];
    let mask5 = v_b.simd_eq(i32x8::splat(*set_a.add(4)));
    let mask6 = v_b.simd_eq(i32x8::splat(*set_a.add(5)));

    let mask = util::or_4(masks_1_to_4) | (mask5 | mask6);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_7x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks_1_to_4 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
    ];

    let mask = util::or_4(masks_1_to_4) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

pub unsafe fn avx2_8x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];

    let mask = util::or_8(masks);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_9x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let last = v_b.simd_eq(i32x8::splat(*set_a.add(8)));

    let mask = util::or_8(masks) | last;
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_10x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
    ];

    let mask = util::or_8(masks8) | (rest[0] | rest[1]);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_11x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
        v_b.simd_eq(i32x8::splat(*set_a.add(10))),
    ];

    let mask = util::or_8(masks8) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_12x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
        v_b.simd_eq(i32x8::splat(*set_a.add(10))),
        v_b.simd_eq(i32x8::splat(*set_a.add(11))),
    ];

    let mask = util::or_8(masks8) | util::or_4(rest);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_13x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let masks4 = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
        v_b.simd_eq(i32x8::splat(*set_a.add(10))),
        v_b.simd_eq(i32x8::splat(*set_a.add(11))),
    ];
    let last = v_b.simd_eq(i32x8::splat(*set_a.add(12)));

    let mask = util::or_8(masks8) | util::or_4(masks4) | last;
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_14x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let masks4 = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
        v_b.simd_eq(i32x8::splat(*set_a.add(10))),
        v_b.simd_eq(i32x8::splat(*set_a.add(11))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(12))),
        v_b.simd_eq(i32x8::splat(*set_a.add(13))),
    ];

    let mask = util::or_8(masks8) | util::or_4(masks4) | (rest[0] | rest[1]);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

unsafe fn avx2_15x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x8 = load_unsafe(set_b);
    let masks8 = [
        v_b.simd_eq(i32x8::splat(*set_a)),
        v_b.simd_eq(i32x8::splat(*set_a.add(1))),
        v_b.simd_eq(i32x8::splat(*set_a.add(2))),
        v_b.simd_eq(i32x8::splat(*set_a.add(3))),
        v_b.simd_eq(i32x8::splat(*set_a.add(4))),
        v_b.simd_eq(i32x8::splat(*set_a.add(5))),
        v_b.simd_eq(i32x8::splat(*set_a.add(6))),
        v_b.simd_eq(i32x8::splat(*set_a.add(7))),
    ];
    let masks4 = [
        v_b.simd_eq(i32x8::splat(*set_a.add(8))),
        v_b.simd_eq(i32x8::splat(*set_a.add(9))),
        v_b.simd_eq(i32x8::splat(*set_a.add(10))),
        v_b.simd_eq(i32x8::splat(*set_a.add(11))),
    ];
    let rest = [
        v_b.simd_eq(i32x8::splat(*set_a.add(12))),
        v_b.simd_eq(i32x8::splat(*set_a.add(13))),
        v_b.simd_eq(i32x8::splat(*set_a.add(14))),
    ];

    let mask = util::or_8(masks8) | util::or_4(masks4) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector8(v_b, mask.to_bitmask());
}

// unsafe fn avx2_16x8<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
//     let v_b: i32x8 = load_unsafe(set_b);
//     let masks = [
//         v_b.simd_eq(i32x8::splat(*set_a)),
//         v_b.simd_eq(i32x8::splat(*set_a.add(1))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(2))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(3))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(4))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(5))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(6))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(7))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(8))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(9))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(10))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(11))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(12))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(13))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(14))),
//         v_b.simd_eq(i32x8::splat(*set_a.add(15))),
//     ];

//     let mask = util::or_16(masks);
//     (*visitor).visit_vector8(v_b, mask.to_bitmask());
// }

pub unsafe fn avx2_1x16<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_a = i32x8::splat(*set_a);
    let v_b0: i32x8 = load_unsafe(set_b);
    let v_b1: i32x8 = load_unsafe(set_b.add(8));
    let mask = v_a.simd_eq(v_b0) | v_a.simd_eq(v_b1);
    if mask.any() {
        (*visitor).visit(*set_a);
    }
}

pub unsafe fn avx2_2x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_2x8(set_a, set_b, visitor);
    avx2_2x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_3x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_3x8(set_a, set_b, visitor);
    avx2_3x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_4x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_4x8(set_a, set_b, visitor);
    avx2_4x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_5x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_5x8(set_a, set_b, visitor);
    avx2_5x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_6x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_6x8(set_a, set_b, visitor);
    avx2_6x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_7x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_7x8(set_a, set_b, visitor);
    avx2_7x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_8x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_8x8(set_a, set_b, visitor);
    avx2_8x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_9x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_9x8(set_a, set_b, visitor);
    avx2_9x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_10x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_10x8(set_a, set_b, visitor);
    avx2_10x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_11x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_11x8(set_a, set_b, visitor);
    avx2_11x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_12x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_12x8(set_a, set_b, visitor);
    avx2_12x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_13x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_13x8(set_a, set_b, visitor);
    avx2_13x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_14x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_14x8(set_a, set_b, visitor);
    avx2_14x8(set_a, set_b.add(8), visitor);
}

pub unsafe fn avx2_15x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx2_15x8(set_a, set_b, visitor);
    avx2_15x8(set_a, set_b.add(8), visitor);
}

// pub unsafe fn avx2_16x16<V: SimdVisitor8<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
//     avx2_16x8(set_a, set_b, visitor);
//     avx2_16x8(set_a, set_b.add(8), visitor);
// }
