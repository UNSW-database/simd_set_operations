#![cfg(all(feature = "simd", target_feature = "avx512f"))]
use std::simd::*;
use crate::{visitor::{Visitor, SimdVisitor16}, instructions::load_unsafe, util};

pub unsafe fn avx512_1x16<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_a = i32x16::splat(*set_a);
    let v_b: i32x16 = load_unsafe(set_b);
    let mask = v_a.simd_eq(v_b);
    if mask.any() {
        (*visitor).visit(*set_a);
    }
}

pub unsafe fn avx512_2x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
    ];
    let mask = masks[0] | masks[1];
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_3x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
    ];
    let mask = masks[0] | masks[1] | mask[2];
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_4x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
    ];
    let mask = util::or_4(masks);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_5x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
    ];
    let last = v_b.simd_eq(i32x16::splat(*set_a.add(4)));
    let mask = util::or_4(masks) | last;
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_6x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
    ];
    let mask = util::or_4(masks) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_7x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
    ];
    let mask = util::or_4(masks) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_8x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let mask = util::or_8(masks);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_9x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let last = v_b.simd_eq(i32x16::splat(*set_a.add(8)));
    let mask = util::or_8(masks) | last;
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_10x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
    ];
    let mask = util::or_8(masks) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_11x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
    ];
    let mask = util::or_8(masks) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_12x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
    ];
    let mask = util::or_8(masks) | util::or_4(rest);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_13x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
    ];
    let last = v_b.simd_eq(i32x16::splat(*set_a.add(12)));
    let mask = util::or_8(masks) | util::or_4(rest) | last;
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_14x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(12))),
        v_b.simd_eq(i32x16::splat(*set_a.add(13))),
    ];

    let mask = util::or_8(mask8) | util::or_4(mask4) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_15x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(12))),
        v_b.simd_eq(i32x16::splat(*set_a.add(13))),
        v_b.simd_eq(i32x16::splat(*set_a.add(14))),
    ];

    let mask = util::or_8(mask8) | util::or_4(mask4) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_16x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
        v_b.simd_eq(i32x16::splat(*set_a.add(12))),
        v_b.simd_eq(i32x16::splat(*set_a.add(13))),
        v_b.simd_eq(i32x16::splat(*set_a.add(14))),
        v_b.simd_eq(i32x16::splat(*set_a.add(15))),
    ];
    let mask = util::or_16(masks);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_17x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x16::splat(*set_a)),
        v_b.simd_eq(i32x16::splat(*set_a.add(1))),
        v_b.simd_eq(i32x16::splat(*set_a.add(2))),
        v_b.simd_eq(i32x16::splat(*set_a.add(3))),
        v_b.simd_eq(i32x16::splat(*set_a.add(4))),
        v_b.simd_eq(i32x16::splat(*set_a.add(5))),
        v_b.simd_eq(i32x16::splat(*set_a.add(6))),
        v_b.simd_eq(i32x16::splat(*set_a.add(7))),
        v_b.simd_eq(i32x16::splat(*set_a.add(8))),
        v_b.simd_eq(i32x16::splat(*set_a.add(9))),
        v_b.simd_eq(i32x16::splat(*set_a.add(10))),
        v_b.simd_eq(i32x16::splat(*set_a.add(11))),
        v_b.simd_eq(i32x16::splat(*set_a.add(12))),
        v_b.simd_eq(i32x16::splat(*set_a.add(13))),
        v_b.simd_eq(i32x16::splat(*set_a.add(14))),
        v_b.simd_eq(i32x16::splat(*set_a.add(15))),
    ];
    let last = v_b.simd_eq(i32x16::splat(*set_a.add(16)));
    let mask = util::or_16(masks) | last;
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}
