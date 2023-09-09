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
    let mask = masks[0] | masks[1] | masks[2];
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

unsafe fn avx512_17x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
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

unsafe fn avx512_18x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
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
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
    ];
    let mask = util::or_16(masks) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_19x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
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
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
    ];
    let mask = util::or_16(masks) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_20x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
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
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
    ];
    let mask = util::or_16(masks) | util::or_4(rest);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_21x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
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
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
    ];
    let last = v_b.simd_eq(i32x16::splat(*set_a.add(20)));
    let mask = util::or_16(masks) | util::or_4(rest) | last;
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_22x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
    ];
    let mask = util::or_16(mask16) | util::or_4(mask4) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_23x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
    ];
    let mask = util::or_16(mask16) | util::or_4(mask4) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_24x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let mask = util::or_16(mask16) | util::or_8(mask8);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_25x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
    ];
    let mask = util::or_16(mask16) | util::or_8(mask8) | rest[0];
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_26x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
    ];
    let mask = util::or_16(mask16) | util::or_8(mask8) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_27x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
    ];
    let mask = util::or_16(mask16) | util::or_8(mask8) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_28x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
        v_b.simd_eq(i32x16::splat(*set_a.add(27))),
    ];
    let mask = util::or_16(mask16) | util::or_8(mask8) | util::or_4(mask4);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_29x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
        v_b.simd_eq(i32x16::splat(*set_a.add(27))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(28))),
    ];
    let mask =
        util::or_16(mask16) | util::or_8(mask8) |
        util::or_4(mask4) | rest[0];
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_30x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
        v_b.simd_eq(i32x16::splat(*set_a.add(27))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(28))),
        v_b.simd_eq(i32x16::splat(*set_a.add(29))),
    ];
    let mask =
        util::or_16(mask16) | util::or_8(mask8) |
        util::or_4(mask4) | (rest[0] | rest[1]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

unsafe fn avx512_31x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let mask16 = [
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
    let mask8 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
    ];
    let mask4 = [
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
        v_b.simd_eq(i32x16::splat(*set_a.add(27))),
    ];
    let rest = [
        v_b.simd_eq(i32x16::splat(*set_a.add(28))),
        v_b.simd_eq(i32x16::splat(*set_a.add(29))),
        v_b.simd_eq(i32x16::splat(*set_a.add(30))),
    ];
    let mask =
        util::or_16(mask16) | util::or_8(mask8) |
        util::or_4(mask4) | (rest[0] | rest[1] | rest[2]);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

#[allow(dead_code)]
unsafe fn avx512_32x16<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    let v_b: i32x16 = load_unsafe(set_b);
    let masks_left = [
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
    let masks_right = [
        v_b.simd_eq(i32x16::splat(*set_a.add(16))),
        v_b.simd_eq(i32x16::splat(*set_a.add(17))),
        v_b.simd_eq(i32x16::splat(*set_a.add(18))),
        v_b.simd_eq(i32x16::splat(*set_a.add(19))),
        v_b.simd_eq(i32x16::splat(*set_a.add(20))),
        v_b.simd_eq(i32x16::splat(*set_a.add(21))),
        v_b.simd_eq(i32x16::splat(*set_a.add(22))),
        v_b.simd_eq(i32x16::splat(*set_a.add(23))),
        v_b.simd_eq(i32x16::splat(*set_a.add(24))),
        v_b.simd_eq(i32x16::splat(*set_a.add(25))),
        v_b.simd_eq(i32x16::splat(*set_a.add(26))),
        v_b.simd_eq(i32x16::splat(*set_a.add(27))),
        v_b.simd_eq(i32x16::splat(*set_a.add(28))),
        v_b.simd_eq(i32x16::splat(*set_a.add(29))),
        v_b.simd_eq(i32x16::splat(*set_a.add(30))),
        v_b.simd_eq(i32x16::splat(*set_a.add(31))),
    ];
    let mask = util::or_16(masks_left) | util::or_16(masks_right);
    (*visitor).visit_vector16(v_b, mask.to_bitmask());
}

pub unsafe fn avx512_1x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_1x16(set_a, set_b, visitor);
    avx512_1x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_2x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_2x16(set_a, set_b, visitor);
    avx512_2x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_3x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_3x16(set_a, set_b, visitor);
    avx512_3x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_4x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_4x16(set_a, set_b, visitor);
    avx512_4x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_5x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_5x16(set_a, set_b, visitor);
    avx512_5x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_6x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_6x16(set_a, set_b, visitor);
    avx512_6x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_7x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_7x16(set_a, set_b, visitor);
    avx512_7x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_8x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_8x16(set_a, set_b, visitor);
    avx512_8x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_9x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_9x16(set_a, set_b, visitor);
    avx512_9x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_10x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_10x16(set_a, set_b, visitor);
    avx512_10x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_11x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_11x16(set_a, set_b, visitor);
    avx512_11x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_12x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_12x16(set_a, set_b, visitor);
    avx512_12x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_13x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_13x16(set_a, set_b, visitor);
    avx512_13x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_14x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_14x16(set_a, set_b, visitor);
    avx512_14x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_15x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_15x16(set_a, set_b, visitor);
    avx512_15x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_16x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_16x16(set_a, set_b, visitor);
    avx512_16x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_17x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_17x16(set_a, set_b, visitor);
    avx512_17x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_18x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_18x16(set_a, set_b, visitor);
    avx512_18x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_19x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_19x16(set_a, set_b, visitor);
    avx512_19x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_20x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_20x16(set_a, set_b, visitor);
    avx512_20x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_21x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_21x16(set_a, set_b, visitor);
    avx512_21x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_22x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_22x16(set_a, set_b, visitor);
    avx512_22x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_23x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_23x16(set_a, set_b, visitor);
    avx512_23x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_24x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_24x16(set_a, set_b, visitor);
    avx512_24x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_25x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_25x16(set_a, set_b, visitor);
    avx512_25x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_26x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_26x16(set_a, set_b, visitor);
    avx512_26x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_27x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_27x16(set_a, set_b, visitor);
    avx512_27x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_28x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_28x16(set_a, set_b, visitor);
    avx512_28x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_29x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_29x16(set_a, set_b, visitor);
    avx512_29x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_30x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_30x16(set_a, set_b, visitor);
    avx512_30x16(set_a, set_b.add(16), visitor);
}

pub unsafe fn avx512_31x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_31x16(set_a, set_b, visitor);
    avx512_31x16(set_a, set_b.add(16), visitor);
}

#[allow(dead_code)]
pub unsafe fn avx512_32x32<V: SimdVisitor16<i32>>(set_a: *const i32, set_b: *const i32, visitor: *mut V) {
    avx512_32x16(set_a, set_b, visitor);
    avx512_32x16(set_a, set_b.add(16), visitor);
}
