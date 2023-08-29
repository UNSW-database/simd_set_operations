use std::simd::*;
use crate::{util::or_4, visitor::{Visitor, SimdVisitor4}, instructions::load_unsafe};

#[inline(always)]
pub unsafe fn sse_1x4<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a = i32x4::splat(*set_a);
    let v_b: i32x4 = load_unsafe(set_b);
    let mask = v_a.simd_eq(v_b);
    if mask.any() {
        visitor.visit(*set_a);
    }
}
#[inline(always)]
pub unsafe fn sse_1x8<V: Visitor<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a = i32x4::splat(*set_a);
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let mask = v_a.simd_eq(v_b0) | v_a.simd_eq(v_b1);
    if mask.any() {
        visitor.visit(*set_a);
    }
}

#[inline(always)]
pub unsafe fn sse_2x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
    ];
    let mask = masks[0] | masks[1];
    visitor.visit_vector4(v_b, mask.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_3x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
        v_b.simd_eq(i32x4::splat(*set_a.add(2))),
    ];
    let mask = masks[0] | masks[1] | masks[2];
    visitor.visit_vector4(v_b, mask.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_4x4<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_b: i32x4 = load_unsafe(set_b);
    let masks = [
        v_b.simd_eq(i32x4::splat(*set_a)),
        v_b.simd_eq(i32x4::splat(*set_a.add(1))),
        v_b.simd_eq(i32x4::splat(*set_a.add(2))),
        v_b.simd_eq(i32x4::splat(*set_a.add(3))),
    ];
    let mask = (masks[0] | masks[1]) | (masks[2] | masks[3]);
    visitor.visit_vector4(v_b, mask.to_bitmask());
}

#[inline(always)]
pub unsafe fn sse_2x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a0 = i32x4::splat(*set_a);
    let v_a1 = i32x4::splat(*set_a.add(1));
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = v_b0.simd_eq(v_a0) | v_b0.simd_eq(v_a1);
    let m_b1 = v_b1.simd_eq(v_a0) | v_b1.simd_eq(v_a1);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_3x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let v_a0 = i32x4::splat(*set_a);
    let v_a1 = i32x4::splat(*set_a.add(1));
    let v_a2 = i32x4::splat(*set_a.add(2));

    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));

    let m_b0 = v_b0.simd_eq(v_a0) | v_b0.simd_eq(v_a1) | v_b0.simd_eq(v_a2);
    let m_b1 = v_b1.simd_eq(v_a0) | v_b1.simd_eq(v_a1) | v_b1.simd_eq(v_a2);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_4x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]),
        v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]),
        v_b0.simd_eq(a[3]),
    ]);
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]),
        v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]),
        v_b1.simd_eq(a[3]),
    ]);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_5x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]),
        v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]),
        v_b0.simd_eq(a[3]),
    ]) | v_b0.simd_eq(a[4]);
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]),
        v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]),
        v_b1.simd_eq(a[3]),
    ]) | v_b1.simd_eq(a[4]);
    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_6x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
        i32x4::splat(*set_a.add(5)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]), v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]), v_b0.simd_eq(a[3]),
    ]) | (v_b0.simd_eq(a[4]) | v_b0.simd_eq(a[5]));
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]), v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]), v_b1.simd_eq(a[3]),
    ]) | (v_b1.simd_eq(a[4]) | v_b1.simd_eq(a[5]));

    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
#[inline(always)]
pub unsafe fn sse_7x8<V: SimdVisitor4<i32>>(set_a: *const i32, set_b: *const i32, visitor: &mut V) {
    let a = [
        i32x4::splat(*set_a),
        i32x4::splat(*set_a.add(1)),
        i32x4::splat(*set_a.add(2)),
        i32x4::splat(*set_a.add(3)),
        i32x4::splat(*set_a.add(4)),
        i32x4::splat(*set_a.add(5)),
        i32x4::splat(*set_a.add(6)),
    ];
    let v_b0: i32x4 = load_unsafe(set_b);
    let v_b1: i32x4 = load_unsafe(set_b.add(4));
    let m_b0 = or_4([
        v_b0.simd_eq(a[0]), v_b0.simd_eq(a[1]),
        v_b0.simd_eq(a[2]), v_b0.simd_eq(a[3]),
    ]) | (v_b0.simd_eq(a[4]) | v_b0.simd_eq(a[5]) | v_b0.simd_eq(a[6]));
    let m_b1 = or_4([
        v_b1.simd_eq(a[0]), v_b1.simd_eq(a[1]),
        v_b1.simd_eq(a[2]), v_b1.simd_eq(a[3]),
    ]) | (v_b1.simd_eq(a[4]) | v_b1.simd_eq(a[5]) | v_b1.simd_eq(a[6]));

    visitor.visit_vector4(v_b0, m_b0.to_bitmask());
    visitor.visit_vector4(v_b1, m_b1.to_bitmask());
}
