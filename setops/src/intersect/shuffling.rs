#![cfg(feature = "simd")]

use std::{
    simd::*,
    simd::cmp::*,
    cmp::Ordering,
};

use crate::{
    visitor::{Visitor, SimdVisitor4,SimdBsrVisitor4},
    intersect, instructions::load_unsafe,
    bsr::BsrRef,
    util::*,
};
#[cfg(target_feature = "avx2")]
use crate::visitor::{
    SimdVisitor8, SimdBsrVisitor8,
};
#[cfg(target_feature = "avx512f")]
use crate::visitor::{SimdVisitor16, SimdBsrVisitor16};

/// SIMD Shuffling set intersection algorithm - Ilya Katsov 2012
/// https://highlyscalable.wordpress.com/2012/06/05/fast-intersection-sorted-lists-sse/
/// Implementation modified from roaring-rs
#[cfg(target_feature = "ssse3")]
pub fn shuffling_sse<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor4,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let v_a: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let v_b: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

        let masks = [
            v_a.simd_eq(v_b),
            v_a.simd_eq(v_b.rotate_elements_left::<1>()),
            v_a.simd_eq(v_b.rotate_elements_left::<2>()),
            v_a.simd_eq(v_b.rotate_elements_left::<3>()),
        ];
        let mask = or_4(masks);

        visitor.visit_vector4(v_a, mask.to_bitmask());

        let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn shuffling_avx2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor8,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 8;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let v_a: i32x8 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let v_b: i32x8 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        let masks = [
                v_a.simd_eq(v_b),
                v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                v_a.simd_eq(v_b.rotate_elements_left::<4>()),
                v_a.simd_eq(v_b.rotate_elements_left::<5>()),
                v_a.simd_eq(v_b.rotate_elements_left::<6>()),
                v_a.simd_eq(v_b.rotate_elements_left::<7>()),
        ];
        let mask = or_8(masks);

        visitor.visit_vector8(v_a, mask.to_bitmask());

        let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}


#[cfg(target_feature = "avx512f")]
pub fn shuffling_avx512<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor16,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 16;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let v_a: i32x16 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let v_b: i32x16 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

        let masks = [
                v_a.simd_eq(v_b),
                v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                v_a.simd_eq(v_b.rotate_elements_left::<4>()),
                v_a.simd_eq(v_b.rotate_elements_left::<5>()),
                v_a.simd_eq(v_b.rotate_elements_left::<6>()),
                v_a.simd_eq(v_b.rotate_elements_left::<7>()),
                v_a.simd_eq(v_b.rotate_elements_left::<8>()),
                v_a.simd_eq(v_b.rotate_elements_left::<9>()),
                v_a.simd_eq(v_b.rotate_elements_left::<10>()),
                v_a.simd_eq(v_b.rotate_elements_left::<11>()),
                v_a.simd_eq(v_b.rotate_elements_left::<12>()),
                v_a.simd_eq(v_b.rotate_elements_left::<13>()),
                v_a.simd_eq(v_b.rotate_elements_left::<14>()),
                v_a.simd_eq(v_b.rotate_elements_left::<15>()),
        ];
        let mask = or_16(masks);

        visitor.visit_vector16(v_a, mask.to_bitmask());

        let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}


// BSR implementations //

#[cfg(target_feature = "ssse3")]
pub fn shuffling_sse_bsr<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor4,
{
    const W: usize = 4;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let base_a: i32x4 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let base_b: i32x4 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let state_a: i32x4 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let state_b: i32x4 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        
        let base_masks = [
            base_a.simd_eq(base_b),
            base_a.simd_eq(base_b.rotate_elements_left::<1>()),
            base_a.simd_eq(base_b.rotate_elements_left::<2>()),
            base_a.simd_eq(base_b.rotate_elements_left::<3>()),
        ];
        let state_masks = [
            base_masks[0].to_int() & (state_a & state_b),
            base_masks[1].to_int() & (state_a & state_b.rotate_elements_left::<1>()),
            base_masks[2].to_int() & (state_a & state_b.rotate_elements_left::<2>()),
            base_masks[3].to_int() & (state_a & state_b.rotate_elements_left::<3>()),
        ];

        let base_mask = or_4(base_masks);
        let state_all = or_4(state_masks);
        let state_mask = state_all.simd_ne(i32x4::from_array([0; 4]));

        let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

        visitor.visit_bsr_vector4(base_a, state_all, total_mask);

        let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn shuffling_avx2_bsr<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor8,
{
    const W: usize = 8;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let base_a: i32x8 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let base_b: i32x8 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let state_a: i32x8 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let state_b: i32x8 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        
        let base_masks = [
            base_a.simd_eq(base_b),
            base_a.simd_eq(base_b.rotate_elements_left::<1>()),
            base_a.simd_eq(base_b.rotate_elements_left::<2>()),
            base_a.simd_eq(base_b.rotate_elements_left::<3>()),
            base_a.simd_eq(base_b.rotate_elements_left::<4>()),
            base_a.simd_eq(base_b.rotate_elements_left::<5>()),
            base_a.simd_eq(base_b.rotate_elements_left::<6>()),
            base_a.simd_eq(base_b.rotate_elements_left::<7>()),
        ];
        let state_masks = [
            base_masks[0].to_int() & (state_a & state_b),
            base_masks[1].to_int() & (state_a & state_b.rotate_elements_left::<1>()),
            base_masks[2].to_int() & (state_a & state_b.rotate_elements_left::<2>()),
            base_masks[3].to_int() & (state_a & state_b.rotate_elements_left::<3>()),
            base_masks[4].to_int() & (state_a & state_b.rotate_elements_left::<4>()),
            base_masks[5].to_int() & (state_a & state_b.rotate_elements_left::<5>()),
            base_masks[6].to_int() & (state_a & state_b.rotate_elements_left::<6>()),
            base_masks[7].to_int() & (state_a & state_b.rotate_elements_left::<7>()),
        ];

        let base_mask = or_8(base_masks);
        let state_all = or_8(state_masks);
        let state_mask = state_all.simd_ne(i32x8::from_array([0; 8]));

        let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

        visitor.visit_bsr_vector8(base_a, state_all, total_mask);

        let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}

#[cfg(target_feature = "avx512f")]
pub fn shuffling_avx512_bsr<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
where
    V: SimdBsrVisitor16,
{
    const W: usize = 16;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let base_a: i32x16 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let base_b: i32x16 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let state_a: i32x16 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let state_b: i32x16 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        
        let base_masks = [
            base_a.simd_eq(base_b),
            base_a.simd_eq(base_b.rotate_elements_left::<1>()),
            base_a.simd_eq(base_b.rotate_elements_left::<2>()),
            base_a.simd_eq(base_b.rotate_elements_left::<3>()),
            base_a.simd_eq(base_b.rotate_elements_left::<4>()),
            base_a.simd_eq(base_b.rotate_elements_left::<5>()),
            base_a.simd_eq(base_b.rotate_elements_left::<6>()),
            base_a.simd_eq(base_b.rotate_elements_left::<7>()),
            base_a.simd_eq(base_b.rotate_elements_left::<8>()),
            base_a.simd_eq(base_b.rotate_elements_left::<9>()),
            base_a.simd_eq(base_b.rotate_elements_left::<10>()),
            base_a.simd_eq(base_b.rotate_elements_left::<11>()),
            base_a.simd_eq(base_b.rotate_elements_left::<12>()),
            base_a.simd_eq(base_b.rotate_elements_left::<13>()),
            base_a.simd_eq(base_b.rotate_elements_left::<14>()),
            base_a.simd_eq(base_b.rotate_elements_left::<15>()),
        ];
        let state_masks = [
            base_masks[ 0].to_int() & (state_a & state_b),
            base_masks[ 1].to_int() & (state_a & state_b.rotate_elements_left::< 1>()),
            base_masks[ 2].to_int() & (state_a & state_b.rotate_elements_left::< 2>()),
            base_masks[ 3].to_int() & (state_a & state_b.rotate_elements_left::< 3>()),
            base_masks[ 4].to_int() & (state_a & state_b.rotate_elements_left::< 4>()),
            base_masks[ 5].to_int() & (state_a & state_b.rotate_elements_left::< 5>()),
            base_masks[ 6].to_int() & (state_a & state_b.rotate_elements_left::< 6>()),
            base_masks[ 7].to_int() & (state_a & state_b.rotate_elements_left::< 7>()),
            base_masks[ 8].to_int() & (state_a & state_b.rotate_elements_left::< 8>()),
            base_masks[ 9].to_int() & (state_a & state_b.rotate_elements_left::< 9>()),
            base_masks[10].to_int() & (state_a & state_b.rotate_elements_left::<10>()),
            base_masks[11].to_int() & (state_a & state_b.rotate_elements_left::<11>()),
            base_masks[12].to_int() & (state_a & state_b.rotate_elements_left::<12>()),
            base_masks[13].to_int() & (state_a & state_b.rotate_elements_left::<13>()),
            base_masks[14].to_int() & (state_a & state_b.rotate_elements_left::<14>()),
            base_masks[15].to_int() & (state_a & state_b.rotate_elements_left::<15>()),
        ];

        let base_mask = or_16(base_masks);
        let state_all = or_16(state_masks);
        let state_mask = state_all.simd_ne(i32x16::from_array([0; 16]));

        let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

        visitor.visit_bsr_vector16(base_a, state_all, total_mask);

        let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
        let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };
        
        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}




// Branch versions
#[cfg(target_feature = "ssse3")]
pub fn shuffling_sse_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor4,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let mut v_b: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        loop {
            let masks = [
                v_a.simd_eq(v_b),
                v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                v_a.simd_eq(v_b.rotate_elements_left::<3>()),
            ];
            let mask = or_4(masks);

            visitor.visit_vector4(v_a, mask.to_bitmask());

            let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
            }
        }
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn shuffling_avx2_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor8,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 8;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x8 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let mut v_b: i32x8 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        loop {
            let masks = [
                 v_a.simd_eq(v_b),
                 v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<4>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<5>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<6>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<7>()),
            ];
            let mask = or_8(masks);

            visitor.visit_vector8(v_a, mask.to_bitmask());

            let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
            }
        }
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}


#[cfg(target_feature = "avx512f")]
pub fn shuffling_avx512_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor16,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 16;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x16 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let mut v_b: i32x16 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        loop {
            let masks = [
                 v_a.simd_eq(v_b),
                 v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<4>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<5>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<6>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<7>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<8>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<9>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<10>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<11>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<12>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<13>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<14>()),
                 v_a.simd_eq(v_b.rotate_elements_left::<15>()),
            ];
            let mask = or_16(masks);

            visitor.visit_vector16(v_a, mask.to_bitmask());

            let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                },
            }
        }
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}


// BSR implementations //

#[cfg(target_feature = "ssse3")]
pub fn shuffling_sse_bsr_branch<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor4,
{
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
                base_a.simd_eq(base_b.rotate_elements_left::<1>()),
                base_a.simd_eq(base_b.rotate_elements_left::<2>()),
                base_a.simd_eq(base_b.rotate_elements_left::<3>()),
            ];
            let state_masks = [
                base_masks[0].to_int() & (state_a & state_b),
                base_masks[1].to_int() & (state_a & state_b.rotate_elements_left::<1>()),
                base_masks[2].to_int() & (state_a & state_b.rotate_elements_left::<2>()),
                base_masks[3].to_int() & (state_a & state_b.rotate_elements_left::<3>()),
            ];

            let base_mask = or_4(base_masks);
            let state_all = or_4(state_masks);
            let state_mask = state_all.simd_ne(i32x4::from_array([0; 4]));

            let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

            visitor.visit_bsr_vector4(base_a, state_all, total_mask);

            let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };
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
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn shuffling_avx2_bsr_branch<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor8,
{
    const W: usize = 8;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut base_a: i32x8 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let mut base_b: i32x8 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let mut state_a: i32x8 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let mut state_b: i32x8 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        loop {
            let base_masks = [
                base_a.simd_eq(base_b),
                base_a.simd_eq(base_b.rotate_elements_left::<1>()),
                base_a.simd_eq(base_b.rotate_elements_left::<2>()),
                base_a.simd_eq(base_b.rotate_elements_left::<3>()),
                base_a.simd_eq(base_b.rotate_elements_left::<4>()),
                base_a.simd_eq(base_b.rotate_elements_left::<5>()),
                base_a.simd_eq(base_b.rotate_elements_left::<6>()),
                base_a.simd_eq(base_b.rotate_elements_left::<7>()),
            ];
            let state_masks = [
                base_masks[0].to_int() & (state_a & state_b),
                base_masks[1].to_int() & (state_a & state_b.rotate_elements_left::<1>()),
                base_masks[2].to_int() & (state_a & state_b.rotate_elements_left::<2>()),
                base_masks[3].to_int() & (state_a & state_b.rotate_elements_left::<3>()),
                base_masks[4].to_int() & (state_a & state_b.rotate_elements_left::<4>()),
                base_masks[5].to_int() & (state_a & state_b.rotate_elements_left::<5>()),
                base_masks[6].to_int() & (state_a & state_b.rotate_elements_left::<6>()),
                base_masks[7].to_int() & (state_a & state_b.rotate_elements_left::<7>()),
            ];

            let base_mask = or_8(base_masks);
            let state_all = or_8(state_masks);
            let state_mask = state_all.simd_ne(i32x8::from_array([0; 8]));

            let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

            visitor.visit_bsr_vector8(base_a, state_all, total_mask);

            let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };
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
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}

#[cfg(target_feature = "avx512f")]
pub fn shuffling_avx512_bsr_branch<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
where
    V: SimdBsrVisitor16,
{
    const W: usize = 16;
    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut base_a: i32x16 = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
        let mut base_b: i32x16 = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
        let mut state_a: i32x16 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let mut state_b: i32x16 = unsafe{ load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32) };
        loop {
            let base_masks = [
                base_a.simd_eq(base_b),
                base_a.simd_eq(base_b.rotate_elements_left::<1>()),
                base_a.simd_eq(base_b.rotate_elements_left::<2>()),
                base_a.simd_eq(base_b.rotate_elements_left::<3>()),
                base_a.simd_eq(base_b.rotate_elements_left::<4>()),
                base_a.simd_eq(base_b.rotate_elements_left::<5>()),
                base_a.simd_eq(base_b.rotate_elements_left::<6>()),
                base_a.simd_eq(base_b.rotate_elements_left::<7>()),
                base_a.simd_eq(base_b.rotate_elements_left::<8>()),
                base_a.simd_eq(base_b.rotate_elements_left::<9>()),
                base_a.simd_eq(base_b.rotate_elements_left::<10>()),
                base_a.simd_eq(base_b.rotate_elements_left::<11>()),
                base_a.simd_eq(base_b.rotate_elements_left::<12>()),
                base_a.simd_eq(base_b.rotate_elements_left::<13>()),
                base_a.simd_eq(base_b.rotate_elements_left::<14>()),
                base_a.simd_eq(base_b.rotate_elements_left::<15>()),
            ];
            let state_masks = [
                base_masks[ 0].to_int() & (state_a & state_b),
                base_masks[ 1].to_int() & (state_a & state_b.rotate_elements_left::< 1>()),
                base_masks[ 2].to_int() & (state_a & state_b.rotate_elements_left::< 2>()),
                base_masks[ 3].to_int() & (state_a & state_b.rotate_elements_left::< 3>()),
                base_masks[ 4].to_int() & (state_a & state_b.rotate_elements_left::< 4>()),
                base_masks[ 5].to_int() & (state_a & state_b.rotate_elements_left::< 5>()),
                base_masks[ 6].to_int() & (state_a & state_b.rotate_elements_left::< 6>()),
                base_masks[ 7].to_int() & (state_a & state_b.rotate_elements_left::< 7>()),
                base_masks[ 8].to_int() & (state_a & state_b.rotate_elements_left::< 8>()),
                base_masks[ 9].to_int() & (state_a & state_b.rotate_elements_left::< 9>()),
                base_masks[10].to_int() & (state_a & state_b.rotate_elements_left::<10>()),
                base_masks[11].to_int() & (state_a & state_b.rotate_elements_left::<11>()),
                base_masks[12].to_int() & (state_a & state_b.rotate_elements_left::<12>()),
                base_masks[13].to_int() & (state_a & state_b.rotate_elements_left::<13>()),
                base_masks[14].to_int() & (state_a & state_b.rotate_elements_left::<14>()),
                base_masks[15].to_int() & (state_a & state_b.rotate_elements_left::<15>()),
            ];

            let base_mask = or_16(base_masks);
            let state_all = or_16(state_masks);
            let state_mask = state_all.simd_ne(i32x16::from_array([0; 16]));

            let total_mask = base_mask.to_bitmask() & state_mask.to_bitmask();

            visitor.visit_bsr_vector16(base_a, state_all, total_mask);

            let a_max = unsafe { *set_a.bases.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.bases.get_unchecked(i_b + W - 1) };
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
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}
