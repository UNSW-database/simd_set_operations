#![cfg(all(feature = "simd", target_feature = "avx512f"))]

use std::{
    simd::*,
    cmp::Ordering,
};
use crate::{
    visitor::{Visitor, SimdVisitor16},
    intersect, instructions::load_unsafe,
};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::arch::asm;

#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
            let mask = unsafe{
                emulate_mm512_2intersect_epi32_mask(v_a.into(), v_b.into())
            };

            visitor.visit_vector16(v_a, mask);

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

/// VP2INTERSECT emulation.
/// Díez-Cañas, G. (2021). Faster-Than-Native Alternatives for x86 VP2INTERSECT
/// Instructions. arXiv preprint arXiv:2112.06342.
#[inline]
unsafe fn emulate_mm512_2intersect_epi32_mask(a: __m512i, b: __m512i) -> u16 {
    let a1 = _mm512_alignr_epi32(a, a, 4);
    let b1 = _mm512_shuffle_epi32(b, _MM_PERM_ADCB);
    let nm00 = _mm512_cmpneq_epi32_mask(a, b);

    let a2 = _mm512_alignr_epi32(a, a, 8);
    let a3 = _mm512_alignr_epi32(a, a, 12);
    let nm01 = _mm512_cmpneq_epi32_mask(a1, b);
    let nm02 = _mm512_cmpneq_epi32_mask(a2, b);

    let nm03 = _mm512_cmpneq_epi32_mask(a3, b);
    let nm10 = _mm512_mask_cmpneq_epi32_mask(nm00, a , b1);
    let nm11 = _mm512_mask_cmpneq_epi32_mask(nm01, a1, b1);

    let b2 = _mm512_shuffle_epi32(b, _MM_PERM_BADC);
    let nm12 = _mm512_mask_cmpneq_epi32_mask(nm02, a2, b1);
    let nm13 = _mm512_mask_cmpneq_epi32_mask(nm03, a3, b1);
    let nm20 = _mm512_mask_cmpneq_epi32_mask(nm10, a , b2);

    let b3 = _mm512_shuffle_epi32(b, _MM_PERM_CBAD);
    let nm21 = _mm512_mask_cmpneq_epi32_mask(nm11, a1, b2);
    let nm22 = _mm512_mask_cmpneq_epi32_mask(nm12, a2, b2);
    let nm23 = _mm512_mask_cmpneq_epi32_mask(nm13, a3, b2);

    let nm0 = _mm512_mask_cmpneq_epi32_mask(nm20, a , b3);
    let nm1 = _mm512_mask_cmpneq_epi32_mask(nm21, a1, b3);
    let nm2 = _mm512_mask_cmpneq_epi32_mask(nm22, a2, b3);
    let nm3 = _mm512_mask_cmpneq_epi32_mask(nm23, a3, b3);

    return !(nm0 & nm1.rotate_left(4) & nm2.rotate_left(8) & nm3.rotate_right(4));
}

/// Intersect using VPCONFLICTD
/// Frank Tetzel (tetzank) https://github.com/tetzank/SIMDSetOperations
#[cfg(target_feature = "avx512cd")]
pub fn conflict_intersect<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + SimdVisitor16,
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
            let (vpool, mask) = unsafe {
                conflict_intersect_vector(v_a.into(), v_b.into())
            };

            visitor.visit_vector16(vpool.into(), mask);

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

#[inline]
#[cfg(target_feature = "avx512cd")]
unsafe fn conflict_intersect_vector(a: __m256i, b: __m256i) -> (__m512i, u16) {

    let za = _mm512_castsi256_si512(a);

    let mut vpool: __m512i;

    //let vpool = _mm512_inserti32x8(v_a, b, 1);
    asm!(
        "vinserti32x8 {vpool}, {za}, {yb}, 1",
        za = in(zmm_reg) za,
        yb = in(ymm_reg) b,
        vpool = out(zmm_reg) vpool,
    );

    let vconflict = _mm512_conflict_epi32(vpool);
    let mask = _mm512_cmpneq_epi32_mask(vconflict, i32x16::from_array([0;16]).into());
    (vpool, mask)
}
