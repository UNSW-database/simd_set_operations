#![cfg(feature = "simd")]

use std::{
    simd::*,
    simd::cmp::*,
    cmp::Ordering,
};
use std::ops::BitOr;
use crate::{
    visitor::{Visitor, SimdVisitor4, SimdBsrVisitor4},
    intersect, instructions::load_unsafe,
    bsr::BsrRef,
    util::*,
};
// #[cfg(target_feature = "avx2")]
use crate::visitor::{SimdVisitor8, SimdBsrVisitor8, HandsOffVisitor, UnsafeWriter, HandsOffVistorReturnable};
// #[cfg(target_feature = "avx512f")]
use crate::visitor::{SimdVisitor16, SimdBsrVisitor16};
const fn twoPower<const N: usize>() -> usize {
    2 << N
}

const fn generateTable<T, const N: usize>() -> [[u8; N*size_of::<T>()];twoPower::<N>()]
where
LaneCount<N>: SupportedLaneCount,
T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement + Sized, [(); twoPower::<{ N }>()]:
{
    let length = twoPower::<N>();
    let mut table: [[u8; N*size_of::<T>()];twoPower::<N>()] = [[u8::MAX;N*size_of::<T>()]; twoPower::<N>()];
    let mut i = 0;
    while i < length {
        let mut j = N - 1;
        let mut curr = 0;
        while j < N {
            if (i & (1usize << j)) != 0 {
                let w = 0;
                while w < size_of::<T>() {
                    table[i][curr+w] = (size_of::<T>() * j + w) as u8;
                }
                curr += 1;
            }
            j += 1;
        }
        i += 1;
    }
    table

}

#[cfg(target_feature = "neon")]
pub fn shuffle4(a: i32x4, b: u8x16) {
    unsafe {
        std::arch::aarch64::vqtbl1q_u8(std::mem::transmute(a),std::mem::transmute(b));
    }
}
// #[cfg(target_feature = "neon")]
// pub fn shuffle8(a: i32x8, b: u8x16) {
//     unsafe {
//         // std::arch::aarch64::vqtbl1q_u8(std::mem::transmute(a), std::mem::transmute(b));
//     }
// }
pub fn broadcast_generic<T, V, const N: usize>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
LaneCount<N>: SupportedLaneCount,
V: Visitor<T> + HandsOffVisitor<T>,
T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement + Sized, [(); twoPower::<{ N }>()]:, [(); N*size_of::<T>()]:
{
    let mut p_a: *const T = set_a.as_ptr();
    let mut p_b: *const T = set_b.as_ptr();
    let table : [[u8; N*size_of::<T>()]; twoPower::<{ N }>()] = generateTable::<T, N >();
    unsafe {
        let end_a: *const T = unsafe { set_a.as_ptr().add((set_a.len() / N) * set_a.len()) };
        let end_b: *const T = unsafe { set_b.as_ptr().add((set_b.len() / N) * set_b.len()) };
        let i_a: usize = 0;
        let i_b: usize = 0;
        while p_a < end_a && p_b < end_b {
            let v_a: std::simd::Simd<T, N> = Simd::from_slice(std::slice::from_raw_parts(p_a, N));
            let mut mask: std::simd::Mask<T,N> = std::intrinsics::simd::simd_eq(v_a, std::simd::Simd::<T, N>::splat(*p_b));
            for i in 1..N {
                mask = mask.bitor(std::intrinsics::simd::simd_eq::<std::simd::Simd<T, N>, std::simd::Mask<T, N>>(v_a, std::simd::Simd::<T, N>::splat(*p_b)));
            }
            let bits = mask.to_bitmask();
            let mut hands_off = visitor.hands_off_visit::<N>();
            match hands_off {
                HandsOffVistorReturnable::counter(counter) => {
                    counter.count = bits.count_ones() as usize;
                },
                HandsOffVistorReturnable::vecWriter(writer) => {
                },
                HandsOffVistorReturnable::unsafeWriter(_) => {}
            }
            let shufflePos = table[bits as usize];
        }
    }
}

pub fn broadcast_sse<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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

        let masks = unsafe {[
            v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b))),
            v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 1))),
            v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 2))),
            v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 3))),
        ]};
        let mask = or_4(masks);

        visitor.visit_vector4(v_a, mask.to_bitmask());

        let a_max = unsafe{ *set_a.get_unchecked(i_a + W - 1) };
        let b_max = unsafe{ *set_b.get_unchecked(i_b + W - 1) };

        i_a += W * (a_max <= b_max) as usize;
        i_b += W * (b_max <= a_max) as usize;
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}
pub fn broadcast_generic2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + HandsOffVisitor<T>,
    T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement, [(); 2*size_of::<T>()]:
{
    broadcast_generic::<T, V, 2>(set_a, set_b, visitor)
}
pub fn broadcast_generic4<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + HandsOffVisitor<T>,
    T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement, [(); 4*size_of::<T>()]:
{
    broadcast_generic::<T, V, 4>(set_a, set_b, visitor)
}
pub fn broadcast_generic8<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + HandsOffVisitor<T>,
    T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement, [(); 8*size_of::<T>()]:
{
    broadcast_generic::<T, V, 8>(set_a, set_b, visitor)
}
pub fn broadcast_generic16<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + HandsOffVisitor<T>,
    T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement, [(); 16*size_of::<T>()]:
{
    broadcast_generic::<T, V, 16>(set_a, set_b, visitor)
}
pub fn broadcast_generic32<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T> + HandsOffVisitor<T>,
    T:  Ord + Copy + std::simd::SimdElement + std::simd::MaskElement, [(); 32*size_of::<T>()]:
{
    broadcast_generic::<T, V, 32>(set_a, set_b, visitor)
}
// #[cfg(any(target_feature = "neon", target_feature = "ssse3"))]


// #[cfg(target_feature = "avx2")]
pub fn broadcast_avx2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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

        let masks = unsafe {[
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 1))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 2))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 3))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 4))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 5))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 6))),
            v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 7))),
        ]};
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

// #[cfg(target_feature = "avx512f")]
pub fn broadcast_avx512<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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

        let masks = unsafe {[
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 1))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 2))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 3))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 4))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 5))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 6))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 7))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 8))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 9))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 10))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 11))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 12))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 13))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 14))),
            v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 15))),
        ]};
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

#[cfg(any(target_feature = "neon", target_feature = "ssse3"))]
pub fn broadcast_sse_bsr<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
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
        let state_a: i32x4 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
        let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

        let base_masks = [
            base_a.simd_eq(i32x4::splat(unsafe { *base_b })),
            base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(1) })),
            base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(2) })),
            base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(3) })),
        ];
        let state_masks = [
            base_masks[0].to_int() & (state_a & i32x4::splat(unsafe { *state_b })),
            base_masks[1].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(1) })),
            base_masks[2].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(2) })),
            base_masks[3].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(3) })),
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
pub fn broadcast_avx2_bsr<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
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
        let state_a: i32x8 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };

        let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
        let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

        let base_masks = [
            base_a.simd_eq(i32x8::splat(unsafe { *base_b })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(1) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(2) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(3) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(4) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(5) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(6) })),
            base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(7) })),
        ];
        let state_masks = [
            base_masks[0].to_int() & (state_a & i32x8::splat(unsafe { *state_b })),
            base_masks[1].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(1) })),
            base_masks[2].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(2) })),
            base_masks[3].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(3) })),
            base_masks[4].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(4) })),
            base_masks[5].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(5) })),
            base_masks[6].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(6) })),
            base_masks[7].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(7) })),
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
pub fn broadcast_avx512_bsr<'a, V>(
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
        let state_a: i32x16 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };

        let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
        let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

        let base_masks = [
            base_a.simd_eq(i32x16::splat(unsafe { *base_b })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(1) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(2) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(3) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(4) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(5) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(6) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(7) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(8) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(9) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(10) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(11) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(12) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(13) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(14) })),
            base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(15) })),
        ];
        let state_masks = [
            base_masks[ 0].to_int() & (state_a & i32x16::splat(unsafe { *state_b })),
            base_masks[ 1].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(1) })),
            base_masks[ 2].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(2) })),
            base_masks[ 3].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(3) })),
            base_masks[ 4].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(4) })),
            base_masks[ 5].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(5) })),
            base_masks[ 6].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(6) })),
            base_masks[ 7].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(7) })),
            base_masks[ 8].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(8) })),
            base_masks[ 9].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(9) })),
            base_masks[10].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(10) })),
            base_masks[11].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(11) })),
            base_masks[12].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(12) })),
            base_masks[13].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(13) })),
            base_masks[14].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(14) })),
            base_masks[15].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(15) })),
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



// Branch
#[cfg(any(target_feature = "neon", target_feature = "ssse3"))]
pub fn broadcast_sse_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        loop {
            let masks = unsafe {[
                v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b))),
                v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 1))),
                v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 2))),
                v_a.simd_eq(i32x4::splat(*ptr_b.add(i_b + 3))),
            ]};
            let mask = or_4(masks);

            visitor.visit_vector4(v_a, mask.to_bitmask());

            let a_max = unsafe{ *set_a.get_unchecked(i_a + W - 1) };
            let b_max = unsafe{ *set_b.get_unchecked(i_b + W - 1) };
            match a_max.cmp(&b_max) {
                Ordering::Equal => {
                    i_a += W;
                    i_b += W;
                    if i_a == st_a || i_b == st_b {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
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
pub fn broadcast_avx2_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        loop {
            let masks = unsafe {[
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 1))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 2))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 3))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 4))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 5))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 6))),
                v_a.simd_eq(i32x8::splat(*ptr_b.add(i_b + 7))),
            ]};
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
pub fn broadcast_avx512_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        loop {
            let masks = unsafe {[
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 1))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 2))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 3))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 4))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 5))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 6))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 7))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 8))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 9))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 10))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 11))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 12))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 13))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 14))),
                v_a.simd_eq(i32x16::splat(*ptr_b.add(i_b + 15))),
            ]};
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
                },
            }
        }
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}

#[cfg(any(target_feature = "neon", target_feature = "ssse3"))]
pub fn broadcast_sse_bsr_branch<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
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
        let mut state_a: i32x4 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        loop {
            let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
            let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

            let base_masks = [
                base_a.simd_eq(i32x4::splat(unsafe { *base_b })),
                base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(1) })),
                base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(2) })),
                base_a.simd_eq(i32x4::splat(unsafe { *base_b.add(3) })),
            ];
            let state_masks = [
                base_masks[ 0].to_int() & (state_a & i32x4::splat(unsafe { *state_b })),
                base_masks[ 1].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(1) })),
                base_masks[ 2].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(2) })),
                base_masks[ 3].to_int() & (state_a & i32x4::splat(unsafe { *state_b.add(3) })),
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
                    state_a = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
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
pub fn broadcast_avx2_bsr_branch<'a, V>(
    set_a: BsrRef<'a>,
    set_b: BsrRef<'a>,
    visitor: &mut V)
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
        let mut state_a: i32x8 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        loop {
            let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
            let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

            let base_masks = [
                base_a.simd_eq(i32x8::splat(unsafe { *base_b })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(1) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(2) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(3) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(4) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(5) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(6) })),
                base_a.simd_eq(i32x8::splat(unsafe { *base_b.add(7) })),
            ];
            let state_masks = [
                base_masks[ 0].to_int() & (state_a & i32x8::splat(unsafe { *state_b })),
                base_masks[ 1].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(1) })),
                base_masks[ 2].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(2) })),
                base_masks[ 3].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(3) })),
                base_masks[ 4].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(4) })),
                base_masks[ 5].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(5) })),
                base_masks[ 6].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(6) })),
                base_masks[ 7].to_int() & (state_a & i32x8::splat(unsafe { *state_b.add(7) })),
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
                    state_a = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
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
pub fn broadcast_avx512_bsr_branch<'a, V>(
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
        let mut state_a: i32x16 = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
        loop {
            let base_b = unsafe { set_b.bases.as_ptr().add(i_b) as *const i32 };
            let state_b = unsafe { set_b.states.as_ptr().add(i_b) as *const i32 };

            let base_masks = [
                base_a.simd_eq(i32x16::splat(unsafe { *base_b })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(1) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(2) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(3) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(4) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(5) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(6) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(7) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(8) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(9) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(10) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(11) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(12) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(13) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(14) })),
                base_a.simd_eq(i32x16::splat(unsafe { *base_b.add(15) })),
            ];
            let state_masks = [
                base_masks[ 0].to_int() & (state_a & i32x16::splat(unsafe { *state_b })),
                base_masks[ 1].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(1) })),
                base_masks[ 2].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(2) })),
                base_masks[ 3].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(3) })),
                base_masks[ 4].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(4) })),
                base_masks[ 5].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(5) })),
                base_masks[ 6].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(6) })),
                base_masks[ 7].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(7) })),
                base_masks[ 8].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(8) })),
                base_masks[ 9].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(9) })),
                base_masks[10].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(10) })),
                base_masks[11].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(11) })),
                base_masks[12].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(12) })),
                base_masks[13].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(13) })),
                base_masks[14].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(14) })),
                base_masks[15].to_int() & (state_a & i32x16::splat(unsafe { *state_b.add(15) })),
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
                    state_a = unsafe{ load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32) };
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
                },
            }
        }
    }
    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}

