#![cfg(all(feature = "simd", target_feature = "ssse3"))]
/// QFilter is a SIMD-based set intersection from the paper below.
///
/// Shuo Han, Lei Zou, and Jeffrey Xu Yu. 2018. Speeding Up Set Intersections in
/// Graph Algorithms using SIMD Instructions. In Proceedings of the 2018
/// International Conference on Management of Data (SIGMOD '18). Association for
/// Computing Machinery, New York, NY, USA, 1587–1602.
/// https://doi.org/10.1145/3183713.3196924
///
/// A significant portion of the implementation is derived from
/// https://github.com/pkumod/GraphSetIntersection (MIT License)

use crate::{
    visitor::{Visitor, SimdVisitor4, SimdBsrVisitor4},
    instructions::load_unsafe,
    intersect,
    instructions::{
        convert, shuffle_epi8,
        BYTE_CHECK_GROUP_A, BYTE_CHECK_GROUP_B,
        BYTE_CHECK_GROUP_A_VEC, BYTE_CHECK_GROUP_B_VEC
    }, bsr::BsrRef,
};
use std::{
    simd::*,
    simd::cmp::*,
    cmp::Ordering,
};

/// Version 2 of the QFilter algorithm as presented by Han et al. (see above)
/// Faster than version 1 (see qfilter_v1)
#[cfg(target_feature = "ssse3")]
pub fn qfilter<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        let v_a: i32x4 = unsafe { load_unsafe(ptr_a.add(i_a)) };
        let v_b: i32x4 = unsafe { load_unsafe(ptr_b.add(i_b)) };

        let byte_group_a: i8x16 = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
        let byte_group_b: i8x16 = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);

        let byte_check_mask = byte_group_a.simd_eq(byte_group_b);
        let bc_mask = byte_check_mask.to_bitmask() as usize;
        let ms_order = unsafe { *BYTE_CHECK_MASK_DICT.get_unchecked(bc_mask) };

        if ms_order != -2 {
            let cmp_mask =
            if ms_order > 0 {
                let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
                v_a.simd_eq(shuffle_epi8(v_b, match_shuffle))
            }
            else {
                let masks = [
                    v_a.simd_eq(v_b),
                    v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                    v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                    v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                ];
                (masks[0] | masks[1]) | (masks[2] | masks[3])
            };

            visitor.visit_vector4(v_a, cmp_mask.to_bitmask());
        }

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

#[cfg(target_feature = "ssse3")]
pub fn qfilter_bsr<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor4,
{
    use std::ops::BitAnd;

    const W: usize = 4;

    let mut i_a = 0;
    let mut i_b = 0;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    while i_a < st_a && i_b < st_b {
        let (base_a, base_b): (i32x4, i32x4) = unsafe {(
            load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32),
            load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32),
        )};
        let (byte_group_a, byte_group_b): (i8x16, i8x16) = (
            simd_swizzle!(convert(base_a), BYTE_CHECK_GROUP_A[0]),
            simd_swizzle!(convert(base_b), BYTE_CHECK_GROUP_B[0]),
        );

        let byte_check_mask = byte_group_a.simd_eq(byte_group_b);
        let bc_mask = byte_check_mask.to_bitmask() as usize;
        let ms_order = BYTE_CHECK_MASK_DICT[bc_mask];

        if ms_order != MS_NO_MATCH {
            let (state_a, state_b): (i32x4, i32x4) = unsafe {(
                load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32),
                load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32),
            )};

            let (cmp_mask, and_state) =
            if ms_order > 0 {
                // Single match
                let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
                let cmp_mask = base_a.simd_eq(shuffle_epi8(base_b, match_shuffle));

                let and_state = state_a & shuffle_epi8(state_b, match_shuffle);

                let state_mask = and_state.simd_ne(i32x4::from_array([0; 4]));

                (cmp_mask & state_mask, and_state)
            }
            else {
                // Multi-match
                let cmp_masks = [
                    base_a.simd_eq(base_b),
                    base_a.simd_eq(base_b.rotate_elements_left::<1>()),
                    base_a.simd_eq(base_b.rotate_elements_left::<2>()),
                    base_a.simd_eq(base_b.rotate_elements_left::<3>()),
                ];
                let state_masks = [
                    state_a & state_b,
                    state_a & state_b.rotate_elements_left::<1>(),
                    state_a & state_b.rotate_elements_left::<2>(),
                    state_a & state_b.rotate_elements_left::<3>(),
                ];
                let and_masks = [
                    state_masks[0] & cmp_masks[0].to_int(),
                    state_masks[1] & cmp_masks[1].to_int(),
                    state_masks[2] & cmp_masks[2].to_int(),
                    state_masks[3] & cmp_masks[3].to_int(),
                ];
                let and_state =
                    (and_masks[0] | and_masks[1]) |
                    (and_masks[2] | and_masks[3]);

                let state_mask = and_state.simd_ne(i32x4::from_array([0; 4]));
                (mask32x4::splat(true).bitand(state_mask), and_state)
            };

            visitor.visit_bsr_vector4(base_a, and_state, cmp_mask.to_bitmask());
        }

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


#[cfg(target_feature = "ssse3")]
pub fn qfilter_v1<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        let byte_group_a: i8x16 = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
        let byte_group_b: i8x16 = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);

        let mut bc_mask = byte_group_a.simd_eq(byte_group_b);
        let mut ms_order = unsafe {
            *BYTE_CHECK_MASK_DICT.get_unchecked(bc_mask.to_bitmask() as usize)
        };

        if ms_order == MS_MULTI_MATCH {
            (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 1);
            if ms_order == MS_MULTI_MATCH {
                (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 2);
                if ms_order == MS_MULTI_MATCH {
                    (_, ms_order) = byte_check(v_a, v_b, bc_mask, 3);
                }
            }
        }
        if ms_order != MS_NO_MATCH {
            debug_assert!(ms_order >= 0);

            let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
            let cmp_mask = v_a.simd_eq( shuffle_epi8(v_b, match_shuffle));

            visitor.visit_vector4(v_a, cmp_mask.to_bitmask())
        }

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

#[inline]
fn byte_check(a: i32x4, b: i32x4, prev_mask: mask8x16, index: usize) -> (mask8x16, i32) {
    let (byte_group_a, byte_group_b): (i8x16, i8x16) = unsafe {(
        shuffle_epi8(convert::<i32x4, i8x16>(a), *BYTE_CHECK_GROUP_A_VEC.get_unchecked(index)),
        shuffle_epi8(convert::<i32x4, i8x16>(b), *BYTE_CHECK_GROUP_B_VEC.get_unchecked(index)),
    )};
    let byte_check_mask = prev_mask & byte_group_a.simd_eq(byte_group_b);
    let bc_mask = byte_check_mask.to_bitmask();

    let byte_check_dict = unsafe { *BYTE_CHECK_MASK_DICT.get_unchecked(bc_mask as usize) };
    (byte_check_mask, byte_check_dict)
}

const BYTE_CHECK_MASK_DICT: [i32; 65536] = prepare_byte_check_mask_dict();
const MATCH_SHUFFLE_DICT: [u8x16; 256] = prepare_match_shuffle_dict4();

const MS_MULTI_MATCH: i32 = -1;
const MS_NO_MATCH: i32 = -2;

// 1) Compare least significant byte with all-pairs comparison
// 2) That output mask is index into this dictionary
// 3) This dictionary returns either multi-match, no match or the offset in which
// the match occurred.
// e.g.,
// v_a ??AB, ??CD, ??31, ??21 matching LSByte on
// v_b ??45, ??55, ??CD, ??33
const fn prepare_byte_check_mask_dict() -> [i32; 65536] {
    let mut dict = [0; 65536];

    let mut mask = 0;
    while mask < 65536 {
        dict[mask as usize] = byte_check_mask_to_offset(mask);
        mask += 1;
    }
    dict
}

const fn byte_check_mask_to_offset(mask: i32) -> i32 {
    // Every 4 bits of mask represent a comparison between some LS-Byte in A with
    // all LS-Bytes in B.
    let offsets: [i32; 4] = [
        cmp_to_offset(0xf & (mask)),
        cmp_to_offset(0xf & (mask >> 4)),
        cmp_to_offset(0xf & (mask >> 8)),
        cmp_to_offset(0xf & (mask >> 12)),
    ];

    if offsets[0] == MS_MULTI_MATCH || offsets[1] == MS_MULTI_MATCH ||
        offsets[2] == MS_MULTI_MATCH || offsets[3] == MS_MULTI_MATCH
    {
        MS_MULTI_MATCH
    }
    else if offsets[0] == MS_NO_MATCH && offsets[1] == MS_NO_MATCH &&
        offsets[2] == MS_NO_MATCH && offsets[3] == MS_NO_MATCH
    {
        MS_NO_MATCH
    }
    else {
        // Single match
        let mut i = 0;
        let mut result = 0;
        while i < 4 {
            let final_offset = if offsets[i] == MS_NO_MATCH {
                i as i32
            }
            else {
                offsets[i]
            };
            // Each offset takes up 2 bits.
            result |= final_offset << (2*i);
            i += 1;
        }
        result
    }
}

const fn cmp_to_offset(c: i32) -> i32 {
    match c {
        0 => MS_NO_MATCH,
        1 => 0,  // 1 << 0 => 0
        2 => 1,  // 1 << 1 => 1
        4 => 2,  // 1 << 2 => 2
        8 => 3,  // 1 << 3 => 3
        _ => MS_MULTI_MATCH,
    }
}

const fn prepare_match_shuffle_dict4() -> [u8x16; 256] {
    let mut dict = [u8x16::from_array([0; 16]); 256];
    let mut offsets = 0;
    while offsets < 256 {
        dict[offsets] = offsets_to_shuffle_mask4(offsets);
        offsets += 1;
    }
    dict
}

const fn offsets_to_shuffle_mask4(offsets: usize) -> u8x16 {
    const WORD_SIZE: usize = 4;
    const WORD_COUNT: usize = 4;

    let mut shuffle_mask = [0; 16];
    let mut word_i = 0;
    while word_i < WORD_COUNT {
        let offset = (offsets >> (word_i*2)) & 0b11;

        let mut byte_i = 0;
        while byte_i < WORD_SIZE {
            let byte_offset = offset * WORD_SIZE + byte_i;
            shuffle_mask[(word_i*WORD_SIZE) + byte_i] = byte_offset as u8;

            byte_i += 1;
        }
        word_i += 1;
    }
    u8x16::from_array(shuffle_mask)
}


// Branch
#[cfg(target_feature = "ssse3")]
pub fn qfilter_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        let mut v_a: i32x4 = unsafe { load_unsafe(ptr_a.add(i_a)) };
        let mut v_b: i32x4 = unsafe { load_unsafe(ptr_b.add(i_b)) };

        let mut byte_group_a: i8x16 = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
        let mut byte_group_b: i8x16 = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);

        loop {
            let byte_check_mask = byte_group_a.simd_eq(byte_group_b);
            let bc_mask = byte_check_mask.to_bitmask() as usize;
            let ms_order = unsafe { *BYTE_CHECK_MASK_DICT.get_unchecked(bc_mask) };

            if ms_order != -2 {
                let cmp_mask =
                if ms_order > 0 {
                    let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
                    v_a.simd_eq(shuffle_epi8(v_b, match_shuffle))
                }
                else {
                    let masks = [
                        v_a.simd_eq(v_b),
                        v_a.simd_eq(v_b.rotate_elements_left::<1>()),
                        v_a.simd_eq(v_b.rotate_elements_left::<2>()),
                        v_a.simd_eq(v_b.rotate_elements_left::<3>()),
                    ];
                    (masks[0] | masks[1]) | (masks[2] | masks[3])
                };

                visitor.visit_vector4(v_a, cmp_mask.to_bitmask());
            }

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
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                },
            }
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}

#[cfg(target_feature = "ssse3")]
pub fn qfilter_bsr_branch<'a, V>(set_a: BsrRef<'a>, set_b: BsrRef<'a>, visitor: &mut V)
where
    V: SimdBsrVisitor4,
{
    use std::ops::BitAnd;

    const W: usize = 4;

    let mut i_a = 0;
    let mut i_b = 0;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    if i_a < st_a && i_b < st_b {
        let (mut base_a, mut base_b): (i32x4, i32x4) = unsafe {(
            load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32),
            load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32),
        )};
        let (mut byte_group_a, mut byte_group_b): (i8x16, i8x16) = (
            simd_swizzle!(convert(base_a), BYTE_CHECK_GROUP_A[0]),
            simd_swizzle!(convert(base_b), BYTE_CHECK_GROUP_B[0]),
        );

        loop {
            let byte_check_mask = byte_group_a.simd_eq(byte_group_b);
            let bc_mask = byte_check_mask.to_bitmask() as usize;
            let ms_order = BYTE_CHECK_MASK_DICT[bc_mask];

            if ms_order != MS_NO_MATCH {
                let (state_a, state_b): (i32x4, i32x4) = unsafe {(
                    load_unsafe(set_a.states.as_ptr().add(i_a) as *const i32),
                    load_unsafe(set_b.states.as_ptr().add(i_b) as *const i32),
                )};

                let (cmp_mask, and_state) =
                if ms_order > 0 {
                    // Single match
                    let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
                    let cmp_mask = base_a.simd_eq(shuffle_epi8(base_b, match_shuffle));

                    let and_state = state_a & shuffle_epi8(state_b, match_shuffle);

                    let state_mask = and_state.simd_ne(i32x4::from_array([0; 4]));

                    // TODO: check that this compiles to ANDNOT
                    (cmp_mask & state_mask, and_state)
                }
                else {
                    // Multi-match
                    let cmp_masks = [
                        base_a.simd_eq(base_b),
                        base_a.simd_eq(base_b.rotate_elements_left::<1>()),
                        base_a.simd_eq(base_b.rotate_elements_left::<2>()),
                        base_a.simd_eq(base_b.rotate_elements_left::<3>()),
                    ];
                    let state_masks = [
                        state_a & state_b,
                        state_a & state_b.rotate_elements_left::<1>(),
                        state_a & state_b.rotate_elements_left::<2>(),
                        state_a & state_b.rotate_elements_left::<3>(),
                    ];
                    let and_masks = [
                        state_masks[0] & cmp_masks[0].to_int(),
                        state_masks[1] & cmp_masks[1].to_int(),
                        state_masks[2] & cmp_masks[2].to_int(),
                        state_masks[3] & cmp_masks[3].to_int(),
                    ];
                    let and_state = (and_masks[0] | and_masks[1]) |
                        (and_masks[2] | and_masks[3]);

                    let state_mask = and_state.simd_ne(i32x4::from_array([0; 4]));
                    (mask32x4::splat(true).bitand(state_mask), and_state)
                };

                visitor.visit_bsr_vector4(base_a, and_state, cmp_mask.to_bitmask());
            }

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
                    byte_group_a = simd_swizzle!(convert(base_a), BYTE_CHECK_GROUP_A[0]);
                    byte_group_b = simd_swizzle!(convert(base_b), BYTE_CHECK_GROUP_B[0]);
                }
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    base_a = unsafe{ load_unsafe(set_a.bases.as_ptr().add(i_a) as *const i32) };
                    byte_group_a = simd_swizzle!(convert(base_a), BYTE_CHECK_GROUP_A[0]);
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    base_b = unsafe{ load_unsafe(set_b.bases.as_ptr().add(i_b) as *const i32) };
                    byte_group_b = simd_swizzle!(convert(base_b), BYTE_CHECK_GROUP_B[0]);
                },
            }
        }
    }

    intersect::branchless_merge_bsr(
        unsafe { set_a.advanced_by_unchecked(i_a) },
        unsafe { set_b.advanced_by_unchecked(i_b) },
        visitor)
}


#[cfg(target_feature = "ssse3")]
pub fn qfilter_v1_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
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
        let mut byte_group_a: i8x16 = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
        let mut byte_group_b: i8x16 = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);

        loop {
            let mut bc_mask = byte_group_a.simd_eq(byte_group_b);
            let mut ms_order = unsafe {
                *BYTE_CHECK_MASK_DICT.get_unchecked(bc_mask.to_bitmask() as usize)
            };

            if ms_order == MS_MULTI_MATCH {
                (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 1);
                if ms_order == MS_MULTI_MATCH {
                    (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 2);
                    if ms_order == MS_MULTI_MATCH {
                        (_, ms_order) = byte_check(v_a, v_b, bc_mask, 3);
                    }
                }
            }
            if ms_order != MS_NO_MATCH {
                debug_assert!(ms_order >= 0);

                let match_shuffle = unsafe { *MATCH_SHUFFLE_DICT.get_unchecked(ms_order as usize) };
                let cmp_mask = v_a.simd_eq( shuffle_epi8(v_b, match_shuffle));

                visitor.visit_vector4(v_a, cmp_mask.to_bitmask())
            }

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
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                },
            }
        }
    }
    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}
