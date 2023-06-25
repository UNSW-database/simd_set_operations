#![cfg(feature = "simd")]
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
    visitor::SimdVisitor,
    instructions::load_unsafe,
    intersect,
    instructions::{
        convert, shuffle_epi8,
        BYTE_CHECK_GROUP_A, BYTE_CHECK_GROUP_B,
        BYTE_CHECK_GROUP_A_VEC, BYTE_CHECK_GROUP_B_VEC
    },
};
use std::{
    simd::*,
    cmp::Ordering,
};

#[cfg(target_feature = "ssse3")]
#[inline(never)]
pub fn qfilter<V>(mut left: &[i32], mut right: &[i32], visitor: &mut V)
where
    V: SimdVisitor<i32, 4>,
{
    const S: usize = 4;

    if left.len() >= S && right.len() >= S {
        let (mut v_a, mut v_b): (i32x4, i32x4) = (
            unsafe{ load_unsafe(left.as_ptr()) },
            unsafe{ load_unsafe(right.as_ptr()) },
        );
        let (mut byte_group_a, mut byte_group_b): (i8x16, i8x16) = (
            simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]),
            simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]),
        );

        while left.len() >= S && right.len() >= S {
            let byte_check_mask = byte_group_a.simd_eq(byte_group_b);
            let bc_mask = byte_check_mask.to_bitmask() as usize;
            let ms_order = BYTE_CHECK_MASK_DICT[bc_mask];

            if ms_order != -2 {
                let cmp_mask =
                if ms_order > 0 {
                    v_a.simd_eq(
                        shuffle_epi8(v_b, MATCH_SHUFFLE_DICT[ms_order as usize])
                    )
                }
                else {
                    let masks = [
                        v_a.simd_eq(v_b),
                        v_a.simd_eq(v_b.rotate_lanes_left::<1>()),
                        v_a.simd_eq(v_b.rotate_lanes_left::<2>()),
                        v_a.simd_eq(v_b.rotate_lanes_left::<3>()),
                    ];
                    (masks[0] | masks[1]) | (masks[2] | masks[3])
                };

                visitor.visit_vector(v_a, cmp_mask.to_bitmask());
            }

            match left[S-1].cmp(&right[S-1]) {
                Ordering::Equal => {
                    left = &left[S..];
                    right = &right[S..];
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                }
                Ordering::Less => {
                    left = &left[S..];
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                    byte_group_a = simd_swizzle!(convert(v_a), BYTE_CHECK_GROUP_A[0]);
                },
                Ordering::Greater => {
                    right = &right[S..];
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                    byte_group_b = simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]);
                },
            }
        }
    }

    intersect::branchless_merge(left, right, visitor)
}


#[cfg(target_feature = "ssse3")]
pub fn qfilter_v1<V>(mut left: &[i32], mut right: &[i32], visitor: &mut V)
where
    V: SimdVisitor<i32, 4>,
{
    const S: usize = 4;

    if left.len() >= S && right.len() >= S {
        let (mut v_a, mut v_b): (i32x4, i32x4) = (
            unsafe{ load_unsafe(left.as_ptr()) },
            unsafe{ load_unsafe(right.as_ptr()) },
        );

        while left.len() >= S && right.len() >= S {
            let (byte_group_a, byte_group_b): (i8x16, i8x16) = (
                shuffle_epi8(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A_VEC[0]),
                shuffle_epi8(convert::<i32x4, i8x16>(v_b), BYTE_CHECK_GROUP_B_VEC[0]),
            );
            let mut bc_mask = byte_group_a.simd_eq(byte_group_b);
            let mut ms_order = BYTE_CHECK_MASK_DICT[bc_mask.to_bitmask() as usize];

            if ms_order == -1 {
                (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 1);
                if ms_order == -1 {
                    (bc_mask, ms_order) = byte_check(v_a, v_b, bc_mask, 2);
                    if ms_order == -1 {
                        (_, ms_order) = byte_check(v_a, v_b, bc_mask, 3);
                    }
                }
            }
            if ms_order != -2 {
                debug_assert!(ms_order >= 0);
                let cmp_mask = v_a.simd_eq(
                    shuffle_epi8(v_b, MATCH_SHUFFLE_DICT[ms_order as usize]));
                visitor.visit_vector(v_a, cmp_mask.to_bitmask())
            }

            match left[S-1].cmp(&right[S-1]) {
                Ordering::Equal => {
                    left = &left[S..];
                    right = &right[S..];
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                }
                Ordering::Less => {
                    left = &left[S..];
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                },
                Ordering::Greater => {
                    right = &right[S..];
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                },
            }
        }
    }

    intersect::branchless_merge(left, right, visitor)
}

#[inline]
fn byte_check(a: i32x4, b: i32x4, prev_mask: mask8x16, index: usize) -> (mask8x16, i32) {
    let (byte_group_a, byte_group_b): (i8x16, i8x16) = (
        shuffle_epi8(convert::<i32x4, i8x16>(a), BYTE_CHECK_GROUP_A_VEC[index]),
        shuffle_epi8(convert::<i32x4, i8x16>(b), BYTE_CHECK_GROUP_B_VEC[index]),
    );
    let byte_check_mask = prev_mask & byte_group_a.simd_eq(byte_group_b);
    let bc_mask = byte_check_mask.to_bitmask();
    (byte_check_mask, BYTE_CHECK_MASK_DICT[bc_mask as usize])
}

const BYTE_CHECK_MASK_DICT: [i32; 65536] = prepare_byte_check_mask_dict();
const MATCH_SHUFFLE_DICT: [u8x16; 256] = prepare_match_shuffle_dict();

const fn prepare_byte_check_mask_dict() -> [i32; 65536] {
    let mut mask = [0; 65536];

    let mut x = 0;
    while x < 65536 {
        let s: [i32; 4] = [
            c_to_s(0xf & (x)),
            c_to_s(0xf & (x >> 4)),
            c_to_s(0xf & (x >> 8)),
            c_to_s(0xf & (x >> 12)),
        ];

        mask[x as usize] = {
            // Multi-match
            if s[0] == 4 || s[1] == 4 || s[2] == 4 || s[3] == 4 {
                -1
            }
            // No match
            else if s[0] == -1 && s[1] == -1 && s[2] == -1 && s[3] == -1 {
                -2
            }
            // Single match
            else {
                let mut j = 0;
                let mut m = 0;
                while j < 4 {
                    let sv = if s[j] == -1 { j as i32 } else { s[j] };
                    m |= sv << (2*j);
                    j += 1;
                }
                m
            }
        };
        x += 1;
    }
    mask
}

const fn c_to_s(c: i32) -> i32 {
    match c {
        0 => -1,
        1 => 0,
        2 => 1,
        4 => 2,
        8 => 3,
        _ => 4,
    }
}

const fn prepare_match_shuffle_dict() -> [u8x16; 256] {
    let mut dict = [u8x16::from_array([0; 16]); 256];
    let mut x = 0;
    while x < 256 {
        let mut vec = [0; 16];
        let mut i = 0;
        while i < 4 {
            let c = (x >> (i << 1)) & 3; // c = 0, 1, 2, 3
            let mut j = 0;
            while j < 4 {
                vec[(i*4) + j] = (c * 4 + j) as u8;
                j += 1;
            }
            i += 1;
        }
        dict[x] = u8x16::from_array(vec);
        x += 1;
    }
    dict
}

pub fn qfilter_mono(left: &[i32], right: &[i32], visitor: &mut crate::visitor::VecWriter<i32>) {
    qfilter_v1(left, right, visitor);
    qfilter(left, right, visitor);
}
