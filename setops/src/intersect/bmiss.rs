#[cfg(feature = "simd")]
use {
    std::{
        cmp::Ordering,
        simd::*,
    },
    crate::instructions::{
        load_unsafe,
        BYTE_CHECK_GROUP_A,
        BYTE_CHECK_GROUP_B
    },
};


use crate::{visitor::Visitor, intersect::branchless_merge};


#[inline(never)]
pub fn bmiss_scalar_3x<T, V>(mut left: &[T], mut right: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    const S: usize = 3;

    while left.len() >= S && right.len() >= S {

        if left[0] == right[0] || left[0] == right[1] || left[0] == right[2] {
            visitor.visit(left[0]);
        }
        if left[1] == right[0] || left[1] == right[1] || left[1] == right[2] {
            visitor.visit(left[1]);
        }
        if left[2] == right[0] || left[2] == right[1] || left[2] == right[2] {
            visitor.visit(left[2]);
        }

        bmiss_advance(&mut left, &mut right, S);
    }

    branchless_merge(left, right, visitor)
}

#[inline(never)]
pub fn bmiss_scalar_4x<T, V>(mut left: &[T], mut right: &[T], visitor: &mut V)
where
    T: Ord + Copy,
    V: Visitor<T>,
{
    const S: usize = 4;

    while left.len() >= S && right.len() >= S {

        if left[0] == right[0] || left[0] == right[1] ||
            left[0] == right[2] || left[0] == right[3]
        {
            visitor.visit(left[0]);
        }
        if left[1] == right[0] || left[1] == right[1] ||
            left[1] == right[2] || left[1] == right[3]
        {
            visitor.visit(left[1]);
        }
        if left[2] == right[0] || left[2] == right[1] ||
            left[2] == right[2] || left[2] == right[3]
        {
            visitor.visit(left[2]);
        }
        if left[3] == right[0] || left[3] == right[1] ||
            left[3] == right[2] || left[3] == right[3]
        {
            visitor.visit(left[3]);
        }

        bmiss_advance(&mut left, &mut right, S);
    }

    branchless_merge(left, right, visitor)
}

#[cfg(feature = "simd")]
const WORD_CHECK_SHUFFLE_A01: [usize; 4] = [0,0,1,1];
#[cfg(feature = "simd")]
const WORD_CHECK_SHUFFLE_A23: [usize; 4] = [2,2,3,3];
#[cfg(feature = "simd")]
const WORD_CHECK_SHUFFLE_B01: [usize; 4] = [0,1,0,1];
#[cfg(feature = "simd")]
const WORD_CHECK_SHUFFLE_B23: [usize; 4] = [2,3,2,3];

// Reference: https://github.com/pkumod/GraphSetIntersection
#[cfg(all(feature = "simd", target_feature = "sse"))]
#[inline(never)]
pub fn bmiss<V>(mut left: &[i32], mut right: &[i32], visitor: &mut V)
where
    V: Visitor<i32>,
{
    use crate::instructions::convert;

    const S: usize = 4;

    if left.len() >= S && right.len() >= S {
        let mut v_a: i32x4 = unsafe{ load_unsafe(left.as_ptr()) };
        let mut v_b: i32x4 = unsafe{ load_unsafe(right.as_ptr()) };

        loop {
            let byte_check_mask0 =
                simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[0])
                .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]));
            let byte_check_mask1 =
                simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[1])
                .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[1]));

            if !(byte_check_mask0 & byte_check_mask1).any() {
                match left[S-1].cmp(&right[S-1]) {
                    Ordering::Equal => {
                        left = &left[S..];
                        right = &right[S..];
                        if left.len() < S || right.len() < S {
                            break;
                        }
                        v_a = unsafe{ load_unsafe(left.as_ptr()) };
                        v_b = unsafe{ load_unsafe(right.as_ptr()) };
                    }
                    Ordering::Less => {
                        left = &left[S..];
                        if left.len() < S {
                            break;
                        }
                        v_a = unsafe{ load_unsafe(left.as_ptr()) };
                    },
                    Ordering::Greater => {
                        right = &(*right)[S..];
                        if right.len() < S {
                            break;
                        }
                        v_b = unsafe{ load_unsafe(right.as_ptr()) };
                    },
                }
                continue;
            }

            let vas = [
                simd_swizzle!(v_a, WORD_CHECK_SHUFFLE_A01),
                simd_swizzle!(v_a, WORD_CHECK_SHUFFLE_A23)
            ];
            let vbs = [
                simd_swizzle!(v_b, WORD_CHECK_SHUFFLE_B01),
                simd_swizzle!(v_b, WORD_CHECK_SHUFFLE_B23)
            ];
            let word_check_mask00 = vas[0].simd_eq(vbs[0]);
            let word_check_mask01 = vas[0].simd_eq(vbs[1]);
            let word_check_mask0 = word_check_mask00 | word_check_mask01;

            let word_check_mask10 = vas[1].simd_eq(vbs[0]);
            let word_check_mask11 = vas[1].simd_eq(vbs[1]);
            let word_check_mask1 = word_check_mask10 | word_check_mask11;

            let wc_mask0: u8 = word_check_mask0.to_bitmask();
            if (wc_mask0 & 0b0011) != 0 { visitor.visit(left[0]) }
            if (wc_mask0 & 0b1100) != 0 { visitor.visit(left[1]) }

            let wc_mask1: u8 = word_check_mask1.to_bitmask();
            if (wc_mask1 & 0b0011) != 0 { visitor.visit(left[2]) }
            if (wc_mask1 & 0b1100) != 0 { visitor.visit(left[3]) }

            match left[S-1].cmp(&right[S-1]) {
                Ordering::Equal => {
                    left = &left[S..];
                    right = &right[S..];
                    if left.len() < S || right.len() < S {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                }
                Ordering::Less => {
                    left = &left[S..];
                    if left.len() < S {
                        break;
                    }
                    v_a = unsafe{ load_unsafe(left.as_ptr()) };
                },
                Ordering::Greater => {
                    right = &(*right)[S..];
                    if right.len() < S {
                        break;
                    }
                    v_b = unsafe{ load_unsafe(right.as_ptr()) };
                },
            }
        }
    }

    branchless_merge(left, right, visitor)
}

#[cfg(feature = "simd")]
const BMISS_STTNI_BC_ARRAY: [u8x16; 2] = [
    u8x16::from_array([0, 1, 4, 5, 8, 9, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255]),
    u8x16::from_array([255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 4, 5, 8, 9, 12, 13]),
];

#[cfg(all(feature = "simd", target_feature = "sse", target_feature = "sse4.2"))]
#[inline(never)]
pub fn bmiss_sttni<V>(mut left: &[i32], mut right: &[i32], visitor: &mut V)
where
    V: Visitor<i32>,
{
    use crate::instructions::shuffle_epi8;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const S: usize = 8;

    if left.len() >= S && right.len() >= S {
        let mut v_a0: i32x4 = unsafe{ load_unsafe(left.as_ptr()) };
        let mut v_b0: i32x4 = unsafe{ load_unsafe(right.as_ptr()) };
        let mut v_a1: i32x4 = unsafe{ load_unsafe(left.as_ptr().add(4)) };
        let mut v_b1: i32x4 = unsafe{ load_unsafe(right.as_ptr().add(4)) };

        loop {
            let byte_group_a =
                shuffle_epi8(v_a0, BMISS_STTNI_BC_ARRAY[0]) |
                shuffle_epi8(v_a1, BMISS_STTNI_BC_ARRAY[1]);
            let byte_group_b =
                shuffle_epi8(v_b0, BMISS_STTNI_BC_ARRAY[0]) |
                shuffle_epi8(v_b1, BMISS_STTNI_BC_ARRAY[1]);

            let bc_mask: i32x4 = unsafe { _mm_cmpestrm(
                byte_group_b.into(), 8,
                byte_group_a.into(), 8,
                _SIDD_UWORD_OPS | _SIDD_CMP_EQUAL_ANY | _SIDD_BIT_MASK)
            }.into();

            let mut r = bc_mask[0];

            while r != 0 {
                let p = ((!r) & (r - 1)).count_ones();
                r &= r - 1;

                let value = left[p as usize];

                let wc_a = i32x4::splat(value);
                if wc_a.simd_eq(v_b0).any() || wc_a.simd_eq(v_b1).any() {
                    visitor.visit(value);
                }
            }

            match left[S-1].cmp(&right[S-1]) {
                Ordering::Equal => {
                    left = &left[S..];
                    right = &right[S..];
                    if left.len() < S || right.len() < S {
                        break;
                    }
                    v_a0 = unsafe{ load_unsafe(left.as_ptr()) };
                    v_a1 = unsafe{ load_unsafe(left.as_ptr().add(4)) };
                    v_b0 = unsafe{ load_unsafe(right.as_ptr()) };
                    v_b1 = unsafe{ load_unsafe(right.as_ptr().add(4)) };
                }
                Ordering::Less => {
                    left = &left[S..];
                    if left.len() < S {
                        break;
                    }
                    v_a0 = unsafe{ load_unsafe(left.as_ptr()) };
                    v_a1 = unsafe{ load_unsafe(left.as_ptr().add(4)) };
                },
                Ordering::Greater => {
                    right = &right[S..];
                    if right.len() < S {
                        break;
                    }
                    v_b0 = unsafe{ load_unsafe(right.as_ptr()) };
                    v_b1 = unsafe{ load_unsafe(right.as_ptr().add(4)) };
                },
            }
        }
    }

    branchless_merge(left, right, visitor)
}


#[inline]
fn bmiss_advance<T: Ord>(left: &mut &[T], right: &mut &[T], s: usize) {
    if (*left)[s-1] == (*right)[s-1] {
        *left = &(*left)[s..];
        *right = &right[s..];
    }
    else {
        let lt = (left[s-1] < right[s-1]) as usize;
        *left = &(*left)[s * lt..];
        *right = &(*right)[s * (lt^1)..];
    }
}
