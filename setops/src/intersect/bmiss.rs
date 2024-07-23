#[cfg(feature = "simd")]

use std::{
    simd::{*, cmp::*},
    cmp::Ordering,
};

use crate::{
    visitor::Visitor,
    intersect,
    instructions::{
        load_unsafe,
        BYTE_CHECK_GROUP_A,
        BYTE_CHECK_GROUP_B
    }
};

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

        unsafe { bmiss_advance(&mut left, &mut right, S) };
    }

    intersect::branchless_merge(left, right, visitor)
}

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

        unsafe { bmiss_advance(&mut left, &mut right, S) };
    }

    intersect::branchless_merge(left, right, visitor)
}

pub trait BMiss<T> {
    fn bmiss<const Out: bool>(set_a: &[T], set_b: &[T], out: &mut [T]) -> usize;
}

impl BMiss<i32> for i32 {
    fn bmiss<const Out: bool>(set_a: &[i32], set_b: &[i32], out: &mut [i32]) -> usize {
        std::todo!();
    }
}

impl BMiss<i64> for i64 {
    fn bmiss<const Out: bool>(set_a: &[i64], set_b: &[i64], out: &mut [i64]) -> usize {
        std::todo!();
    }
}

pub fn bmiss_test<T: BMiss<T>, const Out: bool>(set_a: &[T], set_b: &[T], out: &mut [T]) -> usize {
    T::bmiss::<Out>(set_a, set_b, out)
}

pub fn test() {
    {
        let a = vec![1, 2, 3];
        let b = vec![3, 4, 5];
        let mut o = Vec::with_capacity(3);
        bmiss_test::<i32, true>(&a, &b, &mut o);
    }
    {
        let a: Vec<i64> = vec![1, 2, 3];
        let b: Vec<i64> = vec![3, 4, 5];
        let mut o = Vec::with_capacity(3);
        bmiss_test::<i64, true>(&a, &b, &mut o);
    }
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
pub fn bmiss<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    use crate::instructions::convert;

    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    while i_a < st_a && i_b < st_b {
        let v_a: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let v_b: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

        let byte_check_mask0 =
            simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[0])
            .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]));
        let byte_check_mask1 =
            simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[1])
            .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[1]));

        if !(byte_check_mask0 & byte_check_mask1).any() {
            let a_max = unsafe { *set_a.get_unchecked(i_a + W - 1) };
            let b_max = unsafe { *set_b.get_unchecked(i_b + W - 1) };
            
            i_a += W * (a_max <= b_max) as usize;
            i_b += W * (b_max <= a_max) as usize;
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

        let wc_mask0: u64 = word_check_mask0.to_bitmask();
        if (wc_mask0 & 0b0011) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 0) }) }
        if (wc_mask0 & 0b1100) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 1) }) }

        let wc_mask1: u64 = word_check_mask1.to_bitmask();
        if (wc_mask1 & 0b0011) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 2) }) }
        if (wc_mask1 & 0b1100) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 3) }) }

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

#[cfg(feature = "simd")]
const BMISS_STTNI_BC_ARRAY: [u8x16; 2] = [
    u8x16::from_array([0, 1, 4, 5, 8, 9, 12, 13, 255, 255, 255, 255, 255, 255, 255, 255]),
    u8x16::from_array([255, 255, 255, 255, 255, 255, 255, 255, 0, 1, 4, 5, 8, 9, 12, 13]),
];

#[cfg(all(feature = "simd", target_feature = "sse", target_feature = "sse4.2"))]
pub fn bmiss_sttni<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;
    use crate::instructions::shuffle_epi8;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const W: usize = 8;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    while i_a < st_a && i_b < st_b {
        let v_a0: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let v_b0: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        let v_a1: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a + 4)) };
        let v_b1: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b + 4)) };

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

            let value_i32 = unsafe { *ptr_a.add(i_a + p as usize) };

            let wc_a = i32x4::splat(value_i32);
            if wc_a.simd_eq(v_b0).any() || wc_a.simd_eq(v_b1).any() {
                visitor.visit(unsafe { std::mem::transmute_copy(&value_i32) });
            }
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
unsafe fn bmiss_advance<T: Ord>(left: &mut &[T], right: &mut &[T], s: usize) {
    let l = left.get_unchecked(s-1);
    let r = right.get_unchecked(s-1);

    *left = left.get_unchecked(s * (l <= r) as usize..);
    *right = right.get_unchecked(s * (r <= l) as usize..);
}


// Branch
#[cfg(all(feature = "simd", target_feature = "sse"))]
pub fn bmiss_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    use crate::instructions::convert;

    const W: usize = 4;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let mut v_b: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

        loop {
            let byte_check_mask0 =
                simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[0])
                .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[0]));
            let byte_check_mask1 =
                simd_swizzle!(convert::<i32x4, i8x16>(v_a), BYTE_CHECK_GROUP_A[1])
                .simd_eq(simd_swizzle!(convert(v_b), BYTE_CHECK_GROUP_B[1]));

            if !(byte_check_mask0 & byte_check_mask1).any() {
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

            let wc_mask0: u64 = word_check_mask0.to_bitmask();
            if (wc_mask0 & 0b0011) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 0) }) }
            if (wc_mask0 & 0b1100) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 1) }) }

            let wc_mask1: u64 = word_check_mask1.to_bitmask();
            if (wc_mask1 & 0b0011) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 2) }) }
            if (wc_mask1 & 0b1100) != 0 { visitor.visit(unsafe { *set_a.get_unchecked(i_a + 3) }) }

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

#[cfg(all(feature = "simd", target_feature = "sse", target_feature = "sse4.2"))]
pub fn bmiss_sttni_branch<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;
    use crate::instructions::shuffle_epi8;
    #[cfg(target_arch = "x86")]
    use std::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    const W: usize = 8;

    let st_a = (set_a.len() / W) * W;
    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;
    if (i_a < st_a) && (i_b < st_b) {
        let mut v_a0: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
        let mut v_b0: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
        let mut v_a1: i32x4 = unsafe{ load_unsafe(ptr_a.add(i_a + 4)) };
        let mut v_b1: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b + 4)) };

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

                let value_i32 = unsafe { *ptr_a.add(i_a + p as usize) };

                let wc_a = i32x4::splat(value_i32);
                if wc_a.simd_eq(v_b0).any() || wc_a.simd_eq(v_b1).any() {
                    visitor.visit(unsafe { std::mem::transmute_copy(&value_i32) });
                }
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
                    v_a0 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    v_a1 = unsafe{ load_unsafe(ptr_a.add(i_a + 4)) };
                    v_b0 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                    v_b1 = unsafe{ load_unsafe(ptr_b.add(i_b + 4)) };
                },
                Ordering::Less => {
                    i_a += W;
                    if i_a == st_a {
                        break;
                    }
                    v_a0 = unsafe{ load_unsafe(ptr_a.add(i_a)) };
                    v_a1 = unsafe{ load_unsafe(ptr_a.add(i_a + 4)) };
                },
                Ordering::Greater => {
                    i_b += W;
                    if i_b == st_b {
                        break;
                    }
                    v_b0 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
                    v_b1 = unsafe{ load_unsafe(ptr_b.add(i_b + 4)) };
                },
            }
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a..) },
        unsafe { set_b.get_unchecked(i_b..) },
        visitor)
}
