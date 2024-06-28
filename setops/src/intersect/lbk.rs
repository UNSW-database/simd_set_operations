#![cfg(feature = "simd")]

use std::{
    simd::*,
    simd::cmp::*,
};

use crate::{
    visitor::Visitor,
    intersect, instructions::load_unsafe,
};

#[cfg(target_feature = "ssse3")]
pub fn lbk_v1x4_sse<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 4;

    let st_b = (set_b.len() / W) * W;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + W - 1) } < target {
                i_b += W;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x4::splat(target_i32);
            let v_b: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
            let mask = v_a.simd_eq(v_b);
            if mask.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "ssse3")]
pub fn lbk_v1x8_sse<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 4;
    const BOUND: usize = W*2;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x4::splat(target_i32);

            let v_b1: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
            let v_b2: i32x4 = unsafe{ load_unsafe(ptr_b.add(i_b + W)) };

            let mask1 = v_a.simd_eq(v_b1);
            let mask2 = v_a.simd_eq(v_b2);
            if mask1.any() || mask2.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}


#[cfg(target_feature = "ssse3")]
pub fn lbk_v1x8_avx2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 8;
    const BOUND: usize = W;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x8::splat(target_i32);
            let v_b: i32x8 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

            let mask = v_a.simd_eq(v_b);
            if mask.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn lbk_v1x16_avx2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 8;
    const BOUND: usize = 2*W;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x8::splat(target_i32);
            let v_b1: i32x8 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
            let v_b2: i32x8 = unsafe{ load_unsafe(ptr_b.add(i_b + W)) };

            let mask1 = v_a.simd_eq(v_b1);
            let mask2 = v_a.simd_eq(v_b2);
            if mask1.any() || mask2.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "avx512f")]
pub fn lbk_v1x16_avx512<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 16;
    const BOUND: usize = W;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x16::splat(target_i32);
            let v_b: i32x16 = unsafe{ load_unsafe(ptr_b.add(i_b)) };

            let mask = v_a.simd_eq(v_b);
            if mask.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "avx512f")]
pub fn lbk_v1x32_avx512<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 16;
    const BOUND: usize = 2*W;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }
            let v_a = i32x16::splat(target_i32);
            let v_b1: i32x16 = unsafe{ load_unsafe(ptr_b.add(i_b)) };
            let v_b2: i32x16 = unsafe{ load_unsafe(ptr_b.add(i_b + W)) };

            let mask1 = v_a.simd_eq(v_b1);
            let mask2 = v_a.simd_eq(v_b2);
            if mask1.any() || mask2.any() {
                visitor.visit(*target);
            }
            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}


const NUM_LANES_IN_BOUND: usize = 32;

#[cfg(target_feature = "ssse3")]
pub fn lbk_v3_sse<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 4;
    const BOUND: usize = W*NUM_LANES_IN_BOUND;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }

            let inner_offset: usize = reduce_search_bound(*target, &set_b[i_b..], BOUND);
            let result = block_compare::<i32, W>(target_i32, inner_offset, unsafe{ ptr_b.add(i_b) });

            if result.any() {
                visitor.visit(*target);
            }

            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "avx2")]
pub fn lbk_v3_avx2<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 8;
    const BOUND: usize = W*NUM_LANES_IN_BOUND;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }

            let inner_offset: usize = reduce_search_bound(*target, &set_b[i_b..], BOUND);
            let result = block_compare::<i32, W>(target_i32, inner_offset, unsafe{ ptr_b.add(i_b) });

            if result.any() {
                visitor.visit(*target);
            }

            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}

#[cfg(target_feature = "avx512f")]
pub fn lbk_v3_avx512<T, V>(set_a: &[T], set_b: &[T], visitor: &mut V)
where
    V: Visitor<T>,
    T: Ord + Copy + std::fmt::Display,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<i32>());
    let ptr_a = set_a.as_ptr() as *const i32;
    let ptr_b = set_b.as_ptr() as *const i32;

    const W: usize = 16;
    const BOUND: usize = W*NUM_LANES_IN_BOUND;

    let st_b = (set_b.len() / BOUND) * BOUND;

    let mut i_a: usize = 0;
    let mut i_b: usize = 0;

    if i_b < st_b {
        'outer:
        while i_a < set_a.len() {
            let target = unsafe { set_a.get_unchecked(i_a) };
            let target_i32 = unsafe{ *ptr_a.add(i_a) };
            
            while unsafe { set_b.get_unchecked(i_b + BOUND - 1) } < target {
                i_b += BOUND;
                if i_b >= st_b {
                    break 'outer;
                }
            }

            let inner_offset: usize = reduce_search_bound(*target, &set_b[i_b..], BOUND);
            let result = block_compare::<i32, W>(target_i32, inner_offset, unsafe{ ptr_b.add(i_b) });

            if result.any() {
                visitor.visit(*target);
            }

            i_a += 1;
        }
    }

    intersect::branchless_merge(
        unsafe { set_a.get_unchecked(i_a.min(set_a.len())..) },
        unsafe { set_b.get_unchecked(i_b.min(set_b.len())..) },
        visitor)
}


#[inline]
fn reduce_search_bound<T>(target: T, large: &[T], bound: usize) -> usize
where
    T: Ord,
{
    if large[bound / 2 - 1] >= target {
        if large[bound / 4 - 1] < target {
            NUM_LANES_IN_BOUND / 4
        }
        else {
            0
        }
    }
    else if large[bound * 3 / 4 - 1] < target {
        NUM_LANES_IN_BOUND * 3 / 4
    }
    else {
        NUM_LANES_IN_BOUND / 2
    }
}

#[inline]
fn block_compare<T, const LANES: usize>(
    target: T,
    inner_offset: usize,
    large: *const T) -> Mask<T, LANES>
where
    T: SimdElement + MaskElement + PartialOrd,
    LaneCount<LANES>: SupportedLaneCount,
    Simd<T, LANES>: SimdPartialEq<Mask=Mask<T, LANES>>,
{
    let target_vec = Simd::<T, LANES>::splat(target);
    let qs = [
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset    ))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 1))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 2))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 3))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 4))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 5))) }),
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 6))) }) |
        target_vec.simd_eq(unsafe { load_unsafe(large.add(LANES * (inner_offset + 7))) })
    ];
    (qs[0] | qs[1]) | (qs[2] | qs[3])
}
