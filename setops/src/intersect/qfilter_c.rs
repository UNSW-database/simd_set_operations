#![allow(dead_code)]
include!(concat!(env!("OUT_DIR"), "/qfilter_c.rs"));
use libc::c_int;

#[cfg(target_feature = "ssse3")]
pub fn qfilter_c<T>(set_a: &[T], set_b: &[T], result: &mut [T]) -> usize
where
    T: Ord + Copy,
{
    assert!(std::mem::size_of::<T>() == std::mem::size_of::<c_int>());
    assert!(result.len() >= set_a.len().min(set_b.len()));
    let ptr_a = set_a.as_ptr() as *const c_int;
    let ptr_b = set_b.as_ptr() as *const c_int;
    let ptr_result = result.as_ptr() as *mut c_int;

    unsafe {
        intersect_qfilter_uint_b4_v2(
            ptr_a, set_a.len() as c_int,
            ptr_b, set_b.len() as c_int,
            ptr_result) as usize
    }
}
