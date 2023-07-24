use std::ops::BitOr;


#[inline]
pub fn slice_i32_to_u32(slice_i32: &[i32]) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            slice_i32.as_ptr() as *const u32, slice_i32.len()
        )
    }
}

#[inline]
pub fn or_8<T: BitOr<T, Output=T> + Copy>(v: [T; 8]) -> T {
    or_4(or_8_to_4(v))
}

#[inline]
pub fn or_4<T: BitOr<T, Output=T> + Copy>(v: [T; 4]) -> T {
    or_2(or_4_to_2(v))
}

#[inline]
fn or_8_to_4<T: BitOr<T, Output=T> + Copy>(v: [T; 8]) -> [T; 4] {
    [
        v[0] | v[1],
        v[2] | v[3],
        v[4] | v[5],
        v[6] | v[7],
    ]
}

#[inline]
fn or_4_to_2<T: BitOr<T, Output=T> + Copy>(v: [T; 4]) -> [T; 2] {
    [v[0] | v[1], v[2] | v[3]]
}

#[inline]
fn or_2<T: BitOr<T, Output=T> + Copy>(v: [T; 2]) -> T {
    v[0] | v[1]
}
