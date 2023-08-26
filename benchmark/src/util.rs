
pub fn slice_i32_to_u32(slice_i32: &[i32]) -> &[u32] {
    unsafe {
        std::slice::from_raw_parts(
            slice_i32.as_ptr() as *const u32, slice_i32.len()
        )
    }
}

pub fn slice_u32_to_i32(slice_u32: &[u32]) -> &[i32] {
    unsafe {
        std::slice::from_raw_parts(
            slice_u32.as_ptr() as *const i32, slice_u32.len()
        )
    }
}
