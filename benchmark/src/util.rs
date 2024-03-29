
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

// Conversion of integers to arrays of bytes and vectors of integers to vectors of bytes, all native orders
pub trait Byteable<const N: usize> {
    fn to_bytes(&self) -> [u8; N];
}

macro_rules! byteable_int {
    ( $( $t:ty ),* ) => {
        $(
impl Byteable<{std::mem::size_of::<$t>()}> for $t {
    fn to_bytes(&self) -> [u8; std::mem::size_of::<$t>()] {
        self.to_ne_bytes()
    }
}
        )*
    }
}

byteable_int!{u8, u16, u32, u64, i8, i16, i32, i64, usize}

pub fn vec_to_bytes<const N: usize, T: Byteable<N>>(vec: &Vec<T>) -> Vec<u8> {
    vec.iter().flat_map(|i| i.to_bytes()).collect()
}

// Trait that allows you to access the maximum value for a type
pub trait Max {
    fn max() -> Self;
}

macro_rules! max_int {
    ( $( $t:ty ), *) => {
        $(
impl Max for $t {
    fn max() -> Self {
        Self::MAX
    }
}
        )*
    }
}

max_int!{u8, u16, u32, u64, i8, i16, i32, i64, usize}
