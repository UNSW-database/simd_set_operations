use crate::intersect::*;


pub fn naive_merge_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    naive_merge(set_a, set_b, visitor);
}
    

pub fn branchless_merge_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    branchless_merge(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn shuffling_sse_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    shuffling_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn shuffling_avx2_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    shuffling_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    shuffling_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn broadcast_sse_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    broadcast_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn broadcast_avx2_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    broadcast_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    broadcast_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    bmiss(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_sttni_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    bmiss_sttni(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    qfilter(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{
    vp2intersect_emulation(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_c_mono(set_a: &[i32], set_b: &[i32], set_c: &mut [i32]) -> usize
{
    qfilter_c(set_a, set_b, set_c)
}
    
