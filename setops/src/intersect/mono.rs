use crate::intersect::*;

use crate::visitor::*;


pub fn naive_merge_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    naive_merge(set_a, set_b, visitor);
}
    

pub fn naive_merge_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    naive_merge(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn naive_merge_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    naive_merge(set_a, set_b, visitor);
}
    

pub fn branchless_merge_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    branchless_merge(set_a, set_b, visitor);
}
    

pub fn branchless_merge_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    branchless_merge(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn branchless_merge_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    branchless_merge(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn shuffling_sse_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn shuffling_sse_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn shuffling_sse_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn shuffling_sse_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_sse_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_sse_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn shuffling_avx2_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn shuffling_avx2_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn shuffling_avx2_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn shuffling_avx2_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx2_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx2_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    shuffling_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    shuffling_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn shuffling_avx512_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    shuffling_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn broadcast_sse_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn broadcast_sse_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn broadcast_sse_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn broadcast_sse_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_sse_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_sse(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_sse_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_sse_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn broadcast_avx2_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn broadcast_avx2_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn broadcast_avx2_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx2"))]
pub fn broadcast_avx2_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx2_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_avx2(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx2_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_avx2_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    broadcast_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    broadcast_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_avx512(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn broadcast_avx512_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    broadcast_avx512_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    bmiss(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    bmiss_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    bmiss(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    bmiss_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn bmiss_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    bmiss(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn bmiss_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    bmiss_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_sttni_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    bmiss_sttni(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_sttni_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    bmiss_sttni_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_sttni_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    bmiss_sttni(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn bmiss_sttni_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    bmiss_sttni_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn bmiss_sttni_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    bmiss_sttni(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn bmiss_sttni_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    bmiss_sttni_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    qfilter(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    qfilter_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    qfilter(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    qfilter_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn qfilter_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    qfilter(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn qfilter_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    qfilter_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    vp2intersect_emulation(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_br_count_mono(set_a: &[i32], set_b: &[i32], visitor: &mut Counter)
{
    vp2intersect_emulation_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    vp2intersect_emulation(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_br_lut_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeLookupWriter<i32>)
{
    vp2intersect_emulation_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    vp2intersect_emulation(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "avx512f"))]
pub fn vp2intersect_emulation_br_comp_mono(set_a: &[i32], set_b: &[i32], visitor: &mut UnsafeCompressWriter<i32>)
{
    vp2intersect_emulation_branch(set_a, set_b, visitor);
}
    
#[cfg(all(feature = "simd", target_feature = "ssse3"))]
pub fn qfilter_c_mono(set_a: &[i32], set_b: &[i32], set_c: &mut [i32]) -> usize
{
    qfilter_c(set_a, set_b, set_c)
}
    
