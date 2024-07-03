#!/usr/bin/env python3
'''
Generates function monomorphisations for assembly analysis.
Usage: scripts/codegen/mono.py > setops/src/intersect/mono.rs
'''

SHUF_ALGOS = [ "shuffling", "broadcast" ]
SIMD_WIDTHS = [("sse", "ssse3"), ("avx2", "avx2"), ("avx512", "avx512f") ]

SCALAR_ALGOS = [ "naive_merge", "branchless_merge" ]
SSE_ALGOS = [ "bmiss", "bmiss_sttni", "qfilter" ]
AVX512_ALGOS = [ "vp2intersect_emulation" ]

def print_algo(name, simd_feature):
    guard = f'#[cfg(all(feature = "simd", target_feature = "{simd_feature}"))]' if simd_feature else ""
    print(f'''{guard}
pub fn {name}_mono(set_a: &[i32], set_b: &[i32], visitor: &mut VecWriter<i32>)
{{
    {name}(set_a, set_b, visitor);
}}
    ''')

def print_c_algo(name, simd_feature):
    guard = f'#[cfg(all(feature = "simd", target_feature = "{simd_feature}"))]' if simd_feature else ""
    print(f'''{guard}
pub fn {name}_mono(set_a: &[i32], set_b: &[i32], set_c: &mut [i32]) -> usize
{{
    {name}(set_a, set_b, set_c)
}}
    ''')

print("use crate::intersect::*;\n")

for algo in SCALAR_ALGOS:
    print_algo(algo, None)

for algo in SHUF_ALGOS:
    for width, feature in SIMD_WIDTHS:
        print_algo(f"{algo}_{width}", feature)
        print_algo(f"{algo}_{width}_branch", feature)

for algo in SSE_ALGOS:
    print_algo(algo, "ssse3")
    print_algo(algo + "_branch", "ssse3")

for algo in AVX512_ALGOS:
    print_algo(algo, "avx512f")
    print_algo(algo + "_branch", "avx512f")

print_c_algo("qfilter_c", "ssse3")

