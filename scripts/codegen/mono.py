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

VISITORS = [
    ("count", "Counter"),
    ("lut", "UnsafeLookupWriter<i32>"),
    ("comp", "UnsafeCompressWriter<i32>"),
];

def print_algo(name, func, simd_feature, visitor):
    guard = f'#[cfg(all(feature = "simd", target_feature = "{simd_feature}"))]' if simd_feature else ""
    print(f'''{guard}
pub fn {name}_mono(set_a: &[i32], set_b: &[i32], visitor: &mut {visitor})
{{
    {func}(set_a, set_b, visitor);
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
print("use crate::visitor::*;\n")

for algo in SCALAR_ALGOS:
    for (vname, vtype) in VISITORS:
        simd_req = "avx512f" if vname == "comp" else None
        print_algo(f"{algo}_{vname}", algo, simd_req, vtype)

for algo in SHUF_ALGOS:
    for width, feature in SIMD_WIDTHS:
        func = f"{algo}_{width}"
        for (vname, vtype) in VISITORS:
            simd_req = "avx512f" if vname == "comp" else feature
            print_algo(f"{func}_{vname}", func, simd_req, vtype)
            print_algo(f"{func}_br_{vname}", func + "_branch", simd_req, vtype)

for algo in SSE_ALGOS:
    for (vname, vtype) in VISITORS:
        simd_req = "avx512f" if vname == "comp" else "ssse3"
        print_algo(f"{algo}_{vname}", algo, simd_req, vtype)
        print_algo(f"{algo}_br_{vname}", algo + "_branch", simd_req, vtype)

for algo in AVX512_ALGOS:
    for (vname, vtype) in VISITORS:
        print_algo(f"{algo}_{vname}", algo, "avx512f", vtype)
        print_algo(f"{algo}_br_{vname}", algo + "_branch", "avx512f", vtype)

print_c_algo("qfilter_c", "ssse3")

