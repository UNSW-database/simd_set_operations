#!/usr/bin/env python3
'''
Generate AVX-512 kernels
'''

import sys

kernels = []

if len(sys.argv) != 2:
    print("usage: kernel.py [sse|avx2|avx512]")
    exit(1)

simd = sys.argv[1]

if simd == "sse":
    WIDTH = 4
    SHIFT = 3
elif simd == "avx2":
    WIDTH = 8
    SHIFT = 4
elif simd == "avx512":
    WIDTH = 16
    SHIFT = 5
else:
    print(f"unknown simd type {simd}")
    exit(1)

for ctrl in range(1, WIDTH + 1):
    kernels.append((ctrl, WIDTH))
for ctrl in range(1, WIDTH * 2):
    kernels.append((ctrl, WIDTH * 2))

def get_ctrl(left, right):
    return (left << SHIFT) | right

def format_kernel(ctrl, kernel_small, kernel_large, small, large):
    if simd == "sse":
        id_str = f"0o{ctrl:02o}"
    elif simd == "avx2":
        id_str = f"0x{ctrl:02x}"
    elif simd == "avx512":
        id_str = str(ctrl)

    kernel = f"unsafe {{ kernels_{simd}::{simd}_{kernel_small}x{kernel_large}({small}, {large}, visitor) }}"
    return f"{id_str} => {kernel}"

for left in range(1, WIDTH * 2):
    for right in range(1, WIDTH * 2):
        if left <= right:
            small_id = "left"
            small_value = left
            large_id = "right"
            large_value = right
        else:
            small_id = "right"
            small_value = right
            large_id = "left"
            large_value = left
        
        ctrl = get_ctrl(left, right)

        if large_value > WIDTH:
            kernel_large = WIDTH * 2
        else:
            kernel_large = WIDTH

        # print()
        # print(f"left: {left} | right: {right}")
        # print(f"small: {small_id} {small_value} | large: {large_id} {large_value} | kernel_large {kernel_large}")
        print(format_kernel(ctrl, small_value, kernel_large, small_id, large_id))


# for kernel in kernels:
#     small, large_max = kernel
#     large_min = large_max - WIDTH + 1

#     actual_large_max = WIDTH * 2 - 1 if large_max == WIDTH * 2 else large_max

#     lower = get_id(small, large_min)
#     upper = get_id(small, actual_large_max)

#     for ctrl in range(lower, upper+1):
#         print(format_kernel(ctrl, small, large_max))
