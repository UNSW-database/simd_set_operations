#!/usr/bin/env python3
'''
Generate AVX-512 kernels
'''

kernels = []

for i in range(1, 16):
    kernels.append((i, 16))
for i in range(1, 32):
    kernels.append((i, 32))

def get_id(small, large):
    return (small << 5) | large

def format_kernel(lower_id, upper_id, kernel_small, kernel_large):
    kernel = f"unsafe {{ kernels_avx512::avx512_{kernel_small}x{kernel_large}(small_ptr, large_ptr, visitor) }}"
    return f"{lower_id}..={upper_id} => {kernel}"

for kernel in kernels:
    small, large_max = kernel
    large_min = large_max - 16 + 1

    actual_large_max = 31 if large_max == 32 else large_max

    lower = get_id(small, large_min)
    upper = get_id(small, actual_large_max)

    print(format_kernel(lower, upper, small, large_max))
