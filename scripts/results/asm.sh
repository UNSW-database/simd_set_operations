#!/usr/bin/env sh

OUT_DIR="${1:-asm}"
CPU="${2:-native}"


for algo in bmiss_branch_mono bmiss_mono bmiss_sttni_branch_mono bmiss_sttni_mono branchless_merge_mono broadcast_avx2_branch_mono broadcast_avx2_mono broadcast_avx512_branch_mono broadcast_avx512_mono broadcast_sse_branch_mono broadcast_sse_mono naive_merge_mono qfilter_branch_mono qfilter_c_mono qfilter_mono shuffling_avx2_branch_mono shuffling_avx2_mono shuffling_avx512_branch_mono shuffling_avx512_mono shuffling_sse_branch_mono shuffling_sse_mono vp2intersect_emulation_branch_mono vp2intersect_emulation_mono
do
    mkdir -p "$OUT_DIR"
    OUT_FILE="$OUT_DIR/$algo.asm"
    RUSTFLAGS="-C target-cpu=$CPU" cargo asm --release --package=setops --lib "$algo"_mono > "$OUT_FILE" 2> /dev/null
    echo "$OUT_FILE"
done

