#!/usr/bin/env sh

OUT_DIR="${1:-asm}"
CPU="${2:-native}"


for algo in bmiss_branch bmiss bmiss_sttni_branch bmiss_sttni branchless_merge broadcast_avx2_branch broadcast_avx2 broadcast_avx512_branch broadcast_avx512 broadcast_sse_branch broadcast_sse naive_merge qfilter_branch qfilter_c qfilter shuffling_avx2_branch shuffling_avx2 shuffling_avx512_branch shuffling_avx512 shuffling_sse_branch shuffling_sse vp2intersect_emulation_branch vp2intersect_emulation
do
    mkdir -p "$OUT_DIR"
    OUT_FILE="$OUT_DIR/$algo.asm"
    RUSTFLAGS="-C target-cpu=$CPU" cargo asm --release --package=setops --lib "$algo"_mono > "$OUT_FILE" 2> /dev/null
    echo "$OUT_FILE"
done

