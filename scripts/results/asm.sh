#!/usr/bin/env sh

OUT_DIR="${1:-asm}"
CPU="${2:-native}"

for algo in bmiss bmiss_sttni branchless_merge broadcast_avx2 broadcast_avx512 broadcast_sse naive_merge qfilter_c qfilter shuffling_avx2 shuffling_avx512 shuffling_sse vp2intersect_emulation
do
    mkdir -p "$OUT_DIR"
    OUT_FILE="$OUT_DIR/$algo.asm"
    RUSTFLAGS="-C target-cpu=$CPU" cargo asm --release --package=setops --lib "$algo"_mono > "$OUT_FILE" 2> /dev/null
    echo "$OUT_FILE"
done

