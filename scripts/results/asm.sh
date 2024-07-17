#!/usr/bin/env sh

OUT_DIR="${1:-asm}"
CPU="${2:-native}"
MCA_ARGS="${3}"

asm() {
    ASM_FILE="$OUT_DIR/$1.asm"
    RUSTFLAGS="-C target-cpu=$CPU" cargo asm \
        --release --package=setops --lib \
        --intel --simplify "$1"_mono > "$ASM_FILE" 2> /dev/null
    echo "$ASM_FILE"
}

mca() {
    MCA_FILE="$OUT_DIR/$1.mca"
    RUSTFLAGS="-C target-cpu=$CPU" cargo asm \
        --release --package=setops --lib \
        --mca-intel "$1"_mono $MCA_ARGS > "$MCA_FILE" 2> /dev/null
    echo "$MCA_FILE"
}

# for algo in bmiss bmiss_sttni branchless_merge broadcast_avx2 broadcast_sse naive_merge qfilter shuffling_avx2 shuffling_sse
for algo in bmiss bmiss_sttni branchless_merge broadcast_avx2 broadcast_avx512 broadcast_sse naive_merge qfilter shuffling_avx2 shuffling_avx512 shuffling_sse vp2intersect_emulation
do
    mkdir -p "$OUT_DIR"

    asm ${algo}_lut
    asm ${algo}_br_lut
    asm ${algo}_comp
    asm ${algo}_br_comp

    # mca ${algo}_lut
    # mca ${algo}_br_lut
    # mca ${algo}_comp
    # mca ${algo}_br_comp
done

