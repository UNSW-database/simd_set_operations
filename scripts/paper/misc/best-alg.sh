#!/usr/bin/env sh
set -e

# SERVER=tods1
# MERGE=branchless_merge
# SSE_ALGS="bmiss bmiss_sttni qfilter broadcast_sse shuffling_sse"
# AVX2_ALGS="broadcast_avx2 shuffling_avx2"
# AVX512_ALGS="broadcast_avx512 shuffling_avx512 vp2intersect_emulation"
# SIMD_ALGS="$SSE_ALGS $AVX2_ALGS $AVX512_ALGS"

EXP="$1"

./scripts/results/process.py results/paper/$EXP.json processed/$EXP

PROCESSED="processed/$EXP/$(ls -1 processed/$EXP)"

mkdir -p plots/auto

./scripts/results/plot.py $PROCESSED plots/auto/$EXP.pdf \
    --y_vs_x --cols throughput_vs_branchless_merge_lut selectivity

open plots/auto/$EXP.pdf

read -p "What is the best? " BEST
echo $EXP: $BEST >> plots/auto/best.txt
