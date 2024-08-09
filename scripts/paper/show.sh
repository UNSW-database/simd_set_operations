#!/usr/bin/env sh
set -e

# for alg in bmiss bmiss_sttni qfilter shuffling_sse broadcast_sse 
# for alg in broadcast_avx2 shuffling_avx2
for alg in broadcast_avx512 shuffling_avx512 vp2intersect_emulation
do
    open plots/tods-newcompare-mem/compare-sel-mem-$alg.pdf
done
