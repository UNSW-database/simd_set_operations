#!/usr/bin/env sh

run_bench() {
    RUSTFLAGS="-C target-cpu=$1" ROARING_ARCH="$1" cargo run --release --bin=benchmark -- \
        --out=$2\
        --datasets=/data/alexb/datasets
}

run_bench "nehalem" "results-sse.json"
run_bench "broadwell" "results-avx2.json"
run_bench "native" "results-avx512.json"
