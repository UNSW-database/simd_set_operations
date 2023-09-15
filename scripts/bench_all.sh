#!/usr/bin/env sh

run_bench() {
    RUSTFLAGS="-C target-cpu=native $1" cargo run --release --bin=benchmark -- \
        --out=$2\
        # --datasets=/data/alexb/datasets
}

run_bench "-C target-feature=-avx2 -C target-feature=-avx512f" "results-sse.json"
run_bench "-C target-feature=-avx512f" "results-avx2.json"
run_bench "" "results-avx512.json"
