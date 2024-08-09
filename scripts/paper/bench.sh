#!/usr/bin/env sh

set -e
RUSTFLAGS='-C target-cpu=native' cargo build --release --bin=benchmark

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <platform> <vary>"
    exit 1
fi

platform=$1
vary=$2

OUT=results-$vary-$platform
mkdir -p $OUT

for size in l1 l3 mem
do
    sudo ./target/release/benchmark \
        --out=$OUT/$vary-$platform-$size.json \
        2set_vary_${vary}_${platform}_${size}
    # If datasets are not in the same directory, add flag:
    #     --datasets=$DIR \
done
