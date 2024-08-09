#!/usr/bin/env sh

set -e

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 INPUT_DIR OUTPUT_DIR"
    exit 1
fi

IN=$1
OUT=$2

mkdir -p $OUT

for algo in 0 10 20 21 40
do
    algo_name=$(sed -n '3p' "$IN/$algo-sel0.0")
    dest="$OUT/$algo_name.csv"

    echo "Selectivity,Throughput_epus,Runtime_us" > "$dest"
    for sel in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        src="$IN/$algo-sel$sel"
        throughput=$(sed -rn 's/^ele_per_usec=([^\s]+) run_time=.*$/\1/g;5p' $src)
        runtime=$(sed -rn 's/^ele_per_usec=[^\s]+ run_time=(.*)(ms|us|ns)$/\1/g;5p' $src)
        echo "$sel,$throughput,$runtime" >> "$dest"
    done
    echo "$dest"
done
