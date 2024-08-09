#!/usr/bin/env sh

set -e

plot() {
    RUN=$1
    PLATFORM=$2
    DATASET=$3
    EXPERIMENT=$4

    IN="processed/$RUN/$PLATFORM/$DATASET/"
    OUT_DIR="plots/$RUN/$PLATFORM"
    OUT="$OUT_DIR/$DATASET.pdf"

    # case $DATASET in
    #     *lowd*) VARY="selectivity";;
    #     *highd*) VARY="density";;
    #     *) echo "Unknown vary for dataset: $DATASET"; exit 1;;
    # esac
    VARY=skewness_factor

    mkdir -p $OUT_DIR

    # ./scripts/results/plot.py "$IN" "$OUT" \
    #     --y_vs_x --cols throughput_vs_branchless_merge_lut $VARY &

    ./scripts/results/plot.py "$IN" "$OUT" --y_vs_x --cols throughput_vs_branchless_merge_lut $VARY &
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <run>"
    exit 1
fi

RUN=$1

for PLATFORM in $(ls -1 processed/$RUN/)
do
    for DATASET in $(ls -1 processed/$RUN/$PLATFORM)
    do
        plot "$RUN" "$PLATFORM" "$DATASET"
        # for EXPERIMENT in $(ls -1 processed/$RUN/$PLATFORM/$DATASET)
        # do
        #     plot "$RUN" "$PLATFORM" "$DATASET" "$EXPERIMENT"
        # done
    done
done

wait $(jobs -p)
