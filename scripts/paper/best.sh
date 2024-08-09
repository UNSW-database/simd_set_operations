#!/usr/bin/env sh

set -e

best() {
    PLATFORM=$1
    DATASET=$2
    EXPERIMENT=$3

    DIR="plots/gcp/$PLATFORM/$DATASET"
    IN="$DIR/$EXPERIMENT.pdf"
    OUT="$DIR/best.txt"

    case $PLATFORM in
        amd) PLAT="AMD";;
        intel) PLAT="Intel";;
        *) echo "Unknown platform: $PLATFORM"; exit 1;;
    esac

    EXP=$(echo $EXPERIMENT | sed -e 's/^compare_//')
    
    open -g -a /Applications/Firefox.app "$IN"

    read -p "$IN: " BEST
    printf "%s: %s\n" "$PLATFORM $EXP" "$BEST" >> "$OUT"
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 PLATFORM"
    exit 1
fi
PLATFORM=$1

for DATASET in $(ls -1 plots/gcp-selectivity/$PLATFORM)
do
    for EXPERIMENT in $(ls -1 plots/gcp-selectivity/$PLATFORM/$DATASET/ | grep ".pdf" | sed -e 's/.pdf//')
    do
        best "$PLATFORM" "$DATASET" "$EXPERIMENT"
    done
done
