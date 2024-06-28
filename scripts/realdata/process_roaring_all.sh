#!/usr/bin/env sh

for d in datasets/roaring/*_srt/; do
    dname="$(basename $d _srt)"
    echo $dname
    ./scripts/process_roaring.py "$d" > "datasets/$dname.dat"
done
