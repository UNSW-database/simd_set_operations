#!/usr/bin/env sh

set -e

process() {
    RUN=$1
    PLATFORM=$2
    EXPERIMENT=$3
    IN="results/$RUN/$PLATFORM/$EXPERIMENT.json"
    OUT="processed/$RUN/$PLATFORM/$EXPERIMENT"

    ./scripts/results/process.py "$IN" "$OUT" > /dev/null
    printf "%-40s ==> %s\n" "$IN" "$OUT"
}

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <run>"
    exit 1
fi

RUN=$1

for PLATFORM in $(ls -1 results/$RUN)
do
    for EXPERIMENT in $(ls -1 results/$RUN/$PLATFORM | sed -e 's/.json//')
    do
        process $RUN $PLATFORM $EXPERIMENT
    done
done
