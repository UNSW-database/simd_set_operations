#!/usr/bin/env bash

DATASETS=${1-'datasets'}
TWITTER='https://snap.stanford.edu/data/twitter_combined.txt.gz'
SKITTER='https://snap.stanford.edu/data/as-skitter.txt.gz'

wget -nc "$TWITTER" -P "$DATASETS"
wget -nc "$SKITTER" -P "$DATASETS"
gunzip -dkcv "$DATASETS/twitter_combined.txt.gz" | $(dirname "$0")/process_graph.py > "$DATASETS/twitter.dat"
gunzip -dkcv "$DATASETS/as-skitter.txt.gz" | $(dirname "$0")/process_graph.py > "$DATASETS/as-skitter.dat"
