#!/usr/bin/env bash

DATASETS=${1-'datasets'}
URL=${2-'http://fimi.uantwerpen.be/data/webdocs.dat.gz'}

wget "$URL" -P "$DATASETS"
gunzip -dkcv "$DATASETS/webdocs.dat.gz" > "$DATASETS/webdocs.dat"
