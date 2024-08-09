#!/usr/bin/env sh
set -e

SRC=plots/tods-compare-bsr
LIST=$(ls -1 $SRC | grep pdf)

for item in $LIST
do
    open plots/tods-compare-bsr/$item

    read -p "best for $item: " BEST
    echo $item: $BEST >> $SRC/best.txt
done
