#!/usr/bin/env sh

wget -nc https://github.com/RoaringBitmap/CRoaring/archive/refs/heads/master.zip -P datasets/roaring

unzip datasets/roaring/master.zip CRoaring-master/benchmarks/realdata/* -d datasets/roaring

mv datasets/roaring/CRoaring-master/benchmarks/realdata/* datasets/roaring
rm -r datasets/roaring/CRoaring-master
