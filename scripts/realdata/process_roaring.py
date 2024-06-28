#!/usr/bin/env python3
import sys
import os

if len(sys.argv) != 2:
    print("usage: ./process_roaring.py <dir>")
    exit(1)

dir = os.scandir(sys.argv[1])

lengths = []

for setpath in dir:
    file = open(setpath)
    for line in file:
        segments = line.split(',')
        lengths.append(len(segments))
        out = ' '.join(segments).strip()
        if out != '':
            print(out)

avg_len = sum(lengths) / len(lengths)
print("avg len: ", avg_len, file=sys.stderr)
print("# sets: ", len(lengths), file=sys.stderr)
