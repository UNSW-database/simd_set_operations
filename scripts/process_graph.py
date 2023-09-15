#!/usr/bin/env python3
import sys

if len(sys.argv) <= 1:
    file = sys.stdin
else:
    file = open(sys.argv[1], "r")

adj = {}

for line in file:
    if line[0] == "#":
        continue
    segments = line.split()
    start = int(segments[0])
    end = int(segments[1])
    adj.setdefault(start, []).append(end)

adj_lists = []

for adj_list in adj.values():
    adj_list_dedup = [str(i) for i in sorted(set(adj_list))]
    adj_lists.append(adj_list_dedup)

adj_lists.sort(key=lambda l: len(l), reverse=True)

for adj_list in adj_lists:
    print(' '.join(adj_list))
