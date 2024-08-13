#!/bin/python3
import json
import sys
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from numpy.random import default_rng

NS = 1_000_000_000

def main():
    if len(sys.argv) != 2:
        raise ValueError("Script must have exactly 1 argument: the data json file.")
    data_path = sys.argv[1]
    with open(data_path) as data_file:
        results = json.load(data_file)
    
    tsc_freq = results["tsc_freq"]
    tsc_overhead = results["tsc_overhead"]
    print(f"tsc_freq[{tsc_freq}], tsc_overhead[{tsc_overhead}]")

    data = np.array(results["data"]) / NS

    median = np.median(data)
    print(f"median[{median}]")

    dist = abs(data - median)
    mean_dist = np.mean(dist)
    max_dist = np.max(dist)
    print(f"mean_dist[{mean_dist}], max_dist[{max_dist}]")

    print(data[abs(data - median) > 0.05])


if __name__ == "__main__":
    main()

