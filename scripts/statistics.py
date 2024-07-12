#!/usr/bin/env python3
import json
import sys
import matplotlib.pyplot as plt
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
    data = np.array(results["data"]) / (tsc_freq / NS)

    rng = default_rng()
    choice = rng.choice(len(data), size=20, replace=False)
    for i in choice:
        plot_ensemble(i, data[i])


def plot_ensemble(i, ensemble):
    x = np.arange(len(ensemble))
    fig, ax = plt.subplots()
    ax.plot(x, ensemble)
    ax.set_ylim(ymin=0)
    ax.set_title(f"Ensemble {i} Runtime")
    ax.set_ylabel("Time (ns)")
    ax.set_xlabel("Trial")
    plt.savefig(f"ensemble{i}.png")


if __name__ == "__main__":
    main()
