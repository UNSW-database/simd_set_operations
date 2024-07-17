#!/usr/bin/env python3
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
    
    # tsc_freq = results["tsc_freq"]
    # tsc_overhead = results["tsc_overhead"]
    data = np.array(results["data"]) / NS
    data_max = np.max(data)
    plot_max = data_max + 0.2

    rng = default_rng()
    choice = rng.choice(len(data), size=20, replace=False)
    for i in choice:
        plot_ensemble(i, data[i], plot_max)


def plot_ensemble(i, ensemble, plot_max):
    x = np.arange(len(ensemble))
    fig, ax = plt.subplots()
    ax.plot(x, ensemble)
    ax.set_ylim(ymin=0, ymax=plot_max)
    ax.set_title(f"Ensemble {i} CPU Frequency")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_xlabel("Trial")
    loc = plticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.grid(visible=True, which="major", axis="y")
    plt.savefig(f"ensemble{i}.png")


if __name__ == "__main__":
    main()
