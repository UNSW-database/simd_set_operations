#!/usr/bin/env python3
import json
import sys
import os
import pathlib
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from numpy.random import default_rng

NS = 1_000_000_000

def main():
    if len(sys.argv) != 2:
        raise ValueError("Script must have exactly 1 argument: the data json file.")
    data_path = pathlib.Path(sys.argv[1])
    with open(data_path, "r") as data_file:
        results = json.load(data_file)

    os.chdir(data_path.parents[0]) 
    
    tsc = results["tsc"]

    raw_error = np.array(tsc["error"])
    cycles = results["cycles"]
    prop_err = raw_error / cycles
    trials = results["trials"]

    data = np.array(results["data"]) / NS
    data_max = np.max(data)
    plot_max = data_max + 0.2

    # rng = default_rng()
    # choice = rng.choice(len(data), size=20, replace=False)
    for i in range(len(data)):
        plot_ensemble(i, data[i], plot_max, prop_err, trials, cycles)


def plot_ensemble(i, ensemble, plot_max, prop_error, trials, cycles):
    fig, ax = plt.subplots()

    fig.set_size_inches(20, 10)

    x = np.arange(len(ensemble))
    ax.plot(x, ensemble)
    plt.fill_between(x, ensemble * (1 - prop_error[0]), ensemble * (1 + prop_error[1]), alpha=0.5)

    ax.set_ylim(ymin=0, ymax=plot_max)
    ax.set_title(f"Ensemble {i} CPU Frequency; {cycles} Cycles; {trials} Trials Median")
    ax.set_ylabel("Frequency (GHz)")
    ax.set_xlabel("Trial")
    loc = plticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    ax.grid(visible=True, which="major", axis="y")

    plt.savefig(f"ensemble{i}.png")
    plt.close()


if __name__ == "__main__":
    main()
