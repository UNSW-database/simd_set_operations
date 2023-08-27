#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import plot

def plot_experiment(experiment, results, reference):
    results_times, info = process_results(experiment, results)
    reference_times, info = process_results(experiment, results)

    reference_times = { k + "_old": v for k, v in reference_times.items() }

    results_df = pd.DataFrame(
        results_times,
        index=[*plot.get_vary_range(info)])

    reference_df = pd.DataFrame(
        reference_times,
        index=[*plot.get_vary_range(info)])

    return plot_comparison_absolute(results_df, reference_df, info)

def process_results(experiment, results):
    algorithms = results["algorithm_sets"][experiment["algorithm_set"]]
    dataset = results["datasets"][experiment["dataset"]]
    info = dataset["info"]

    times = {}
    for algorithm in algorithms:
        algorithm_times = []
        for xrec in dataset["algos"][algorithm]:
            algorithm_times.append(sum(xrec["times"]) / len(xrec["times"]))
        times[algorithm] = algorithm_times
    
    return (times, info)

def plot_comparison_absolute(results_df, reference_df, info):
    fig, ax = plt.subplots()

    ax.plot(results_df, linestyle="solid")
    ax.plot(reference_df, linestyle="dashed")
    
    ax.set_xlabel(plot.format_xlabel(info))
    ax.set_ylabel("intersection time")

    if plot.use_log(info):
        ax.set_yscale("log")

    ax.xaxis.set_major_formatter(lambda x, _: plot.format_x(x, info))
    ax.yaxis.set_major_formatter(lambda y, _: plot.format_time(y))
    ax.grid()

    return fig

def main():
    results_file   = open("results.json", "r")
    reference_file = open("reference.json", "r")
    results   = json.loads(results_file.read())
    reference = json.loads(reference_file.read())

    os.makedirs("plots", exist_ok=True)

    for experiment in results["experiments"]:
        figpath = f"plots/{experiment['name']}.svg"
        print(figpath)

        figure = plot_experiment(experiment, results, reference)
        figure.savefig(figpath)

if __name__ == "__main__":
    main()
