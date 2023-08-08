#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import os

def get_vary_range(info):
    vary = info["vary"]
    if vary == "selectivity":
        start = info["selectivity"]
    elif vary == "density":
        start = info["density"]
    elif vary == "size":
        start = info["max_len"]
    elif vary == "skew":
        start = info["skewness_factor"]
    elif vary == "set_count":
        start = info["set_count"]
    
    return range(start, info["to"]+1, info["step"])

def format_x(x: int, info) -> str:
    vary = info["vary"]
    if vary in ["selectivity", "density"]:
        return f"{x / 1000 :.2}"
    elif vary == "size":
        return format_size(x)
    elif vary == "skew":
        if info["set_count"] == 2:
            skew = pow(2, x / 1000)
            return f"1:{skew}"
        else:
            return f"f={x / 1000}"
    elif vary == "set_count":
        return str(x)


def format_size(size: int) -> str:
    if size < 10:
        exp = size
        unit = ""
    elif size < 20:
        exp = size - 10
        unit = "KiB"
    elif size < 30:
        exp = size - 20
        unit = "MiB"
    elif size < 40:
        exp = size - 30
        unit = "GiB"
    else:
        return "Too large"
    return str(1 << exp) + unit

def format_time(nanos: int) -> str:
    if nanos < pow(10, 3):
        value = nanos
        unit = "ns"
    elif nanos < pow(10, 6):
        value = nanos / pow(10, 3)
        unit = "µs"
    elif nanos < pow(10, 9):
        value = nanos / pow(10, 6)
        unit = "ms"
    else:
        value = nanos / pow(10, 9)
        unit = "s"
    return str(value) + unit

def format_xlabel(param: str) -> str:
    return param.replace("_", " ")

def plot_experiment(experiment, results):
    algorithms = results["algorithm_sets"][experiment["algorithm_set"]]
    dataset = results["datasets"][experiment["dataset"]]
    info = dataset["info"]

    times_table = {}
    for algorithm in algorithms:
        times = []
        for xrec in dataset["algos"][algorithm]:
            times.append(sum(xrec["times"]) / len(xrec["times"]))
        times_table[algorithm] = times

    df = pd.DataFrame(
        times_table,
        index=[*get_vary_range(info)]
    )

    ax = df.plot()
    ax.set_xlabel(format_xlabel(info["vary"]))
    ax.set_ylabel("intersection time")

    ax.xaxis.set_major_formatter(lambda x, _: format_x(x, info))
    ax.yaxis.set_major_formatter(lambda y, _: format_time(y))

    return ax.get_figure()

def main():
    results_file = open("results.json", "r")
    results = json.loads(results_file.read())

    os.makedirs("plots", exist_ok=True)

    for experiment in results["experiments"]:
        figure = plot_experiment(experiment, results)
        
        figpath = f"plots/{experiment['name']}.svg"
        print(figpath)
        figure.savefig(figpath)

if __name__ == "__main__":
    main()
