#!/usr/bin/env python3
import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt

if "--plotly" in sys.argv:
    pd.options.plotting.backend = "plotly"
    plotly = True
else:
    plotly = False

figsize = (16, 9)

def get_vary_range(info):
    if info["type"] == "synthetic":
        return get_vary_range_synthetic(info)
    else:
        assert(info["type"] == "real")
        return range(
            info["set_count_start"],
            info["set_count_end"] + 1, 1)

def get_vary_range_synthetic(info):
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
    if info["type"] == "synthetic":
        return format_x_synthetic(x, info)
    else:
        return str(x)

def format_x_synthetic(x: int, info) -> str:
    vary = info["vary"]
    if vary in ["selectivity", "density"]:
        return f"{x / 1000 :.2}"
    elif vary == "size":
        return format_size(x)
    elif vary == "skew":
        if info["set_count"] == 2:
            skew = pow(2, x / 1000)
            return f"1:{int(skew)}"
        else:
            return f"f={x / 1000}"
    elif vary == "set_count":
        return str(x)

def format_size(size: int) -> str:
    size = int(size)
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

def format_xlabel(info) -> str:
    if "vary" in info:
        return info["vary"].replace("_", " ")
    else:
        assert(info["type"] == "real")
        return "set count"

def use_log(info) -> bool:
    if info["type"] == "synthetic":
        return info["vary"] in ["skew", "size", "density"]
    else:
        assert(info["type"] == "real")
        return False

def use_bar(info) -> bool:
    if info["type"] == "synthetic":
        return info["vary"] == "set_count"
    else:
        assert(info["type"] == "real")
        return True

def plot_experiment(experiment, results):
    if "algorithm_set" in experiment:
        algorithms = results["algorithm_sets"][experiment["algorithm_set"]]
    else:
        algorithms = experiment["algorithms"]
    dataset = results["datasets"][experiment["dataset"]]
    info = dataset["info"]

    times = {}
    for algorithm in algorithms:
        algorithm_times = []
        for xrec in dataset["algos"][algorithm]:
            if len(xrec["times"]) > 0:
                algorithm_times.append(sum(xrec["times"]) / len(xrec["times"]))
        
        if len(algorithm_times) > 0:
            times[algorithm] = algorithm_times

    df = pd.DataFrame(
        times,
        index=[*get_vary_range(info)]
    )

    if "relative_to" in experiment and experiment["relative_to"] is not None:
        if experiment["relative_to"] in df:
            return plot_experiment_relative(df, info, experiment["relative_to"], experiment["name"])
        else:
            print(f"warn: invalid relative_to {experiment['relative_to']}")

    return plot_experiment_absolute(df, info)

def plot_experiment_absolute(times_df, info):
    if use_bar(info):
        ax = times_df.plot(kind="bar", width=0.8, rot=0, figsize=figsize)
    else:
        ax = times_df.plot(figsize=figsize)
    
    if plotly:
        ax.update_layout(
            xaxis_title=format_xlabel(info),
            yaxis_title="intersection time")

        if use_log(info):
            ax.update_yaxes(type="log")

        ax.show()
        return None
    else:
        ax.set_xlabel(format_xlabel(info))
        ax.set_ylabel("intersection time")

        if use_log(info):
            ax.set_yscale("log")

        if use_bar(info):
            ax.xaxis.set_major_formatter(lambda _, pos: format_x(times_df.index[pos], info))
        else:
            ax.xaxis.set_major_formatter(lambda x, _: format_x(x, info))

        ax.yaxis.set_major_formatter(lambda y, _: format_time(y))
        ax.grid()
        return ax.get_figure()

def plot_experiment_relative(times_df, info, relative_to, name):
    base = times_df[relative_to]

    speed_relative = 1 / times_df.div(base, axis="index")

    if use_bar(info):
        ax = speed_relative.plot(kind="bar", width=0.8, rot=0, figsize=figsize)
    else:
        ax = speed_relative.plot(figsize=figsize)

    if plotly:
        ax.update_layout(
            xaxis_title=format_xlabel(info),
            yaxis_title=f"relative speed ({relative_to})")
        ax.show()
        return None
    else:
        ax.set_xlabel(format_xlabel(info))
        ax.set_ylabel(f"relative speed ({relative_to})")

        if use_bar(info):
            ax.xaxis.set_major_formatter(lambda _, pos: format_x(speed_relative.index[pos], info))
        else:
            ax.xaxis.set_major_formatter(lambda x, _: format_x(x, info))
        ax.grid()

        if name in ["bsr_2set_vary_density",
                    "2set_vary_density",
                    "2set_vary_density_culled",
                    "2set_vary_density_skewed_culled"
                    # ,
                    # "census1881", "census-income",
                    # "weather_sept_85", "wikileaks-noquotes"
                    ]:
            ax.set_yscale("log")

        (y_min, y_max) = ax.get_ylim()
        ax.set_ylim(max(y_min, -1), y_max)

        return ax.get_figure()


def main():
    if len(sys.argv) == 2 and sys.argv[1] != "--plotly":
        results_path = sys.argv[1]
    else:
        results_path = "results.json"
    results_file = open(results_path, "r")
    results = json.loads(results_file.read())

    os.makedirs("plots", exist_ok=True)

    for experiment in results["experiments"]:
        figpath = f"plots/{experiment['name']}.svg"
        print(figpath)

        figure = plot_experiment(experiment, results)
        if not plotly:
            figure.savefig(figpath)
            plt.close(figure)

if __name__ == "__main__":
    main()
