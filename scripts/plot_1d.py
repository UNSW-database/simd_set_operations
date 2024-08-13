import argparse
import pathlib
import json
import os
from itertools import product
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

NS = 1_000_000_000
US = 1_000_000
MS = 1_000

VARIABLES = {"skew", "density", "selectivity", "size", "datatype", "distribution"}

def main():
    parser = argparse.ArgumentParser(
        description = "Graphs relative performance of algorithms along for a \
            single variable.",
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("description", help="Path to databin description.")
    parser.add_argument("variable", help="Variable to graph.")
    parser.add_argument("reference", help="Reference algorithm.")
    args = parser.parse_args()

    if args.variable not in VARIABLES:
        raise ValueError(f"\"{args.variable}\" is not a valid variable.")

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    if args.reference not in algorithms(results):
        raise ValueError(f"\"{args.reference}\" is not a valid algorithm.")

    description_path = pathlib.Path(args.description)
    with open(description_path, "r") as description_file:
        description = json.load(description_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    # TODO
    # NB: Data cleaning should be done separately
    # For the selected variable, create bins of databins that only vary in that variable
    # Graph each bin

    bins = databin_bins(description, args.variable)

    plot(bins, results, args.variable)


def algorithms(results):
    names_lists = [[algorithm["algorithm_name"] for algorithm in experiment["algorithm_results"]] for experiment in results["experiment_results"]]
    return {name for names in names_lists for name in names}


def databin_bins(description, variable):
    datatype = {}
    max_value = {}
    max_length = {}
    min_length = {}
    intersection_length = {}
    distribution = {}
    trials = {}

    for i, databin in enumerate(description):
        datatype.setdefault(databin["datatype"], set()).add(i)
        max_value.setdefault(databin["max_value"], set()).add(i)
        if "set_lengths" in databin["lengths"]:
            max_length.setdefault(databin["lengths"]["set_lengths"][0], set()).add(i)
            min_length.setdefault(databin["lengths"]["set_lengths"][1], set()).add(i)
        else:
            raise NotImplementedError("samples")
        intersection_length.setdefault(databin["lengths"]["intersection_length"], set()).add(i)
        distribution.setdefault(databin["distribution"]["type"], set()).add(i)
        trials.setdefault(databin["trials"], set()).add(i)

    datatype = list(datatype.values())
    max_value = list(max_value.values())
    max_length = list(max_length.values())
    min_length = list(min_length.values())
    intersection_length = list(intersection_length.values())
    distribution = list(distribution.values())
    trials = list(trials.values())
    
    match variable:
        case "selectivity":
            i = datatype
            i = [x & y for (x, y) in product(i, max_value)]
            i = [x & y for (x, y) in product(i, max_length)]
            i = [x & y for (x, y) in product(i, min_length)]
            i = [x & y for (x, y) in product(i, distribution)]
            i = [x & y for (x, y) in product(i, trials)]
        case _:
            raise NotImplementedError(f"{variable}")

    ret = [sorted(list(v)) for v in i]
    return ret


def bins_lists_to_data(bins_lists, results):
    experiment_data = {}
    for experiment in results["experiment_results"]:
        for algorithm in experiment["algorithm_results"]:
            bins_lists_results = []
            databins = merge_repeats(algorithm("repeat_results"))["databin_results"]
            for bins in bins_lists:
                bins_results = []  
                for bin_index in bins:
                    bins_results.append(databins[bin_index]["deltas"])
                bins_lists_results.append({
                    "experiment": experiment["experiment_name"],
                    "algorithm": algorithm["algorithm_name"],
                    "deltas": bins_results,
                })


def merge_repeats(repeats):
    merged = deepcopy(repeats[0])
    for repeat in repeats[1:]:
        for (databin, acc) in zip(repeat["databin_results"], merged["databin_results"]):
            results_type = databin["results"]
            if "pair" in results_type:
                for (trials, acc) in zip(results_type["pair"], merged["results"]["pair"]):
                    acc["deltas"].extend(trials["deltas"])
    return merged




if __name__ == "__main__":
    main()