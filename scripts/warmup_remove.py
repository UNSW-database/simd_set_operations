import argparse
import pathlib
import json
import os

def main():
    parser = argparse.ArgumentParser(
        description = "Provides removal of initial cold-cache datapoints."
    )
    parser.add_argument("results", help="Path to json results file.")
    parser.add_argument("count", type=int, help="Number of beginning measurements to remove.")
    args = parser.parse_args()

    results_path = pathlib.Path(args.results)
    with open(results_path, "r") as data_file:
        results = json.load(data_file)

    # We output the graphs to the same directory as the input results file
    os.chdir(results_path.parents[0]) 

    # We remove args.count values from the start of every trial
    for experiment_result in results["experiment_results"]:
        for algorithm_result in experiment_result["algorithm_results"]:
            for repeat_result in algorithm_result["repeat_results"]:
                for databin_result in repeat_result["databin_results"]: 
                    if "pair" in databin_result["results"]:
                        for trial_result in databin_result["results"]["pair"]:
                            trial_result['deltas'] = trial_result['deltas'][args.count:]
                    else:
                        raise NotImplementedError("sample")

    cleaned_path = results_path.stem + ".nowarmup.json"
    with open(cleaned_path, "w") as out_file:
        json.dump(results, out_file, separators=(',', ':'))


if __name__ == '__main__':
    main()
