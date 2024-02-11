#!/usr/bin/env python3
from yattag import Doc
import json
import os
import sys
import tomllib

doc, tag, text = Doc().tagtext()

if len(sys.argv) >= 2:
    results_path = sys.argv[1]
else:
    results_path = "results.json"

toml_path = None
if len(sys.argv) >= 3:
    toml_path = sys.argv[2]

results_file = open(results_path, "r")
results = json.loads(results_file.read())

os.makedirs("plots", exist_ok=True)

if toml_path:
    toml_file = open(toml_path, "rb")
    toml_results = tomllib.load(toml_file)
    experiments = toml_results["experiment"]
else:
    experiments = results["experiments"]

datasets = results["datasets"]

os.makedirs("plots", exist_ok=True)

def write_experiment(experiment):
    title = experiment["title"]
    name = experiment["name"]
    with tag("h2"):
        text(title)
    with tag("div"):
        with tag("i"):
            text(name)
    doc.stag("img", src=f"{name}.pdf")

def write_field(name, value):
    with tag("div"):
        with tag("b"):
            text(name + ": ")
        with tag("span"):
            text(value)

def write_dataset(dataset):
    name = dataset["name"]
    dataset_type = dataset["type"]
    with tag("h2"):
        text(name)
    
    write_field("type", dataset_type)
    if dataset_type == "synthetic":
        write_field("vary", dataset["vary"])
        write_field("to", dataset["to"])
        write_field("step", dataset["step"])
        write_field("gen_count", dataset["gen_count"])

        set_count = dataset["set_count"]
        write_field("set_count", set_count)
        write_field("density", dataset["density"] / 1000)
        write_field("selectivity", dataset["selectivity"] / 1000)
        
        max_len = dataset["max_len"]
        max_len_count = 2 ** max_len
        write_field("max_len", f"2^{max_len} elements ({max_len_count})")
        write_field("skewness_factor", dataset["skewness_factor"])
    else:
        write_field("gen_count", dataset["gen_count"])
        write_field("set_count_start", dataset["set_count_start"])
        write_field("set_count_end", dataset["set_count_end"])

with tag("html"):
    with tag("body"):
        with tag("h1"):
            text("Experiments")
        
        for experiment in experiments:
            write_experiment(experiment)
        
        with tag("h1"):
            text("Datasets")
        for dataset in datasets.values():
            write_dataset(dataset["info"])

print(doc.getvalue())
