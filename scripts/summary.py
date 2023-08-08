#!/usr/bin/env python3
from yattag import Doc
import json
import os

doc, tag, text = Doc().tagtext()

results_file = open("results.json", "r")
results = json.loads(results_file.read())

os.makedirs("plots", exist_ok=True)

with tag("html"):
    with tag("body"):
        for experiment in results["experiments"]:
            name = experiment["name"]
            with tag("h1"):
                text(name)
            doc.stag("img", src=f"{name}.svg")

print(doc.getvalue())
