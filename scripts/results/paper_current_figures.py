#!/usr/bin/env python3
import json
import math
import statistics as st
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "linux"
PAPER_IMAGES = ROOT / "68cd21f53eb78be3a611be2e" / "images"


def load_json(path: Path):
    return json.loads(path.read_text())


def median_time(entry):
    return st.median(entry["times"])


def gmean(values):
    return math.exp(sum(math.log(v) for v in values) / len(values))


def dataset_algos(data, dataset_name):
    return data["datasets"][dataset_name]["algos"]


def save_fig(fig, name: str):
    out = PAPER_IMAGES / name
    fig.savefig(out, bbox_inches="tight")
    print(out)


def short_name(alg: str):
    names = {
        "shuffling_sse": "Shuffle SSE",
        "broadcast_sse": "Broadcast SSE",
        "shuffling_avx2": "Shuffle AVX2",
        "broadcast_avx2": "Broadcast AVX2",
        "shuffling_avx512": "Shuffle AVX-512",
        "broadcast_avx512": "Broadcast AVX-512",
        "vp2intersect_emulation": "VP2 Emul.",
        "conflict_intersect": "Conflict",
        "branchless_merge": "Merge",
        "qfilter": "QFilter",
        "qfilter_branch": "QFilter (BR)",
        "bmiss": "BMiss",
        "bmiss_branch": "BMiss (BR)",
        "bmiss_sttni": "BMiss STTNI",
        "bmiss_sttni_branch": "BMiss STTNI (BR)",
        "lbk_v1x16_avx2": "V1 AVX2",
        "lbk_v1x16_avx2_prefilter": "V1 AVX2 + PF",
        "galloping_avx2": "Galloping AVX2",
        "galloping_avx2_prefilter": "Galloping AVX2 + PF",
    }
    return names.get(alg, alg)


def plot_branch_summary():
    data = load_json(RESULTS / "radonduo-20260413-173312" / "stage3-kernels.json")
    algos = dataset_algos(data, "2set_vary_size")
    pairs = [
        ("shuffling_sse", "shuffling_sse_branch"),
        ("broadcast_sse", "broadcast_sse_branch"),
        ("shuffling_avx2", "shuffling_avx2_branch"),
        ("broadcast_avx2", "broadcast_avx2_branch"),
        ("shuffling_avx512", "shuffling_avx512_branch"),
        ("broadcast_avx512", "broadcast_avx512_branch"),
        ("vp2intersect_emulation", "vp2intersect_emulation_branch"),
        ("conflict_intersect", "conflict_intersect_branch"),
    ]

    labels = []
    ratios = []
    colors = []
    for base_alg, branch_alg in pairs:
        base_entries = {e["x"]: e for e in algos[base_alg]}
        branch_entries = {e["x"]: e for e in algos[branch_alg]}
        xs = sorted(set(base_entries) & set(branch_entries))
        vals = [median_time(branch_entries[x]) / median_time(base_entries[x]) for x in xs]
        ratios.append(gmean(vals))
        labels.append(short_name(base_alg))
        if "sse" in base_alg:
            colors.append("#9aa0a6")
        elif "avx2" in base_alg:
            colors.append("#4c78a8")
        else:
            colors.append("#d95f02")

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.barh(y, ratios, color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Branch / branchless runtime (gmean, lower is better)")
    ax.set_xlim(0, 1.3)
    ax.grid(axis="x", alpha=0.25)
    for yi, ratio in zip(y, ratios):
        ax.text(ratio + 0.02, yi, f"{ratio:.2f}x", va="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "opt-branching-current.pdf")


def plot_writeback_summary():
    count = load_json(RESULTS / "radonduo-20260414-writeback" / "writeback-count.json")
    unsafe = load_json(RESULTS / "radonduo-20260414-writeback" / "writeback-unsafe.json")
    vec = load_json(RESULTS / "radonduo-20260414-writeback" / "writeback-vec.json")
    count_algos = dataset_algos(count, "2set_vary_size")
    unsafe_algos = dataset_algos(unsafe, "2set_vary_size")
    vec_algos = dataset_algos(vec, "2set_vary_size")

    families = [
        "broadcast_avx2_branch",
        "broadcast_avx512_branch",
        "conflict_intersect_branch",
        "shuffling_avx2_branch",
        "shuffling_avx512_branch",
        "vp2intersect_emulation_branch",
    ]

    labels = [short_name(x.replace("_branch", "")) for x in families]
    unsafe_over_count = []
    vec_over_unsafe = []
    for alg in families:
        ce = {e["x"]: e for e in count_algos[alg]}
        ue = {e["x"]: e for e in unsafe_algos[alg]}
        ve = {e["x"]: e for e in vec_algos[alg]}
        xs = sorted(ce)
        unsafe_over_count.append(gmean([median_time(ue[x]) / median_time(ce[x]) for x in xs]))
        vec_over_unsafe.append(gmean([median_time(ve[x]) / median_time(ue[x]) for x in xs]))

    y = np.arange(len(labels))
    h = 0.36
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.barh(y - h / 2, unsafe_over_count, height=h, color="#4c78a8", label="Unsafe / count-only")
    ax.barh(y + h / 2, vec_over_unsafe, height=h, color="#59a14f", label="Vec / unsafe")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Runtime ratio (gmean, lower is better)")
    ax.set_xlim(0.8, 1.34)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for yi, ratio in zip(y - h / 2, unsafe_over_count):
        ax.text(ratio + 0.01, yi, f"{ratio:.2f}x", va="center", fontsize=8)
    for yi, ratio in zip(y + h / 2, vec_over_unsafe):
        ax.text(ratio + 0.01, yi, f"{ratio:.2f}x", va="center", fontsize=8)
    fig.tight_layout()
    save_fig(fig, "opt-writeback-current.pdf")


def plot_stage2_summary():
    data = load_json(RESULTS / "radonduo-20260414-030603" / "stage2-filtering.json")
    size_algos = dataset_algos(data, "2set_vary_size")
    sel_algos = dataset_algos(data, "2set_vary_selectivity")
    skew_algos = dataset_algos(data, "2set_vary_skew")

    merge_gmean = gmean([median_time(e) for e in size_algos["branchless_merge"]])
    fused_families = [
        "qfilter",
        "qfilter_branch",
        "bmiss",
        "bmiss_branch",
        "bmiss_sttni",
        "bmiss_sttni_branch",
    ]
    fused_speedups = [
        merge_gmean / gmean([median_time(e) for e in size_algos[alg]]) for alg in fused_families
    ]

    explicit_labels = ["V1 sel.", "Gallop sel.", "V1 skew", "Gallop skew"]
    explicit_slowdowns = [
        gmean([median_time(e) for e in sel_algos["lbk_v1x16_avx2_prefilter"]])
        / gmean([median_time(e) for e in sel_algos["lbk_v1x16_avx2"]]),
        gmean([median_time(e) for e in sel_algos["galloping_avx2_prefilter"]])
        / gmean([median_time(e) for e in sel_algos["galloping_avx2"]]),
        gmean([median_time(e) for e in skew_algos["lbk_v1x16_avx2_prefilter"]])
        / gmean([median_time(e) for e in skew_algos["lbk_v1x16_avx2"]]),
        gmean([median_time(e) for e in skew_algos["galloping_avx2_prefilter"]])
        / gmean([median_time(e) for e in skew_algos["galloping_avx2"]]),
    ]

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8.2, 3.8), gridspec_kw={"wspace": 0.5})

    y0 = np.arange(len(explicit_labels))
    ax0.barh(y0, explicit_slowdowns, color="#e15759")
    ax0.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax0.set_yticks(y0, explicit_labels)
    ax0.invert_yaxis()
    ax0.set_xlim(0.9, 2.2)
    ax0.set_xlabel("Prefilter / base runtime")
    ax0.set_title("Search-based explicit prefilter")
    ax0.grid(axis="x", alpha=0.25)
    for yi, ratio in zip(y0, explicit_slowdowns):
        ax0.text(ratio + 0.03, yi, f"{ratio:.2f}x", va="center", fontsize=9)

    y1 = np.arange(len(fused_families))
    ax1.barh(y1, fused_speedups, color="#59a14f")
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax1.set_yticks(y1, [short_name(x) for x in fused_families])
    ax1.invert_yaxis()
    ax1.set_xlim(1.0, 5.4)
    ax1.set_xlabel("Speedup over Merge")
    ax1.set_title("Linear fused gating")
    ax1.grid(axis="x", alpha=0.25)
    for yi, ratio in zip(y1, fused_speedups):
        ax1.text(ratio + 0.08, yi, f"{ratio:.2f}x", va="center", fontsize=9)

    fig.tight_layout()
    save_fig(fig, "stage2-summary-current.pdf")


def main():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    PAPER_IMAGES.mkdir(parents=True, exist_ok=True)
    plot_branch_summary()
    plot_writeback_summary()
    plot_stage2_summary()


if __name__ == "__main__":
    main()
