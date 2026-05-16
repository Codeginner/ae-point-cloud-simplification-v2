"""
plot_results.py
===============
Buat figure dari hasil evaluate_downstream.py.

Usage:
    python plot_results.py --results ./eval_results/results.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

STYLE = {
    "ours":      dict(color="#E05C5C", lw=2.5, marker="*", ms=10, zorder=5, label="Ours"),
    "fps":       dict(color="#5BA3E8", lw=1.8, marker="o", ms=7,  zorder=3, label="FPS"),
    "random":    dict(color="#A0A0A0", lw=1.5, marker="s", ms=6,  zorder=2, label="Random"),
    "voxel":     dict(color="#F0A030", lw=1.8, marker="^", ms=7,  zorder=3, label="Voxel"),
    "apes":      dict(color="#4DCCA0", lw=2.0, marker="D", ms=7,  zorder=4, label="APES"),
    "samplenet": dict(color="#C07AD8", lw=2.0, marker="v", ms=7,  zorder=4, label="SampleNet"),
}


def load(path: str) -> tuple[list[int], dict]:
    with open(path) as f:
        data = json.load(f)
    return data["M_values"], data["results"]


def plot(M_values, results, metric: str, ylabel: str, title: str, save_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0f0f14")
    ax.set_facecolor("#0f0f14")

    for name, style in STYLE.items():
        if name not in results:
            continue
        ys = [results[name].get(str(M), results[name].get(M, {})).get(metric, None)
              for M in M_values]
        valid = [(x, y) for x, y in zip(M_values, ys) if y is not None]
        if not valid:
            continue
        xs, ys = zip(*valid)
        ax.plot(xs, ys, **style)

    ax.set_xlabel("M (number of simplified points)", color="#aaa", fontsize=11)
    ax.set_ylabel(ylabel,                            color="#aaa", fontsize=11)
    ax.set_title(title,                              color="#eee", fontsize=13, pad=12)

    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(color="#222", linestyle="--", linewidth=0.7)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))

    legend = ax.legend(facecolor="#1a1a22", edgecolor="#333",
                       labelcolor="#ddd",  fontsize=10,
                       loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {save_path}")


def plot_time(M_values, results, save_path: Path):
    """Bar chart: avg inference time per method at each M."""
    names  = [n for n in STYLE if n in results]
    colors = [STYLE[n]["color"] for n in names]
    x      = np.arange(len(M_values))
    width  = 0.8 / len(names)

    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0f0f14")
    ax.set_facecolor("#0f0f14")

    for i, (name, color) in enumerate(zip(names, colors)):
        ys = [results[name].get(str(M), results[name].get(M, {})).get("avg_time_ms", 0)
              for M in M_values]
        ax.bar(x + i * width, ys, width, label=STYLE[name]["label"],
               color=color, alpha=0.85, edgecolor="#0f0f14")

    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels([f"M={m}" for m in M_values], color="#aaa")
    ax.set_ylabel("Avg inference time (ms/sample)", color="#aaa", fontsize=11)
    ax.set_title("Inference Time Comparison", color="#eee", fontsize=13, pad=12)
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.grid(axis="y", color="#222", linestyle="--", linewidth=0.7)
    ax.legend(facecolor="#1a1a22", edgecolor="#333", labelcolor="#ddd", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved → {save_path}")


def print_table(M_values, results, metric="overall_acc"):
    methods = [n for n in STYLE if n in results]
    header  = f"{'Method':<12}" + "".join(f"  M={m:<5}" for m in M_values)
    print("\n" + "="*len(header))
    print(header)
    print("-"*len(header))
    for name in methods:
        row = f"{name:<12}"
        for M in M_values:
            val = results[name].get(str(M), results[name].get(M, {})).get(metric)
            row += f"  {val:>5.1f}%  " if val is not None else "   —      "
        print(row)
    print("="*len(header))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="./eval_results/results.json")
    p.add_argument("--out_dir", default=None)
    args = p.parse_args()

    result_path = Path(args.results)
    out_dir     = Path(args.out_dir) if args.out_dir else result_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    M_values, results = load(args.results)

    plot(M_values, results,
         metric="overall_acc",
         ylabel="Overall Accuracy (%)",
         title="Downstream Classification Accuracy vs M",
         save_path=out_dir / "acc_overall.png")

    plot(M_values, results,
         metric="per_class_acc",
         ylabel="Mean Class Accuracy (%)",
         title="Mean Per-Class Accuracy vs M",
         save_path=out_dir / "acc_per_class.png")

    plot_time(M_values, results, out_dir / "inference_time.png")

    print_table(M_values, results, "overall_acc")
    print_table(M_values, results, "per_class_acc")


if __name__ == "__main__":
    main()
