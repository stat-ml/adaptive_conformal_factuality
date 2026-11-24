#!/usr/bin/env python
"""Figure 1: target vs empirical coverage when calibrating on two categories, one panel per method.

Writes one figure per method (Original, Adaptive).
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["tab:orange", "tab:green", "tab:purple"]


def panel(res, method, path):
    target = 1 - np.asarray([float(a) for a in res["config"]["alphas"].split(",")])
    names = [n.capitalize() for n in res["category_names"]]
    cov = np.asarray([s["metrics"][method]["coverage"] for s in res["per_seed"]], dtype=float)
    # One curve per category and one for both together (mean of the two category coverages).
    curves = [(n, cov[:, :, c]) for c, n in enumerate(names)] + [("&".join(names), np.nanmean(cov, 2))]
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0.4, 1.0], [0.4, 1.0], ls="--", lw=0.8, color="gray")
    for color, (label, values) in zip(COLORS, curves):
        mean, std = np.nanmean(values, 0), np.nanstd(values, 0)
        ax.fill_between(target, mean - std, mean + std, color=color, alpha=0.2, lw=0)
        ax.plot(target, mean, color=color, marker="o", ms=4, label=label)
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Target coverage", fontsize=16)
    ax.set_ylabel("Empirical coverage", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(alpha=0.4)
    ax.legend(loc="upper left", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", default="results/longform/two_categories_mistral.json")
    p.add_argument("--out", default="results/longform/figures/fig1", help="writes <out>_{a_original,b_adaptive}.pdf")
    args = p.parse_args()
    res = json.loads(Path(args.results).read_text())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for method, tag in (("orig", "a_original"), ("adaptive", "b_adaptive")):
        panel(res, method, Path(f"{args.out}_{tag}.pdf"))
    print(f"wrote {args.out}_*")


if __name__ == "__main__":
    main()
