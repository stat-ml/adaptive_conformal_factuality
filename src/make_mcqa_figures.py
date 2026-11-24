#!/usr/bin/env python
"""Figures 2 and 4: MCQA target vs empirical coverage for two categories and for all calibration
categories together, one figure per method (Original, Adaptive), in the style of the paper.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = ["tab:orange", "tab:green", "tab:purple"]


def coverage_figure(target, curves, path):
    plt.figure(figsize=(7, 6))
    for color, (label, cov) in zip(COLORS, curves):
        mean, std = np.nanmean(cov, 0), np.nanstd(cov, 0)
        plt.plot(target, mean, marker="o", label=label, color=color)
        plt.fill_between(target, mean - std, mean + std, alpha=0.2, color=color)
    plt.plot([0.4, 1.0], [0.4, 1.0], "--", color="gray")
    plt.xlabel("Target coverage", fontsize=16)
    plt.ylabel("Empirical coverage", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=14)
    plt.xlim(0.45, 1.0)
    plt.ylim(0.4, 1.0)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(f"{path}.pdf", bbox_inches="tight")
    plt.close()


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", required=True, help="JSON written by run_mcqa.py")
    p.add_argument("--categories", required=True, help="two MMLU subjects, e.g. marketing,professional_accounting")
    p.add_argument("--labels", default=None, help="legend names for the two subjects (default: capitalized)")
    p.add_argument("--all-label", required=True, help='legend name of all categories, e.g. "16 Categories"')
    p.add_argument("--out", required=True, help="output prefix; writes <out>_{a_original,b_adaptive}.pdf")
    args = p.parse_args()

    run = json.loads(open(args.results, encoding="utf-8").read())
    names = run["category_names"]
    target = 1 - np.asarray(run["config"]["alphas"])
    subjects = args.categories.split(",")
    labels = args.labels.split(",") if args.labels else [s.capitalize() for s in subjects]
    for method, tag in (("orig", "a_original"), ("adaptive", "b_adaptive")):
        cov = np.asarray(run["metrics"][method]["coverage"], dtype=float)  # (splits, alphas, groups + overall)
        curves = [(label, cov[:, :, names.index(s)]) for label, s in zip(labels, subjects)]
        curves.append((args.all_label, cov[:, :, names.index("overall")]))
        coverage_figure(target, curves, f"{args.out}_{tag}")
        gaps = [np.nanmean(c, 0) - target for _, c in curves[:2]]
        print(f"{tag}: mean (coverage - target) over targets: "
              + ", ".join(f"{l} {100 * g.mean():+.1f}" for l, g in zip(labels, gaps)))


if __name__ == "__main__":
    main()
