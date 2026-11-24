#!/usr/bin/env python
"""Figure 3: Dolan-More performance profiles of the MCQA calibration error, Original vs Adaptive.

A problem is one (category, split, target coverage) triple; the error is |coverage - target|.
The profile is the fraction of problems on which a method is within a factor delta of the best one.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def calibration_errors(run, coverage_min, coverage_max):
    """|coverage - target| of both methods, flattened over splits, target coverages and categories."""
    target = 1.0 - np.asarray(run["config"]["alphas"], dtype=float)
    keep = (target >= coverage_min - 1e-9) & (target <= coverage_max + 1e-9)
    categories = [i for i, c in enumerate(run["category_names"]) if c != "overall"]
    errors = []
    for method in ("orig", "adaptive"):
        cov = np.asarray(run["metrics"][method]["coverage"], dtype=float)[:, keep][:, :, categories]
        errors.append(np.abs(cov - target[keep][None, :, None]).reshape(-1))
    return errors


def profile(ratios, deltas):
    return np.array([(ratios <= d).mean() for d in deltas])


def plot(err_orig, err_adaptive, out, ymin):
    best = np.minimum(err_orig, err_adaptive)
    ratios = []
    for err in (err_orig, err_adaptive):
        r = np.ones_like(err)
        r[best >= 1e-12] = err[best >= 1e-12] / best[best >= 1e-12]  # both exact: ratio 1
        ratios.append(r)
    delta_max = min(max(3.0, float(np.quantile(np.concatenate(ratios), 0.95))), 10.0)
    deltas = np.linspace(1.0, delta_max, 400)

    fig, ax = plt.subplots(figsize=(6, 4))
    for r, label in zip(ratios, ("Original", "Adaptive")):
        ax.plot(deltas, profile(r, deltas), label=label, linewidth=2.5)
    ax.set_xlabel(r"$\delta$", fontsize=16)
    ax.set_ylabel("Fraction of problems", fontsize=16)
    ax.set_xscale("log")
    ax.set_xlim(1.0, delta_max)
    ax.set_ylim(ymin, 1.0)
    ax.set_xticks(np.arange(1, int(delta_max) + 1))
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.tick_params(axis="both", labelsize=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, which="major", alpha=0.3)
    ax.legend(frameon=False, fontsize=14)
    fig.tight_layout()
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    for r, label in zip(ratios, ("Original", "Adaptive")):
        print(f"{label}: best on {np.isclose(r, 1.0).mean():.3f} of {len(r)} problems")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--results", required=True, help="JSON written by run_mcqa.py")
    p.add_argument("--coverage-range", default="0.5,0.8", help="target coverages included")
    p.add_argument("--ymin", type=float, default=0.35)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    lo, hi = (float(x) for x in args.coverage_range.split(","))
    run = json.loads(open(args.results, encoding="utf-8").read())
    plot(*calibration_errors(run, lo, hi), args.out, args.ymin)


if __name__ == "__main__":
    main()
