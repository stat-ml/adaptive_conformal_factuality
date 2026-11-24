#!/usr/bin/env python
"""Run the multiple-choice QA experiment (Original and Adaptive) on many random splits.
"""

import argparse
import json
from fractions import Fraction
from pathlib import Path

import numpy as np

from mcqa_conformal import load_mcqa_data, run_mcqa


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores", required=True, help="merged answer probabilities (merge_mcqa_scores.py)")
    p.add_argument("--embeddings", required=True, help="question embeddings, one row per line of --scores")
    p.add_argument("--subjects", default="", help="keep only these MMLU subjects, e.g. marketing,professional_accounting")
    p.add_argument("--n-seeds", type=int, default=20)
    p.add_argument("--splits", default="0.3,0.4,0.3", help="cal1,cal2,test fractions within each subject, e.g. 1/3,1/3,1/3")
    p.add_argument("--epochs", type=int, default=1300, help="training epochs of the quantile network")
    p.add_argument("--hidden-dim", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--n-alphas", type=int, default=10, help="miscoverage levels np.linspace(0.05, 0.5, n)")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--out", required=True)
    args = p.parse_args()

    alphas = np.linspace(0.05, 0.5, args.n_alphas)
    cal1, cal2, _ = (float(Fraction(x)) for x in args.splits.split(","))
    probs, targets, categories, embeddings = load_mcqa_data(args.scores, args.embeddings,
                                                               [s for s in args.subjects.split(",") if s])
    names, cov = run_mcqa(probs, targets, categories, embeddings, alphas, args.n_seeds, cal1, cal2, args.hidden_dim,
                          args.epochs, args.lr, args.weight_decay, device=args.device)
    config = {**{k: v for k, v in vars(args).items() if k != "out"}, "alphas": alphas.tolist()}
    result = {"config": config, "category_names": names,
              "metrics": {m: {"coverage": c.tolist()} for m, c in cov.items()}}
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result) + "\n")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
