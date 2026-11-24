#!/usr/bin/env python
"""Run the Boosted baseline (Cherian et al., 2024) on the same random splits as run_longform.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from boosted import boosted_on_split
from longform_conformal import METRICS, aggregate_by_category, evaluate_kept, paired_split
from longform_data import load_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data/long_data")
    p.add_argument("--model", default="mistral", help="mistral or gemma")
    p.add_argument("--alpha", type=float, default=0.2)
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--splits", default="0.3,0.4,0.3", help="cal1,cal2,test fractions")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    prompts = load_prompts(args.data_dir, args.model)
    categories = np.asarray([int(p["category"]) for p in prompts])
    labels = [np.asarray(p["labels"]) for p in prompts]
    cal1, cal2, test = (float(x) for x in args.splits.split(","))

    def one_split(seed):
        torch.set_num_threads(1)
        cal1_idx, cal2_idx, test_idx = paired_split(len(prompts), seed, cal1, cal2, test)
        kept = boosted_on_split(prompts, args.alpha, seed, cal1_idx, cal2_idx, test_idx)
        per_prompt = [evaluate_kept(labels[i], k) for i, k in zip(test_idx, kept)]
        agg = aggregate_by_category(per_prompt, categories[test_idx], 8)
        return {"seed": seed, "metrics": {"boosted": {m: [agg[m].tolist()] for m in METRICS}}}

    seeds = range(args.seed_base, args.seed_base + args.n_seeds)
    per_seed = Parallel(n_jobs=args.n_jobs)(delayed(one_split)(s) for s in seeds)
    cov = np.nanmean([s["metrics"]["boosted"]["coverage"][0] for s in per_seed], axis=0)
    print("mean coverage per category: " + " ".join(f"{100 * c:5.1f}" for c in cov))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("n_jobs", "out")}
    out.write_text(json.dumps({"config": config, "per_seed": per_seed}) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
