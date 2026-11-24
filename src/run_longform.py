#!/usr/bin/env python
"""Run the long-form QA experiment on many random splits and save the per-split results.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from joblib import Parallel, delayed

from longform_conformal import REGRESSORS, prepare, run_split
from longform_data import CATEGORIES, load_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data/long_data")
    p.add_argument("--embeddings", default="data/emb/longformqa_mqamp32_8categor_300.npy")
    p.add_argument("--model", default="mistral", help="mistral or gemma")
    p.add_argument("--score", default="claim,CCP_claim_fact_pref", help="claim-level uncertainty score")
    p.add_argument("--regressors", default="adaptive", help=f"comma-separated, from {', '.join(REGRESSORS)}")
    p.add_argument("--n-seeds", type=int, default=100)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--alphas", default="0.2")
    p.add_argument("--splits", default="0.3,0.4,0.3", help="cal1,cal2,test fractions")
    p.add_argument("--categories", default="", help="restrict to these categories, e.g. inventions,landmarks")
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    alphas = np.array([float(a) for a in args.alphas.split(",")])
    prompts = load_prompts(args.data_dir, args.model)
    embeddings = np.load(args.embeddings)
    z, labels, V, categories = prepare(prompts, tuple(args.score.split(",")))
    assert len(embeddings) == len(V), f"{len(embeddings)} embedding rows for {len(V)} prompts"

    names = list(CATEGORIES)
    if args.categories:
        names = args.categories.split(",")
        ids = [CATEGORIES.index(n) for n in names]
        keep = np.flatnonzero(np.isin(categories, ids))
        z, labels, V, embeddings = [z[i] for i in keep], [labels[i] for i in keep], V[keep], embeddings[keep]
        categories = np.asarray([ids.index(c) for c in categories[keep]])

    regressors = args.regressors.split(",")
    cal1, cal2, test = (float(x) for x in args.splits.split(","))

    def one_split(seed):
        torch.set_num_threads(1)
        metrics = run_split(z, labels, V, categories, embeddings, alphas, seed, regressors,
                            cal1=cal1, cal2=cal2, test=test, n_categories=len(names))
        return {"seed": seed, "metrics": {m: {k: v.tolist() for k, v in r.items()} for m, r in metrics.items()}}

    seeds = [args.seed_base + s for s in range(args.n_seeds)]
    per_seed = Parallel(n_jobs=args.n_jobs)(delayed(one_split)(s) for s in seeds)

    for a_idx, alpha in enumerate(alphas):
        print(f"\ntarget coverage {1 - alpha:.2f}: mean coverage per category")
        for method in per_seed[0]["metrics"]:
            cov = np.nanmean([s["metrics"][method]["coverage"][a_idx] for s in per_seed], axis=0)
            print(f"  {method:<24}" + " ".join(f"{100 * c:5.1f}" for c in cov)
                  + f"   bias {100 * np.mean(np.abs(cov - (1 - alpha))):.2f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    config = {k: v for k, v in vars(args).items() if k not in ("n_jobs", "out")}
    out.write_text(json.dumps({"config": config, "category_names": names, "per_seed": per_seed}) + "\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
