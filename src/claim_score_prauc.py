#!/usr/bin/env python
"""Table 1: how well each claim-level uncertainty score detects incorrect claims (PR-AUC)."""

import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import auc, precision_recall_curve

from longform_data import INCORRECT_LABEL, MODELS, load_prompts

# Rows of Table 1: name in the paper and LM-Polygraph score key.
SCORES = [
    ("Random Baseline", ("claim", "RandomBaselineClaim")),
    ("Maximum Probability", ("claim", "MaximumClaimProbability")),
    ("Perplexity", ("claim", "PerplexityClaim")),
    ("Max Token Entropy", ("claim", "MaxTokenEntropyClaim")),
    ("Pointwise Mutual Information", ("claim", "PointwiseMutualInformationClaim")),
    ("Claim Conditioned Probability", ("claim", "CCP_claim_fact_pref")),
    ("TokenSAR", ("claim", "TokenSARClaim")),
]
MODEL_NAMES = {"mistral": "Mistral 7B", "llama": "Llama3 8B", "gemma": "Gemma3 12B"}


def pr_auc(prompts, key):
    """PR-AUC of the score for the "incorrect" class over all claims (higher score = more uncertain)."""
    scores = np.concatenate([np.asarray(p["unc_scores"][key]) for p in prompts])
    labels = np.concatenate([np.asarray(p["labels"]) for p in prompts]) == INCORRECT_LABEL
    precision, recall, _ = precision_recall_curve(labels, scores)
    return auc(recall, precision)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data/long_data")
    p.add_argument("--out", default="results/longform/table1.md")
    args = p.parse_args()
    values = {}
    for model in MODELS:
        prompts = load_prompts(args.data_dir, model)
        values[model] = [pr_auc(prompts, key) for _, key in SCORES]
    lines = ["| Claim scoring method | " + " | ".join(MODEL_NAMES[m] for m in MODELS) + " |", "|---" * (len(MODELS) + 1) + "|"]
    for i, (name, _) in enumerate(SCORES):
        lines.append(f"| {name} | " + " | ".join(f"{values[m][i]:.3f}" for m in MODELS) + " |")
    text = "## Table 1: PR-AUC of claim-level uncertainty scores\n\n" + "\n".join(lines) + "\n"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
