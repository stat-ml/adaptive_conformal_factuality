#!/usr/bin/env python
"""MCQA question embeddings, one row per line of a merged scores file: sentence encoder -> standardize -> PCA.

This produced data/emb/mcqa_mqamp32_16categor.npy (16 subjects) and, with
--subjects marketing,professional_accounting, data/emb/mcqa_mqamp32_2categor.npy.
"""

import argparse
import json

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scores", default="mcqa/data/meta-llama__Llama-2-7b-hf/all_categories_merged_results.jsonl")
    p.add_argument("--subjects", default="", help="keep only these MMLU subjects")
    p.add_argument("--encoder", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.scores, encoding="utf-8")]
    if args.subjects:
        rows = [r for r in rows if r["subject"] in args.subjects.split(",")]
    raw = SentenceTransformer(args.encoder).encode([r["asked_question"] for r in rows], convert_to_numpy=True)
    emb = PCA(args.pca_dim, random_state=42).fit_transform(StandardScaler().fit_transform(raw))
    np.save(args.out, emb)
    print(f"saved {args.out}: {emb.shape}")


if __name__ == "__main__":
    main()
