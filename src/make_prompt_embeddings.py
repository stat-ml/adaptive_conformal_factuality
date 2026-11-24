#!/usr/bin/env python
"""Long-form prompt embeddings, one row per prompt in loading order: sentence encoder -> standardize -> PCA.

This produced data/emb/longformqa_mqamp32_8categor_300.npy (Mistral and Gemma share the prompts) and, with
another --encoder, the embeddings of the ablation in data/emb/longqa_ablation.
"""

import argparse

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from longform_data import load_prompts


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data/long_data")
    p.add_argument("--model", default="mistral")
    p.add_argument("--encoder", default="multi-qa-mpnet-base-dot-v1")
    p.add_argument("--pca-dim", type=int, default=32)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    data = load_prompts(args.data_dir, args.model)
    texts = [str(d["prompt"]) for d in data]
    raw = SentenceTransformer(args.encoder).encode(texts, batch_size=64, convert_to_numpy=True)
    emb = PCA(args.pca_dim, random_state=42).fit_transform(StandardScaler().fit_transform(raw))
    np.save(args.out, emb)
    print(f"saved {args.out}: {emb.shape}")


if __name__ == "__main__":
    main()
