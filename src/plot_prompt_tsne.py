#!/usr/bin/env python
"""Figure 5: t-SNE of the long-form prompt embeddings, coloured by category."""

import argparse

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

from longform_data import CATEGORY_LABELS


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--embeddings", default="data/emb/longformqa_mqamp32_8categor_300.npy")
    p.add_argument("--per-category", type=int, default=300, help="prompts per category, stored in order")
    p.add_argument("--out", default="results/longform/figures/fig5_tsne.pdf")
    args = p.parse_args()
    emb = np.load(args.embeddings)
    xy = TSNE(n_components=2, perplexity=40, init="pca", random_state=42).fit_transform(emb)
    category = np.arange(len(emb)) // args.per_category
    fig, ax = plt.subplots(figsize=(6, 5))
    for c, label in enumerate(CATEGORY_LABELS):
        ax.scatter(*xy[category == c].T, s=8, label=label)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Clustering of Long-Form QA Prompts")
    ax.legend(fontsize=8, markerscale=1.5)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")


if __name__ == "__main__":
    main()
