#!/usr/bin/env python
"""Tables 2-7 of the paper and the numbers of Appendix B.2, as Markdown, from results/longform."""

import json
import os
from pathlib import Path

import numpy as np

from longform_data import CATEGORY_LABELS

RESULTS = Path(os.environ.get("RESULTS_DIR", "results/longform"))
MODELS = {"mistral": "Mistral 7B", "gemma": "Gemma-3 12B"}


def load(name):
    return json.loads((RESULTS / name).read_text())


def per_seed(res, method, metric, alpha):
    """(splits, categories) array of one metric at one target level."""
    alphas = [float(a) for a in str(res["config"].get("alphas", res["config"].get("alpha"))).split(",")]
    a_idx = int(np.argmin(np.abs(np.asarray(alphas) - alpha)))
    return np.asarray([s["metrics"][method][metric][a_idx] for s in res["per_seed"]], dtype=float)


def bias(res, method, alpha=0.2):
    """Coverage bias: mean over categories of |category coverage - target|, in %."""
    return 100 * np.mean(np.abs(np.nanmean(per_seed(res, method, "coverage", alpha), 0) - (1 - alpha)))


def category_table(sources, metric, alpha):
    """Rows: categories (mean ± std over splits, in %). sources: [(column, results, method)]."""
    arrays = [(col, per_seed(res, m, metric, alpha)) for col, res, m in sources]
    lines = ["| Category | " + " | ".join(col for col, _ in arrays) + " |", "|---" * (len(arrays) + 1) + "|"]
    for c, label in enumerate(CATEGORY_LABELS):
        cells = [f"{100 * np.nanmean(a[:, c]):.2f} ± {100 * np.nanstd(a[:, c]):.2f}" for _, a in arrays]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    if metric == "coverage":
        lines.append("| Cov. bias | " + " | ".join(f"{bias(res, m, alpha):.2f}" for _, res, m in sources) + " |")
    return "\n".join(lines)


def ablation_table():
    """Table 4: coverage bias of Adaptive with one setting changed (Mistral, 50 splits)."""
    pca, mlp = load("ablation/pca.json"), load("ablation/mlp.json")
    split = [load(f"ablation/split_{s}.json") for s in ("0.25_0.45_0.3", "0.35_0.35_0.3", "0.4_0.3_0.3", "0.45_0.25_0.3")]
    emb = [load(f"ablation/embedding_{e}.json") for e in
           ("all_mpnet_base_v2", "all_distilroberta_v1", "sentence_transformers__all_MiniLM_L12_v2")]
    default = (pca, "adaptive")
    components = {
        "Number of PCA components": [(pca, m) for m in ("adaptive_d4", "adaptive", "adaptive_d16", "adaptive_d32")],
        "Embedding model": [default] + [(r, "adaptive") for r in emb],
        "D_cal1/D_cal2 split": [default] + [(r, "adaptive") for r in split],
        "MLP configuration": [default] + [(mlp, m) for m in
                                          ("adaptive_h16", "adaptive_2layers", "adaptive_early_stopping", "adaptive_bagging")],
    }
    orig = bias(pca, "orig")
    lines = ["| Ablation component | Adaptive wins | Mean bias reduction | Bias range |", "|---|---|---|---|"]
    for component, settings in components.items():
        b = np.array([bias(res, m) for res, m in settings])
        lines.append(f"| {component} | {int((b < orig).sum())}/{len(b)} | {100 * np.mean(1 - b / orig):.1f}% | "
                     f"{b.min():.2f}–{b.max():.2f} |")
    lines.append(f"\nOriginal has a bias of {orig:.2f} and the oracle Group-wise {bias(pca, 'groupwise'):.2f}.")
    return "\n".join(lines)


def efficiency_table():
    """Appendix B.2: claims kept per prompt and % of correct claims removed."""
    lines = ["| Model | Method | Retained claims per prompt | Correct claims removed (%) |", "|---|---|---|---|"]
    for model, name in MODELS.items():
        res = load(f"main_{model}.json")
        for label, method in (("Original", "orig"), ("Adaptive", "adaptive")):
            retained = np.nanmean(per_seed(res, method, "retained", 0.2))
            removed = 100 * np.nanmean(per_seed(res, method, "removed_true", 0.2))
            lines.append(f"| {name} | {label} | {retained:.2f} | {removed:.1f} |")
    return "\n".join(lines)


def main():
    sections = []
    for (model, name), number in zip(MODELS.items(), (2, 3)):
        res, boosted = load(f"main_{model}.json"), load(f"boosted_{model}.json")
        sources = [("Original", res, "orig"), ("Adaptive", res, "adaptive"),
                   ("Group-wise", res, "groupwise"), ("Boosted", boosted, "boosted")]
        sections.append((f"Table {number}: coverage (%), {name}, target 80%", category_table(sources, "coverage", 0.2)))
        sections.append((f"Table {number + 3}: removed claims (%), {name}, target 80%",
                         category_table(sources, "removed", 0.2)))
    sections.sort(key=lambda s: int(s[0].split(":")[0].split()[1]))
    sections.insert(2, ("Table 4: ablation, Mistral 7B, 50 splits, target 80%", ablation_table()))
    high = load("high_coverage_mistral.json")
    sections.append(("Table 7: coverage (%), Mistral 7B, target 95%", category_table(
        [("Original", high, "orig"), ("Adaptive", high, "adaptive"), ("Group-wise", high, "groupwise")],
        "coverage", 0.05)))
    sections.append(("Appendix B.2: efficiency, target 80%", efficiency_table()))
    text = "\n\n".join(f"## {title}\n\n{table}" for title, table in sections) + "\n"
    (RESULTS / "tables.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
