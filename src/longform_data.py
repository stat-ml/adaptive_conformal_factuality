"""Loading the long-form QA data (LM-Polygraph manager files) into one record per prompt."""

from pathlib import Path

import numpy as np
import torch

CATEGORIES = ["inventions", "persons", "artworks", "books", "cities", "movies", "landmarks", "events"]
CATEGORY_LABELS = ["Inventions", "Biographies", "Artworks", "Books", "Cities", "Movies", "Landmarks", "Events"]
MODELS = ("mistral", "llama", "gemma")  # Table 1 uses all three; the conformal experiments Mistral and Gemma
INCORRECT_LABEL = 1
EPS = 1e-8
LABEL_KEY = ("claim", "OpenAIFactCheck")  # GPT-4o claim labels: 0 correct, 1 incorrect


def manager_file(data_dir, category, model):
    return Path(data_dir) / f"ue_manager_seed1_{category}_{model}"


def prompts_from_manager(manager, category_id):
    """One dict per prompt: prompt text, category, claims, labels and all claim-level uncertainty scores.
    Claims without a label are dropped, and so are prompts left without claims."""
    stats, estimations = manager["stats"], manager["estimations"]
    labels_all = np.asarray(manager["gen_metrics"][LABEL_KEY])
    claim_prompts = np.asarray(stats["claim_input_texts_concatenated"])
    claim_texts = np.asarray(stats["claim_texts_concatenated"])
    records = []
    for prompt in np.asarray(stats["input_texts"]):
        in_prompt = claim_prompts == prompt
        labels = labels_all[in_prompt]
        labelled = ~np.isnan(labels)
        if not labelled.any():
            continue
        records.append({
            "prompt": prompt,
            "category": category_id,
            "claims": claim_texts[in_prompt][labelled],
            "labels": labels[labelled].astype(int),
            "unc_scores": {key: np.asarray(scores)[in_prompt][labelled] for key, scores in estimations.items()},
        })
    return records


def load_prompts(data_dir, model):
    """Prompt records of all eight categories, in the order the prompt embeddings are stored."""
    records = []
    for category_id, category in enumerate(CATEGORIES):
        manager = torch.load(manager_file(data_dir, category, model), weights_only=False)
        records.extend(prompts_from_manager(manager, category_id))
    return records


def safe_threshold(scores, labels):
    """Largest threshold t such that keeping the claims with score <= t keeps no incorrect claim.
    This is the conformity score V of a prompt (Eq. 4 of the paper)."""
    scores, labels = np.asarray(scores), np.asarray(labels)
    incorrect = scores[labels == INCORRECT_LABEL]
    if len(incorrect) == 0:
        return scores.max() + EPS
    return incorrect.min() - EPS
