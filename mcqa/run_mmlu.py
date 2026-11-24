#!/usr/bin/env python3
"""
run_mmlu.py

What this script does
---------------------
1) Loads MMLU subjects from Hugging Face (lukaemon/mmlu) with trust_remote_code=True.
2) Builds prompts as the original ConformalLLM code (MMLU-based or GPT-based 1-shot).
3) Saves the questions, prompts and question-prompt pairs once each (no prompt is repeated per question):
   - questions/<subject>.jsonl              (question-only + options + target)  [saved once]
   - prompts/<regime>/<subject>.jsonl       (prompt_add per run)               [saved once per run]
   - pairs/<regime>/<subject>/run_XX.jsonl  (links run_idx <-> (split, idx))  [saved per run]

4) Unless --no_infer is given, runs the model and saves its answer probabilities:
   - outputs/<model_tag>/<regime>/<subject>_scores.npy
   - outputs/<model_tag>/<regime>/<subject>_targets.npy
   - outputs/<model_tag>/<regime>/<subject>_acc.pkl

Notes
-----
- For 'lukaemon/mmlu' you MUST pass trust_remote_code=True.
- For Llama-2 weights: you need `huggingface-cli login` and access to the model.
"""

import argparse
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from datasets import load_dataset

# GPT prompt examples live here (repo file)
import prompt_questions as p

from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(pth: Path) -> None:
    pth.mkdir(parents=True, exist_ok=True)


def append_jsonl(path: Path, row: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def format_question_only(task_data: dict, split: str, i: int) -> Tuple[str, Dict[str, str], str]:
    """Question + options only (no example, no instruction)."""
    q = task_data[split]["input"][i]
    opts = {L: task_data[split][L][i] for L in ["A", "B", "C", "D"]}
    q_only = q + "\n" + "\n".join([f"({L}) {opts[L]}" for L in ["A", "B", "C", "D"]])
    return q, opts, q_only


# ----------------------------
# Prompt construction and scoring follow the original ConformalLLM code (Kumar et al., 2023)
# ----------------------------
def modify_task_data(task_data, token_limit: int, max_size_prompt_len: int):
    """
    As in ConformalLLM (uses string lengths as a proxy).
    """
    new_task_data = {"train": defaultdict(list), "validation": defaultdict(list), "test": defaultdict(list)}
    for split in new_task_data.keys():
        for i in range(len(task_data[split])):
            q = task_data[split]["input"][i]
            a = task_data[split]["A"][i]
            b = task_data[split]["B"][i]
            c = task_data[split]["C"][i]
            d = task_data[split]["D"][i]
            target = task_data[split]["target"][i]
            if len(q) + max(map(len, [a, b, c, d])) + max_size_prompt_len < token_limit:
                new_task_data[split]["input"].append(q)
                new_task_data[split]["A"].append(a)
                new_task_data[split]["B"].append(b)
                new_task_data[split]["C"].append(c)
                new_task_data[split]["D"].append(d)
                new_task_data[split]["target"].append(target)
    return new_task_data


def get_prompt(task_data, task: str, question_num: int = 0, prompt_q: Optional[str] = None) -> str:
    """
    Same formatting as get_prompt() in ConformalLLM.
    """
    if prompt_q is None:
        prompt_set = "test"
        if question_num > len(task_data["test"]["input"]) - 1:
            question_num = len(task_data["test"]["input"]) - 1

        prompt_add = f'This is a question from {task.replace("_", " ")}.\n'
        prompt_add += f"{task_data[prompt_set]['input'][question_num]}\n"
        for letter in ["A", "B", "C", "D"]:
            prompt_add += "    " + letter + ". " + task_data[prompt_set][letter][question_num] + "\n"
        prompt_add += f"The correct answer is option: {task_data[prompt_set]['target'][question_num]}\n"
    else:
        prompt_add = f'This is a question from {task.replace("_", " ")}.'
        prompt_add += prompt_q
        prompt_add += "\n"

    prompt_add += f"You are the world's best expert in {task.replace('_', ' ')}. "
    prompt_add += "Reason step-by-step and answer the following question. "
    return prompt_add


def get_max_size_prompt_len(task_data, task: str, n: int = 10, max_allowed_prompt_len: int = 700):
    """
    As in ConformalLLM: choose n test questions as 1-shot prompts, constrained by prompt length.
    Returns (max_len_among_selected, selected_prompt_question_ids).
    """
    max_len = 0
    i = 0
    prompt_question_ids = []
    while len(prompt_question_ids) < n:
        if i >= len(task_data["test"]["input"]):
            break
        prompt_add = get_prompt(task_data, task=task, question_num=i, prompt_q=None)
        prompt_len = len(prompt_add)
        if prompt_len <= max_allowed_prompt_len:
            prompt_question_ids.append(i)
            max_len = max(max_len, prompt_len)
        i += 1
    return max_len, prompt_question_ids


# ----------------------------
# Prompt list mapping (GPT-based)
# ----------------------------
def gpt_prompt_list_for_subject(subject: str) -> List[str]:
    mapping = {
        "college_computer_science": getattr(p, "prompt_q_list_college_cs"),
        "formal_logic": getattr(p, "prompt_q_list_formal_logic"),
        "high_school_computer_science": getattr(p, "prompt_q_list_high_school_cs"),
        "computer_security": getattr(p, "prompt_q_list_computer_security"),
        "machine_learning": getattr(p, "prompt_q_list_machine_learning"),
        "clinical_knowledge": getattr(p, "prompt_q_list_clinical_knowledge"),
        "high_school_biology": getattr(p, "prompt_q_list_high_school_bio"),
        "anatomy": getattr(p, "prompt_q_list_anatomy"),
        # Fix: avoid eager evaluation
        "college_chemistry": getattr(p, "promtp_q_list_college_chemistry", None) or getattr(p, "prompt_q_list_college_chemistry"),
        "college_medicine": getattr(p, "prompt_q_list_college_medicine"),
        "professional_medicine": getattr(p, "prompt_q_list_professional_medicine"),
        "business_ethics": getattr(p, "prompt_q_list_business_ethics"),
        "professional_accounting": getattr(p, "prompt_q_list_professional_accounting"),
        "public_relations": getattr(p, "prompt_q_list_pr"),
        "management": getattr(p, "prompt_q_list_management"),
        "marketing": getattr(p, "prompt_q_list_marketing"),
    }
    if subject not in mapping:
        raise KeyError(f"No GPT prompt list mapping found for subject '{subject}'.")
    return list(mapping[subject])



# ----------------------------
# Logging of questions, prompts and pairs
# ----------------------------
def save_questions_once(task_data: dict, subject: str, out_dir: Path, splits: Tuple[str, ...]) -> Path:
    """
    Writes: out_dir/questions/<subject>.jsonl
    One row per question containing: question_only + target + split + idx.
    """
    q_dir = out_dir / "questions"
    ensure_dir(q_dir)
    path = q_dir / f"{subject}.jsonl"

    rows = []
    for split in splits:
        start = 1 if split == "train" else 0
        for i in range(start, len(task_data[split]["input"])):
            q, opts, q_only = format_question_only(task_data, split, i)
            rows.append(
                {
                    "subject": subject,
                    "split": split,
                    "idx_in_split": i,
                    "question": q,
                    "options": opts,
                    "target": task_data[split]["target"][i],
                    "question_only": q_only,
                }
            )
    dump_jsonl(path, rows)
    return path


def log_prompt_run(out_dir: Path, regime: str, subject: str, run_idx: int, prompt_add: str, meta: dict) -> None:
    """
    Appends one row per run:
      out_dir/prompts/<regime>/<subject>.jsonl
    """
    p_dir = out_dir / "prompts" / regime
    ensure_dir(p_dir)
    row = {"regime": regime, "subject": subject, "run_idx": run_idx, "prompt_add": prompt_add, **meta}
    append_jsonl(p_dir / f"{subject}.jsonl", row)


def log_pairs_for_run(out_dir: Path, regime: str, subject: str, run_idx: int, pairs: List[dict]) -> None:
    """
    Writes:
      out_dir/pairs/<regime>/<subject>/run_XX.jsonl
    Each row links run_idx to a question via (split, idx_in_split).
    """
    pair_dir = out_dir / "pairs" / regime / subject
    ensure_dir(pair_dir)
    dump_jsonl(pair_dir / f"run_{run_idx:02d}.jsonl", pairs)


# ----------------------------
# Optional inference (scores/targets)
# ----------------------------
@torch.no_grad()
def score_one_question(model, tokenizer, base_input: str) -> np.ndarray:
    """
    Scores choices A,B,C,D by computing log-softmax prob of the final option token (as in ConformalLLM).
    base_input ends with: "\\nThe correct answer is option: "
    We create 4 variants by appending A/B/C/D, and return probs shape (4,).
    """
    device = model.device if hasattr(model, "device") else torch.device("cuda")
    texts = [base_input + L for L in ["A", "B", "C", "D"]]

    all_logits = []
    all_option_ids = []
    for t in texts:
        enc = tokenizer(t, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"].to(device)
        out = model(input_ids)
        logits = out.logits  # (1, seq, vocab)

        # option token is last token; we want logits at previous position (the -2:-1 slice of ConformalLLM)
        option_id = input_ids[:, -1:]             # (1,1)
        logits_pos = logits[:, -2:-1, :]          # (1,1,vocab)
        log_probs = torch.log_softmax(logits_pos.float(), dim=-1)  # (1,1,vocab)
        opt_logp = torch.gather(log_probs, 2, option_id[:, :, None]).squeeze().item()
        all_logits.append(opt_logp)
        all_option_ids.append(option_id.squeeze().item())

    # convert 4 log-probs to probs
    logits_np = np.array(all_logits, dtype=np.float64)
    ex = np.exp(logits_np - logits_np.max())
    probs = ex / ex.sum()
    return probs  # (4,)


def load_llama_from_hf(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.pad_token_id = tok.pad_token_id
    model.eval()
    return model, tok


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", required=True, help="MMLU subject names, e.g. professional_accounting marketing")
    ap.add_argument("--regime", choices=["mmlu_based", "gpt_based"], required=True)
    ap.add_argument("--n_prompts", type=int, default=1, help="Number of prompt variants to use (per subject).")
    ap.add_argument("--token_limit", type=int, default=1500)
    ap.add_argument("--max_allowed_prompt_len", type=int, default=700)
    ap.add_argument("--out_dir", type=str, default="mcqa/data")

    ap.add_argument("--model_id", type=str, default="meta-llama/Llama-2-7b-hf",
                    help="HF model id (requires access), used if inference enabled.")
    ap.add_argument("--no_infer", action="store_true", help="Only log prompts/questions/pairs; do not run model.")
    ap.add_argument("--hf_cache_dir", type=str, default=None, help="Optional cache_dir for datasets + models.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Dataset loader wrapper
    def load_mmlu(subject: str):
        return load_dataset(
            "lukaemon/mmlu",
            subject,
            trust_remote_code=True,
            cache_dir=args.hf_cache_dir,
        )

    # Load model optionally
    model = tokenizer = None
    model_tag = None
    if not args.no_infer:
        model, tokenizer = load_llama_from_hf(args.model_id)
        model_tag = args.model_id.replace("/", "__")

    # Output dir for scores
    if model_tag is not None:
        scores_dir = out_dir / "outputs" / model_tag / args.regime
        ensure_dir(scores_dir)

    # Run per subject
    for subject in args.subjects:
        raw = load_mmlu(subject)

        if args.regime == "mmlu_based":
            # Choose prompt questions from test set
            max_len, prompt_ids = get_max_size_prompt_len(
                raw, task=subject, n=args.n_prompts, max_allowed_prompt_len=args.max_allowed_prompt_len
            )
            if len(prompt_ids) == 0:
                raise RuntimeError(f"No prompt ids found for {subject} under max_allowed_prompt_len.")

            # Filter using max_len among chosen prompts (same as original)
            filtered = modify_task_data(raw, token_limit=args.token_limit, max_size_prompt_len=max_len)

            # Save questions once (train+val+test) because mmlu_based uses all splits in ConformalLLM
            save_questions_once(filtered, subject, out_dir, splits=("train", "validation", "test"))

            acc_list = []
            for run_idx, prompt_q_id in enumerate(prompt_ids):
                prompt_add = get_prompt(raw, task=subject, question_num=prompt_q_id, prompt_q=None)

                # Log prompt run
                log_prompt_run(
                    out_dir, "mmlu_based", subject, run_idx, prompt_add,
                    meta={"prompt_question_id_test": prompt_q_id}
                )

                # Log pairs for this run (exclude prompt question from test)
                pairs = []
                for split in ("train", "validation", "test"):
                    start = 1 if split == "train" else 0
                    for i in range(start, len(filtered[split]["input"])):
                        if split == "test" and i == prompt_q_id:
                            continue
                        pairs.append({"run_idx": run_idx, "split": split, "idx_in_split": i})
                log_pairs_for_run(out_dir, "mmlu_based", subject, run_idx, pairs)

                # Optional inference
                if not args.no_infer:
                    # Score each pair in order (same order as pairs file)
                    scores = []
                    targets = []
                    amap = {"A": 0, "B": 1, "C": 2, "D": 3}
                    for pr in pairs:
                        split = pr["split"]
                        i = pr["idx_in_split"]
                        q, opts, q_only = format_question_only(filtered, split, i)
                        base_input = prompt_add + q_only + "\nThe correct answer is option: "
                        probs = score_one_question(model, tokenizer, base_input)
                        scores.append(probs)
                        targets.append(amap[filtered[split]["target"][i]])

                    scores = np.array(scores, dtype=np.float32)          # (N,4)
                    targets = np.array(targets, dtype=np.int64)          # (N,)

                    # Save per-run scores as (1,N,4) to match "num_prompts" axis
                    np.save(scores_dir / f"{subject}_run{run_idx:02d}_scores.npy", scores[None, :, :])
                    np.save(scores_dir / f"{subject}_run{run_idx:02d}_targets.npy", targets)

                    acc = float((scores.argmax(1) == targets).mean())
                    acc_list.append(acc)

            if not args.no_infer:
                with open(scores_dir / f"{subject}_acc.pkl", "wb") as f:
                    pickle.dump({"subject": subject, "acc_list": acc_list}, f)

        else:
            # GPT-based prompts from prompt_questions.py
            prompt_q_list = gpt_prompt_list_for_subject(subject)[: args.n_prompts]
            max_prompt_len = int(np.max([len(x) for x in prompt_q_list]))

            # Filter using max GPT prompt length (same as original)
            filtered = modify_task_data(raw, token_limit=args.token_limit, max_size_prompt_len=max_prompt_len)

            # Save questions once. For GPT-based artifacts (repo-style) you often want validation+test.
            save_questions_once(filtered, subject, out_dir, splits=("validation", "test"))

            # Prepare a stable list of question pairs (val+test)
            base_pairs = []
            for split in ("validation", "test"):
                for i in range(len(filtered[split]["input"])):
                    base_pairs.append({"split": split, "idx_in_split": i})

            acc_list = []
            all_scores_runs = []
            amap = {"A": 0, "B": 1, "C": 2, "D": 3}

            for run_idx, prompt_q in enumerate(prompt_q_list):
                prompt_add = get_prompt(filtered, task=subject, prompt_q=prompt_q)

                log_prompt_run(
                    out_dir, "gpt_based", subject, run_idx, prompt_add,
                    meta={"gpt_prompt_idx": run_idx, "prompt_q": prompt_q}
                )

                # pairs for this run (same base pairs)
                pairs = [{"run_idx": run_idx, **bp} for bp in base_pairs]
                log_pairs_for_run(out_dir, "gpt_based", subject, run_idx, pairs)

                if not args.no_infer:
                    scores = []
                    targets = []
                    for bp in base_pairs:
                        split = bp["split"]
                        i = bp["idx_in_split"]
                        q, opts, q_only = format_question_only(filtered, split, i)
                        base_input = prompt_add + q_only + "\nThe correct answer is option: "
                        probs = score_one_question(model, tokenizer, base_input)
                        scores.append(probs)
                        targets.append(amap[filtered[split]["target"][i]])

                    scores = np.array(scores, dtype=np.float32)  # (N,4)
                    all_scores_runs.append(scores)
                    acc_list.append(float((scores.argmax(1) == np.array(targets)).mean()))

            if not args.no_infer:
                # Save combined scores as (num_prompts, N, 4) and targets as (N,)
                scores_3d = np.stack(all_scores_runs, axis=0)  # (P,N,4)
                targets_1d = np.array([amap[filtered[bp["split"]]["target"][bp["idx_in_split"]]] for bp in base_pairs], dtype=np.int64)

                np.save(scores_dir / f"{subject}_scores.npy", scores_3d)
                np.save(scores_dir / f"{subject}_targets.npy", targets_1d)
                with open(scores_dir / f"{subject}_acc.pkl", "wb") as f:
                    pickle.dump({"subject": subject, "acc_list": acc_list}, f)

    print(f"Done. Written to: {out_dir}")
    if args.no_infer:
        print("   (Inference was disabled: --no_infer)")
    else:
        print(f"   Scores/targets written under: {out_dir / 'outputs' / model_tag / args.regime}")


if __name__ == "__main__":
    main()
