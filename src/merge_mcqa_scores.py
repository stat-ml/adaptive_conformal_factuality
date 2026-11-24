#!/usr/bin/env python
"""Merge the per-subject MCQA outputs (questions, prompts, answer probabilities, targets) of one model
into a single JSONL file with one line per question and prompt."""

import argparse
import json
from pathlib import Path

import numpy as np


CATEGORY_MAP = {
    "marketing": 0,
    "professional_accounting": 1,
    "college_computer_science": 2,
    "formal_logic": 3,
    "high_school_computer_science": 4,
    "computer_security": 5,
    "machine_learning": 6,
    "clinical_knowledge": 7,
    "high_school_biology": 8,
    "anatomy": 9,
    "college_chemistry": 10,
    "college_medicine": 11,
    "professional_medicine": 12,
    "business_ethics": 13,
    "public_relations": 14,
    "management": 15,
}

DEFAULT_SUBJECTS = list(CATEGORY_MAP.keys())


def read_jsonl(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON in {path}, line {line_idx}") from e


def write_jsonl(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_questions_map(path: Path):
    return {
        (row["split"], row["idx_in_split"]): row
        for row in read_jsonl(path)
    }


def load_prompts_map(path: Path):
    return {
        int(row["run_idx"]): row
        for row in read_jsonl(path)
    }


def extract_instruction_only(prompt_add: str) -> str:
    anchor = "You are the world's best expert"
    pos = prompt_add.find(anchor)

    if pos == -1:
        return prompt_add.strip()

    return prompt_add[pos:].strip()


def merge_subject(
    base_dir: Path,
    subject: str,
    regime: str,
    model_tag: str,
    store_full_prompt_no_example: bool = True,
):
    questions_path = base_dir / "questions" / f"{subject}.jsonl"
    prompts_path = base_dir / "prompts" / regime / f"{subject}.jsonl"
    pairs_dir = base_dir / "pairs" / regime / subject
    outputs_dir = base_dir / "outputs" / model_tag / regime

    questions_map = load_questions_map(questions_path)
    prompts_map = load_prompts_map(prompts_path)

    scores = np.load(outputs_dir / f"{subject}_scores.npy")
    targets = np.load(outputs_dir / f"{subject}_targets.npy")

    if scores.ndim != 3 or scores.shape[-1] != 4:
        raise ValueError(
            f"Expected scores shape (P, N, 4), got {scores.shape} for {subject}"
        )

    num_prompts, num_questions, _ = scores.shape

    if len(targets) != num_questions:
        raise ValueError(
            f"{subject}: targets length {len(targets)} != scores N {num_questions}"
        )

    if subject not in CATEGORY_MAP:
        raise KeyError(f"Subject {subject!r} is missing from CATEGORY_MAP")

    category_id = CATEGORY_MAP[subject]
    merged_rows = []

    for run_idx in range(num_prompts):
        if run_idx not in prompts_map:
            raise KeyError(f"{subject}: missing prompt for run_idx={run_idx}")

        prompt_add = prompts_map[run_idx]["prompt_add"]
        instruction = extract_instruction_only(prompt_add)

        pairs_path = pairs_dir / f"run_{run_idx:02d}.jsonl"
        pairs = list(read_jsonl(pairs_path))

        if len(pairs) != num_questions:
            raise ValueError(
                f"{subject}, run_idx={run_idx}: "
                f"pairs length {len(pairs)} != scores N {num_questions}"
            )

        for question_idx, pair_record in enumerate(pairs):
            key = (pair_record["split"], pair_record["idx_in_split"])

            if key not in questions_map:
                raise KeyError(f"{subject}: missing question record for key={key}")

            question_record = questions_map[key]
            asked_question = question_record["question_only"]

            score = scores[run_idx, question_idx]
            pred = int(np.argmax(score))
            target = int(targets[question_idx])

            row = {
                "subject": subject,
                "category": category_id,
                "run_idx": run_idx,
                "j": question_idx,
                "asked_question": asked_question,
                "scores": score.astype(float).tolist(),
                "pred": pred,
                "target": target,
            }

            if store_full_prompt_no_example:
                row["full_prompt_no_example"] = (
                    f"{instruction}\n"
                    f"{asked_question}\n"
                    "The correct answer is option: "
                )

            merged_rows.append(row)

    return merged_rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge ConformalLLM question, prompt, pair, score, and target files."
    )

    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("mcqa/data"),
    )
    parser.add_argument(
        "--regime",
        default="gpt_based",
    )
    parser.add_argument(
        "--model-tag",
        required=True,
        help="Example: google__gemma-3-12b-it",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=DEFAULT_SUBJECTS,
        help="Subjects to merge. Defaults to all known subjects.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. Defaults to BASE_DIR / MODEL_TAG / all_categories_merged_results.jsonl",
    )
    parser.add_argument(
        "--no-full-prompt",
        action="store_true",
        help="Do not store full_prompt_no_example.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    output_path = args.output
    if output_path is None:
        output_path = args.base_dir / args.model_tag / "all_categories_merged_results.jsonl"

    all_rows = []

    for subject in args.subjects:
        rows = merge_subject(
            base_dir=args.base_dir,
            subject=subject,
            regime=args.regime,
            model_tag=args.model_tag,
            store_full_prompt_no_example=not args.no_full_prompt,
        )

        print(f"{subject}: {len(rows)} rows")
        all_rows.extend(rows)

    write_jsonl(all_rows, output_path)

    print(f"\nTotal rows: {len(all_rows)}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()