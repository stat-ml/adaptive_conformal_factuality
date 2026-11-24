# MCQA scores

Answer probabilities of the LLMs on 16 MMLU subjects. The prompting follows
[ConformalLLM](https://github.com/bhaweshiitk/ConformalLLM) (Kumar et al., 2023):
one GPT-generated 1-shot example per subject (`prompt_questions.py`) and the probability of each answer
option from the logits of the option letter.

`run_mmlu.py` writes, under `data/`, the questions, the prompts, the
question-prompt pairs and, with a model, the answer probabilities (`outputs/<model>/`).
`src/merge_mcqa_scores.py` merges them into `data/<model>/all_categories_merged_results.jsonl`,
one line per question, which is what the experiments read.

```bash
python mcqa/run_mmlu.py --regime gpt_based --n_prompts 1 \
    --model_id mistralai/Mistral-7B-Instruct-v0.2 \
    --subjects college_computer_science formal_logic high_school_computer_science machine_learning \
               clinical_knowledge high_school_biology anatomy college_chemistry college_medicine \
               professional_medicine business_ethics public_relations management computer_security \
               professional_accounting marketing
python src/merge_mcqa_scores.py --model-tag mistralai__Mistral-7B-Instruct-v0.2
```

The same commands with `meta-llama/Llama-3.1-8B-Instruct` give the Llama-3.1 scores. For
Llama-2-7B (`meta-llama/Llama-2-7b-hf`) only the merged file is included.
