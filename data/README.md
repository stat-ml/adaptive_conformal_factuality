---
license: mit
language:
  - en
tags:
  - uncertainty-quantification
  - conformal-prediction
  - hallucination-detection
  - lm-polygraph
pretty_name: Long-form QA with claim-level uncertainty and factuality labels
---

# Long-form QA data for "Adaptive Conformal Prediction for Improving Factuality of Generations by LLMs"

Data of the long-form QA experiments of [the paper](https://arxiv.org/abs/2604.13991)
(ACML 2026; code: https://github.com/stat-ml/adaptive_conformal_factuality).

For 8 categories (inventions, persons, artworks, books, cities, movies, landmarks, events) and three
models (Mistral-7B-Instruct-v0.2, Llama-3.1-8B-Instruct, Gemma-3-12B-Instruct), 300 prompts per
category were answered by the model. Each answer was split into atomic claims, every claim was labelled
as correct or incorrect by GPT-4o, and claim-level uncertainty scores were computed with
[LM-Polygraph](https://github.com/IINemo/lm-polygraph).

| Model | Prompts | Labelled claims | Incorrect claims |
|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | 2400 | 56,075 | 18.6% |
| Llama-3.1-8B-Instruct | 2400 | 66,026 | 16.7% |
| Gemma-3-12B-Instruct | 2400 | 65,439 | 14.0% |

## Files

`ue_manager_seed1_<category>_<model>`: one LM-Polygraph UE manager per category and model. A manager is
a dictionary saved with `torch.save`; reading it needs `lm-polygraph` 0.3.0 or later:

```python
import torch
manager = torch.load("ue_manager_seed1_artworks_mistral", weights_only=False)
manager["stats"]["input_texts"]                       # prompts
manager["stats"]["greedy_texts"]                      # answers
manager["stats"]["claim_texts_concatenated"]          # claims of all answers
manager["stats"]["claim_input_texts_concatenated"]    # prompt of each claim
manager["gen_metrics"][("claim", "OpenAIFactCheck")]  # GPT-4o label per claim: 0 correct, 1 incorrect, NaN none
manager["estimations"][("claim", "CCP_claim_fact_pref")]  # claim-level uncertainty scores (9 methods)
```

## Citation

```bibtex
@article{rubashevskii2026adaptive,
  title   = {Adaptive Conformal Prediction for Improving Factuality of Generations by Large Language Models},
  author  = {Rubashevskii, Aleksandr and Piatrashyn, Dzianis and Nakov, Preslav and Panov, Maxim},
  journal = {arXiv preprint arXiv:2604.13991},
  year    = {2026}
}
```
