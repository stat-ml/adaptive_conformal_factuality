#!/usr/bin/env bash
# Multiple-choice QA results of the paper: Figures 2, 3 and 4.
#
# Usage: scripts/reproduce_mcqa.sh [step ...]     (default: fig2 fig3 fig4 figures)
#   merge          merge the per-subject outputs of Mistral and Llama-3.1 (already in mcqa/data)
#   fig2           Llama-2-7B, Marketing + Accounting, 40 splits 0.4/0.4/0.2       (Figure 2)
#   fig3           Mistral and Llama-3.1, 16 subjects, 20 splits 0.3/0.4/0.3       (Figure 3)
#   fig4           Llama-2-7B, 16 subjects, 20 splits into equal thirds            (Figure 4)
#   figures        Figures 2, 3 and 4
# Environment variable: RESULTS (default results/mcqa).
set -euo pipefail
cd "$(dirname "$0")/.."
RESULTS=${RESULTS:-results/mcqa}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore OMP_NUM_THREADS=1
declare -A TAG=([mistral]=mistralai__Mistral-7B-Instruct-v0.2 [llama]=meta-llama__Llama-3.1-8B-Instruct)
LLAMA2=mcqa/data/meta-llama__Llama-2-7b-hf/all_categories_merged_results.jsonl
EMB_ALL=data/emb/mcqa_mqamp32_16categor.npy
EMB_PAIR=data/emb/mcqa_mqamp32_2categor.npy
log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { python src/run_mcqa.py "$@" > /dev/null; }

step_merge() {
  for m in mistral llama; do python src/merge_mcqa_scores.py --model-tag "${TAG[$m]}"; done
}

step_fig2() {
  log "fig2: Llama-2"
  run --scores $LLAMA2 --subjects marketing,professional_accounting --embeddings $EMB_PAIR \
      --n-seeds 40 --splits 0.4,0.4,0.2 --epochs 600 \
      --out "$RESULTS/fig2_llama2.json"
}

step_fig3() {
  for m in mistral llama; do
    log "fig3: $m"
    run --scores mcqa/data/${TAG[$m]}/all_categories_merged_results.jsonl --embeddings $EMB_ALL \
        --n-seeds 20 --splits 0.3,0.4,0.3 --epochs 1300 --out "$RESULTS/fig3_$m.json"
  done
}

step_fig4() {
  log "fig4: Llama-2"
  run --scores $LLAMA2 --embeddings $EMB_ALL --n-seeds 20 --splits 1/3,1/3,1/3 --epochs 1000 \
      --out "$RESULTS/fig4_llama2.json"
}

step_figures() {
  local F="$RESULTS/figures"; mkdir -p "$F"
  declare -A YMIN=([mistral]=0.4 [llama]=0.35)
  for m in mistral llama; do
    python src/make_figure3.py --results "$RESULTS/fig3_$m.json" --ymin ${YMIN[$m]} --out "$F/fig3_$m.pdf"
  done
  python src/make_mcqa_figures.py --results "$RESULTS/fig2_llama2.json" --categories marketing,professional_accounting \
      --labels Marketing,Accounting --all-label "Marketing&Accounting" --out "$F/fig2"
  python src/make_mcqa_figures.py --results "$RESULTS/fig4_llama2.json" --categories professional_medicine,clinical_knowledge \
      --labels Professional_medicine,Clinical_knowledge --all-label "16 Categories" --out "$F/fig4"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(fig2 fig3 fig4 figures)
for s in "${steps[@]}"; do "step_$s"; done
log "done: ${steps[*]}"
