#!/usr/bin/env bash
# Long-form QA results of the paper: Tables 1-7 and Figure 1.
#
# Usage: scripts/reproduce_longform.sh [step ...]     (default: all steps, in this order)
#   table1     PR-AUC of the claim-level uncertainty scores                   (Table 1)
#   main       Original, Adaptive, Group-wise; Mistral and Gemma, 100 splits   (Tables 2, 3, 5, 6)
#   boosted    Boosted on the same splits                                      (Tables 2, 3, 5, 6)
#   highcov    target coverage 0.95, Mistral                                   (Table 7)
#   ablation   one setting of Adaptive changed at a time, Mistral, 50 splits   (Table 4)
#   fig1       calibration on Inventions and Landmarks, Mistral                (Figure 1)
#   tables     tables and Figure 1 from the results
#
# Environment variables: JOBS (splits run in parallel, default: all CPU cores), RESULTS (default results/longform).
set -euo pipefail
cd "$(dirname "$0")/.."
JOBS=${JOBS:-$(nproc)}
RESULTS=${RESULTS:-results/longform}
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" PYTHONWARNINGS=ignore
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
log() { echo "[$(date +%H:%M:%S)] $*"; }
run() { python src/run_longform.py --n-jobs "$JOBS" "$@" > /dev/null; }

step_table1() {
  log "table1"
  python src/claim_score_prauc.py --out "$RESULTS/table1.md"
}

step_main() {
  for m in mistral gemma; do
    log "main: $m"
    run --model $m --out "$RESULTS/main_$m.json"
  done
}

step_boosted() {
  for m in mistral gemma; do
    log "boosted: $m"
    python src/run_boosted.py --n-jobs "$JOBS" --model $m --out "$RESULTS/boosted_$m.json" > /dev/null
  done
}

step_highcov() {
  log "highcov"
  run --model mistral --alphas 0.05 --out "$RESULTS/high_coverage_mistral.json"
}

step_ablation() {
  local A="$RESULTS/ablation"
  abl() { run --model mistral --n-seeds 50 "$@"; }
  log "ablation: PCA components"
  abl --regressors adaptive_d4,adaptive,adaptive_d16,adaptive_d32 --out "$A/pca.json"
  log "ablation: embedding model"
  for e in all_mpnet_base_v2 all_distilroberta_v1 sentence_transformers__all_MiniLM_L12_v2; do
    abl --embeddings data/emb/longqa_ablation/${e}_pca32_embeddings.npy --out "$A/embedding_$e.json"
  done
  log "ablation: cal1/cal2 split"
  for s in 0.25,0.45,0.3 0.35,0.35,0.3 0.4,0.3,0.3 0.45,0.25,0.3; do
    abl --splits $s --out "$A/split_${s//,/_}.json"
  done
  log "ablation: MLP settings"
  abl --regressors adaptive_h16,adaptive_2layers,adaptive_early_stopping,adaptive_bagging --out "$A/mlp.json"
}

step_fig1() {
  log "fig1"
  run --model mistral --categories inventions,landmarks --alphas 0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5 \
      --out "$RESULTS/two_categories_mistral.json"
}

step_tables() {
  log "tables and Figure 1"
  RESULTS_DIR="$RESULTS" python src/make_longform_tables.py
  python src/make_figure1.py --results "$RESULTS/two_categories_mistral.json" --out "$RESULTS/figures/fig1"
}

steps=("$@"); [ ${#steps[@]} -eq 0 ] && steps=(table1 main boosted highcov ablation fig1 tables)
for s in "${steps[@]}"; do "step_$s"; done
log "done: ${steps[*]}"
