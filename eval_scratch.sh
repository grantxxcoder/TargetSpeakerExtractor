#!/usr/bin/env bash
# Hail mary (bsrnn_interf_scratch.yaml): one epoch through render -> offline ASR
# -> Gemini judge -> turbo ASR. Usage: bash eval_scratch.sh 8
# EXPLORATORY: the registered epochs are e21/e27 (decisions-m2.md 2026-10-03);
# any other epoch is reported as "selected". One-off driver. Not committed.
set -u
cd "$(dirname "$0")"
E=${1:?usage: bash eval_scratch.sh <epoch>}
T=interf-scratch-e$E
PY=../tse_venv/bin/python
LOG="eval-scratch-e$E-$(date +%F).log"

set -a; . ./.env; set +a    # GEMINI_API_KEY for the judge
[ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY is not in .env"; exit 1; }
[ -f "models/model_sir0_$T.pt" ] || { echo "no models/model_sir0_$T.pt"; exit 1; }

{
    $PY scripts/run_eval_suite.py --tag "$T" --checkpoint "models/model_sir0_$T.pt" \
        --config experiments/configs/bsrnn_interf_scratch.yaml --steps estimates,asr,judge \
        || { echo "FAILED on $T, stopping"; exit 1; }
    echo "== judge done. Turbo ASR (the report's offline ASR) next; Ctrl-C is safe here."
    EST=$(ls -d experiments/results/*-est-$T | tail -1)
    $PY scripts/evaluate.py --config experiments/configs/eval_offline_asr_turbo.yaml \
        --systems estimate --est "$EST" --out "experiments/results/$(date +%F)-eval-asr-turbo-$T"
} 2>&1 | tee "$LOG"
