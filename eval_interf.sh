#!/usr/bin/env bash
# D10 arm vs control, matched epochs 5 then 3: render, offline ASR, Gemini judge.
# One-off driver for decisions-m2.md 2026-10-02. Not committed.
set -u
cd "$(dirname "$0")"
LOG="eval-interf-$(date +%F).log"
for r in "interf-ft-e5 bsrnn_interf_ft" \
         "interf-ft-control-e5 bsrnn_interf_ft_control" \
         "interf-ft-e3 bsrnn_interf_ft" \
         "interf-ft-control-e3 bsrnn_interf_ft_control"; do
    set -- $r
    ../tse_venv/bin/python scripts/run_eval_suite.py --tag "$1" \
        --checkpoint "models/model_sir0_$1.pt" \
        --config "experiments/configs/$2.yaml" \
        --steps estimates,asr,judge || { echo "FAILED on $1, stopping"; exit 1; }
done 2>&1 | tee "$LOG"
