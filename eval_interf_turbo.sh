#!/usr/bin/env bash
# D10 arm vs control: the REPORT's offline ASR (large-v3-turbo, decisions-m3.md
# 2026-09-28) on the estimates eval_interf.sh rendered. Run after it finishes.
# One-off driver. Not committed.
set -u
cd "$(dirname "$0")"
LOG="eval-interf-turbo-$(date +%F).log"
for t in interf-ft-e5 interf-ft-control-e5 interf-ft-e3 interf-ft-control-e3; do
    EST=$(ls -d experiments/results/*-est-$t 2>/dev/null | tail -1)
    [ -n "$EST" ] || { echo "no estimates for $t yet, stopping"; exit 1; }
    ../tse_venv/bin/python scripts/evaluate.py \
        --config experiments/configs/eval_offline_asr_turbo.yaml \
        --systems estimate --est "$EST" \
        --out "experiments/results/$(date +%F)-eval-asr-turbo-$t" \
        || { echo "FAILED on $t, stopping"; exit 1; }
done 2>&1 | tee "$LOG"
