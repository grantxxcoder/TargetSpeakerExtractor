#!/usr/bin/env bash
# Selection-free headline on eval_private (decisions-m4.md 2026-10-07).
# Checkpoints were chosen on sir0_val; eval_private was never trained, tuned or
# selected on. Scored ONCE, after all selection: nothing may be selected on it.
#
#   bash eval_private.sh render   # CPU only, no API key: baseline e6, e21, WeSep
#   bash eval_private.sh judge    # Gemini (reads GEMINI_API_KEY from .env), one run each
#
# Same commands, settings and checkpoints as the sir0_val / eval_public runs:
# make_estimates.py (ours), make_estimates_wesep.py under ../wesep_venv (WeSep,
# output_norm false), evaluate.py --listener judge --judge-rpm 10.
set -u
cd "$(dirname "$0")"
PY=../tse_venv/bin/python
WPY=../wesep_venv/bin/python
R=experiments/results
D=2026-10-07
EST_E21=$R/$D-est-private-cuecontext-wer-e21
EST_BASE=$R/$D-est-private-10000-e6
EST_WESEP=$R/$D-est-private-wesep-tfmap-causal
PHASE=${1:?usage: bash eval_private.sh render|judge}
LOG="eval-private-$PHASE-$(date +%F-%H%M).log"

{
case "$PHASE" in
render)
    $PY scripts/make_estimates.py --split sir0priv --condition both \
        --checkpoint models/model_sir0_cuecontext-wer-e21.pt \
        --config experiments/configs/bsrnn_cue_context.yaml --out "$EST_E21" || exit 1
    $PY scripts/make_estimates.py --split sir0priv --condition both \
        --checkpoint models/model_sir0_10000-e6.pt \
        --config experiments/configs/bsrnn_baseline.yaml --out "$EST_BASE" || exit 1
    $WPY scripts/make_estimates_wesep.py --split sir0priv --condition both \
        --pretrain ../wesep_pretrained/tfmap_context_causal_100 --out "$EST_WESEP" || exit 1
    echo "== render done: $EST_E21 $EST_BASE $EST_WESEP"
    ;;
judge)
    set -a; . ./.env; set +a
    [ -n "${GEMINI_API_KEY:-}" ] || { echo "GEMINI_API_KEY is not in .env"; exit 1; }
    # First run also judges the anchors (No processing, Target alone); later runs hit the cache.
    $PY scripts/evaluate.py --split eval_private --condition both --est "$EST_E21" \
        --metrics content --listener judge --judge-rpm 10 \
        --out "$R/$(date +%F)-eval-private-cuecontext-wer-e21-judge" || exit 1
    $PY scripts/evaluate.py --split eval_private --condition both --est "$EST_BASE" \
        --metrics content --listener judge --judge-rpm 10 \
        --out "$R/$(date +%F)-eval-private-10000-e6-judge" || exit 1
    $PY scripts/evaluate.py --split eval_private --condition both --est "$EST_WESEP" \
        --metrics content --listener judge --judge-rpm 10 \
        --out "$R/$(date +%F)-eval-private-wesep-judge" || exit 1
    echo "== judge done"
    ;;
*) echo "usage: bash eval_private.sh render|judge"; exit 1 ;;
esac
} 2>&1 | tee "$LOG"
