#!/usr/bin/env bash
# Score the holes-versus-leakage family through one listener.
# decisions-m4.md 2026-09-21 (REGISTERED prediction).
#
#   bash scripts/run_holes_panel.sh asr
#   bash scripts/run_holes_panel.sh gemini-3.7-flash
#   bash scripts/run_holes_panel.sh gemini-3.5-transcribe
#
# WHY A SCRIPT AND NOT A PASTED LOOP. Measured the hard way 2026-09-21: the
# terminal split a pasted one-liner after `--systems estimate`, so --listener,
# --metrics and --out were all silently dropped. The run then used the DEFAULTS
# -- offline ASR, all three metric families, no output file -- and looked like
# it was working for ten minutes per arm while producing nothing and calling the
# wrong listener. A short command cannot be truncated into a different valid
# command.
set -euo pipefail

LISTENER="${1:?usage: run_holes_panel.sh <asr|gemini-model-id> [rpm]}"
RPM="${2:-12}"
R=experiments/results
# PREFIX and ARMS are overridable so the same family can be run on a DIFFERENT
# extractor -- the control that answers "is this rule about listeners, or about
# our model's flat mask?". decisions-m4.md 2026-09-21.
PREFIX="${PREFIX:-2026-09-12-est-}"
ARMS="${ARMS:-floor0.20 floor0.10 floor0.05 hyst-control hystfix-mild hystfix-sharp}"

if [ "$LISTENER" = "asr" ]; then
    TAG=asr
    EXTRA=(--listener asr)
else
    TAG="$LISTENER"
    # --judge-max-new-calls caps each arm at 103 + margin, so a bug cannot run
    # away with the quota. --judge-structured auto lets gemini-3.5-transcribe
    # fall back to schema-free on its 400 without contaminating flash's rows.
    EXTRA=(--listener judge --judge-model "$LISTENER"
           --judge-structured auto --judge-rpm "$RPM" --judge-max-new-calls 120)
fi

N=$(echo $ARMS | wc -w)
echo "listener=$LISTENER  prefix=$PREFIX  arms=$N  -> $((N * 103)) cells"
for arm in $ARMS; do
    OUT="$R/2026-09-21-holes-${LABEL:-}$TAG-$arm"
    echo "=== $arm -> $OUT"
    python3 scripts/evaluate.py \
        --split sir0_val --condition both \
        --est "$R/$PREFIX$arm" \
        --systems estimate --metrics content \
        "${EXTRA[@]}" \
        --out "$OUT"
done
echo "done: $LISTENER"
