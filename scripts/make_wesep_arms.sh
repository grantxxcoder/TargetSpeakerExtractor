#!/usr/bin/env bash
# Apply the SAME sharpening family to WeSep, as the control for
# "is the design rule about listeners, or about our model's flat mask?"
#
#   bash scripts/make_wesep_arms.sh
#
# WeSep is 25 points ahead of our baseline and its mask is not a broadband
# volume knob, so if the rule survives here it is a property of the LISTENERS.
# If it does not, the rule is specific to flat-mask extractors -- narrower, but
# far better found here than in a viva. decisions-m4.md 2026-09-21.
set -euo pipefail
R=experiments/results
SRC="$R/2026-09-03-est-wesep-tfmap-causal"
# Three arms, not six: control and the two sharpening settings. The floor arms
# lost on BOTH listeners, so they buy nothing here and cost 103 calls each.
python3 scripts/postprocess_mask.py --est "$SRC" --out "$R/2026-09-21-wesep-hyst-control"  --down 1.0
python3 scripts/postprocess_mask.py --est "$SRC" --out "$R/2026-09-21-wesep-hystfix-mild"  --hi 1.4 --lo 0.6 --down 0.2
python3 scripts/postprocess_mask.py --est "$SRC" --out "$R/2026-09-21-wesep-hystfix-sharp" --hi 1.5 --lo 0.5 --down 0.0
echo "done. now: PREFIX=2026-09-21-wesep- ARMS='hyst-control hystfix-mild hystfix-sharp' LABEL=wesep- bash scripts/run_holes_panel.sh asr"
