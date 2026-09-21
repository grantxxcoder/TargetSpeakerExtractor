#!/usr/bin/env bash
# The WeSep control, end to end. No environment state to lose.
#
#   bash scripts/run_wesep_control.sh                     # both listeners
#   bash scripts/run_wesep_control.sh gemini-3.7-flash    # just one
#
# WHY THIS EXISTS. The env-var form (PREFIX=... ARMS=... LABEL=...) fails
# silently when a new terminal drops the exports: the command then re-runs the
# ORIGINAL six-arm family, which is fully cached, so it finishes in seconds and
# looks like it worked. Observed 2026-09-21. Hardcoding removes the failure.
#
# THE QUESTION. Is the listener-specific result a property of the LISTENERS, or
# an artefact of our extractor's mask being a broadband volume knob (84.2 % of
# its variance is one number per frame)? WeSep is 25 points ahead and does not
# have that mask. decisions-m4.md 2026-09-21.
set -euo pipefail
R=experiments/results
export PREFIX=2026-09-21-wesep-
export ARMS='hyst-control hystfix-mild hystfix-sharp'
export LABEL=wesep-

for arm in $ARMS; do
    if [ ! -d "$R/$PREFIX$arm" ]; then
        echo "missing $R/$PREFIX$arm -- run scripts/make_wesep_arms.sh first" >&2
        exit 1
    fi
done

for listener in "${@:-asr gemini-3.7-flash}"; do
    echo "########## $listener ##########"
    bash scripts/run_holes_panel.sh "$listener"
done
