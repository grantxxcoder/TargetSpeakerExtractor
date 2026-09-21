"""Turn the repeat-keyed judge calls into the two numbers that gate the sweep.

    python3 scripts/analyse_judge_spread.py
    python3 scripts/analyse_judge_spread.py --model gemini-3.5-transcribe --date 2026-09-22

WHY A SEPARATE SCRIPT. scripts/judge_spread.py buys the repeats and prints the
spread for ONE clip. Nothing aggregated them, and the aggregate is what decides
how the panel is run -- specifically whether k=1 is enough, which is a
three-fold difference in the whole panel's cost.

THE TWO NUMBERS, AND THE SECOND IS THE ONE THAT MATTERS.

1. PER-CLIP SPREAD. How much one clip's score moves across identical calls.
   M4's registered gate: this must be smaller than the floor-to-ceiling gap
   (59.4 points on sir0_val through the offline ASR). It is a property of the
   instrument and it bounds any PER-TRIAL claim.

2. SE OF THE DIFFERENCE BETWEEN TWO ALPHA CELLS. What actually gates the sweep,
   because the sweep compares alpha cells averaged over every trial, not single
   clips. Averaging n trials shrinks the noise by sqrt(n), so a metric far too
   noisy per clip can still rank alphas cleanly.

   Pooled as RMS of the per-clip standard deviations, NOT their mean. Variances
   add, standard deviations do not, and the per-clip sigmas here are strongly
   heterogeneous -- averaging them understates the true SE (Jensen). Two cells
   are compared on the same trials but the judge's noise is independent between
   them, so the difference carries both variances:

       SE(difference) = sigma_RMS * sqrt(2 / (n * k))

WHAT A BIG PER-CLIP SPREAD DOES *NOT* MEAN. It does not invalidate the metric.
It means per-trial claims are unavailable and aggregate claims are fine, which
is what project-state.md already says in its "Cannot" list. Read the SE column
before concluding anything about cost.

decisions-m4.md 2026-09-21, milestones.md M4.
"""

import argparse
import collections
import csv
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.judge import DEFAULT_CACHE, DEFAULT_MODEL_ID  # noqa: E402
from src.live_model_metric.lcf_wer import count_errors                   # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
# The reference transcripts. sweep_alpha_rows.json carries target text per
# trial and is already on disk; the manifest route needs meta.json per trial.
REFERENCES = REPO_ROOT / "experiments/results/sweep_alpha_rows.json"

# sir0_val, offline ASR: floor 65.22, ceiling 5.85. decisions-m3.md.
FLOOR_TO_CEILING_GAP = 59.4


def word_error_rate(reference, hypothesis):
    counts = count_errors(reference, hypothesis)
    if not counts.reference_word_count:
        return None
    return 100.0 * counts.total_errors / counts.reference_word_count


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--date", default=None, help="restrict to one run_date")
    parser.add_argument("--clip", default="mixture.wav",
                        help="mixture.wav by default: 2026-09-02 found ceilings "
                             "reproducible and floors not, so the variance lives "
                             "on the mixtures")
    parser.add_argument("--references", default=str(REFERENCES))
    args = parser.parse_args()

    references = {r["tid"]: r["target"]
                  for r in json.loads(Path(args.references).read_text())}

    rows = list(csv.DictReader(open(args.cache, newline="")))
    scores = collections.defaultdict(dict)
    for row in rows:
        if row["model"] != args.model:
            continue
        if args.date and row["run_date"] != args.date:
            continue
        if not row["file"].endswith(args.clip):
            continue
        if "|r" not in row["key"]:
            continue                      # not a repeat-keyed call
        repeat = row["key"].rsplit("|r", 1)[-1]
        reference = references.get(row["trial_id"])
        if reference is None:
            continue
        scores[row["trial_id"]][repeat] = word_error_rate(reference, row["text"])

    complete = {t: [v for v in reps.values() if v is not None]
                for t, reps in scores.items()}
    complete = {t: v for t, v in complete.items() if len(v) >= 2}
    if not complete:
        raise SystemExit(f"no repeat-keyed {args.clip} rows for {args.model}. "
                         f"Run scripts/judge_spread.py first.")

    k_observed = min(len(v) for v in complete.values())
    ranges = [max(v) - min(v) for v in complete.values()]
    sds = [statistics.stdev(v) for v in complete.values()]
    variances = [statistics.variance(v) for v in complete.values()]
    sigma_rms = math.sqrt(statistics.mean(variances))
    identical = sum(1 for r in ranges if r == 0)

    print(f"model {args.model}  clip {args.clip}  "
          f"{len(complete)} trials, k>={k_observed}\n")
    print("1. PER-CLIP SPREAD across identical calls (WER points)")
    print(f"   range   mean {statistics.mean(ranges):6.1f}   "
          f"median {statistics.median(ranges):6.1f}   max {max(ranges):6.1f}")
    print(f"   stdev   mean {statistics.mean(sds):6.1f}   "
          f"median {statistics.median(sds):6.1f}   max {max(sds):6.1f}")
    print(f"   sigma (RMS, the one to pool with): {sigma_rms:.1f}")
    print(f"   identical every time: {identical} of {len(ranges)} trials")
    gate = statistics.mean(ranges) < FLOOR_TO_CEILING_GAP
    print(f"\n   M4 GATE: mean range {statistics.mean(ranges):.1f} vs "
          f"floor-to-ceiling {FLOOR_TO_CEILING_GAP} -> "
          f"{'PASS' if gate else 'FAIL'} "
          f"(margin {FLOOR_TO_CEILING_GAP / max(statistics.mean(ranges), 1e-9):.1f}x)")
    if max(ranges) > FLOOR_TO_CEILING_GAP:
        print(f"   NOTE: the WORST clip ({max(ranges):.1f}) exceeds the gap. "
              f"Passes on average, not universally.")

    print("\n2. SE OF THE DIFFERENCE BETWEEN TWO ALPHA CELLS  "
          "(sigma_RMS * sqrt(2/(n*k)))")
    print(f"   {'n':>6} {'k':>3} {'SE':>8} {'95% CI +/-':>12}   calls/listener")
    for n, k in [(103, 1), (103, 2), (103, 3), (50, 1), (200, 1)]:
        se = sigma_rms * math.sqrt(2.0 / (n * k))
        print(f"   {n:6d} {k:3d} {se:8.2f} {1.96 * se:12.2f}   "
              f"{n * 5 * k:>6d}")

    print("\n   The 2026-09-01 mix-back curve spanned 10.5 points end to end "
          "(59.1 to 69.6),")
    print("   with adjacent-alpha steps of roughly 2 to 6 points. Compare the "
          "95% column:")
    print("   a step smaller than it cannot be called, however many alphas "
          "are run.")

    print("\n3. PER-TRIAL alpha*, the Tier 3 question")
    single = sigma_rms * math.sqrt(2.0)
    print(f"   Ranking two alphas WITHIN one trial carries SE {single:.1f} "
          f"points at k=1.")
    needed = 2.0 * (sigma_rms / 3.0) ** 2
    print(f"   Resolving a 3-point within-trial difference would need "
          f"k ~ {needed:.0f} repeats per (trial, alpha) cell.")
    print("   If that is infeasible, per-trial alpha* is not measurable through "
          "this listener,")
    print("   and it has to come from a DETERMINISTIC one (small.en, greedy) "
          "instead.")


if __name__ == "__main__":
    main()
