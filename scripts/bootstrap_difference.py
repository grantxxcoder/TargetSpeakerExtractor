"""Is a difference between two systems bigger than the test set's own noise?

    ../tse_venv/bin/python scripts/bootstrap_difference.py \
        --per-trial experiments/results/2026-09-12-leakage-share/per_trial.json \
        --reference "ours (baseline)" --compare "ours (state teacher)"

WHY THIS IS NOT OPTIONAL. MEASURED 2026-09-12: the 95 % interval on a LCF-WER
difference between two systems on `sir0_val` `both` (n=103) is about **+-8
points**. The state-teacher extension measured +1.72 and 37 % of resampled test
sets put it on the other side of zero. Every headline delta this project has
quoted is smaller than that interval, so reporting one without this check states
as a finding something the data cannot support. `report-todo.md` #9 anticipated
exactly this: "a system difference smaller than it cannot honestly be claimed."

WHAT IT DOES. Draws n clips with replacement from the n you have, scores BOTH
systems on that same draw, and records the difference. Ten thousand times. The
spread of those differences is how much your answer depends on which clips you
happened to render.

PAIRED, and that matters. Both systems are resampled on the SAME indices every
draw, so clip difficulty cancels and what is left is the difference between the
systems. An unpaired interval would be wider and would understate the test's
power.

WHAT IT DOES NOT COVER. Run-to-run training variance -- the same config at a
different seed. No same-config replicate exists anywhere in
`experiments/results/`, so these intervals are a LOWER BOUND on the real
scatter.

Corpus-level word error rate, matching `evaluate.py`: total errors over total
reference words, not the mean of per-clip rates.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def corpus_wer(rows):
    errors = sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows)
    words = sum(r["reference_words"] for r in rows)
    return 100.0 * errors / max(words, 1)


def mean_field(rows, field):
    return float(np.mean([r[field] for r in rows]))


def report(name, samples, point, lower_is_better=True):
    low, high = np.percentile(samples, [2.5, 97.5])
    opposite = float((samples > 0).mean() if point < 0 else (samples < 0).mean())
    verdict = "INSIDE THE NOISE" if low < 0 < high else "outside the interval"
    direction = "better" if (point < 0) == lower_is_better else "worse"
    print(f"\n{name}")
    print(f"  measured difference        {point:+.2f}   ({direction})")
    print(f"  95 % interval              [{low:+.2f}, {high:+.2f}]")
    print(f"  resamples with other sign  {100*opposite:.1f} %")
    print(f"  verdict                    {verdict}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-trial", required=True,
                    help="per_trial.json from scripts/analyse_leakage_share.py")
    ap.add_argument("--reference", required=True, help="system name to compare AGAINST")
    ap.add_argument("--compare", required=True, help="system name under test")
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = json.load(open(args.per_trial))
    for name in (args.reference, args.compare):
        if name not in data:
            raise SystemExit(f"{name!r} not in {args.per_trial}. "
                             f"Available: {list(data)}")

    reference = {r["trial_id"]: r for r in data[args.reference]}
    compare = {r["trial_id"]: r for r in data[args.compare]}
    ids = sorted(set(reference) & set(compare))
    if not ids:
        raise SystemExit("the two systems share no trials")

    R = [reference[i] for i in ids]
    C = [compare[i] for i in ids]
    print(f"{len(ids)} paired trials, {args.draws:,} draws, seed {args.seed}")
    print(f"reference {args.reference!r}   compare {args.compare!r}")
    print(f"\ncorpus WER   reference {corpus_wer(R):.2f}   compare {corpus_wer(C):.2f}")

    rng = np.random.default_rng(args.seed)
    wer_samples, leak_samples = [], []
    for _ in range(args.draws):
        idx = rng.integers(0, len(ids), len(ids))
        rr = [R[i] for i in idx]
        cc = [C[i] for i in idx]
        wer_samples.append(corpus_wer(cc) - corpus_wer(rr))
        leak_samples.append(100.0 * (mean_field(cc, "leaked_fraction")
                                     - mean_field(rr, "leaked_fraction")))

    report("LCF-WER", np.array(wer_samples), corpus_wer(C) - corpus_wer(R))
    report("mean leaked %", np.array(leak_samples),
           100.0 * (mean_field(C, "leaked_fraction") - mean_field(R, "leaked_fraction")))

    better = sum(1 for i in ids if compare[i]["wer"] < reference[i]["wer"])
    tied = sum(1 for i in ids if compare[i]["wer"] == reference[i]["wer"])
    print(f"\nper trial: compare better on {better}, worse on "
          f"{len(ids)-better-tied}, tied on {tied}")
    print("\nA result inside the interval is a DIRECTION, not an effect.")


if __name__ == "__main__":
    main()
