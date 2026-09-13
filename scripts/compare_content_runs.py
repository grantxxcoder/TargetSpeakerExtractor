"""Put several content evaluations side by side, with the directions marked.

    ../tse_venv/bin/python scripts/compare_content_runs.py \
        baseline=experiments/results/2026-09-04-train-sir0-10000 \
        floor05=experiments/results/2026-09-12-eval-floor0.05

WHY IT EXISTS. `results.txt` shows one run. Every question this project actually
asks is a comparison, and assembling those by hand is how a metric gets quoted
in the wrong direction -- `fr_at_2` is "fraction of trials with >=2 invented
words", so LOWER is better, and it was misread once already (2026-09-12).
Direction is declared here once, in code, rather than remembered each time.

**It prints differences, not verdicts.** A paired bootstrap on 2026-09-12 put the
95 % interval on a LCF-WER difference at roughly +-8 points on n=103
(decisions-m3.md), so a delta of a point or two is inside the noise. The footer
repeats that, because a table of small deltas invites exactly the reading the
measurement forbids.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# metric -> (label, lower_is_better)
FIELDS = [
    ("lcf_wer",            "LCF-WER",          True),
    ("substitutions",      "  substitutions",  True),
    ("deletions",          "  deletions",      True),
    ("insertions",         "  insertions",     True),
    ("icr_at_2",           "ICR@2 (leakage)",  True),
    ("mean_leak",          "mean leaked %",    True),
    ("fr_at_2",            "FR@2 (fabricated)", True),
    ("invented_per_trial", "invented/trial",   True),
    ("no_response",        "no response %",    True),
]


def load(path):
    path = Path(path)
    candidate = path if path.suffix == ".json" else path / "results.json"
    with open(candidate) as handle:
        return json.load(handle)["systems"]["estimate"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", metavar="NAME=DIR",
                    help="the FIRST one is the reference every other is "
                         "compared against")
    args = ap.parse_args()

    names, systems = [], []
    for item in args.runs:
        if "=" not in item:
            sys.exit(f"expected NAME=DIR, got {item!r}")
        name, directory = item.split("=", 1)
        names.append(name)
        systems.append(load(directory))

    reference = systems[0]
    width = max(len(n) for n in names) + 2

    print(f"\n{'metric':<20}" + "".join(f"{n:>{width+8}}" for n in names))
    print("-" * (20 + len(names) * (width + 8)))
    for key, label, lower_better in FIELDS:
        cells = []
        for i, system in enumerate(systems):
            value = system[key]
            if i == 0:
                cells.append(f"{value:>{width+8}.2f}")
            else:
                delta = value - reference[key]
                better = (delta < 0) if lower_better else (delta > 0)
                mark = " " if abs(delta) < 1e-9 else ("+" if better else "-")
                cells.append(f"{value:>{width+2}.2f} {mark}{abs(delta):>4.2f}")
        print(f"{label:<20}" + "".join(cells))

    print(f"\n'+' = better than {names[0]}, '-' = worse. Lower is better for "
          f"every metric shown.")
    print("The 95 % interval on a LCF-WER difference is about +-8 points at "
          "n=103 (decisions-m3.md 2026-09-12).")
    print("Read these as directions. A delta under ~5 points is not an effect.")


if __name__ == "__main__":
    main()
