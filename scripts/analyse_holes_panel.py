"""Did the two listeners want DIFFERENT settings of the holes-vs-leakage knob?

    python3 scripts/analyse_holes_panel.py
    python3 scripts/analyse_holes_panel.py --judge gemini-3.5-transcribe

THE REGISTERED QUESTION (decisions-m4.md 2026-09-21, written before the run).
The judge punishes leakage 5.6:1 over deletion; `small.en` weighs them 1.1:1.
Prediction: the judge should prefer HARDER sharpening than the ASR -- the same
audio, opposite optima.

WHY A SEPARATE SCRIPT FROM analyse_alpha_sweep.py. That one walks alpha* dirs
under one root. This family is six independently produced estimate directories
and two listeners whose per-trial scores live in two DIFFERENT caches:

  judge     experiments/results/judge_responses.csv, keyed by audio CONTENT HASH
  small.en  experiments/results/transcripts.csv,     keyed by mtime and size

Both are reconstructed exactly, so no scoring is re-bought and no aggregate is
taken on trust.

WHAT IT REPORTS.
1. Both curves side by side, with paired bootstrap CIs.
2. Each listener's argmin as a DISTRIBUTION, against a NULL CALIBRATION. The
   null matters: measured on synthetic data 2026-09-21, five cells drawn from ONE
   distribution still handed 56.9 % of the win share to a single arm. A lopsided
   win share is not evidence of an optimum until it beats its null.
3. THE DIFFERENCE-IN-DIFFERENCES, which is the registered test:
       [judge(arm) - judge(control)] - [asr(arm) - asr(control)]
   Non-zero means the listeners disagree about what that arm DOES. Three
   verdicts -- differ / agree-by-equivalence / underpowered -- because "p > 0.05"
   alone cannot distinguish agreement from an inconclusive test.

WHAT THE INTERVALS INCLUDE. Judge re-measurement noise is inside them: each cell
is k=1, so that noise is already part of the per-trial spread being resampled.
`small.en` is deterministic and contributes sampling noise only, which is why a
DiD against it is tighter than a judge-vs-judge one would be. Training variance
is absent rather than unmeasured -- all six arms are post-processing of ONE
checkpoint.

Corpus-level WER throughout, matching evaluate.py: total errors over total
reference words, never the mean of per-clip rates.
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.lcf_wer import count_errors                  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = REPO_ROOT / "experiments/results/sweep_alpha_rows.json"
ARMS = ["floor0.20", "floor0.10", "floor0.05", "hyst-control",
        "hystfix-mild", "hystfix-sharp"]
CONTROL = "hyst-control"


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def judge_transcripts(cache_path, model_id):
    out = {}
    with open(cache_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model_id:
                continue
            match = re.search(r"\|sha([0-9a-f]+)\|", row["key"])
            if match:
                out[match.group(1)] = row["text"]
    return out


def asr_transcripts(cache_path):
    with open(cache_path, newline="") as handle:
        return {row["key"]: row["text"] for row in csv.DictReader(handle)}


def collect(arms, references, judge, asr, est_root, asr_model="small.en"):
    """{listener: {arm: {trial: (errors, words)}}}. Missing cells are skipped."""
    out = {"judge": {}, "asr": {}}
    for arm in arms:
        base = Path(est_root) / f"2026-09-12-est-{arm}"
        cells_j, cells_a = {}, {}
        for trial, reference in references.items():
            wav = base / trial / "estimate.wav"
            if not wav.exists():
                continue
            text = judge.get(fingerprint(wav))
            if text is not None:
                counts = count_errors(reference, text)
                if counts.reference_word_count:
                    cells_j[trial] = (counts.total_errors,
                                      counts.reference_word_count)
            key = (f"{asr_model}|{trial}|estimate.wav|"
                   f"{int(os.path.getmtime(wav))}|{os.path.getsize(wav)}")
            text = asr.get(key)
            if text is not None:
                counts = count_errors(reference, text)
                if counts.reference_word_count:
                    cells_a[trial] = (counts.total_errors,
                                      counts.reference_word_count)
        out["judge"][arm], out["asr"][arm] = cells_j, cells_a
    return out


def holm(pvalues):
    """Holm-Bonferroni over the five arm-vs-control comparisons per listener."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def boot_matrix(curves, arms, trials, draws):
    errs = {a: np.array([curves[a][t][0] for t in trials], float) for a in arms}
    wrds = {a: np.array([curves[a][t][1] for t in trials], float) for a in arms}
    out = np.empty((draws.shape[0], len(arms)))
    for j, a in enumerate(arms):
        out[:, j] = 100.0 * errs[a][draws].sum(1) / wrds[a][draws].sum(1)
    return out, errs, wrds


def null_matrix(errs, wrds, arms, draws):
    """Every cell rescaled to the grand-mean WER, keeping its own spread."""
    grand = np.mean([100.0 * errs[a].sum() / wrds[a].sum() for a in arms])
    out = np.empty((draws.shape[0], len(arms)))
    for j, a in enumerate(arms):
        cell = 100.0 * errs[a].sum() / wrds[a].sum()
        scaled = errs[a] * (grand / cell if cell else 1.0)
        out[:, j] = 100.0 * scaled[draws].sum(1) / wrds[a][draws].sum(1)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", default="gemini-3.7-flash")
    parser.add_argument("--est-root", default=str(REPO_ROOT / "experiments/results"))
    parser.add_argument("--judge-cache",
                        default=str(REPO_ROOT / "experiments/results/judge_responses.csv"))
    parser.add_argument("--asr-cache",
                        default=str(REPO_ROOT / "experiments/results/transcripts.csv"))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--material", type=float, default=5.0,
                        help="WER points. Agreement is claimed only when the DiD "
                             "interval EXCLUDES this, not merely when it "
                             "contains zero.")
    args = parser.parse_args()

    references = {r["tid"]: r["target"]
                  for r in json.loads(Path(REFERENCES).read_text())}
    data = collect(ARMS, references,
                   judge_transcripts(args.judge_cache, args.judge),
                   asr_transcripts(args.asr_cache), args.est_root)

    arms = [a for a in ARMS if data["judge"].get(a) and data["asr"].get(a)]
    shared = sorted(set.intersection(
        *[set(data[l][a]) for l in ("judge", "asr") for a in arms]))
    print(f"judge={args.judge}   seed={args.seed}   draws={args.draws}")
    print(f"{len(arms)} arms, {len(shared)} trials scored by BOTH listeners "
          f"in EVERY arm\n")
    if len(shared) < 30:
        raise SystemExit("too few shared trials to say anything.")

    rng = np.random.default_rng(args.seed)
    draws = rng.choice(np.arange(len(shared)), size=(args.draws, len(shared)),
                       replace=True)
    boot, stats = {}, {}
    for listener in ("judge", "asr"):
        b, e, w = boot_matrix(data[listener], arms, shared, draws)
        boot[listener] = b
        stats[listener] = (e, w)

    print("1. THE TWO CURVES  (corpus LCF-WER, lower is better)")
    print(f"   {'arm':<16} {'judge':>8} {'95% CI':>16}   {'small.en':>9} {'95% CI':>16}")
    point = {}
    for listener in ("judge", "asr"):
        e, w = stats[listener]
        point[listener] = [100.0 * e[a].sum() / w[a].sum() for a in arms]
    for j, a in enumerate(arms):
        lj, hj = np.percentile(boot["judge"][:, j], [2.5, 97.5])
        la, ha = np.percentile(boot["asr"][:, j], [2.5, 97.5])
        print(f"   {a:<16} {point['judge'][j]:8.2f} [{lj:6.2f},{hj:6.2f}]   "
              f"{point['asr'][j]:9.2f} [{la:6.2f},{ha:6.2f}]")

    print("\n2. WHERE IS EACH LISTENER'S OPTIMUM?  (share of resampled test sets)")
    for listener, label in (("judge", args.judge), ("asr", "small.en")):
        e, w = stats[listener]
        wins = np.bincount(boot[listener].argmin(1), minlength=len(arms)) / args.draws
        nulls = np.bincount(null_matrix(e, w, arms, draws).argmin(1),
                            minlength=len(arms)) / args.draws
        best = int(np.argmax(wins))
        print(f"   {label}:")
        for j, a in enumerate(arms):
            bar = "#" * int(round(wins[j] * 28))
            print(f"     {a:<16} {100*wins[j]:5.1f} %  (null {100*nulls[j]:4.1f} %) {bar}")
        print(f"     -> argmin {arms[best]}, {100*wins[best]:.1f} % vs null "
              f"{100*nulls[best]:.1f} %")

    print(f"\n2b. WITHIN EACH LISTENER: each arm vs {CONTROL}")
    print("    Holm-corrected over the five comparisons. Added 2026-09-21 after")
    print("    Grant asked whether the small movements were significant at all --")
    print("    the curve and the argmin do not answer that, and the answer")
    print("    changed the claim.")
    base = arms.index(CONTROL)
    for listener, label in (("judge", args.judge), ("asr", "small.en")):
        others = [j for j in range(len(arms)) if j != base]
        diffs = [boot[listener][:, j] - boot[listener][:, base] for j in others]
        pvals = np.array([max(2 * min((d <= 0).mean(), (d >= 0).mean()),
                              1.0 / args.draws) for d in diffs])
        adjusted = holm(pvals)
        print(f"\n    {label}:")
        print(f"      {'arm':<16} {'diff':>7} {'95% CI':>18} {'p(Holm)':>9}  verdict")
        for slot, j in enumerate(others):
            lo, hi = np.percentile(diffs[slot], [2.5, 97.5])
            v = "SIGNIFICANT" if adjusted[slot] < 0.05 else "not distinguishable"
            print(f"      {arms[j]:<16} {point[listener][j] - point[listener][base]:7.2f}"
                  f"   [{lo:6.2f},{hi:6.2f}] {adjusted[slot]:9.4f}  {v}")

    print(f"\n3. DIFFERENCE-IN-DIFFERENCES vs {CONTROL}  (the registered test)")
    print("   negative = the judge likes this arm MORE than small.en does")
    print(f"   {'arm':<16} {'DiD':>8} {'95% CI':>18} {'p':>8}  verdict")
    verdicts = []
    for j, a in enumerate(arms):
        if j == base:
            continue
        did = ((boot["judge"][:, j] - boot["judge"][:, base])
               - (boot["asr"][:, j] - boot["asr"][:, base]))
        lo, hi = np.percentile(did, [2.5, 97.5])
        p = max(2 * min((did <= 0).mean(), (did >= 0).mean()), 1.0 / args.draws)
        if p < 0.05:
            v, tag = "DIFFER", "differ"
        elif max(abs(lo), abs(hi)) < args.material:
            v, tag = "agree (equivalent)", "agree"
        else:
            v, tag = "underpowered", "unknown"
        verdicts.append(tag)
        print(f"   {a:<16} {did.mean():8.2f}   [{lo:6.2f},{hi:6.2f}] {p:8.4f}  {v}")
    print()
    if "differ" in verdicts:
        print("   -> The listeners DISAGREE about what this knob does.")
        print("      A setting is worth more to one listener than to the other,")
        print("      which is the listener-specific design rule the panel was for.")
    elif "unknown" in verdicts:
        print("   -> UNDERPOWERED. Not evidence that the listeners agree.")
    else:
        print("   -> Listeners AGREE within the material threshold.")


if __name__ == "__main__":
    main()
