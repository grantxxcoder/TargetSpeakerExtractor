"""Statistical reading of a mix-back sweep. Is any of the curve real?

    python3 scripts/analyse_alpha_sweep.py --sweep experiments/results/2026-09-21-mixback-baseline
    python3 scripts/analyse_alpha_sweep.py --sweep ... --listener gemini-3.5-transcribe
    python3 scripts/analyse_alpha_sweep.py --sweep ... --listener A --compare-listener B

WHY THIS EXISTS. `evaluate.py` writes aggregates only, so there is no per-trial
file to resample. Every transcript is in the judge cache though, keyed by the
audio's CONTENT HASH, so per-trial scores are recoverable exactly by hashing each
alpha's wav and looking it up. One source of truth, and no extra API calls.

WHAT IT REPORTS, and the second and third are the ones that matter.

1. THE CURVE, with a paired bootstrap CI on every cell.

2. WHERE THE MINIMUM IS -- as a DISTRIBUTION, not a point. Resample the trials,
   recompute the whole curve, record which alpha won; repeat. "alpha=0 wins in
   62 % of resampled test sets" is the honest statement. A single argmin off one
   sample is exactly the kind of claim that does not replicate, and it also
   sidesteps the multiple-comparison problem that pairwise testing creates.

3. LISTENER INTERACTION -- the panel's actual hypothesis. With
   --compare-listener it tests whether the SHAPE differs between two listeners:

       [WER_A(a) - WER_A(b)] - [WER_B(a) - WER_B(b)] != 0

   a difference-in-differences, bootstrapped on shared trials. This is the test
   behind "the best knob depends on who is listening". Nothing else on the page
   tests it, and without it the panel has an anecdote rather than a finding.

PAIRED THROUGHOUT. One set of resampled trial indices per draw, applied to every
alpha cell, so clip difficulty cancels and what is left is the alpha effect.

WHAT THE INTERVALS DO AND DO NOT INCLUDE. They include judge re-measurement
noise, because each cell is a single (k=1) reading and that noise is already
inside the per-trial spread being resampled. They do NOT include training
variance -- every alpha here shares one checkpoint, so that term is genuinely
absent rather than merely unmeasured, which is the methodological advantage of a
sweep over a family of retrained models (decisions-pending.md 2026-09-16).

Corpus-level WER, matching evaluate.py and bootstrap_difference.py: total errors
over total reference words, never the mean of per-clip rates.

decisions-m4.md 2026-09-21.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.judge import (DEFAULT_CACHE, DEFAULT_MODEL_ID,   # noqa: E402
                                         audio_fingerprint)
from src.live_model_metric.lcf_wer import count_errors                      # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCES = REPO_ROOT / "experiments/results/sweep_alpha_rows.json"


def load_cache_by_hash(cache_path, model_id):
    """content-hash -> transcript, for one listener."""
    out = {}
    with open(cache_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row["model"] != model_id:
                continue
            match = re.search(r"\|sha([0-9a-f]+)\|", row["key"])
            if match:
                out[match.group(1)] = row["text"]
    return out


def collect(sweep_root, cache_by_hash, references):
    """alpha -> {trial_id: (errors, reference_words)}. Missing cells are skipped
    and counted, never silently imputed."""
    curves, missing = {}, 0
    for alpha_dir in sorted(Path(sweep_root).glob("alpha*")):
        alpha = float(alpha_dir.name[len("alpha"):])
        cells = {}
        for trial_dir in sorted(alpha_dir.iterdir()):
            wav = trial_dir / "estimate.wav"
            if not wav.exists():
                continue
            reference = references.get(trial_dir.name)
            if reference is None:
                continue
            text = cache_by_hash.get(audio_fingerprint(wav))
            if text is None:
                missing += 1
                continue
            counts = count_errors(reference, text)
            if counts.reference_word_count:
                cells[trial_dir.name] = (counts.total_errors,
                                         counts.reference_word_count)
        if cells:
            curves[alpha] = cells
    return curves, missing


def collect_from_sweep_json(path, references):
    """Load a listener's curve from a sweep_alpha_rows.json instead of the cache.

    The 2026-09-01 Whisper sweep stored (trial, alpha) -> transcript directly.
    Joining on (trial_id, alpha) rather than on the audio's content hash is
    valid here and only here: that sweep blended the SAME formula over the SAME
    source checkpoint (2026-09-01-est-sir0-5000), so cell (t, a) is the same
    audio even though the file on disk is long gone.

    It gives the panel its first listener comparison for nothing, and the
    comparison is the informative one -- small.en is DETERMINISTIC, so any shape
    difference against the judge cannot be judge noise dressed up.
    """
    curves = {}
    for row in json.loads(Path(path).read_text()):
        reference = references.get(row["tid"])
        if reference is None:
            continue
        counts = count_errors(reference, row["text"])
        if counts.reference_word_count:
            curves.setdefault(float(row["alpha"]), {})[row["tid"]] = (
                counts.total_errors, counts.reference_word_count)
    return curves, 0


def corpus_wer(cells, trials):
    errors = sum(cells[t][0] for t in trials)
    words = sum(cells[t][1] for t in trials)
    return 100.0 * errors / words if words else float("nan")


def holm(pvalues):
    """Holm-Bonferroni. Controls the chance of ANY false positive across the
    family. Five alphas give four comparisons against the reference; testing
    them each at 0.05 would fail ~19 % of the time under a true null."""
    order = np.argsort(pvalues)
    m = len(pvalues)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(running, 1.0)
    return adjusted


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sweep", required=True, help="a make_mixback.py out-root")
    parser.add_argument("--listener", default=DEFAULT_MODEL_ID)
    parser.add_argument("--compare-listener", default=None,
                        help="second listener (a judge model id); runs the "
                             "interaction test")
    parser.add_argument("--compare-asr", action="store_true",
                        help="shorthand for --compare-sweep-json "
                             "experiments/results/sweep_alpha_rows.json, the "
                             "canonical 2026-09-01 offline-ASR sweep. Takes no "
                             "value, so it survives a terminal that wraps long "
                             "pasted lines.")
    parser.add_argument("--compare-sweep-json", default=None,
                        help="second listener taken from a sweep_alpha_rows.json "
                             "instead of the judge cache -- i.e. the offline ASR. "
                             "Free, and small.en is DETERMINISTIC, so a shape "
                             "difference cannot be judge noise.")
    parser.add_argument("--reference-alpha", type=float, default=1.0,
                        help="the alpha every other is tested against. 1.0 = the "
                             "current model untouched")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--references", default=str(REFERENCES))
    parser.add_argument("--draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.compare_asr and not args.compare_sweep_json:
        args.compare_sweep_json = str(REFERENCES)

    references = {r["tid"]: r["target"]
                  for r in json.loads(Path(args.references).read_text())}
    curves, missing = collect(args.sweep,
                              load_cache_by_hash(args.cache, args.listener),
                              references)
    if not curves:
        raise SystemExit(f"no scored cells for {args.listener} under {args.sweep}. "
                         f"Has evaluate.py run on these directories?")

    alphas = sorted(curves)
    shared = sorted(set.intersection(*(set(curves[a]) for a in alphas)))
    print(f"listener {args.listener}   seed {args.seed}   draws {args.draws}")
    print(f"{len(alphas)} alphas, {len(shared)} trials scored in EVERY cell"
          f"{f', {missing} cells not yet scored' if missing else ''}\n")
    if len(shared) < 30:
        print("  WARNING: fewer than 30 shared trials. Intervals will be wide "
              "and the argmin distribution close to meaningless.\n")

    rng = np.random.default_rng(args.seed)
    index = np.arange(len(shared))
    draws = rng.choice(index, size=(args.draws, len(shared)), replace=True)

    # One resample per draw, applied to every cell. This is what makes it paired.
    boot = np.empty((args.draws, len(alphas)))
    for j, alpha in enumerate(alphas):
        err = np.array([curves[alpha][t][0] for t in shared], dtype=float)
        wrd = np.array([curves[alpha][t][1] for t in shared], dtype=float)
        boot[:, j] = 100.0 * err[draws].sum(axis=1) / wrd[draws].sum(axis=1)

    print("1. THE CURVE  (corpus LCF-WER, lower is better)")
    print(f"   {'alpha':>6} {'WER':>8} {'95% CI':>18}")
    point = []
    for j, alpha in enumerate(alphas):
        value = corpus_wer(curves[alpha], shared)
        point.append(value)
        lo, hi = np.percentile(boot[:, j], [2.5, 97.5])
        print(f"   {alpha:6g} {value:8.2f}   [{lo:6.2f}, {hi:6.2f}]")

    adjusted = np.ones(max(len(alphas) - 1, 1))
    if args.reference_alpha in curves:
        ref = alphas.index(args.reference_alpha)
        print(f"\n2. EACH ALPHA vs alpha={args.reference_alpha:g}  "
              f"(paired; negative = better than the reference)")
        others = [j for j in range(len(alphas)) if j != ref]
        diffs = [boot[:, j] - boot[:, ref] for j in others]
        pvals = np.array([2 * min((d <= 0).mean(), (d >= 0).mean()) for d in diffs])
        pvals = np.clip(pvals, 1.0 / args.draws, 1.0)
        adjusted = holm(pvals)
        print(f"   {'alpha':>6} {'diff':>8} {'95% CI':>18} {'p':>8} {'p(Holm)':>9}  verdict")
        for slot, j in enumerate(others):
            d = diffs[slot]
            lo, hi = np.percentile(d, [2.5, 97.5])
            call = "significant" if adjusted[slot] < 0.05 else "not distinguishable"
            print(f"   {alphas[j]:6g} {point[j] - point[ref]:8.2f}   "
                  f"[{lo:6.2f}, {hi:6.2f}] {pvals[slot]:8.4f} "
                  f"{adjusted[slot]:9.4f}  {call}")
        print("   Holm controls the chance of ANY false positive across the family.")

    print("\n3. WHERE IS THE MINIMUM?  (share of resampled test sets each alpha wins)")
    wins = np.bincount(boot.argmin(axis=1), minlength=len(alphas)) / args.draws

    # NULL CALIBRATION, and it is not optional. MEASURED on synthetic data
    # 2026-09-21: a curve with NO real effect at all -- five cells drawn from
    # one distribution -- still gave 55 % of the win share to a single alpha and
    # "interior 60 %". The bootstrap resamples THIS sample, so whichever cell
    # happened to come out lowest wins most redraws. Read raw, the win share
    # manufactures an optimum out of noise.
    #
    # So: rescale every cell's errors to share the grand-mean WER, keeping each
    # cell's own per-trial spread, and bootstrap THAT. It answers "how lopsided
    # would this picture look if nothing were going on?"
    err_all = {a: np.array([curves[a][t][0] for t in shared], dtype=float)
               for a in alphas}
    wrd_all = {a: np.array([curves[a][t][1] for t in shared], dtype=float)
               for a in alphas}
    grand = np.mean([100.0 * err_all[a].sum() / wrd_all[a].sum() for a in alphas])
    null = np.empty((args.draws, len(alphas)))
    for j, alpha in enumerate(alphas):
        cell = 100.0 * err_all[alpha].sum() / wrd_all[alpha].sum()
        scaled = err_all[alpha] * (grand / cell if cell else 1.0)
        null[:, j] = 100.0 * scaled[draws].sum(axis=1) / wrd_all[alpha][draws].sum(axis=1)
    null_wins = np.bincount(null.argmin(axis=1), minlength=len(alphas)) / args.draws

    print(f"   {'alpha':>6} {'observed':>9} {'under null':>11}")
    for j, alpha in enumerate(alphas):
        bar = "#" * int(round(wins[j] * 30))
        print(f"   {alpha:6g} {100 * wins[j]:8.1f} % {100 * null_wins[j]:10.1f} %  {bar}")
    best = int(np.argmax(wins))
    interior, null_interior = wins[1:-1].sum(), null_wins[1:-1].sum()
    print(f"   Best guess alpha={alphas[best]:g}, winning {100 * wins[best]:.1f} % "
          f"(null would give it {100 * null_wins[best]:.1f} %).")
    print(f"   Minimum INTERIOR in {100 * interior:.1f} % of draws "
          f"(null: {100 * null_interior:.1f} %).")

    significant = (args.reference_alpha in curves) and bool((adjusted < 0.05).any())
    if not significant:
        print("\n   *** DO NOT READ THE WIN SHARE ABOVE AS A RESULT. ***")
        print("   No alpha survived Holm correction against the reference, so the")
        print("   curve is flat as far as this data can tell. A flat curve still")
        print("   produces a lopsided win share -- that is the bootstrap tracking")
        print("   one sample's accident, not an optimum. Report 'no detectable")
        print("   difference between alphas', and quote the CIs, not the argmin.")
    elif interior <= null_interior:
        print("\n   -> No evidence for an INTERIOR optimum: the observed interior")
        print("      share does not exceed what a flat curve would produce. Some")
        print("      alpha differs from the reference, but the best setting is an")
        print("      endpoint -- mix-back is a trade, not a free lunch.")
    else:
        print(f"\n   -> Interior optimum supported: {100 * interior:.1f} % against a "
              f"null of {100 * null_interior:.1f} %,")
        print("      and at least one alpha survives Holm. This is the case that")
        print("      justifies a per-clip alpha head.")

    if args.compare_listener or args.compare_sweep_json:
        if args.compare_sweep_json:
            other_curves, _ = collect_from_sweep_json(args.compare_sweep_json,
                                                      references)
            args.compare_listener = f"sweep-json:{Path(args.compare_sweep_json).name}"
        else:
            other_curves, _ = collect(
                args.sweep, load_cache_by_hash(args.cache, args.compare_listener),
                references)
        common_alphas = sorted(set(alphas) & set(other_curves))
        common_trials = sorted(set(shared).intersection(
            *(set(other_curves[a]) for a in common_alphas)))
        print(f"\n4. INTERACTION: does the SHAPE differ between listeners?")
        print(f"   {args.listener}  vs  {args.compare_listener}")
        print(f"   {len(common_trials)} trials scored by both in every cell")
        if len(common_trials) < 30 or len(common_alphas) < 2:
            print("   Not enough shared coverage. Score both listeners first.")
            return
        idx = np.arange(len(common_trials))
        d2 = rng.choice(idx, size=(args.draws, len(common_trials)), replace=True)

        def curve(cs):
            out = np.empty((args.draws, len(common_alphas)))
            for j, a in enumerate(common_alphas):
                e = np.array([cs[a][t][0] for t in common_trials], dtype=float)
                w = np.array([cs[a][t][1] for t in common_trials], dtype=float)
                out[:, j] = 100.0 * e[d2].sum(axis=1) / w[d2].sum(axis=1)
            return out

        ca, cb = curve(curves), curve(other_curves)
        base = common_alphas.index(args.reference_alpha) \
            if args.reference_alpha in common_alphas else 0
        print(f"\n   Difference-in-differences against alpha="
              f"{common_alphas[base]:g}. Non-zero = the listeners")
        print(f"   disagree about what that alpha does, i.e. alpha* is "
              f"listener-specific.")
        print(f"   {'alpha':>6} {'DiD':>8} {'95% CI':>18} {'p':>8}")
        any_sig = False
        for j, a in enumerate(common_alphas):
            if j == base:
                continue
            did = (ca[:, j] - ca[:, base]) - (cb[:, j] - cb[:, base])
            lo, hi = np.percentile(did, [2.5, 97.5])
            p = max(2 * min((did <= 0).mean(), (did >= 0).mean()), 1.0 / args.draws)
            any_sig |= p < 0.05
            print(f"   {a:6g} {did.mean():8.2f}   [{lo:6.2f}, {hi:6.2f}] {p:8.4f}")
        print(f"\n   -> {'Listeners DIFFER in shape: alpha* is listener-specific.' if any_sig else 'No detectable shape difference: alpha* may be SEPARABLE (shared shape, per-listener level).'}")
        print("   Feeds the branch table in decisions-m4.md 2026-09-21.")


if __name__ == "__main__":
    main()
