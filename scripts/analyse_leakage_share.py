"""How much of our transcription error is the OTHER speaker's words?

    ../tse_venv/bin/python scripts/analyse_leakage_share.py

D14 records this as the measurement that gates the whole speaker-state family:
"if leakage is not the dominant error, both D10 and head B lose their
motivation". The aggregate ICR that answers it was already computed on
2026-09-04; this script goes underneath the aggregate to the per-trial level,
because an ICR of 50 % says how OFTEN the interferer leaks, not how much of the
damage leakage accounts for.

READS NOTHING NEW. Every transcript comes from experiments/results/
transcripts.csv (`allow_new=False`), so this cannot silently start a 10-minute
ASR pass or score a different trial set than the published numbers did.

THREE MEASURES, none of which the aggregate gives:

  1. Error-mass attribution. Take the content words the listener reported that
     are NOT the target's. Split them into words the interferer actually said
     (leakage) and words neither speaker said (invention/mishearing). Uses the
     same normaliser and stopword list as icr.py, so the two are consistent.

  2. Per-trial rank correlation between how much leaked and how bad the WER
     was. The published comparison is across two systems (n=2) and confounded,
     since WeSep is better at everything at once. Within one system across 103
     trials, leakage and WER vary for reasons that are not "one model is
     better", which is a far stronger test of the same claim.

  3. Leakage quartiles. The WER of the trials that leaked most against those
     that leaked least, within a single system.

  4. The confound control. A positive correlation could be nothing but trial
     difficulty: a heavily overlapped trial both leaks more AND is harder for
     every other reason too. Measure 2 is therefore repeated inside overlap and
     SIR strata, where difficulty is held roughly fixed. If it survives there,
     difficulty is not the explanation.

Citations: ICR and the content-word/stopword treatment are ours
(docs/data/metric-definitions.md 3.2). Word-error decomposition via jiwer.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.evaluate import load_trials, transcribe  # noqa: E402
from src.live_model_metric.icr import (  # noqa: E402
    content_words, interferer_exclusive_content, measure_leakage)
from src.live_model_metric.lcf_wer import count_errors  # noqa: E402

SPLIT, CONDITION = "sir0_val", "both"
MANIFEST_DIR = Path("data/manifests")
MINIMUM_STRATUM = 8   # below this a rank correlation is noise, so it is skipped
OUT = Path("experiments/results/2026-09-12-leakage-share")

SYSTEMS = {
    "floor (mixture)": None,
    "ours (baseline)": "experiments/results/2026-09-04-train-sir0-10000/",
    # The M5 state-teacher arm, added 2026-09-12. Its aggregate says leakage
    # FELL (ICR@2 50.5 -> 46.6) while WER ROSE (59.5 -> 61.2) entirely through
    # insertions (+2.21). Whether those are the SAME trials is the question the
    # aggregate cannot answer and this script can.
    "ours (state teacher)": "experiments/results/2026-09-12-est-state-e6",
    "WeSep": "experiments/results/2026-09-03-est-wesep-tfmap-causal",
    "ceiling (clean)": None,
}


def spearman(xs, ys):
    """Rank correlation, written out rather than pulling in scipy."""
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        i = 0
        while i < len(order):           # average ties, or the tie mass in
            j = i                       # `available_count` biases the answer
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            shared = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[order[k]] = shared
            i = j + 1
        return ranks

    rx, ry = rank(xs), rank(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def _cached_or_gated(paths):
    """Cached transcripts, with a gate-blocked estimate read as "".

    evaluate.py's speech gate answers a speech-free estimate locally, so that
    clip is never transcribed and has no cache row -- and evaluate.py scores it
    as the empty hypothesis (metric-definitions.md 3.1). Reading it as "" here
    keeps this script on the SAME hypothesis as the reported LCF-WER;
    transcribing it would score audio the headline never listened to. Added
    2026-09-23: 1a e15 on sir0_privval gated 18 clips and this script raised.

    Only UNCACHED clips are re-gated, with the same `decide` and VAD, so the
    model loads only when needed. A clip the gate would pass still raises.
    """
    from src.live_model_metric.evaluate import (ASR_MODEL_SIZE, TRANSCRIPT_CACHE,
                                                _cache_key, _load_cache, _resolve)
    from src.live_model_metric.speech_gate import decide, vad_seconds_fn

    cache = _load_cache(_resolve(TRANSCRIPT_CACHE))
    missing = [p for p in paths if p is not None and Path(p).exists()
               and _cache_key(p, ASR_MODEL_SIZE) not in cache]
    blocked = set()
    if missing:
        vad = vad_seconds_fn()
        blocked = {str(p) for p in missing if not decide(p, vad_detect=vad).has_speech}
        print(f"  {len(blocked)} uncached clip(s) gate-blocked, read as \"\"")
    texts = transcribe([None if p is not None and str(p) in blocked else p
                        for p in paths], allow_new=False, verbose=False)
    return ["" if p is not None and str(p) in blocked else t
            for p, t in zip(paths, texts)]


def per_trial(trials, audio_for):
    """One row per trial: its error counts and what leaked into it."""
    paths = [audio_for(t) for t in trials]
    hypotheses = _cached_or_gated(paths)

    rows = []
    for trial, hypothesis in zip(trials, hypotheses):
        if hypothesis is None:
            continue
        errors = count_errors(trial.target_text, hypothesis)
        if errors.reference_word_count == 0:
            continue
        leak = measure_leakage(hypothesis, trial.target_text, trial.interferer_text)

        # Error-mass attribution. Content words reported that the target did
        # not say, split by whether the interferer said them.
        said_by_target = content_words(trial.target_text)
        available = interferer_exclusive_content(trial.target_text,
                                                 trial.interferer_text)
        wrong = content_words(hypothesis) - said_by_target
        rows.append({
            "trial_id": trial.trial_id,
            "wer": 100.0 * errors.total_errors / errors.reference_word_count,
            "substitutions": errors.substitutions,
            "deletions": errors.deletions,
            "insertions": errors.insertions,
            "reference_words": errors.reference_word_count,
            "wrong_content_words": len(wrong),
            "wrong_from_interferer": len(wrong & available),
            "leaked_count": leak.leaked_count,
            "available_count": leak.available_count,
            "leaked_fraction": leak.leaked_fraction,
        })
    return rows


def report(name, rows):
    wrong = sum(r["wrong_content_words"] for r in rows)
    from_interferer = sum(r["wrong_from_interferer"] for r in rows)
    share = 100.0 * from_interferer / wrong if wrong else float("nan")

    usable = [r for r in rows if r["leaked_fraction"] is not None]
    rho = (spearman([r["leaked_fraction"] for r in usable],
                    [r["wer"] for r in usable]) if len(usable) > 2
           else float("nan"))

    ordered = sorted(usable, key=lambda r: r["leaked_fraction"])
    quarter = max(1, len(ordered) // 4)
    low = statistics.fmean(r["wer"] for r in ordered[:quarter])
    high = statistics.fmean(r["wer"] for r in ordered[-quarter:])

    print(f"\n{name}   n={len(rows)} trials")
    print(f"  wrong content words reported        {wrong:6d}")
    print(f"    of those, words the interferer said {from_interferer:6d}"
          f"   = {share:5.1f} %  <- leakage share of the error")
    print(f"    of those, words NEITHER said        {wrong - from_interferer:6d}"
          f"   = {100 - share:5.1f} %")
    print(f"  per-trial rank correlation, leaked vs WER   rho = {rho:+.3f}"
          f"   (n={len(usable)})")
    print(f"  WER of the least-leaky quartile  {low:5.1f} %")
    print(f"  WER of the most-leaky quartile   {high:5.1f} %"
          f"   ({high - low:+.1f} points)")

    return {"n_trials": len(rows), "wrong_content_words": wrong,
            "wrong_from_interferer": from_interferer,
            "leakage_share_of_error_pct": share,
            "spearman_leak_vs_wer": rho, "n_for_correlation": len(usable),
            "wer_low_leak_quartile": low, "wer_high_leak_quartile": high}


STRATA = [
    ("overlap_achieved", [0.01, 0.3, 0.6], ["none", "low", "mid", "high"]),
    ("sir_db", [-1.0, 1.0], ["interferer louder", "balanced", "target louder"]),
]


def stratum_of(manifest_row, field, edges, labels):
    raw = manifest_row.get(field) or ""
    if raw == "":
        return "n/a"
    value = float(raw)
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def confound_control(rows, manifest):
    """Measure 2 again inside strata, where trial difficulty is held fixed."""
    usable = [r for r in rows
              if r["leaked_fraction"] is not None and r["trial_id"] in manifest]
    out = {}
    for field, edges, labels in STRATA:
        print(f"  within {field}:")
        out[field] = {}
        for label in labels:
            subset = [r for r in usable
                      if stratum_of(manifest[r["trial_id"]], field, edges,
                                    labels) == label]
            if len(subset) < MINIMUM_STRATUM:
                print(f"    {label:<20} n={len(subset):3d}   too few to rank")
                continue
            rho = spearman([r["leaked_fraction"] for r in subset],
                           [r["wer"] for r in subset])
            mean_wer = statistics.fmean(r["wer"] for r in subset)
            mean_leak = statistics.fmean(r["leaked_fraction"] for r in subset)
            print(f"    {label:<20} n={len(subset):3d}   rho {rho:+.3f}"
                  f"   mean WER {mean_wer:5.1f} %"
                  f"   mean leaked {100 * mean_leak:5.1f} %")
            out[field][label] = {"n": len(subset), "spearman": rho,
                                 "mean_wer": mean_wer,
                                 "mean_leaked_fraction": mean_leak}
    return out


def parse_systems(argv):
    """`--systems name=dir,name=dir` overrides the built-in SYSTEMS table.

    Added 2026-09-12 so a post-processed or floored estimate directory can be
    measured without editing this file. `floor` and `ceiling` keep their special
    meaning: a name starting with either reads the mixture or the clean stem
    instead of an estimate, so those two still need no directory.
    """
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", default=None,
                    help="comma-separated name=directory pairs; replaces the "
                         "built-in table")
    ap.add_argument("--out", default=None)
    # Added 2026-09-13. The split was hardcoded to sir0_val, the only dev split
    # that existed when this was written. sir0_privval carries 1,421 `both`
    # trials against sir0_val's 103, and the paired bootstrap downstream reads
    # this script's per_trial.json -- so the wider split is unusable for a
    # system comparison until this is an argument. decisions-m3.md 2026-09-13.
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--condition", default=CONDITION)
    args = ap.parse_args(argv)
    systems = None
    if args.systems:
        systems = {}
        for item in args.systems.split(","):
            name, _, directory = item.partition("=")
            systems[name] = directory or None
    return (systems, (Path(args.out) if args.out else None),
            args.split, args.condition)


def main(systems=None, out=None, split=None, condition=None):
    systems = systems or SYSTEMS
    out = out or OUT
    split = split or SPLIT
    condition = condition or CONDITION
    with open(MANIFEST_DIR / f"{split}.csv", newline="") as handle:
        manifest = {r["trial_id"]: r for r in csv.DictReader(handle)}

    summary = {"split": split, "condition": condition, "systems": {}}
    per_trial_dump = {}

    for name, estimate_directory in systems.items():
        trials = load_trials(split, condition=condition,
                             estimate_directory=estimate_directory)
        if name.startswith("floor"):
            audio_for = lambda t: t.mixture          # noqa: E731
        elif name.startswith("ceiling"):
            audio_for = lambda t: t.clean            # noqa: E731
        else:
            audio_for = lambda t: t.estimate         # noqa: E731

        rows = per_trial(trials, audio_for)
        summary["systems"][name] = report(name, rows)
        if not name.startswith("ceiling"):     # nothing leaks, so nothing to rank
            summary["systems"][name]["strata"] = confound_control(rows, manifest)
        per_trial_dump[name] = rows

    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "per_trial.json").write_text(json.dumps(per_trial_dump, indent=2))
    print(f"\nwrote {out}/summary.json and per_trial.json")


if __name__ == "__main__":
    import sys as _sys
    _systems, _out, _split, _condition = parse_systems(_sys.argv[1:])
    main(_systems, _out, _split, _condition)
