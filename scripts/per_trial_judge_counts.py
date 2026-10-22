#!/usr/bin/env python3
"""Per-clip judge error counts for scripts/bootstrap_difference.py, from CACHED judge answers only.

    ../tse_venv/bin/python scripts/per_trial_judge_counts.py \
        --run "extension e21=experiments/results/2026-09-25-est-cuecontext-wer-e21" \
        --run "baseline e11=experiments/results/2026-10-08-est-baseline-matched-e11" \
        --exclude sir0_val-42-000116 --out experiments/results/<date>-paired-<name>/per_trial.json

evaluate.py reports corpus totals only; the paired bootstrap needs each clip's substitutions, deletions,
insertions, reference words and leaked fraction. This calls the same evaluate() with the CLI's judge
defaults and allow_new_transcripts=False, and records the transcripts it scores, so the per-clip rows sum
back to evaluate.py's own LCF-WER (checked and printed). No API call is made: a clip with no cached answer
stops the run, and --exclude drops it from EVERY system so the comparison stays paired.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.live_model_metric.lcf_wer as L  # noqa: E402
from src.live_model_metric.evaluate import evaluate, load_trials  # noqa: E402
from src.live_model_metric.icr import measure_leakage  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=EST_DIR")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--condition", default="both")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--exclude", action="append", default=[], metavar="TRIAL_ID")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    man = args.manifest_dir
    if args.exclude:   # a filtered copy of the manifest, beside the output, so the exclusion is on record
        rows = list(csv.DictReader(open(Path(args.manifest_dir) / f"{args.split}.csv")))
        keep = [r for r in rows if r["trial_id"] not in set(args.exclude)]
        man = args.out.parent / "manifests"
        man.mkdir(exist_ok=True)
        with open(man / f"{args.split}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(keep)
        print(f"excluded {len(rows) - len(keep)} trial(s) from every system: {args.exclude}")

    captured = []
    orig = L.compute_lcf_wer
    L.compute_lcf_wer = lambda refs, hyps: (captured.append((list(refs), list(hyps))), orig(refs, hyps))[1]
    out = {}
    for spec in args.run:
        name, est = spec.split("=", 1)
        captured.clear()
        res = evaluate(split=args.split, condition=args.condition, estimate_directory=est, manifest_dir=str(man),
                       systems=("estimate",), metrics=("content",), allow_new_transcripts=False, listener="judge",
                       speech_gate=True, verbose=False,
                       judge_kwargs={"requests_per_minute": 10, "max_new_calls": 600, "repeat": 0})
        trials = load_trials(args.split, args.condition, None, "data", str(man), est, None)
        refs, hyps = captured[-1]
        assert len(trials) == len(refs) == len(hyps)
        rows = []
        for t, ref, hyp in zip(trials, refs, hyps):
            c = L.count_errors(ref, hyp)
            lk = measure_leakage(hyp, t.target_text, t.interferer_text)
            rows.append({"trial_id": t.trial_id, "wer": 100.0 * c.total_errors / max(c.reference_word_count, 1),
                         "substitutions": c.substitutions, "deletions": c.deletions, "insertions": c.insertions,
                         "reference_words": c.reference_word_count, "leaked_fraction": lk.leaked_fraction or 0.0,
                         "empty_response": not (hyp or "").strip()})
        total = 100.0 * sum(r["substitutions"] + r["deletions"] + r["insertions"] for r in rows) \
            / sum(r["reference_words"] for r in rows)
        reported = res.systems["estimate"]["lcf_wer"]
        assert abs(total - reported) < 1e-6, (name, total, reported)
        print(f"{name}: LCF-WER {total:.2f} (= evaluate.py), n={len(rows)}, empty responses "
              f"{sum(r['empty_response'] for r in rows)}")
        out[name] = rows
    args.out.write_text(json.dumps(out, indent=1))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
