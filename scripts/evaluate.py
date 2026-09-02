"""Score every metric on one split, from the command line.

    # everything, all three systems
    python scripts/evaluate.py --split sir0_val \
        --est experiments/results/2026-09-01-est-sir0-5000

    # quick check: content metrics only, 20 trials, nothing new transcribed
    python scripts/evaluate.py --split sir0_val --metrics content --limit 20 --cached-only

    # just the anchors, no model needed
    python scripts/evaluate.py --split sir0_val --systems floor,ceiling

The library behind this is src/live_model_metric/evaluate.py, which the notebook
notebooks/evaluate_metrics.ipynb calls directly. Same code either way.

Timings, so nothing is a surprise: content is seconds when the transcripts are
cached and ~3 s per clip when they are not; signal is about a minute; perceptual
is about 2.8 s per clip per system. A full three-system run on 103 trials is
roughly 15 minutes.

It does NOT render estimates. That is scripts/make_estimates.py and takes ~25
minutes; if the estimate directory is missing this tells you the exact command.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_model_metric.evaluate import (ALL_LISTENERS, ALL_METRICS,  # noqa: E402
                                            ALL_SYSTEMS, ASR, JUDGE,
                                            TRANSCRIPT_CACHE, evaluate)
from src.run_log import timed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--condition", default="both",
                        help="'both' is the only row that has an interferer to remove; "
                             "pass '' for every condition")
    parser.add_argument("--est", "--estimate-directory", dest="estimate_directory",
                        default=None, help="a scripts/make_estimates.py output directory")
    parser.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    parser.add_argument("--metrics", default=",".join(ALL_METRICS),
                        help="content, signal, perceptual (comma-separated)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--cache", default=str(TRANSCRIPT_CACHE))
    parser.add_argument("--cached-only", action="store_true",
                        help="refuse to run the ASR; error instead of a silent "
                             "10-minute transcription pass")
    parser.add_argument("--listener", default=ASR, choices=list(ALL_LISTENERS),
                        help="asr = faster-whisper small.en, the STAND-IN. "
                             "judge = the live model, i.e. an actual LCF result. "
                             "Anchors are cached and never re-bought (run-once "
                             "rule); estimates are keyed by content so a new "
                             "checkpoint is judged fresh.")
    parser.add_argument("--judge-model", default=None,
                        help="override the judge model id")
    parser.add_argument("--judge-prompt", default=None,
                        help="override the judge prompt file. THE PROMPT IS PART "
                             "OF THE INSTRUMENT: a different file has its own "
                             "cache and cannot reuse another prompt's answers.")
    parser.add_argument("--judge-rpm", type=int, default=10)
    parser.add_argument("--judge-max-new-calls", type=int, default=600,
                        help="hard cap on NEW judge calls. Refuses rather than "
                             "truncating; cached work is kept and a re-run "
                             "resumes. sir0_val floor+ceiling+estimate is ~520.")
    parser.add_argument("--no-gate", action="store_true",
                        help="disable the speech gate. ONLY for characterising a "
                             "listener's invention rate on speech-free audio "
                             "(metric-definitions.md 3.3) -- never for scoring a "
                             "system.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    systems = tuple(s.strip() for s in args.systems.split(",") if s.strip())
    metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
    unknown = [m for m in metrics if m not in ALL_METRICS]
    if unknown:
        raise SystemExit(f"unknown metric(s) {unknown}; known: {list(ALL_METRICS)}")
    unknown = [s for s in systems if s not in ALL_SYSTEMS]
    if unknown:
        raise SystemExit(f"unknown system(s) {unknown}; known: {list(ALL_SYSTEMS)}")

    results = evaluate(
        split=args.split,
        condition=args.condition,
        estimate_directory=args.estimate_directory,
        systems=systems,
        metrics=metrics,
        limit=args.limit,
        data_root=args.data_root,
        manifest_dir=args.manifest_dir,
        cache_path=args.cache,
        allow_new_transcripts=not args.cached_only,
        listener=args.listener,
        speech_gate=not args.no_gate,
        judge_kwargs=({k: v for k, v in {
            "model_id": args.judge_model,
            "prompt_file": args.judge_prompt,
            "requests_per_minute": args.judge_rpm,
            "max_new_calls": args.judge_max_new_calls,
        }.items() if v is not None} if args.listener == JUDGE else None),
    )

    print(f"\n{args.split}  condition={args.condition or 'all'}  n={results.n_trials}\n")
    print(results.table())
    if args.listener == ASR:
        print("\nThe listener is an offline ASR standing in for the judge. "
              "NOT a live-model result.")
    else:
        prov = results.provenance
        print(f"\nLIVE-MODEL RESULT. judge={prov['listener']} "
              f"({prov['judge_modality']}) via {prov['judge_backend']}, "
              f"prompt sha256[:12]={prov['judge_prompt_sha256_12']}, "
              f"run {prov['date']}, speech gate {prov['speech_gate']}.")
        print("Closed models change silently: a comparison across dates is "
              "invalid unless re-run.")

    out = Path(args.out or
               f"experiments/results/{date.today().isoformat()}-evaluate-{args.split}")
    results.save(out)
    print(f"\nwritten to {out}/")


if __name__ == "__main__":
    with timed("scripts/evaluate.py", lambda: " ".join(sys.argv[1:])):
        main()
