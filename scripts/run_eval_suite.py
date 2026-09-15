"""Run the whole evaluation battery on one checkpoint, in the right order.

    ../tse_venv/bin/python scripts/run_eval_suite.py --tag state-e6 \
        --checkpoint models/model_sir0_state-e6.pt \
        --config experiments/configs/bsrnn_state.yaml --dry-run

WHY A DRIVER AND NOT A README. Six scripts have to run in a fixed order, three
of them consume the output directory of the first, and every one of them needs
the same `--split` / `--condition` or the numbers are not comparable. Every
previous evaluation was assembled by hand from `docs/run_times.md`, which is how
`2026-09-06-eval-10000-signal-perceptual` ended up on a different flag set from
`2026-09-04-train-sir0-10000`. This makes "the battery" one command and one
recorded definition.

IT ORCHESTRATES; IT MEASURES NOTHING ITSELF. Every number still comes from the
script that owns it, written to its own results directory with its own meta.
This file adds no metric and changes no default -- if a step is wrong here it is
wrong in the same way it was wrong by hand.

THE ORDER IS NOT ARBITRARY.
  1. estimates      everything downstream reads this directory
  2. signal         SI-SDR / DNSMOS, no network, no API budget
  3. content (ASR)  WER / ICR from the cached whisper transcripts
  4. content (judge) the project metric, LCF. LAST, because it costs API calls
                    and because a mistake in 1-3 is cheaper to find first
  5. rtf            latency, independent of the estimates
  6. cue            enrolment sensitivity, independent of the estimates
  7. leakage        reads the transcripts step 3 produced

RUN IT IN TWO STAGES, NOT ONE. THE GATE IS `asr`.
-------------------------------------------------
The judge costs API calls and the signal step measures the one axis this project
does not optimise, so neither should run before the cheap question is answered.

  STAGE 1, ~33 min, no API budget:   --steps estimates,asr
      ICR@2 and mean_leak are the DIRECT measures of interferer leakage, which
      is 58.5 % of our content-word error mass and correlates +0.622 with
      per-trial WER (2026-09-11). Any arm aimed at leakage either moves them or
      it has failed on its own logic.

      The numbers to beat, `models/model_sir0_10000-e6.pt` on sir0_val `both`,
      n=103 (experiments/results/2026-09-04-train-sir0-10000/results.json):

          LCF-WER 59.52    ICR@2 50.49 %    mean leaked 34.58 %
          floor (raw mixture)   65.22 / 66.99 % / 51.30 %
          ceiling (clean target) 5.85 /  0.00 % /  0.00 %

  STAGE 2, ~31 min + API:   --steps judge,rtf,cue
      Only if stage 1 moved. `signal` is optional and least informative -- it
      scores SI-SDR and DNSMOS, which is the axis a leakage-targeted arm is
      expected to lose on and which the project does not optimise anyway.

WHY RUN ANY OF IT WHEN A TRAINING DIAGNOSTIC LOOKS BAD. Because the training
diagnostics are signal-fidelity measures and the thesis is that signal fidelity
is not the objective (CLAUDE.md). Abandoning an arm on `L_MR` alone would be
exactly the inference this project exists to argue against. Judge it on content,
or do not judge it.

WALL TIMES ARE MEASURED, NOT ESTIMATED -- every one is a row in
docs/run_times.md from the 2026-09-04/06 baseline evaluation on this machine.
`--dry-run` prints them with the commands so a session can be planned before it
is started. `diagnose_cue.py` has no recorded row and is printed as unknown
rather than guessed.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import date
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = sys.executable

# (name, measured wall time, needs the estimate directory, needs the judge API)
STEPS = ("estimates", "signal", "asr", "judge", "rtf", "cue", "leakage")
MEASURED = {
    "estimates": "27 min",    # 2026-09-04, 200 trials, cpu, whole-clip
    "signal":    "17 min",    # 2026-09-06, signal,perceptual
    "asr":       "6 min",     # 2026-09-04, content, cached transcripts
    "judge":     "12 min",    # 2026-09-06, content --listener judge --judge-rpm 10
    "rtf":       "2 min",     # 2026-09-06, chunk 80 ms, 4 threads, cpu
    "cue":       "unmeasured",
    "leakage":   "1.2 s",     # 2026-09-11, reads cached transcripts only
}


def build(args):
    """The commands, in order. Returns [(step, argv, note)]."""
    today = date.today().isoformat()
    results = REPO / "experiments" / "results"
    estimates = results / f"{today}-est-{args.tag}"

    if args.limit:
        # A limited run must not be able to masquerade as a full one six weeks
        # later, so the limit goes in the directory name, not just the meta.
        args.tag = f"{args.tag}-limit{args.limit}"
        estimates = results / f"{today}-est-{args.tag}"
    limit = ["--limit", str(args.limit)] if args.limit else []

    def script(name, *rest):
        return [PYTHON, str(REPO / "scripts" / name), *rest]

    common = ["--split", args.split, "--data-root", args.data_root]
    plan = [
        ("estimates", script(
            "make_estimates.py", *common,
            "--checkpoint", args.checkpoint, "--config", args.config,
            "--condition", args.condition, "--out", str(estimates), *limit),
         "renders one estimate.wav per trial; everything below reads it"),

        ("signal", script(
            "evaluate.py", "--split", args.eval_split, "--condition", args.condition,
            "--est", str(estimates), "--metrics", "signal,perceptual", *limit,
            "--out", str(results / f"{today}-eval-{args.tag}-signal")),
         "SI-SDR and DNSMOS. NOT the project metric -- signal quality is "
         "explicitly not what this project optimises"),

        ("asr", script(
            "evaluate.py", "--split", args.eval_split, "--condition", args.condition,
            "--est", str(estimates), "--metrics", "content", *limit,
            "--out", str(results / f"{today}-eval-{args.tag}-asr")),
         "offline WER / ICR from cached transcripts; writes the transcripts the "
         "leakage step reads"),

        ("judge", script(
            "evaluate.py", "--split", args.eval_split, "--condition", args.condition,
            "--est", str(estimates), "--metrics", "content", "--listener", "judge",
            "--judge-rpm", str(args.judge_rpm), *limit,
            "--out", str(results / f"{today}-eval-{args.tag}-judge")),
         "THE PROJECT METRIC. Costs API calls; record the model id, prompt, "
         "modality and date with the result (CLAUDE.md)"),

        ("rtf", script(
            "measure_rtf.py", "--checkpoint", args.checkpoint, "--config", args.config,
            "--chunk-ms", "80", "--threads", "4", "--device", "cpu",
            "--out", str(results / f"{today}-rtf-{args.tag}")),
         "streaming latency against the 200-300 ms budget"),

        ("cue", script(
            "diagnose_cue.py", *common,
            "--checkpoint", args.checkpoint, "--config", args.config,
            "--n-crops", "200",
            "--out-dir", str(results / f"{today}-cue-{args.tag}")),
         "enrolment sensitivity, gender-stratified. Never report it pooled"),

        ("leakage", script("analyse_leakage_share.py"),
         "how much of the content error is the interferer's words"),
    ]
    return [p for p in plan if p[0] in args.steps]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--tag", required=True,
                    help="short name for this system, e.g. state-e6. Goes into "
                         "every output directory name, so it is how the run is "
                         "found again six weeks later")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml",
                    help="the config the checkpoint was TRAINED with. Only used "
                         "for the drift check and by measure_rtf; build_model "
                         "always reads the config stored inside the checkpoint")
    ap.add_argument("--split", default="sir0", help="for the scripts that take a split name")
    ap.add_argument("--eval-split", default="sir0_val", help="for evaluate.py")
    ap.add_argument("--condition", default="both")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--judge-rpm", type=int, default=10)
    ap.add_argument("--limit", type=int, default=None,
                    help="first N trials only, passed to the steps that accept "
                         "it. FOR SMOKE TESTS ONLY -- a limited run is not "
                         "comparable with any published number, and the output "
                         "directory is tagged so it cannot be mistaken for one")
    ap.add_argument("--steps", default=",".join(STEPS),
                    help=f"comma-separated subset of: {', '.join(STEPS)}")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands and their measured wall times, run nothing")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing step instead of stopping. OFF "
                         "by default: step 1 feeding steps 2-4 means a silent "
                         "failure there produces three empty evaluations")
    args = ap.parse_args()

    args.steps = [s.strip() for s in args.steps.split(",") if s.strip()]
    unknown = [s for s in args.steps if s not in STEPS]
    if unknown:
        ap.error(f"unknown step(s) {unknown}; choose from {list(STEPS)}")

    if not (REPO / args.checkpoint).exists() and not Path(args.checkpoint).exists():
        ap.error(f"no checkpoint at {args.checkpoint}")

    if "judge" in args.steps and not args.dry_run and not os.environ.get("GEMINI_API_KEY"):
        print("WARNING: GEMINI_API_KEY is not set, so the judge step will fail. "
              "Put it in .env and export it, or drop 'judge' from --steps.\n",
              file=sys.stderr)

    plan = build(args)
    print(f"\n{len(plan)} step(s) for {args.tag}  ({args.checkpoint})\n")
    for name, argv, note in plan:
        print(f"--- {name}   [measured: {MEASURED[name]}]")
        print(f"    {note}")
        print(f"    {' '.join(shlex.quote(a) for a in argv)}\n")

    if args.dry_run:
        print("dry run: nothing executed.")
        return

    for name, argv, _ in plan:
        print(f"\n===== {name} =====", flush=True)
        result = subprocess.run(argv, cwd=REPO)
        if result.returncode != 0:
            print(f"\n{name} FAILED with exit code {result.returncode}",
                  file=sys.stderr)
            if not args.keep_going:
                sys.exit(result.returncode)


if __name__ == "__main__":
    main()
