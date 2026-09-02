"""How much does the judge's answer change between identical calls?

    python3 scripts/judge_spread.py                          # 5 calls, one mixture
    python3 scripts/judge_spread.py --repeats 5 --clip target
    python3 scripts/judge_spread.py --trial sir0_val-42-000004 --dry-run

WHY THIS MATTERS MORE THAN IT LOOKS. milestones.md M4 sets the gate: run-to-run
spread must be SMALLER than the floor-to-ceiling gap, or the metric cannot tell
systems apart. Nothing in the project has measured that spread, so no delta is
currently interpretable -- not a prompt comparison, and not "the extractor
improved LCF-WER by 6.1 points".

The number this prints is the noise floor. A system difference smaller than it
cannot honestly be claimed.

WHAT IT SUSPENDS. target/mixture/interferer are normally judged once, ever
(the run-once rule). This script opts INTO repeat keying via
repeat_run_once=True, because measuring variance requires deliberately buying
the same answer more than once. That is the only sanctioned reason to do it.

Evidence that motivated this, 2026-09-02: across three prompt variants the two
CEILING controls returned 0.0 % every time, while one FLOOR control moved
88.0 -> 74.0 -> 70.0 %. Ceilings look reproducible and floors do not, but with
one call per prompt there was no way to separate a prompt effect from sampling
noise. This separates them.

decisions-pending.md J2. Config: experiments/configs/judge_gate.yaml.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_model_metric.evaluate import load_trials                # noqa: E402
from src.live_model_metric.judge import (DEFAULT_MODEL_ID, Judge,     # noqa: E402
                                         NewCallLimitReached, QuotaExhausted,
                                         prompt_sha, prompt_text)
from src.live_model_metric.lcf_wer import count_errors                # noqa: E402

CLIP_ATTR = {"mixture": "mixture", "target": "clean", "interferer": "interferer"}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--trial", default=None,
                        help="trial id; default is the first 'both' trial")
    parser.add_argument("--clip", default="mixture", choices=sorted(CLIP_ATTR),
                        help="mixture = the floor (expected noisy); "
                             "target = the ceiling (expected stable)")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--backend", default="aistudio", choices=["aistudio", "vertex"])
    parser.add_argument("--project", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-new-calls", type=int, default=10)
    args = parser.parse_args()

    trials = load_trials(args.split, condition="both", limit=200)
    if args.trial:
        trials = [t for t in trials if t.trial_id == args.trial]
        if not trials:
            sys.exit(f"trial {args.trial} not found in {args.split} 'both' trials")
    trial = trials[0]
    audio = getattr(trial, CLIP_ATTR[args.clip])
    reference = trial.target_text

    print(f"judge   : {args.model}  via {args.backend}")
    print(f"prompt  : sha256[:12]={prompt_sha(args.prompt_file)}")
    print(f"clip    : {trial.trial_id}  {Path(audio).name}  ({args.clip})")
    print(f"repeats : {args.repeats}")
    print(f"reference words: {len(reference.split())}")
    print()

    # repeat_run_once=True is what makes this measurement possible at all.
    judge = Judge(model_id=args.model, requests_per_minute=args.rpm,
                  backend=args.backend, project=args.project,
                  prompt_file=args.prompt_file,
                  max_new_calls=args.max_new_calls,
                  repeat_run_once=True)

    already, new = 0, 0
    for repeat in range(args.repeats):
        probe = Judge(model_id=args.model, prompt_file=args.prompt_file,
                      backend=args.backend, repeat=repeat, repeat_run_once=True,
                      requests_per_minute=0)
        if probe.cached(audio) is not None:
            already += 1
        else:
            new += 1
    print(f"already paid for : {already}")
    print(f"WOULD SPEND      : {new} new call(s)  (cap {args.max_new_calls})")
    print()

    if args.dry_run:
        print("dry run: no API calls made, nothing cached.")
        return

    wers, texts = [], []
    for repeat in range(args.repeats):
        judge.repeat = repeat
        try:
            status, text = judge.judge(audio)
        except (QuotaExhausted, NewCallLimitReached) as exc:
            print(f"  stopped: {exc}")
            break
        errors = count_errors(reference, text)
        if errors.reference_word_count == 0:
            print(f"  r{repeat}: [{status}] no reference words -- WER undefined")
            texts.append(text)
            continue
        wer = errors.total_errors / errors.reference_word_count * 100.0
        wers.append(wer)
        texts.append(text)
        print(f"  r{repeat}: [{status:<10}] WER {wer:6.1f} %   "
              f"S/D/I {errors.substitutions}/{errors.deletions}/{errors.insertions}")

    if len(wers) < 2:
        print("\nNeed at least 2 answers to report a spread.")
        return

    spread = max(wers) - min(wers)
    print()
    print(f"n            : {len(wers)}")
    print(f"mean         : {statistics.mean(wers):.1f} %")
    print(f"min / max    : {min(wers):.1f} % / {max(wers):.1f} %")
    print(f"SPREAD       : {spread:.1f} points")
    if len(wers) > 2:
        print(f"stdev        : {statistics.stdev(wers):.1f} points")
    print(f"identical answers: {len(set(texts))} distinct text(s) in {len(texts)}")
    print()
    print("HOW TO READ IT. This spread is the noise floor of the metric on this")
    print("clip. A system difference smaller than it cannot be claimed. M4's gate")
    print("is that the spread must be smaller than the floor-to-ceiling gap --")
    print("which on sir0_val through the offline ASR is 65.2 - 5.8 = 59.4 points.")


if __name__ == "__main__":
    main()
