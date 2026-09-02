"""Five calls. See what the judge actually returns before spending a quota on it.

    python scripts/judge_smoke.py                  # 5 calls, sir0_val
    python scripts/judge_smoke.py --n 3
    python scripts/judge_smoke.py --dry-run        # no API calls, shows the plan

WHY THIS EXISTS SEPARATELY FROM THE GATE. The gate (~150 calls) answers whether
the judge can rank systems. This answers something cheaper and prior: what shape
does the response come back in, does the structured schema hold, and does the
prompt need rewording. Rewording the prompt after the anchor run would void
every anchor call, so it is worth five calls to find out first.

WHAT IT PRINTS, per clip: the raw judge status and text, the cached offline-ASR
transcript for the same audio, and the reference. Reading those three side by
side is the point -- it shows whether the judge is doing something different
from small.en, which is what metric-definitions.md section 1 hypothesises.

decisions-pending.md J2. Config: experiments/configs/judge_gate.yaml.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_model_metric.evaluate import load_trials, transcribe   # noqa: E402
from src.live_model_metric.judge import (DEFAULT_MODEL_ID, Judge,    # noqa: E402
                                         QuotaExhausted, prompt_sha, prompt_text)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--n", type=int, default=2,
                        help="present trials; each contributes a floor AND a "
                             "ceiling call, plus one absent trial. Default 2 -> 5 calls.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--backend", default="aistudio", choices=["aistudio", "vertex"],
                        help="aistudio = API key (free tier available); "
                             "vertex = GCP project on ADC (higher limits, paid-tier "
                             "data terms). Pick ONE for all scored calls.")
    parser.add_argument("--project", default=None,
                        help="GCP project id, required for --backend vertex")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the prompt, make no API calls")
    args = parser.parse_args()

    present = load_trials(args.split, condition="both", limit=args.n)
    absent = load_trials(args.split, condition="interferer_only", limit=1)
    if not present:
        sys.exit(f"no 'both' trials found for split {args.split}")

    # floor = the unprocessed mixture. ceiling = the clean target. Neither is
    # touched by the model, which is why these are scored once and reused.
    plan = []
    for trial in present:
        plan.append((trial, "floor", trial.mixture))
        plan.append((trial, "ceiling", trial.clean))
    for trial in absent:
        plan.append((trial, "floor(absent)", trial.mixture))

    print(f"judge   : {args.model}  via {args.backend}")
    print(f"prompt  : sha256[:12]={prompt_sha()}  ({len(prompt_text())} chars)")
    print(f"split   : {args.split}   calls: {len(plan)}   rpm cap: {args.rpm}")
    print("-" * 72)
    print(prompt_text().strip())
    print("-" * 72)

    if args.dry_run:
        for trial, system, path in plan:
            print(f"  would call  {system:14s} {trial.trial_id}  {Path(path).name}")
        print("\ndry run: no API calls made, nothing cached.")
        return

    judge = Judge(model_id=args.model, requests_per_minute=args.rpm,
                  backend=args.backend, project=args.project)
    asr_texts = transcribe([p for _, _, p in plan], allow_new=False, verbose=False)

    for (trial, system, path), asr_text in zip(plan, asr_texts):
        print(f"\n=== {trial.trial_id}  [{system}]  {Path(path).name}")
        reference = trial.target_text if "absent" not in system else "(target silent)"
        print(f"  reference : {reference[:100]}")
        print(f"  small.en  : {(asr_text or '(not cached)')[:100]}")
        try:
            status, text = judge.judge(path)
        except QuotaExhausted as exc:
            print(f"  JUDGE     : quota spent -- {exc}")
            break
        except Exception as exc:                                  # noqa: BLE001
            print(f"  JUDGE     : FAILED -- {type(exc).__name__}: {exc}")
            break
        print(f"  JUDGE     : [{status}] {text[:100]}")

    print(f"\ncalls made: {judge.calls_made}   cache hits: {judge.cache_hits}")
    print(f"raw responses appended to: {judge.cache_path}")


if __name__ == "__main__":
    main()
