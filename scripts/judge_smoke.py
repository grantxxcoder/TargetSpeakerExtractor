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
                                         NewCallLimitReached, QuotaExhausted,
                                         prompt_sha, prompt_text)
from src.live_model_metric.speech_gate import (GateDecision,          # noqa: E402
                                               condition_lookup, decide,
                                               log_decision)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="sir0_val")
    parser.add_argument("--n", type=int, default=2,
                        help="'both' trials; each contributes a floor AND a ceiling "
                             "call. Default 2.")
    parser.add_argument("--absent", type=int, default=1,
                        help="absent trials; each contributes a SILENCE call "
                             "(clean target, RMS 0 -- the invented-speech test) and "
                             "an absent-mixture call. Default 1.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-file", default=None,
                        help="prompt to test instead of the default. THE PROMPT IS "
                             "PART OF THE INSTRUMENT: its sha256 is in every cache "
                             "key, so a different file has its own cache entries "
                             "and cannot be served answers from another prompt. "
                             "e.g. src/live_model_metric/judge_prompt_v2.txt")
    parser.add_argument("--rpm", type=int, default=10)
    parser.add_argument("--backend", default="aistudio", choices=["aistudio", "vertex"],
                        help="aistudio = API key (free tier available); "
                             "vertex = GCP project on ADC (higher limits, paid-tier "
                             "data terms). Pick ONE for all scored calls.")
    parser.add_argument("--project", default=None,
                        help="GCP project id, required for --backend vertex")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and the prompt, make no API calls")
    parser.add_argument("--no-gate", action="store_true",
                        help="send speech-free clips to the judge anyway. ONLY for "
                             "characterising the judge's invention rate "
                             "(metric-definitions.md 3.3) -- never for scoring a "
                             "system, because the judge fabricates 17-42 words on "
                             "silence.")
    parser.add_argument("--max-new-calls", type=int, default=40,
                        help="hard cap on NEW (uncached) calls this run. Refuses "
                             "rather than truncating, so an accidental large run "
                             "cannot quietly bill through. Cached work is kept and "
                             "a re-run resumes. Default 40.")
    args = parser.parse_args()

    present = load_trials(args.split, condition="both", limit=args.n)
    absent = load_trials(args.split, condition="interferer_only", limit=args.absent)
    if not present:
        sys.exit(f"no 'both' trials found for split {args.split}")

    # floor = the unprocessed mixture. ceiling = the clean target. Neither is
    # touched by the model, which is why these are scored once and reused.
    plan = []
    for trial in present:
        plan.append((trial, "floor", trial.mixture))
        plan.append((trial, "ceiling", trial.clean))
    for trial in absent:
        # SILENCE is the invented-speech test (B4, metric-definitions.md 3.1).
        # On an absent trial target.wav is measured RMS 0 -- true digital
        # silence -- so the only correct answer is no_speech, and any words
        # that come back are invented. The offline ASR fails this: small.en
        # emits "you" on silence in 8 of 8 absent trials.
        # The absent MIXTURE is a different question and is not this test: it
        # still contains the interferer talking, so speech is correct there.
        plan.append((trial, "silence", trial.clean))
        plan.append((trial, "floor(absent)", trial.mixture))

    print(f"judge   : {args.model}  via {args.backend}")
    print(f"prompt  : {args.prompt_file or 'judge_prompt.txt (default)'}")
    print(f"          sha256[:12]={prompt_sha(args.prompt_file)}  "
          f"({len(prompt_text(args.prompt_file))} chars)")
    print(f"split   : {args.split}   calls: {len(plan)}   rpm cap: {args.rpm}")
    print("-" * 72)
    print(prompt_text(args.prompt_file).strip())
    print("-" * 72)

    # Constructed before the dry-run gate on purpose: Judge builds no client
    # until a call is actually made, so preflight is free -- and the cost is the
    # main thing a dry run should tell you.
    judge = Judge(model_id=args.model, requests_per_minute=args.rpm,
                  backend=args.backend, project=args.project,
                  max_new_calls=args.max_new_calls,
                  prompt_file=args.prompt_file)

    # THE SPEECH GATE, in front of the judge. metric-definitions.md 3.1.
    # A speech-free clip is answered locally and never costs a call. Anchors are
    # decided by construction from the manifest condition; no VAD runs here
    # because judge_smoke only touches anchors.
    lookup = condition_lookup(args.split)
    gate = {}
    for _t, _sys, path in plan:
        gate[str(path)] = (GateDecision(True, "gate-disabled:--no-gate")
                           if args.no_gate else decide(path, lookup(path)))
    blocked = [p for p, d in gate.items() if d.fired]

    # PREFLIGHT. Nothing already paid for is ever bought twice, and nothing the
    # gate blocks is paid for at all.
    payable = [p for _, _, p in plan if not gate[str(p)].fired]
    per_call = [judge.cached(p) is not None for _, _, p in plan]
    already, new = judge.preflight(payable)
    print(f"gate blocked     : {len(blocked)} clip(s) — answered locally, 0 calls")
    print(f"already paid for : {already}")
    print(f"WOULD SPEND      : {new} new call(s)"
          f"{'' if args.max_new_calls is None else f'  (cap {args.max_new_calls})'}")
    if args.max_new_calls is not None and new > args.max_new_calls:
        print(f"  NOTE: plan needs {new} new calls but the cap is "
              f"{args.max_new_calls}; the run will stop at the cap and keep "
              f"everything it bought. Raise --max-new-calls to go further.")
    if new == 0:
        print("\nEverything in this plan is already cached. No calls needed.")
    print()

    if args.dry_run:
        for (trial, system, path), hit in zip(plan, per_call):
            d = gate[str(path)]
            mark = "BLOCKED" if d.fired else ("cached " if hit else "SPEND  ")
            note = f"   {d.reason}" if d.fired else ""
            print(f"  {mark} {system:14s} {trial.trial_id}  {Path(path).name}{note}")
        print("\ndry run: no API calls made, nothing cached.")
        return

    # The offline-ASR column is a CONVENIENCE, not the measurement. allow_new
    # is False so this never silently starts a transcription pass, but a cache
    # miss must not abort the judge calls -- they are the point of the run.
    try:
        asr_texts = transcribe([p for _, _, p in plan], allow_new=False, verbose=False)
    except RuntimeError as exc:
        print(f"note: offline-ASR column unavailable ({exc}); continuing.")
        asr_texts = [None] * len(plan)

    for (trial, system, path), asr_text in zip(plan, asr_texts):
        print(f"\n=== {trial.trial_id}  [{system}]  {Path(path).name}")
        reference = trial.target_text if "absent" not in system else "(target silent)"
        print(f"  reference : {reference[:100]}")
        # The gate applies to BOTH listeners or the comparison is void.
        gate_decision = gate[str(path)]
        if gate_decision.fired:
            log_decision(gate_decision, path, "small.en", split=args.split,
                         condition=lookup(path))
            log_decision(gate_decision, path, "judge", split=args.split,
                         condition=lookup(path))
            print(f"  small.en  : (gate blocked)")
            print(f"  JUDGE     : (gate blocked — {gate_decision.reason}) "
                  f"no call made")
            continue
        print(f"  small.en  : {(asr_text or '(not cached)')[:100]}")
        try:
            status, text = judge.judge(path)
        except QuotaExhausted as exc:
            print(f"  JUDGE     : quota spent -- {exc}")
            break
        except NewCallLimitReached as exc:
            print(f"  JUDGE     : spend cap reached -- {exc}")
            break
        except Exception as exc:                                  # noqa: BLE001
            print(f"  JUDGE     : FAILED -- {type(exc).__name__}: {exc}")
            break
        print(f"  JUDGE     : [{status}] {text[:100]}")

    print(f"\ncalls made: {judge.calls_made}   cache hits: {judge.cache_hits}")
    print(f"raw responses appended to: {judge.cache_path}")


if __name__ == "__main__":
    main()



