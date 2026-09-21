"""Which models can this API key actually reach, and which of them take audio?

    python3 scripts/enumerate_models.py                     # list everything
    python3 scripts/enumerate_models.py --audio-only        # just the candidates
    python3 scripts/enumerate_models.py --probe gemini-3.7-flash,gemini-X
                                                            # one real call each

WHY THIS EXISTS. decisions-m4.md 2026-09-21 commits to a PANEL of listeners, and
the panel is budgeted at ~1,854 calls per listener. Budgeting that against a
model ID taken from a docs page is how you discover at call 900 that the ID was
renamed, is not enabled on your tier, or is socket-only and never accepted audio
on this endpoint. Model IDs are not stable and documentation lags the API.

This asks the API instead. Listing is free and takes seconds.

--probe is the second half and it is the one that actually settles it: it sends
ONE real audio call per named model and reports status, latency and whether the
structured response parsed. A model that lists fine can still refuse audio.
Latency is reported because a Pro tier can be several times slower per call than
a Flash tier, and wall clock -- not money -- is the binding cost of the panel.

decisions-m4.md 2026-09-21. Config: experiments/configs/judge_gate.yaml.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.live_model_metric.evaluate import load_trials              # noqa: E402
from src.live_model_metric.judge import DEFAULT_MODEL_ID, Judge     # noqa: E402


# Measured 2026-09-21: this SDK version reports no per-modality field, so every
# one of the 52 listed models came back "unknown" and the flag was useless. These
# patterns do the shortlisting instead. They are a NAME HEURISTIC, not a
# capability query -- they exist to stop you spending probe calls on an image
# model, never to rule a candidate out. Anything unmatched is a candidate.
NOT_AUDIO_IN = ("-tts", "-image", "image-", "lyria", "nano-banana", "gemma",
                "robotics", "computer-use", "deep-research", "antigravity",
                "/aqa", "-translate")

# Socket-only. Judge speaks generateContent, so probing these here FAILS BY
# DESIGN and tells you nothing. They need the Live client (decisions-m4.md
# 2026-09-21) before they can be scored at all.
LIVE_ONLY = ("-live", "native-audio")


def classify(name):
    lowered = name.lower()
    if any(token in lowered for token in LIVE_ONLY):
        return "live-only"
    if any(token in lowered for token in NOT_AUDIO_IN):
        return "not-audio"
    return "candidate"


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", default="aistudio", choices=["aistudio", "vertex"])
    parser.add_argument("--project", default=None)
    parser.add_argument("--audio-only", action="store_true",
                        help="hide models that certainly do not take audio")
    parser.add_argument("--probe", default=None,
                        help="comma-separated model IDs to send ONE real audio "
                             "call each. Costs one clip per model.")
    parser.add_argument("--split", default="sir0_val",
                        help="where --probe takes its one clip from")
    args = parser.parse_args()

    # Credentials and SDK construction live in Judge so there is exactly one
    # copy of that logic; this borrows the client rather than rebuilding it.
    client = Judge(backend=args.backend, project=args.project)._ensure_client()

    print(f"=== models reachable on backend={args.backend} ===\n")
    buckets = {"candidate": [], "live-only": [], "not-audio": []}
    for model in client.models.list():
        name = getattr(model, "name", "?")
        buckets[classify(name)].append(name)

    order = ["candidate", "live-only"] if args.audio_only else \
            ["candidate", "live-only", "not-audio"]
    headings = {
        "candidate": "WHOLE-CLIP CANDIDATES -- probe these",
        "live-only": "LIVE / SOCKET-ONLY -- need the Live client, do NOT probe here",
        "not-audio": "not audio-in (image, TTS, music, text-only)",
    }
    for bucket in order:
        names = buckets[bucket]
        if not names:
            continue
        print(f"  {headings[bucket]}  ({len(names)})")
        for name in names:
            print(f"    {name}")
        print()

    print("Pass BARE ids to --probe and to --judge-model: 'gemini-3.7-flash', not")
    print("'models/gemini-3.7-flash'. The id goes into the judge cache key, so the")
    print("prefixed form is a DIFFERENT listener and re-buys all 653 cached calls.")

    if not args.probe:
        print("\nNothing probed. Re-run with --probe id1,id2 before budgeting a "
              "panel listener on any of these.")
        return

    trial = load_trials(args.split, condition="both", limit=1)[0]
    clip = trial.clean
    print(f"\n=== probing with one clip: {trial.trial_id} / {clip.name} ===\n")
    for model_id in [m.strip() for m in args.probe.split(",") if m.strip()]:
        judge = Judge(model_id=model_id, backend=args.backend,
                      project=args.project, max_new_calls=1)
        started = time.time()
        try:
            status, text = judge.judge(clip)
            elapsed = time.time() - started
            preview = (text or "")[:70].replace("\n", " ")
            print(f"  {model_id:<45} OK   {elapsed:5.1f}s  status={status:<12} {preview!r}")
        except Exception as exc:                                    # noqa: BLE001
            elapsed = time.time() - started
            print(f"  {model_id:<45} FAIL {elapsed:5.1f}s  {type(exc).__name__}: "
                  f"{str(exc)[:90]}")

    print("\nRead the latency column before committing. Panel budget is ~1,854 "
          "calls per listener, so a 5 s/call model is ~2.6 h of wall clock.")


if __name__ == "__main__":
    main()
