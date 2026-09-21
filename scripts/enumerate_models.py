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


def _audio_capable(model):
    """Best-effort read of whether a listed model accepts audio input.

    The SDK does not expose a reliable per-modality flag across versions, so
    this is a HINT for shortlisting, never a substitute for --probe. When the
    field is missing the model is kept, not dropped: a false positive costs one
    probe call, a false negative silently removes a valid candidate.
    """
    modalities = getattr(model, "supported_input_modalities", None)
    if modalities:
        return any("audio" in str(m).lower() for m in modalities)
    actions = getattr(model, "supported_actions", None) or []
    if actions and not any("generate" in str(a).lower() for a in actions):
        return False
    return None            # unknown -- keep it, probe to find out


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
    rows = []
    for model in client.models.list():
        name = getattr(model, "name", "?")
        capable = _audio_capable(model)
        if args.audio_only and capable is False:
            continue
        rows.append((name, capable))
        flag = {True: "audio", False: "no-audio", None: "unknown"}[capable]
        print(f"  {name:<55} {flag}")
    print(f"\n{len(rows)} shown. 'unknown' means the SDK did not say -- probe it.")

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
