"""Is the frozen state teacher MONOTONIC in the thing it is supposed to price?

    ../tse_venv/bin/python scripts/diagnose_state_teacher.py --limit 40

WHY THIS EXISTS. `L_state` asks a frozen teacher "is a non-target voice audible
in this output?" and pushes the answer towards no. For that to be a usable
training signal the teacher's reading must FALL as the interferer is removed.
`derive_w_state.py` already computes three synthetic suppression anchors, and
its own docstring says they exist to show "whether it distinguishes -6 dB of
leakage from -20 dB". Nothing ever checked the SIGN of that distinction --
`oracle_wiring_ok` only asserts oracle < 0.5 x passthrough, which a
non-monotonic teacher passes comfortably.

This script walks the full ladder from "the interferer at full level" down to
"no interferer at all", with the noise bed held fixed, and separates the two
things the existing anchors conflate: removing the INTERFERER and removing the
NOISE. `oracle` in derive_w_state.py is the clean target with no noise, so the
gap between it and `partial_20db` mixes both.

    interferer only     target + 1.0 x interferer + noise   (the raw mixture)
    ...                 target + beta x interferer + noise
    target + noise      target + 0.0 x interferer + noise   <- the new anchor
    target              target alone, no noise              (derive's `oracle`)
    noise               the noise bed alone, no speech

Stems are additive by construction (verified 2026-09-10: mixture == target +
interferer + noise exactly, because each stem is stored at the gain it
contributes), so every rung is a real suppression level rather than an
approximation. They carry no masking artefact, which a real extractor output
does -- so this measures the teacher's response to LEAKAGE, cleanly, and says
nothing about its response to spectral holes.

Reported per rung: the mean probability the teacher assigns to "a non-target is
audible", and `L_state` itself (binary cross-entropy against zero, the exact
quantity the training term minimises).

Nothing here trains and nothing is written into any model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.state_teacher import NON_TARGET_AUDIBLE, StateTeacher  # noqa: E402
from src.run_log import timed  # noqa: E402

# Interferer attenuations, in dB. -inf is "no interferer", which is the rung
# derive_w_state.py is missing and the one that separates leakage from noise.
LADDER_DB = (0.0, -6.0, -12.0, -20.0, -30.0)


def read(path, sample_rate):
    audio, rate = sf.read(str(path), dtype="float32", always_2d=False)
    assert rate == sample_rate, f"{path} is {rate} Hz, expected {sample_rate}"
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def score(teacher, waveform, enrolment_embedding):
    """Mean P(non-target audible) and mean L_state over the clip's windows."""
    with torch.no_grad():
        logits = teacher(waveform[None], enrolment_embedding)[..., NON_TARGET_AUDIBLE]
        probability = torch.sigmoid(logits).mean()
        loss = F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="mean")
    return float(probability), float(loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="models/state_detector_notebook.pt")
    ap.add_argument("--ecapa-dir", default="../ecapa_pretrained")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--data-root", default="data/rendered")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/results/2026-09-12-state-teacher-monotonicity")
    args = ap.parse_args()

    device = torch.device(args.device)
    teacher = StateTeacher(args.teacher, args.ecapa_dir, device=device)
    print(f"teacher {teacher.describe()}\n")

    root = Path(args.data_root) / args.split
    trials = sorted(d for d in root.iterdir() if (d / "mixture.wav").exists())[:args.limit]

    rungs = [f"{int(db)}dB" for db in LADDER_DB] + ["no interferer", "target only", "noise only"]
    probabilities = {r: [] for r in rungs}
    losses = {r: [] for r in rungs}

    for count, directory in enumerate(trials, 1):
        mixture    = read(directory / "mixture.wav", args.sample_rate)
        target     = read(directory / "target.wav", args.sample_rate)
        interferer = read(directory / "interferer.wav", args.sample_rate)
        enrolment  = read(directory / "enrollment.wav", args.sample_rate)
        noise = mixture - target - interferer

        embedding = teacher.embed_enrolment(enrolment[None].to(device))

        signals = {f"{int(db)}dB": target + (10.0 ** (db / 20.0)) * interferer + noise
                   for db in LADDER_DB}
        signals["no interferer"] = target + noise
        signals["target only"] = target
        signals["noise only"] = noise

        for name, waveform in signals.items():
            p, l = score(teacher, waveform.to(device), embedding)
            probabilities[name].append(p)
            losses[name].append(l)

        if count % 10 == 0:
            print(f"  {count}/{len(trials)} trials", flush=True)

    print(f"\n{len(trials)} trials of {args.split}. Lower is 'no second voice heard'.\n")
    print(f"{'rung':<18}{'P(non-target)':>15}{'L_state':>12}")
    print("-" * 45)
    summary = {}
    for name in rungs:
        p, l = float(np.mean(probabilities[name])), float(np.mean(losses[name]))
        summary[name] = {"p_non_target": p, "L_state": l}
        print(f"{name:<18}{p:>15.4f}{l:>12.4f}")

    # THE CHECK. Removing the interferer must LOWER the reading at every step.
    ladder = [summary[f"{int(db)}dB"]["L_state"] for db in LADDER_DB]
    monotonic = all(b <= a for a, b in zip(ladder, ladder[1:]))
    print(f"\nmonotonic in interferer level: {'YES' if monotonic else 'NO'}")
    if not monotonic:
        print("  L_state does not fall as the interferer is removed, so the "
              "gradient it supplies does not point at removing the interferer. "
              "As a training signal for leakage this teacher is INVALID.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(
        {"teacher": json.loads(teacher.describe()), "split": args.split,
         "n_trials": len(trials), "ladder_db": list(LADDER_DB),
         "monotonic_in_interferer": monotonic, "rungs": summary}, indent=2))
    print(f"\nwrote {out}/results.json")


if __name__ == "__main__":
    with timed("scripts/diagnose_state_teacher.py", scope=lambda: "state teacher suppression ladder, sir0_val"):
        main()
