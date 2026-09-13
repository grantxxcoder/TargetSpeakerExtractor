"""Is the mask a per-instant volume knob, a fixed EQ curve, or a real decision?

    ../tse_venv/bin/python scripts/diagnose_mask_structure.py \
        --split sir0_val --n 12 \
        --checkpoint models/model_sir0_10000-e6.pt \
        --out experiments/results/2026-09-13-mask-structure

THE QUESTION. 2026-09-12 measured that 84.2 % of our mask's variance is
explained by one number per frame (`postprocess_mask.py`), and the mask-grid
picture shows vertical stripes running the full height of the spectrogram. But
"not frequency-selective" has two very different readings, and the picture
cannot tell them apart:

  (a) the model applies a flat broadband gain -- no frequency shape at all;
  (b) the model applies a FIXED frequency shape (a baked-in EQ curve, e.g. the
      persistent dark band below 500 Hz) scaled up and down by one gain per
      frame.

Both look like stripes. Only (b) means the model learned anything about
frequency at all, and in neither case can it separate two overlapping voices,
because separation requires the frequency decision to CHANGE from instant to
instant depending on who owns which bin.

THE DECOMPOSITION. Two-way additive (ANOVA-style) on the mask magnitude:

    M(f,t) = mu + eq(f) + gain(t) + interaction(f,t)

  gain(t)        deviation of each frame's mean from the global mean -- the
                 volume knob.
  eq(f)          deviation of each frequency's mean from the global mean -- a
                 fixed spectral shape, constant over the whole clip.
  interaction    what is left. THE ONLY TERM THAT CAN SEPARATE TWO VOICES,
                 because it is the only one that says "this frequency, at this
                 instant, belongs to the target" -- a statement whose truth
                 changes from frame to frame.

The three terms are orthogonal by construction, so their variances sum to the
total and the shares are exact, not a fitted approximation.

THE IDEAL MASK IS DECOMPOSED THE SAME WAY, on the same trials and bins. It is
the correct answer we can compute because we own both stems, and its
interaction share is the target this model would have to reach.

FRAME SELECTION MATTERS AND IS REPORTED THREE WAYS. A clip that is half silence
gives the volume knob a large share for free -- ducking silence is not a skill.
  all        every frame
  speech     frames where the mixture carries real energy
  overlap    frames where BOTH the target and the interferer are active. This is
             the only regime where separation is even a question, and the
             overlap interaction share is the number the hypothesis rests on.

WHAT THIS CANNOT SETTLE. It describes the mask, not the output: the additive
residual branch R bypasses the mask entirely (`diagnose_residual.py`). A low
interaction share is evidence the MASK is not separating, not that the model's
output carries no frequency structure at all. Read-only; nothing trains.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train import build_model                                   # noqa: E402
from src.estimates.runner import git_commit                     # noqa: E402
from plot_mask_grid import capture_mask                         # noqa: E402


def decompose(M, frames):
    """Additive two-way variance shares of M[f, t] over the selected frames."""
    A = M[:, frames]
    if A.shape[1] < 4:
        return None
    mu = A.mean()
    eq = A.mean(axis=1, keepdims=True) - mu            # one value per frequency
    gain = A.mean(axis=0, keepdims=True) - mu          # one value per frame
    interaction = A - mu - eq - gain
    total = ((A - mu) ** 2).mean()
    if total <= 0:
        return None
    share = lambda D: float((D ** 2).mean() / total)   # noqa: E731
    return {
        "gain_share": share(np.broadcast_to(gain, A.shape)),
        "eq_share": share(np.broadcast_to(eq, A.shape)),
        "interaction_share": share(interaction),
        "n_frames": int(A.shape[1]),
    }


def active(mag, floor_db):
    """Frames whose energy is within floor_db of the clip's loudest frame."""
    energy = 20 * np.log10(np.sqrt((mag ** 2).sum(axis=0)) + 1e-9)
    return energy > (energy.max() - floor_db)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="models/model_sir0_10000-e6.pt")
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--data-root", default="data/rendered")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--active-db", type=float, default=35.0,
                    help="a frame is active if within this many dB of the clip peak")
    ap.add_argument("--max-hz", type=float, default=8000.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="experiments/results/2026-09-13-mask-structure")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.eval()
    model.estimator.mask_floor = 0.0
    store = capture_mask(model)

    root = Path(args.data_root) / args.split
    rows = []
    for directory in sorted(root.glob(f"{args.split}-*")):
        if len(rows) >= args.n:
            break
        if not (directory / "interferer.wav").exists():
            continue
        read = lambda n: torch.from_numpy(                       # noqa: E731
            sf.read(directory / n, dtype="float32")[0].astype(np.float32))
        target, interferer = read("target.wav"), read("interferer.wav")
        if target.abs().max() < 1e-4 or interferer.abs().max() < 1e-4:
            continue                                             # not a `both` trial
        mixture, enrolment = read("mixture.wav"), read("enrollment.wav")

        with torch.no_grad():
            model(mixture[None], enrolment[None])
            mask = store["mask"][0].numpy()
            X = model.stft(mixture[None])[0].abs().numpy()
            S = model.stft(target[None])[0].abs().numpy()
            I = model.stft(interferer[None])[0].abs().numpy()

        sample_rate = checkpoint["config"]["data"]["sample_rate"]
        bins = int(args.max_hz / (sample_rate / 2) * X.shape[0])
        T = min(mask.shape[1], X.shape[1])
        X, S, I, mask = X[:bins, :T], S[:bins, :T], I[:bins, :T], mask[:bins, :T]
        ideal = np.clip(S / np.maximum(X, 1e-8), 0.0, 2.0)

        selections = {
            "all": np.ones(T, dtype=bool),
            "speech": active(X, args.active_db),
            "overlap": active(S, args.active_db) & active(I, args.active_db),
        }
        row = {"trial": directory.name}
        for name, frames in selections.items():
            row[name] = {"ours": decompose(mask, frames),
                         "ideal": decompose(ideal, frames),
                         "frame_fraction": float(frames.mean())}
        rows.append(row)

    summary = {}
    for name in ("all", "speech", "overlap"):
        for which in ("ours", "ideal"):
            present = [r[name][which] for r in rows if r[name][which]]
            if not present:
                continue
            summary[f"{name}/{which}"] = {
                k: float(np.mean([p[k] for p in present]))
                for k in ("gain_share", "eq_share", "interaction_share")
            } | {"n_trials": len(present)}
        summary[f"{name}/frame_fraction"] = float(np.mean([r[name]["frame_fraction"] for r in rows]))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "n_trials": len(rows), "checkpoint": args.checkpoint, "split": args.split,
        "active_db": args.active_db, "max_hz": args.max_hz, "seed": args.seed,
        "git_commit": git_commit(), "date": str(date.today()),
        "summary": summary, "per_trial": rows,
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=1))

    print(f"{len(rows)} trials, {args.split}, {Path(args.checkpoint).name}\n")
    print(f"{'frames':<10}{'mask':<8}{'gain(t)':>9}{'eq(f)':>9}{'interaction':>13}")
    for name in ("all", "speech", "overlap"):
        print(f"  ({100*summary[f'{name}/frame_fraction']:.0f} % of frames)")
        for which in ("ours", "ideal"):
            key = f"{name}/{which}"
            if key in summary:
                s = summary[key]
                print(f"{name:<10}{which:<8}{100*s['gain_share']:>8.1f}%"
                      f"{100*s['eq_share']:>8.1f}%{100*s['interaction_share']:>12.1f}%")
    print(f"\nwrote {out/'summary.json'}")


if __name__ == "__main__":
    main()
