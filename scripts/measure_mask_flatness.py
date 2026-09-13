"""Does the mask get LESS flat as the training set grows? Objective, or data?

    ../tse_venv/bin/python scripts/measure_mask_flatness.py \
        --checkpoints models/model_sir0.pt,models/model_sir0_5000-e7.pt,models/model_sir0_10000-e6.pt \
        --split sir0 --condition both --limit 50 \
        --out experiments/results/2026-09-13-mask-flatness

THE QUESTION. MEASURED 2026-09-12: 84.2 % of our mask's variance is explained by
a single number per frame, and it varies 6.5x less across frequency than the
ideal mask. The model applies a broadband gain rather than deciding, per
time-frequency cell, which voice owns it -- which is the one operation that can
separate two overlapping talkers. That finding has TWO readings and they demand
opposite fixes:

  OBJECTIVE FAILURE  nothing in the loss ever asks for frequency structure, so
                     none was learned. Fix: add a term.
  DATA FAILURE       the model cannot learn reliable per-cell decisions from
                     ~10k trials and falls back on the robust broadband
                     solution. Fix: more data; a structure term treats a
                     symptom. WeSep, trained on 100 h, reaches 49.4 WER at
                     SIR < -5 dB where we reach 88.2, so this is not idle.

This script reads flatness off checkpoints that ALREADY EXIST at three training
sizes (~1,989 / ~4,976 / ~9,955 trials -- the scaling curve in generator.yaml),
so the fork costs CPU hours instead of a training run.

THE CONFOUND, AND IT IS THE REASON THIS SCRIPT EXISTS RATHER THAN A ONE-LINER.
Epoch runs BACKWARDS against data size in the checkpoints we own: the ~2k model
is at epoch 9-14, the ~5k at 7, the ~10k at 6. A flatness difference between
them is therefore data size AND training length together. Pass the epoch sweep
at fixed data size (model_sir0_10000-e6 / -e7 / -last, epochs 6 / 7 / 15) in the
same run: if flatness moves as much across epochs as across data sizes, the
comparison is inconclusive and this script says so rather than being read anyway.

WHAT IS MEASURED, per trial, on the model's own STFT grid, against the ideal
mask |target| / |mixture| clipped to [0, 2] (Yu et al., Interspeech 2023 predict
a complex mask; magnitude is what "flat" is about, so magnitude is what is
compared):

  frame_mean_share   1 - var(M - per-frame mean) / var(M). The 84.2 % number.
                     1.0 = a pure volume knob, 0.0 = no broadband component.
  d_time, d_freq     mean |first difference| along each axis.
  freq_over_time     d_freq / d_time. The ideal mask measures ~1.01; ours 0.53.

Reported over ALL bins (comparable with the 2026-09-12 figure) and over LOUD
bins only -- target energy within 50 dB of its peak, the same restriction
plot_mask_grid.py uses -- because silence is most of a spectrogram and a
statistic dominated by it says little about the decision the model is making.

NOT A CONTENT RESULT. Flatness is a property of the mask, not of what a listener
transcribes. It selects which intervention to try; it never scores one.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.estimates.runner import read_trials, git_commit      # noqa: E402
from src.run_log import timed                                  # noqa: E402
from train import SPLIT_MANIFESTS, build_model                 # noqa: E402


def read_mono(path):
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def flatness(mask):
    """(frame_mean_share, d_time, d_freq) for one (F, T) magnitude array."""
    if mask.shape[1] < 2 or mask.shape[0] < 2:
        return np.nan, np.nan, np.nan
    frame_mean = mask.mean(axis=0, keepdims=True)
    total = mask.var()
    residual = (mask - frame_mean).var()
    share = np.nan if total <= 0 else 1.0 - residual / total
    return (float(share),
            float(np.abs(np.diff(mask, axis=1)).mean()),
            float(np.abs(np.diff(mask, axis=0)).mean()))


def measure(model, mixture, target, enrolment, device):
    with torch.no_grad():
        mix_t = torch.from_numpy(mixture).unsqueeze(0).to(device)
        model(mix_t, torch.from_numpy(enrolment).unsqueeze(0).to(device))
        mask = model.estimator.last_parts["mask_mag"][0].cpu().numpy()
        X = model.stft(mix_t)[0].abs().cpu().numpy()
        S = model.stft(torch.from_numpy(target).unsqueeze(0).to(device))[0].abs().cpu().numpy()

    frames = min(mask.shape[1], X.shape[1], S.shape[1])
    bins = min(mask.shape[0], X.shape[0], S.shape[0])
    mask, X, S = mask[:bins, :frames], X[:bins, :frames], S[:bins, :frames]
    ideal = np.clip(S / np.maximum(X, 1e-8), 0.0, 2.0)

    row = {}
    # Same "loud" rule as plot_mask_grid.py: bins carrying real target energy.
    db = 20 * np.log10(S + 1e-6)
    loud = db > (db.max() - 50)
    for scope, sel in (("all", None), ("loud", loud)):
        if sel is None:
            m, idl = mask, ideal
        else:
            # Keep the grid shape -- a first difference needs neighbours -- and
            # zero what is not selected rather than flattening to a vector.
            keep_rows = sel.any(axis=1)
            if keep_rows.sum() < 2:
                continue
            m, idl = mask[keep_rows], ideal[keep_rows]
        for name, arr in (("ours", m), ("ideal", idl)):
            share, dt, df = flatness(arr)
            row[f"{scope}_{name}_frame_mean_share"] = share
            row[f"{scope}_{name}_d_time"] = dt
            row[f"{scope}_{name}_d_freq"] = df
            row[f"{scope}_{name}_freq_over_time"] = df / dt if dt else np.nan
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoints", required=True,
                    help="comma-separated. Give the data-size sweep AND the "
                         "fixed-data epoch sweep in one run, or the confound "
                         "cannot be subtracted.")
    ap.add_argument("--split", default="sir0", choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--condition", default="both")
    ap.add_argument("--limit", type=int, default=50,
                    help="the 2026-09-12 figure rested on 12 trials; 50 is "
                         "cheap and tightens it. The SAME trials are used for "
                         "every checkpoint.")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    trials = read_trials(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        audio_root=Path(args.data_root) / "rendered" / val_audio,
        limit=args.limit, condition=args.condition)

    audio = [(t.trial_id,
              read_mono(t.directory / "mixture.wav"),
              read_mono(t.directory / "target.wav"),
              read_mono(t.directory / "enrollment.wav")) for t in trials]

    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-mask-flatness")
    out_root.mkdir(parents=True, exist_ok=True)

    per_checkpoint, done = {}, []
    with timed("scripts/measure_mask_flatness.py",
               scope=lambda: f"{len(done)} checkpoints x {len(audio)} trials",
               rate=lambda: "cpu, whole-clip"):
        for path in args.checkpoints.split(","):
            path = path.strip()
            ckpt = torch.load(path, map_location=device, weights_only=False)
            model = build_model(ckpt["config"])
            model.load_state_dict(ckpt["model"])
            model.to(device).eval()
            model.estimator.capture_parts = True

            rows = []
            for i, (tid, mixture, target, enrolment) in enumerate(audio, 1):
                row = measure(model, mixture, target, enrolment, device)
                row["trial_id"] = tid
                rows.append(row)
                if i % 25 == 0 or i == len(audio):
                    print(f"  {Path(path).name}  {i}/{len(audio)}", flush=True)
            per_checkpoint[path] = {"epoch": ckpt.get("epoch"), "rows": rows}
            done.append(path)

    summary = {
        "n_trials": len(audio),
        "split": args.split,
        "condition": args.condition,
        "seed": 42,
        "git_commit": git_commit(),
        "date": date.today().isoformat(),
        "checkpoints": {},
    }
    for path, blob in per_checkpoint.items():
        keys = [k for k in blob["rows"][0] if k != "trial_id"]
        summary["checkpoints"][path] = {
            "epoch": blob["epoch"],
            "mean": {k: float(np.nanmean([r.get(k, np.nan) for r in blob["rows"]]))
                     for k in keys},
        }

    json.dump({p: b["rows"] for p, b in per_checkpoint.items()},
              open(out_root / "per_trial.json", "w"), indent=1)
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)

    print(f"\n  wrote {out_root}\n")
    hdr = f"  {'checkpoint':34s} {'ep':>3s} {'share(all)':>11s} {'share(loud)':>12s} " \
          f"{'f/t ours':>9s} {'f/t ideal':>10s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for path, blob in summary["checkpoints"].items():
        m = blob["mean"]
        print(f"  {Path(path).name:34s} {str(blob['epoch']):>3s} "
              f"{m.get('all_ours_frame_mean_share', float('nan')):11.4f} "
              f"{m.get('loud_ours_frame_mean_share', float('nan')):12.4f} "
              f"{m.get('loud_ours_freq_over_time', float('nan')):9.4f} "
              f"{m.get('loud_ideal_freq_over_time', float('nan')):10.4f}")
    print("\n  share = fraction of mask variance explained by ONE number per "
          "frame. 1.0 = pure volume knob.")
    print("  f/t   = how much the mask varies across frequency relative to "
          "across time. The ideal mask's value is the target.")


if __name__ == "__main__":
    main()
