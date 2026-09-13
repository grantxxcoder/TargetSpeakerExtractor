"""Sharpen an existing estimate's mask across frequency, by region growing.

    ../tse_venv/bin/python scripts/postprocess_mask.py \
        --est experiments/results/2026-09-04-train-sir0-10000 \
        --out experiments/results/2026-09-12-est-hyst-13-07 --hi 1.3 --lo 0.7

WHAT IT TESTS. MEASURED 2026-09-12 over 12 sir0_val trials: **84.2 % of our
mask's variance is explained by a single number per frame.** The model has
learned a broadband volume knob, not a time-frequency mask. Against the ideal
mask it falls 3.5x short of the right amount of variation along time and 6.5x
short along frequency, and the ideal mask varies equally in both axes (ratio
1.01) while ours varies half as much across frequency as across time (0.53).

The hypothesis this script tests is therefore NOT "the mask is too aggressive".
It is: **the per-frame gain is roughly right and the distribution of that gain
across frequency is wrong.**

HOW IT WORKS, and why it is region growing rather than a threshold. A single
threshold on a smooth mask just produces a smooth binary blob. Hysteresis keeps
a bin if it is strongly target-like, OR if it is weakly target-like AND
connected to a bin already kept. Faint but connected structure survives; faint
and isolated structure does not. That is Canny's edge-linking rule (Canny, IEEE
PAMI 1986) and the grouping-by-continuity rule of computational auditory scene
analysis (Bregman 1990; Brown & Cooke 1994; Wang & Brown 2006). BORROWED WITH A
DIFFERENCE: in CASA the grown regions ARE the system, built from harmonicity and
onset cues; here they are a post-hoc sharpening of a learned mask, and the
acceptance test is content fidelity for a downstream listener rather than an
ideal-binary-mask overlap score.

IT IS CAUSAL, so it does not spend latency. Growth runs forward in time only.
Within a frame it grows freely across frequency, which is legal because every
frequency of a frame arrives at the same instant -- the same argument that lets
the band-wise LSTM be bidirectional in a causal model.

THE LEVEL IS HELD FIXED, DELIBERATELY. After sharpening, every frame is rescaled
so its mean gain is what it was before. The model's broadband decision is left
alone and only the distribution across frequency changes. Without this the
experiment would confound "sharpen the mask" with "turn the output down", and
turning the output down has already been measured to move these metrics on its
own.

THRESHOLDS ARE RELATIVE TO EACH FRAME'S OWN MEAN GAIN, not absolute. The mask's
overall level moves with the input, so a fixed 0.6 would mean different things
in loud and quiet passages.

It reads finished `estimate.wav` files and never loads a checkpoint, so it
applies to ANY system -- ours, WeSep, or a floored variant -- and costs no GPU.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_log import timed  # noqa: E402

N_FFT, HOP = 512, 128


def stft(x):
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                      window=torch.hann_window(N_FFT), center=True,
                      return_complex=True)


def istft(X, length):
    return torch.istft(X, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT,
                       window=torch.hann_window(N_FFT), center=True, length=length)


def grow(mask, hi_rel, lo_rel):
    """Hysteresis region growing over a (F, T) gain map. Returns a bool keep map.

    Per frame: seeds are bins above hi, candidates are bins above lo. A run of
    adjacent candidate bins is kept if it contains a seed, or if it touches a
    bin that was kept in the PREVIOUS frame. Time flows one way only.
    """
    n_bins, n_frames = mask.shape
    frame_mean = mask.mean(axis=0)
    frame_mean = np.maximum(frame_mean, 1e-8)
    keep = np.zeros_like(mask, dtype=bool)
    previous = np.zeros(n_bins, dtype=bool)

    for t in range(n_frames):
        column = mask[:, t]
        candidate = column >= lo_rel * frame_mean[t]
        seed = (column >= hi_rel * frame_mean[t]) | (candidate & previous)
        if not candidate.any():
            previous = np.zeros(n_bins, dtype=bool)
            continue
        # Runs of adjacent candidate bins, found from the edges of the boolean
        # array. A run survives if any bin inside it is a seed.
        edges = np.flatnonzero(np.diff(np.concatenate(([0], candidate.view(np.int8), [0]))))
        kept = np.zeros(n_bins, dtype=bool)
        for start, end in zip(edges[0::2], edges[1::2]):
            if seed[start:end].any():
                kept[start:end] = True
        keep[:, t] = kept
        previous = kept
    return keep


def sharpen(mixture, estimate, hi_rel, lo_rel, down,
            floor=0.0, floor_follows_gain=False):
    """Redistribute the estimate's per-bin gain across frequency. Same length."""
    length = len(estimate)
    X = stft(torch.from_numpy(mixture[:length]))
    E = stft(torch.from_numpy(estimate))
    magnitude_x = X.abs().numpy()
    mask = E.abs().numpy() / np.maximum(magnitude_x, 1e-8)

    keep = grow(mask, hi_rel, lo_rel)
    sharpened = np.where(keep, mask, mask * down)

    if floor > 0.0:
        per_frame = mask.mean(axis=0)
        if floor_follows_gain:
            # 1.0 in the frame where the model opened up most, ~0 where it shut.
            strength = per_frame / max(per_frame.max(), 1e-8)
        else:
            strength = np.ones_like(per_frame)
        sharpened = np.maximum(sharpened, (floor * strength)[None, :])

    # HOLD THE LEVEL. Rescale each frame back to the mean gain it had, so the
    # only thing this function changes is the shape across frequency.
    before, after = mask.mean(axis=0), sharpened.mean(axis=0)
    sharpened = sharpened * (before / np.maximum(after, 1e-8))[None, :]

    scale = sharpened / np.maximum(mask, 1e-8)
    out = istft(E * torch.from_numpy(scale.astype(np.float32)), length)
    return out.numpy().astype(np.float32), keep.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--est", required=True, help="directory of finished estimates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="sir0_val")
    ap.add_argument("--data-root", default="data/rendered")
    ap.add_argument("--hi", type=float, default=1.3,
                    help="seed threshold, as a multiple of the frame's mean gain")
    ap.add_argument("--lo", type=float, default=0.7,
                    help="growth threshold, same units. Must be below --hi")
    ap.add_argument("--down", type=float, default=0.3,
                    help="what an unkept bin's gain is multiplied by. 0.0 removes "
                         "it outright; 1.0 is a no-op and is the control")
    ap.add_argument("--floor", type=float, default=0.0,
                    help="fill holes to at least this gain. Applied AFTER the "
                         "sharpening and before the level is restored.")
    ap.add_argument("--floor-follows-gain", action="store_true",
                    help="scale the floor by how far the model has opened its "
                         "broadband gain in that frame. MEASURED 2026-09-12: a "
                         "flat floor of 0.05 cut deletions 27 % (11.79 -> 8.64) "
                         "and RAISED leakage 7.8 points, because it fills holes "
                         "with the raw mixture -- which contains both speakers "
                         "-- even in frames where the target is silent. The "
                         "model's per-frame mean gain is 84 % of its mask's "
                         "information and is effectively its speech detector, so "
                         "following it fills holes only where the target is "
                         "believed present.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    assert args.lo < args.hi, "--lo must be below --hi or there is no hysteresis"

    source, out = Path(args.est), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rendered = Path(args.data_root) / args.split

    trials = sorted(d for d in source.iterdir()
                    if d.is_dir() and (d / "estimate.wav").exists())
    if args.limit:
        trials = trials[:args.limit]

    kept_fractions = []
    for count, directory in enumerate(trials, 1):
        estimate, rate = sf.read(directory / "estimate.wav", dtype="float32")
        mixture, _ = sf.read(rendered / directory.name / "mixture.wav", dtype="float32")
        sharpened, kept = sharpen(mixture, estimate, args.hi, args.lo, args.down,
                                  floor=args.floor,
                                  floor_follows_gain=args.floor_follows_gain)
        kept_fractions.append(kept)
        target = out / directory.name
        target.mkdir(exist_ok=True)
        sf.write(target / "estimate.wav", sharpened, rate)
        if count % 25 == 0:
            print(f"  {count}/{len(trials)}", flush=True)

    for name in ("meta.yaml", "results.json"):
        if (source / name).exists():
            shutil.copy(source / name, out / name)
    (out / "postprocess.txt").write_text(
        f"source {source}\nhi {args.hi}\nlo {args.lo}\ndown {args.down}\n"
        f"floor {args.floor} follows_gain {args.floor_follows_gain}\n"
        f"mean fraction of bins kept {np.mean(kept_fractions):.4f}\n")
    print(f"\nwrote {len(trials)} estimates -> {out}")
    print(f"mean fraction of bins kept: {100*np.mean(kept_fractions):.1f} %")


if __name__ == "__main__":
    with timed("scripts/postprocess_mask.py", scope=lambda: "sir0_val estimates"):
        main()
