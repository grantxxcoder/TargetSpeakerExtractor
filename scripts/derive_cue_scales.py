#!/usr/bin/env python3
"""DERIVE the per-channel scales for item 1a's cue. Not chosen -- measured.

    ../tse_venv/bin/python scripts/derive_cue_scales.py --split sir0

Writes experiments/results/<date>-cue-scales-<split>/{meta.yaml,per_batch.csv}
and prints the yaml lines to paste.

WHY THIS EXISTS
---------------
Item 1a replaced one well-scaled cue channel with three badly-scaled ones.
MEASURED 2026-09-22 at random init, as the share of the separator's input each
channel drives (zero it and see how far z moves):

    baseline's single cue   50.9 %      mean |value| 0.159
    direction                0.5 %                   0.028
    match_fraction         135.3 %                   0.851
    unmatched                4.4 %                   0.102
    (Re / Im, the reference)                         0.106 / 0.105

The old cue happened to sit at the same magnitude as the STFT values beside it.
The decomposition spans 30x, and `SubbandNorm` standardises all five channels
JOINTLY within a band, so it preserves their relative sizes rather than
equalising them. The channel carrying the target's spectral shape is therefore
nearly invisible to the separator, and the one carrying a ratio in [0,1] swamps
everything.

THE ARCHITECTURE CAN FIX THIS AND DOES NOT DO IT FAST ENOUGH. Measured on the
2-epoch probe checkpoint: the per-channel LayerNorm gain for `direction` moved
1.0000 -> 1.1687, the right direction, ~17 % of the way in two epochs against
the ~280 % needed -- while `match_fraction` grew 135.3 % -> 185.1 %. So this is
an OPTIMISATION problem, not a capability one, and the fix is conditioning the
input rather than adding capacity. decisions-m2.md 2026-09-22.

THE CRITERION, STATED SO IT CAN BE ARGUED WITH
----------------------------------------------
Match each cue channel's RMS to the RMS of the real and imaginary channels.

RMS, not mean-absolute: `ChannelWiseLayerNorm` standardises by a STANDARD
DEVIATION, so a channel's influence on the joint statistics is governed by its
spread, not by its mean magnitude. Both are reported; only RMS is used.

Two of the three have known closed forms, which is a check on the measurement
rather than a substitute for it:
  * `direction` is unit-norm over F bins, so its per-bin RMS is exactly
    1/sqrt(F) = 0.0624 at F=257, the same for every frame and every trial.
  * `match_fraction` is one number broadcast across bins, so its RMS over a
    frame IS cos(theta) for that frame.

DERIVED, THEN VERIFIED. The script applies the scales it derived and re-measures
the shares, so the number in the config is one whose EFFECT has been measured --
the same discipline as scripts/derive_w_g.py, which measured the anchor rather
than reasoning about it.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import SPLIT_MANIFESTS, build_model, git_commit  # noqa: E402

CUE = ("direction", "match_fraction", "unmatched")


def channel_shares(model, batches, scales=None):
    """How much of the separator's input does each cue channel drive?

    Zero one channel at a time and measure ||dz||^2 / ||z||^2. Done on the model
    as given: at init this reads the ARCHITECTURE, on a checkpoint it reads what
    training made of it.
    """
    totals = {k: [0.0, 0.0] for k in CUE}
    with torch.no_grad():
        for mixture, enrollment in batches:
            X, Xe = model.stft(mixture), model.stft(enrollment)
            tf = model.tfmap(X.abs(), Xe.abs())
            if scales is not None:
                tf = tf * torch.tensor(scales).view(1, -1, 1, 1)
            Xri = torch.stack([X.real, X.imag], dim=1)
            z_ref = model.subband_norm(model.split(torch.cat([Xri, tf], dim=1)))
            den = float(z_ref.pow(2).sum())
            for channel, name in enumerate(CUE):
                zeroed = tf.clone()
                zeroed[:, channel] = 0.0
                z = model.subband_norm(model.split(torch.cat([Xri, zeroed], dim=1)))
                totals[name][0] += float((z - z_ref).pow(2).sum())
                totals[name][1] += den
    return {k: 100.0 * v[0] / v[1] for k, v in totals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--config", default="experiments/configs/bsrnn_cue_parts.yaml")
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed); np.random.seed(seed)

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    dataset = TrialDataset(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        data_root=Path(args.data_root), split=val_audio,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed, random_crop=False, both_directions=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs)

    # TFMap is parameter-free and SubbandNorm's LayerNorm is plain
    # standardisation at init, so no checkpoint is needed or wanted: the scales
    # are a property of the DATA, and deriving them from a trained model would
    # bake that model's partial rebalancing into a constant.
    model = build_model(config).eval()

    rows, cached = [], []
    seen = 0
    sums = {k: [0.0, 0.0] for k in list(CUE) + ["reference"]}   # [sum sq, count]
    abs_sums = {k: [0.0, 0.0] for k in list(CUE) + ["reference"]}

    scope = lambda: f"{seen} crops, {args.split}"
    with timed("scripts/derive_cue_scales.py", scope=scope,
               rate=lambda: f"cpu, batch {args.batch_size}, no checkpoint"):
        with torch.no_grad():
            for batch in loader:
                if seen >= args.n_trials:
                    break
                mixture, enrollment = batch["mixture"], batch["enrollment"]
                cached.append((mixture, enrollment))
                X, Xe = model.stft(mixture), model.stft(enrollment)
                tf = model.tfmap(X.abs(), Xe.abs())
                Xri = torch.stack([X.real, X.imag], dim=1)

                def accumulate(name, t):
                    sums[name][0] += float(t.pow(2).sum()); sums[name][1] += t.numel()
                    abs_sums[name][0] += float(t.abs().sum()); abs_sums[name][1] += t.numel()

                for channel, name in enumerate(CUE):
                    accumulate(name, tf[:, channel])
                accumulate("reference", Xri)      # Re and Im together
                row = {"batch": len(rows), "n": mixture.shape[0]}
                row.update({f"rms_{n}": (sums[n][0] / sums[n][1]) ** 0.5
                            for n in list(CUE) + ["reference"]})
                rows.append(row)
                seen += mixture.shape[0]

    rms = {k: (v[0] / v[1]) ** 0.5 for k, v in sums.items()}
    mean_abs = {k: v[0] / v[1] for k, v in abs_sums.items()}
    scales = {k: rms["reference"] / rms[k] for k in CUE}

    before = channel_shares(model, cached)
    after = channel_shares(model, cached, scales=[scales[k] for k in CUE])

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today()}-cue-scales-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_batch.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out_dir / "meta.yaml", "w") as fh:
        yaml.safe_dump({
            "date": str(date.today()), "git_commit": git_commit(), "seed": seed,
            "split": args.split, "config": str(args.config), "n_trials": seen,
            "criterion": "match each cue channel's RMS to the RMS of Re and Im",
            "checkpoint": "NONE -- TFMap is parameter-free, LayerNorm is plain "
                          "standardisation at init",
            "rms": rms, "mean_abs": mean_abs, "scales": scales,
            "shares_pct_before": before, "shares_pct_after": after,
        }, fh, sort_keys=False)

    print(f"\n  {seen} crops from {val_manifest}, no checkpoint\n")
    print(f"  {'channel':<18}{'RMS':>10}{'mean|.|':>10}{'SCALE':>10}")
    for k in CUE:
        print(f"  {k:<18}{rms[k]:>10.4f}{mean_abs[k]:>10.4f}{scales[k]:>10.3f}")
    print(f"  {'reference (Re,Im)':<18}{rms['reference']:>10.4f}{mean_abs['reference']:>10.4f}"
          f"{'1.000':>10}")
    print(f"\n  closed-form check: direction RMS should be 1/sqrt(F) = "
          f"{1.0 / (model.stft.n_fft // 2 + 1) ** 0.5:.4f}")

    print(f"\n  {'channel':<18}{'share before':>14}{'share after':>13}")
    for k in CUE:
        print(f"  {k:<18}{before[k]:>13.1f}%{after[k]:>12.1f}%")
    print("  (the baseline's single cue drove 50.9 % at init, for scale)")

    print("\n  PASTE INTO THE ARM'S CONFIG, under `model:`\n")
    print("  tfmap_part_scales: [" + ", ".join(f"{scales[k]:.4f}" for k in CUE) + "]")
    print("                         # DERIVED " + str(date.today())
          + f", scripts/derive_cue_scales.py, {seen} crops.")
    print("                         # direction, match_fraction, unmatched.")
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
