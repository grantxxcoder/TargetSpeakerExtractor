#!/usr/bin/env python3
"""Item 1a's acceptance test: does the cue still track how LOUD the frame is?

    ../tse_venv/bin/python scripts/measure_cue_loudness.py --split sir0

Writes experiments/results/<date>-cue-loudness-<split>/{per_frame.csv,meta.yaml}.

NO CHECKPOINT, BY DESIGN. TFMap has no learned parameters -- it is arithmetic
over the enrollment and the mixture -- so the quantity this measures is a
property of the DATA and of `tfmap_scale`, not of training. That is what makes
1a's acceptance test free: it can fail before a single GPU hour is spent.
The same reasoning is already recorded in diagnose_cue.py.

WHAT IS MEASURED
----------------
The cue's last step projects the UN-normalised mixture frame onto the target's
template direction:

    matched_level = <x_t, direction_t> = ||x_t|| * cos(theta_t)

so the single number the baseline hands over is loudness TIMES identity
evidence. MEASURED 2026-09-21: corr(matched_level, ||x_t||) = 0.990.

Item 1a hands over `match_fraction = matched_level / ||x_t||` instead. This
script reports both correlations on the same frames, so the pair is read off one
run rather than compared across scripts and dates.

HOW TO READ IT, AND THE CAVEAT THAT MUST TRAVEL WITH THE NUMBER
----------------------------------------------------------------
`match_fraction` is `matched_level` divided by the very quantity it is being
correlated against, so A LARGE DROP IS PARTLY GUARANTEED BY THE ALGEBRA. Do not
report the drop as if it were a discovery.

THE INFORMATIVE NUMBER IS WHAT SURVIVES. If match_fraction still tracks loudness
appreciably, that is a real property of the data -- loud frames genuinely being
more target-like, e.g. because the target dominates when it is loud -- and it
would mean 1a leaves a residual confound that 1c would then inherit. Report the
surviving correlation, and say plainly that the mechanical part was expected.

Spearman is reported beside Pearson because "tracks loudness" is a monotone
question, not a linear one, and a Pearson number alone can be moved by a handful
of very loud frames.
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
from scipy import stats
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.models.conditioning import TFMap  # noqa: E402
from src.models.stft import STFT  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import SPLIT_MANIFESTS, git_commit  # noqa: E402

# Frames this far below the clip's loudest are near-silence: the template
# direction there is fitted to numerical dust, and including them would let the
# correlation be decided by frames no listener can hear. Reported alongside the
# unfiltered number so the choice is visible rather than buried.
QUIET_FLOOR_DB = -60.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
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

    stft = STFT(config["model"]["stft"]["n_fft"], config["model"]["stft"]["hop"],
                config["data"]["sample_rate"])
    scale = float(config["model"]["tfmap_scale"])
    # ONE module, asked for both forms. Constructing two would risk them drifting
    # apart; `return_parts` is the only difference and it is set per call site.
    cue_product = TFMap(scale=scale)
    cue_parts = TFMap(scale=scale, return_parts=True)

    rows = []
    seen = 0
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs)
    scope = lambda: f"{seen} crops, {args.split}"
    with timed("scripts/measure_cue_loudness.py", scope=scope,
               rate=lambda: f"cpu, batch {args.batch_size}, no checkpoint"):
        with torch.no_grad():
            for batch in loader:
                if seen >= args.n_trials:
                    break
                mix_mag = stft(batch["mixture"]).abs()
                enrol_mag = stft(batch["enrollment"]).abs()

                product = cue_product(mix_mag, enrol_mag)[:, 0]     # (B,F,Tx)
                direction, match_fraction, unmatched = cue_parts(
                    mix_mag, enrol_mag).unbind(dim=1)

                loudness = mix_mag.norm(dim=1)                      # (B,Tx) = ||x||
                # matched_level is the product's length along a UNIT direction,
                # so its per-frame norm recovers it without recomputing the dot.
                matched_level = product.norm(dim=1)                 # (B,Tx)
                cos = match_fraction[:, 0, :]                       # (B,Tx), broadcast
                unmatched_frac = unmatched.norm(dim=1) / loudness.clamp_min(1e-12)

                peak = loudness.max(dim=1, keepdim=True).values.clamp_min(1e-12)
                rel_db = 20.0 * torch.log10(loudness.clamp_min(1e-12) / peak)

                for b in range(mix_mag.shape[0]):
                    for t in range(mix_mag.shape[2]):
                        rows.append({
                            "trial_id": batch["trial_id"][b],
                            "frame": t,
                            "loudness": float(loudness[b, t]),
                            "rel_db": float(rel_db[b, t]),
                            "matched_level": float(matched_level[b, t]),
                            "match_fraction": float(cos[b, t]),
                            "unmatched_fraction": float(unmatched_frac[b, t]),
                        })
                seen += mix_mag.shape[0]

    loud = np.array([r["loudness"] for r in rows])
    keep = np.array([r["rel_db"] for r in rows]) > QUIET_FLOOR_DB

    def corrs(y, mask):
        y, x = np.asarray(y)[mask], loud[mask]
        ok = np.isfinite(x) & np.isfinite(y)
        return (float(stats.pearsonr(x[ok], y[ok])[0]),
                float(stats.spearmanr(x[ok], y[ok])[0]), int(ok.sum()))

    out = {}
    for name in ("matched_level", "match_fraction", "unmatched_fraction"):
        y = [r[name] for r in rows]
        for label, mask in (("all", np.ones(len(rows), bool)), ("audible", keep)):
            p, s, n = corrs(y, mask)
            out[f"{name}/{label}"] = {"pearson": p, "spearman": s, "n": n}

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today()}-cue-loudness-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    with open(out_dir / "meta.yaml", "w") as fh:
        yaml.safe_dump({"date": str(date.today()), "git_commit": git_commit(),
                        "seed": seed, "split": args.split, "manifest": val_manifest,
                        "config": str(args.config), "tfmap_scale": scale,
                        "n_trials": seen, "n_frames": len(rows),
                        "quiet_floor_db": QUIET_FLOOR_DB,
                        "frames_below_quiet_floor": int((~keep).sum()),
                        "checkpoint": "NONE -- TFMap is parameter-free",
                        "correlations": out}, fh, sort_keys=False)

    print(f"\n  {seen} crops, {len(rows):,} frames from {val_manifest}, "
          f"tfmap_scale={scale}, NO checkpoint\n")
    excluded = int((~keep).sum())
    print(f"  quiet floor {QUIET_FLOOR_DB:.0f} dB rel. peak excludes "
          f"{excluded:,} of {len(rows):,} frames "
          f"({100 * excluded / max(len(rows), 1):.1f} %)"
          f"{'  -- so `audible` and `all` are the SAME frames' if not excluded else ''}")
    print(f"\n  {'quantity':<30}{'frames':>9}{'Pearson':>10}{'Spearman':>10}")
    for name in ("matched_level", "match_fraction", "unmatched_fraction"):
        for label in ("all", "audible"):
            k = out[f"{name}/{label}"]
            print(f"  {name + ' (' + label + ')':<30}{k['n']:>9,}"
                  f"{k['pearson']:>10.3f}{k['spearman']:>10.3f}")

    old = out["matched_level/audible"]["pearson"]
    new = out["match_fraction/audible"]["pearson"]
    print("\n  READING (audible frames)")
    print(f"    the baseline cue tracks loudness at   r = {old:.3f}")
    print(f"    item 1a's match fraction tracks it at r = {new:.3f}")
    print("    A LARGE DROP IS PARTLY GUARANTEED: match_fraction is")
    print("    matched_level divided by the quantity being correlated against.")
    print(f"    THE INFORMATIVE NUMBER IS WHAT SURVIVES: r = {new:.3f}.")
    if abs(new) > 0.5:
        print("    That is substantial. Loud frames really are more target-like in")
        print("    this data, so 1a leaves a residual confound for 1c to inherit.")
    else:
        print("    Small. The loudness coupling was mechanical, and removing the")
        print("    multiplication removes essentially all of it.")
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
