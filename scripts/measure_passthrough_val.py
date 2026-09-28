"""Score "do nothing" (output = the mixture) EXACTLY as train.py's validation
loop scores a model. The like-for-like reference for every val_* number in a
history.csv.

    ../tse_venv/bin/python scripts/measure_passthrough_val.py --split sir0 \
        --config experiments/configs/bsrnn_cue_parts.yaml

WHY THIS EXISTS. The 1.593 dB pass-through quoted as "the margin over doing
nothing" since 2026-08-28 comes from derive_w_g.py, which scores the TARGET
direction only (200 crops). The validation loop scores BOTH directions (400
crops, 261 present on sir0_val). decisions-m2.md 2026-08-29 flagged the two as
"not directly subtractable"; they were subtracted anyway. This measures the
reference on the validation loop's own crops, with the loss's own functions,
so the subtraction is valid.

No model, no checkpoint: the only input is the data and the loss settings.
Same dataset arguments as train.py's val_dataset and the same
present-crop-weighted mean as its epoch accumulator.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from train import SPLIT_MANIFESTS, build_loss_fn, git_commit, selection_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--config", required=True,
                    help="supplies the data settings and loss weights; pass the "
                         "config of the run being compared against")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--batch-size", type=int, default=8, help="counts TRIALS")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    both_directions = bool(config["data"].get("both_directions", False))
    dataset = TrialDataset(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        data_root=Path(args.data_root),
        split=val_audio,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed,
        random_crop=False,
        both_directions=both_directions,
    )
    loss_fn = build_loss_fn(config)

    sums = {"L_pres": 0.0, "L_MR": 0.0, "L_gain": 0.0, "L_abs": 0.0}
    n_present = n_absent = 0
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_pairs):
            mixture, target = batch["mixture"], batch["target"]
            absent = batch["crop_absent"].bool()
            present = ~absent
            if present.any():
                t, y = target[present], mixture[present]      # y = pass-through
                sums["L_pres"] += float(loss_fn._loss_target_present(
                    t, y, loss_fn.tau_pres).sum())
                sums["L_MR"] += float(loss_fn._loss_multi_res_stft(
                    t, y, loss_fn.windows, loss_fn.p).sum())
                sums["L_gain"] += float(loss_fn._loss_gain_match(
                    t, y, loss_fn.gain_delta_db).sum())
                n_present += int(present.sum())
            if absent.any():
                x = mixture[absent]
                sums["L_abs"] += float(loss_fn._loss_target_absent(
                    x, x, loss_fn.tau_abs).sum())
                n_absent += int(absent.sum())

    row = {"L_pres": sums["L_pres"] / n_present, "L_MR": sums["L_MR"] / n_present,
           "L_gain": sums["L_gain"] / n_present, "L_abs": sums["L_abs"] / n_absent}
    score = selection_score(row, config)

    print(f"  pass-through on {val_manifest}: {n_present} present / {n_absent} absent crops"
          f"{', both directions' if both_directions else ''}")
    print(f"    separation {-row['L_pres']:+.3f} dB  (L_pres {row['L_pres']:.4f})")
    print(f"    L_MR {row['L_MR']:.4f}   L_gain {row['L_gain']:.4f}   L_abs {row['L_abs']:.4f}")
    print(f"    selection score ({config['training'].get('select_on', 'present_branch')}) {score:.4f}")

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today()}-passthrough-val-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "meta.yaml", "w") as fh:
        yaml.safe_dump({
            "date": str(date.today()), "git_commit": git_commit(), "seed": seed,
            "split": args.split, "manifest": val_manifest, "config": str(args.config),
            "system": "pass-through (output = mixture)",
            "protocol": "train.py val_dataset: random_crop=False, "
                        f"both_directions={both_directions}; present-crop-weighted mean",
            "n_present": n_present, "n_absent": n_absent,
            "val": {k: float(v) for k, v in row.items()},
            "separation_db": float(-row["L_pres"]),
            "selection_score": float(score),
        }, fh, sort_keys=False)
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
