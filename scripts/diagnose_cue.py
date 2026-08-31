#!/usr/bin/env python3
"""D3a: does the speaker cue itself carry identity, or does the network lose it?

    ../tse_venv/bin/python scripts/diagnose_cue.py --split sir0 \
        --checkpoint models/model_sir0_e50es.pt

Writes experiments/results/<date>-cue-diag-<split>/{per_batch.csv,meta.yaml}.

WHY THIS EXISTS
---------------
Every run so far has failed the same way: swapping a stranger's enrollment in
moves the output by about half (`val_enrol_sens_db` ~ -3 dB, i.e. ~50 %, across
the control and both augmentation arms, 2026-08-30). Half the output is decided
without reference to who was asked for.

The conditioning path has TWO halves and no run distinguishes which one fails:

    build a cue                    make the network use it
    TFMap(mix_mag, enroll_mag)  ->  concat as a 3rd channel -> 6 BSNet blocks

**Rebuilding the wrong half is the expensive mistake available here.** This
measures both with the same statistic, one layer apart, so the failure is
attributed rather than guessed. decisions-pending.md D3a.

THE TEST
--------
Roll the enrollment within the batch -- the same trick `diagnostic_accumulate()`
in train.py uses at the output -- and measure relative movement at each stage:

    cue sensitivity     ||tf_true - tf_swap||^2 / ||tf_true||^2
    output sensitivity  ||y_true  - y_swap ||^2 / ||y_true ||^2

Reported as a power ratio in dB and as a percentage, `pct = 10 ** (dB / 10)`,
matching how `val_enrol_sens_db` is read everywhere else.

HOW TO READ IT
--------------
    cue LOW                  -> nothing downstream can help. The cue is the bug:
                                fix the cue (D2 temperature, then D5 encoder).
    cue HIGH, output LOW     -> the cue is fine and the INJECTION PATH discards
                                it. D4a (re-inject at every block) is the fix.
    cue HIGH, output HIGH    -> conditioning is not the bottleneck; look at the
                                separator or the objective.

TFMap is parameter-free, so **cue sensitivity is a property of the data and
`tfmap_scale`, not of training**. It can be measured with no checkpoint at all;
pass one only to get the output half and the partition.

Two secondary read-outs, nearly free because the same softmax is already
computed, and both are the mechanism behind whatever the partition says:

  * attention concentration (D2). Measured 2026-08-25: the softmax blended ~620
    of 628 enrollment frames. `tfmap_scale` was added to fix that; this says
    whether it did.
  * same-gender split (D3c). `sir0_train` is balanced 50/50, which caps but does
    not remove a pitch shortcut -- a model using pitch alone gets the
    cross-gender half right and coin-flips the rest. If sensitivity collapses on
    same-gender trials the model is riding gender, not identity, and that is
    invisible in the pooled number we currently log.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
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


def rel_movement(a, b):
    """||a - b||^2 / ||a||^2, summed over the batch. The same statistic
    `diagnostic_accumulate()` applies at the output, so the two stages are
    directly comparable rather than merely similar."""
    return float((a - b).pow(2).sum()), float(a.pow(2).sum())


def db(num, den):
    """Power ratio in dB. Guarded: an empty stratum reports NaN rather than
    -inf, matching how the loss terms report a missing half."""
    if den <= 0 or num <= 0:
        return float("nan")
    # float(), not np.float64: meta.yaml is written with yaml.safe_dump, which
    # refuses numpy scalars outright rather than coercing them.
    return float(10.0 * np.log10(num / den))


def attention_stats(tfmap, mix_mag, enroll_mag):
    """Re-run TFMap's softmax to see how many enrollment frames it actually uses.

    Duplicates four lines of TFMap.forward rather than changing its signature:
    the module is on the training path and returning diagnostics from it would
    put a tuple in the model's forward for the sake of a script.
    """
    import torch.nn.functional as F
    bx = F.normalize(mix_mag,    p=2, dim=1, eps=tfmap.eps)
    be = F.normalize(enroll_mag, p=2, dim=1, eps=tfmap.eps)
    sim = torch.matmul(bx.transpose(1, 2), be)
    scale = mix_mag.shape[1] ** 0.5 if tfmap.scale is None else tfmap.scale
    h = torch.softmax(sim * scale, dim=-1)            # (B, Tx, Te)
    te = h.shape[-1]
    # Effective number of frames used, per mixture frame: the perplexity of the
    # attention distribution, exp(entropy). Uniform over Te gives exactly Te, a
    # one-hot gives 1 -- so it reads directly as "how many frames is it
    # blending?" without a threshold to argue about.
    ent = -(h.clamp_min(1e-12).log() * h).sum(-1)
    return {"eff_frames": float(ent.exp().mean()),
            "n_enroll_frames": int(te),
            "max_weight": float(h.max(-1).values.mean()),
            "uniform_weight": 1.0 / te}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", default=None,
                    help="without it only the CUE half is measured -- TFMap has "
                         "no parameters, so that half needs no trained model.")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--n-crops", type=int, default=200,
                    help="B6's floor for a scored measurement is 200")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cpu")

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    dataset = TrialDataset(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        data_root=Path(args.data_root),
        split=val_audio,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed,
        random_crop=False,     # fixed crops: this must be re-runnable exactly
        both_directions=False,  # one direction is enough; the roll supplies the swap
    )

    ckpt_meta = None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        ckpt_meta = {"path": str(args.checkpoint),
                     "epoch": int(ckpt["epoch"]) if "epoch" in ckpt else None,
                     "best_val": float(ckpt["best_val"]) if "best_val" in ckpt else None}
    else:
        # TFMap is parameter-free, so an untrained model measures the cue
        # identically. Only the output half needs trained weights.
        model = build_model(config)
    model.to(device).eval()

    # Strata: pooled, plus D3c's same-gender split. Read from the manifest,
    # which the loader already carries through in `meta`.
    sums = defaultdict(lambda: defaultdict(float))
    rows = []
    seen = 0
    att = defaultdict(float)
    n_batches = 0

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs)

    scope = lambda: f"{seen} crops, {args.split}"
    with timed("scripts/diagnose_cue.py", scope=scope,
               rate=lambda: f"cpu, batch {args.batch_size}"):
        with torch.no_grad():
            for batch in loader:
                if seen >= args.n_crops:
                    break
                mixture = batch["mixture"].to(device)
                enroll = batch["enrollment"].to(device)
                if mixture.shape[0] < 2:
                    # roll(1, 0) on a batch of 1 returns the SAME enrollment and
                    # would read a false 0 dB. train.py skips it for the same
                    # reason; skipping is honest, padding would not be.
                    continue
                swap = enroll.roll(1, 0)

                # --- the cue, one layer upstream of the network ---------------
                X = model.stft(mixture)
                tf_true = model.tfmap(X.abs(), model.stft(enroll).abs())
                tf_swap = model.tfmap(X.abs(), model.stft(swap).abs())

                # --- the output, for the partition ----------------------------
                y_true = model(mixture, enroll)
                y_swap = model(mixture, swap)

                same = batch["meta"]["same_gender"].to(device).bool()
                for name, sel in (("all", torch.ones_like(same)),
                                  ("same_gender", same), ("cross_gender", ~same)):
                    if not sel.any():
                        continue
                    cn, cd = rel_movement(tf_true[sel], tf_swap[sel])
                    on, od = rel_movement(y_true[sel], y_swap[sel])
                    sums[name]["cue_num"] += cn; sums[name]["cue_den"] += cd
                    sums[name]["out_num"] += on; sums[name]["out_den"] += od
                    sums[name]["n"] += int(sel.sum())

                a = attention_stats(model.tfmap, X.abs(), model.stft(enroll).abs())
                for k, v in a.items():
                    att[k] += v
                n_batches += 1
                rows.append({"batch": n_batches, "n": int(mixture.shape[0]),
                             "cue_db": db(*rel_movement(tf_true, tf_swap)),
                             "out_db": db(*rel_movement(y_true, y_swap)), **a})
                seen += mixture.shape[0]

    att = {k: v / max(n_batches, 1) for k, v in att.items()}

    print(f"\n  {seen} crops from {val_manifest}, tfmap_scale="
          f"{config['model'].get('tfmap_scale')}"
          f"{'' if ckpt_meta else '  (NO CHECKPOINT: output half is untrained)'}\n")
    print(f"  {'stratum':<15}{'n':>5}{'CUE moves':>14}{'OUTPUT moves':>16}")
    out = {}
    for name in ("all", "same_gender", "cross_gender"):
        if name not in sums:
            continue
        s = sums[name]
        c = db(s["cue_num"], s["cue_den"]); o = db(s["out_num"], s["out_den"])
        out[name] = {"n": int(s["n"]), "cue_db": c,
                     "cue_pct": float(100 * 10 ** (c / 10)), "out_db": o,
                     "out_pct": float(100 * 10 ** (o / 10))}
        print(f"  {name:<15}{int(s['n']):>5}{c:>9.2f} dB "
              f"({100 * 10 ** (c / 10):>5.1f} %){o:>10.2f} dB "
              f"({100 * 10 ** (o / 10):>5.1f} %)")

    print(f"\n  attention: blending {att['eff_frames']:.0f} of "
          f"{att['n_enroll_frames']:.0f} enrollment frames "
          f"(max weight {att['max_weight']:.5f} vs uniform "
          f"{att['uniform_weight']:.5f})")

    a = out.get("all", {})
    if ckpt_meta and a:
        print("\n  PARTITION")
        if a["cue_pct"] < 20:
            print("    The CUE barely moves. Nothing downstream can recover "
                  "identity the cue never carried -- fix the cue (D2, then D5). "
                  "D4a would re-inject a cue that says nothing.")
        elif a["out_pct"] < 0.5 * a["cue_pct"]:
            print("    The cue moves and the OUTPUT does not follow: the "
                  "injection path is discarding it. D4a (re-inject at every "
                  "block) is the indicated fix, and it needs no new cue.")
        else:
            print("    The cue moves and the output follows it. Conditioning is "
                  "NOT the bottleneck -- look at the separator or the objective.")
    elif not ckpt_meta:
        print("\n  No --checkpoint: the OUTPUT column is an untrained network "
              "and means nothing. Re-run with a checkpoint for the partition.")

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today().isoformat()}-cue-diag-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_batch.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out_dir / "meta.yaml").write_text(yaml.safe_dump({
        "date": date.today().isoformat(), "script": "scripts/diagnose_cue.py",
        "git_commit": git_commit(), "seed": seed, "config": args.config,
        "split": args.split, "manifest": val_manifest, "n_crops": seen,
        "tfmap_scale": config["model"].get("tfmap_scale"),
        "checkpoint": ckpt_meta, "strata": out, "attention": att,
    }, sort_keys=False))
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
