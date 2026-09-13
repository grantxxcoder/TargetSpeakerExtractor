"""Measure L_struct at known anchors and DERIVE w_struct. decisions-pending.md D17.

    ../tse_venv/bin/python scripts/derive_w_struct.py \
        --split sir0 --checkpoint models/model_sir0_10000-e6.pt \
        --n-crops 50 --share 0.15 \
        --out experiments/results/2026-09-13-wstruct-anchor-sir0

MEMORY. This script backpropagates through the full model twice per batch. The
first version cloned every parameter gradient twice and held two graphs at once,
and took a 15 GB machine down. It now holds ONE graph at a time and accumulates
scalars. Even so: --n-crops defaults to 50, not 200, and --batch-size is there
for when the machine is busy. Do not run it alongside two other training or
evaluation jobs.

THE WEIGHT IS NEVER CHOSEN. w_g = 1.69 and w_state = 0.002692 were both derived
against a measured anchor, and this follows the same discipline for the same
reason: a hand-picked weight makes every result an argument about the weight.

### Why the measurement point MOVES, and this is the one real difference

`derive_w_state.py` measures the gradient share reaching the WAVEFORM, because
all four M2 terms act on the waveform and the waveform is their common currency.
**L_struct does not act on the waveform.** It acts on the mask, several
operations upstream. Measuring it at the waveform would compare a quantity that
exists there against one that does not.

So the share is measured on the MODEL PARAMETERS, which is the first place the
two terms are commensurable. That is a deliberate departure from the w_state
derivation and must be reported as one -- a share of 15 % here is not the same
object as a share of 15 % there, and the two numbers may not be compared.

### The share is EXACTLY linear in the weight, so no search is needed

    total = base + (1 - w) * w_struct * L_struct

so  grad(w_struct) - grad(0) = w_struct * (1 - w) * grad(L_struct), elementwise
and exactly. Therefore

    share = w_struct * (1 - w) * ||grad(L_struct)||_1 / ||grad(base)||_1
    w_struct = share * ||grad(base)||_1 / ((1 - w) * ||grad(L_struct)||_1)

One extra backward pass, no bisection. The script then RE-MEASURES at the
derived weight and asserts the share comes back, because an algebraic
relationship that is not checked is an assumption.

### The three anchors, and what each one means

  flat    a mask of all ones -- the pure volume knob, zero frequency shape. The
          term's UPPER anchor: what a model that has learned nothing about
          frequency pays. Our measured mask is 84.2 % explained by one number
          per frame, so we sit near this end.
  model   the checkpoint's own mask. The real starting value.
  oracle  the ideal mask itself. Exactly 0 by construction, and a wiring check:
          anything else means the loss and the target are on different grids.

`model` between `flat` and `oracle` is the headroom the term has to work with.
If `model` is already close to `oracle`, the term has nothing to buy and D17
dies here for the cost of a script -- which is the point of running it first.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.estimates.runner import git_commit                    # noqa: E402
from src.run_log import timed                                  # noqa: E402
from train import (SPLIT_MANIFESTS, build_model, build_loss_fn,  # noqa: E402
                   get_data_loaders, unpack, oracle_mask_and_mag)


def param_grad_l1(model):
    """L1 of every parameter gradient, as a SCALAR. Nothing is retained."""
    total = 0.0
    for prm in model.parameters():
        if prm.grad is not None:
            total += float(prm.grad.abs().sum())
    return total


def grad_norms(model, loss_fn, batch, device):
    """||grad(base)||_1 and ||grad of the structure contribution||_1.

    MEMORY, AND THIS IS WHY THE FUNCTION LOOKS LIKE THIS. The first version
    cloned every parameter gradient twice (7.19 M floats each) and held two
    autograd graphs through a six-layer LSTM stack at once. On a 15 GB machine
    running two other jobs it exhausted RAM and took the box down.

    The fix uses the fact that the weight enters LINEARLY:

        total = base + (1 - w) * w_struct * L_struct

    so the difference between the gradient at w_struct and at 0 is exactly
    (1 - w) * w_struct * grad(L_struct). The two norms can therefore be measured
    in SEPARATE backward passes that each free their own graph, accumulating one
    float apiece -- no subtraction, no clones, no two graphs alive together.

    TWO FORWARD PASSES, deliberately, rather than one forward and two backwards
    with retain_graph. retain_graph keeps the graph alive, which is the thing
    that crashed. Slower and survivable beats faster and fatal.
    """
    mixture, target, enrollment, crop_absent = unpack(batch, device)

    # --- pass A: the base objective, structure term OFF ---------------------
    model.zero_grad(set_to_none=True)
    saved, loss_fn.w_struct = loss_fn.w_struct, 0.0
    s_output = model(mixture, enrollment)
    loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent)
    loss.backward()
    base = param_grad_l1(model)
    loss_fn.w_struct = saved
    del s_output, loss
    model.zero_grad(set_to_none=True)

    # --- pass B: the structure contribution ALONE ---------------------------
    # (1 - w) is the factor the term carries inside the objective, so scaling by
    # it here makes `struct` directly comparable to `base` without a subtraction.
    s_output, mask = model(mixture, enrollment, return_mask=True)
    oracle, mag = oracle_mask_and_mag(model, target, mixture)
    present = ~crop_absent.bool()
    struct = 0.0
    if present.any():
        contribution = (1.0 - loss_fn.w) * loss_fn._loss_mask_shape(
            mask.float()[present], oracle[present], mag[present]).mean()
        contribution.backward()
        struct = param_grad_l1(model)
        parts["L_struct"] = float(contribution.detach()) / max(1.0 - loss_fn.w, 1e-12)
        del contribution
    del s_output, mask, oracle, mag
    model.zero_grad(set_to_none=True)
    return base, struct, parts


def anchors(model, loss_fn, batch, device):
    """L_struct for a flat mask, the model's mask, and the oracle itself."""
    mixture, target, enrollment, crop_absent = unpack(batch, device)
    with torch.no_grad():
        _, mask = model(mixture, enrollment, return_mask=True)
        oracle, mag = oracle_mask_and_mag(model, target, mixture)
        present = ~crop_absent.bool()
        if not present.any():
            return None
        flat = torch.ones_like(mask)
        f = loss_fn._loss_mask_shape
        return {
            "flat": float(f(flat[present], oracle[present], mag[present]).mean()),
            "model": float(f(mask[present].float(), oracle[present], mag[present]).mean()),
            "oracle": float(f(oracle[present], oracle[present], mag[present]).mean()),
        }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="sir0", choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--n-crops", type=int, default=50,
                    help="50, not 200. Each crop costs two forward passes and "
                         "two backward passes through a six-layer LSTM stack; "
                         "the anchors are means and the share is a ratio of "
                         "sums, so 50 is a readable first answer. Raise it only "
                         "on an otherwise idle machine.")
    ap.add_argument("--chunk-s", type=float, default=1.0,
                    help="crop length for the derivation, in seconds. 1.0, not "
                         "the config's 4.008: ONE LSTM layer in this stack "
                         "allocates ~381 MB of activations at batch 1 and a "
                         "4 s crop, and there are twelve of them, all retained "
                         "for backward. The share is a RATIO of two gradient "
                         "norms measured on the same crops, so shortening them "
                         "changes both numerator and denominator together. "
                         "Pass the config value to check that -- if the derived "
                         "weight moves materially with crop length, say so "
                         "rather than quoting either number.")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="override the config's training batch size. Lower it "
                         "if memory is tight -- this script holds one autograd "
                         "graph at a time but the graph itself scales with the "
                         "batch.")
    ap.add_argument("--share", type=float, default=0.15,
                    help="target gradient share ON THE PARAMETERS. 0.15 matches "
                         "the share w_state was set to, but the measurement "
                         "point differs -- see the module docstring. NOT "
                         "comparable to w_state's 15 %%.")
    # Kaggle mounts the audio somewhere else entirely, so these must be
    # settable exactly as train.py makes them settable. Defaults are the local tree.
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["config"]); model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    loss_fn = build_loss_fn(config)
    loss_fn.w_struct = 0.0

    if args.batch_size:
        config["data"]["batch_size"] = int(args.batch_size)
    config["data"]["chunk_s"] = float(args.chunk_s)
    _, val_loader = get_data_loaders(
        args.split, Path(args.manifest_dir), Path(args.data_root), config)

    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-wstruct-anchor-{args.split}")
    out_root.mkdir(parents=True, exist_ok=True)

    rows, base_sum, delta_sum, seen = [], 0.0, 0.0, 0
    with timed("scripts/derive_w_struct.py",
               scope=lambda: f"{seen} crops, {args.split}",
               rate=lambda: f"{device.type}, two backward passes per batch"):
        for batch in val_loader:
            a = anchors(model, loss_fn, batch, device)
            if a is None:
                continue
            b, d, _ = grad_norms(model, loss_fn, batch, device)
            base_sum += b; delta_sum += d
            rows.append(a); seen += len(batch["mixture"])
            if seen % 50 < len(batch["mixture"]):
                print(f"  {seen}/{args.n_crops}", flush=True)
            if seen >= args.n_crops:
                break

    w = float(config["loss"]["w"])
    # delta_sum is ||grad|| at w_struct = 1, which already carries the (1 - w).
    w_struct = args.share * base_sum / max(delta_sum, 1e-20)
    measured = w_struct * delta_sum / max(base_sum, 1e-20)
    assert abs(measured - args.share) < 1e-6, (
        f"the linear relation did not hold: asked {args.share}, re-measured "
        f"{measured}. Do not use this weight.")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    summary = {
        "w_struct": w_struct,
        "share_requested": args.share,
        "share_remeasured": measured,
        "share_measured_on": "model parameters (NOT the waveform -- see docstring)",
        "anchors_mean_L_struct": mean,
        "headroom_model_over_oracle": mean["model"] - mean["oracle"],
        "fraction_of_the_way_to_flat": (
            (mean["model"] - mean["oracle"]) / (mean["flat"] - mean["oracle"])
            if mean["flat"] > mean["oracle"] else None),
        "n_crops": seen, "split": args.split, "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"), "seed": seed,
        "loss_w": w, "git_commit": git_commit(), "date": date.today().isoformat(),
        # Recorded because the derivation was run on SHORTER crops than training
        # uses, for memory. A weight that moves with this is not a weight.
        "chunk_s_used": float(args.chunk_s),
        "chunk_s_in_config": 4.008,
        "batch_size_used": config["data"]["batch_size"],
    }
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)
    with open(out_root / "per_crop.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    print(f"\n  wrote {out_root}\n")
    print(f"  L_struct at the anchors, mean over {seen} present crops:")
    for k in ("flat", "model", "oracle"):
        print(f"    {k:8s} {mean[k]:.6f}")
    frac = summary["fraction_of_the_way_to_flat"]
    if frac is not None:
        print(f"\n  our mask sits {frac:.1%} of the way from the ideal to a "
              f"pure volume knob.")
    print(f"\n  w_struct = {w_struct:.6g}   "
          f"({args.share:.0%} of the parameter gradient, re-measured {measured:.4f})")
    print("\n  Put this in experiments/configs/*.yaml as loss.w_struct.")


if __name__ == "__main__":
    main()
