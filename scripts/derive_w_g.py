"""Measure L_gain at four anchors and derive w_g. decisions-m2.md 2026-08-28.

    ../tse_venv/bin/python scripts/derive_w_g.py --split sir0 \
        --checkpoint kaggle_out/models/model_sir0.pt

Writes experiments/results/<date>-wg-anchor-<split>/{per_crop.csv,meta.yaml}.

w_m's rule ("30 % of |L_pres| at the do-nothing anchor") does not transfer:
L_pres is ~-0.18 there on sir0, so it divides by ~0. Instead compare two states
differing ONLY in gain -- the mixture at its own level, and the mixture scaled
to the collapsed checkpoint's level:

    buys on absent :      w  * (L_abs_pt - L_abs_muted)
    costs via L_MR : (1 - w) * w_m * (L_MR_muted - L_MR_pt)
    costs via L_gain:(1 - w) * w_g * (L_gain_muted - L_gain_pt)

    w_g* = (buys - costs_MR) / ((1 - w) * (L_gain_muted - L_gain_pt))

Below w_g* the mute still pays. A ceiling is reported too: past it the level
term outweighs L_pres's whole range and pass-through becomes the cheap answer.

Measured 2026-08-28: costs_MR comes out NEGATIVE. L_MR, documented as "the term
that pins the output gain", REWARDS muting. Before L_gain nothing opposed it.

Assumes the mute is global (justified at a 2.45 dB present/absent gap). A
checkpoint with a real gate invalidates it -- re-derive, do not inherit.

Terms are accumulated by train.py's own add_parts/epoch_report, so the
arithmetic matches history.csv rather than resembling it.
"""

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
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import (SPLIT_MANIFESTS, add_parts, build_loss_fn, build_model,  # noqa: E402
                   epoch_report, git_commit)

# The three systems, in the order they are reported. `model` is dropped when no
# --checkpoint is given, in which case no derivation is possible and the script
# says so rather than inventing one.
ORACLE, PASSTHROUGH, MUTED, MODEL = "oracle", "passthrough", "passthrough_muted", "model"


def systems_for(batch, model, device):
    """Output per anchor.

    oracle            the clean target. L_gain and L_MR must be ~0: wiring check.
    passthrough       the mixture. Correct level, zero separation.
    passthrough_muted the mixture scaled per crop to the MODEL's level.
    model             the checkpoint under test.

    model-vs-passthrough conflates gain and separation and would price both.
    passthrough_muted holds the audio fixed and moves only gain, isolating the
    policy L_gain prices. L_pres must be identical across the two -- second
    wiring check.
    """
    mixture, target, enrollment = batch["mixture"], batch["target"], batch["enrollment"]
    out = {ORACLE: target, PASSTHROUGH: mixture}
    if model is not None:
        with torch.no_grad():
            estimate = model(mixture.to(device), enrollment.to(device)).cpu()
        out[MODEL] = estimate
        # per-crop gain that puts the mixture at the model's own output level
        eps = 1e-12
        gain = ((estimate.pow(2).mean(dim=-1) + eps).sqrt()
                / (mixture.pow(2).mean(dim=-1) + eps).sqrt())
        out[MUTED] = mixture * gain.unsqueeze(-1)
    return out


def derive(stats, w, wm):
    """Module-docstring arithmetic. Never raises: a degenerate denominator is a
    RESULT (the mute is not a level problem) and must reach the log."""
    # pt -> muted, NOT pt -> model: same audio, only the gain changed, so this
    # isolates the policy being priced. See systems_for().
    pt, md = stats[PASSTHROUGH], stats[MUTED]
    buys = w * (pt["L_abs"] - md["L_abs"])
    costs_mr = (1 - w) * wm * (md["L_MR"] - pt["L_MR"])
    gain_headroom = md["L_gain"] - pt["L_gain"]
    denominator = (1 - w) * gain_headroom

    out = {
        "mute_buys_on_absent": buys,
        "mute_already_costs_via_L_MR": costs_mr,
        "L_gain_headroom": gain_headroom,
        "break_even_w_g": None,
        "ceiling_w_g": None,
        "suggested_w_g": None,
    }
    if not np.isfinite(denominator) or denominator <= 1e-9:
        out["note"] = ("L_gain headroom is zero or negative: the checkpoint is no "
                       "further off-level than the mixture, so this term cannot "
                       "price its behaviour. Do NOT set w_g from this run.")
        return out

    break_even = (buys - costs_mr) / denominator
    # L_pres runs 0 -> -30 (tau_pres = 0.001). Past this the level term outweighs
    # everything the separation term can ever say.
    ceiling = 30.0 / gain_headroom
    out["break_even_w_g"] = break_even
    out["ceiling_w_g"] = ceiling

    if break_even <= 0:
        out["note"] = ("break-even is <= 0: L_MR alone already makes the mute "
                       "unprofitable, so the mute is not being driven by the "
                       "level trade this term models. Investigate before setting w_g.")
    elif break_even >= ceiling:
        out["note"] = (f"break-even ({break_even:.3f}) is at or above the ceiling "
                       f"({ceiling:.3f}): no weight both kills the mute and leaves "
                       f"the separation term dominant. The objective needs more "
                       f"than a reweighting.")
    else:
        # Midway between "the mute is exactly neutral" and "level outweighs
        # content", in log space, so the choice is not sitting on either edge.
        suggested = float(np.sqrt(break_even * ceiling))
        out["suggested_w_g"] = suggested
        out["note"] = (f"usable window [{break_even:.3f}, {ceiling:.3f}]. "
                       f"Suggested {suggested:.3f} (geometric midpoint). "
                       f"Record the choice in decisions-m2.md before training.")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", default=None,
                    help="the collapsed checkpoint. Without it only the two "
                         "reference anchors are measured and no w_g is derived.")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--n-crops", type=int, default=300,
                    help="matches the 300-crop 2026-08-20 loss anchor")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")   # loader appends "rendered/"
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")

    # The loss comes from the CURRENT config -- it must carry gain_delta_db,
    # which no checkpoint written before 2026-08-27 has.
    loss_fn = build_loss_fn(config)
    w, wm = float(config["loss"]["w"]), float(config["loss"]["w_m"])

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    dataset = TrialDataset(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        data_root=Path(args.data_root),
        split=val_audio,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed,
        random_crop=False,      # fixed crops: this must be re-runnable exactly
    )

    model, ckpt_meta = None, None
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        # Model built from the CHECKPOINT's config, not the current one. The
        # current file has three loss keys the checkpoint predates, so a strict
        # equality check would refuse a checkpoint whose WEIGHTS are perfectly
        # valid. Shape comes from the checkpoint, objective from the config.
        model = build_model(ckpt["config"])
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        ckpt_meta = {"path": args.checkpoint, "epoch": ckpt.get("epoch"),
                     "best_val": ckpt.get("best_val"), "seed": ckpt.get("seed")}

    names = [ORACLE, PASSTHROUGH] + ([MUTED, MODEL] if model is not None else [])
    sums = {n: defaultdict(float) for n in names}
    counts = {n: defaultdict(int) for n in names}
    rows, seen = [], 0

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs)

    scope = lambda: f"{seen} crops x {len(names)} systems, {args.split}"
    with timed("scripts/derive_w_g.py", scope=scope, rate=lambda: f"cpu, batch {args.batch_size}"):
        for batch in loader:
            if seen >= args.n_crops:
                break
            outputs = systems_for(batch, model, device)
            for name in names:
                _, parts = loss_fn(batch["target"], outputs[name],
                                   batch["mixture"], batch["crop_absent"])
                add_parts(sums[name], counts[name], parts)
                rows.append({"system": name, "n_present": parts["n_present"],
                             "n_absent": parts["n_absent"],
                             **{k: parts[k] for k in ("L_pres", "L_MR", "L_gain", "L_abs")}})
            seen += batch["mixture"].shape[0]

    stats = {n: epoch_report(sums[n], counts[n], w, wm, loss_fn.wg) for n in names}

    print(f"\n  {seen} crops from {val_manifest}, w={w} w_m={wm} "
          f"delta_db={loss_fn.gain_delta_db}\n")
    print(f"  {'system':<14}{'L_pres':>10}{'L_MR':>10}{'L_gain':>10}{'L_abs':>10}")
    for n in names:
        s = stats[n]
        fmt = lambda v: "     n/a " if (v is None or np.isnan(v)) else f"{v:9.4f}"
        print(f"  {n:<14}{fmt(s['L_pres'])}{fmt(s['L_MR'])}{fmt(s['L_gain'])}{fmt(s['L_abs'])}")

    # The oracle row is a wiring check, not a result: output IS the target, so
    # L_gain and L_MR must both be ~0. A nonzero one means the term is broken
    # and every number below it is meaningless.
    oracle_gain = stats[ORACLE]["L_gain"]
    ok = np.isnan(oracle_gain) or abs(oracle_gain) < 1e-3
    print(f"\n  wiring check 1: oracle L_gain = {oracle_gain:.6f} "
          f"({'OK' if ok else 'BROKEN -- stop here'})")
    # L_pres is scale-invariant, so muting the mixture must not move it at all.
    # If this drifts, either the invariance is broken or the muted anchor is not
    # actually the same audio -- and the derivation below would be meaningless.
    ok_si = True
    if MUTED in stats:
        drift = abs(stats[MUTED]["L_pres"] - stats[PASSTHROUGH]["L_pres"])
        ok_si = drift < 1e-3
        print(f"  wiring check 2: L_pres drift pt -> muted = {drift:.2e} "
              f"({'OK, scale-invariant' if ok_si else 'BROKEN -- stop here'})")
    ok = ok and ok_si

    derivation = None
    if model is not None:
        derivation = derive(stats, w, wm)
        print(f"\n  mute buys on absent crops   : {derivation['mute_buys_on_absent']:+.4f}")
        print(f"  already costs via L_MR      : {derivation['mute_already_costs_via_L_MR']:+.4f}")
        print(f"  L_gain headroom             : {derivation['L_gain_headroom']:+.4f}")
        if derivation["break_even_w_g"] is not None:
            print(f"\n  break-even w_g : {derivation['break_even_w_g']:.4f}")
            print(f"  ceiling    w_g : {derivation['ceiling_w_g']:.4f}")
        print(f"\n  {derivation['note']}")
        if derivation["suggested_w_g"] is not None:
            be, ce = derivation["break_even_w_g"], derivation["ceiling_w_g"]
            print(f"\n  ablate_w_g: [0.0, {derivation['suggested_w_g']:.2f}, {ce:.2f}]"
                  f"   # 0 arm required")
    else:
        print("\n  no --checkpoint: reference anchors only, no derivation.")

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today().isoformat()}-wg-anchor-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_crop.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    clean = lambda d: {k: (None if isinstance(v, float) and np.isnan(v) else v)
                       for k, v in d.items()}
    yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/derive_w_g.py",
        "git_commit": git_commit(),
        "seed": seed,
        "config": args.config,
        "split": args.split,
        "manifest": val_manifest,
        "n_crops": seen,
        "gain_delta_db": float(loss_fn.gain_delta_db),
        "w": w, "w_m": wm,
        "checkpoint": ckpt_meta,
        "anchors": {n: clean(stats[n]) for n in names},
        "wiring_checks_passed": bool(ok),
        "derivation": derivation,
    }, open(out_dir / "meta.yaml", "w"), sort_keys=False)
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
