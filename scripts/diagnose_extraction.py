#!/usr/bin/env python3
"""Is the model doing target extraction, or has it collapsed to a uniform mute?

WHY THIS EXISTS
---------------
None of the four logged loss terms can show a mute. `L_pres` is scale-invariant
(it projects onto the target), so an output at 1/30th volume scores identically
to one at correct volume. `L_abs` rewards silence outright. `L_MR` notices, but
only indirectly and late. The 2026-08-24 smoke run collapsed to a near-silent,
enrolment-blind output while its loss curve looked healthy the whole way down.

This runs the two tests that actually settle it, on a saved checkpoint.

    TEST 1  enrolment sensitivity
            Feed the same mixture twice: once with the correct enrolment, once
            with another crop's. The enrolment is the ONLY thing saying which
            voice to extract, so if the output barely moves the model is
            ignoring it and doing generic enhancement, not extraction.

    TEST 2  present/absent discrimination
            Output loudness on crops where the target speaks, minus crops where
            they do not. A working extractor is loud when the target speaks and
            quiet when they do not. Near zero means it is equally quiet in both
            cases -- the mute.

REFERENCE (smoke, 20 speakers, epoch 95, decisions-m1.md 2026-08-25):
    enrolment sensitivity   -17.15 dB      (output moved 2% of its energy)
    L_pres cost of a swap    +0.62 dB
    discrimination gap       +1.34 dB      (equally quiet either way)

These are the same two quantities train.py now logs per epoch as
`val_enrol_sens_db` and `val_pres_abs_gap_db`. This script is for checkpoints
that predate that logging, and for a closer look at one checkpoint.

USAGE
-----
    python scripts/diagnose_extraction.py --checkpoint models/model_mid.pt --split mid
    python scripts/diagnose_extraction.py --checkpoint models/model_smoke.pt --split smoke

CPU is slow: two forward passes over the whole val set. Pass --limit to cut it
short while iterating.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import (build_loss_fn, build_model,  # noqa: E402
                           get_data_loaders, unpack)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", required=True, help="smoke | mid | full")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--limit", type=int, default=None,
                    help="stop after N val batches (for a quick look)")
    ap.add_argument("--device", default=None, help="default: cuda if available")
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # weights_only=False: the checkpoint carries the config dict, not just
    # tensors. Safe because we wrote it. Same reasoning as train.py's resume.
    ck = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ck["config"]

    # The checkpoint's OWN config, never the yaml on disk: the file may have
    # moved on since, and rebuilding a different model would silently measure
    # something else.
    torch.manual_seed(int(cfg["seed"]))
    model = build_model(cfg)
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    loss_fn = build_loss_fn(cfg)
    tau_pres = cfg["loss"]["tau_pres"]

    _, val = get_data_loaders(args.split, Path(args.manifest_dir),
                              Path(args.data_root), cfg)
    print(f"checkpoint  {args.checkpoint}")
    print(f"  epoch {ck.get('epoch')}  best_val {ck.get('best_val'):.4f}  "
          f"batch_size {cfg['data']['batch_size']}  seed {cfg['seed']}  device {device}")

    # Accumulated in the loop rather than stored: the val set is small but there
    # is no reason to hold two full copies of the audio in memory.
    swap_num = swap_den = 0.0       # TEST 1, summed energies
    e_pres = e_abs = 0.0            # TEST 2, summed per-crop dB
    n_pres = n_abs = 0
    lp_ok_sum = lp_sw_sum = 0.0     # L_pres with correct vs swapped enrolment
    n_swap_crops = 0

    with torch.no_grad():
        for i, batch in enumerate(val):
            if args.limit is not None and i >= args.limit:
                break
            x, s, e, a = unpack(batch, device)
            absent = a.bool()
            present = ~absent
            y = model(x, e)

            # TEST 1. Roll by one within the batch: the wrong enrolment, with no
            # second pass over the data. At batch size 1 roll() returns the SAME
            # enrolment and would read a false 0 dB, so skip those batches.
            # With 940 target speakers a rolled pair sharing a speaker is rare.
            if x.shape[0] > 1:
                y_sw = model(x, e.roll(1, 0))
                # ratio of SUMMED energies, not a mean of per-crop ratios: a
                # near-silent crop has a tiny denominator and would dominate an
                # average of ratios.
                swap_num += float((y - y_sw).pow(2).sum())
                swap_den += float(y.pow(2).sum())
                if present.any():
                    lp_ok_sum += float(loss_fn._loss_target_present(
                        s[present], y[present], tau_pres).sum())
                    lp_sw_sum += float(loss_fn._loss_target_present(
                        s[present], y_sw[present], tau_pres).sum())
                    n_swap_crops += int(present.sum())

            # TEST 2. Per-crop output energy relative to its own mixture, in dB.
            e_db = 10 * torch.log10(y.pow(2).sum(-1) / x.pow(2).sum(-1) + 1e-12)
            if present.any():
                e_pres += float(e_db[present].sum()); n_pres += int(present.sum())
            if absent.any():
                e_abs += float(e_db[absent].sum()); n_abs += int(absent.sum())

            print(f"  batch {i + 1}", end="\r", flush=True)

    nan = float("nan")
    sens = 10 * math.log10(swap_num / swap_den) if swap_den > 0 and swap_num > 0 else nan
    lp_ok = lp_ok_sum / n_swap_crops if n_swap_crops else nan
    lp_sw = lp_sw_sum / n_swap_crops if n_swap_crops else nan
    ep = e_pres / n_pres if n_pres else nan
    ea = e_abs / n_abs if n_abs else nan

    print(f"\nval  {n_pres} present, {n_abs} absent crops"
          f"  ({n_swap_crops} present crops usable for TEST 1)")

    print("\nTEST 1  enrolment sensitivity")
    print(f"  output change when the enrolment is swapped : {sens:+7.2f} dB")
    print(f"  L_pres, correct enrolment                   : {lp_ok:7.4f}")
    print(f"  L_pres, WRONG enrolment                     : {lp_sw:7.4f}")
    print(f"  cost of the wrong enrolment                 : {lp_sw - lp_ok:+7.4f} dB")
    print( "    smoke epoch 95 reference                  :  -17.15 dB / +0.6200 dB")
    print( "  near 0 dB  = strongly conditioned on the enrolment")
    print( "  very -ve   = ignoring it; generic enhancement, not extraction")

    print("\nTEST 2  present/absent discrimination")
    print(f"  output vs mixture, PRESENT crops            : {ep:+7.2f} dB")
    print(f"  output vs mixture, ABSENT  crops            : {ea:+7.2f} dB")
    print(f"  gap                                         : {ep - ea:+7.2f} dB")
    print( "    smoke epoch 95 reference                  :  +1.34 dB (the mute)")
    print( "  large +ve  = loud when the target speaks, quiet when not")
    print( "  near 0     = equally quiet either way, i.e. muted")


if __name__ == "__main__":
    main()
