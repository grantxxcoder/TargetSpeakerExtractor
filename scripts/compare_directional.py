#!/usr/bin/env python3
"""Paired comparison of two `diagnose_cue_directional.py` runs on the same crops.

    ../tse_venv/bin/python scripts/compare_directional.py \
        --a experiments/results/<run-a>/per_pair.csv --a-name "1a e15" \
        --b experiments/results/<run-b>/per_pair.csv --b-name "1c e12" \
        --out experiments/results/<dir>/comparison.yaml

WHAT IS COMPARED. On every crop with two live voices, each model makes two
decisions: asked for the target, does its output land nearer the target's stem
than the interferer's; asked for the interferer, nearer the interferer's. A
decision is "right" when it lands on the requested voice. The two models are
compared decision by decision on the SAME crops, so the only thing that differs
is the model.

TWO TESTS, because the two directions of one crop are not independent:
  * McNemar, exact two-sided (binomial on the discordant decisions), treating
    every decision as independent. Kept because decisions-m2.md 2026-09-23
    reported it that way, so the numbers are comparable.
  * Crop-level paired bootstrap and sign-flip permutation, resampling whole
    crops. This respects the pairing of the two directions and is the one to
    believe if the two disagree.

Selectivity (dB, higher is better) is also compared: the mean over both
directions of how much more of a speaker you get by asking for them. It is
continuous, so it can detect a smaller difference than the right/wrong rate.

SI-SDR as defined by Le Roux et al., ICASSP 2019, computed in
diagnose_cue_directional.py; this script only reads its per_pair.csv.
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import binomtest


def decisions(path):
    """per_pair.csv -> one row per crop with two live voices."""
    df = pd.read_csv(path)
    live = df[(df["target_live"] == True) & (df["interferer_live"] == True)].copy()  # noqa: E712
    live["right_t"] = live["sisdr_yt_st"] > live["sisdr_yt_si"]
    live["right_i"] = live["sisdr_yi_si"] > live["sisdr_yi_st"]
    live["sel"] = (live["sel_t"] + live["sel_i"]) / 2.0
    return live.set_index("trial_id")[
        ["same_gender", "sir_band", "right_t", "right_i", "sel"]]


def compare(a, b, rng, n_boot):
    """a, b: decision tables over the same crops. Returns a dict of results."""
    ra = np.stack([a["right_t"].values, a["right_i"].values], 1).astype(float)
    rb = np.stack([b["right_t"].values, b["right_i"].values], 1).astype(float)
    n_crops = len(a)
    if n_crops == 0:
        return {"n_crops": 0}

    # McNemar over decisions.
    b_only = int(((rb == 1) & (ra == 0)).sum())   # b right, a wrong
    a_only = int(((ra == 1) & (rb == 0)).sum())
    disc = a_only + b_only
    p_mcnemar = float(binomtest(b_only, disc, 0.5).pvalue) if disc else 1.0

    # Crop-level: per-crop difference in rate (b - a), in percentage points.
    d_rate = (rb.mean(1) - ra.mean(1)) * 100.0
    d_sel = b["sel"].values - a["sel"].values
    boot_rate, boot_sel = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n_crops, n_crops)
        boot_rate.append(d_rate[idx].mean())
        boot_sel.append(np.nanmean(d_sel[idx]))
    # Sign-flip permutation: under "no difference", swapping which model is
    # which within a crop is equally likely.
    flips = rng.choice([-1.0, 1.0], size=(n_boot, n_crops))
    perm_rate = (flips * d_rate).mean(1)
    perm_sel = np.nanmean(flips * np.nan_to_num(d_sel), 1)
    obs_rate, obs_sel = d_rate.mean(), np.nanmean(d_sel)

    return {
        "n_crops": int(n_crops),
        "n_decisions": int(2 * n_crops),
        "a_right_pct": round(float(ra.mean() * 100), 2),
        "b_right_pct": round(float(rb.mean() * 100), 2),
        "diff_pp": round(float(obs_rate), 2),
        "diff_pp_95ci": [round(float(np.percentile(boot_rate, 2.5)), 2),
                         round(float(np.percentile(boot_rate, 97.5)), 2)],
        "p_crop_permutation": round(float((np.abs(perm_rate) >= abs(obs_rate)).mean()), 4),
        "mcnemar_b_only": b_only,
        "mcnemar_a_only": a_only,
        "p_mcnemar": round(p_mcnemar, 4),
        "a_selectivity_db": round(float(np.nanmean(a["sel"])), 3),
        "b_selectivity_db": round(float(np.nanmean(b["sel"])), 3),
        "selectivity_diff_db": round(float(obs_sel), 3),
        "selectivity_diff_db_95ci": [round(float(np.percentile(boot_sel, 2.5)), 3),
                                     round(float(np.percentile(boot_sel, 97.5)), 3)],
        "p_selectivity_permutation": round(float((np.abs(perm_sel) >= abs(obs_sel)).mean()), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    a, b = decisions(args.a), decisions(args.b)
    common = a.index.intersection(b.index)
    # Both runs read the same crops, so "live" must agree; if it does not, the
    # runs did not score the same audio and nothing below is paired.
    assert len(common) == len(a) == len(b), (len(a), len(b), len(common))
    a, b = a.loc[common], b.loc[common]
    assert (a["same_gender"].values == b["same_gender"].values).all()

    groups = {"all": slice(None),
              "same_gender": a["same_gender"].values == 1,
              "cross_gender": a["same_gender"].values == 0}
    for band in sorted(a["sir_band"].unique()):
        groups[f"sir {band}"] = a["sir_band"].values == band

    rng = np.random.default_rng(args.seed)
    results = {}
    for name, mask in groups.items():
        results[name] = compare(a[mask], b[mask], rng, args.n_boot)

    meta = {"date": str(date.today()), "seed": args.seed, "n_boot": args.n_boot,
            "a": {"name": args.a_name, "per_pair": args.a},
            "b": {"name": args.b_name, "per_pair": args.b},
            "diff_sign": "b minus a; positive means b lands on the requested voice more often",
            "results": results}

    print(f"\n  {args.b_name} minus {args.a_name}, on the same crops "
          f"(positive = {args.b_name} better)\n")
    print(f"  {'group':<16}{'crops':>6}{args.a_name:>10}{args.b_name:>10}"
          f"{'diff pp':>9}{'95% CI':>18}{'p crop':>8}{'McNemar':>14}{'sel dB diff':>13}")
    for name, r in results.items():
        if not r.get("n_crops"):
            continue
        ci = f"[{r['diff_pp_95ci'][0]:+.1f}, {r['diff_pp_95ci'][1]:+.1f}]"
        mc = f"+{r['mcnemar_b_only']}/-{r['mcnemar_a_only']} p={r['p_mcnemar']:.3f}"
        print(f"  {name:<16}{r['n_crops']:>6}{r['a_right_pct']:>9.1f}%{r['b_right_pct']:>9.1f}%"
              f"{r['diff_pp']:>+9.1f}{ci:>18}{r['p_crop_permutation']:>8.3f}{mc:>14}"
              f"{r['selectivity_diff_db']:>+13.2f}")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(yaml.safe_dump(meta, sort_keys=False))
        print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
