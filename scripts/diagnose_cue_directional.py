#!/usr/bin/env python3
"""Item 2: the DIRECTIONAL enrollment-swap test. Does the output move TOWARD the
other speaker, or does it only change level?

    ../tse_venv/bin/python scripts/diagnose_cue_directional.py --split sir0 \
        --checkpoint models/model_sir0_10000-e6.pt

Writes experiments/results/<date>-cue-directional-<split>/{per_pair.csv,meta.yaml}.

WHY THIS EXISTS
---------------
`diagnose_cue.py` measures ||a - b||^2 / ||a||^2 -- a MAGNITUDE with no
direction. "The output moves 48.2 % when the enrollment is swapped" is fully
consistent with the output only changing VOLUME, which is the behaviour under
investigation. D5 (the discriminative speaker encoder) was demoted on that
evidence on 2026-08-30 and REOPENED on 2026-09-21 for exactly this reason: the
directional test was never run. decisions-pending.md D5,
ranked-next-steps.md item 2.

THE TEST
--------
The data already contains the swap. `both_directions` renders every trial twice
-- the SAME mixture asked for the target and asked for the other speaker, each
with its own enrollment and its own ground-truth stem (dataset_loader.py:253).
So the swap is not a perturbation we invent; it is a second, equally valid
request with a known correct answer.

    y_t = model(mixture, enrollment_target)       should reconstruct s_t
    y_i = model(mixture, enrollment_interferer)   should reconstruct s_i

Score every output against BOTH stems and read two different things:

    SELECTIVITY (dB, higher is better, 0 means the enrollment was ignored)
        sel_t = SI-SDR(y_t, s_t) - SI-SDR(y_i, s_t)
        sel_i = SI-SDR(y_i, s_i) - SI-SDR(y_t, s_i)

    How much more of a speaker do you get by actually asking for them? A model
    that ignores the enrollment emits the same audio either way, so both
    differences are exactly 0 however good or bad that audio is. This is what
    `rel_movement` cannot see: it reports the two outputs differing without
    saying whether the difference is the requested speaker or a gain change.

    FOLLOWS_REQUEST (%, 50 is a coin flip)
        Did the output land NEARER the stem that was asked for than the other?

    These two answer different questions and BOTH are needed. Selectivity is a
    continuous nudge; follows_request is whether the nudge was big enough to
    arrive. An output can move toward the right speaker and still end up closer
    to the wrong one, which is exactly what this model does on same-gender
    pairs (2026-09-22).

    TRACKS_LOUDER (%)
        Nearer the LOUDER stem, whoever was asked for. The loudness-meter
        hypothesis as a number: corr(alpha_t, mixture loudness) = 0.990 predicts
        it is high and roughly equal in both directions.

BOTH-LIVE IS THE HONEST DENOMINATOR
------------------------------------
`condition` labels the CLIP; this scores a 4 s CROP. Measured 2026-09-22: 26 of
103 `both`-labelled trials have one speaker silent in the crop actually scored.
Those crops cannot answer an identity question -- with one voice there is no
choice to make -- so every rate here is computed over crops with TWO live
voices, and `n_both_live` is reported beside `n`. Same trap as losses.py:240,
which is why `crop_absent` comes from the loader and never from the manifest.

WHY PLAIN SI-SDR AND NOT THE TRAINING TERM
-------------------------------------------
Scale-invariant SDR as defined by Le Roux, Wisdom, Erdogan & Hershey, "SDR --
half-baked or well done?", ICASSP 2019. NOT `losses.py::_loss_target_present`,
which is the floored CARTSE variant: its tau floor caps the score at +30 dB and
is tuned for gradients, not for reading. Sign convention here is the ordinary
one -- HIGHER IS BETTER -- where the loss term is negated. Do not quote a number
from this script against a `L_pres` number without converting.

NOT SI-SDRi. ranked-next-steps.md item 2: improvement-over-input rises as the
input gets worse, so on the low-SIR slice it improves while the audio degrades.
Every number here is a difference between two OUTPUTS on the SAME input, which
has no such pathology.

HOW TO READ IT
--------------
    selectivity ~ 0, tracks_louder high    the cue is a loudness meter. The
                                           48.2 % movement was level. D5 stands.
    selectivity > 0, follows_request       the cue separates voices that differ
    high cross-gender and ~50 % same       in PITCH and fails on voices that do
                                           not. The case for a discriminative
                                           speaker encoder (D5 / item 1c).
    follows_request high in BOTH           the cue carries identity. Conditioning
                                           is not the bottleneck -- look at the
                                           separator or the objective.
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
from train import (SPLIT_MANIFESTS, build_context_encoder,  # noqa: E402
                   build_model, context_kwargs, git_commit)

# A stem whose energy is below this fraction of the mixture's is treated as
# silent and every score against it is NaN rather than a number. SI-SDR against
# an all-zero reference is 0/0; against a near-zero one it is dominated by
# whatever numerical dust is in the stem.
SILENT_REL_ENERGY = 1e-8

SIR_BANDS = ((-np.inf, -5.0, "< -5"), (-5.0, 0.0, "-5..0"),
             (0.0, 5.0, "0..+5"), (5.0, np.inf, ">= +5"))


def si_sdr(estimate, reference, eps=1e-12):
    """Scale-invariant SDR in dB, HIGHER IS BETTER. (B, T) -> (B,).

    Le Roux et al., ICASSP 2019. NaN where the reference is silent.
    """
    ref_energy = reference.pow(2).sum(dim=-1)
    alpha = ((estimate * reference).sum(dim=-1) / (ref_energy + eps)).unsqueeze(-1)
    projection = alpha * reference
    residual = estimate - projection
    out = 10.0 * torch.log10((projection.pow(2).sum(dim=-1) + eps)
                             / (residual.pow(2).sum(dim=-1) + eps))
    return torch.where(ref_energy > 0, out, torch.full_like(out, float("nan")))


def sir_band(sir_db):
    for low, high, name in SIR_BANDS:
        if low <= sir_db < high:
            return name
    return SIR_BANDS[-1][2]


def mean_or_nan(values):
    """np.nanmean over an all-NaN slice warns and returns NaN; say it once,
    quietly, rather than letting a stratum table fill with warnings."""
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else float("nan")


def summarise(subset):
    """Rates over BOTH-LIVE crops; dB means over whatever is finite."""
    live = [r for r in subset if r["target_live"] and r["interferer_live"]]
    asked = []
    louder = []
    for r in live:
        asked.append(r["sisdr_yt_st"] > r["sisdr_yt_si"])
        asked.append(r["sisdr_yi_si"] > r["sisdr_yi_st"])
        # sir_db is the TARGET's level relative to the interferer, so sir > 0
        # means the target is the loud one for BOTH requests.
        louder.append((r["sisdr_yt_st"] > r["sisdr_yt_si"]) == (r["sir_db"] > 0))
        louder.append((r["sisdr_yi_st"] > r["sisdr_yi_si"]) == (r["sir_db"] > 0))
    sel_t = mean_or_nan([r["sel_t"] for r in subset])
    sel_i = mean_or_nan([r["sel_i"] for r in subset])
    return {
        "n": len(subset), "n_both_live": len(live),
        "sel_t_db": sel_t, "sel_i_db": sel_i,
        "sel_mean_db": mean_or_nan([sel_t, sel_i]),
        "follows_request": float(np.mean(asked)) if asked else float("nan"),
        "tracks_louder": float(np.mean(louder)) if louder else float("nan"),
        "rel_movement_pct": 100.0 * mean_or_nan([r["rel_movement"] for r in subset]),
        "level_yt_db": mean_or_nan([r["level_yt_db"] for r in subset]),
        "level_yi_db": mean_or_nan([r["level_yi_db"] for r in subset]),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", required=True,
                    help="unlike diagnose_cue.py this NEEDS a trained model: the "
                         "whole question is about the output, and an untrained "
                         "separator has no behaviour to attribute.")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--n-trials", type=int, default=200,
                    help="B6's floor for a scored measurement is 200")
    ap.add_argument("--batch-size", type=int, default=4, help="counts TRIALS")
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
        random_crop=False,      # fixed crops: this must be re-runnable exactly
        both_directions=True,   # THE POINT: same mixture, two requests, two stems
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    # ITEM 1c. THIS SCRIPT IS THE ARM'S REGISTERED ACCEPTANCE TEST, so it has to
    # run on a 1c checkpoint or the arm cannot be judged at all. Encoder from
    # the CHECKPOINT's config, so the swap below is scored with the same
    # embedding the run was trained with.
    encoder = build_context_encoder(ckpt["config"], device)

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_pairs)

    rows = []
    seen = 0
    scope = lambda: f"{seen} trials x 2 directions, {args.split}"
    with timed("scripts/diagnose_cue_directional.py", scope=scope,
               rate=lambda: f"cpu, batch {args.batch_size} trials"):
        with torch.no_grad():
            for batch in loader:
                if seen >= args.n_trials:
                    break
                # collate_pairs flattens [[t, i], [t, i], ...] so the directions
                # interleave. ASSERTED, not assumed: a loader change that
                # reordered them would silently invert every number below.
                direction = batch["direction"]
                assert list(direction[0::2]) == ["target"] * len(direction[0::2]), direction
                assert list(direction[1::2]) == ["interferer"] * len(direction[1::2]), direction

                mixture = batch["mixture"].to(device)
                enroll = batch["enrollment"].to(device)
                stems = batch["target"].to(device)

                mix_t, mix_i = mixture[0::2], mixture[1::2]
                # Both directions are handed the SAME mixture by
                # dataset_loader._get_item. If that stops being true the
                # cross-scores below compare outputs of different inputs.
                assert torch.allclose(mix_t, mix_i), "directions saw different mixtures"

                enroll_t, enroll_i = enroll[0::2], enroll[1::2]
                s_t, s_i = stems[0::2], stems[1::2]

                # BOTH HALVES OF THE CONDITIONING SWAP, which is the whole
                # point of this script: the enrolment feeds the cue AND the
                # identity anchor, and embedding only one of them would measure
                # a model that was never asked for the other speaker.
                y_t = model(mix_t, enroll_t, **context_kwargs(encoder, enroll_t))
                y_i = model(mix_t, enroll_i, **context_kwargs(encoder, enroll_i))

                # Silence guard, applied to the STEM not the output: a silent
                # stem makes its column of the matrix undefined, and that is a
                # property of the trial, not of the model.
                mix_energy = mix_t.pow(2).sum(dim=-1)
                live_t = s_t.pow(2).sum(dim=-1) > SILENT_REL_ENERGY * mix_energy
                live_i = s_i.pow(2).sum(dim=-1) > SILENT_REL_ENERGY * mix_energy

                def scored(est, ref, live):
                    out = si_sdr(est, ref)
                    return torch.where(live, out,
                                       torch.full_like(out, float("nan"))).cpu().numpy()

                tt, ti = scored(y_t, s_t, live_t), scored(y_t, s_i, live_i)
                it, ii = scored(y_i, s_t, live_t), scored(y_i, s_i, live_i)

                # The old statistic, computed here so the two are read off one
                # run rather than compared across scripts and dates.
                movement = ((y_t - y_i).pow(2).sum(dim=-1)
                            / y_t.pow(2).sum(dim=-1).clamp_min(1e-12)).cpu().numpy()

                # OUTPUT LEVEL relative to the mixture, per direction. Without it
                # a large `sel` is ambiguous: a model that MUTES when asked for an
                # absent speaker scores like one that swaps to the right voice.
                # Three failures hide inside one WER -- wrong speaker,
                # near-silence, passthrough -- and they need opposite fixes.
                def level_db(y):
                    return (10.0 * torch.log10((y.pow(2).sum(dim=-1) + 1e-12)
                                               / (mix_energy + 1e-12))).cpu().numpy()
                lvl_t, lvl_i = level_db(y_t), level_db(y_i)

                meta = batch["meta"]
                for k in range(mix_t.shape[0]):
                    sir = float(meta["sir_db"][2 * k])
                    rows.append({
                        "trial_id": batch["trial_id"][2 * k],
                        "condition": str(meta["condition"][2 * k]),
                        "sir_db": sir,
                        "sir_band": sir_band(sir),
                        "same_gender": float(meta["same_gender"][2 * k]),
                        "target_live": bool(live_t[k]),
                        "interferer_live": bool(live_i[k]),
                        "sisdr_yt_st": float(tt[k]), "sisdr_yt_si": float(ti[k]),
                        "sisdr_yi_st": float(it[k]), "sisdr_yi_si": float(ii[k]),
                        "sel_t": float(tt[k] - it[k]),
                        "sel_i": float(ii[k] - ti[k]),
                        "rel_movement": float(movement[k]),
                        "level_yt_db": float(lvl_t[k]),
                        "level_yi_db": float(lvl_i[k]),
                    })
                seen += mix_t.shape[0]

    strata = {"all": rows}
    for _, _, name in SIR_BANDS:
        strata[f"sir {name}"] = [r for r in rows if r["sir_band"] == name]
    strata["same_gender"] = [r for r in rows if r["same_gender"] == 1.0]
    strata["cross_gender"] = [r for r in rows if r["same_gender"] != 1.0]
    for cond in sorted({r["condition"] for r in rows}):
        strata[f"cond {cond}"] = [r for r in rows if r["condition"] == cond]
    summary = {k: summarise(v) for k, v in strata.items() if v}

    out_dir = Path(args.out_dir or
                   f"experiments/results/{date.today()}-cue-directional-{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_pair.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    with open(out_dir / "meta.yaml", "w") as fh:
        yaml.safe_dump({
            "date": str(date.today()), "git_commit": git_commit(), "seed": seed,
            "split": args.split, "manifest": val_manifest,
            "checkpoint": str(args.checkpoint),
            "checkpoint_epoch": int(ckpt["epoch"]) if "epoch" in ckpt else None,
            "config": str(args.config), "n_trials": seen,
            "si_sdr": "Le Roux et al. ICASSP 2019, unfloored, higher is better",
            "rates_computed_over": "crops with two live voices (n_both_live)",
            "summary": summary,
        }, fh, sort_keys=False)

    print(f"\n  {seen} trials x 2 directions from {val_manifest}, "
          f"{Path(args.checkpoint).name}\n")
    print(f"  {'stratum':<22}{'n':>5}{'live':>6}{'sel_t':>8}{'sel_i':>8}"
          f"{'asked':>8}{'louder':>8}{'move':>8}{'lvl_yt':>8}{'lvl_yi':>8}")
    print(f"  {'':<22}{'':>5}{'':>6}{'dB':>8}{'dB':>8}{'%':>8}{'%':>8}"
          f"{'%':>8}{'dB':>8}{'dB':>8}")
    for name, row in summary.items():
        print(f"  {name:<22}{row['n']:>5}{row['n_both_live']:>6}"
              f"{row['sel_t_db']:>8.2f}{row['sel_i_db']:>8.2f}"
              f"{100 * row['follows_request']:>8.1f}{100 * row['tracks_louder']:>8.1f}"
              f"{row['rel_movement_pct']:>8.0f}{row['level_yt_db']:>8.2f}"
              f"{row['level_yi_db']:>8.2f}")

    # ITEM 1c's OTHER REGISTERED CRITERION. gamma near zero means the identity
    # anchor is being IGNORED, in which case every other number on this page is
    # about item 1a and must not be reported as 1c's. Cheap, so there is no
    # excuse for not having it beside the result it qualifies.
    if hasattr(model, "context"):
        with torch.no_grad():
            g = model.context.gamma(encoder.embed(
                next(iter(loader))["enrollment"].to(device)))
        print(f"\n  IDENTITY ANCHOR  ||gamma|| mean {g.norm(dim=-1).mean():.4f}  "
              f"max |gamma| {g.abs().max():.4f}  over {g.shape[1]} features")
        if g.abs().max() < 1e-3:
            print("    NEAR ZERO -- the anchor is being ignored. Everything below")
            print("    is about item 1a, not 1c. Do not report it as 1c's result.")

    a = summary["all"]
    print("\n  READING")
    if not a["n_both_live"]:
        print("    No two-voice crops in this sample. Nothing to read.")
        print(f"\n  wrote {out_dir}/")
        return
    print(f"    {a['n_both_live']} crops hold TWO live voices (of {a['n']} scored) --")
    print(f"    the manifest labels the CLIP, the crop may hold one voice.")
    print(f"    Asking for the other speaker buys {a['sel_mean_db']:.2f} dB of that")
    print(f"    speaker. 0 dB would mean the enrollment was ignored entirely.")
    if abs(a["sel_mean_db"]) < 1.0:
        print("    NEAR ZERO -- the output does not move TOWARD whoever was asked")
        print(f"    for, although it moves {a['rel_movement_pct']:.0f} % in magnitude.")
        print("    That movement is level, not identity. D5 stands.")
    else:
        print("    The enrollment DOES steer the output. But selectivity alone")
        print("    cannot say whether it steers FAR ENOUGH, so read the split:")

    # THE DECIDING SPLIT, and the reason the pooled number must not be read
    # alone. A model riding PITCH gets the cross-gender half right and coin-flips
    # the rest (D3c). `follows_request` is the binary "did it land nearer the
    # voice that was asked for", which selectivity -- a continuous dB nudge --
    # cannot tell you.
    same, cross = summary.get("same_gender"), summary.get("cross_gender")
    if same and cross and same["n_both_live"] and cross["n_both_live"]:
        print(f"      cross-gender {100 * cross['follows_request']:>5.1f} % of requests "
              f"land on the right voice ({cross['n_both_live']} crops)")
        print(f"      same-gender  {100 * same['follows_request']:>5.1f} % "
              f"({same['n_both_live']} crops) -- 50 % is a coin flip")
        if same["follows_request"] < 0.6 <= cross["follows_request"]:
            print("    PITCH, NOT IDENTITY. The cue separates voices that differ in")
            print("    pitch and fails on voices that do not -- which is the case for")
            print("    a discriminative speaker encoder (D5 / ranked item 1c),")
            print("    trained to tell speakers apart WITHIN gender.")
        elif same["follows_request"] >= 0.6:
            print("    The cue carries identity beyond pitch. Conditioning is not the")
            print("    bottleneck -- look at the separator or the objective.")
    else:
        print("    No gender split available; do not read the pooled number alone.")
    print(f"\n  wrote {out_dir}/")


if __name__ == "__main__":
    main()
