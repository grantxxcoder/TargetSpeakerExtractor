"""D19: is the flat mask OURS, or is it what causal band-split masking does?

    ../tse_venv/bin/python scripts/measure_effective_mask_flatness.py \
        --systems wesep=experiments/results/2026-09-03-est-wesep-tfmap-causal,\
baseline=experiments/results/2026-09-04-train-sir0-10000,\
struct=experiments/results/2026-09-15-est-struct-e12-sir0val \
        --split sir0 --condition both --limit 103 \
        --out experiments/results/2026-09-15-effective-mask-flatness

THE QUESTION. MEASURED 2026-09-12/13: 84 % of our mask's variance is one number
per frame, and it varies ~6.5x less across frequency than the ideal mask -- a
broadband volume knob, which cannot in principle separate two voices that
overlap in time AND frequency. Three interventions have now been aimed at that
diagnosis (D14 state teacher, D15/D17 mask structure) and none improved what a
listener transcribes. Before a fourth, the diagnosis itself is worth testing.

WeSep `tfmap_context_causal_100` scores 26.40 LCF-WER on the same 103 trials
where we score 55.59 (floor 63.27, ceiling 1.05). It is CAUSAL, it is the same
TF-Map/band-split family, and it is out of domain by its own config. So:

  IF WESEP'S MASK IS STRUCTURED (f/t near the ideal's ~1.01) -- a better causal
  band-split model on our own trials does NOT use a volume knob, so the knob is
  a property of OUR OBJECTIVE. That is the strongest available support for the
  structure line of work, and it comes from a system we did not build.

  IF WESEP'S MASK IS ALSO FLAT (f/t ~0.3-0.6, like ours) -- the knob is what
  this architecture class does, it does NOT explain a 29-point gap, and D15/D17/
  D18 are the wrong tree. M5's remaining weeks should go elsewhere.

Either answer redirects M5, and neither costs a training run.

WHY THE *EFFECTIVE* MASK, AND THIS IS THE LOAD-BEARING DESIGN CHOICE.
scripts/measure_mask_flatness.py reads our model's INTERNAL mask tensor
(`estimator.last_parts["mask_mag"]`). WeSep has no such tensor to read: its
`band_masker` emits an estimated complex spectrogram directly
(wesep/models/tse_bsrnn_spk.py S5), so there is no internal magnitude mask to
extract, and inventing one would mean reimplementing its masker.

So both systems are measured by the mask their OUTPUT IMPLIES:

    M_eff = |STFT(estimate)| / |STFT(mixture)|,  clipped to [0, 2]

This is defined identically for every system, needs nothing but the rendered
wavs, and is exactly the quantity a downstream listener is affected by. It is
the definition this script uses for ALL systems INCLUDING ours, so every number
in one table is like-for-like.

DO NOT COMPARE THESE NUMBERS TO THE 2026-09-13 INTERNAL-MASK TABLE. Different
definition. Our internal mask is one factor inside a model that also has a
residual branch and a complex/phase path; the effective mask folds all of that
in. The baseline appears in this table precisely so the comparison stays inside
one definition. (Cross-check available: R is inert, decisions-m3.md 2026-09-13,
so the two definitions should not be far apart for our model -- if they are,
that is itself worth knowing and this script's baseline row is what reveals it.)

THE ANALYSIS GRID IS NOT ARBITRARY. n_fft 512 / hop 128 at 16 kHz is OUR
config's STFT (bsrnn_baseline.yaml) and ALSO WeSep's (`win: 512, stride: 128` in
tfmap_context_causal_100/config.yaml). The two models happen to share a grid, so
this is each system's own framing rather than a third one imposed on both.

CAVEATS THAT TRAVEL WITH ANY WESEP ROW (see make_estimates_wesep.py for the
full list, quoted from its own config):
  * WeSep was trained with SISDR against Libri2Mix's DRY source, so it
    DEREVERBERATES. Our reference is the full reverberant target (A1). Removing
    reverb tail shows up in M_eff as extra time-frequency structure that our
    model is not being asked to produce. This alone could raise its f/t, and
    it is NOT evidence about separation.
  * 33.5 M parameters against our 7.19 M, ~14.6 M of it a VoxCeleb ECAPA-TDNN.
  * Different data, different objective, different budget. Nothing here is a
    comparison claim against a published REAL-TSE number; it is a diagnostic on
    OUR model, using a second system as a reference point on OUR trials.

NOT A CONTENT RESULT. Flatness is a property of a mask, not of what a listener
transcribes (the same warning measure_mask_flatness.py carries). It selects
which intervention to try next. It never scores one.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.estimates.runner import read_trials, git_commit          # noqa: E402
from src.run_log import timed                                     # noqa: E402
from measure_mask_flatness import flatness, read_mono             # noqa: E402
from train import SPLIT_MANIFESTS                                 # noqa: E402

N_FFT, HOP = 512, 128
CLIP_MAX = 2.0          # same clip measure_mask_flatness.py puts on the ideal
LOUD_DB = 50.0          # same "loud" rule as plot_mask_grid.py


def magnitude(x):
    """|STFT| on the shared analysis grid. (F, T) float64."""
    spec = torch.stft(torch.from_numpy(np.asarray(x, dtype=np.float32)),
                      n_fft=N_FFT, hop_length=HOP,
                      window=torch.hann_window(N_FFT),
                      center=True, return_complex=True)
    return spec.abs().numpy().astype(np.float64)


def effective_mask(estimate, mixture):
    """|E| / |X|, clipped. Both already trimmed to a common length."""
    return np.clip(estimate / np.maximum(mixture, 1e-8), 0.0, CLIP_MAX)


def row_for(mask, ideal, target_mag):
    """flatness of one mask and of the ideal, over all bins and over loud bins."""
    out = {}
    db = 20 * np.log10(target_mag + 1e-6)
    loud = db > (db.max() - LOUD_DB)
    for scope, sel in (("all", None), ("loud", loud)):
        if sel is None:
            m, idl = mask, ideal
        else:
            keep = sel.any(axis=1)
            if keep.sum() < 2:
                continue
            m, idl = mask[keep], ideal[keep]
        for name, arr in (("ours", m), ("ideal", idl)):
            share, dt, df = flatness(arr)
            out[f"{scope}_{name}_frame_mean_share"] = share
            out[f"{scope}_{name}_d_time"] = dt
            out[f"{scope}_{name}_d_freq"] = df
            out[f"{scope}_{name}_freq_over_time"] = df / dt if dt else np.nan
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--systems", required=True,
                    help="comma-separated NAME=ESTIMATE_DIR. Every system is "
                         "scored on the SAME trials -- the intersection of the "
                         "trial ids all of them rendered.")
    ap.add_argument("--split", default="sir0", choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--condition", default="both")
    ap.add_argument("--limit", type=int, default=103)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    systems = {}
    for item in args.systems.split(","):
        name, _, directory = item.strip().partition("=")
        if not directory:
            raise SystemExit(f"--systems wants NAME=DIR, got {item!r}")
        systems[name] = Path(directory)
        if not systems[name].is_dir():
            raise SystemExit(f"no such estimate directory: {directory}")

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    trials = read_trials(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        audio_root=Path(args.data_root) / "rendered" / val_audio,
        limit=None, condition=args.condition)

    # SAME TRIALS FOR EVERY SYSTEM. The renders differ in trial count (200 for
    # the baseline, which was rendered over every condition; 103 for the rest),
    # so the intersection is taken rather than the first N of each -- otherwise
    # the systems would be compared on different audio.
    usable = [t for t in trials
              if all((d / t.trial_id / "estimate.wav").exists() for d in systems.values())]
    if not usable:
        raise SystemExit("no trial is rendered by every system named in --systems")
    usable = usable[:args.limit]

    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-effective-mask-flatness")
    out_root.mkdir(parents=True, exist_ok=True)

    per_system = {name: [] for name in systems}
    with timed("scripts/measure_effective_mask_flatness.py",
               scope=lambda: f"{len(systems)} systems x {len(usable)} trials",
               rate=lambda: "cpu, whole-clip, no model inference"):
        for i, t in enumerate(usable, 1):
            mixture = magnitude(read_mono(t.directory / "mixture.wav"))
            target = magnitude(read_mono(t.directory / "target.wav"))
            for name, directory in systems.items():
                est = magnitude(read_mono(directory / t.trial_id / "estimate.wav"))
                # Renders can differ from the mixture by a frame of padding.
                f = min(mixture.shape[0], target.shape[0], est.shape[0])
                n = min(mixture.shape[1], target.shape[1], est.shape[1])
                X, S, E = mixture[:f, :n], target[:f, :n], est[:f, :n]
                row = row_for(effective_mask(E, X), effective_mask(S, X), S)
                row["trial_id"] = t.trial_id
                per_system[name].append(row)
            if i % 25 == 0 or i == len(usable):
                print(f"  {i}/{len(usable)}", flush=True)

    summary = {
        "n_trials": len(usable),
        "split": args.split,
        "condition": args.condition,
        "n_fft": N_FFT, "hop": HOP,
        "mask": "effective: |STFT(estimate)| / |STFT(mixture)|, clipped [0, 2]",
        "git_commit": git_commit(),
        "date": date.today().isoformat(),
        "systems": {},
    }
    for name, rows in per_system.items():
        keys = [k for k in rows[0] if k != "trial_id"]
        summary["systems"][name] = {
            "estimate_directory": str(systems[name]),
            "mean": {k: float(np.nanmean([r.get(k, np.nan) for r in rows]))
                     for k in keys},
        }

    json.dump(per_system, open(out_root / "per_trial.json", "w"), indent=1)
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)

    hdr = (f"  {'system':14s} {'share(all)':>11s} {'share(loud)':>12s} "
           f"{'f/t all':>8s} {'f/t loud':>9s} {'d_freq':>8s} {'d_time':>8s}")
    lines = [f"\n  wrote {out_root}", "", hdr, "  " + "-" * (len(hdr) - 2)]
    for name, blob in summary["systems"].items():
        m = blob["mean"]
        lines.append(f"  {name:14s} {m['all_ours_frame_mean_share']:11.4f} "
                     f"{m['loud_ours_frame_mean_share']:12.4f} "
                     f"{m['all_ours_freq_over_time']:8.4f} "
                     f"{m['loud_ours_freq_over_time']:9.4f} "
                     f"{m['loud_ours_d_freq']:8.4f} {m['loud_ours_d_time']:8.4f}")
    any_mean = next(iter(summary["systems"].values()))["mean"]
    lines.append(f"  {'IDEAL':14s} {any_mean['all_ideal_frame_mean_share']:11.4f} "
                 f"{any_mean['loud_ideal_frame_mean_share']:12.4f} "
                 f"{any_mean['all_ideal_freq_over_time']:8.4f} "
                 f"{any_mean['loud_ideal_freq_over_time']:9.4f} "
                 f"{any_mean['loud_ideal_d_freq']:8.4f} "
                 f"{any_mean['loud_ideal_d_time']:8.4f}")
    lines += ["",
              "  share -> 1.0 is a pure volume knob. LOWER is better.",
              "  f/t   -> variation across frequency over across time. The ideal",
              "           mask sets the target; ours must RISE toward it.",
              "  Effective masks: NOT comparable with the 2026-09-13 internal-mask",
              "  table. WeSep dereverberates by design -- see the module docstring.",
              ""]
    print("\n".join(lines))
    (out_root / "results.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
