"""What does the additive residual branch R actually carry?

    ../tse_venv/bin/python scripts/diagnose_residual.py \
        --split sir0_val --condition both \
        --checkpoint models/model_sir0_10000-e6.pt \
        --out experiments/results/2026-09-13-residual-share

THE QUESTION. `Estimator` returns S = M (x) X + R (Yu et al., Interspeech 2023,
eq. 2). The mask is MULTIPLIED by the mixture, so in a bin where |X| = 0 the
mask contributes exactly 0 no matter what it predicts. R is ADDED, comes from a
raw Conv1d with neither GLU nor bound, and is therefore the ONLY path by which
this model can place energy in a time-frequency cell the microphone never
recorded. Emitting sound that was never said is, physically, what an invented
word is -- and invented words are 308 of our 742 wrong content words, 41.5 % of
the error mass, with no mechanism assigned to them (decisions-pending.md,
2026-09-11). D6 flagged R as unbounded and unconditioned. Nothing has ever
looked inside it.

WHAT IS MEASURED, per trial, on the model's own STFT grid:

  residual_energy_share   |R|^2 / |S|^2 summed over all bins. Blunt: R and the
                          masked path are complex and can cancel, so this alone
                          overstates nothing but explains little.
  residual_projection     Re<R, S> / |S|^2. How much of the output R actually
                          ACCOUNTS FOR once cancellation is taken into account.
                          Negative means R is mostly subtracting.
  silent_bin_fraction     share of mixture bins below the silence threshold.
  out_energy_in_silent    |S|^2 in those bins / |S|^2 everywhere. Energy the
                          mask CANNOT have produced. This is the fabrication
                          measure, and it is the number the hypothesis rests on.
  residual_share_in_silent  |R|^2 in silent bins / |R|^2 everywhere.

SILENCE IS RELATIVE TO THE CLIP, not absolute: a bin counts as silent when
|X| is below `--silent-percentile` of that clip's own |X| distribution. An
absolute floor would call a quiet recording entirely silent.

WHAT THIS CANNOT SETTLE. It describes a model TRAINED WITH R. A large residual
share is not evidence that R is harmful -- the mask learned to lean on it, and
removing it may break the scaling the mask relies on. Only a trained-without-R
arm answers that. Pair this with the inference ablation
(`make_estimates.py --residual-scale 0`) and read both as "is R implicated",
never as "R is bad".
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from src.estimates.runner import read_trials, git_commit      # noqa: E402
from src.run_log import timed                                  # noqa: E402
from train import SPLIT_MANIFESTS, build_model                 # noqa: E402


def read_mono(path):
    audio, _ = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1)


def load_model(checkpoint_path, device):
    """Same split as make_estimates.build_extractor -- built from the CHECKPOINT's
    config so the weights always fit."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    model.estimator.capture_parts = True
    return model, ckpt


def trial_row(model, mixture, enrollment, device, silent_percentile):
    with torch.no_grad():
        model(torch.from_numpy(mixture).unsqueeze(0).to(device),
              torch.from_numpy(enrollment).unsqueeze(0).to(device))
    parts = model.estimator.last_parts
    if parts["residual"] is None:
        raise SystemExit("this checkpoint was built with residual_branch: false "
                         "-- there is no R to measure.")

    masked = parts["masked"][0]                  # (F, T) complex
    residual = parts["residual"][0]
    mix = parts["mixture"][0]
    out = masked + residual

    e_out = float(out.abs().pow(2).sum())
    e_res = float(residual.abs().pow(2).sum())
    e_masked = float(masked.abs().pow(2).sum())
    # Re<R, S> / |S|^2 -- the share of the output R is responsible for once
    # cancellation against the masked path is accounted for.
    projection = float((residual.conj() * out).real.sum()) / max(e_out, 1e-20)

    mix_mag = mix.abs()
    threshold = torch.quantile(mix_mag.flatten().float(), silent_percentile)
    silent = mix_mag <= threshold

    e_out_silent = float(out.abs().pow(2)[silent].sum())
    e_res_silent = float(residual.abs().pow(2)[silent].sum())
    e_masked_silent = float(masked.abs().pow(2)[silent].sum())

    return {
        "residual_energy_share": e_res / max(e_out, 1e-20),
        "masked_energy_share": e_masked / max(e_out, 1e-20),
        "residual_projection": projection,
        "silent_bin_fraction": float(silent.float().mean()),
        "out_energy_in_silent": e_out_silent / max(e_out, 1e-20),
        "residual_share_in_silent": e_res_silent / max(e_res, 1e-20),
        # Sanity: the mask cannot create energy where the mixture has none, so
        # this should be tiny. If it is not, the silence threshold is too loose.
        "masked_energy_in_silent_bins": e_masked_silent / max(e_out, 1e-20),
        "residual_rms_db": 10 * np.log10(max(e_res, 1e-20)),
        "output_rms_db": 10 * np.log10(max(e_out, 1e-20)),
    }


def correlate(rows, invented_json, key):
    """Pearson r between a residual measure and per-trial invented-word count.

    Same shape as D14 step 0 (leakage vs WER) and D15 step 0. A residual measure
    that does not track invention is not the mechanism, and saying so cheaply is
    the point of this function.
    """
    if not invented_json:
        return None
    data = json.load(open(invented_json))
    system = "ours (baseline)"
    if system not in data:
        return {"error": f"{system!r} not in {invented_json}; "
                         f"have {sorted(data)}"}
    invented = {t["trial_id"]: t["wrong_content_words"] - t["wrong_from_interferer"]
                for t in data[system]}
    pairs = [(r[key], invented[r["trial_id"]])
             for r in rows if r["trial_id"] in invented]
    if len(pairs) < 3:
        return {"error": f"only {len(pairs)} trials matched"}
    x = np.array([p[0] for p in pairs], dtype=float)
    y = np.array([p[1] for p in pairs], dtype=float)
    if x.std() == 0 or y.std() == 0:
        return {"n": len(pairs), "pearson_r": None, "note": "zero variance"}
    return {"n": len(pairs), "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "mean_invented": float(y.mean()), "measure": key}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--condition", default="both")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--silent-percentile", type=float, default=0.10,
                    help="a mixture bin counts as SILENT when its magnitude is "
                         "below this quantile of the clip's own |X|. Relative, "
                         "not absolute -- an absolute floor would call a quiet "
                         "recording entirely silent.")
    ap.add_argument("--invented-from", default=None,
                    help="a per_trial.json from analyse_leakage_share.py, used "
                         "to correlate residual share against invented words.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(args.checkpoint, device)
    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    trials = read_trials(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        audio_root=Path(args.data_root) / "rendered" / val_audio,
        limit=args.limit, condition=args.condition)

    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-residual-share")
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []
    with timed("scripts/diagnose_residual.py",
               scope=lambda: f"{len(rows)} trials, {args.split}",
               rate=lambda: f"{device.type}, whole-clip"):
        for i, trial in enumerate(trials, 1):
            row = trial_row(model,
                            read_mono(trial.directory / "mixture.wav"),
                            read_mono(trial.directory / "enrollment.wav"),
                            device, args.silent_percentile)
            row["trial_id"] = trial.trial_id
            rows.append(row)
            if i % 25 == 0 or i == len(trials):
                print(f"  {i}/{len(trials)}", flush=True)

    keys = [k for k in rows[0] if k != "trial_id"]
    summary = {
        "n_trials": len(rows),
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "split": args.split,
        "condition": args.condition,
        "silent_percentile": args.silent_percentile,
        "seed": seed,
        "git_commit": git_commit(),
        "date": date.today().isoformat(),
        "mean": {k: float(np.mean([r[k] for r in rows])) for k in keys},
        "median": {k: float(np.median([r[k] for r in rows])) for k in keys},
    }
    for measure in ("residual_energy_share", "out_energy_in_silent",
                    "residual_projection"):
        summary.setdefault("correlation_with_invented_words", {})[measure] = \
            correlate(rows, args.invented_from, measure)

    json.dump(rows, open(out_root / "per_trial.json", "w"), indent=1)
    json.dump(summary, open(out_root / "summary.json", "w"), indent=1)

    print(f"\n  wrote {out_root}")
    print(f"\n  {'measure':32s} {'mean':>10s} {'median':>10s}")
    for k in keys:
        print(f"  {k:32s} {summary['mean'][k]:10.4f} {summary['median'][k]:10.4f}")
    corr = summary.get("correlation_with_invented_words") or {}
    for measure, c in corr.items():
        if c and c.get("pearson_r") is not None:
            print(f"\n  r({measure}, invented words) = {c['pearson_r']:+.3f}  "
                  f"n={c['n']}")


if __name__ == "__main__":
    main()
