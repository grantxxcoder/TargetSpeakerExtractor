"""Pass ONE validation trial through a checkpoint and write what it sounds like.

    ../tse_venv/bin/python scripts/pass_a_test_case_through.py --split sir0 --index 0

Listening check, not a measurement. Writes estimate.wav + a meta.yaml, and
prints the loss terms for that crop beside the pass-through anchor.

Only the estimate is written: mixture/target/enrollment already exist under
data/rendered/<split>/<trial_id>/ and the meta records that path.

WHOLE CLIP, ONE FORWARD PASS. The model is causal, so appending later audio
cannot change earlier output -- measured 2026-08-24, interior diff 1.68e-08.
Stitching 4 s chunks is worse, not merely unnecessary: each seam reinjects the
384-sample overlap-add tail (max diff 4.37e-03, rel L2 1.04e-02).

CAUSAL IS NOT CONTEXT-FREE. A mid-clip crop starts LSTM/cLN state COLD; inside a
full pass it is warm. Measured 5.60e-03 max, rel L2 3.06e-01. So the AUDIO here
is the warm version (what deployment emits) while the LOSS is the cold-start
crop (the only number comparable to history.csv). Safe by measurement: the two
score -2.4197 vs -2.4150 over the same window.

build_loss_fn/build_model/SPLIT_MANIFESTS are IMPORTED from train.py, never
re-declared -- a copy is how the two silently diverge.

Reading a number off this:
  1. The single-crop `total` is not on an epoch total's scale -- one crop is
     either present or absent, so the other half contributes 0, not its mean.
  2. The anchor is computed on THIS crop, not the 300-crop median in
     experiments/results/2026-08-20-loss-anchor. Single-crop anchors swing by
     many dB with SIR and target activity.
  3. Audio is float32, UNNORMALISED -- normalising would hide the very gain
     error L_gain exists to catch. Peak is reported and may exceed 1.0.
"""

import argparse
import hashlib
import sys
from collections import OrderedDict
from datetime import date
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

# Same line as scripts/train.py: `python scripts/...` puts scripts/ on sys.path,
# not the repo root, so the src.* imports below need this first.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402
from train import SPLIT_MANIFESTS, build_loss_fn, build_model, git_commit  # noqa: E402

# DERIVED from train.py. Was its own literal and had drifted -- `mid` and `sir0`
# were never added here, so the two splits that actually trained could not be
# inspected. Two-valued: manifest name != audio dir for `mid` ("mid_val", "val").
VAL_MANIFESTS = {split: pairs[1] for split, pairs in SPLIT_MANIFESTS.items()}


def pick_index(dataset, index, trial_id):
    """--index or --trial-id -> a row number. Exactly one of the two."""
    if trial_id is not None:
        matches = dataset.manifest_df.index[
            dataset.manifest_df["trial_id"].astype(str) == trial_id].tolist()
        if not matches:
            raise SystemExit(f"trial_id {trial_id!r} is not in this manifest")
        return int(matches[0])
    if not 0 <= index < len(dataset):
        raise SystemExit(f"--index {index} out of range for {len(dataset)} trials")
    return index


def load_checkpoint(model, checkpoint_path, config, device):
    """Load weights, return provenance.

    SHAPE COMPATIBILITY IS DECIDED BY load_state_dict, not by comparing configs.
    A hand-kept list of "shape keys" drifts: `tfmap_scale` (2026-08-26) changes
    behaviour but no tensor shape, and a whole-dict comparison is worse still --
    adding the loss key w_g on 2026-08-27 locked every existing checkpoint out of
    this script. PyTorch is the authority; config drift is reported, not enforced.
    """
    if not checkpoint_path.exists():
        raise SystemExit(f"no checkpoint at {checkpoint_path} -- train first, "
                         f"or pass --checkpoint")
    # weights_only=False: our own checkpoint carries the config dict. Never
    # point this at a file this project did not write.
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = ckpt.get("config") or {}

    def flat(d, prefix=""):
        for k, v in (d or {}).items():
            if isinstance(v, dict):
                yield from flat(v, f"{prefix}{k}.")
            else:
                yield f"{prefix}{k}", v

    a, b = dict(flat(ckpt_config)), dict(flat(config))
    drift = {k: (a.get(k), b.get(k)) for k in set(a) | set(b) if a.get(k) != b.get(k)}
    # `model.*` drift changes what the weights MEAN even when they still load,
    # so it is called out separately from loss/training drift, which does not.
    model_drift = {k: v for k, v in drift.items() if k.startswith("model.")}
    if model_drift:
        print("  WARNING: the model config differs from the one this checkpoint "
              "was trained under, so its output does NOT reflect what it learned:")
        for k, (was, now) in sorted(model_drift.items()):
            print(f"    {k}: trained={was!r}  now={now!r}")
    elif drift:
        print(f"  NOTE: config drift outside `model` (weights unaffected): "
              f"{sorted(drift)}")

    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as e:
        raise SystemExit(f"{checkpoint_path} does not fit the model this config "
                         f"builds:\n{e}")
    return {
        "path": str(checkpoint_path),
        "trained_to_epoch": ckpt.get("epoch"),
        "best_val_total": ckpt.get("best_val"),
        "seed": ckpt.get("seed"),
    }


def levels(x):
    """Peak and RMS in dBFS, for the meta. Silence reports -inf, not a crash."""
    x = x.detach().cpu().flatten()
    peak = float(x.abs().max())
    rms = float(x.pow(2).mean().sqrt())
    to_db = lambda v: (round(20 * float(np.log10(v)), 2) if v > 0 else float("-inf"))
    return {"peak": round(peak, 6), "peak_dbfs": to_db(peak),
            "rms_dbfs": to_db(rms), "clipped": bool(peak > 1.0)}


def write_estimate(out_dir, sample_rate, estimate):
    """Write estimate.wav. FLOAT not PCM_16: the estimate is unnormalised and
    PCM_16 would clip it silently, turning a gain bug into a fake artefact."""
    path = out_dir / "estimate.wav"
    sf.write(str(path), estimate.detach().cpu().flatten().numpy(),
             sample_rate, subtype="FLOAT")
    return path


def fmt(v):
    """Loss terms print as a fixed width, or as `n/a` for the absent half."""
    return "    n/a " if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:8.4f}"


def report(parts, anchor, w, wm, crop_absent, wg=0.0):
    """Loss for this crop beside the do-nothing anchor for the SAME crop.

    L_gain is listed even at w_g = 0. Use scripts/derive_w_g.py for the real
    derivation -- single-crop anchors swing by many dB."""
    branch = "ABSENT (target silent in this crop)" if crop_absent else "PRESENT"
    print(f"\n  crop branch : {branch}")
    print(f"  {'term':<12}{'model':>10}{'pass-through':>16}{'delta':>10}")
    for key in ("L_pres", "L_MR", "L_gain", "L_abs", "total"):
        m, a = parts.get(key), anchor.get(key)
        both = all(x is not None and not np.isnan(x) for x in (m, a))
        print(f"  {key:<12}{fmt(m):>10}{fmt(a):>16}"
              f"{(f'{m - a:+8.4f}' if both else '     n/a'):>10}")

    if not np.isnan(parts["total"]) and not np.isnan(anchor["total"]):
        delta = parts["total"] - anchor["total"]
        verdict = "BETTER than" if delta < 0 else "WORSE than"
        print(f"\n  {verdict} emitting the mixture unchanged, by {abs(delta):.4f}")
    # Floor = w-weighted sum of 10log10(tau_pres) and 10log10(tau_abs), so w
    # moves it (-25.42 shipped). L_gain adds nothing: 0 at perfect reconstruction.
    print(f"  single-crop scale: this branch only, w={w} w_m={wm} w_g={wg}. "
          f"Not comparable to an epoch total.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(VAL_MANIFESTS))
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--index", type=int, default=0,
                       help="row in the val manifest (default 0)")
    group.add_argument("--trial-id", default=None,
                       help="pick by trial_id instead of row number")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--checkpoint", default=None,
                    help="default models/model_<split>.pt")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--results-dir", default=None,
                    help="default experiments/results/<today>-passthrough-<split>")
    ap.add_argument("--crop-only", action="store_true",
                    help="write just the 4 s crop instead of the whole clip "
                         "(smaller, but you hear 22%% of the trial)")
    args = ap.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())

    # Non-negotiable rule: set and log a seed for every run. Matters here even
    # with no training -- it is what fixes the crop offset below.
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)

    val_manifest, val_audio = VAL_MANIFESTS[args.split]
    manifest_csv = Path(args.manifest_dir) / f"{val_manifest}.csv"
    # `split` below is the AUDIO directory, which is val_audio -- not the
    # manifest name. They differ for `mid`. See VAL_MANIFESTS.
    val_split = val_audio

    # random_crop=False pins epoch 0 inside _crop_offset_start, so re-running
    # this script on the same index returns the same 4 s window every time. A
    # listening check you cannot reproduce is worthless.
    dataset = TrialDataset(
        manifest_csv=manifest_csv,
        data_root=Path(args.data_root),
        split=val_split,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed,
        random_crop=False,
    )

    idx = pick_index(dataset, args.index, args.trial_id)
    # [0] because __getitem__ returns a LIST of directions (decisions-m2.md
    # 2026-08-26), and does so even at both_directions=False, which is the
    # default above. [0] is the TARGET direction, which is what a listening
    # check wants. This script predated that change and indexed the list as a
    # dict.
    sample = dataset[idx][0]

    # No DataLoader: one example needs no collation, and unsqueezing here keeps
    # the batch dim visible rather than hidden in a loader with batch_size=1.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mixture = sample["mixture"].unsqueeze(0).to(device)
    target = sample["target"].unsqueeze(0).to(device)
    enrollment = sample["enrollment"].unsqueeze(0).to(device)
    crop_absent = torch.tensor([sample["crop_absent"]], device=device)

    model = build_model(config).to(device)
    ckpt_meta = load_checkpoint(model, Path(args.checkpoint) if args.checkpoint
                                else Path("models") / f"model_{args.split}.pt",
                                config, device)
    model.eval()
    loss_fn = build_loss_fn(config)

    print(f"seed {seed}  config {config_path}  device {device}")
    print(f"{val_split}[{idx}] = {sample['trial_id']}  "
          f"checkpoint {ckpt_meta['path']} @ epoch {ckpt_meta['trained_to_epoch']}")

    with torch.no_grad():
        estimate = model(mixture, enrollment)
        _, parts = loss_fn(target, estimate, mixture, crop_absent)
        # The anchor: substitute the mixture for the model output on this exact
        # crop. Says whether the model did anything at all here, which a raw
        # loss value on one crop cannot.
        _, anchor = loss_fn(target, mixture, mixture, crop_absent)

    report(parts, anchor, loss_fn.w, loss_fn.wm, sample["crop_absent"], loss_fn.wg)

    # Where the three input stems live. They are NOT copied into the results
    # dir -- the meta points at them instead, which is the whole reason this
    # script costs 256 kB per trial rather than ~1.3 MB.
    source_dir = (Path(args.data_root) / "rendered" / val_split
                  / str(sample["trial_id"]))
    # Deterministic given (seed, epoch=0, idx) because random_crop is False.
    # Recomputed via the dataset's own method so it cannot disagree with the
    # audio that was actually read.
    crop_start = dataset._crop_offset_start(
        idx, sf.info(str(source_dir / "mixture.wav")).frames)

    results_dir = Path(args.results_dir) if args.results_dir else (
        Path("experiments/results")
        / f"{date.today().isoformat()}-passthrough-{args.split}" / str(sample["trial_id"]))
    results_dir.mkdir(parents=True, exist_ok=True)

    # The audio to listen to is the WHOLE clip, in one pass. Not the 4 s crop the
    # loss was computed on, and NOT chunked-and-stitched: the model is causal, so
    # one full-length pass is what streaming emits, while concatenating
    # independent chunks would add a 23.4 ms edge artefact at every seam.
    # Re-read through the dataset's own reader so the sample-rate and mono
    # assertions apply here too.
    if args.crop_only:
        audio_out, audio_scope = estimate, "crop"
    else:
        full_mixture = dataset._read_in_wav(source_dir / "mixture.wav").unsqueeze(0).to(device)
        with torch.no_grad():
            audio_out = model(full_mixture, enrollment)
        audio_scope = "full_clip"

    write_estimate(results_dir, dataset.sample_rate, audio_out)

    # Levels for all four signals, but only the estimate on disk. Measuring the
    # inputs costs nothing -- they are already in memory -- and the estimate's
    # level is only meaningful against the target's. Crop-window levels
    # throughout, so the four are comparable to each other and to the loss.
    signal_levels = OrderedDict([
        ("mixture", levels(mixture)),        # what went in
        ("enrollment", levels(enrollment)),  # who to listen for
        ("target", levels(target)),          # the right answer
        ("estimate", levels(estimate)),      # what came out, over the crop
    ])
    if audio_scope == "full_clip":
        signal_levels["estimate_full_clip"] = levels(audio_out)

    # The manifest's own provenance, carried through the way train.py does it --
    # the result is only interpretable against the data build that produced it.
    manifest_meta_path = manifest_csv.with_suffix(".meta.yaml")
    manifest_meta = (yaml.safe_load(manifest_meta_path.read_text())
                     if manifest_meta_path.exists() else {})

    (results_dir / "meta.yaml").write_text(yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/pass_a_test_case_through.py",
        "git_commit": git_commit(),
        "seed": seed,
        "config": str(config_path),
        "config_md5": hashlib.md5(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "checkpoint": ckpt_meta,
        "trial": {
            "split": val_split,
            "manifest": str(manifest_csv),
            "manifest_built_at_commit": manifest_meta.get("git_commit"),
            "index": idx,
            "trial_id": str(sample["trial_id"]),
            # From the CROPPED stem, not the manifest label -- 5.8 % of
            # `both`/`target_only` crops contain no target speech at all.
            "crop_absent": bool(sample["crop_absent"]),
            # The 4 s window of the full stems that the estimate corresponds
            # to. This is what lets the uncopied inputs be lined up by ear.
            "source_dir": str(source_dir),
            "crop_start_sample": crop_start,
            "crop_window_s": [round(crop_start / dataset.sample_rate, 3),
                              round((crop_start + dataset.chunk_frames)
                                    / dataset.sample_rate, 3)],
            "chunk_s": float(config["data"]["chunk_s"]),
            "conditions": {k: (bool(v) if isinstance(v, bool) else v)
                           for k, v in sample["meta"].items()},
        },
        "audio": {
            "sample_rate": int(dataset.sample_rate),
            # The ONLY wav written here. float32, unnormalised -- `clipped` true
            # means peak > 1.0, which is a gain problem in the model, not in the wav.
            "written": "estimate.wav (wav float32, no normalisation)",
            # full_clip: one causal pass over the entire mixture, so it lines
            # up 1:1 with the source stems and needs no stitching. NOTE the loss
            # below is on the COLD-START crop at crop_window_s, not on this
            # signal's warm-state version of that window -- measured 0.005 apart
            # on total, but they are not the same tensor.
            "estimate_scope": audio_scope,
            "estimate_seconds": round(audio_out.shape[-1] / dataset.sample_rate, 3),
            # Not copied. Listen to these in place; the crop window above says
            # which 4 s of them the estimate corresponds to.
            "source_stems_not_copied": str(source_dir),
            "levels_dbfs": dict(signal_levels),
        },
        "loss": {
            # tau was SPLIT into tau_pres/tau_abs on 2026-08-25; this still read
            # loss_fn.tau and died with AttributeError after printing the report
            # but before writing this file. w_g added at the same time.
            "w": loss_fn.w, "w_m": loss_fn.wm, "w_g": loss_fn.wg,
            "tau_pres": loss_fn.tau_pres, "tau_abs": loss_fn.tau_abs,
            "gain_delta_db": loss_fn.gain_delta_db,
            "p": loss_fn.p, "windows_ms": list(loss_fn.windows),
            # Per-branch floors, reached at exact reconstruction only. Two now,
            # because the present and absent branches no longer share a tau.
            "floor_pres": round(10 * float(np.log10(loss_fn.tau_pres)), 4),
            "floor_abs": round(10 * float(np.log10(loss_fn.tau_abs)), 4),
            "model": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                      for k, v in parts.items()},
            "pass_through_anchor": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                    for k, v in anchor.items()},
            "note": "single crop: one branch only, not comparable to an epoch total",
        },
    }, sort_keys=False))

    secs = audio_out.shape[-1] / dataset.sample_rate
    scope = ("whole clip, one causal pass -- no stitching"
             if audio_scope == "full_clip" else "crop only (--crop-only)")
    print(f"\nwrote {results_dir}/  (estimate.wav {secs:.2f} s, meta.yaml)")
    print(f"  estimate  : {scope}")
    print(f"  inputs    : {source_dir}/ (not copied)")
    print(f"  loss above: the {dataset.chunk_s:.0f} s crop at "
          f"{crop_start / dataset.sample_rate:.2f}-"
          f"{(crop_start + dataset.chunk_frames) / dataset.sample_rate:.2f} s")
    for name, lv in signal_levels.items():
        flag = "  <-- CLIPS" if lv["clipped"] else ""
        print(f"  {name:<18} peak {lv['peak_dbfs']:>7.2f} dBFS   "
              f"rms {lv['rms_dbfs']:>7.2f} dBFS{flag}")


if __name__ == "__main__":
    main()
