"""Write estimate.wav for EVERY trial in a split's val manifest.

    python scripts/make_estimates.py --split smoke --checkpoint kaggle_out/models/model_sir0.pt
    python scripts/make_estimates.py --split sir0  --out experiments/results/2026-08-28-est-sir0

Batch counterpart to scripts/pass_a_test_case_through.py, which does ONE trial
per invocation and reloads the model each time. At 500 trials that is 500
process launches and 500 model loads; here the model is built once.

Writes <out>/<trial_id>/estimate.wav and one meta.yaml for the whole run, not
one per trial -- provenance is a property of the pass, not of each file.

WHOLE CLIP, ONE FORWARD PASS, no chunking: the model is causal, so appending
later audio cannot change earlier output (measured 2026-08-24, 1.68e-08), and
stitching chunks reinjects the overlap-add tail at every seam.

Audio is float32 and UNNORMALISED, matching pass_a_test_case_through.py --
normalising would hide the gain error L_gain exists to catch.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import SPLIT_MANIFESTS, build_model, git_commit  # noqa: E402


def resolve_checkpoint(given, split):
    """--checkpoint, or the conventional location for this split.

    Errors list what was tried AND what exists, because the alternative is
    torch.load's FileNotFoundError on a path the caller mistyped.
    """
    if given is not None:
        p = Path(given)
        if p.exists():
            return p
        found = sorted(str(q) for q in Path(".").rglob("model_*.pt")
                       if ".git" not in q.parts and "kaggle_bundle" not in q.parts)
        raise SystemExit(f"no checkpoint at {given!r}.\nCheckpoints on disk:\n  "
                         + "\n  ".join(found or ["(none)"]))
    for cand in (Path("models") / f"model_{split}.pt",
                 Path("kaggle_out/models") / f"model_{split}.pt"):
        if cand.exists():
            return cand
    raise SystemExit(f"no checkpoint for --split {split}: looked in "
                     f"models/ and kaggle_out/models/ for model_{split}.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", default=None,
                    help="default: models/model_<split>.pt, else "
                         "kaggle_out/models/model_<split>.pt")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--out", default=None,
                    help="default experiments/results/<today>-est-<split>")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")   # loader appends "rendered/"
    ap.add_argument("--limit", type=int, default=None, help="first N trials only")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(args.checkpoint, args.split)
    print(f"  checkpoint: {checkpoint}")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    # Model from the CHECKPOINT's config so the weights always fit; the current
    # config may carry loss keys it predates. Same split as scripts/derive_w_g.py.
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # Report model-config drift rather than refusing: it does not stop the
    # weights loading, but it does change what they MEAN.
    drift = {k: (ckpt["config"].get("model", {}).get(k), v)
             for k, v in config.get("model", {}).items()
             if ckpt["config"].get("model", {}).get(k) != v}
    if drift:
        print(f"  WARNING: model config drift, output does not reflect training: {drift}")

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    dataset = TrialDataset(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        data_root=Path(args.data_root),
        split=val_audio,
        chunk_s=config["data"]["chunk_s"],
        sample_rate=config["data"]["sample_rate"],
        seed=seed,
        random_crop=False,
    )
    n = len(dataset) if args.limit is None else min(args.limit, len(dataset))
    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-est-{args.split}")
    out_root.mkdir(parents=True, exist_ok=True)
    sr = int(config["data"]["sample_rate"])
    written = 0

    with timed("scripts/make_estimates.py",
               scope=lambda: f"{written} trials, {args.split}",
               rate=lambda: f"{device.type}, whole-clip"):
        for i in range(n):
            row = dataset.manifest_df.iloc[i]
            trial_id = str(row["trial_id"])
            d = Path(args.data_root) / "rendered" / val_audio / trial_id
            # Whole clip, not the 4 s training crop -- the dataset crops, so read
            # the files directly here.
            mixture, _ = sf.read(str(d / "mixture.wav"), dtype="float32", always_2d=False)
            enrol, _ = sf.read(str(d / "enrollment.wav"), dtype="float32", always_2d=False)
            with torch.no_grad():
                est = model(torch.from_numpy(mixture).unsqueeze(0).to(device),
                            torch.from_numpy(enrol).unsqueeze(0).to(device))
            trial_dir = out_root / trial_id
            trial_dir.mkdir(parents=True, exist_ok=True)
            sf.write(str(trial_dir / "estimate.wav"),
                     est.squeeze(0).cpu().numpy(), sr, subtype="FLOAT")
            written += 1
            if written % 25 == 0 or written == n:
                print(f"  {written}/{n}", flush=True)

    yaml.safe_dump({
        "date": date.today().isoformat(),
        "script": "scripts/make_estimates.py",
        "git_commit": git_commit(),
        "seed": seed,
        "config": args.config,
        "split": args.split,
        "manifest": val_manifest,
        "checkpoint": {"path": str(checkpoint), "epoch": ckpt.get("epoch"),
                       "best_val": ckpt.get("best_val"), "seed": ckpt.get("seed")},
        "model_config_drift": {k: list(v) for k, v in drift.items()} or None,
        "n_trials": written,
        "device": device.type,
        "audio": "estimate.wav, float32, unnormalised, whole clip",
    }, open(out_root / "meta.yaml", "w"), sort_keys=False)
    print(f"\n  wrote {written} estimates -> {out_root}/")


if __name__ == "__main__":
    main()
