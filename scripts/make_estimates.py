"""Write estimate.wav for every trial in a split, using OUR BSRNN checkpoint.

    python scripts/make_estimates.py --split smoke --checkpoint kaggle_out/models/model_sir0.pt
    python scripts/make_estimates.py --split sir0  --out experiments/results/2026-09-03-est-sir0
    python scripts/make_estimates.py --split sir0  --condition both   # the 103 scored trials only

This is the OUR-MODEL front-end. The walk over trials, the audio conventions
and the provenance file live in src/estimates/runner.py, shared with
scripts/make_estimates_wesep.py so that a second system differs from this one
by its model and by nothing else. Anything that would change the comparison
belongs in the runner; anything specific to loading a torch checkpoint of ours
belongs here.

Batch counterpart to scripts/pass_a_test_case_through.py, which does ONE trial
per invocation and reloads the model each time. At 500 trials that is 500
process launches and 500 model loads; here the model is built once.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimates.runner import read_trials, write_estimates  # noqa: E402
from src.run_log import timed  # noqa: E402
from train import SPLIT_MANIFESTS, build_model  # noqa: E402


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


def build_extractor(checkpoint_path, config, device, mask_floor=0.0,
                    mask_hysteresis=None, residual_scale=1.0):
    """Load the checkpoint and return (extractor, checkpoint dict, drift dict).

    The model is built from the CHECKPOINT's config so the weights always fit;
    the current config may carry loss keys it predates. Same split as
    scripts/derive_w_g.py.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    # SET AFTER LOADING, deliberately. The floor is an inference-time clamp on a
    # model trained without it, so it must not be part of the architecture the
    # state dict is matched against -- that keeps every existing checkpoint
    # loading with strict=True and keeps floor 0.0 bit-identical to the
    # published result.
    if mask_floor:
        model.estimator.mask_floor = float(mask_floor)
    if mask_hysteresis:
        model.estimator.mask_hysteresis = tuple(mask_hysteresis)
    # Same argument as the floor: set after loading, so residual_scale 1.0 is
    # bit-identical to every published result and no checkpoint changes shape.
    if residual_scale != 1.0:
        model.estimator.residual_scale = float(residual_scale)
    model.to(device).eval()

    # Report model-config drift rather than refusing: it does not stop the
    # weights loading, but it does change what they MEAN.
    drift = {k: (ckpt["config"].get("model", {}).get(k), v)
             for k, v in config.get("model", {}).items()
             if ckpt["config"].get("model", {}).get(k) != v}

    def extract(mixture, enrollment, sample_rate):    # noqa: ARG001
        with torch.no_grad():
            est = model(torch.from_numpy(mixture).unsqueeze(0).to(device),
                        torch.from_numpy(enrollment).unsqueeze(0).to(device))
        return est.squeeze(0).cpu().numpy()

    return extract, ckpt, drift


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
    ap.add_argument("--data-root", default="data")   # audio lives under rendered/
    ap.add_argument("--condition", default=None,
                    help="render only this condition, e.g. 'both'. Default: every "
                         "trial in the manifest, which is what earlier runs did. "
                         "Whatever you choose, choose the SAME for every system "
                         "being compared -- a system rendered on a different "
                         "subset is not comparable.")
    ap.add_argument("--limit", type=int, default=None, help="first N trials only")
    ap.add_argument("--mask-floor", type=float, default=0.0,
                    help="smallest magnitude the mask may take, applied at "
                         "INFERENCE on an already-trained model. 0.0 = off, "
                         "which is every result before 2026-09-12. See "
                         "Estimator.mask_floor.")
    ap.add_argument("--mask-hysteresis", default=None, metavar="HI,LO,DOWN",
                    help="region-grow the mask across frequency at inference, "
                         "e.g. 1.5,0.5,0.0. HI and LO are multiples of the "
                         "frame's own mean gain; DOWN is what a bin that fails "
                         "to survive is multiplied by. MEASURED 2026-09-12: "
                         "1.5,0.5,0.0 raises the mask's frequency variation "
                         "0.0238 -> 0.0596 against the ideal mask's 0.1725, and "
                         "drops the share explained by one number per frame from "
                         "82.5 % to 2.0 %. Milder settings barely move either "
                         "and cannot test the hypothesis.")
    ap.add_argument("--residual-scale", type=float, default=1.0,
                    help="scale on the additive residual branch R, applied at "
                         "INFERENCE. 1.0 = off (the trained model unchanged, "
                         "every result before 2026-09-13); 0.0 deletes R so the "
                         "output is the masked mixture alone. R is the only "
                         "path that can place energy in a bin the microphone "
                         "never recorded, which is what an invented word IS. "
                         "See Estimator.residual_scale for what this cannot "
                         "settle.")
    args = ap.parse_args()
    if args.mask_hysteresis:
        args.mask_hysteresis = [float(v) for v in args.mask_hysteresis.split(",")]
        assert len(args.mask_hysteresis) == 3, "--mask-hysteresis wants HI,LO,DOWN"
        assert args.mask_hysteresis[1] < args.mask_hysteresis[0], \
            "LO must be below HI or there is no hysteresis, only a threshold"

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(args.checkpoint, args.split)
    print(f"  checkpoint: {checkpoint}")
    extract, ckpt, drift = build_extractor(checkpoint, config, device,
                                          mask_floor=args.mask_floor,
                                          mask_hysteresis=args.mask_hysteresis,
                                          residual_scale=args.residual_scale)
    if drift:
        print(f"  WARNING: model config drift, output does not reflect training: {drift}")

    val_manifest, val_audio = SPLIT_MANIFESTS[args.split][1]
    trials = read_trials(
        manifest_csv=Path(args.manifest_dir) / f"{val_manifest}.csv",
        audio_root=Path(args.data_root) / "rendered" / val_audio,
        limit=args.limit,
        condition=args.condition,
    )
    out_root = Path(args.out or
                    f"experiments/results/{date.today().isoformat()}-est-{args.split}")
    written = 0

    with timed("scripts/make_estimates.py",
               scope=lambda: f"{written} trials, {args.split}",
               rate=lambda: f"{device.type}, whole-clip"):
        meta = write_estimates(
            extract=extract,
            trials=trials,
            out_root=out_root,
            sample_rate=int(config["data"]["sample_rate"]),
            provenance={
                "script": "scripts/make_estimates.py",
                "system": "ours-bsrnn",
                "seed": seed,
                "config": args.config,
                # A floored run is NOT the same system as an unfloored one, and
                # the directory name may not say so. Record it.
                "mask_floor": float(args.mask_floor),
                "mask_hysteresis": list(args.mask_hysteresis) if args.mask_hysteresis else None,
                "residual_scale": float(args.residual_scale),
                "split": args.split,
                "manifest": val_manifest,
                "condition": args.condition,
                "checkpoint": {"path": str(checkpoint), "epoch": ckpt.get("epoch"),
                               "best_val": ckpt.get("best_val"),
                               "seed": ckpt.get("seed")},
                "model_config_drift": {k: list(v) for k, v in drift.items()} or None,
                "device": device.type,
            },
        )
        written = meta["n_trials"]


if __name__ == "__main__":
    main()
