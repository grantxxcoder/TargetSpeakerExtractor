"""Write estimate.wav for every trial in a split, using OUR BSRNN checkpoint.

    python scripts/make_estimates.py --split sir0 --checkpoint models/model_sir0_cuecontext-wer-e21.pt \
        --config experiments/configs/bsrnn_cue_context.yaml --condition both   # the 103 scored trials

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
from train import (SPLIT_MANIFESTS, build_context_encoder,  # noqa: E402
                   build_model, context_kwargs)


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
                       if ".git" not in q.parts)
        raise SystemExit(f"no checkpoint at {given!r}.\nCheckpoints on disk:\n  "
                         + "\n  ".join(found or ["(none)"]))
    cand = Path("models") / f"model_{split}.pt"
    if cand.exists():
        return cand
    raise SystemExit(f"no checkpoint for --split {split}: looked for {cand}")


def build_extractor(checkpoint_path, config, device):
    """Load the checkpoint and return (extractor, checkpoint dict, drift dict).

    The model is built from the CHECKPOINT's config so the weights always fit;
    the current config may carry loss keys it predates. Same split as
    scripts/derive_w_g.py.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["config"])
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()

    # ITEM 1c. Built from the CHECKPOINT's config, like the model itself, so an
    # eval cannot silently use different encoder settings from the run it is
    # scoring. None for the baseline, and context_kwargs() is then a no-op.
    encoder = build_context_encoder(ckpt["config"], device)

    # Report model-config drift rather than refusing: it does not stop the
    # weights loading, but it does change what they MEAN.
    drift = {k: (ckpt["config"].get("model", {}).get(k), v)
             for k, v in config.get("model", {}).items()
             if ckpt["config"].get("model", {}).get(k) != v}

    def extract(mixture, enrollment, sample_rate):    # noqa: ARG001
        with torch.no_grad():
            enrol = torch.from_numpy(enrollment).unsqueeze(0).to(device)
            # ONE forward per clip here -- runner.py is whole-clip, not chunked --
            # so the encoder runs once per trial and costs nothing per frame.
            est = model(torch.from_numpy(mixture).unsqueeze(0).to(device), enrol,
                        **context_kwargs(encoder, enrol))
        return est.squeeze(0).cpu().numpy()

    return extract, ckpt, drift


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=sorted(SPLIT_MANIFESTS))
    ap.add_argument("--checkpoint", default=None,
                    help="default: models/model_<split>.pt")
    ap.add_argument("--config", default="experiments/configs/bsrnn_cue_context.yaml")
    ap.add_argument("--out", default=None,
                    help="default experiments/results/<today>-est-<split>")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--data-root", default="data")   # audio lives under rendered/
    ap.add_argument("--condition", default=None,
                    help="render only this condition, e.g. 'both', or a comma-separated "
                         "list. Default: every "
                         "trial in the manifest, which is what earlier runs did. "
                         "Whatever you choose, choose the SAME for every system "
                         "being compared -- a system rendered on a different "
                         "subset is not comparable.")
    ap.add_argument("--limit", type=int, default=None, help="first N trials only")
    ap.add_argument("--force", action="store_true",
                    help="re-render every trial even if estimate.wav is already "
                         "there. OFF by default: a pass resumes, keeping the "
                         "estimates a previous interrupted run completed. Use "
                         "this when the files on disk are known bad -- a "
                         "directory written by a DIFFERENT run is refused "
                         "outright rather than resumed, so --force is not "
                         "needed for that.")
    args = ap.parse_args()

    config = yaml.safe_load(open(args.config))
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(args.checkpoint, args.split)
    print(f"  checkpoint: {checkpoint}")
    extract, ckpt, drift = build_extractor(checkpoint, config, device)
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
    written, reused = 0, 0

    # The run_times.md row must name what this pass actually COMPUTED. A resumed
    # pass that rendered 200 of 1,421 trials in 20 minutes would otherwise log a
    # 1,421-trial rate that is off by a factor of seven, and that row is what the
    # next session plans against.
    with timed("scripts/make_estimates.py",
               scope=lambda: (f"{written} trials rendered"
                              + (f" (+{reused} reused)" if reused else "")
                              + f", {args.split}"),
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
                "split": args.split,
                "manifest": val_manifest,
                "condition": args.condition,
                "checkpoint": {"path": str(checkpoint), "epoch": ckpt.get("epoch"),
                               "best_val": ckpt.get("best_val"),
                               "seed": ckpt.get("seed")},
                "model_config_drift": {k: list(v) for k, v in drift.items()} or None,
                "device": device.type,
            },
            force=args.force,
        )
        written, reused = meta["n_written"], meta["n_reused"]


if __name__ == "__main__":
    main()
