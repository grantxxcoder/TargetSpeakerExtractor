"""Write estimate.wav for every trial in a split, using one of our checkpoints.

    python scripts/make_estimates.py --split sir0 --checkpoint models/model_sir0_cuecontext-wer-e21.pt --condition both   # the 103 scored trials

The walk over trials and the audio conventions live in src/estimates/runner.py, shared with scripts/make_estimates_wesep.py, so the two systems differ only by their model.
The model is built from the checkpoint's own config, so the weights always fit, and it is built once and reused for every trial.
"""

import argparse
import sys
from datetime import date
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimates.runner import read_trials, write_estimates
from src.models.bsrnn import extract
from src.run_log import timed
from train import build_model, build_speaker_encoder, speaker_embedding

# Each split a system is scored on, and the name of its manifest and audio folder.
EVAL_SPLITS = {
    "sir0": "sir0_val",  # validation, where checkpoints were selected
    "sir0pub": "eval_public",  # public test split (decisions-m3.md 2026-09-24)
    "sir0priv": "eval_private",  # scored once, after selection (decisions-m4.md 2026-10-07)
    "ami": "ami_eval",  # AMI transfer set
}


# this loads a checkpoint and returns a function that extracts one whole clip with it, plus the checkpoint itself
def build_extractor(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(checkpoint["config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    # The speaker encoder is built from the checkpoint's config too, so evaluation embeds a speaker exactly as training did. It is None for the baseline.
    encoder = build_speaker_encoder(checkpoint["config"], device)

    def run(mixture, enrollment, sample_rate):
        with torch.no_grad():
            enrollment_tensor = torch.from_numpy(enrollment).unsqueeze(0).to(device)
            mixture_tensor = torch.from_numpy(mixture).unsqueeze(0).to(device)
            # One forward pass per clip, so the encoder runs once per trial and costs nothing per frame.
            estimate = extract(model, mixture_tensor, enrollment_tensor, enrol_embedding=speaker_embedding(encoder, enrollment_tensor))
        return estimate.squeeze(0).cpu().numpy()
    return run, checkpoint


# this is where rendering starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=sorted(EVAL_SPLITS))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", default=None, help="default experiments/results/<today>-est-<split>")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--condition", default=None, help="render only this condition, for example 'both'. Use the same choice for every system being compared")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  checkpoint: {args.checkpoint}")
    run, checkpoint = build_extractor(args.checkpoint, device)
    seed = int(checkpoint["config"]["seed"])
    torch.manual_seed(seed)

    manifest = EVAL_SPLITS[args.split]
    trials = read_trials(Path(args.manifest_dir) / f"{manifest}.csv", Path(args.data_root) / "rendered" / manifest, None, args.condition)
    out_root = Path(args.out or f"experiments/results/{date.today().isoformat()}-est-{args.split}")
    provenance = {
        "script": "scripts/make_estimates.py",
        "system": "ours-bsrnn",
        "seed": seed,
        "split": args.split,
        "manifest": manifest,
        "condition": args.condition,
        "checkpoint": {"path": str(args.checkpoint), "epoch": checkpoint.get("epoch"), "best_val": checkpoint.get("best_val"), "seed": checkpoint.get("seed")},
        "device": device.type,
    }
    with timed("scripts/make_estimates.py", scope=lambda: f"{len(trials)} trials rendered, {args.split}", rate=lambda: f"{device.type}, whole-clip"):
        write_estimates(run, trials, out_root, int(checkpoint["config"]["data"]["sample_rate"]), provenance)


if __name__ == "__main__":
    main()
