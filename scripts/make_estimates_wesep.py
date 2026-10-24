"""Write estimate.wav for every trial in a split, using the pretrained WeSep baseline. Run it under the WeSep virtualenv, not ours:

    ../wesep_venv/bin/python scripts/make_estimates_wesep.py --split sir0 --condition both --pretrain ../wesep_pretrained/tfmap_context_causal_100 --out experiments/results/<date>-est-wesep-tfmap-causal

It is a separate script because the two systems cannot share an interpreter: ours pins torch 2.13 and numpy 2.5.2, and WeSep needs torch 2.7.1 and numpy 1.26.4. They meet in src/estimates/runner.py.
Borrowed, not ours: the REAL-TSE Challenge baseline toolkit (Wang et al., Interspeech 2024, "WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction"), run on our data with our metric. No number from it is comparable to a published REAL-TSE result.
The checkpoint is out of domain on our trials: it was trained on clean, anechoic Libri2Mix with no augmentation, against the dry source, while our trials are reverberant and noisy and scored against the reverberant target.
So a worse score here is not evidence our model is better. Its value is a system we did not build or tune.
"""

import argparse
import hashlib
import sys
from datetime import date
from pathlib import Path

import torch
import wesep
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.estimates.runner import read_trials, write_estimates
from src.run_log import timed

SAMPLE_RATE = 16000
# Every WeSep estimate was rendered on the CPU (docs/run_times.md).
DEVICE = "cpu"
MANIFEST_DIR = Path("data/manifests")
RENDERED_DIR = Path("data/rendered")

# The same table as EVAL_SPLITS in scripts/make_estimates.py. It is copied rather than imported, because that script needs torch 2.13 and our model code, which the WeSep virtualenv does not have.
EVAL_SPLITS = {
    "sir0": "sir0_val",
    "sir0pub": "eval_public",
    "sir0priv": "eval_private",
    "ami": "ami_eval",
}


# this hashes the weights, because a download link can be re-uploaded silently and the hash is all that ties a result to these exact parameters
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# this reads the facts about the borrowed model from its own config and weights, so the out-of-domain caveat travels with every result
def describe_checkpoint(pretrain_dir):
    pretrain_dir = Path(pretrain_dir)
    config_path = pretrain_dir / "config.yaml"
    weights_path = pretrain_dir / "avg_model.pt"
    for path in (config_path, weights_path):
        if not path.exists():
            raise SystemExit(f"Missing {path}. --pretrain must be a folder holding avg_model.pt and config.yaml, for example ../wesep_pretrained/tfmap_context_causal_100.")
    config = yaml.safe_load(open(config_path))
    separator = config.get("model_args", {}).get("tse_model", {}).get("separator", {})
    dataset = config.get("dataset_args", {})
    state = torch.load(weights_path, map_location="cpu", weights_only=False)
    weights = state["models"][0] if isinstance(state.get("models"), list) else state
    parameter_count = 0
    for value in weights.values():
        if hasattr(value, "numel"):
            parameter_count += value.numel()
    return {
        "name": pretrain_dir.name,
        "path": str(pretrain_dir),
        "avg_model_sha256": sha256(weights_path),
        "parameters_millions": round(parameter_count / 1e6, 2),
        "causal": separator.get("causal"),
        "native_resample_rate": dataset.get("resample_rate"),
        "trained_on": config.get("train_data"),
        "training_loss": config.get("loss"),
        "noise_prob": dataset.get("noise_prob"),
        "reverb_enroll_prob": dataset.get("reverb_enroll_prob"),
        "num_avg": config.get("num_avg"),
        "seed": config.get("seed"),
    }


# this loads WeSep and wraps it in the runner's contract: whole mixture and enrolment in, the estimate out
def build_extractor(pretrain_dir):
    """WeSep's own VAD is switched off, because our speech gate does that job for every listener, and a second gate inside a system under test would mute clips before ours saw them.
    Its output normalisation is switched off too, because our estimates are written unnormalised and normalising would hand WeSep a level correction our model does not get.
    """
    model = wesep.load_model_local(str(pretrain_dir))
    model.set_resample_rate(SAMPLE_RATE)
    model.set_vad(False)
    model.set_device(DEVICE)
    model.set_output_norm(False)

    def run(mixture, enrollment, sample_rate):
        estimate = model.extract_speech_from_pcm(torch.from_numpy(mixture).unsqueeze(0), sample_rate, torch.from_numpy(enrollment).unsqueeze(0), sample_rate)
        if estimate is None:
            return None
        return estimate.reshape(-1).cpu().numpy()
    return run


# this is where rendering starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True, choices=sorted(EVAL_SPLITS))
    parser.add_argument("--pretrain", required=True, help="a WeSep checkpoint folder holding avg_model.pt and config.yaml")
    parser.add_argument("--out", default=None, help="default experiments/results/<today>-est-wesep-<checkpoint>")
    parser.add_argument("--condition", default=None, help="render only this condition. It must match what scripts/make_estimates.py was run with")
    args = parser.parse_args()

    checkpoint = describe_checkpoint(args.pretrain)
    native = checkpoint["native_resample_rate"]
    if native is not None and int(native) != SAMPLE_RATE:
        raise SystemExit(f"{checkpoint['name']} was trained at {native} Hz but our audio is {SAMPLE_RATE} Hz, and a resampler must not silently join the instrument.")
    print(f"  checkpoint: {checkpoint['name']}  {checkpoint['parameters_millions']} M params  causal={checkpoint['causal']}")
    print(f"  trained on: {checkpoint['trained_on']}  loss={checkpoint['training_loss']}")
    print("  OUT OF DOMAIN on our trials -- see this script's docstring before quoting anything from this run.")

    run = build_extractor(args.pretrain)
    manifest = EVAL_SPLITS[args.split]
    trials = read_trials(MANIFEST_DIR / f"{manifest}.csv", RENDERED_DIR / manifest, args.condition)
    out_root = Path(args.out or f"experiments/results/{date.today().isoformat()}-est-wesep-{checkpoint['name']}")
    provenance = {
        "script": "scripts/make_estimates_wesep.py",
        "system": "wesep-real-tse-pretrained",
        "borrowed_from": "REAL-TSE/wesep-real-tse; Wang et al., Interspeech 2024. Their checkpoint, our data, our metric. NOT comparable to published REAL-TSE numbers.",
        "out_of_domain": "trained on clean anechoic Libri2Mix with no augmentation, against the dry source; our trials are reverberant and noisy and are scored against the reverberant target (A1). A worse score here is not evidence our model is better.",
        "split": args.split,
        "manifest": manifest,
        "condition": args.condition,
        "checkpoint": checkpoint,
        "device": DEVICE,
        "wesep_vad": False,
        "wesep_output_norm": False,
        # str(), because torch.__version__ is a str subclass that yaml.safe_dump refuses.
        "torch": str(torch.__version__),
    }
    with timed("scripts/make_estimates_wesep.py", scope=lambda: f"{len(trials)} trials rendered, {args.split}, {checkpoint['name']}", rate=lambda: f"{DEVICE}, whole-clip"):
        write_estimates(run, trials, out_root, SAMPLE_RATE, provenance)


if __name__ == "__main__":
    main()
