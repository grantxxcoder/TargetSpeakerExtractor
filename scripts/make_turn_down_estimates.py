"""Writes estimate folders of e21 with the turn-down head, or with its oracle, on the 103 two-speaker validation trials, for scripts/evaluate.py.

    ../tse_venv/bin/python scripts/make_turn_down_estimates.py --head models/turn_down_head.pt --out experiments/results/<date>-est-e21-turn-down
    ../tse_venv/bin/python scripts/make_turn_down_estimates.py --oracle --out experiments/results/<date>-est-e21-turn-down-oracle

Whole clips, one forward pass each, no gradients, as scripts/make_estimates.py runs every extractor.
--oracle applies the head's training target instead of the head: each band of e21's estimate brought down to the clean target's energy in target-active frames.
That is the ceiling for this head, by the same rule it was trained on (decisions-pending.md 2026-10-29), and it differs from D1 of 2026-10-28, which used Silero labels and acted only in overlap.
Claude Code wrote this script on 2026-10-29.
"""

import argparse
import csv
import subprocess
import sys
from datetime import date
from pathlib import Path

import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import build_speaker_encoder, speaker_embedding
from scripts.train_turn_down import load_frozen_base
from src.models.bsrnn import extract_spectrogram
from src.models.stft import stft_forward, stft_inverse
from src.models.turn_down_head import apply_band_gains, build_turn_down_head, oracle_gain_db, turn_down_logits
from src.run_log import timed

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_RATE = 16000


# this reads a wav file as a (1, samples) float32 tensor
def read_clip(path):
    waveform, sample_rate = sf.read(str(path), dtype="float32")
    if sample_rate != SAMPLE_RATE:
        raise SystemExit(f"{path} is at {sample_rate} Hz, not {SAMPLE_RATE}.")
    return torch.from_numpy(waveform)[None]


# this reads the two-speaker validation trials in manifest order
def validation_trials():
    trial_ids = []
    with open(REPO_ROOT / "data/manifests/sir0_val.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] == "both":
                trial_ids.append(row["trial_id"])
    return trial_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--head", default=None, help="a head checkpoint from scripts/train_turn_down.py")
    parser.add_argument("--oracle", action="store_true", help="apply the oracle gain instead of a head")
    parser.add_argument("--config", default="experiments/configs/turn_down_head.yaml", help="for the oracle's settings and e21's weights when --oracle is used")
    parser.add_argument("--out", required=True)
    arguments = parser.parse_args()
    if arguments.oracle == (arguments.head is not None):
        raise SystemExit("Give exactly one of --head or --oracle.")

    device = torch.device("cpu")
    if arguments.head:
        head_record = torch.load(arguments.head, map_location=device, weights_only=False)
        config = head_record["config"]
    else:
        config = yaml.safe_load((REPO_ROOT / arguments.config).read_text())
    base_model, base_config = load_frozen_base(REPO_ROOT / config["base"]["weights"], device)
    speaker_encoder = build_speaker_encoder(base_config, device)
    head = None
    if arguments.head:
        head = build_turn_down_head(len(base_model.band_widths), int(base_config["model"]["separator"]["feature_dim"]), int(config["head"]["hidden_dim"]))
        head.load_state_dict(head_record["head"])
        head.eval()

    out_directory = REPO_ROOT / arguments.out
    out_directory.mkdir(parents=True, exist_ok=True)
    trial_ids = validation_trials()
    for count, trial_id in enumerate(trial_ids, 1):
        trial_directory = REPO_ROOT / "data/rendered/sir0_val" / trial_id
        mixture = read_clip(trial_directory / "mixture.wav")
        enrollment = read_clip(trial_directory / "enrollment.wav")
        with torch.no_grad():
            estimate_spectrogram, features, mixture_spectrogram = extract_spectrogram(base_model, mixture, enrollment, speaker_embedding(speaker_encoder, enrollment))
            if head is not None:
                gains = torch.sigmoid(turn_down_logits(head, features, estimate_spectrogram, mixture_spectrogram, base_model.band_widths))
            else:
                target_spectrogram = stft_forward(base_model.stft, read_clip(trial_directory / "target.wav"))
                gain_db = oracle_gain_db(estimate_spectrogram, target_spectrogram, torch.tensor([False]), base_model.band_widths, float(config["target"]["activity_floor_db"]), float(config["target"]["gain_floor_db"]))
                gains = 10.0 ** (gain_db / 20.0)
            estimate = stft_inverse(base_model.stft, apply_band_gains(estimate_spectrogram, gains, base_model.band_widths), mixture.shape[-1])[0].numpy()
        (out_directory / trial_id).mkdir(exist_ok=True)
        sf.write(str(out_directory / trial_id / "estimate.wav"), estimate, SAMPLE_RATE, subtype="FLOAT")
        if count % 25 == 0:
            print(f"  {count}/{len(trial_ids)} trials", flush=True)

    system = "e21 + turn-down oracle (clean target, forward-only)"
    if head is not None:
        system = f"e21 + turn-down head {arguments.head} (epoch {head_record['epoch']})"
    meta = {"date": date.today().isoformat(), "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip(), "script": "scripts/make_turn_down_estimates.py", "system": system, "base_weights": config["base"]["weights"], "seed": int(config["seed"]), "split": "sir0", "manifest": "sir0_val", "condition": "both", "n_trials": len(trial_ids), "target": config["target"], "sample_rate": SAMPLE_RATE, "audio": "estimate.wav, float32, unnormalised, whole clip"}
    with open(out_directory / "meta.yaml", "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)
    print(f"wrote {len(trial_ids)} estimates to {arguments.out}")


if __name__ == "__main__":
    with timed("scripts/make_turn_down_estimates.py", lambda: " ".join(sys.argv[1:])):
        main()
