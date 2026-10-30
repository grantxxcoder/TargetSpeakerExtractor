"""Trains the turn-down head on a frozen e21 (decisions-pending.md 2026-10-29). Only the head's weights change.

    python scripts/train_turn_down.py --config experiments/configs/turn_down_head.yaml               # on a GPU, never the laptop
    python scripts/train_turn_down.py --config experiments/configs/turn_down_head.yaml --check       # two validation batches, no gradients

Each step runs e21 without gradients to get its estimate, its separator state and the mixture's spectrogram, then trains the head to predict the oracle gain:
each band of e21's estimate brought down to the clean target's energy, in frames where the target is active (src/models/turn_down_head.py).
The data, crops and loaders are e21's own (scripts/train.get_data_loaders on e21's data block), and selection is on the sir0_val loss, so no reported holdout is ever scored.
The best and the last head go to --outdir; the history, config and commit go to experiments/results/<date>-train-turn-down/.
Claude Code wrote this script on 2026-10-29 from the option the author chose.
"""

import argparse
import copy
import csv
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import build_model, build_speaker_encoder, get_data_loaders, mixed_precision, speaker_embedding, unpack
from src.models.bsrnn import extract_spectrogram
from src.models.stft import stft_forward
from src.models.turn_down_head import DB_PER_NEPER, build_turn_down_head, oracle_gain_db, turn_down_logits, turn_down_loss
from src.run_log import timed

HISTORY_FIELDS = ["epoch", "train_loss_db2", "val_loss_db2", "val_predicted_gain_db", "val_oracle_gain_db", "lr", "seconds"]
PROGRESS_EVERY = 200


# this loads e21 from its checkpoint, with every weight frozen, and gives its config too
def load_frozen_base(weights_path, device):
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)  # our own file: it carries the config dict
    base_config = checkpoint["config"]
    base_model = build_model(base_config)
    base_model.load_state_dict(checkpoint["model"])
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False
    base_model.to(device)
    return base_model, base_config


# this builds e21's own loaders, with only the batch size and the worker count taken from this run's config
def build_loaders(base_config, config, manifest_dir, data_root):
    loader_config = copy.deepcopy(base_config)
    loader_config["data"]["batch_size"] = int(config["training"]["batch_size"])
    loader_config["data"]["num_workers"] = int(config["training"]["num_workers"])
    return get_data_loaders(manifest_dir, data_root, loader_config)


# this runs one batch: e21 without gradients, then the head, and gives the loss with the mean predicted and oracle gains in dB
def batch_loss(batch, base_model, speaker_encoder, head, config, device, use_mixed_precision):
    mixture, target, enrollment, crop_absent = unpack(batch, device)
    with torch.no_grad():
        with mixed_precision(use_mixed_precision):
            estimate_spectrogram, features, mixture_spectrogram = extract_spectrogram(base_model, mixture, enrollment, speaker_embedding(speaker_encoder, enrollment))
        # The head and the loss run in float32, as e21's loss did.
        estimate_spectrogram = estimate_spectrogram.to(torch.complex64)
        mixture_spectrogram = mixture_spectrogram.to(torch.complex64)
        features = features.float()
        target_spectrogram = stft_forward(base_model.stft, target)
        band_widths = base_model.band_widths
        target_gain_db = oracle_gain_db(estimate_spectrogram, target_spectrogram, crop_absent, band_widths, float(config["target"]["activity_floor_db"]), float(config["target"]["gain_floor_db"]))
    logits = turn_down_logits(head, features, estimate_spectrogram, mixture_spectrogram, band_widths)
    loss = turn_down_loss(logits, target_gain_db, estimate_spectrogram, band_widths, float(config["target"]["audible_floor_db"]))
    predicted_gain_db = float((DB_PER_NEPER * torch.nn.functional.logsigmoid(logits.detach())).mean())
    return loss, predicted_gain_db, float(target_gain_db.mean())


# this runs one pass over a loader, training when an optimizer is given, and gives the mean loss and gains
def run_epoch(loader, base_model, speaker_encoder, head, config, device, optimizer, max_batches):
    use_mixed_precision = device.type == "cuda"
    head.train(optimizer is not None)
    pass_start = time.time()
    loss_sum = 0.0
    predicted_sum = 0.0
    oracle_sum = 0.0
    count = 0
    for batch in loader:
        if optimizer is not None:
            loss, predicted_gain_db, oracle_gain = batch_loss(batch, base_model, speaker_encoder, head, config, device, use_mixed_precision)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(config["training"]["grad_clip"]))
            optimizer.step()
        else:
            with torch.no_grad():
                loss, predicted_gain_db, oracle_gain = batch_loss(batch, base_model, speaker_encoder, head, config, device, use_mixed_precision)
        loss_sum += float(loss.detach())
        predicted_sum += predicted_gain_db
        oracle_sum += oracle_gain
        count += 1
        if optimizer is not None and count % PROGRESS_EVERY == 0:
            # The elapsed time and the steps left give the epoch's length long before it ends.
            elapsed = time.time() - pass_start
            print(f"    step {count}/{len(loader)}: loss {loss_sum / count:.3f} dB^2, {elapsed / 60:.1f} min so far, about {elapsed / count * len(loader) / 60:.0f} min for this epoch's training", flush=True)
        if max_batches is not None and count >= max_batches:
            break
    return {"loss": loss_sum / count, "predicted_gain_db": predicted_sum / count, "oracle_gain_db": oracle_sum / count}


# this gives the commit the run came from, marked dirty when the working tree has changes. On Kaggle there is no git history, so it reads the commit the bundle was built from
def git_commit():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit:
        status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        if status:
            return commit + "-dirty"
        return commit
    if Path("bundle_manifest.json").exists():
        return json.loads(Path("bundle_manifest.json").read_text())["git_commit"] + " (from the Kaggle bundle)"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="experiments/configs/turn_down_head.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--outdir", default="models/")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--manifest-dir", default="data/manifests")
    parser.add_argument("--resume", action="store_true", help="carry on from turn_down_head_last.pt in --outdir")
    parser.add_argument("--check", action="store_true", help="two validation batches with no gradients, then stop")
    arguments = parser.parse_args()

    config_path = Path(arguments.config)
    config = yaml.safe_load(config_path.read_text())
    seed = int(config["seed"])
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not arguments.check:
        raise SystemExit("No GPU. This script never trains on a CPU; use --check for a forward-only test.")

    base_model, base_config = load_frozen_base(config["base"]["weights"], device)
    speaker_encoder = build_speaker_encoder(base_config, device)
    head = build_turn_down_head(len(base_model.band_widths), int(base_config["model"]["separator"]["feature_dim"]), int(config["head"]["hidden_dim"]))
    head.to(device)
    parameter_count = 0
    for parameter in head.parameters():
        parameter_count += parameter.numel()
    train_loader, val_loader = build_loaders(base_config, config, Path(arguments.manifest_dir), Path(arguments.data_root))
    print(f"seed {seed}, device {device}, base {config['base']['weights']} (frozen), head {parameter_count:,} weights, train trials {len(train_loader.dataset)}, val trials {len(val_loader.dataset)}", flush=True)

    if arguments.check:
        result = run_epoch(val_loader, base_model, speaker_encoder, head, config, device, None, 2)
        print(f"check: val loss {result['loss']:.3f} dB^2, predicted gain {result['predicted_gain_db']:.2f} dB, oracle gain {result['oracle_gain_db']:.2f} dB")
        return

    optimizer = torch.optim.AdamW(head.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))  # AdamW (Loshchilov & Hutter, 2019)
    out_directory = Path(arguments.outdir)
    out_directory.mkdir(parents=True, exist_ok=True)
    best_path = out_directory / "turn_down_head.pt"
    last_path = out_directory / "turn_down_head_last.pt"
    results_directory = Path("experiments/results") / f"{date.today().isoformat()}-train-turn-down"
    results_directory.mkdir(parents=True, exist_ok=True)
    (results_directory / "config.yaml").write_text(config_path.read_text())
    history_path = results_directory / "history.csv"

    start_epoch = 0
    best_val = float("inf")
    if arguments.resume:
        state = torch.load(last_path, map_location=device, weights_only=False)
        head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_val = state["best_val"]
        print(f"resumed {last_path} at epoch {start_epoch}, best val {best_val:.3f}", flush=True)
    elif not history_path.exists():
        with open(history_path, "w", newline="") as handle:
            csv.writer(handle).writerow(HISTORY_FIELDS)

    num_epochs = arguments.epochs if arguments.epochs is not None else int(config["training"]["epochs"])
    progress = {"epochs": 0}
    start_time = time.time()
    with timed(f"scripts/train_turn_down.py --config {config_path}", scope=lambda: f"{len(train_loader.dataset):,} trials x {progress['epochs']} epochs, head only on frozen e21", rate=lambda: f"batch {config['training']['batch_size']}, {device}, {(time.time() - start_time) / max(progress['epochs'], 1):.0f} s/epoch"):
        for epoch in range(start_epoch, num_epochs):
            epoch_start = time.time()
            train_result = run_epoch(train_loader, base_model, speaker_encoder, head, config, device, optimizer, None)
            val_result = run_epoch(val_loader, base_model, speaker_encoder, head, config, device, None, None)
            seconds = time.time() - epoch_start
            row = [epoch, round(train_result["loss"], 4), round(val_result["loss"], 4), round(val_result["predicted_gain_db"], 3), round(val_result["oracle_gain_db"], 3), optimizer.param_groups[0]["lr"], round(seconds)]
            with open(history_path, "a", newline="") as handle:
                csv.writer(handle).writerow(row)
            record = {"head": head.state_dict(), "config": config, "base_weights": config["base"]["weights"], "epoch": epoch, "val_loss": val_result["loss"], "seed": seed}
            if val_result["loss"] < best_val:
                best_val = val_result["loss"]
                torch.save(record, best_path)
            torch.save({"head": head.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch, "best_val": best_val, "config": config}, last_path)
            progress["epochs"] += 1
            # Generated by Claude Code
            print(f"epoch {epoch}: train {train_result['loss']:.3f} dB^2, val {val_result['loss']:.3f} dB^2 (best {best_val:.3f}), val gain predicted {val_result['predicted_gain_db']:.2f} dB vs oracle {val_result['oracle_gain_db']:.2f} dB, {seconds / 60:.1f} min", flush=True)

    meta = {"date": date.today().isoformat(), "git_commit": git_commit(), "config": str(config_path), "seed": seed, "base_weights": config["base"]["weights"], "head_weights": parameter_count, "epochs_run": progress["epochs"], "best_val_loss_db2": round(best_val, 4), "device": str(device), "wall_seconds": round(time.time() - start_time)}
    with open(results_directory / "meta.yaml", "w") as handle:
        yaml.safe_dump(meta, handle, sort_keys=False)
    print(f"best head: {best_path} (val {best_val:.3f} dB^2); history in {history_path}")


if __name__ == "__main__":
    main()
