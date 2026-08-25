import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import yaml
import hashlib
import csv
import subprocess
import time
from pathlib import Path
import sys
from collections import defaultdict
from datetime import date
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
# Agg before pyplot is imported, never after -- the backend is fixed at import.
# Training runs on a headless server and on Kaggle, where the default backend
# has no display and plt.show() either warns or blocks.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# `python scripts/train.py` puts scripts/ on sys.path, not the repo root, so
# the src.* imports below fail without this. Same line as
# scripts/measure_vad_impact.py -- which is why the src imports come after it.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402
from src.models.bsrnn import BSRNN_TFMAP  # noqa: E402
from src.models.losses import LossBSRNN  # noqa: E402
from src.run_log import timed  # noqa: E402

def build_loss_fn(config):
    w = float(config["loss"]["w"])
    wm = float(config["loss"]["w_m"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    p = float(config["loss"]["p"])
    windows = config["loss"]["windows_ms"]
    # the loss turns windows_ms into n_fft with this, so it cannot be defaulted
    # inside the loss without the run using a rate the config does not record
    sample_rate = int(config["data"]["sample_rate"])

    # w is a convex weight between the two halves. A typo of 4.58 for 0.458
    # makes (1 - w) negative, which trains the model to destroy the target
    # while the curve still looks like it is descending.
    assert 0.0 <= w <= 1.0, f"loss.w must be in [0, 1], got {w}"
    assert wm >= 0.0, f"loss.w_m must be >= 0, got {wm}"

    return LossBSRNN(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p, windows=windows,
                     sample_rate=sample_rate)


def total_loss_floor(config):
    """Best total the objective can reach, for the reference line on the plot.

    At exact reconstruction L_pres hits 10*log10(tau_pres), L_MR hits 0 and
    L_abs hits 10*log10(tau_abs), so the floor is the w-weighted sum of the two.
    Was 10*log10(tau) off a single shared tau; once tau_pres and tau_abs differ
    (2026-08-25) no single tau defines the floor and that read the wrong one.
    """
    w = float(config["loss"]["w"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    return ((1 - w) * 10 * np.log10(tau_pres) + w * 10 * np.log10(tau_abs))


def build_model(config):
    """Config -> BSRNN_TFMAP. Every ctor argument comes from the yaml.

    Separate from main() so scripts/measure_train_cost.py measures the model
    that actually trains, rather than a second copy of this call that can drift.

    Two config keys are deliberately not passed: separator.norm (cLN is implied
    by causal=True inside SubbandNorm) and n_hidden (ctor default 1, chosen in
    decisions-m1.md 2026-08-18). Both belong in the yaml eventually so they are
    logged rather than inherited.
    """
    return BSRNN_TFMAP(
        sample_rate=config["data"]["sample_rate"],
        n_fft=config["model"]["stft"]["n_fft"],
        hop=config["model"]["stft"]["hop"],
        band_segments=config["model"]["bands"]["plan"],
        feature_dim=config["model"]["separator"]["feature_dim"],
        hidden_dim=config["model"]["separator"]["lstm_hidden"],
        num_repeat=config["model"]["separator"]["num_repeat"],
        causal=config["model"]["separator"]["causal"],
        mlp_hidden=config["model"]["mask"]["mlp_hidden"],
        residual_branch=config["model"]["mask"]["residual_branch"],
        lookahead_frames=config["model"]["lookahead_frames"],
    )


def unpack(batch, device):
    """Loader dict -> the four signals the objective needs.

    Three signals, not two: L_abs has no target to compare against (the right
    answer is silence) so the MIXTURE is its yardstick.

    trial_id (list of str) and meta (dict) stay on the CPU. They are for
    logging, and neither has a .to() -- that was the AttributeError.
    """
    mixture = batch["mixture"].to(device)          # x_input
    target = batch["target"].to(device)            # s_target
    enrollment = batch["enrollment"].to(device)
    crop_absent = batch["crop_absent"].to(device)  # (B,) bool, from the CROPPED stem
    return mixture, target, enrollment, crop_absent


def add_parts(sums, counts, parts):
    """Accumulate each loss term against its own crop count.

    NOT loss.item() * batch_size. A batch loss is
    (1 - w) * mean_present[L_pres + wm*L_MR] + w * mean_absent[L_abs]: two
    means over different subsets whose sizes change from batch to batch. A
    batch-size weighting therefore makes one present crop in one batch count
    as much as eleven in another, and the epoch number moves when only the
    shuffle changes.

    Gated on the counts, never on isnan(): LossBSRNN returns NaN for a half
    with no crops (~1.5 % of batches have no absent crop at batch 12), but a
    NaN from a real numerical failure must still reach the log.
    """
    if parts["n_present"]:
        sums["L_pres"] += parts["L_pres"] * parts["n_present"]
        sums["L_MR"] += parts["L_MR"] * parts["n_present"]
        counts["present"] += parts["n_present"]
    if parts["n_absent"]:
        sums["L_abs"] += parts["L_abs"] * parts["n_absent"]
        counts["absent"] += parts["n_absent"]


def epoch_report(sums, counts, w, wm):
    """Recombine the accumulated terms with the same w and wm as the batch loss."""
    n_present, n_absent = counts["present"], counts["absent"]
    L_pres = sums["L_pres"] / n_present if n_present else float("nan")
    L_MR = sums["L_MR"] / n_present if n_present else float("nan")
    L_abs = sums["L_abs"] / n_absent if n_absent else float("nan")

    return {
        "total": (1 - w) * (L_pres + wm * L_MR) + w * L_abs,
        "L_pres": L_pres,
        "L_MR": L_MR,
        "L_abs": L_abs,
        "n_present": n_present,
        "n_absent": n_absent,
    }


def train(model, train_loader, val_loader, optimizer, num_epochs, device, print_debug=False, save_path=None, config=None, scheduler=None, start_epoch=0, best_val=float("inf"), best_row=None):
    model.to(device)
    val_loss_history = []
    train_loss_history = []

    if config is None:
        raise ValueError("Config must be provided to build the loss function.")

    loss_fn = build_loss_fn(config)
    grad_clip = float(config["training"]["grad_clip"])
    patience = int(config["training"]["patience"])
    epochs_since_best = 0

    for epoch in range(start_epoch, num_epochs):
        # Re-crop. Offsets are derived from (seed, epoch, idx), so without this
        # every epoch reads the same 4 s window of every clip -- reproducible,
        # and it throws away five sixths of the audio.
        train_loader.dataset.set_epoch(epoch)

        model.train()
        sums, counts = defaultdict(float), defaultdict(int)
        # TRAINING LOSS
        # leave=False so a finished epoch's bar is erased and the scrollback
        # keeps only the per-epoch print, not one stale bar per epoch.
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{num_epochs} train",
                    unit="batch", leave=False, dynamic_ncols=True)
        for batch in pbar:
            mixture, target, enrollment, crop_absent = unpack(batch, device)

            optimizer.zero_grad()
            s_output = model(mixture, enrollment)
            # arg order is (reference, output, mixture, mask) -- reference
            # FIRST, the reverse of the usual (pred, target). See LossBSRNN.
            loss, parts = loss_fn(target, s_output, mixture, crop_absent)
            loss.backward()
            # A six-layer LSTM stack on an SI-SDR-family loss: a near-silent
            # present crop puts a very large gradient through alpha. Clip value
            # comes from the config, never from here.
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            add_parts(sums, counts, parts)
            # Running epoch loss, not this batch's: the batch number swings on
            # how many present/absent crops the shuffle happened to put in it.
            # Same recombination as the epoch report, so the bar converges to
            # the number that gets logged.
            pbar.set_postfix_str(
                f"loss {epoch_report(sums, counts, loss_fn.w, loss_fn.wm)['total']:.4f}")

        pbar.close()
        epoch_loss = epoch_report(sums, counts, loss_fn.w, loss_fn.wm)
        train_loss_history.append(epoch_loss)

        if print_debug and (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss["total"]:.4f}')


        # VALIDATION LOSS
        model.eval()
        val_sums, val_counts = defaultdict(float), defaultdict(int)
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"epoch {epoch+1}/{num_epochs} val",
                              unit="batch", leave=False, dynamic_ncols=True):
                mixture, target, enrollment, crop_absent = unpack(batch, device)

                s_output = model(mixture, enrollment)
                _, parts = loss_fn(target, s_output, mixture, crop_absent)
                add_parts(val_sums, val_counts, parts)

        val_loss = epoch_report(val_sums, val_counts, loss_fn.w, loss_fn.wm)
        val_loss["epoch"] = epoch
        val_loss["lr"] = optimizer.param_groups[0]["lr"]
        val_loss_history.append(val_loss)

        if scheduler is not None:
            scheduler.step(val_loss["total"])

        # Save the model if it has the best validation loss so far.
        if val_loss["total"] < best_val:
            best_val = val_loss["total"]
            best_row = val_loss
            epochs_since_best = 0
            if save_path:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "best_val": best_val,
                    "best_row": best_row,
                    "config": config,
                    "seed": config["seed"],
                }, save_path)
        else:
            epochs_since_best += 1
            # EARLY STOPPING. Counted in validations since the best, so it is
            # patience epochs of no improvement, not patience epochs total.
            if epochs_since_best >= patience:
                if print_debug:
                    print(f"Early stop at epoch {epoch}: {patience} epochs "
                          f"without improving on {best_val:.4f}")
                break


    return train_loss_history, val_loss_history, best_row


# split -> ((train_manifest, train_audio_dir), (val_manifest, val_audio_dir))
#
# The manifest name and the audio directory used to be one string, because for
# smoke and full they happen to be equal. `mid` breaks that: it is a SUBSET
# manifest over train/val audio that was already rendered, so it must read
# data/rendered/train while carrying its own row list. Keeping them as one
# string would have meant either re-rendering 2,000 duplicate trials or
# symlinking 2,000 directories. See experiments/configs/generator.yaml
# `subsets:` and scripts/make_subset_manifest.py.
SPLIT_MANIFESTS = {
    "smoke": (("smoke_train", "smoke_train"), ("smoke_val", "smoke_val")),
    "mid":   (("mid_train",   "train"),       ("mid_val",   "val")),
    "full":  (("train",       "train"),       ("val",       "val")),
}


def get_data_loaders(split, csv_path, data_path, config):
    # The dataset's `split` is the directory name under data/rendered/, so it
    # must track the manifest -- hardcoding "smoke_train" made --split full
    # read smoke audio against the full manifest. It is a SEPARATE string from
    # the manifest name so a subset split can point at already-rendered audio.
    if split not in SPLIT_MANIFESTS:
        raise ValueError(f"Unknown split: {split}. Known: {sorted(SPLIT_MANIFESTS)}")
    (train_manifest, train_audio), (val_manifest, val_audio) = SPLIT_MANIFESTS[split]

    csv_train = csv_path / f"{train_manifest}.csv"
    csv_val = csv_path / f"{val_manifest}.csv"

    train_dataset = TrialDataset(
        manifest_csv=csv_train,
        data_root=data_path,
        split=train_audio,
        chunk_s=config["data"]["chunk_s"], # follow CARTSE
        sample_rate=config["data"]["sample_rate"],
        seed=config["seed"],
    )

    val_dataset = TrialDataset(
        manifest_csv=csv_val,
        data_root=data_path,
        split=val_audio,
        chunk_s=config["data"]["chunk_s"], # follow CARTSE
        sample_rate=config["data"]["sample_rate"],
        seed=config["seed"],
        random_crop=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=0)

    return train_loader, val_loader


def git_commit():
    # Same helper as scripts/measure_vad_impact.py. Duplicated rather than
    # shared, matching how the other scripts do it -- and -dirty matters: a
    # result logged against a dirty tree is not reproducible from that hash.
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def log_results(out_dir, config, config_path, args, model, device, manifest_csv, train_loss_history, val_loss_history, best_row, wall_s, num_epochs, save_path):
    """Write experiments/results/<dir>/{meta.yaml,history.csv}.

    The CLAUDE.md rule: every experiment result gets the config used, the git
    commit hash, the metrics, the seed and the date. Written in the same shape
    as experiments/results/2026-08-20-loss-anchor/meta.yaml so a training run
    and the anchor measurement read side by side.

    history.csv is per-epoch and WIDE -- train_* and val_* on one row -- because
    the thing actually plotted is the two curves against each other. The lr
    column is what tells a plateau apart from a scheduler step.

    Never raises: a logging bug must not throw away a finished training run.
    Same reasoning as src/run_log.py.
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        fields = ["total", "L_pres", "L_MR", "L_abs", "n_present", "n_absent"]
        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch"] + [f"train_{k}" for k in fields]
                            + [f"val_{k}" for k in fields] + ["lr"])
            # Epoch comes from the VAL row: on a resume the histories start at
            # start_epoch, so enumerate() would relabel epoch 40 as epoch 0.
            for tr, va in zip(train_loss_history, val_loss_history):
                writer.writerow([va["epoch"]] + [tr[k] for k in fields]
                                + [va[k] for k in fields] + [va["lr"]])

        # The manifest's own provenance, carried through the way
        # measure_vad_impact.py does it -- the result is only interpretable
        # against the data build that produced it.
        manifest_meta_path = Path(manifest_csv).with_suffix(".meta.yaml")
        manifest_meta = (yaml.safe_load(manifest_meta_path.read_text())
                         if manifest_meta_path.exists() else {})

        epochs_run = len(train_loss_history)
        last_epoch = val_loss_history[-1]["epoch"] if val_loss_history else -1

        (out_dir / "meta.yaml").write_text(yaml.safe_dump({
            "date": date.today().isoformat(),
            "script": "scripts/train.py",
            "git_commit": git_commit(),
            "seed": int(config["seed"]),
            "config": str(config_path),
            # md5 of the config FILE, matching the manifests' config_md5. A
            # silent yaml edit then cannot be mistaken for the same experiment.
            "config_md5": hashlib.md5(Path(config_path).read_bytes()).hexdigest(),
            "split": args.split,
            "manifest": str(manifest_csv),
            "manifest_built_at_commit": manifest_meta.get("git_commit"),
            "manifest_config_md5": manifest_meta.get("config_md5"),
            "device": str(device),
            "resumed": bool(args.resume),
            "model": {
                "class": type(model).__name__,
                "n_parameters": sum(p.numel() for p in model.parameters()),
                "n_bands": len(model.band_widths),
            },
            "epochs_requested": num_epochs,
            "epochs_run": epochs_run,
            # True means patience ran out, not that the epoch budget did.
            "early_stopped": bool(epochs_run and last_epoch + 1 < num_epochs),
            "wall_seconds": round(wall_s, 1),
            "seconds_per_epoch": round(wall_s / epochs_run, 1) if epochs_run else None,
            # Copied verbatim so the result is readable without opening the
            # yaml -- config_md5 above is what proves they match.
            "loss": config["loss"],
            "training": config["training"],
            # The best val row, whichever run produced it -- on a resume that
            # can be an earlier invocation, which is why it rides in the
            # checkpoint rather than being recomputed from val_loss_history.
            # `or {}` because a checkpoint predating best_row has none.
            "best_val": {k: (best_row or {}).get(k) for k in
                         ["epoch", "total", "L_pres", "L_MR", "L_abs", "lr"]},
            "final_train": train_loss_history[-1] if train_loss_history else {},
            "checkpoint": str(save_path),
        }, sort_keys=False))

        print(f"Wrote {out_dir}/meta.yaml and {out_dir}/history.csv")
        return True
    except Exception as e:                      # noqa: BLE001 - never fatal
        print(f"  log_results: could not write {out_dir}: {e}", file=sys.stderr)
        return False


def plot_history(out_dir, train_loss_history, val_loss_history, best_row, loss_floor):
    """Write loss_plot.png: train and val `total` against epoch.

    The histories are lists of DICTS from epoch_report, not floats -- passing
    them straight to plt.plot raises `TypeError: unhashable type: 'dict'`,
    which is what happened on 2026-08-24 *after* the run had finished and the
    results were already written. Hence `total` pulled out explicitly, and hence
    the try block: a plotting bug must not be the last thing a 12-hour run does.
    Same never-raise contract as log_results.

    X axis comes from the VAL row's own `epoch`, not enumerate(), so a resumed
    run plots epochs 40-60 rather than relabelling them 0-20.
    """
    try:
        if len(val_loss_history) < 2:
            return False                    # a single point is not a curve
        epochs = [v["epoch"] for v in val_loss_history]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, [t["total"] for t in train_loss_history], label="train")
        ax.plot(epochs, [v["total"] for v in val_loss_history], label="val")

        # The two reference lines that say whether the curve is any good. The
        # floor is total_loss_floor(config), reachable only at exact
        # reconstruction; the anchor is the do-nothing baseline measured over 300
        # crops in experiments/results/2026-08-20-loss-anchor/. NOTE -2.24 was
        # computed at the old shared tau=0.001; at tau_abs=0.01 it is -2.22. The
        # 0.02 shift is invisible on this plot, but the constant is tau-dependent
        # and hardcoded, so it needs recomputing if w, w_m or tau_abs move.
        ax.axhline(loss_floor, ls=":", lw=1, color="grey")
        ax.axhline(-2.24, ls="--", lw=1, color="crimson")
        # Labels right-aligned inside the axes, not anchored to a data point --
        # at epochs[0] the anchor label sat directly on top of both curves.
        ax.text(0.995, loss_floor, f" floor {loss_floor:.0f} ", fontsize=8,
                color="grey", ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.text(0.995, -2.24, " do-nothing anchor -2.24 ", fontsize=8, color="crimson",
                ha="right", va="top", transform=ax.get_yaxis_transform())

        if best_row and best_row.get("epoch") is not None:
            ax.plot(best_row["epoch"], best_row["total"], "o", ms=7, mfc="none",
                    color="black", label=f"best val {best_row['total']:.3f}")

        ax.set_xlabel("epoch")
        ax.set_ylabel("total loss")
        # Integer ticks: epochs are counts, and the default locator was showing
        # 0.5 and 1.5 on short runs.
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_plot.png", dpi=130)
        plt.close(fig)                      # or figures accumulate across calls
        print(f"Wrote {out_dir}/loss_plot.png")
        return True
    except Exception as e:                  # noqa: BLE001 - never fatal
        print(f"  plot_history: could not plot: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--epochs", type=int, default=None,
                    help="override training.epochs from the config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--outdir", default="models/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--results-dir", default=None,
                    help="default experiments/results/<today>-train-<split>")

    args = ap.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    data_path_root = Path(args.data_root)
    csv_path = Path(args.manifest_dir)

    if args.split not in SPLIT_MANIFESTS:
        raise ValueError(f"Unknown split: {args.split}")

    # Non-negotiable rule: set and log a seed for every run. Seeded before the
    # model is built, so the weight init is reproducible too, not just the data.
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])
    print(f"seed {seed}  epochs {num_epochs}  config {config_path}")

    train_loader, val_loader = get_data_loaders(args.split, csv_path, data_path_root, config)

    # Build model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)

    # optimiser
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config["training"]["lr_factor"]),
        patience=int(config["training"]["lr_patience"]))

    save_path = Path(args.outdir) / f"model_{args.split}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, best_row = 0, float("inf"), None
    if args.resume:
        if not save_path.exists():
            raise FileNotFoundError(f"--resume but no checkpoint at {save_path}")
        # weights_only=False: the checkpoint carries the config dict, not just
        # tensors. Safe because we wrote it; never point this at a file you did not.
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt["best_val"]
        # .get, not [...]: checkpoints written before best_row existed have no
        # such key, and a missing best row is not a reason to refuse a resume.
        best_row = ckpt.get("best_row")
        # A resume across a config change is two experiments in one curve.
        if ckpt.get("config") != config:
            raise ValueError(f"{save_path} was trained under a different config, start a fresh run or pass a matching --config")
        print(f"resumed {save_path} at epoch {start_epoch}, best_val {best_val:.4f}")

    # Train the model.
    # `timed` writes the docs/run_times.md row (the over-a-minute rule) and
    # logs it even if training dies, marked (failed)
    ran = {"epochs": 0}
    t0 = time.time()
    with timed(f"scripts/train.py --split {args.split}",
               scope=lambda: f"{len(train_loader.dataset):,} trials x "
                             f"{ran['epochs']} epochs, {args.split}",
               rate=lambda: f"batch {config['data']['batch_size']}, {device}, "
                            f"{(time.time() - t0) / max(ran['epochs'], 1):.0f} s/epoch"):
        train_loss_history, val_loss_history, best_row = train(
            model,
            train_loader,
            val_loader,
            optimizer,
            num_epochs=num_epochs,
            device=device,
            print_debug=True,
            save_path=save_path,
            config=config,
            scheduler=scheduler,
            start_epoch=start_epoch,
            best_val=best_val,
            best_row=best_row,
        )
        ran["epochs"] = len(train_loss_history)
    wall_s = time.time() - t0

    results_dir = (Path(args.results_dir) if args.results_dir else
                   Path("experiments/results") /
                   f"{date.today().isoformat()}-train-{args.split}")
    
    train_manifest = csv_path / f"{SPLIT_MANIFESTS[args.split][0][0]}.csv"
    log_results(results_dir, config, config_path, args, model, device,
                train_manifest, train_loss_history, val_loss_history, best_row,
                wall_s, num_epochs, save_path)
    plot_history(results_dir, train_loss_history, val_loss_history, best_row,
                 loss_floor=total_loss_floor(config))

if __name__ == "__main__":
    main()