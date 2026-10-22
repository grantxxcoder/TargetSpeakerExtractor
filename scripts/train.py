"""Train the extractor on one split: python scripts/train.py --split sir0 --config experiments/configs/<run>.yaml"""

import argparse
import contextlib
import copy
import csv
import hashlib
import math
import subprocess
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib
import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

matplotlib.use("Agg")  # before pyplot: training runs headless
import matplotlib.pyplot as plt  # noqa: E402

# `python scripts/train.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.holdouts import HOLDOUT_SPLITS  # noqa: E402
from src.models.bsrnn import BSRNN_TFMAP, BSRNN_TFMAP_CONTEXT  # noqa: E402
from src.models.losses import LossBSRNN  # noqa: E402
from src.run_log import timed  # noqa: E402


# Experiment options removed 2026-10-22. Neither final model (baseline-matched e11, extension e21) uses any of them, and
# git history before that date has them. A config that still switches one on is refused, never silently ignored.
REMOVED_OPTIONS = {
    ("model", "state_head"): False, ("model", "presence_gate"): False, ("model", "tfmap_inject"): False,
    ("loss", "w_state"): 0.0, ("loss", "w_struct"): 0.0, ("loss", "w_interf"): 1.0, ("loss", "w_artif"): 1.0,
    ("training", "data_parallel"): False, ("training", "init_weights"): None, ("training", "freeze_extractor"): False,
}


def refuse_removed_options(config, blocks=("model", "loss", "training")):
    """Raise if `config` switches on any REMOVED_OPTIONS entry in the given blocks."""
    on = [f"{block}.{key}" for (block, key), off in REMOVED_OPTIONS.items()
          if block in blocks and (config.get(block) or {}).get(key, off) not in (off, None)]
    if on:
        raise ValueError(f"{on}: experiment options removed 2026-10-22; only the final baseline and extension remain")


def build_loss_fn(config):
    """Config -> loss object."""
    refuse_removed_options(config, blocks=("loss",))
    w = float(config["loss"]["w"])
    wm = float(config["loss"]["w_m"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    p = float(config["loss"]["p"])
    windows = config["loss"]["windows_ms"]
    sample_rate = int(config["data"]["sample_rate"])  # windows_ms -> n_fft needs it
    wg = float(config["loss"].get("w_g", 0.0))
    gain_delta_db = float(config["loss"].get("gain_delta_db", 3.0))
    return LossBSRNN(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p, windows=windows, sample_rate=sample_rate, wg=wg, gain_delta_db=gain_delta_db)


# Absent-branch weight schedule (decisions-m2.md 2026-08-25). With L_abs on from the start the model learns silence before it
# learns to use the enrolment, so w starts low and ramps up. Indexed in optimiser steps since 2026-09-03, so the schedule does
# not stretch when the training set grows.

def _w_ramp(w_start, w_final, warmup, ramp, t):
    """Schedule shape in any unit: w_start until warmup, linear to w_final over ramp, then w_final."""
    if t < warmup:
        return w_start
    if ramp <= 0 or t >= warmup + ramp:
        return w_final
    frac = (t - warmup + 1) / ramp  # +1 so the last ramp step reaches w_final
    return w_start + frac * (w_final - w_start)


def schedule_in_steps(config):
    """(warmup, ramp, w_start, w_final) in optimiser steps, or None for a constant w."""
    w_final = float(config["loss"]["w"])
    sched = config["loss"].get("w_schedule")
    if not sched:
        return None
    assert not ({"warmup_epochs", "ramp_epochs"} & set(sched)), "w_schedule in epochs was removed 2026-10-22; use warmup_steps/ramp_steps"

    w_start = float(sched.get("w_start", 0.0))
    assert 0.0 <= w_start <= 1.0, f"w_schedule.w_start must be in [0, 1], got {w_start}"
    return int(sched.get("warmup_steps", 0)), int(sched.get("ramp_steps", 0)), w_start, w_final


def w_at_step(config, global_step):
    """Absent-branch weight for this step. global_step counts batches, including AMP-skipped ones, so it is reproducible."""
    resolved = schedule_in_steps(config)
    if resolved is None:
        return float(config["loss"]["w"])
    warmup, ramp, w_start, w_final = resolved
    return _w_ramp(w_start, w_final, warmup, ramp, global_step)


# Shared by the stdout rows and history.csv, so a log pasted from a killed run is a valid history.csv.
HISTORY_FIELDS = ["total", "L_pres", "L_MR", "L_gain", "L_abs", "n_present", "n_absent"]

# Validation-only diagnostics:
#   enrol_sens_db     output change when the enrolment is swapped. Near 0 dB = conditioned, very negative = ignored.
#   pres_abs_gap_db   output loudness on present minus absent crops. Large positive = it knows when to speak.
#   content_wer(_ema) in-loop validation WER and its smoothed value (src/live_model_metric/content_probe.py). NaN when the probe did not run.
VAL_DIAGNOSTICS = ["enrol_sens_db", "pres_abs_gap_db", "content_wer", "content_wer_ema"]


def history_header():
    # `w` is the weight that trained the epoch; `total` always uses the final w (see epoch_report).
    return ["epoch"] + [f"train_{k}" for k in HISTORY_FIELDS] + [f"val_{k}" for k in HISTORY_FIELDS] + ["lr", "w"] + [f"val_{k}" for k in VAL_DIAGNOSTICS]


def history_row(tr, va):
    """One history.csv row. Epoch comes from the val dict so a resume keeps its numbering; .get so older rows still write."""
    return ([va["epoch"]] + [tr.get(k, float("nan")) for k in HISTORY_FIELDS] + [va.get(k, float("nan")) for k in HISTORY_FIELDS]
            + [va["lr"], va.get("w", float("nan"))] + [va.get(k, float("nan")) for k in VAL_DIAGNOSTICS])


def format_epoch_breakdown(epoch, num_epochs, tr, va, epoch_seconds, w_trained):
    """Per-epoch table of every term, for stderr (stdout carries the CSV rows). gap = val - train, so a growing gap means memorising."""
    lines = [f"epoch {epoch + 1}/{num_epochs}  {epoch_seconds:.0f} s  lr {va['lr']:.2e}  w_trained {w_trained:.3f}",
             f"  {'term':<7} {'train':>10} {'val':>10} {'gap(val-train)':>15}"]
    for term in ("total", "L_pres", "L_MR", "L_gain", "L_abs"):
        train_value, val_value = tr.get(term, float("nan")), va.get(term, float("nan"))
        lines.append(f"  {term:<7} {train_value:>10.4f} {val_value:>10.4f} {val_value - train_value:>15.4f}")
    lines.append(f"  crops   train {tr['n_present']} present / {tr['n_absent']} absent   val {va['n_present']} / {va['n_absent']}")
    lines.append(f"  diag    enrol_sens {va.get('enrol_sens_db', float('nan')):.2f} dB   pres_abs_gap {va.get('pres_abs_gap_db', float('nan')):.2f} dB")
    return "\n".join(lines)


def build_context_encoder(config, device="cpu"):
    """Frozen ECAPA speaker encoder (Desplanques et al., 2020) for item 1c, or None.

    One constructor for training and evaluation, so they cannot embed differently. Pair it with context_kwargs().
    """
    if not bool(config["model"].get("context_embedding", False)):
        return None
    from src.models.context_encoder import ContextEncoder
    encoder = ContextEncoder(ecapa_dir=config["model"].get("ecapa_dir", "../ecapa_pretrained"), device=device, expected_hashes=config["model"].get("ecapa_sha256"), normalise=bool(config["model"].get("context_normalise", True)))
    want = int(config["model"].get("context_embedding_dim", 192))
    if encoder.embedding_dim != want:
        raise ValueError(f"the snapshot emits {encoder.embedding_dim}-d embeddings but the model was built for {want}. Set model.context_embedding_dim.")
    return encoder


def context_kwargs(encoder, enrollment):
    """{"enrol_embedding": ...} for item 1c, else {}."""
    if encoder is None:
        return {}
    return {"enrol_embedding": encoder.embed(enrollment)}


def diagnostic_accumulate(diag, model, mixture, enrollment, s_output, crop_absent, amp=False, enrol_embedding=None):
    """Accumulate the two validation diagnostics over one batch. The enrolment swap rolls within the batch, so it needs batch > 1."""
    if mixture.shape[0] > 1:
        # Roll the item-1c embedding with the waveform, or only half the conditioning is swapped.
        swapped = {} if enrol_embedding is None else {"enrol_embedding": enrol_embedding.roll(1, 0)}
        with amp_ctx(amp):
            y_swapped = model(mixture, enrollment.roll(1, 0), **swapped)
        y_swapped = y_swapped.float()  # sums of squares over 64k samples overflow fp16
        diag["swap_num"] += float((s_output - y_swapped).pow(2).sum())
        diag["swap_den"] += float(s_output.pow(2).sum())

    # per-crop output energy relative to its own mixture, in dB
    e = 10 * torch.log10(s_output.pow(2).sum(-1) / mixture.pow(2).sum(-1) + 1e-12)
    absent = crop_absent.bool()
    present = ~absent
    if present.any():
        diag["e_pres"] += float(e[present].sum()); diag["n_pres"] += int(present.sum())
    if absent.any():
        diag["e_abs"] += float(e[absent].sum()); diag["n_abs"] += int(absent.sum())


def diagnostic_report(diag):
    """Accumulators -> the two logged numbers, NaN when a half was never seen."""
    nan = float("nan")
    sens = 10 * np.log10(diag["swap_num"] / diag["swap_den"]) if diag["swap_den"] > 0 and diag["swap_num"] > 0 else nan
    gap = diag["e_pres"] / diag["n_pres"] - diag["e_abs"] / diag["n_abs"] if diag["n_pres"] and diag["n_abs"] else nan
    return {"enrol_sens_db": sens, "pres_abs_gap_db": gap}


def total_loss_floor(config):
    """Best reachable total, for the plot's reference line: the w-weighted 10 log10 of tau_pres and tau_abs."""
    w = float(config["loss"]["w"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    return (1 - w) * 10 * np.log10(tau_pres) + w * 10 * np.log10(tau_abs)


def build_model(config):
    """Config -> the baseline (BSRNN_TFMAP) or, with model.context_embedding, the extension (BSRNN_TFMAP_CONTEXT)."""
    refuse_removed_options(config, blocks=("model",))
    cls = BSRNN_TFMAP_CONTEXT if bool(config["model"].get("context_embedding", False)) else BSRNN_TFMAP  # item 1c (decisions-m2.md 2026-09-22)
    extra = {"embedding_dim": int(config["model"].get("context_embedding_dim", 192))} if cls is BSRNN_TFMAP_CONTEXT else {}
    return cls(
        **extra,
        sample_rate=config["data"]["sample_rate"],
        n_fft=config["model"]["stft"]["n_fft"],
        hop=config["model"]["stft"]["hop"],
        band_segments=config["model"]["bands"]["plan"],
        feature_dim=config["model"]["separator"]["feature_dim"],
        hidden_dim=config["model"]["separator"]["lstm_hidden"],
        num_repeat=config["model"]["separator"]["num_repeat"],
        causal=config["model"]["separator"]["causal"],
        mlp_hidden=config["model"]["mask"]["mlp_hidden"],
        n_hidden=int(config["model"]["mask"].get("n_hidden", 1)),  # 1 deviates from the WeSep reference's 2 (Wang et al., 2024; decisions-m1.md 2026-08-19)
        residual_branch=config["model"]["mask"]["residual_branch"],
        lookahead_frames=config["model"]["lookahead_frames"],
        tfmap_scale=_tfmap_scale(config),
        tfmap_parts=bool(config["model"].get("tfmap_parts", False)),  # item 1a: +66,820 params, not parameter-matched
        tfmap_part_scales=config["model"].get("tfmap_part_scales"),  # from scripts/derive_cue_scales.py; absent = all ones
    )


def _tfmap_scale(config):
    # Absent = 1.0, the pre-2026-08-25 behaviour: right for an old checkpoint, wrong for a new run.
    if "tfmap_scale" in config["model"]:
        return float(config["model"]["tfmap_scale"])
    print("WARNING: config has no model.tfmap_scale -- assuming 1.0, the pre-2026-08-25 behaviour. Correct for an old checkpoint, WRONG for a new run: add the key to the config.", file=sys.stderr)
    return 1.0


def amp_ctx(enabled):
    """fp16 autocast for the model forward only (Micikevicius et al., 2018). The loss stays fp32: its 1e-12 epsilons underflow in fp16."""
    return torch.amp.autocast("cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def unpack(batch, device):
    """Loader dict -> (mixture, target, enrollment, crop_absent) on device. L_abs uses the mixture as its reference."""
    mixture = batch["mixture"].to(device)
    target = batch["target"].to(device)
    enrollment = batch["enrollment"].to(device)
    crop_absent = batch["crop_absent"].to(device)  # from the cropped stem, not the manifest label
    return mixture, target, enrollment, crop_absent


def add_parts(sums, counts, parts):
    """Accumulate each term weighted by the number of crops it was averaged over, so epoch means do not depend on the shuffle."""
    if parts["n_present"]:
        sums["L_pres"] += parts["L_pres"] * parts["n_present"]
        sums["L_MR"] += parts["L_MR"] * parts["n_present"]
        sums["L_gain"] += parts["L_gain"] * parts["n_present"]
        counts["present"] += parts["n_present"]
    if parts["n_absent"]:
        sums["L_abs"] += parts["L_abs"] * parts["n_absent"]
        counts["absent"] += parts["n_absent"]


def epoch_report(sums, counts, w, wm, wg):
    """Accumulated sums -> epoch means.

    `total` uses the reporting w (the schedule's final value), so it is one objective across epochs (decisions-m2.md 2026-08-25).
    """
    nan = float("nan")
    n_present, n_absent = counts["present"], counts["absent"]
    L_pres = sums["L_pres"] / n_present if n_present else nan
    L_MR = sums["L_MR"] / n_present if n_present else nan
    L_gain = sums["L_gain"] / n_present if n_present else nan
    L_abs = sums["L_abs"] / n_absent if n_absent else nan
    return {
        "total": (1 - w) * (L_pres + wm * L_MR + wg * L_gain) + w * L_abs,
        "L_pres": L_pres,
        "L_MR": L_MR,
        "L_gain": L_gain,
        "L_abs": L_abs,
        "n_present": n_present,
        "n_absent": n_absent,
    }


def lr_schedule_metric(val_loss, config):
    """The number ReduceLROnPlateau watches (training.lr_schedule_on): the same smoothed validation WER the selector uses."""
    mode = str(config["training"].get("lr_schedule_on", "content_wer"))
    if mode != "content_wer":
        raise ValueError(f"training.lr_schedule_on: {mode!r} was removed 2026-10-22; only content_wer remains.")
    return selection_score(val_loss, config)


def selection_score(val_loss, config):
    """The number that decides which epoch to keep (training.select_on). Lower is better.

    The smoothed in-loop validation WER, the only mode that tracks content (decisions-m2.md 2026-09-23). Smoothed, because
    per-epoch LCF-WER is too noisy to step a scheduler on. Raises rather than falling back.
    """
    mode = str(config["training"].get("select_on", "content_wer"))
    if mode != "content_wer":
        raise ValueError(f"training.select_on: {mode!r} was removed 2026-10-22; only content_wer remains.")
    value = val_loss.get("content_wer_ema", val_loss.get("content_wer"))
    if value is None or value != value:
        raise ValueError("select_on/lr_schedule_on is `content_wer` but no validation WER was computed this epoch. Enable the probe with a `content_probe:` block, and set `every_n_epochs: 1` -- the scheduler needs a number every epoch.")
    return float(value)


def selection_eligible(val_loss, config):
    """Whether an epoch may be kept at all: L_abs at or below training.select_abs_max (null = no bar)."""
    bar = config["training"].get("select_abs_max", None)
    return True if bar is None else float(val_loss["L_abs"]) <= float(bar)


# Data locations main() fills in for the probe. They are not part of what makes two runs the same.
PROBE_PATH_KEYS = ("data_root", "manifest_dir")


def comparable_config(config):
    """Config minus run-time data paths, for checking a resume uses the same config."""
    out = copy.deepcopy(config or {})
    probe = out.get("content_probe")
    if isinstance(probe, dict):
        for key in PROBE_PATH_KEYS:
            probe.pop(key, None)
    return out


def build_content_probe(config, verbose=True):
    """The in-loop validation WER probe, or None unless content_probe.enabled.

    Optional keys: split (never a holdout), n_trials, every_n_epochs, asr_device, ema_span, cap_errors_at_spoken, asr_model.
    Cost is mostly the CPU ASR: 40 clips took ~2.7 min per epoch (measured 2026-09-23).
    """
    block = config.get("content_probe") or {}
    if not block.get("enabled", False):
        return None
    from src.live_model_metric.content_probe import ASR_MODEL_SIZE, ContentProbe, load_probe_trials

    split = str(block.get("split", "sir0_val"))
    if split in HOLDOUT_SPLITS:
        raise ValueError(f"content_probe.split={split!r} is a reported holdout. Steering training on it would destroy the only clean number the project has. CLAUDE.md; decisions-m4.md 2026-09-15.")
    data_root = Path(block.get("data_root", "data"))
    trials = load_probe_trials(Path(block.get("manifest_dir", "data/manifests")) / f"{split}.csv", data_root / "rendered" / split, limit=int(block.get("n_trials", 40)), condition=str(block.get("condition", "both")))
    # cap_errors_at_spoken absent = off, so an older run resumes under the rule it was scored by.
    probe = ContentProbe(trials, model_size=str(block.get("asr_model", ASR_MODEL_SIZE)), device=str(block.get("asr_device", "cpu")), ema_span=int(block.get("ema_span", 3)), cap_errors_at_spoken=bool(block.get("cap_errors_at_spoken", False)))
    if verbose:
        print(f"  content probe ON: {probe.describe()}, split {split}, every {int(block.get('every_n_epochs', 1))} epoch(s)")
    return probe


def live_extractor(model, context_encoder, device):
    """Extractor over the model being trained: whole clip, one forward, the same path as make_estimates.py. Caller sets eval() and no_grad()."""
    def extract(mixture, enrollment, sample_rate):  # noqa: ARG001
        enrol = torch.from_numpy(enrollment).unsqueeze(0).to(device)
        estimate = model(torch.from_numpy(mixture).unsqueeze(0).to(device), enrol, **context_kwargs(context_encoder, enrol))
        return estimate.squeeze(0).float().cpu().numpy()
    return extract


def checkpoints_to_drop(kept, keep_top_k, keep_stride=0, last_epoch=None):
    """Epochs whose weights may be deleted. `kept` is a sorted (score, epoch) list.

    Keeps the top-k by score, plus every keep_stride-th epoch and the last: the signal score can rank epochs very differently
    from content, and a deleted checkpoint can only be recovered by retraining (decisions-m2.md 2026-09-23).
    """
    survivors = set()
    if keep_top_k > 0:
        survivors |= {epoch for _, epoch in kept[:keep_top_k]}
    if keep_stride and keep_stride > 0:
        survivors |= {epoch for _, epoch in kept if epoch % keep_stride == 0}
        if last_epoch is not None:
            survivors.add(last_epoch)
    return sorted({epoch for _, epoch in kept} - survivors)


def train(model, train_loader, val_loader, optimizer, num_epochs, device, print_debug=False, save_path=None, config=None, scheduler=None, start_epoch=0, best_val=float("inf"), best_row=None, start_step=0, context_encoder=None, resumed_wer_history=None):
    model.to(device)
    val_loss_history = []
    train_loss_history = []

    if config is None:
        raise ValueError("Config must be provided to build the loss function.")
    resumed_wer_history = list(resumed_wer_history or [])

    loss_fn = build_loss_fn(config)
    grad_clip = float(config["training"]["grad_clip"])
    patience = int(config["training"]["patience"])
    keep_top_k = int(config["training"].get("keep_top_k", 3))
    keep_stride = int(config["training"].get("keep_stride", 0))
    kept = []  # (score, epoch) of the checkpoints still on disk
    print(f"  selecting on `{config['training'].get('select_on', 'content_wer')}`, silence bar L_abs <= {config['training'].get('select_abs_max', 'none')}, keeping top {keep_top_k}")
    epochs_since_best = 0
    content_probe = build_content_probe(config)
    probe_every = int((config.get("content_probe") or {}).get("every_n_epochs", 1))
    if content_probe is not None and resumed_wer_history:
        content_probe.warm_start(resumed_wer_history)
        print(f"  content probe warm-started from {len(resumed_wer_history)} earlier epochs; EMA resumes at {content_probe.smoothed[-1]:.2f}")
    elif content_probe is not None and start_epoch > 0:
        print("  WARNING: resuming with a COLD content probe -- this checkpoint predates content_wer_history, so the EMA and the trend restart. The first epoch or two after the resume will read low-confidence.")

    w_report = float(config["loss"]["w"])  # every reported `total` uses this, so the curve is one objective
    steps_per_epoch = len(train_loader)  # the loader counts trials; both_directions widens batches, not epochs
    resolved = schedule_in_steps(config)
    if resolved is not None:
        warmup_steps, ramp_steps, _, _ = resolved
        full_at = warmup_steps + ramp_steps
        print(f"w schedule: {config['loss']['w_schedule']}", flush=True)
        print(f"  {steps_per_epoch} steps/epoch  warmup {warmup_steps} steps  ramp {ramp_steps} steps  full w at step {full_at} (epoch {full_at / max(steps_per_epoch, 1):.2f})", flush=True)
        shown = min(num_epochs, math.ceil(full_at / max(steps_per_epoch, 1)) + 1)  # only up to the end of the ramp
        ws = [w_at_step(config, e * steps_per_epoch) for e in range(shown)]
        print(f"  w at each epoch START: {[round(v, 4) for v in ws]}", flush=True)
        print(f"  reporting/selection w held at {w_report}", flush=True)

    # Mixed precision, CUDA only. With enabled=False the scaler's calls are no-ops, so one code path serves both precisions.
    use_amp = bool(config["training"].get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  mixed precision: {'ON (fp16 forward, fp32 loss)' if use_amp else 'off'}", flush=True)

    global_step = int(start_step)  # carried across a resume so the w warmup does not restart

    for epoch in range(start_epoch, num_epochs):
        train_loader.dataset.set_epoch(epoch)  # new crops each epoch

        model.train()
        sums, counts = defaultdict(float), defaultdict(int)
        w_sum, w_steps = 0.0, 0  # for the mean w that trained this epoch
        epoch_start = time.time()
        for batch in train_loader:
            loss_fn.w = w_at_step(config, global_step)
            w_sum += loss_fn.w
            w_steps += 1

            mixture, target, enrollment, crop_absent = unpack(batch, device)
            ctx = context_kwargs(context_encoder, enrollment)

            optimizer.zero_grad()
            with amp_ctx(use_amp):
                s_output = model(mixture, enrollment, **ctx)
            # Reference first, then output (see LossBSRNN). The loss runs in fp32.
            loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent)
            scaler.scale(loss).backward()
            # Unscale before clipping, or the clip sees gradients inflated by the loss scale (Micikevicius et al., 2018).
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)  # gradient-norm clipping (Pascanu et al., 2013)
            # step() skips updates with inf/NaN gradients; a few early skips are the scaler calibrating.
            scaler.step(optimizer)
            scaler.update()

            add_parts(sums, counts, parts)
            global_step += 1

        w_trained = w_sum / max(w_steps, 1)
        epoch_loss = epoch_report(sums, counts, w_report, loss_fn.wm, loss_fn.wg)
        train_loss_history.append(epoch_loss)

        # Validation, in the same precision as training.
        model.eval()
        val_sums, val_counts = defaultdict(float), defaultdict(int)
        diag = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                mixture, target, enrollment, crop_absent = unpack(batch, device)
                ctx = context_kwargs(context_encoder, enrollment)

                with amp_ctx(use_amp):
                    s_output = model(mixture, enrollment, **ctx)
                s_output = s_output.float()
                _, parts = loss_fn(target, s_output, mixture, crop_absent)
                add_parts(val_sums, val_counts, parts)
                diagnostic_accumulate(diag, model, mixture, enrollment, s_output, crop_absent, amp=use_amp, enrol_embedding=ctx.get("enrol_embedding"))

        val_loss = epoch_report(val_sums, val_counts, w_report, loss_fn.wm, loss_fn.wg)
        val_loss.update(diagnostic_report(diag))
        # Content probe: whole clips with reference text, which the 4 s validation crops do not have.
        if content_probe is not None and epoch % probe_every == 0:
            with torch.no_grad():
                val_loss["content_wer"] = content_probe.score(live_extractor(model, context_encoder, device))
            val_loss["content_wer_ema"] = content_probe.smoothed[-1]
        val_loss["epoch"] = epoch
        val_loss["lr"] = optimizer.param_groups[0]["lr"]
        val_loss["w"] = w_trained
        val_loss_history.append(val_loss)

        # One CSV row per epoch on stdout (flushed, so a killed run keeps it), and the readable table on stderr.
        if epoch == start_epoch:
            print(",".join(history_header()), flush=True)
        print(",".join(str(v) for v in history_row(epoch_loss, val_loss)), flush=True)
        print(format_epoch_breakdown(epoch, num_epochs, epoch_loss, val_loss, time.time() - epoch_start, w_trained), file=sys.stderr, flush=True)
        if content_probe is not None and content_probe.history:
            print("  " + content_probe.verdict(), file=sys.stderr, flush=True)

        if scheduler is not None:
            scheduler.step(lr_schedule_metric(val_loss, config))

        # The newest weights, every epoch. Not what --resume reads, which is the best checkpoint.
        if save_path:
            last_path = Path(save_path).with_name(Path(save_path).stem + "_last.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                "global_step": global_step,
                "content_wer_history": content_probe.history if content_probe is not None else [],  # so the probe's EMA survives a resume
                "best_val": best_val,
                "best_row": best_row,
                "config": config,
                "seed": config["seed"],
            }, last_path)

        score = selection_score(val_loss, config)
        eligible = selection_eligible(val_loss, config)
        # Keep top-k and strided checkpoints so epochs can be re-scored on content later (see checkpoints_to_drop).
        # Not gated on `eligible`: on a run that never clears the silence bar these are the only weights that survive.
        if save_path and (keep_top_k > 0 or keep_stride > 0):
            kept.append((score, epoch))
            kept.sort()
            rank_path = Path(save_path).with_name(f"{Path(save_path).stem}_e{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch, "score": score, "eligible": eligible, "row": val_loss, "config": config, "seed": config["seed"]}, rank_path)
            for dropped in checkpoints_to_drop(kept, keep_top_k, keep_stride, last_epoch=epoch):
                Path(save_path).with_name(f"{Path(save_path).stem}_e{dropped:03d}.pt").unlink(missing_ok=True)
                kept[:] = [(sc, ep) for sc, ep in kept if ep != dropped]

        if eligible and score < best_val:
            best_val = score
            best_row = val_loss
            epochs_since_best = 0
            if save_path:
                torch.save({
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "epoch": epoch,
                    "global_step": global_step,
                    "content_wer_history": content_probe.history if content_probe is not None else [],
                    "best_val": best_val,
                    "best_row": best_row,
                    "config": config,
                    "seed": config["seed"],
                }, save_path)
        else:
            epochs_since_best += 1
            if epochs_since_best >= patience:  # early stopping: patience epochs without a new best
                if print_debug:
                    print(f"Early stop at epoch {epoch}: {patience} epochs without improving on {best_val:.4f}")
                break

    # No epoch cleared the silence bar, so no best checkpoint exists. Say so rather than relax the bar.
    if best_row is None:
        print(f"WARNING: no epoch met training.select_abs_max={config['training'].get('select_abs_max')} on L_abs, so no best checkpoint was written. The top-{keep_top_k} by score and _last.pt are on disk; kept epochs: {sorted(e for _, e in kept)}. Either the run is too short to reach the bar or the bar is wrong for this split.", file=sys.stderr)

    return train_loss_history, val_loss_history, best_row


# split -> ((train_manifest, train_audio_dir), (val_manifest, val_audio_dir)).
SPLIT_MANIFESTS = {
    "sir0":  (("sir0_train",  "sir0_train"),  ("sir0_val",  "sir0_val")),  # symmetric target/interferer loudness
    # Render-only from here down: each validates on a test split.
    "sir0pub": (("sir0_train", "sir0_train"), ("eval_public", "eval_public")),  # public test split (decisions-m3.md 2026-09-24)
    "sir0priv": (("sir0_train", "sir0_train"), ("eval_private", "eval_private")),  # scored once, after selection (decisions-m4.md 2026-10-07)
    "ami":   (("sir0_train", "sir0_train"), ("ami_eval",    "ami_eval")),  # AMI transfer set, eval only
}
RENDER_ONLY_SPLITS = frozenset({"sir0pub", "sir0priv", "ami"})


def get_data_loaders(split, csv_path, data_path, config):
    if split not in SPLIT_MANIFESTS:
        raise ValueError(f"Unknown split: {split}. Known: {sorted(SPLIT_MANIFESTS)}")
    if split in RENDER_ONLY_SPLITS:
        raise ValueError(f"split {split!r} is for rendering estimates only; training on it would select checkpoints on {SPLIT_MANIFESTS[split][1][0]}.")
    (train_manifest, train_audio), (val_manifest, val_audio) = SPLIT_MANIFESTS[split]
    both_directions = bool(config["data"].get("both_directions", False))  # each trial trained once per speaker (decisions-m2.md 2026-08-26)
    enrollment_variants = int(config["data"].get("enrollment_variants", 1))  # train only; val stays on enrollment.wav (2026-08-30)
    remix_gains = bool(config["data"].get("remix_gains", False))  # per-epoch SIR/SNR re-draw, train only

    # chunk_s = 4 s follows CARTSE (Li & Seki, 2026).
    train_dataset = TrialDataset(manifest_csv=csv_path / f"{train_manifest}.csv", data_root=data_path, split=train_audio, chunk_s=config["data"]["chunk_s"], sample_rate=config["data"]["sample_rate"], seed=config["seed"], both_directions=both_directions, enrollment_variants=enrollment_variants, remix_gains=remix_gains)
    val_dataset = TrialDataset(manifest_csv=csv_path / f"{val_manifest}.csv", data_root=data_path, split=val_audio, chunk_s=config["data"]["chunk_s"], sample_rate=config["data"]["sample_rate"], seed=config["seed"], random_crop=False, both_directions=both_directions)

    # num_workers 0 suits the laptop; on a GPU the wav reads need workers. Pinned memory lets host-to-GPU copies run async.
    num_workers = int(config["data"].get("num_workers", 0))
    extra = dict(persistent_workers=True, prefetch_factor=4) if num_workers else {}
    extra["pin_memory"] = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_pairs, **extra)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=num_workers, collate_fn=collate_pairs, **extra)
    return train_loader, val_loader


def git_commit():
    """HEAD, with -dirty if the tree has changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def log_results(out_dir, config, config_path, args, model, device, manifest_csv, train_loss_history, val_loss_history, best_row, wall_s, num_epochs, save_path):
    """Write experiments/results/<dir>/{meta.yaml, history.csv}: config, commit, metrics, seed and date (CLAUDE.md). Never raises."""
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "history.csv", "w", newline="") as f:
            writer = csv.writer(f, lineterminator="\n")  # LF, matching the rows printed to stdout
            writer.writerow(history_header())
            for tr, va in zip(train_loss_history, val_loss_history):
                writer.writerow(history_row(tr, va))

        manifest_meta_path = Path(manifest_csv).with_suffix(".meta.yaml")
        manifest_meta = yaml.safe_load(manifest_meta_path.read_text()) if manifest_meta_path.exists() else {}
        epochs_run = len(train_loss_history)
        last_epoch = val_loss_history[-1]["epoch"] if val_loss_history else -1

        (out_dir / "meta.yaml").write_text(yaml.safe_dump({
            "date": date.today().isoformat(),
            "script": "scripts/train.py",
            "git_commit": git_commit(),
            "seed": int(config["seed"]),
            "config": str(config_path),
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
            "early_stopped": bool(epochs_run and last_epoch + 1 < num_epochs),
            "wall_seconds": round(wall_s, 1),
            "seconds_per_epoch": round(wall_s / epochs_run, 1) if epochs_run else None,
            # Copied so the result is readable without the yaml.
            "data": config["data"],
            "loss": config["loss"],
            "training": config["training"],
            "best_val": {k: (best_row or {}).get(k) for k in ["epoch", "total", "L_pres", "L_MR", "L_abs", "lr"]},
            "selection": {
                "select_on": config["training"].get("select_on", "content_wer"),
                "select_abs_max": config["training"].get("select_abs_max", None),
                "keep_top_k": config["training"].get("keep_top_k", 3),
            },
            "final_train": train_loss_history[-1] if train_loss_history else {},
            "checkpoint": str(save_path),
        }, sort_keys=False))

        print(f"Wrote {out_dir}/meta.yaml and {out_dir}/history.csv")
        return True
    except Exception as e:  # noqa: BLE001 - a logging bug must not discard a finished run
        print(f"  log_results: could not write {out_dir}: {e}", file=sys.stderr)
        return False


def plot_history(out_dir, train_loss_history, val_loss_history, best_row, loss_floor):
    """Write loss_plot.png: train and val `total` per epoch, with the loss floor and the do-nothing anchor. Never raises."""
    try:
        if len(val_loss_history) < 2:
            return False
        epochs = [v["epoch"] for v in val_loss_history]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(epochs, [t["total"] for t in train_loss_history], label="train")
        ax.plot(epochs, [v["total"] for v in val_loss_history], label="val")

        # Do-nothing anchor from experiments/results/2026-08-20-loss-anchor/ (computed at tau 0.001; -2.22 at tau_abs 0.01).
        ax.axhline(loss_floor, ls=":", lw=1, color="grey")
        ax.axhline(-2.24, ls="--", lw=1, color="crimson")
        ax.text(0.995, loss_floor, f" floor {loss_floor:.0f} ", fontsize=8, color="grey", ha="right", va="bottom", transform=ax.get_yaxis_transform())
        ax.text(0.995, -2.24, " do-nothing anchor -2.24 ", fontsize=8, color="crimson", ha="right", va="top", transform=ax.get_yaxis_transform())

        if best_row and best_row.get("epoch") is not None:
            ax.plot(best_row["epoch"], best_row["total"], "o", ms=7, mfc="none", color="black", label=f"best val {best_row['total']:.3f}")

        ax.set_xlabel("epoch")
        ax.set_ylabel("total loss")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / "loss_plot.png", dpi=130)
        plt.close(fig)
        print(f"Wrote {out_dir}/loss_plot.png")
        return True
    except Exception as e:  # noqa: BLE001 - a plotting bug must not end a finished run
        print(f"  plot_history: could not plot: {e}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--epochs", type=int, default=None, help="override training.epochs from the config")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--config", default="experiments/configs/bsrnn_cue_context.yaml")
    ap.add_argument("--outdir", default="models/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--results-dir", default=None, help="default experiments/results/<today>-train-<split>")
    args = ap.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    refuse_removed_options(config)
    data_path_root = Path(args.data_root)
    csv_path = Path(args.manifest_dir)
    # The probe reads the same data as the val loader, so it takes the CLI paths unless its block names its own.
    if (config.get("content_probe") or {}).get("enabled", False):
        config["content_probe"].setdefault("data_root", str(data_path_root))
        config["content_probe"].setdefault("manifest_dir", str(csv_path))

    if args.split not in SPLIT_MANIFESTS:
        raise ValueError(f"Unknown split: {args.split}")

    # Seed every run, before the model is built so the weight init is reproducible too.
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    num_epochs = args.epochs if args.epochs is not None else int(config["training"]["epochs"])
    print(f"seed {seed}  epochs {num_epochs}  config {config_path}")

    train_loader, val_loader = get_data_loaders(args.split, csv_path, data_path_root, config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config)
    model.to(device)

    # Item 1c's encoder lives beside the model: frozen, not optimised, not saved in checkpoints.
    context_encoder = build_context_encoder(config, device)
    if context_encoder is not None:
        print(f"  context encoder: {context_encoder.describe()}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))  # AdamW (Loshchilov & Hutter, 2019)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(config["training"]["lr_factor"]), patience=int(config["training"]["lr_patience"]))

    save_path = Path(args.outdir) / f"model_{args.split}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, best_row, start_step = 0, float("inf"), None, 0
    resumed_wer_history = []

    if args.resume:
        if not save_path.exists():
            raise FileNotFoundError(f"--resume but no checkpoint at {save_path}")
        ckpt = torch.load(save_path, map_location=device, weights_only=False)  # our own file: it carries the config dict
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        start_step = int(ckpt.get("global_step", start_epoch * len(train_loader)))  # pre-2026-09-03 checkpoints have no global_step
        best_val = ckpt["best_val"]
        best_row = ckpt.get("best_row")
        resumed_wer_history = list(ckpt.get("content_wer_history") or [])
        if comparable_config(ckpt.get("config")) != comparable_config(config):
            raise ValueError(f"{save_path} was trained under a different config, start a fresh run or pass a matching --config")
        print(f"resumed {save_path} at epoch {start_epoch}, step {start_step}, best_val {best_val:.4f}")

    # `timed` writes the docs/run_times.md row, marked (failed) if training dies.
    ran = {"epochs": 0}
    t0 = time.time()
    with timed(f"scripts/train.py --split {args.split}",
               scope=lambda: f"{len(train_loader.dataset):,} trials x {ran['epochs']} epochs, {args.split}",
               rate=lambda: f"batch {config['data']['batch_size']}, {device}, {(time.time() - t0) / max(ran['epochs'], 1):.0f} s/epoch"):
        train_loss_history, val_loss_history, best_row = train(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, device=device, print_debug=True, save_path=save_path, config=config, scheduler=scheduler, context_encoder=context_encoder, start_epoch=start_epoch, best_val=best_val, best_row=best_row, start_step=start_step, resumed_wer_history=resumed_wer_history)
        ran["epochs"] = len(train_loss_history)
    wall_s = time.time() - t0

    results_dir = Path(args.results_dir) if args.results_dir else Path("experiments/results") / f"{date.today().isoformat()}-train-{args.split}"
    train_manifest = csv_path / f"{SPLIT_MANIFESTS[args.split][0][0]}.csv"
    log_results(results_dir, config, config_path, args, model, device, train_manifest, train_loss_history, val_loss_history, best_row, wall_s, num_epochs, save_path)
    plot_history(results_dir, train_loss_history, val_loss_history, best_row, loss_floor=total_loss_floor(config))


if __name__ == "__main__":
    main()
