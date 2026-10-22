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

matplotlib.use("Agg")  # before pyplot: Kaggle is headless
import matplotlib.pyplot as plt  # noqa: E402

# `python scripts/train.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.holdouts import HOLDOUT_SPLITS  # noqa: E402
from src.models.bsrnn import BSRNN_TFMAP, BSRNN_TFMAP_CONTEXT  # noqa: E402
from src.models.losses import LossBSRNN  # noqa: E402
from src.run_log import timed  # noqa: E402


def build_loss_fn(config):
    """Config -> loss object. Every optional term is opt-in, so an old config reproduces its own objective."""
    w = float(config["loss"]["w"])
    wm = float(config["loss"]["w_m"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    p = float(config["loss"]["p"])
    windows = config["loss"]["windows_ms"]
    sample_rate = int(config["data"]["sample_rate"])  # windows_ms -> n_fft needs it
    wg = float(config["loss"].get("w_g", 0.0))
    gain_delta_db = float(config["loss"].get("gain_delta_db", 3.0))
    w_struct = float(config["loss"].get("w_struct", 0.0))  # D17
    struct_floor_db = float(config["loss"].get("struct_floor_db", -40.0))

    # D10 interference weight. Needs the other speaker's stem, which is only loaded with both_directions or remix_gains.
    w_interf = float(config["loss"].get("w_interf", 1.0))
    assert w_interf > 0.0, f"loss.w_interf must be > 0, got {w_interf}"
    if w_interf != 1.0:
        assert config["data"].get("both_directions", False) or config["data"].get("remix_gains", False), "loss.w_interf != 1 needs data.both_directions or data.remix_gains, or the other speaker's stem is never loaded"

    # Artefact weight from AB-SDR (Ochiai et al., 2024). An energy weight: their alpha = 2 is w_artif = 4 (decisions-m2.md 2026-10-03).
    w_artif = float(config["loss"].get("w_artif", 1.0))
    assert w_artif > 0.0, f"loss.w_artif must be > 0, got {w_artif}"
    if w_artif != 1.0:
        assert config["data"].get("both_directions", False) or config["data"].get("remix_gains", False), "loss.w_artif != 1 needs data.both_directions or data.remix_gains, or the other speaker's stem is never loaded"

    # The D14 state-teacher arm was removed 2026-10-22 (git history has it). Refuse its keys rather than silently ignore them.
    assert not config["loss"].get("w_state") and not config.get("model", {}).get("state_head"), "loss.w_state / model.state_head: the state-teacher arm was removed 2026-10-22"

    # Presence gate, eq. (12) of docs/extra/variational-presence-gate.md. Imported here so runs without the gate never need it.
    if config.get("model", {}).get("presence_gate", False):
        from src.models.losses_presence import LossBSRNNPresence
        g = config["loss"]["presence"]
        return LossBSRNNPresence(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p, windows=windows, sample_rate=sample_rate, wg=wg, gain_delta_db=gain_delta_db, beta=float(g["beta"]), lam=float(g["lam"]), d_on=float(g["d_on"]), d_off=float(g["d_off"]))

    return LossBSRNN(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p, windows=windows, sample_rate=sample_rate, wg=wg, gain_delta_db=gain_delta_db, w_struct=w_struct, struct_floor_db=struct_floor_db, w_interf=w_interf, w_artif=w_artif)


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


def schedule_in_steps(config, steps_per_epoch):
    """(warmup, ramp, w_start, w_final) in steps, or None for a constant w.

    Accepts warmup_steps/ramp_steps, or the legacy warmup_epochs/ramp_epochs kept for runs up to 2026-09-01. Not both.
    """
    w_final = float(config["loss"]["w"])
    sched = config["loss"].get("w_schedule")
    if not sched:
        return None

    step_keys = sorted({"warmup_steps", "ramp_steps"} & set(sched))
    epoch_keys = sorted({"warmup_epochs", "ramp_epochs"} & set(sched))
    assert not (step_keys and epoch_keys), f"w_schedule mixes units: {step_keys} with {epoch_keys}. Use steps; epochs are only for reproducing runs up to 2026-09-01."

    w_start = float(sched.get("w_start", 0.0))
    assert 0.0 <= w_start <= 1.0, f"w_schedule.w_start must be in [0, 1], got {w_start}"

    if epoch_keys:
        warmup = int(sched.get("warmup_epochs", 0)) * steps_per_epoch
        ramp = int(sched.get("ramp_epochs", 0)) * steps_per_epoch
    else:
        warmup = int(sched.get("warmup_steps", 0))
        ramp = int(sched.get("ramp_steps", 0))
    return warmup, ramp, w_start, w_final


def w_at_step(config, global_step, steps_per_epoch):
    """Absent-branch weight for this step. global_step counts batches, including AMP-skipped ones, so it is reproducible."""
    resolved = schedule_in_steps(config, steps_per_epoch)
    if resolved is None:
        return float(config["loss"]["w"])
    warmup, ramp, w_start, w_final = resolved
    return _w_ramp(w_start, w_final, warmup, ramp, global_step)


def w_at_epoch(config, epoch):
    """Legacy epoch-indexed schedule, for runs up to 2026-09-01 and as the reference in tests/test_w_schedule.py."""
    sched = config["loss"].get("w_schedule") or {}
    assert not ({"warmup_steps", "ramp_steps"} & set(sched)), "w_at_epoch() is the legacy epoch-indexed entry point and cannot read a step-indexed w_schedule; call w_at_step()"
    return w_at_step(config, epoch, steps_per_epoch=1)  # one "step" per epoch is the old arithmetic


# Shared by the stdout rows and history.csv, so a log pasted from a killed run is a valid history.csv.
HISTORY_FIELDS = ["total", "L_pres", "L_MR", "L_gain", "L_abs", "L_state", "L_struct", "L_kl", "L_bce", "pi_mean", "L_pres_w", "interf_share", "artif_share", "n_present", "n_absent"]

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
    """Per-epoch table of every term, for stderr (stdout carries the CSV rows).

    gap = val - train, so a growing gap means memorising. If L_state falls while L_pres stalls, the frozen state teacher is
    being gamed (D14). Optional terms are skipped when they are NaN.
    """
    lines = [f"epoch {epoch + 1}/{num_epochs}  {epoch_seconds:.0f} s  lr {va['lr']:.2e}  w_trained {w_trained:.3f}",
             f"  {'term':<7} {'train':>10} {'val':>10} {'gap(val-train)':>15}"]
    optional = ("L_state", "L_struct", "L_kl", "L_bce", "pi_mean", "L_pres_w", "interf_share", "artif_share")
    for term in ("total", "L_pres", "L_MR", "L_gain", "L_abs") + optional:
        train_value, val_value = tr.get(term, float("nan")), va.get(term, float("nan"))
        if term in optional and not np.isfinite(train_value) and not np.isfinite(val_value):
            continue
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


def unwrap(model):
    """The module behind an nn.DataParallel wrapper. Use it for attributes and state_dict, so checkpoints stay interchangeable (E7)."""
    return model.module if isinstance(model, nn.DataParallel) else model


def build_model(config):
    """Config -> model. Absent optional keys give the architecture earlier runs trained."""
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
        presence_gate=bool(config["model"].get("presence_gate", False)),  # docs/extra/variational-presence-gate.md
        presence_mode=str(config["model"].get("presence_mode", "joint")),  # "postfilter": detached gate, decisions-m2.md 2026-10-18
        presence_floor=float(config["model"].get("presence_floor", 0.0)),
        tfmap_inject=bool(config["model"].get("tfmap_inject", False)),  # D4a: +37,698 params, not parameter-matched to its control
        tfmap_gate_init=float(config["model"].get("tfmap_gate_init", 0.0)),
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


def model_block_without_gate(config):
    """config["model"] minus the presence-gate keys, so e21's weights can seed a gated model (the gate is extra, not a change)."""
    block = dict(config.get("model") or {})
    for k in ("presence_gate", "presence_mode", "presence_floor"):
        block.pop(k, None)
    return block


def init_weights_record(config):
    """{"path", "md5"} of training.init_weights, or None. The md5 is logged so the starting weights are pinned."""
    path = (config.get("training") or {}).get("init_weights")
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training.init_weights: no file at {path}")
    return {"path": str(path), "md5": hashlib.md5(path.read_bytes()).hexdigest()}


def other_kwargs(loss_fn, batch, device):
    """{"s_other": the other speaker's stem} for a loss that takes it (D10), else {}."""
    if not getattr(loss_fn, "ACCEPTS_OTHER", False) or "other" not in batch:
        return {}
    return {"s_other": batch["other"].to(device)}


def gate_kwargs(loss_fn, batch, device, gate_logits, s_output_ungated=None):
    """Presence-gate inputs (logits, labels v, v_valid, and in "postfilter" mode the extractor's own output) for a loss that takes them, else {}."""
    if not getattr(loss_fn, "ACCEPTS_GATE", False):
        return {}
    kw = {"gate_logits": gate_logits.float(), "v": batch["v"].to(device), "v_valid": batch["v_valid"].to(device)}
    if s_output_ungated is not None:
        kw["s_output_ungated"] = s_output_ungated.float()
    return kw


def split_gate_output(out):
    """model(..., return_gate=True) -> (s_output, gate_logits, s_output_ungated or None). Two outputs in "joint" mode, three in "postfilter"."""
    if len(out) == 3:
        return out
    s_output, gate_logits = out
    return s_output, gate_logits, None


def add_parts(sums, counts, parts):
    """Accumulate each term weighted by the number of crops it was averaged over, so epoch means do not depend on the shuffle."""
    if parts["n_present"]:
        sums["L_pres"] += parts["L_pres"] * parts["n_present"]
        sums["L_MR"] += parts["L_MR"] * parts["n_present"]
        sums["L_gain"] += parts["L_gain"] * parts["n_present"]
        # "postfilter" mode logs the extractor's own output beside the gated one (losses_presence.py).
        for term in ("L_pres_ungated", "L_MR_ungated"):
            if not math.isnan(parts.get(term, float("nan"))):
                sums[term] += parts[term] * parts["n_present"]
        # D17 and D10 terms are present-only and NaN when off; NaN * n would poison the sum.
        for term in ("L_struct", "L_pres_w", "interf_share", "artif_share"):
            if not math.isnan(parts.get(term, float("nan"))):
                sums[term] += parts[term] * parts["n_present"]
        counts["present"] += parts["n_present"]
    if parts["n_absent"]:
        sums["L_abs"] += parts["L_abs"] * parts["n_absent"]
        if not math.isnan(parts.get("L_abs_ungated", float("nan"))):
            sums["L_abs_ungated"] += parts["L_abs_ungated"] * parts["n_absent"]
        counts["absent"] += parts["n_absent"]
    # L_state and the presence-gate terms cover every crop.
    if parts.get("L_state") is not None:
        n_all = parts["n_present"] + parts["n_absent"]
        sums["L_state"] += parts["L_state"] * n_all
        counts["all"] += n_all
    if parts.get("L_kl") is not None:
        n_all = parts["n_present"] + parts["n_absent"]
        for term in ("L_kl", "L_bce", "pi_mean"):
            sums[term] += parts[term] * n_all
        counts["gate"] += n_all


def oracle_mask_and_mag(model, target, mixture, clip=2.0):
    """Ideal mask |target| / |mixture| on the model's own STFT grid, clipped to [0, clip], and the mixture magnitude. No gradient."""
    with torch.no_grad():
        S = model.stft(target).abs()
        X = model.stft(mixture).abs()
        frames = min(S.shape[-1], X.shape[-1])
        S, X = S[..., :frames], X[..., :frames]
        return (S / X.clamp_min(1e-8)).clamp(0.0, clip), X


def epoch_report(sums, counts, w, wm, wg):
    """Accumulated sums -> epoch means.

    `total` uses the reporting w (the schedule's final value), so it is one objective across epochs (decisions-m2.md 2026-08-25).
    Optional terms are NaN when off and stay out of `total`, so runs before and after each term was added remain comparable.
    """
    nan = float("nan")
    n_present, n_absent = counts["present"], counts["absent"]
    L_pres = sums["L_pres"] / n_present if n_present else nan
    L_MR = sums["L_MR"] / n_present if n_present else nan
    L_gain = sums["L_gain"] / n_present if n_present else nan
    L_abs = sums["L_abs"] / n_absent if n_absent else nan
    n_all = counts.get("all", 0)
    L_state = sums["L_state"] / n_all if n_all else nan
    present_only = {t: (sums[t] / n_present if n_present and t in sums else nan) for t in ("L_struct", "L_pres_w", "interf_share", "artif_share")}
    n_gate = counts.get("gate", 0)
    gate_terms = {t: (sums[t] / n_gate if n_gate else nan) for t in ("L_kl", "L_bce", "pi_mean")}
    # The extractor alone, "postfilter" mode only; NaN otherwise, so older histories keep their columns' meaning.
    ungated = {t: (sums[t] / n_present if n_present and t in sums else nan) for t in ("L_pres_ungated", "L_MR_ungated")}
    ungated["L_abs_ungated"] = sums["L_abs_ungated"] / n_absent if n_absent and "L_abs_ungated" in sums else nan
    return {
        "total": (1 - w) * (L_pres + wm * L_MR + wg * L_gain) + w * L_abs,
        "L_pres": L_pres,
        "L_MR": L_MR,
        "L_gain": L_gain,
        "L_abs": L_abs,
        "L_state": L_state,
        **present_only,
        "n_present": n_present,
        "n_absent": n_absent,
        **gate_terms,
        **ungated,
    }


def lr_schedule_metric(val_loss, config):
    """The number ReduceLROnPlateau watches (training.lr_schedule_on, default `total`).

    `total` keeps falling as the model goes quiet, so new runs should use present_branch or content_wer (2026-09-21).
    Reuses selection_score's modes so the scheduler and the selector cannot drift apart.
    """
    mode = str(config["training"].get("lr_schedule_on", "total"))
    shim = {**config, "training": {**config["training"], "select_on": mode}}
    return selection_score(val_loss, shim)


def selection_score(val_loss, config):
    """The number that decides which epoch to keep (training.select_on). Lower is better.

      content_wer     smoothed in-loop validation WER; the only mode that tracks content (decisions-m2.md 2026-09-23)
      present_branch  the present half of the loss (default); silence is handled by selection_eligible (2026-08-30)
      total           pre-2026-08-30 behaviour; falls as the model goes quiet
      separation      L_pres alone; ignores what the model does on absent crops
    L_state is never selected on: the frozen teacher can be gamed (D14).
    """
    mode = str(config["training"].get("select_on", "present_branch"))
    if mode == "content_wer":
        # Smoothed, because per-epoch LCF-WER is too noisy to step a scheduler on. Raises rather than falling back.
        value = val_loss.get("content_wer_ema", val_loss.get("content_wer"))
        if value is None or value != value:
            raise ValueError("select_on/lr_schedule_on is `content_wer` but no validation WER was computed this epoch. Enable the probe with a `content_probe:` block, and set `every_n_epochs: 1` -- the scheduler needs a number every epoch.")
        return float(value)
    if mode == "total":
        return float(val_loss["total"])
    if mode == "present_branch":
        w_m = float(config["loss"]["w_m"])
        w_g = float(config["loss"].get("w_g", 0.0))
        return float(val_loss["L_pres"] + w_m * val_loss["L_MR"] + w_g * val_loss.get("L_gain", 0.0))
    if mode == "separation":
        return float(val_loss["L_pres"])
    raise ValueError(f"training.select_on: unknown mode {mode!r}. Known: content_wer, present_branch, total, separation.")


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
    core = unwrap(model)  # weights are saved, and the STFT read, from the real module
    val_loss_history = []
    train_loss_history = []

    if config is None:
        raise ValueError("Config must be provided to build the loss function.")
    resumed_wer_history = list(resumed_wer_history or [])

    loss_fn = build_loss_fn(config)
    # log_struct logs L_struct without training on it (D17). The presence gate cannot also return the mask, so gated runs skip it.
    log_struct = bool(config["loss"].get("log_struct", True))
    gate_on = getattr(core, "presence_gate", None) is not None
    want_mask = (getattr(loss_fn, "w_struct", 0.0) > 0.0 or log_struct) and not gate_on
    grad_clip = float(config["training"]["grad_clip"])
    patience = int(config["training"]["patience"])
    keep_top_k = int(config["training"].get("keep_top_k", 3))
    keep_stride = int(config["training"].get("keep_stride", 0))
    kept = []  # (score, epoch) of the checkpoints still on disk
    print(f"  selecting on `{config['training'].get('select_on', 'present_branch')}`, silence bar L_abs <= {config['training'].get('select_abs_max', 'none')}, keeping top {keep_top_k}")
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
    resolved = schedule_in_steps(config, steps_per_epoch)
    if resolved is not None:
        warmup_steps, ramp_steps, _, _ = resolved
        full_at = warmup_steps + ramp_steps
        print(f"w schedule: {config['loss']['w_schedule']}", flush=True)
        print(f"  {steps_per_epoch} steps/epoch  warmup {warmup_steps} steps  ramp {ramp_steps} steps  full w at step {full_at} (epoch {full_at / max(steps_per_epoch, 1):.2f})", flush=True)
        shown = min(num_epochs, math.ceil(full_at / max(steps_per_epoch, 1)) + 1)  # only up to the end of the ramp
        ws = [w_at_step(config, e * steps_per_epoch, steps_per_epoch) for e in range(shown)]
        print(f"  w at each epoch START: {[round(v, 4) for v in ws]}", flush=True)
        print(f"  reporting/selection w held at {w_report}", flush=True)

    # Mixed precision, CUDA only. With enabled=False the scaler's calls are no-ops, so one code path serves both precisions.
    use_amp = bool(config["training"].get("amp", False)) and str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  mixed precision: {'ON (fp16 forward, fp32 loss)' if use_amp else 'off'}", flush=True)

    global_step = int(start_step)  # carried across a resume so the w warmup does not restart

    for epoch in range(start_epoch, num_epochs):
        train_loader.dataset.set_epoch(epoch)  # new crops each epoch
        if gate_on and core.presence_mode == "joint":
            # Temperature of the soft draw, eq. (20): linear from tau_start to tau_end over the run.
            # Annealing tau follows Jang et al. (2017); the linear per-epoch schedule is ours.
            # NOTE it is spread over --epochs, so a --resume restarts it (seen 2026-10-17/18). "postfilter" has no draw.
            g = config["loss"]["presence"]
            frac = epoch / max(num_epochs - 1, 1)
            core.gate_tau = g["tau_start"] + frac * (g["tau_end"] - g["tau_start"])

        model.train()
        sums, counts = defaultdict(float), defaultdict(int)
        w_sum, w_steps = 0.0, 0  # for the mean w that trained this epoch
        epoch_start = time.time()
        for batch in train_loader:
            loss_fn.w = w_at_step(config, global_step, steps_per_epoch)
            w_sum += loss_fn.w
            w_steps += 1

            mixture, target, enrollment, crop_absent = unpack(batch, device)
            if hasattr(loss_fn, "enrolment_embedding"):  # the state teacher needs the enrolment
                loss_fn.enrolment_embedding = loss_fn.teacher.embed_enrolment(enrollment)
            ctx = context_kwargs(context_encoder, enrollment)

            optimizer.zero_grad()
            gate_logits = s_ungated = None
            with amp_ctx(use_amp):
                if gate_on:
                    s_output, gate_logits, s_ungated = split_gate_output(model(mixture, enrollment, return_gate=True, **ctx))
                    mask = None
                elif want_mask:
                    s_output, mask = model(mixture, enrollment, return_mask=True, **ctx)
                else:
                    s_output, mask = model(mixture, enrollment, **ctx), None
            oracle, mix_mag = oracle_mask_and_mag(core, target, mixture) if want_mask else (None, None)
            # Reference first, then output (see LossBSRNN). The loss runs in fp32.
            loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent, mask=None if mask is None else mask.float(), oracle_mask=oracle, mixture_mag=mix_mag, **other_kwargs(loss_fn, batch, device), **gate_kwargs(loss_fn, batch, device, gate_logits, s_ungated))
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

        # Validation, in the same precision as training. In eval() the gate multiplies by pi, eq. (21).
        model.eval()
        val_sums, val_counts = defaultdict(float), defaultdict(int)
        diag = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                mixture, target, enrollment, crop_absent = unpack(batch, device)
                if hasattr(loss_fn, "enrolment_embedding"):
                    loss_fn.enrolment_embedding = loss_fn.teacher.embed_enrolment(enrollment)
                ctx = context_kwargs(context_encoder, enrollment)

                gate_logits = s_ungated = None
                with amp_ctx(use_amp):
                    if gate_on:
                        s_output, gate_logits, s_ungated = split_gate_output(model(mixture, enrollment, return_gate=True, **ctx))
                        mask = None
                    elif want_mask:
                        s_output, mask = model(mixture, enrollment, return_mask=True, **ctx)
                    else:
                        s_output, mask = model(mixture, enrollment, **ctx), None
                s_output = s_output.float()
                oracle, mix_mag = oracle_mask_and_mag(core, target, mixture) if want_mask else (None, None)
                _, parts = loss_fn(target, s_output, mixture, crop_absent, mask=None if mask is None else mask.float(), oracle_mask=oracle, mixture_mag=mix_mag, **other_kwargs(loss_fn, batch, device), **gate_kwargs(loss_fn, batch, device, gate_logits, s_ungated))
                add_parts(val_sums, val_counts, parts)
                diagnostic_accumulate(diag, model, mixture, enrollment, s_output, crop_absent, amp=use_amp, enrol_embedding=ctx.get("enrol_embedding"))

        val_loss = epoch_report(val_sums, val_counts, w_report, loss_fn.wm, loss_fn.wg)
        val_loss.update(diagnostic_report(diag))
        # Content probe: whole clips with reference text, which the 4 s validation crops do not have.
        if content_probe is not None and epoch % probe_every == 0:
            with torch.no_grad():
                val_loss["content_wer"] = content_probe.score(live_extractor(core, context_encoder, device))
            val_loss["content_wer_ema"] = content_probe.smoothed[-1]
        val_loss["epoch"] = epoch
        val_loss["lr"] = optimizer.param_groups[0]["lr"]
        val_loss["w"] = w_trained
        val_loss_history.append(val_loss)

        # One CSV row per epoch on stdout (flushed, so a killed Kaggle session keeps it), and the readable table on stderr.
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
                "model": core.state_dict(),
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
            torch.save({"model": core.state_dict(), "epoch": epoch, "score": score, "eligible": eligible, "row": val_loss, "config": config, "seed": config["seed"]}, rank_path)
            for dropped in checkpoints_to_drop(kept, keep_top_k, keep_stride, last_epoch=epoch):
                Path(save_path).with_name(f"{Path(save_path).stem}_e{dropped:03d}.pt").unlink(missing_ok=True)
                kept[:] = [(sc, ep) for sc, ep in kept if ep != dropped]

        if eligible and score < best_val:
            best_val = score
            best_row = val_loss
            epochs_since_best = 0
            if save_path:
                torch.save({
                    "model": core.state_dict(),
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


# split -> ((train_manifest, train_audio_dir), (val_manifest, val_audio_dir)). Manifest and audio dir are separate because a
# subset manifest (e.g. `mid`) reads audio rendered for another split.
SPLIT_MANIFESTS = {
    "smoke": (("smoke_train", "smoke_train"), ("smoke_val", "smoke_val")),
    "mid":   (("mid_train",   "train"),       ("mid_val",   "val")),
    "sir0":  (("sir0_train",  "sir0_train"),  ("sir0_val",  "sir0_val")),  # symmetric target/interferer loudness
    # Render-only from here down to amitrchk, except amift and amimix: each validates on a holdout or test split.
    "sir0ext": (("sir0_train", "sir0_train"), ("sir0_privval", "sir0_privval")),  # expanded dev set (decisions-m3.md 2026-09-13)
    "sir0pub": (("sir0_train", "sir0_train"), ("eval_public", "eval_public")),  # public test split (decisions-m3.md 2026-09-24)
    "sir0priv": (("sir0_train", "sir0_train"), ("eval_private", "eval_private")),  # scored once, after selection (decisions-m4.md 2026-10-07)
    "ami":   (("sir0_train", "sir0_train"), ("ami_eval",    "ami_eval")),  # AMI transfer set, eval only
    # M8 AMI fine-tunes (decisions-m8.md 2026-10-12). Rows carry their own audio_dir; val stays on sir0_val as a forgetting monitor.
    "amift":  (("amift_train",  "ami_train"), ("sir0_val",  "sir0_val")),
    "amimix": (("amimix_train", "ami_train"), ("sir0_val",  "sir0_val")),
    "amitr":  (("ami_train",    "ami_train"), ("ami_train", "ami_train")),  # the ami_train windows as Gemini hears them
    "amitrchk": (("ami_train_check", "ami_train_check"), ("ami_train_check", "ami_train_check")),  # the 150-window subset Gemini already scored
    "full":  (("train",       "train"),       ("val",       "val")),
}
RENDER_ONLY_SPLITS = frozenset({"sir0pub", "sir0priv", "ami", "sir0ext", "amitr", "amitrchk"})


def get_data_loaders(split, csv_path, data_path, config):
    if split not in SPLIT_MANIFESTS:
        raise ValueError(f"Unknown split: {split}. Known: {sorted(SPLIT_MANIFESTS)}")
    if split in RENDER_ONLY_SPLITS:
        raise ValueError(f"split {split!r} is for rendering estimates only; training on it would select checkpoints on {SPLIT_MANIFESTS[split][1][0]}.")
    (train_manifest, train_audio), (val_manifest, val_audio) = SPLIT_MANIFESTS[split]
    both_directions = bool(config["data"].get("both_directions", False))  # each trial trained once per speaker (decisions-m2.md 2026-08-26)
    enrollment_variants = int(config["data"].get("enrollment_variants", 1))  # train only; val stays on enrollment.wav (2026-08-30)
    remix_gains = bool(config["data"].get("remix_gains", False))  # per-epoch SIR/SNR re-draw, train only

    def labels_for(audio):
        """Presence-gate labels v_t from data/index/state_<split>.csv, or {} when the gate is off."""
        if not config["model"].get("presence_gate", False):
            return {}
        stft = config["model"]["stft"]
        return dict(state_index=Path(config["data"]["state_index_dir"]) / f"state_{audio}.csv", label_n_fft=stft["n_fft"], label_hop=stft["hop"])

    # chunk_s = 4 s follows CARTSE (Li & Seki, 2026).
    train_dataset = TrialDataset(manifest_csv=csv_path / f"{train_manifest}.csv", data_root=data_path, split=train_audio, chunk_s=config["data"]["chunk_s"], sample_rate=config["data"]["sample_rate"], seed=config["seed"], both_directions=both_directions, enrollment_variants=enrollment_variants, remix_gains=remix_gains, **labels_for(train_audio))
    val_dataset = TrialDataset(manifest_csv=csv_path / f"{val_manifest}.csv", data_root=data_path, split=val_audio, chunk_s=config["data"]["chunk_s"], sample_rate=config["data"]["sample_rate"], seed=config["seed"], random_crop=False, both_directions=both_directions, **labels_for(val_audio))

    # num_workers 0 suits the laptop; on a GPU the wav reads need workers. Pinned memory lets host-to-GPU copies run async.
    num_workers = int(config["data"].get("num_workers", 0))
    extra = dict(persistent_workers=True, prefetch_factor=4) if num_workers else {}
    extra["pin_memory"] = torch.cuda.is_available()

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True, num_workers=num_workers, drop_last=True, collate_fn=collate_pairs, **extra)
    val_loader = DataLoader(val_dataset, batch_size=config["data"]["batch_size"], shuffle=False, num_workers=num_workers, collate_fn=collate_pairs, **extra)
    return train_loader, val_loader


def git_commit():
    """HEAD, with -dirty if the tree has changes. Inside the Kaggle bundle (no .git) it reads the hash stamped at build time."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        stamp = Path(__file__).resolve().parents[1] / "docs/bundle_commit.txt"
        try:
            return "bundle:" + stamp.read_text().strip()
        except OSError:
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
            "init_weights": init_weights_record(config),  # None when trained from scratch
            "model": {
                "class": type(unwrap(model)).__name__,
                "n_parameters": sum(p.numel() for p in unwrap(model).parameters()),
                "n_bands": len(unwrap(model).band_widths),
            },
            "epochs_requested": num_epochs,
            "epochs_run": epochs_run,
            "early_stopped": bool(epochs_run and last_epoch + 1 < num_epochs),
            "wall_seconds": round(wall_s, 1),
            "seconds_per_epoch": round(wall_s / epochs_run, 1) if epochs_run else None,
            # Copied so the result is readable without the yaml; the Kaggle notebook rewrites batch_size and num_workers.
            "data": config["data"],
            "loss": config["loss"],
            "training": config["training"],
            "best_val": {k: (best_row or {}).get(k) for k in ["epoch", "total", "L_pres", "L_MR", "L_abs", "lr"]},
            "selection": {
                "select_on": config["training"].get("select_on", "present_branch"),
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
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--outdir", default="models/")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--results-dir", default=None, help="default experiments/results/<today>-train-<split>")
    args = ap.parse_args()

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
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

    # Optional DataParallel across both Kaggle T4s (E7). It buys batch room, not a wider model; the loss still sees the gathered batch.
    n_gpu = torch.cuda.device_count()
    use_dp = bool(config["training"].get("data_parallel", False)) and n_gpu > 1
    if use_dp:
        model = nn.DataParallel(model)
        print(f"DataParallel across {n_gpu} GPUs, batch {config['data']['batch_size']} split {config['data']['batch_size'] // n_gpu} per card")
    elif bool(config["training"].get("data_parallel", False)):
        print(f"WARNING: data_parallel requested but device_count() == {n_gpu}; running on one device", file=sys.stderr)
    core = unwrap(model)

    # training.freeze_extractor: train the presence gate ALONE on a fixed extractor (decisions-m2.md 2026-10-18). The
    # extractor's weights get no gradient, so no autograd graph is built for them: the extractor is then e21 exactly,
    # and gate-on vs e21 is a paired comparison on identical extractor weights. Only meaningful with a gate.
    if bool(config["training"].get("freeze_extractor", False)):
        if getattr(core, "presence_gate", None) is None:
            raise ValueError("training.freeze_extractor needs model.presence_gate: there would be nothing left to train")
        gate_ids = {id(p) for p in core.presence_gate.parameters()}
        n_frozen = 0
        for p in core.parameters():
            if id(p) not in gate_ids:
                p.requires_grad_(False)
                n_frozen += p.numel()
        print(f"  extractor FROZEN ({n_frozen:,} weights); training the gate's {sum(p.numel() for p in core.presence_gate.parameters())} weights only", flush=True)


    optimizer = torch.optim.AdamW([p for p in core.parameters() if p.requires_grad], lr=float(config["training"]["lr"]), weight_decay=float(config["training"]["weight_decay"]))  # AdamW (Loshchilov & Hutter, 2019)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=float(config["training"]["lr_factor"]), patience=int(config["training"]["lr_patience"]))

    save_path = Path(args.outdir) / f"model_{args.split}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, best_row, start_step = 0, float("inf"), None, 0
    resumed_wer_history = []

    # training.init_weights: a new run from another run's weights, with a fresh optimiser (D10, 2026-10-02). Skipped on --resume.
    init_record = init_weights_record(config)
    if init_record and not args.resume:
        source = torch.load(init_record["path"], map_location=device, weights_only=False)
        if model_block_without_gate(source.get("config") or {}) != model_block_without_gate(config):
            raise ValueError(f"{init_record['path']} was trained with a different model block; init_weights needs the same architecture (the presence-gate keys may differ)")
        # A gate-less source (e21) into a gated model: only the gate's own weights may be missing; they keep their
        # seeded init (PresenceGate.init_seed). Nothing else may be missing or unexpected.
        missing, unexpected = core.load_state_dict(source["model"], strict=False)
        if unexpected or any(not k.startswith("presence_gate.") for k in missing):
            raise ValueError(f"init_weights mismatch: missing {missing}, unexpected {unexpected}")
        print(f"initialised from {init_record['path']} (md5 {init_record['md5']}), fresh optimiser at lr {config['training']['lr']}"
              + (f"; gate weights left at their seeded init ({len(missing)} tensors)" if missing else ""), flush=True)

    if args.resume:
        if not save_path.exists():
            raise FileNotFoundError(f"--resume but no checkpoint at {save_path}")
        ckpt = torch.load(save_path, map_location=device, weights_only=False)  # our own file: it carries the config dict
        core.load_state_dict(ckpt["model"])
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
               rate=lambda: f"batch {config['data']['batch_size']}, {device}{f' x{n_gpu} (DataParallel)' if use_dp else ''}, {(time.time() - t0) / max(ran['epochs'], 1):.0f} s/epoch"):
        train_loss_history, val_loss_history, best_row = train(model, train_loader, val_loader, optimizer, num_epochs=num_epochs, device=device, print_debug=True, save_path=save_path, config=config, scheduler=scheduler, context_encoder=context_encoder, start_epoch=start_epoch, best_val=best_val, best_row=best_row, start_step=start_step, resumed_wer_history=resumed_wer_history)
        ran["epochs"] = len(train_loss_history)
    wall_s = time.time() - t0

    results_dir = Path(args.results_dir) if args.results_dir else Path("experiments/results") / f"{date.today().isoformat()}-train-{args.split}"
    train_manifest = csv_path / f"{SPLIT_MANIFESTS[args.split][0][0]}.csv"
    log_results(results_dir, config, config_path, args, model, device, train_manifest, train_loss_history, val_loss_history, best_row, wall_s, num_epochs, save_path)
    plot_history(results_dir, train_loss_history, val_loss_history, best_row, loss_floor=total_loss_floor(config))


if __name__ == "__main__":
    main()
