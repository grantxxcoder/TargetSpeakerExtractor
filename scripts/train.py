import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import contextlib
import math
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
import matplotlib
# Agg BEFORE pyplot: the backend is fixed at import, and Kaggle is headless.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# `python scripts/train.py` puts scripts/ on sys.path, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
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
    # windows_ms -> n_fft needs this, so it cannot be defaulted inside the loss
    sample_rate = int(config["data"]["sample_rate"])
    # .get(), so a pre-2026-08-27 config still loads and trains its own objective.
    wg = float(config["loss"].get("w_g", 0.0))
    gain_delta_db = float(config["loss"].get("gain_delta_db", 3.0))

    # Convex weight: a typo of 4.58 for 0.458 makes (1 - w) negative, training
    # the model to destroy the target while the curve still looks like it falls.
    assert 0.0 <= w <= 1.0, f"loss.w must be in [0, 1], got {w}"
    assert wm >= 0.0, f"loss.w_m must be >= 0, got {wm}"
    assert wg >= 0.0, f"loss.w_g must be >= 0, got {wg}"
    # A negative deadzone punishes a PERFECT match -- reads as a dead term.
    assert gain_delta_db >= 0.0, f"loss.gain_delta_db must be >= 0, got {gain_delta_db}"

    return LossBSRNN(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p, windows=windows,
                     sample_rate=sample_rate, wg=wg, gain_delta_db=gain_delta_db)


# ---------------------------------------------------------------------------
# THE ABSENT-BRANCH WEIGHT SCHEDULE
#
# WHY A SCHEDULE AT ALL. decisions-m2.md 2026-08-25. On the 2-epoch `mid` run
# the model had muted to -18.5 dB by epoch 1, enrolment sensitivity -14.31 dB:
# it learned silence before conditioning. Going quiet is worth ~9 loss units
# immediately; using the enrolment is slow and earns nothing for level (L_pres
# is scale-invariant). Weights cannot fix it -- w_m would need ~243, and
# tau_abs is inert. The lever is WHEN the absent branch turns on. At w = 0
# silence pays nothing, so the only way down is to reconstruct the target,
# which on `both` crops needs the enrolment.
#
# WHY INDEXED IN GRADIENT STEPS, since 2026-09-03. The schedule used to be
# indexed in EPOCHS, so its length in optimiser steps moved with the size of
# the training set: warmup 4 + ramp 3 epochs is 11,606 steps at 4,976 trials
# but 23,212 at 9,955 -- a silent 2x lengthening of the one knob that stops the
# early mute. Two runs at different data volumes were therefore not running the
# same schedule, which is precisely the confound the data-scaling curve
# (1,989 -> 4,976 -> 9,955) exists to measure. Steps are the unit the optimiser
# moves in, so a step-indexed schedule is invariant to dataset size.
# decisions-m2.md 2026-09-03.
# ---------------------------------------------------------------------------


def _w_ramp(w_start, w_final, warmup, ramp, t):
    """The schedule's SHAPE, in whatever unit `warmup`, `ramp` and `t` share.

        t <  warmup                  -> w_start
        warmup <= t < warmup + ramp  -> linear w_start -> w_final
        t >= warmup + ramp           -> w_final

    Split out on 2026-09-03 so the step-indexed schedule and the legacy
    epoch-indexed one are provably the SAME CURVE, differing only in the unit
    of `t`. Do not inline it.
    """
    if t < warmup:
        return w_start
    if ramp <= 0 or t >= warmup + ramp:
        return w_final
    # +1 so the LAST unit of the ramp reaches w_final rather than stopping one
    # increment short of it.
    frac = (t - warmup + 1) / ramp
    return w_start + frac * (w_final - w_start)


def schedule_in_steps(config, steps_per_epoch):
    """(warmup_steps, ramp_steps, w_start, w_final), or None for a constant w.

    THE ONE PLACE the config is converted into steps, so the training loop, the
    startup print and the tests cannot disagree about where the ramp ends.

    Two accepted forms, and mixing them is refused:

      warmup_steps / ramp_steps    -- current. Invariant to dataset size.
      warmup_epochs / ramp_epochs  -- LEGACY. Multiplied by this run's
                                      steps_per_epoch here, which IS the bug
                                      described above. Kept only so every run
                                      up to 2026-09-01 reproduces exactly; a
                                      new run should not use it.
    """
    w_final = float(config["loss"]["w"])
    sched = config["loss"].get("w_schedule")
    if not sched:
        return None

    step_keys = sorted({"warmup_steps", "ramp_steps"} & set(sched))
    epoch_keys = sorted({"warmup_epochs", "ramp_epochs"} & set(sched))
    assert not (step_keys and epoch_keys), (
        f"w_schedule mixes units: {step_keys} with {epoch_keys}. Pick one -- "
        f"steps are the current form, epochs are kept only for reproducing "
        f"runs up to 2026-09-01.")

    w_start = float(sched.get("w_start", 0.0))
    assert 0.0 <= w_start <= 1.0, f"w_schedule.w_start must be in [0, 1], got {w_start}"

    if epoch_keys:
        assert steps_per_epoch > 0, (
            "an epoch-indexed w_schedule needs steps_per_epoch > 0 to convert")
        warmup = int(sched.get("warmup_epochs", 0)) * steps_per_epoch
        ramp = int(sched.get("ramp_epochs", 0)) * steps_per_epoch
    else:
        warmup = int(sched.get("warmup_steps", 0))
        ramp = int(sched.get("ramp_steps", 0))

    assert warmup >= 0 and ramp >= 0, "w_schedule warmup/ramp must be >= 0"
    return warmup, ramp, w_start, w_final


def w_at_step(config, global_step, steps_per_epoch):
    """The absent-branch weight for this optimiser step.

    `global_step` counts BATCHES since the start of training, carried across a
    resume. Batches and not *successful* optimiser updates on purpose: AMP's
    GradScaler skips a step whose gradients hold inf/NaN, and a schedule that
    moved with those skips would not be reproducible from the config and seed
    alone.

    Absent `w_schedule`, returns loss.w for every step, so an unscheduled
    config behaves exactly as before.
    """
    resolved = schedule_in_steps(config, steps_per_epoch)
    if resolved is None:
        return float(config["loss"]["w"])
    warmup, ramp, w_start, w_final = resolved
    return _w_ramp(w_start, w_final, warmup, ramp, global_step)


def w_at_epoch(config, epoch):
    """LEGACY, epoch-indexed. Reproduces every run up to 2026-09-01.

    The training loop no longer calls this -- it calls w_at_step(). Kept
    because decisions-m2.md 2026-08-25 cites it by name, and because it is the
    reference the step-indexed schedule is checked against in
    tests/test_w_schedule.py. Only meaningful for a config that uses
    warmup_epochs / ramp_epochs.
    """
    sched = config["loss"].get("w_schedule") or {}
    assert not ({"warmup_steps", "ramp_steps"} & set(sched)), (
        "w_at_epoch() is the legacy epoch-indexed entry point and cannot read a "
        "step-indexed w_schedule -- call w_at_step() instead.")
    # steps_per_epoch=1 makes one "step" one epoch, which is exactly the old
    # arithmetic.
    return w_at_step(config, epoch, steps_per_epoch=1)


# One definition, used by both the stdout line and history.csv -- so a log
# pasted out of a killed run is a valid history.csv with no editing.
HISTORY_FIELDS = ["total", "L_pres", "L_MR", "L_gain", "L_abs", "n_present", "n_absent"]

# VAL-ONLY leading indicators; the loss terms are lagging ones.
#   enrol_sens_db    output movement on an enrolment swap. Near 0 dB = strongly
#                    conditioned; very negative = ignoring the enrolment.
#   pres_abs_gap_db  output loudness, present crops minus absent. Large
#                    positive = it knows when to speak.
# Until 2026-08-27 no loss term could show a mute, and the 2026-08-24 smoke run
# collapsed to one with a healthy-looking curve throughout. L_gain now prices it
# directly, but read these first: L_gain says the level is wrong,
# pres_abs_gap_db says whether the correction was SELECTIVE. Fixing level by
# turning everything up scores well on L_gain and leaves this flat -- that is a
# pass-through, not an extractor.
VAL_DIAGNOSTICS = ["enrol_sens_db", "pres_abs_gap_db"]


def history_header():
    # `w` is the weight that actually TRAINED this epoch; `total` is always at
    # the final w (see epoch_report). Without this column a reader cannot tell a
    # real improvement from a schedule step.
    return (["epoch"] + [f"train_{k}" for k in HISTORY_FIELDS]
            + [f"val_{k}" for k in HISTORY_FIELDS] + ["lr", "w"]
            + [f"val_{k}" for k in VAL_DIAGNOSTICS])


def history_row(tr, va):
    """One row. Epoch comes from the VAL dict, not enumerate(), so a resume does
    not relabel epoch 40 as 0. .get on the diagnostics: hand-built val rows (the
    tests, older histories) lack them, and that must not kill the row."""
    return ([va["epoch"]] + [tr[k] for k in HISTORY_FIELDS]
            + [va[k] for k in HISTORY_FIELDS] + [va["lr"], va.get("w", float("nan"))]
            + [va.get(k, float("nan")) for k in VAL_DIAGNOSTICS])


def format_epoch_breakdown(epoch, num_epochs, tr, va, epoch_seconds, w_trained):
    """The per-epoch term breakdown, for STDERR. Replaced tqdm on 2026-08-31.

    Why this shape. Progress bars emitted one line per batch, which at 1,666
    batches an epoch buried the only output that matters. This prints once per
    epoch instead, and shows the thing the run is actually being judged on.

    `gap` is val minus train, so it is POSITIVE when the model does worse on
    audio it has not seen, and GROWING gap = memorising. That is the number to
    watch, not `total`: in the 2026-08-29 run both totals fell the whole way
    down while held-out separation collapsed below pass-through
    (decisions-m2.md 2026-08-29). L_pres is negated SI-SDR, so a train L_pres of
    -5.51 against a val +0.17 is the 5.68 dB gap that run ended with.

    Goes to stderr on purpose: stdout carries one CSV row per epoch and must
    stay a valid history.csv so a killed Kaggle session can be recovered by
    pasting it into a file. See scripts/make_kaggle_notebook.py.
    """
    lines = [
        f"epoch {epoch + 1}/{num_epochs}  {epoch_seconds:.0f} s  "
        f"lr {va['lr']:.2e}  w_trained {w_trained:.3f}",
        f"  {'term':<7} {'train':>10} {'val':>10} {'gap(val-train)':>15}",
    ]
    for term in ("total", "L_pres", "L_MR", "L_gain", "L_abs"):
        train_value, val_value = tr[term], va[term]
        lines.append(f"  {term:<7} {train_value:>10.4f} {val_value:>10.4f} "
                     f"{val_value - train_value:>15.4f}")
    lines.append(f"  crops   train {tr['n_present']} present / {tr['n_absent']} absent"
                 f"   val {va['n_present']} / {va['n_absent']}")
    lines.append(f"  diag    enrol_sens {va.get('enrol_sens_db', float('nan')):.2f} dB"
                 f"   pres_abs_gap {va.get('pres_abs_gap_db', float('nan')):.2f} dB")
    return "\n".join(lines)


def diagnostic_accumulate(diag, model, mixture, enrollment, s_output, crop_absent,
                          amp=False):
    """Accumulate the two leading indicators over one val batch.

    Costs one extra val forward per epoch. Rolls within the batch rather than
    shuffling globally. Skipped at batch 1, where roll() returns the same
    enrolment and would read a false 0 dB.
    """
    if mixture.shape[0] > 1:
        # Forward in fp16 when training does, but .float() IMMEDIATELY: the sums
        # below are sums of squares over 64k samples and would overflow fp16's
        # 65504 ceiling, silently turning the diagnostic into inf.
        with amp_ctx(amp):
            y_swapped = model(mixture, enrollment.roll(1, 0))
        y_swapped = y_swapped.float()
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
    """Accumulators -> the two logged numbers. NaN when a half was never seen,
    matching how the loss terms report a missing half."""
    nan = float("nan")
    sens = (10 * np.log10(diag["swap_num"] / diag["swap_den"])
            if diag["swap_den"] > 0 and diag["swap_num"] > 0 else nan)
    gap = (diag["e_pres"] / diag["n_pres"] - diag["e_abs"] / diag["n_abs"]
           if diag["n_pres"] and diag["n_abs"] else nan)
    return {"enrol_sens_db": sens, "pres_abs_gap_db": gap}


def total_loss_floor(config):
    """Best total the objective can reach, for the reference line on the plot.

    w-weighted sum of 10log10(tau_pres) and 10log10(tau_abs). Since the
    2026-08-25 tau split no single tau defines it. w_g does not appear: L_gain
    is 0 at perfect reconstruction, so only tau_pres, tau_abs and w move it.
    """
    w = float(config["loss"]["w"])
    tau_pres = float(config["loss"]["tau_pres"])
    tau_abs = float(config["loss"]["tau_abs"])
    return ((1 - w) * 10 * np.log10(tau_pres) + w * 10 * np.log10(tau_abs))


def build_model(config):
    """Config -> BSRNN_TFMAP. Every ctor argument comes from the yaml.

    Separate from main() so measure_train_cost.py measures the model that
    actually trains. Two keys deliberately not passed: separator.norm (implied
    by causal=True) and n_hidden (ctor default 1). Both belong in the yaml.
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
        # Without this the config key is dead: BSRNN_TFMAP's own default (16.0)
        # would win and editing the yaml would change nothing.
        #
        # Missing key => a checkpoint saved BEFORE 2026-08-25, when TFMap had no
        # logit scale at all, i.e. an effective scale of 1.0. Those weights must
        # be reloaded at 1.0: defaulting to sqrt(F) would run them against a cue
        # they never saw in training, and every diagnostic on them would be
        # measuring a model that never existed. Loud, because silently reviving
        # the flat softmax on a NEW run is the bug this whole file is about.
        tfmap_scale=_tfmap_scale(config),
    )


def _tfmap_scale(config):
    if "tfmap_scale" in config["model"]:
        return float(config["model"]["tfmap_scale"])
    print("WARNING: config has no model.tfmap_scale -- assuming 1.0, the "
          "pre-2026-08-25 behaviour. Correct for an old checkpoint, WRONG for "
          "a new run: add the key to the config.", file=sys.stderr)
    return 1.0


def amp_ctx(enabled):
    """fp16 autocast, or a no-op. ONE definition so train, val and the
    diagnostic cannot drift into different precisions.

    Scope is deliberately the MODEL FORWARD ONLY. LossBSRNN carries 1e-12
    epsilons inside log10 and divisions; fp16's smallest normal is ~6e-5, so
    those underflow to zero and the loss returns inf/NaN. Every caller casts the
    model output back with .float() before the loss sees it.
    """
    return (torch.amp.autocast("cuda", dtype=torch.float16)
            if enabled else contextlib.nullcontext())


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

    NOT loss.item() * batch_size: the two halves are means over subsets whose
    sizes vary per batch, so that weighting makes the epoch number move when
    only the shuffle changes. Gated on counts, never isnan() -- a NaN from a
    real numerical failure must still reach the log.
    """
    if parts["n_present"]:
        sums["L_pres"] += parts["L_pres"] * parts["n_present"]
        sums["L_MR"] += parts["L_MR"] * parts["n_present"]
        sums["L_gain"] += parts["L_gain"] * parts["n_present"]
        counts["present"] += parts["n_present"]
    if parts["n_absent"]:
        sums["L_abs"] += parts["L_abs"] * parts["n_absent"]
        counts["absent"] += parts["n_absent"]


def epoch_report(sums, counts, w, wm, wg):
    """Recombine the accumulated terms.

    `w` is the REPORTING w (loss.w, the schedule's final value), never the w
    that trained this epoch -- otherwise `total` is a different objective each
    epoch and the curve falls with the schedule alone, corrupting both
    ReduceLROnPlateau and best-checkpoint selection. The training w is its own
    column. decisions-m2.md 2026-08-25.
    """
    n_present, n_absent = counts["present"], counts["absent"]
    L_pres = sums["L_pres"] / n_present if n_present else float("nan")
    L_MR = sums["L_MR"] / n_present if n_present else float("nan")
    L_gain = sums["L_gain"] / n_present if n_present else float("nan")
    L_abs = sums["L_abs"] / n_absent if n_absent else float("nan")

    return {
        "total": (1 - w) * (L_pres + wm * L_MR + wg * L_gain) + w * L_abs,
        "L_pres": L_pres,
        "L_MR": L_MR,
        "L_gain": L_gain,
        "L_abs": L_abs,
        "n_present": n_present,
        "n_absent": n_absent,
    }


def selection_score(val_loss, config):
    """The number that decides which epoch's weights we KEEP. Not a loss.

    WHY THIS IS NOT `val_total`. The training objective and the model-selection
    rule are different jobs. `w` = 0.458 was derived from the absent-crop rate
    (CARTSE's eta) to balance GRADIENTS between the two branches; it was never
    derived to rank finished models, and used that way it ranks them badly.

    Measured 2026-08-30 across three runs: at control epoch 14 the absent branch
    contributes 0.458 x -11.699 = -5.358 to `val_total` while the whole present
    branch contributes +3.180, so the total keeps falling as the model gets
    quieter on absent crops long after separation has stopped improving. On the
    remix arm that cost a real model -- `val_total` kept epoch 14 at 1.13 dB
    held-out separation when epoch 10 of the same run reached 2.36 dB.

    `present_branch` is the same combination the loss applies to target-present
    crops, with the absent branch removed rather than reweighted. Silence is
    handled by an eligibility bar (see `selection_eligible`) rather than by a
    second arbitrary exchange rate between two quantities that are not
    commensurable. decisions-m2.md 2026-08-30.
    """
    mode = str(config["training"].get("select_on", "present_branch"))
    if mode == "total":
        return float(val_loss["total"])          # pre-2026-08-30 behaviour
    if mode == "present_branch":
        w_m = float(config["loss"]["w_m"])
        w_g = float(config["loss"].get("w_g", 0.0))
        return float(val_loss["L_pres"] + w_m * val_loss["L_MR"]
                     + w_g * val_loss.get("L_gain", 0.0))
    if mode == "separation":
        # L_pres alone. Available, and NOT the default: it is computed only on
        # target-present crops, so absent behaviour is unconstrained by it.
        # Measured 2026-08-30, it picks epoch 4-5 where L_abs is -5.1 to -6.3
        # against -10 to -12 elsewhere -- a model that separates well and then
        # keeps talking when nobody is there, on the quarter of trials that have
        # no target at all.
        return float(val_loss["L_pres"])
    raise ValueError(f"training.select_on: unknown mode {mode!r}. "
                     "Known: present_branch, total, separation.")


def selection_eligible(val_loss, config):
    """Whether an epoch is allowed to be kept at all -- the silence bar.

    A CONSTRAINT, not another weighted sum: "must be quiet enough, then be the
    best separator". A weighted sum would just invent a second exchange rate of
    the kind that caused the problem in the first place. `null` disables it.
    """
    bar = config["training"].get("select_abs_max", None)
    return True if bar is None else float(val_loss["L_abs"]) <= float(bar)


def train(model, train_loader, val_loader, optimizer, num_epochs, device, print_debug=False, save_path=None, config=None, scheduler=None, start_epoch=0, best_val=float("inf"), best_row=None, start_step=0):
    model.to(device)
    val_loss_history = []
    train_loss_history = []

    if config is None:
        raise ValueError("Config must be provided to build the loss function.")

    loss_fn = build_loss_fn(config)
    grad_clip = float(config["training"]["grad_clip"])
    patience = int(config["training"]["patience"])
    keep_top_k = int(config["training"].get("keep_top_k", 3))
    kept = []          # (score, epoch) for the top-k checkpoints still on disk
    print(f"  selecting on `{config['training'].get('select_on', 'present_branch')}`"
          f", silence bar L_abs <= {config['training'].get('select_abs_max', 'none')}"
          f", keeping top {keep_top_k}")
    epochs_since_best = 0
    # Fixed for the whole run. Every `total` reported anywhere uses this, so the
    # curve is one objective even while the schedule moves the training w.
    w_report = float(config["loss"]["w"])
    # STEPS PER EPOCH. len(train_loader) with drop_last=True is
    # floor(n_trials / batch_size): the loader counts TRIALS, and
    # both_directions widens each batch rather than lengthening the epoch.
    steps_per_epoch = len(train_loader)
    resolved = schedule_in_steps(config, steps_per_epoch)
    if resolved is not None:
        warmup_steps, ramp_steps, _, _ = resolved
        full_at = warmup_steps + ramp_steps
        print(f"w schedule: {config['loss']['w_schedule']}", flush=True)
        print(f"  {steps_per_epoch} steps/epoch  warmup {warmup_steps} steps  "
              f"ramp {ramp_steps} steps  full w at step {full_at} "
              f"(epoch {full_at / max(steps_per_epoch, 1):.2f})", flush=True)
        # Only as far as the end of the ramp: past it every epoch sits at
        # w_report, and printing 100 identical numbers buries the ones that differ.
        shown = min(num_epochs, math.ceil(full_at / max(steps_per_epoch, 1)) + 1)
        ws = [w_at_step(config, e * steps_per_epoch, steps_per_epoch)
              for e in range(shown)]
        print(f"  w at each epoch START: {[round(v, 4) for v in ws]}", flush=True)
        print(f"  reporting/selection w held at {w_report}", flush=True)

    # Mixed precision. Config-driven so history.csv is readable next to the flag
    # that produced it; CUDA-only because autocast("cuda") and GradScaler are.
    # GradScaler(enabled=False) makes scale/unscale_/step/update exact no-ops, so
    # ONE code path serves both precisions -- no branch in the hot loop and no
    # second path to keep correct.
    use_amp = (bool(config["training"].get("amp", False))
               and str(device).startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"  mixed precision: {'ON (fp16 forward, fp32 loss)' if use_amp else 'off'}",
          flush=True)

    # Position in the schedule, carried across a resume so a resumed run does
    # not restart the warmup and re-run the early-mute risk.
    global_step = int(start_step)

    for epoch in range(start_epoch, num_epochs):
        # Re-crop. Offsets are derived from (seed, epoch, idx), so without this
        # every epoch reads the same 4 s window of every clip -- reproducible,
        # and it throws away five sixths of the audio.
        train_loader.dataset.set_epoch(epoch)

        model.train()
        sums, counts = defaultdict(float), defaultdict(int)
        # For the reported `w`: the mean over this epoch's steps, since a
        # step-indexed schedule can move WITHIN an epoch.
        w_sum, w_steps = 0.0, 0
        epoch_start = time.time()
        # TRAINING LOSS
        # No progress bar: at ~1,666 batches an epoch it emitted more lines than
        # the whole rest of the run and buried the per-epoch numbers. The
        # breakdown printed at the end of the epoch replaces it (2026-08-31).
        for batch in train_loader:
            # THE ONE PLACE THE SCHEDULE TAKES EFFECT: the loss used for this
            # backward pass. Per STEP since 2026-09-03, so the warmup covers the
            # same amount of optimisation whatever the training set size.
            # Everything downstream reports at w_report.
            loss_fn.w = w_at_step(config, global_step, steps_per_epoch)
            w_sum += loss_fn.w
            w_steps += 1

            mixture, target, enrollment, crop_absent = unpack(batch, device)

            optimizer.zero_grad()
            with amp_ctx(use_amp):
                s_output = model(mixture, enrollment)
            # arg order is (reference, output, mixture, mask) -- reference
            # FIRST, the reverse of the usual (pred, target). See LossBSRNN.
            # .float() is not cosmetic: see amp_ctx on why the loss stays fp32.
            loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent)
            scaler.scale(loss).backward()
            # UNSCALE BEFORE CLIPPING. scale() multiplied the loss by ~65536 so
            # small gradients survive fp16, so the gradients sitting here are
            # inflated by that factor. Clipping them unscaled would compare an
            # inflated norm against grad_clip and crush every gradient to near
            # zero -- training would look stable and learn nothing.
            scaler.unscale_(optimizer)
            # A six-layer LSTM stack on an SI-SDR-family loss: a near-silent
            # present crop puts a very large gradient through alpha. Clip value
            # comes from the config, never from here.
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step() SKIPS the update when the gradients hold inf/NaN and
            # update() then lowers the scale. A few skipped steps at the start of
            # training is the scaler calibrating, not a bug.
            scaler.step(optimizer)
            scaler.update()

            add_parts(sums, counts, parts)
            global_step += 1

        # The MEAN w over the epoch's steps. Identical to the old per-epoch
        # value wherever w is constant -- every epoch after the ramp, and every
        # epoch of a legacy epoch-indexed run -- and inside the ramp it is the
        # weight that actually trained the epoch rather than one endpoint of it.
        w_trained = w_sum / max(w_steps, 1)

        epoch_loss = epoch_report(sums, counts, w_report, loss_fn.wm, loss_fn.wg)
        train_loss_history.append(epoch_loss)


        # VALIDATION LOSS
        model.eval()
        val_sums, val_counts = defaultdict(float), defaultdict(int)
        diag = defaultdict(float)
        with torch.no_grad():
            for batch in val_loader:
                mixture, target, enrollment, crop_absent = unpack(batch, device)

                # Val runs in the same precision as training on purpose: a
                # metric measured in a precision the model was not trained in
                # describes a model that does not exist. The loss is still fp32.
                with amp_ctx(use_amp):
                    s_output = model(mixture, enrollment)
                s_output = s_output.float()
                _, parts = loss_fn(target, s_output, mixture, crop_absent)
                add_parts(val_sums, val_counts, parts)
                diagnostic_accumulate(diag, model, mixture, enrollment,
                                      s_output, crop_absent, amp=use_amp)

        val_loss = epoch_report(val_sums, val_counts, w_report, loss_fn.wm, loss_fn.wg)
        val_loss.update(diagnostic_report(diag))
        val_loss["epoch"] = epoch
        val_loss["lr"] = optimizer.param_groups[0]["lr"]
        # the w that TRAINED this epoch, not w_report -- see history_header()
        val_loss["w"] = w_trained
        val_loss_history.append(val_loss)

        # One CSV row per epoch, same columns and same order as history.csv. The
        # header is printed once above the first row, so if the run dies the
        # printed block can be pasted straight into a .csv file and read back.
        # flush: stdout is a pipe under the Kaggle notebook, so without this the
        # rows sit in the buffer and a killed session loses exactly what this
        # exists to preserve.
        if epoch == start_epoch:
            print(",".join(history_header()), flush=True)
        print(",".join(str(v) for v in history_row(epoch_loss, val_loss)), flush=True)

        # Human-readable twin of the row above, on STDERR so stdout stays a
        # valid history.csv. Replaced the tqdm bars on 2026-08-31.
        print(format_epoch_breakdown(epoch, num_epochs, epoch_loss, val_loss,
                                     time.time() - epoch_start, w_trained),
              file=sys.stderr, flush=True)

        if scheduler is not None:
            scheduler.step(val_loss["total"])

        # A SECOND checkpoint, written every epoch regardless of improvement.
        #
        # The best-only save below can be many epochs stale: the 2026-08-25
        # warmup run improved on 9 of 10 epochs, but a run that plateaus early
        # leaves nothing newer than the plateau. On Kaggle a session that hits
        # the 12 h wall loses /kaggle/working entirely unless it was committed,
        # so "the newest weights" and "the best weights" are different insurance
        # policies and both are cheap (87 MB, ~1 s).
        #
        # Deliberately NOT the file --resume reads: resuming from a worse-but-
        # newer checkpoint silently changes which model a run continues from.
        if save_path:
            last_path = Path(save_path).with_name(Path(save_path).stem + "_last.pt")
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                # Where the w schedule had got to. Without it a resume restarts
                # the warmup. See w_at_step().
                "global_step": global_step,
                "best_val": best_val,
                "best_row": best_row,
                "config": config,
                "seed": config["seed"],
            }, last_path)

        # Keep this epoch's weights if it is the best SELECTION SCORE so far --
        # which is not `val_total`. See selection_score() for why, and for the
        # measurement that forced the change (decisions-m2.md 2026-08-30).
        score = selection_score(val_loss, config)
        eligible = selection_eligible(val_loss, config)
        # TOP-K INSURANCE. The criterion is a judgement call and this run's
        # history can be re-scored later, but only if the weights still exist.
        # Before 2026-08-30 just best-and-last were kept, so when the criterion
        # turned out to be wrong the good checkpoints were already gone and the
        # only recovery was a re-run. 87 MB each; keep a few.
        if save_path and keep_top_k > 0:
            # Deliberately NOT gated on `eligible`: on a run too short to ever
            # clear the silence bar these are the only weights that survive, and
            # the flag below is what tells you which ones cleared it.
            kept.append((score, epoch))
            kept.sort()
            rank_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_e{epoch:03d}.pt")
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "score": score, "eligible": eligible, "row": val_loss,
                        "config": config, "seed": config["seed"]}, rank_path)
            for _, dropped in kept[keep_top_k:]:
                stale = Path(save_path).with_name(
                    f"{Path(save_path).stem}_e{dropped:03d}.pt")
                stale.unlink(missing_ok=True)
            del kept[keep_top_k:]

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

    # NO EPOCH CLEARED THE SILENCE BAR, so `save_path` was never written and the
    # run would otherwise finish looking successful with no best checkpoint. Say
    # so loudly and point at the weights that do exist, rather than silently
    # relaxing the bar -- a run that never got quiet enough is a result about the
    # run, not a reason to lower the standard behind the user's back.
    if best_row is None:
        print(f"WARNING: no epoch met training.select_abs_max="
              f"{config['training'].get('select_abs_max')} on L_abs, so no best "
              f"checkpoint was written. The top-{keep_top_k} by score and "
              f"_last.pt are on disk; kept epochs: "
              f"{sorted(e for _, e in kept)}. Either the run is too short to "
              f"reach the bar or the bar is wrong for this split.",
              file=sys.stderr)

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
    # sir0: generated trials with its OWN audio, so manifest and audio dir match
    # (unlike `mid`, which is a row-subset over train/val audio). Symmetric
    # target/interferer loudness -- the arm that tests whether the model only
    # ignores the enrollment because "keep the loud voice" already works.
    "sir0":  (("sir0_train",  "sir0_train"),  ("sir0_val",  "sir0_val")),
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
    # Every trial trained twice, once per speaker. Config-driven so the arm is
    # recorded with the run; absent key = the old single-direction behaviour, so
    # older configs and checkpoints are unaffected. decisions-m2.md 2026-08-26.
    both_directions = bool(config["data"].get("both_directions", False))
    # Rotate the enrollment recording per epoch. TRAIN ONLY, on purpose: the val
    # set has to be fixed or its curve cannot be read across epochs, and pinning
    # it to variant 0 (= enrollment.wav) keeps val numbers comparable with runs
    # from before the bank existed. Absent key = 1 = the old behaviour, so older
    # configs and checkpoints are unaffected. decisions-m2.md 2026-08-30.
    enrollment_variants = int(config["data"].get("enrollment_variants", 1))
    # Per-epoch SIR/SNR re-draw. Train only, and the dataset pins it off on any
    # fixed set anyway. Absent key = False = the pre-2026-08-30 behaviour.
    remix_gains = bool(config["data"].get("remix_gains", False))

    csv_train = csv_path / f"{train_manifest}.csv"
    csv_val = csv_path / f"{val_manifest}.csv"

    train_dataset = TrialDataset(
        manifest_csv=csv_train,
        data_root=data_path,
        split=train_audio,
        chunk_s=config["data"]["chunk_s"], # follow CARTSE
        sample_rate=config["data"]["sample_rate"],
        seed=config["seed"],
        both_directions=both_directions,
        enrollment_variants=enrollment_variants,
        remix_gains=remix_gains,
    )

    val_dataset = TrialDataset(
        manifest_csv=csv_val,
        data_root=data_path,
        split=val_audio,
        chunk_s=config["data"]["chunk_s"], # follow CARTSE
        sample_rate=config["data"]["sample_rate"],
        seed=config["seed"],
        random_crop=False,
        both_directions=both_directions,
    )

    # num_workers from the config, default 0. 0 is right on the laptop (4 cores,
    # already saturated by the model) but starves a GPU: every crop is 3 windowed
    # wav reads, and single-threaded that dominates the step. Config-driven so the
    # figure that produced a given wall time is logged with the run.
    num_workers = int(config["data"].get("num_workers", 0))
    # persistent_workers only legal when num_workers > 0; without it each epoch
    # re-forks the pool, which set_epoch()'s per-epoch re-crop makes very visible.
    extra = dict(persistent_workers=True, prefetch_factor=4) if num_workers else {}
    # Page-locked staging buffers. Pageable host memory is not DMA-readable, so
    # every H2D copy goes through a bounce buffer and BLOCKS; pinned memory lets
    # the copy run async and overlap compute. Free, and only meaningful on GPU.
    extra["pin_memory"] = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=collate_pairs,
        **extra)

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_pairs,
        **extra)

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
        # No .git here. That is the normal case inside the Kaggle bundle, which
        # ships source files only -- so fall back to the hash stamped in at
        # bundle-build time by scripts/make_kaggle_bundle.py. Prefixed so it is
        # never mistaken for a hash read from a live checkout.
        stamp = Path(__file__).resolve().parents[1] / "docs/bundle_commit.txt"
        try:
            return "bundle:" + stamp.read_text().strip()
        except OSError:
            return "UNKNOWN-not-a-git-checkout"


def log_results(out_dir, config, config_path, args, model, device, manifest_csv, train_loss_history, val_loss_history, best_row, wall_s, num_epochs, save_path):
    """Write experiments/results/<dir>/{meta.yaml,history.csv}.

    CLAUDE.md: config, commit, metrics, seed, date on every result. Same shape
    as the 2026-08-20 anchor meta.yaml so the two read side by side. history.csv
    is WIDE (train_* and val_* on one row) because the plot is the two curves
    against each other; `lr` tells a plateau from a scheduler step.

    Never raises: a logging bug must not discard a finished run.
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        with open(out_dir / "history.csv", "w", newline="") as f:
            # lineterminator="\n": csv.writer defaults to CRLF, but the per-epoch
            # rows printed to stdout are LF. Matching them makes a block pasted
            # out of a killed run byte-identical to this file rather than merely
            # equivalent. pandas reads either, so no prior result is invalidated.
            writer = csv.writer(f, lineterminator="\n")
            writer.writerow(history_header())
            for tr, va in zip(train_loss_history, val_loss_history):
                writer.writerow(history_row(tr, va))

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
            # data as well as loss/training: the Kaggle notebook rewrites
            # batch_size (GPU memory) and num_workers, so a run logged without
            # this cannot say what batch it actually trained at. The 2-epoch
            # mid run on 2026-08-25 trained at batch 6 and did not record it.
            "data": config["data"],
            "loss": config["loss"],
            "training": config["training"],
            # The best val row, whichever run produced it -- on a resume that
            # can be an earlier invocation, which is why it rides in the
            # checkpoint rather than being recomputed from val_loss_history.
            # `or {}` because a checkpoint predating best_row has none.
            "best_val": {k: (best_row or {}).get(k) for k in
                         ["epoch", "total", "L_pres", "L_MR", "L_abs", "lr"]},
            # WHAT CHOSE that row. Without it a checkpoint cannot be compared
            # with one selected under a different rule, and runs either side of
            # 2026-08-30 were selected differently.
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
    except Exception as e:                      # noqa: BLE001 - never fatal
        print(f"  log_results: could not write {out_dir}: {e}", file=sys.stderr)
        return False


def plot_history(out_dir, train_loss_history, val_loss_history, best_row, loss_floor):
    """Write loss_plot.png: train and val `total` against epoch.

    Histories are lists of DICTS, so `total` is pulled out explicitly -- passing
    them to plt.plot raised TypeError on 2026-08-24 after a finished run. Hence
    also the try block: a plotting bug must not be the last thing a 12-hour run
    does. X axis is the VAL row's own `epoch`, not enumerate(), so a resumed run
    plots 40-60 rather than relabelling to 0-20.
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

    start_epoch, best_val, best_row, start_step = 0, float("inf"), None, 0
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
        # THE SCHEDULE'S POSITION MUST SURVIVE THE RESUME. .get falls back to
        # reconstructing it from the epoch count, which is exact for every
        # checkpoint written before 2026-09-03 provided the training set has not
        # changed size -- and a resume across a config change, which is how the
        # split changes, is refused below.
        start_step = int(ckpt.get("global_step", start_epoch * len(train_loader)))
        best_val = ckpt["best_val"]
        # .get, not [...]: checkpoints written before best_row existed have no
        # such key, and a missing best row is not a reason to refuse a resume.
        best_row = ckpt.get("best_row")
        # A resume across a config change is two experiments in one curve.
        if ckpt.get("config") != config:
            raise ValueError(f"{save_path} was trained under a different config, start a fresh run or pass a matching --config")
        print(f"resumed {save_path} at epoch {start_epoch}, step {start_step}, "
              f"best_val {best_val:.4f}")

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
            start_step=start_step,
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