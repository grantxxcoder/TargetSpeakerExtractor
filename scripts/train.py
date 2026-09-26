import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import contextlib
import copy
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
from src.models.bsrnn import BSRNN_TFMAP, BSRNN_TFMAP_CONTEXT  # noqa: E402
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
    # D17. Absent key = 0.0 = term disabled, so every pre-2026-09-13 config
    # reproduces its own objective exactly.
    w_struct = float(config["loss"].get("w_struct", 0.0))
    struct_floor_db = float(config["loss"].get("struct_floor_db", -40.0))
    assert w_struct >= 0.0, f"loss.w_struct must be >= 0, got {w_struct}"
    assert struct_floor_db <= 0.0, (
        f"loss.struct_floor_db is dB BELOW the clip peak and must be <= 0, "
        f"got {struct_floor_db}")

    # Convex weight: a typo of 4.58 for 0.458 makes (1 - w) negative, training
    # the model to destroy the target while the curve still looks like it falls.
    assert 0.0 <= w <= 1.0, f"loss.w must be in [0, 1], got {w}"
    assert wm >= 0.0, f"loss.w_m must be >= 0, got {wm}"
    assert wg >= 0.0, f"loss.w_g must be >= 0, got {wg}"
    # A negative deadzone punishes a PERFECT match -- reads as a dead term.
    assert gain_delta_db >= 0.0, f"loss.gain_delta_db must be >= 0, got {gain_delta_db}"

    # THE STATE TERM (decisions-pending.md D14) IS OPT-IN, AND THE BRANCH IS THE
    # OPT-IN. A config without `w_state` gets the plain LossBSRNN object, so a
    # pre-2026-09-11 run reproduces by construction rather than by a weight
    # multiplied by zero -- and no old config can reach the new code path even
    # by accident.
    w_state = float(config["loss"].get("w_state", 0.0))
    assert w_state >= 0.0, f"loss.w_state must be >= 0, got {w_state}"

    # this is the option to disable the effect of the head B for the trainer teacher setup
    if w_state <= 0.0:
        return LossBSRNN(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p,
                         windows=windows, sample_rate=sample_rate, wg=wg,
                         gain_delta_db=gain_delta_db,
                         w_struct=w_struct, struct_floor_db=struct_floor_db)

    # Imported here, not at module scope: the teacher pulls in speechbrain and
    # a 21 M-parameter checkpoint, and a baseline run should not pay for either.
    from src.models.losses_state import LossBSRNNState              
    from src.models.state_teacher import StateTeacher               

    teacher_path = config["loss"].get("state_teacher")
    assert teacher_path, (
        "loss.w_state > 0 needs loss.state_teacher, the frozen detector "
        "checkpoint. It defines part of the objective, so it is named in the "
        "config and hash-pinned, never defaulted.")
    teacher = StateTeacher(teacher_path,
                           config["loss"].get("ecapa_dir", "../ecapa_pretrained"),
                           device=config["loss"].get("state_device", "cpu"))

    return LossBSRNNState(wm=wm, w=w, tau_pres=tau_pres, tau_abs=tau_abs, p=p,
                          windows=windows, sample_rate=sample_rate, wg=wg,
                          gain_delta_db=gain_delta_db,
                          w_struct=w_struct, struct_floor_db=struct_floor_db,
                          teacher=teacher, w_state=w_state)


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
HISTORY_FIELDS = ["total", "L_pres", "L_MR", "L_gain", "L_abs", "L_state", "L_struct",
                  "n_present", "n_absent"]

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
# `content_wer` is the in-loop validation WORD ERROR RATE and `content_wer_ema`
# its smoothed twin. NaN on runs and epochs where the probe did not run, which
# is every run before 2026-09-23 -- history_row uses .get, so old histories and
# hand-built val rows keep working. src/live_model_metric/content_probe.py.
VAL_DIAGNOSTICS = ["enrol_sens_db", "pres_abs_gap_db",
                   "content_wer", "content_wer_ema"]


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

    `L_state` is printed only when the state term is in use, because otherwise
    it is a row of NaNs. THE PATTERN TO WATCH THERE IS NOT THE GAP: it is
    `L_state` falling while `L_pres` stalls or worsens. The teacher is a frozen
    learned scorer, so the model can improve on it by finding its blind spots
    instead of by removing the interferer -- and it has never heard masked
    audio, only real mixtures and synthetic suppressions. That divergence is
    the signature of reward-model overoptimisation, and it is why `L_state`
    appears in no selection mode. decisions-pending.md D14.

    `L_struct` (D17) prints on the same terms. It is the mask's frequency SHAPE
    error against the ideal mask with each frame's mean removed, so it is the
    only column that says whether the model is still applying a broadband gain
    rather than choosing bins. LOWER IS BETTER and 0.0 is the oracle's score.
    It is deliberately NOT in `total` (see epoch_report), so this column is the
    only place it can be read -- before 2026-09-14 it was computed every batch
    and then dropped here and from HISTORY_FIELDS, which made the D17 arm
    unreadable while still paying its cost.

    Goes to stderr on purpose: stdout carries one CSV row per epoch and must
    stay a valid history.csv so a killed Kaggle session can be recovered by
    pasting it into a file. See scripts/make_kaggle_notebook.py.
    """
    lines = [
        f"epoch {epoch + 1}/{num_epochs}  {epoch_seconds:.0f} s  "
        f"lr {va['lr']:.2e}  w_trained {w_trained:.3f}",
        f"  {'term':<7} {'train':>10} {'val':>10} {'gap(val-train)':>15}",
    ]
    for term in ("total", "L_pres", "L_MR", "L_gain", "L_abs", "L_state", "L_struct"):
        # L_state and L_struct are NaN when their term is not in use -- skip the
        # row rather than print a line of NaNs. The other four always apply.
        train_value, val_value = tr.get(term, float("nan")), va.get(term, float("nan"))
        if term in ("L_state", "L_struct") and not np.isfinite(train_value) \
                and not np.isfinite(val_value):
            continue
        lines.append(f"  {term:<7} {train_value:>10.4f} {val_value:>10.4f} "
                     f"{val_value - train_value:>15.4f}")
    lines.append(f"  crops   train {tr['n_present']} present / {tr['n_absent']} absent"
                 f"   val {va['n_present']} / {va['n_absent']}")
    lines.append(f"  diag    enrol_sens {va.get('enrol_sens_db', float('nan')):.2f} dB"
                 f"   pres_abs_gap {va.get('pres_abs_gap_db', float('nan')):.2f} dB")
    return "\n".join(lines)


def build_context_encoder(config, device="cpu"):
    """The frozen speaker encoder item 1c needs, or None for every other arm.

    ONE constructor, used by training AND by every evaluation script, so an eval
    cannot silently differ from the run it is scoring -- a different L2-norm
    setting or a different snapshot would change the embedding and therefore the
    output, with nothing appearing in a diff.

    Pair it with `context_kwargs()`:

        encoder = build_context_encoder(ckpt["config"], device)
        y = model(mixture, enrollment, **context_kwargs(encoder, enrollment))

    which is a no-op on the baseline and 1a paths.
    """
    if not bool(config["model"].get("context_embedding", False)):
        return None
    from src.models.context_encoder import ContextEncoder
    encoder = ContextEncoder(
        ecapa_dir=config["model"].get("ecapa_dir", "../ecapa_pretrained"),
        device=device,
        expected_hashes=config["model"].get("ecapa_sha256"),
        normalise=bool(config["model"].get("context_normalise", True)))
    want = int(config["model"].get("context_embedding_dim", 192))
    if encoder.embedding_dim != want:
        raise ValueError(
            f"the snapshot emits {encoder.embedding_dim}-d embeddings but the "
            f"model was built for {want}. Set model.context_embedding_dim.")
    return encoder


def context_kwargs(encoder, enrollment):
    """`enrol_embedding=...` for item 1c, or nothing at all.

    ONE place decides, so the training forward, the validation forward and the
    swap diagnostic cannot disagree about whether the arm is on. `encoder` is
    None for every other arm and this returns {}, leaving those call sites
    byte-identical to their pre-2026-09-22 form.
    """
    if encoder is None:
        return {}
    return {"enrol_embedding": encoder.embed(enrollment)}


def diagnostic_accumulate(diag, model, mixture, enrollment, s_output, crop_absent,
                          amp=False, enrol_embedding=None):
    """Accumulate the two leading indicators over one val batch.

    Costs one extra val forward per epoch. Rolls within the batch rather than
    shuffling globally. Skipped at batch 1, where roll() returns the same
    enrolment and would read a false 0 dB.
    """
    if mixture.shape[0] > 1:
        # Forward in fp16 when training does, but .float() IMMEDIATELY: the sums
        # below are sums of squares over 64k samples and would overflow fp16's
        # 65504 ceiling, silently turning the diagnostic into inf.
        # ITEM 1c: ROLL THE EMBEDDING TOO. The enrolment feeds the model twice --
        # the TF-Map cue and the identity anchor -- and rolling only the
        # waveform would swap half the conditioning while leaving the anchor
        # pointing at the original speaker. The diagnostic would then read as
        # "the model barely responds to the enrolment" for a model that
        # responds correctly to the half it was actually given. Rolling the
        # precomputed embedding is exactly equivalent to re-embedding the
        # rolled enrolment, because the embedding is per-example.
        swapped = ({} if enrol_embedding is None
                   else {"enrol_embedding": enrol_embedding.roll(1, 0)})
        with amp_ctx(amp):
            y_swapped = model(mixture, enrollment.roll(1, 0), **swapped)
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


def unwrap(model):
    """The real module behind a possible `nn.DataParallel` wrapper.

    `DataParallel` forwards `__call__` but NOT attribute access, and it prefixes
    every `state_dict()` key with `module.`. So anything that touches the model
    as an object rather than as a function -- `model.stft`, `model.band_widths`,
    saving and loading weights -- must go through here. A checkpoint written
    from the wrapper would not load into an unwrapped model, which would break
    `--resume`, `make_estimates.py` and every checkpoint already on disk.
    decisions-pending.md E7.
    """
    return model.module if isinstance(model, nn.DataParallel) else model


def build_model(config):
    """Config -> BSRNN_TFMAP. Every ctor argument comes from the yaml.

    Separate from main() so measure_train_cost.py measures the model that
    actually trains. One key deliberately not passed: separator.norm (implied
    by causal=True). It belongs in the yaml.
    """
    # ITEM 1c. Absent key => the parent class, i.e. every run up to 2026-09-22.
    # The subclass's forward REQUIRES enrol_embedding, so a config that turns
    # this on cannot be run by a script that has not been taught to supply one:
    # it is a TypeError, not a silently unconditioned model. decisions-m2.md
    # 2026-09-22, decisions-pending.md E8.
    cls = (BSRNN_TFMAP_CONTEXT if bool(config["model"].get("context_embedding", False))
           else BSRNN_TFMAP)
    extra = ({"embedding_dim": int(config["model"].get("context_embedding_dim", 192))}
             if cls is BSRNN_TFMAP_CONTEXT else {})
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
        # Estimator DEPTH. Absent key => 1, the architecture every run up to
        # 2026-09-21 trained -- so no existing config or checkpoint changes
        # meaning. It was hardcoded at the ctor default until now, which made
        # `n_hidden` one of the two sizing knobs the yaml could not reach
        # (decisions-m1.md 2026-08-19 records 1 as a deliberate deviation from
        # the wesep reference's 2, chosen because the paper specifies the
        # estimator's WIDTH but not its depth).
        n_hidden=int(config["model"]["mask"].get("n_hidden", 1)),
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
        # Head A, decisions-pending.md D14 piece A. Absent key => False, which
        # is the architecture every run up to 2026-09-11 trained, so no old
        # config or checkpoint changes meaning. Training-only: the head is
        # stripped before inference (state_head.drop_state_head), so a model
        # trained with it deploys at the same 7.19 M parameters and the same
        # latency as the baseline.
        state_head=bool(config["model"].get("state_head", False)),
        state_head_detach=bool(config["model"].get("state_head_detach", False)),
        # D4a, decisions-pending.md D4. Absent key => False, the 2026-08-28
        # frozen architecture. Unlike the state head this one adds parameters to
        # the audio path (+37,698, +0.52 %), so the arm is NOT parameter-matched
        # to its control and the write-up has to say so.
        tfmap_inject=bool(config["model"].get("tfmap_inject", False)),
        tfmap_gate_init=float(config["model"].get("tfmap_gate_init", 0.0)),
        # ITEM 1a, ranked-next-steps.md. Absent key => False, the 1-channel
        # product cue every run up to 2026-09-22 trained on. True hands the
        # network the cue's PARTS instead (direction, normalised similarity,
        # unexplained residual), widening the input 3 -> 5 channels and
        # SubbandNorm by +66,820 parameters (+0.93 %), MEASURED at the shipped
        # width -- 128*257*2 of conv weight plus 2*257*2 of LayerNorm gain and
        # bias, the second term easy to forget. Like tfmap_inject this
        # is NOT parameter-matched to its control and the write-up must say so.
        tfmap_parts=bool(config["model"].get("tfmap_parts", False)),
        # DERIVED by scripts/derive_cue_scales.py. Absent => all ones, which is
        # the un-scaled form the first 1a probe trained, kept reproducible.
        tfmap_part_scales=config["model"].get("tfmap_part_scales"),
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
        # D17, present branch only -- see LossBSRNN._loss_mask_shape on why the
        # ideal mask is meaningless on a target-absent crop. NaN when the term
        # is not in use, and NaN * n would poison the sum, so it is gated.
        if not math.isnan(parts.get("L_struct", float("nan"))):
            sums["L_struct"] += parts["L_struct"] * parts["n_present"]
        counts["present"] += parts["n_present"]
    if parts["n_absent"]:
        sums["L_abs"] += parts["L_abs"] * parts["n_absent"]
        counts["absent"] += parts["n_absent"]
    # L_state applies to EVERY crop, present and absent alike: there is no
    # situation in which a second voice belongs in the output, so it has its own
    # count rather than sharing either branch's. Absent unless the state loss is
    # in use, so `.get` rather than a key.
    if parts.get("L_state") is not None:
        n_all = parts["n_present"] + parts["n_absent"]
        sums["L_state"] += parts["L_state"] * n_all
        counts["all"] += n_all


def oracle_mask_and_mag(model, target, mixture, clip=2.0):
    """The ideal mask |target| / |mixture| on the MODEL'S OWN STFT grid, and the
    mixture magnitude the structure term weights by.

    Computed here rather than inside the loss because the grid belongs to the
    model -- n_fft, hop and window are the model's, and a loss that built its own
    would supervise the mask on a different grid from the one it was predicted
    on, which is a misalignment nothing else would catch.

    CLIPPED to [0, clip]. The ratio is unbounded: it exceeds 1 whenever the
    interferer is out of phase with the target, and explodes wherever the mixture
    is near silent. Same clip as scripts/plot_mask_grid.py so the training target
    and the diagnostic picture are the same object.

    NO GRADIENT. The oracle is data, not a prediction.
    """
    with torch.no_grad():
        S = model.stft(target).abs()
        X = model.stft(mixture).abs()
        frames = min(S.shape[-1], X.shape[-1])
        S, X = S[..., :frames], X[..., :frames]
        return (S / X.clamp_min(1e-8)).clamp(0.0, clip), X


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
    # NaN, not 0.0, when the term is off: a missing term is a GAP in the curve,
    # not a zero. Plotting it as zero would suggest a perfectly satisfied
    # objective rather than an absent one.
    n_all = counts.get("all", 0)
    L_state = sums["L_state"] / n_all if n_all else float("nan")
    # NaN, not 0.0, when the term never ran, and here that matters MORE than it
    # does for the four above: 0.0 is the ORACLE's score on this term (a perfect
    # shape match, see derive_w_struct.py), so a disabled term logged as 0.0
    # reads as a solved one. The key exists only where add_parts accumulated it,
    # which it does only for a non-NaN L_struct.
    L_struct = (sums["L_struct"] / n_present
                if n_present and "L_struct" in sums else float("nan"))

    # `total` deliberately EXCLUDES the state term. It is the number
    # ReduceLROnPlateau and the curve read, and the four terms above are what it
    # has always meant -- adding a fifth would make every run before 2026-09-11
    # incomparable to every run after. w_state's contribution is visible in its
    # own column, and derive_w_state.py reports its gradient share.
    return {
        "total": (1 - w) * (L_pres + wm * L_MR + wg * L_gain) + w * L_abs,
        "L_pres": L_pres,
        "L_MR": L_MR,
        "L_gain": L_gain,
        "L_abs": L_abs,
        "L_state": L_state,
        # EXCLUDED from `total` for the same reason L_state is: `total` is what
        # ReduceLROnPlateau and the curve read, and adding a term would make
        # every run before 2026-09-13 incomparable to every run after. D17's
        # contribution is visible in its own column.
        "L_struct": L_struct,
        "n_present": n_present,
        "n_absent": n_absent,
    }


def lr_schedule_metric(val_loss, config):
    """The number `ReduceLROnPlateau` watches. Config key `lr_schedule_on`.

    WHY THIS EXISTS. Until 2026-09-21 the scheduler stepped on
    `val_loss["total"]` directly, which is the quantity `selection_score`'s own
    docstring spends two paragraphs explaining is unfit for ranking models --
    `total` contains `L_abs`, `L_abs` rewards silence, so it keeps falling as
    the model goes quiet long after separation has stopped improving. Selection
    was fixed on 2026-08-30; the schedule was not, so the two disagreed for
    three weeks. MEASURED CONSEQUENCE: the 14.73 M run halved its lr at epoch
    idx 13 (`2026-09-21-train-sir0-wesepref/history.csv`) on a number a model
    can improve by muting itself.

    DEFAULTS TO `total`, deliberately. Changing the schedule changes training
    dynamics, so every config written before 2026-09-21 must keep the behaviour
    it actually ran under or its curves stop being comparable. New arms opt in
    with `lr_schedule_on: present_branch`.

    NOT `separation` (L_pres alone), even though it is available and sounds like
    the obvious choice. `selection_score` records it being measured on
    2026-08-30 and rejected: computed only on target-present crops, it leaves
    absent behaviour unconstrained and picks epochs that are loud on crops where
    the target never speaks. The same objection applies here -- a schedule that
    only sees separation would hold the lr up while the model learns to shout
    through silence.

    The modes and their arithmetic are `selection_score`'s, reused rather than
    re-derived, so the schedule and the selector cannot drift apart again.
    """
    mode = str(config["training"].get("lr_schedule_on", "total"))
    shim = {**config, "training": {**config["training"], "select_on": mode}}
    return selection_score(val_loss, shim)


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

    NO MODE INCLUDES `L_state`, AND THAT IS DELIBERATE. It is a frozen learned
    scorer, and a model can improve on it by finding its blind spots rather than
    by removing the interferer -- reward-model overoptimisation. Selecting on it
    would make that failure invisible, because the thing being gamed would also
    be the thing choosing the checkpoint.

    This project has watched a headline number improve for a bad reason twice
    already: 2026-08-25, total loss falling the whole way into a mute; and
    2026-09-04, `enrol_sens` and `pres_abs_gap` reaching their best values of the
    run on the epoch where held-out separation dropped below pass-through. The
    state term is a more gameable quantity than either.

    So `L_state` is LOGGED and never SELECTED ON. If it falls while
    `present_branch` stalls, that is the signature, and it should be reported in
    the same breath as any result. decisions-pending.md D14.
    """
    mode = str(config["training"].get("select_on", "present_branch"))
    if mode == "content_wer":
        # THE ONLY MODE THAT WATCHES WHAT THE PROJECT MEASURES. Every
        # signal-domain mode below ranked item 1a's epoch 9 first, and epoch 9
        # is worse than doing nothing on content (ASR LCF-WER 65.63 against a
        # 65.22 floor) while epoch 15, tenth of fourteen on `present_branch`,
        # reads 52.77. No reweighting of the present branch reaches 15;
        # scripts/reselect_epochs.py checks all of them. decisions-m2.md
        # 2026-09-23.
        #
        # Reads the SMOOTHED value, and that is not a detail. LCF-WER is not a
        # smooth function of audio quality -- the 2026-09-01 mix-back sweep
        # moved the signal monotonically while WER went 65.2, 63.4, 69.6, 67.2,
        # 59.1 -- so a raw per-epoch WER would plateau-trigger the scheduler on
        # noise. The raw number is logged beside it.
        #
        # Raises rather than falling back when the probe is off: silently
        # selecting on something other than what the config asked for is the
        # class of bug this whole entry is about.
        value = val_loss.get("content_wer_ema", val_loss.get("content_wer"))
        if value is None or value != value:
            raise ValueError(
                "select_on/lr_schedule_on is `content_wer` but no validation "
                "WER was computed this epoch. Enable the probe with a "
                "`content_probe:` block, and set `every_n_epochs: 1` -- the "
                "scheduler needs a number every epoch.")
        return float(value)
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
                     "Known: content_wer, present_branch, total, separation.")


def selection_eligible(val_loss, config):
    """Whether an epoch is allowed to be kept at all -- the silence bar.

    A CONSTRAINT, not another weighted sum: "must be quiet enough, then be the
    best separator". A weighted sum would just invent a second exchange rate of
    the kind that caused the problem in the first place. `null` disables it.
    """
    bar = config["training"].get("select_abs_max", None)
    return True if bar is None else float(val_loss["L_abs"]) <= float(bar)


HOLDOUT_SPLITS = frozenset({"sir0_privval", "eval_private"})

# Where the probe's data is MOUNTED, filled from --data-root / --manifest-dir by
# main(). A location, not a training setting -- --data-root itself is not in the
# config -- so it must never decide whether two configs are the same run.
PROBE_PATH_KEYS = ("data_root", "manifest_dir")


def comparable_config(config):
    """The config minus run-time data locations, for "is this the same run?".

    Every checkpoint since 2026-09-23 stores the Kaggle mount paths inside
    `content_probe`, and the config FILE does not, so a plain != refused the
    first resume of the content_wer run. The notebook's pre-check mirrors this.
    """
    out = copy.deepcopy(config or {})
    probe = out.get("content_probe")
    if isinstance(probe, dict):
        for key in PROBE_PATH_KEYS:
            probe.pop(key, None)
    return out


def build_content_probe(config, verbose=True):
    """The in-loop validation WER probe, or None when the run does not want one.

    Config, all optional inside `content_probe:`:

        content_probe:
          enabled: true
          split: sir0_val      # NEVER a holdout -- refused below
          n_trials: 40         # the cost knob
          every_n_epochs: 1    # 1 if the scheduler steps on it
          asr_device: cpu      # cpu keeps it identical to the reported instrument
          ema_span: 3
          cap_errors_at_spoken: true   # loop guard; see content_probe.clip_errors.
                                       # Absent = false. Set it on a FRESH run only:
                                       # a resume refuses the change, by design.

    COST, and it is the reason for `n_trials`. MEASURED 2026-09-23 end to end:
    ~4 s per clip for the ASR on cpu, ~16 s per clip for extraction on cpu. On
    Kaggle extraction runs on the T4 and is the cheap half, so the ASR sets the
    cost: 40 clips is ~2.7 min per epoch, about 7 % of a 2,575 s epoch, or ~48
    min over a 16-epoch run. `asr_device: cuda` is far faster, but the
    transcripts are then not bit-identical to the reported instrument, which is
    why cpu is the default and the device is recorded.
    """
    block = (config.get("content_probe") or {})
    if not block.get("enabled", False):
        return None
    from src.live_model_metric.content_probe import ContentProbe, load_probe_trials

    split = str(block.get("split", "sir0_val"))
    if split in HOLDOUT_SPLITS:
        raise ValueError(
            f"content_probe.split={split!r} is a reported holdout. Steering "
            f"training on it would destroy the only clean number the project "
            f"has. CLAUDE.md; decisions-m4.md 2026-09-15.")
    data_root = Path(block.get("data_root", "data"))
    trials = load_probe_trials(
        Path(block.get("manifest_dir", "data/manifests")) / f"{split}.csv",
        data_root / "rendered" / split,
        limit=int(block.get("n_trials", 40)),
        condition=str(block.get("condition", "both")))
    probe = ContentProbe(trials,
                         device=str(block.get("asr_device", "cpu")),
                         ema_span=int(block.get("ema_span", 3)),
                         # 2026-09-24 loop guard. Absent = off, so a run begun
                         # without it resumes under the rule it was scored by.
                         cap_errors_at_spoken=bool(block.get("cap_errors_at_spoken", False)))
    if verbose:
        print(f"  content probe ON: {probe.describe()}, split {split}, "
              f"every {int(block.get('every_n_epochs', 1))} epoch(s)")
    return probe


def live_extractor(model, context_encoder, device):
    """runner.py's Extractor contract over the model currently being trained.

    Whole clip, one forward pass, no chunking -- the same path
    scripts/make_estimates.py:build_extractor uses, so the probe scores the
    audio the evaluation harness would score rather than a training crop. The
    caller is responsible for model.eval() and torch.no_grad().
    """
    def extract(mixture, enrollment, sample_rate):    # noqa: ARG001
        enrol = torch.from_numpy(enrollment).unsqueeze(0).to(device)
        estimate = model(torch.from_numpy(mixture).unsqueeze(0).to(device), enrol,
                         **context_kwargs(context_encoder, enrol))
        return estimate.squeeze(0).float().cpu().numpy()
    return extract


def checkpoints_to_drop(kept, keep_top_k, keep_stride=0, last_epoch=None):
    """Which epochs' weights may be deleted. `kept` is a sorted (score, epoch) list.

    A PURE FUNCTION so the retention policy can be tested without a training
    run. The failure it guards against is unrecoverable: once the weights are
    gone, re-scoring an epoch means training again.

    TWO KINDS OF INSURANCE, AND THEY COVER DIFFERENT RISKS.

    `keep_top_k` keeps the best few BY THE SIGNAL SCORE. It covers "the exchange
    rate between the terms was slightly wrong".

    `keep_stride` keeps every k-th epoch regardless of score, plus the last. It
    covers the larger risk, which is that the signal score does not rank epochs
    the way CONTENT does at all. Measured 2026-09-23: item 1a's best content
    epoch (15, ASR LCF-WER 52.77 against a 65.22 floor) sits TENTH OF FOURTEEN
    eligible epochs on `present_branch`, so no top-k cut reaches it and no
    reweighting of the present branch reaches it either. Only a cut that ignores
    the score entirely -- a stride -- is guaranteed to span the run.

    Cost, so the trade is explicit: ~87 MB per checkpoint. Over a 16-epoch run,
    `keep_stride: 3` retains 6 by stride, union'd with top-k, and lands near
    600-700 MB. Keeping every epoch would be 1.4 GB, which is why this is a
    stride and not a flag.
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
    # Weights are saved and the STFT grid is read from the REAL module, never
    # from a DataParallel wrapper. See unwrap().
    core = unwrap(model)
    val_loss_history = []
    train_loss_history = []

    if config is None:
        raise ValueError("Config must be provided to build the loss function.")
    resumed_wer_history = list(resumed_wer_history or [])

    loss_fn = build_loss_fn(config)
    # D17. `log_struct` computes and LOGS L_struct while leaving it out of the
    # gradient, which is what a derivation script reads to set w_struct -- the
    # same arrangement that let derive_w_g.py set w_g while L_gain was shipped
    # off. Defaults ON when the key is absent so the column exists for every
    # future run; the cost is one extra STFT pair per batch and a retained mask.
    log_struct = bool(config["loss"].get("log_struct", True))
    want_mask_val = getattr(loss_fn, "w_struct", 0.0) > 0.0 or log_struct
    grad_clip = float(config["training"]["grad_clip"])
    patience = int(config["training"]["patience"])
    keep_top_k = int(config["training"].get("keep_top_k", 3))
    # Defaults 0 (off) so every config written before 2026-09-23 keeps the disk
    # footprint it ran with. New arms that will be selected on content opt in
    # with `keep_stride: 3`.
    keep_stride = int(config["training"].get("keep_stride", 0))
    kept = []          # (score, epoch) for the top-k checkpoints still on disk
    print(f"  selecting on `{config['training'].get('select_on', 'present_branch')}`"
          f", silence bar L_abs <= {config['training'].get('select_abs_max', 'none')}"
          f", keeping top {keep_top_k}")
    epochs_since_best = 0
    content_probe = build_content_probe(config)
    probe_every = int((config.get("content_probe") or {}).get("every_n_epochs", 1))
    if content_probe is not None and resumed_wer_history:
        content_probe.warm_start(resumed_wer_history)
        print(f"  content probe warm-started from {len(resumed_wer_history)} "
              f"earlier epochs; EMA resumes at {content_probe.smoothed[-1]:.2f}")
    elif content_probe is not None and start_epoch > 0:
        print("  WARNING: resuming with a COLD content probe -- this checkpoint "
              "predates content_wer_history, so the EMA and the trend restart. "
              "The first epoch or two after the resume will read low-confidence.")
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

            # I can only use the teacher loss if I have the enrollment to supply the teacher with.
            if hasattr(loss_fn, "enrolment_embedding"):
                loss_fn.enrolment_embedding = loss_fn.teacher.embed_enrolment(enrollment)

            # ITEM 1c. {} for every other arm, so this line is a no-op on the
            # baseline and 1a paths. Computed ONCE per batch: the enrolment does
            # not change within a step, and the encoder is the larger forward.
            ctx = context_kwargs(context_encoder, enrollment)

            optimizer.zero_grad()
            # D17: only ask for the mask when the term is configured. The flag
            # retains a (B, F, T) tensor in the graph, so a run that does not use
            # it should not pay for it.
            want_mask = getattr(loss_fn, "w_struct", 0.0) > 0.0 or log_struct
            with amp_ctx(use_amp):
                if want_mask:
                    s_output, mask = model(mixture, enrollment, return_mask=True, **ctx)
                else:
                    s_output, mask = model(mixture, enrollment, **ctx), None
            oracle, mix_mag = (oracle_mask_and_mag(core, target, mixture)
                               if want_mask else (None, None))
            # arg order is (reference, output, mixture, mask) -- reference
            # FIRST, the reverse of the usual (pred, target). See LossBSRNN.
            # .float() is not cosmetic: see amp_ctx on why the loss stays fp32.
            loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent,
                                  mask=None if mask is None else mask.float(),
                                  oracle_mask=oracle, mixture_mag=mix_mag)
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

                # Same setter as the training loop for the teacher loss
                if hasattr(loss_fn, "enrolment_embedding"):
                    loss_fn.enrolment_embedding = loss_fn.teacher.embed_enrolment(enrollment)

                ctx = context_kwargs(context_encoder, enrollment)

                # Val runs in the same precision as training on purpose: a
                # metric measured in a precision the model was not trained in
                # describes a model that does not exist. The loss is still fp32.
                with amp_ctx(use_amp):
                    if want_mask_val:
                        s_output, mask = model(mixture, enrollment, return_mask=True, **ctx)
                    else:
                        s_output, mask = model(mixture, enrollment, **ctx), None
                s_output = s_output.float()
                oracle, mix_mag = (oracle_mask_and_mag(core, target, mixture)
                                   if want_mask_val else (None, None))
                _, parts = loss_fn(target, s_output, mixture, crop_absent,
                                   mask=None if mask is None else mask.float(),
                                   oracle_mask=oracle, mixture_mag=mix_mag)
                add_parts(val_sums, val_counts, parts)
                diagnostic_accumulate(diag, model, mixture, enrollment,
                                      s_output, crop_absent, amp=use_amp,
                                      enrol_embedding=ctx.get("enrol_embedding"))

        val_loss = epoch_report(val_sums, val_counts, w_report, loss_fn.wm, loss_fn.wg)
        val_loss.update(diagnostic_report(diag))
        # THE CONTENT PROBE. Still inside model.eval(); torch.no_grad() is the
        # probe's own business because it calls the model itself. Whole clips,
        # so this is a second pass over a small fixed subset rather than a
        # re-read of the 4 s validation crops -- LCF-WER needs a whole utterance
        # and its reference text, and a crop has neither.
        if content_probe is not None and epoch % probe_every == 0:
            with torch.no_grad():
                val_loss["content_wer"] = content_probe.score(
                    live_extractor(core, context_encoder, device))
            val_loss["content_wer_ema"] = content_probe.smoothed[-1]
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

        # "Is this run worth continuing?" -- the question item 1a could not
        # answer, because its content was still improving at the last epoch and
        # nothing in the loop was watching.
        if content_probe is not None and content_probe.history:
            print("  " + content_probe.verdict(), file=sys.stderr, flush=True)

        if scheduler is not None:
            # NOT val_loss["total"] -- see lr_schedule_metric(). Defaults to
            # `total`, so this is a no-op for every config written before
            # 2026-09-21.
            scheduler.step(lr_schedule_metric(val_loss, config))

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
                "model": core.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler else None,
                "epoch": epoch,
                # Where the w schedule had got to. Without it a resume restarts
                # the warmup. See w_at_step().
                "global_step": global_step,
                # THE PROBE'S HISTORY MUST SURVIVE A RESUME. The EMA the
                # scheduler reads, and trend(), are functions of every epoch
                # so far. A 12 h Kaggle session caps this run at ~16 epochs
                # (2,575 s/epoch measured), so going further MEANS resuming,
                # and a resumed probe that restarts its EMA hands
                # ReduceLROnPlateau a step change that is an artefact of the
                # session boundary. decisions-m2.md 2026-09-23.
                "content_wer_history": (content_probe.history
                                        if content_probe is not None else []),
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
        #
        # 2026-09-23: TOP-K IS NOT ENOUGH INSURANCE, MEASURED. Item 1a's epoch
        # 15 is the best content result the project has produced (ASR LCF-WER
        # 52.77 against a 65.22 floor) and it ranks TENTH OF FOURTEEN eligible
        # epochs on `present_branch` (4.319 at the kept epoch 9, 6.443 at 15).
        # No top-k shortlist reaches it and no reweighting of the present branch
        # reaches it either -- dropping L_gain still keeps 9. It survived only
        # because `_last.pt` is written unconditionally, which is luck, not
        # policy: a 20-epoch run would have deleted it unscored.
        # `keep_stride` is what makes post-hoc selection on content possible
        # at all. See scripts/select_by_wer.py and decisions-m2.md 2026-09-23.
        if save_path and (keep_top_k > 0 or keep_stride > 0):
            # Deliberately NOT gated on `eligible`: on a run too short to ever
            # clear the silence bar these are the only weights that survive, and
            # the flag below is what tells you which ones cleared it.
            kept.append((score, epoch))
            kept.sort()
            rank_path = Path(save_path).with_name(
                f"{Path(save_path).stem}_e{epoch:03d}.pt")
            torch.save({"model": core.state_dict(), "epoch": epoch,
                        "score": score, "eligible": eligible, "row": val_loss,
                        "config": config, "seed": config["seed"]}, rank_path)
            for dropped in checkpoints_to_drop(kept, keep_top_k, keep_stride,
                                               last_epoch=epoch):
                stale = Path(save_path).with_name(
                    f"{Path(save_path).stem}_e{dropped:03d}.pt")
                stale.unlink(missing_ok=True)
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
                # THE PROBE'S HISTORY MUST SURVIVE A RESUME. The EMA the
                # scheduler reads, and trend(), are functions of every epoch
                # so far. A 12 h Kaggle session caps this run at ~16 epochs
                # (2,575 s/epoch measured), so going further MEANS resuming,
                # and a resumed probe that restarts its EMA hands
                # ReduceLROnPlateau a step change that is an artefact of the
                # session boundary. decisions-m2.md 2026-09-23.
                "content_wer_history": (content_probe.history
                                        if content_probe is not None else []),
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
    # sir0ext: sir0's TRAINING data, evaluated on the expanded 2,800-trial dev
    # split built 2026-09-13 from eval_private's released speakers. EVAL ONLY in
    # practice -- the train half is deliberately identical to `sir0` so a
    # checkpoint trained under `sir0` is scored on the wider set WITHOUT
    # retraining, and the two splits differ in exactly one axis.
    #
    # WHY IT EXISTS. The paired bootstrap on 2026-09-12 put sir0_val's
    # resolution at +-8 LCF-WER over its 103 `both` trials, so an arm moving the
    # metric less than ~5 points is unreadable. sir0_privval carries 1,421
    # `both` trials, and SIR spans [-10, +15] against sir0_val's [-10, +10] --
    # a SUPERSET, reported per SIR band, never as one blended mean.
    # decisions-m3.md 2026-09-13.
    "sir0ext": (("sir0_train", "sir0_train"), ("sir0_privval", "sir0_privval")),
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
            # unwrap(): with DataParallel on, `type(model).__name__` would
            # record "DataParallel" and the meta.yaml would no longer say which
            # architecture ran. band_widths is not forwarded at all.
            "model": {
                "class": type(unwrap(model)).__name__,
                "n_parameters": sum(p.numel() for p in unwrap(model).parameters()),
                "n_bands": len(unwrap(model).band_widths),
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
    # The content probe reads the SAME data as the val loader, so it takes the
    # CLI's --data-root / --manifest-dir unless its config block names its own.
    # Without this it fell back to "data/manifests", which does not exist on
    # Kaggle: 2026-09-23 run died at startup, "no manifest at
    # data/manifests/sir0_val.csv".
    if (config.get("content_probe") or {}).get("enabled", False):
        config["content_probe"].setdefault("data_root", str(data_path_root))
        config["content_probe"].setdefault("manifest_dir", str(csv_path))

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

    # ITEM 1c. Built BESIDE the model, never inside it: it is frozen, it is not
    # optimised, it must not be replicated per card by DataParallel, and it must
    # not bloat every saved checkpoint by ~83 MB. src/models/context_encoder.py
    # gives the full reasoning. None for every other arm, and `context_kwargs`
    # then returns {} so those paths are byte-identical.
    context_encoder = build_context_encoder(config, device)
    if context_encoder is not None:
        print(f"  context encoder: {context_encoder.describe()}", flush=True)

    # BOTH T4s. decisions-pending.md E7: Kaggle's "GPU T4 x2" gives two cards and
    # every run before 2026-09-21 used one, because `torch.device("cuda")` is
    # cuda:0 and nothing here asked for more.
    #
    # DataParallel scatters the BATCH across the cards, runs the forward on
    # each, and gathers the outputs back to cuda:0. So `LossBSRNN` still sees
    # the whole batch on one device, which matters: the loss means over SUBSETS
    # (n_present, n_absent) and a per-device reduction would silently reweight
    # them. It does not break the direction pairing either -- `collate_pairs`
    # puts both directions of a trial in one batch and splitting that batch does
    # separate some pairs in the forward, but the model is per-example
    # independent and the contrast lives in the loss, which sees the gathered
    # batch.
    #
    # WHAT IT DOES NOT DO: it replicates the model on both cards, so it buys
    # BATCH headroom, not room for a wider model. Per-card activation memory is
    # unchanged, and the measured ceiling (0.12 GB fixed + 2.15 GB per trial
    # against 14.56 GiB, E3b-E3f) still applies PER CARD.
    #
    # Default OFF so every existing run reproduces exactly.
    n_gpu = torch.cuda.device_count()
    use_dp = bool(config["training"].get("data_parallel", False)) and n_gpu > 1
    if use_dp:
        model = nn.DataParallel(model)
        print(f"DataParallel across {n_gpu} GPUs, "
              f"batch {config['data']['batch_size']} split {config['data']['batch_size'] // n_gpu} per card")
    elif bool(config["training"].get("data_parallel", False)):
        print(f"WARNING: data_parallel requested but device_count() == {n_gpu}; "
              "running on one device", file=sys.stderr)
    core = unwrap(model)

    # optimiser
    optimizer = torch.optim.AdamW(core.parameters(),
                                  lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min",
        factor=float(config["training"]["lr_factor"]),
        patience=int(config["training"]["lr_patience"]))

    save_path = Path(args.outdir) / f"model_{args.split}.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch, best_val, best_row, start_step = 0, float("inf"), None, 0
    # Empty on a fresh run; the resume branch below replaces it. Declared here
    # so the train() call is valid on both paths.
    resumed_wer_history = []
    if args.resume:
        if not save_path.exists():
            raise FileNotFoundError(f"--resume but no checkpoint at {save_path}")
        # weights_only=False: the checkpoint carries the config dict, not just
        # tensors. Safe because we wrote it; never point this at a file you did not.
        ckpt = torch.load(save_path, map_location=device, weights_only=False)
        # `core`, not `model`: checkpoints are written unwrapped (see unwrap()),
        # so a DataParallel run resumes from an ordinary checkpoint and the
        # files stay interchangeable between the two modes.
        core.load_state_dict(ckpt["model"])
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
        # .get with a default: only runs from 2026-09-23 onward carry it, and a
        # missing probe history is not a reason to refuse a resume -- it just
        # means the EMA starts cold, which the warning below makes visible.
        resumed_wer_history = list(ckpt.get("content_wer_history") or [])
        # A resume across a config change is two experiments in one curve.
        if comparable_config(ckpt.get("config")) != comparable_config(config):
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
               rate=lambda: f"batch {config['data']['batch_size']}, {device}"
                            f"{f' x{n_gpu} (DataParallel)' if use_dp else ''}, "
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
            context_encoder=context_encoder,
            start_epoch=start_epoch,
            best_val=best_val,
            best_row=best_row,
            start_step=start_step,
            resumed_wer_history=resumed_wer_history,
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