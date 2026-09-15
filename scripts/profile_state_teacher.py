"""What does the state teacher cost per training step? decisions-pending.md D14.

    ../tse_venv/bin/python scripts/profile_state_teacher.py \
        --teacher models/state_detector_notebook.pt --device cuda

Writes experiments/results/<date>-state-teacher-cost/{results.json,cost.pdf}.

THIS IS THE NUMBER THAT DECIDES WHETHER THE ARM RUNS AT ALL. Every timing
recorded for the teacher so far is CPU; training is a Kaggle T4. Scoring the
full 4.008 s chunk is 13 one-second windows per example, forward AND backward
through a 21 M-parameter frozen encoder, against a current 0.674 s/step at
batch 3 (decisions-m2.md 2026-08-28). A 16-epoch run is 10.5 h against a 12 h
session cap, so anything beyond roughly +10 % per step no longer fits.

WHY IT SWEEPS SEGMENT LENGTH RATHER THAN A RANDOM SUBSET OF WINDOWS
-------------------------------------------------------------------
The term is a mean over windows, so scoring a random subset would be an
unbiased estimate of it -- the mini-batch argument, and the reason even a
single-sample estimate converges given enough steps.

That reasoning is sound and mostly INAPPLICABLE here, because the teacher's
head is a BiLSTM over the window sequence:

  - the loss on 4 windows still needs all 13 ECAPA embeddings to feed the
    recurrence, so the expensive forward is unchanged;
  - the backward flows through the recurrence to all 13 hidden states anyway.

So random subsampling buys almost nothing. What buys time is scoring a shorter
CONTIGUOUS segment, which shortens the recurrence as well as the encoder.
The cost is a mild distribution shift -- the head was fitted on 13 windows of
context and would see fewer -- which is why this script reports the TERM'S
VALUE at each length beside the timing. A length that is fast and changes what
the teacher says is not a saving.

Reports, per segment length:
  seconds per step, forward and backward, against the no-teacher baseline
  peak GPU memory, against the ~15 GB card and the batch-3 activation ceiling
  L_state itself, so a cheap setting that changes the measurement is visible
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from src.models.losses import LossBSRNN  # noqa: E402
from src.models.losses_state import LossBSRNNState  # noqa: E402
from src.models.state_teacher import StateTeacher  # noqa: E402
from train import amp_ctx, build_model, git_commit  # noqa: E402

# Seconds of the chunk handed to the teacher. None = the whole thing.
SEGMENT_SECONDS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.008)


def synthetic_batch(batch_size, chunk_samples, sample_rate, device, seed=42):
    """Seeded noise at a realistic level. Audio CONTENT does not affect timing,
    and using noise keeps this runnable on Kaggle without the rendered corpus."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    make = lambda n: (torch.randn(batch_size, n, generator=generator) * 0.05
                      ).to(device)
    return (make(chunk_samples), make(chunk_samples),
            make(5 * sample_rate),
            torch.zeros(batch_size, dtype=torch.bool, device=device))


def time_steps(model, loss_fn, batch, n_steps, warmup, device, teacher=None,
               window_starts=None, use_amp=True):
    """Mean seconds per full training step: forward, loss, backward, step.

    MIRRORS scripts/train.py EXACTLY, and it has to. Measured on a T4
    2026-09-11: without AMP the baseline is 4.765 s/step and 12.23 GB peak,
    against the RECORDED 0.674 s/step -- 7x slow, because real training runs
    fp16 with a GradScaler. Every overhead percentage measured against an fp32
    baseline would have been meaningless.

    autocast wraps the MODEL FORWARD ONLY. The losses carry 1e-12 epsilons
    inside log10 and fp16's smallest normal is ~6e-5, so they underflow to zero
    and return NaN -- see amp_ctx in train.py. The teacher is part of the loss
    and stays fp32 with it.

    Warmup matters on CUDA: the first steps pay kernel autotuning and allocator
    growth, and including them would overstate the cost badly.
    """
    mixture, target, enrolment, crop_absent = batch
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def one_step():
        if teacher is not None:
            loss_fn.enrolment_embedding = teacher.embed_enrolment(enrolment)
        optimiser.zero_grad()
        with amp_ctx(use_amp):
            output = model(mixture, enrolment)
        if window_starts is not None:
            loss_fn.window_starts = window_starts
        loss, parts = loss_fn(target, output.float(), mixture, crop_absent)
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()
        return parts

    for _ in range(warmup):
        one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    started = time.time()
    for _ in range(n_steps):
        parts = one_step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = (time.time() - started) / n_steps

    peak_gb = (torch.cuda.max_memory_allocated() / 2**30
               if device.type == "cuda" else float("nan"))
    return seconds, peak_gb, parts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="experiments/configs/bsrnn_state.yaml")
    parser.add_argument("--teacher", required=True)
    parser.add_argument("--ecapa-dir", default="../ecapa_pretrained")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="TRIALS. both_directions doubles it into examples.")
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true",
                        help="fp32. The config default is amp: true, and the "
                             "recorded 0.674 s/step baseline is an AMP number, "
                             "so fp32 is a different experiment.")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    config = yaml.safe_load(Path(arguments.config).read_text())
    device = torch.device(arguments.device)
    sample_rate = int(config["data"]["sample_rate"])
    chunk_samples = int(round(float(config["data"]["chunk_s"]) * sample_rate))
    # batch_size counts TRIALS; both_directions makes a step see twice as many
    # examples, which is what the teacher's cost actually scales with.
    trials = arguments.batch_size or int(config["data"]["batch_size"])
    examples = trials * (2 if config["data"].get("both_directions") else 1)

    print(f"device {device}   batch {trials} trials = {examples} examples   "
          f"chunk {config['data']['chunk_s']} s")
    if device.type == "cuda":
        print(f"gpu    {torch.cuda.get_device_name(0)}   "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")

    use_amp = (not arguments.no_amp) and bool(config["training"].get("amp", False)) \
        and device.type == "cuda"
    print(f"amp    {use_amp}   (config says {config['training'].get('amp')})")

    model = build_model(config).to(device)
    batch = synthetic_batch(examples, chunk_samples, sample_rate, device)

    loss_kwargs = dict(
        wm=float(config["loss"]["w_m"]), w=float(config["loss"]["w"]),
        wg=float(config["loss"].get("w_g", 0.0)),
        tau_pres=float(config["loss"]["tau_pres"]),
        tau_abs=float(config["loss"]["tau_abs"]),
        p=float(config["loss"]["p"]), windows=config["loss"]["windows_ms"],
        sample_rate=sample_rate,
        gain_delta_db=float(config["loss"].get("gain_delta_db", 3.0)))

    results = []

    baseline_seconds, baseline_gb, _ = time_steps(
        model, LossBSRNN(**loss_kwargs), batch, arguments.steps,
        arguments.warmup, device, use_amp=use_amp)
    print(f"\nbaseline, no teacher: {baseline_seconds:.3f} s/step   "
          f"{baseline_gb:.2f} GB peak")
    # The recorded figure for this exact configuration, so a misconfigured
    # profile is visible immediately rather than after the sweep.
    print("  recorded 2026-08-28 at batch 3 with AMP on a T4: 0.674 s/step")
    if use_amp and baseline_seconds > 2.0:
        print("  *** that is far above the recorded baseline. Check amp and")
        print("      batch size before trusting any overhead below. ***")
    results.append(dict(segment_seconds=0.0, n_windows=0,
                        seconds_per_step=baseline_seconds, peak_gb=baseline_gb,
                        L_state=None, overhead_pct=0.0))

    teacher = StateTeacher(arguments.teacher, arguments.ecapa_dir,
                           device=str(device))
    loss_fn = LossBSRNNState(teacher=teacher, w_state=1.0, **loss_kwargs)

    print(f"\n{'segment':>9} {'windows':>8} {'s/step':>9} {'overhead':>10} "
          f"{'peak GB':>9} {'L_state':>9}")
    for segment in SEGMENT_SECONDS:
        segment_samples = min(int(round(segment * sample_rate)), chunk_samples)
        starts = teacher.window_starts(segment_samples)
        if not starts:
            continue
        seconds, peak_gb, parts = time_steps(
            model, loss_fn, batch, arguments.steps, arguments.warmup, device,
            teacher=teacher, window_starts=starts, use_amp=use_amp)
        overhead = 100 * (seconds - baseline_seconds) / baseline_seconds
        print(f"{segment:>8.2f}s {len(starts):>8} {seconds:>9.3f} "
              f"{overhead:>9.0f}% {peak_gb:>9.2f} {parts['L_state']:>9.4f}")
        results.append(dict(segment_seconds=segment, n_windows=len(starts),
                            seconds_per_step=seconds, peak_gb=peak_gb,
                            L_state=parts["L_state"], overhead_pct=overhead))

    # What the numbers mean for the session cap, which is the binding constraint.
    print()
    print("A 16-epoch run at 9,955 trials took 10.5 h against a 12 h Kaggle cap")
    print("(decisions-m2.md 2026-09-04), so the budget is about +14 % per step.")
    for row in results[1:]:
        verdict = "FITS" if row["overhead_pct"] <= 14 else "does not fit"
        print(f"  {row['segment_seconds']:.2f} s / {row['n_windows']:2d} windows"
              f"  ->  10.5 h becomes {10.5 * row['seconds_per_step'] / baseline_seconds:.1f} h"
              f"   {verdict}")
    print()
    print("L_state must be roughly CONSTANT across segment lengths. If it drifts,")
    print("the shorter context is changing what the teacher says, and a cheap")
    print("setting that changes the measurement is not a saving.")

    out_dir = Path(arguments.out or
                   f"experiments/results/{date.today().isoformat()}"
                   f"-state-teacher-cost")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(dict(
        date=date.today().isoformat(), git_commit=git_commit(),
        device=str(device),
        gpu=(torch.cuda.get_device_name(0) if device.type == "cuda" else None),
        config=arguments.config, teacher=arguments.teacher,
        teacher_describe=teacher.describe(),
        batch_trials=trials, batch_examples=examples,
        chunk_s=float(config["data"]["chunk_s"]),
        steps=arguments.steps, warmup=arguments.warmup,
        amp=use_amp,
        baseline_seconds_per_step=baseline_seconds,
        rows=results), indent=2))
    plot(results, baseline_seconds, out_dir, device)
    print(f"\nwrote {out_dir}")


def plot(results, baseline_seconds, out_dir, device):
    """Two panels: what it costs, and whether it changes what the teacher says."""
    import matplotlib as mpl
    mpl.use("Agg")
    import matplotlib.pyplot as plt

    # Journal style, matching notebooks/eda_data_construction.ipynb so the
    # figure drops into the report without being redrawn.
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6,
        "ytick.major.width": 0.6, "axes.spines.top": False,
        "axes.spines.right": False, "figure.dpi": 140, "savefig.dpi": 300,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
    })
    INK, ACCENT, ACCENT2 = "#222222", "#0B6E99", "#C1440E"
    TEXTWIDTH_IN = 6.3

    rows = [r for r in results if r["n_windows"] > 0]
    windows = [r["n_windows"] for r in rows]
    hours = [10.5 * r["seconds_per_step"] / baseline_seconds for r in rows]
    memory = [r["peak_gb"] for r in rows]
    state = [r["L_state"] for r in rows]

    figure, (cost_axis, drift_axis) = plt.subplots(
        1, 2, figsize=(TEXTWIDTH_IN, 2.5))

    cost_axis.plot(windows, hours, "o-", color=ACCENT, lw=1.3, ms=4)
    cost_axis.axhline(12.0, color=ACCENT2, lw=1.0, ls="--")
    cost_axis.text(windows[0], 12.1, "Kaggle session cap", fontsize=7,
                   color=ACCENT2, va="bottom")
    cost_axis.axhline(10.5, color=INK, lw=0.6, ls=":", alpha=0.6)
    cost_axis.text(windows[-1], 10.4, "baseline, no teacher", fontsize=7,
                   color=INK, ha="right", va="top", alpha=0.7)
    cost_axis.set_xlabel("windows scored (1 s each, 0.25 s hop)")
    cost_axis.set_ylabel("projected 16-epoch run (h)")
    cost_axis.set_title("Cost against the session cap")

    if device.type == "cuda" and np.isfinite(memory).all():
        memory_axis = cost_axis.twinx()
        memory_axis.plot(windows, memory, "s--", color=INK, lw=0.8, ms=3,
                         alpha=0.5)
        memory_axis.set_ylabel("peak GPU memory (GB)", color=INK, alpha=0.7)
        memory_axis.spines["top"].set_visible(False)

    drift_axis.plot(windows, state, "o-", color=ACCENT, lw=1.3, ms=4)
    if state and state[-1] is not None:
        drift_axis.axhline(state[-1], color=INK, lw=0.6, ls=":", alpha=0.6)
        drift_axis.text(windows[0], state[-1], " full chunk", fontsize=7,
                        color=INK, va="bottom", alpha=0.7)
    drift_axis.set_xlabel("windows scored")
    drift_axis.set_ylabel("L_state (nats)")
    drift_axis.set_title("Does a shorter context change the answer?")

    path = out_dir / "state_teacher_cost.pdf"
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"))
    print(f"figure -> {path}")


if __name__ == "__main__":
    main()
