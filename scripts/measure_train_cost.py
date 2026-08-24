"""What one training step costs, in peak memory and wall time, per batch size.

The question this answers: which batch size fits on this machine, and what does
an epoch cost before committing hours to one. Measures only -- it changes no
config and picks nothing. Companion to scripts/measure_vad_impact.py.

    ../tse_venv/bin/python scripts/measure_train_cost.py --batches 1,2,3,4

SYNTHETIC tensors, not the real loader. Peak memory and step time depend on
tensor SHAPES, not on their contents, so a random batch of the right shape
measures the same thing while removing 27 GB of disk I/O from the timing. The
one content-dependent thing that matters is the present/absent split, because
the two loss branches cost differently, so absent crops are planted at the
measured 0.297 rate (decisions-m1.md 2026-08-18).

Each batch size runs in its own SUBPROCESS. A step that exhausts memory then
dies alone instead of taking the sweep -- or the desktop -- with it; the parent
records the failure and stops climbing. This script exists because
systemd-oomd killed VSCode during setup on 2026-08-24, and guessing a batch
size is how that happens again.

Reports peak RSS from VmHWM (the kernel's own high-water mark, not a sampled
guess) at three stages, so the number points at a cause:

    model      weights + optimiser state, no activations
    forward    + everything autograd stored for the backward pass
    step       + gradients, and the multi-resolution loss's own STFTs

Projections to a full epoch are labelled as projections and are NOT written to
docs/run_times.md. That file holds measured wall times only.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

ABSENT_RATE = 0.297      # measured, decisions-m1.md 2026-08-18
TRAIN_TRIALS = 19938     # data/manifests/train.meta.yaml n_trials


def peak_rss_mb():
    """VmHWM: the highest RSS this process has ever reached, kernel-tracked.

    Not a sample of current RSS -- a sampled reading misses the peak inside a
    backward pass, which is the only number that decides whether a batch fits.
    """
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            return int(line.split()[1]) / 1024
    return float("nan")


def measure_one(config, batch, chunk_s, steps):
    """One batch size, in this process. Returns the dict the parent prints."""
    import torch
    from train import build_loss_fn, build_model     # the model that trains

    sample_rate = int(config["data"]["sample_rate"])
    n_samples = int(chunk_s * sample_rate)
    n_enroll = int(5.0 * sample_rate)               # generator.yaml enrollment_length_s
    baseline_mb = peak_rss_mb()

    torch.manual_seed(int(config["seed"]))
    model = build_model(config)
    loss_fn = build_loss_fn(config)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=float(config["training"]["lr"]),
                                  weight_decay=float(config["training"]["weight_decay"]))
    n_params = sum(p.numel() for p in model.parameters())

    # One step first, so the optimiser has allocated its two moment buffers per
    # parameter -- measuring "model" before that understates the resting cost.
    crop_absent = torch.zeros(batch, dtype=torch.bool)
    n_absent = max(1, int(round(batch * ABSENT_RATE))) if batch > 1 else 0
    crop_absent[:n_absent] = True

    mixture = torch.randn(batch, n_samples) * 0.1
    enrollment = torch.randn(batch, n_enroll) * 0.1
    target = torch.randn(batch, n_samples) * 0.1
    target[crop_absent] = 0.0            # an absent crop's target is exactly zero

    model_mb = peak_rss_mb()

    # forward only
    t0 = time.perf_counter()
    s_output = model(mixture, enrollment)
    t_forward = time.perf_counter() - t0
    forward_mb = peak_rss_mb()

    loss, parts = loss_fn(target, s_output, mixture, crop_absent)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step_mb = peak_rss_mb()

    # Timed steps after the graph has been built once: the first step pays for
    # lazy allocator growth and would flatter every later one by comparison.
    times = []
    for _ in range(steps):
        t0 = time.perf_counter()
        s_output = model(mixture, enrollment)
        loss, parts = loss_fn(target, s_output, mixture, crop_absent)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        times.append(time.perf_counter() - t0)

    return {
        "batch": batch,
        "chunk_s": chunk_s,
        "n_parameters": n_params,
        "n_absent_planted": int(n_absent),
        "baseline_mb": round(baseline_mb, 1),
        "model_mb": round(model_mb, 1),
        "forward_mb": round(forward_mb, 1),
        "step_mb": round(step_mb, 1),
        # What the training itself costs, with the torch import subtracted.
        "step_over_baseline_mb": round(step_mb - baseline_mb, 1),
        "per_crop_mb": round((step_mb - baseline_mb) / batch, 1),
        "t_forward_s": round(t_forward, 3),
        "t_step_median_s": round(float(np.median(times)), 3),
        "t_step_min_s": round(min(times), 3),
        "loss_total": parts["total"],
        "threads": torch.get_num_threads(),
        "torch": torch.__version__,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="experiments/configs/bsrnn_baseline.yaml")
    ap.add_argument("--batches", default="1,2,3,4",
                    help="comma-separated batch sizes, climbing")
    ap.add_argument("--chunks", default=None,
                    help="comma-separated chunk_s values to sweep; default data.chunk_s. "
                         "chunk_s is the OTHER memory lever and scales the same way")
    ap.add_argument("--chunk-s", default=None, type=float, help=argparse.SUPPRESS)
    ap.add_argument("--steps", type=int, default=3, help="timed steps per batch size")
    ap.add_argument("--budget-mb", type=int, default=3500,
                    help="stop climbing once a step's peak exceeds this")
    ap.add_argument("--out", default=None,
                    help="default experiments/results/<today>-train-cost")
    # --child is the single-measurement worker the parent spawns. Not for hand use.
    ap.add_argument("--child", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    chunk_s = args.chunk_s if args.chunk_s is not None else float(config["data"]["chunk_s"])

    if args.child is not None:
        # One measurement, JSON on stdout. Isolated so an out-of-memory death
        # is one dead child, not a dead sweep.
        print(json.dumps(measure_one(config, args.child, chunk_s, args.steps)))
        return

    batches = [int(b) for b in args.batches.split(",")]
    chunks = ([float(c) for c in args.chunks.split(",")] if args.chunks else [chunk_s])
    rows = []
    for chunk in chunks:
        for batch in batches:
            print(f"chunk {chunk} s, batch {batch} ... ", end="", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", str(batch), "--config", args.config,
                 "--chunk-s", str(chunk), "--steps", str(args.steps)],
                capture_output=True, text=True)
            if proc.returncode != 0:
                tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
                print(f"FAILED ({proc.returncode}): {tail[0][:120]}")
                rows.append({"batch": batch, "chunk_s": chunk, "failed": True,
                             "returncode": proc.returncode, "stderr_tail": tail[0][:400]})
                break
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"{row['step_mb']:.0f} MB peak, {row['t_step_median_s']:.2f} s/step")
            # Climb no further on THIS chunk length; a shorter one may still fit.
            if row["step_mb"] > args.budget_mb:
                print(f"  stopping this chunk: past the {args.budget_mb} MB budget")
                break

    ok = [r for r in rows if not r.get("failed")]
    if not ok:
        sys.exit("no batch size completed -- nothing to report")

    out = Path(args.out) if args.out else (
        Path("experiments/results") / f"{date.today().isoformat()}-train-cost")
    out.mkdir(parents=True, exist_ok=True)

    import csv
    with open(out / "cost.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ok[0].keys()))
        writer.writeheader()
        writer.writerows(ok)

    # Linear fit per chunk length. Activation memory is linear in batch size,
    # so the slope IS the per-crop cost and the intercept is the fixed cost --
    # and reporting the residual lets the reader see whether it really is
    # linear rather than taking the extrapolation on faith.
    fits = {}
    for chunk in sorted({r["chunk_s"] for r in ok}):
        g = [r for r in ok if r["chunk_s"] == chunk]
        b = np.array([r["batch"] for r in g], dtype=float)
        m = np.array([r["step_mb"] for r in g], dtype=float)
        t = np.array([r["t_step_median_s"] for r in g], dtype=float)
        mfit = np.polyfit(b, m, 1) if len(g) > 1 else np.array([float("nan"), m[0]])
        tfit = np.polyfit(b, t, 1) if len(g) > 1 else np.array([float("nan"), t[0]])
        resid = float(np.max(np.abs(np.polyval(mfit, b) - m))) if len(g) > 1 else 0.0
        fits[chunk] = {"mem": mfit, "time": tfit, "resid": resid, "rows": g}

    plot(out, fits)
    print(f"\nWrote {out}/cost.csv, meta.yaml, batch_size_vs_memory.png")


def plot(out, fits):
    """Two panels: memory against batch size, and projected epoch cost.

    One series per chunk length, because chunk_s and batch size are the same
    lever -- both multiply the number of frames autograd has to keep.
    """
    import matplotlib
    matplotlib.use("Agg")           # no display in a background run
    import matplotlib.pyplot as plt

    span = np.linspace(1, 16, 64)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.5))

    for i, (chunk, f) in enumerate(sorted(fits.items())):
        g, c = f["rows"], f"C{i}"
        b = [r["batch"] for r in g]
        ax.plot(span, np.polyval(f["mem"], span), "--", color=c, alpha=.45)
        ax.plot(b, [r["step_mb"] for r in g], "o-", color=c,
                label=f'chunk {chunk} s: {f["mem"][1]:.0f} MB + {f["mem"][0]:.0f} MB/crop')

        hours = [(TRAIN_TRIALS // r["batch"]) * r["t_step_median_s"] / 3600 for r in g]
        ax2.plot(b, hours, "o-", color=c, label=f"chunk {chunk} s")
        for x, y in zip(b, hours):
            ax2.annotate(f"{y:.0f} h", (x, y), textcoords="offset points",
                         xytext=(0, 7), fontsize=8, ha="center", color=c)

    # The budget lines are what make the left panel readable off the page.
    for mb, txt, c in [(9200, "free with VSCode open", "0.35"),
                       (13000, "free with VSCode closed", "0.55")]:
        ax.axhline(mb, color=c, ls=":", lw=1.5)
        ax.text(0.3, mb, f" {txt}", color=c, fontsize=8, va="bottom")

    ax.set_xlabel("batch size")
    ax.set_ylabel("peak RSS for one training step (MB)")
    ax.set_title("Memory per training step -- CPU, measured")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.3)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16000)

    ax2.set_xlabel("batch size")
    ax2.set_ylabel("hours per epoch (train split, 19,938 trials)")
    ax2.set_title("PROJECTED epoch cost -- extrapolated, not measured")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=.3, which="both")
    ax2.set_xlim(0, 16)

    fig.tight_layout()
    fig.savefig(out / "batch_size_vs_memory.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
