"""Separate figures from the state-teacher cost sweep. decisions-pending.md D14.

    ../tse_venv/bin/python scripts/plot_state_teacher_cost.py \
        --results experiments/results/2026-09-11-state-teacher-cost/results.json

Writes THREE single-panel PDFs next to the results, each sized to half the text
width so any two sit side by side in LaTeX:

    state_teacher_cost.pdf     what it costs against the session cap
    state_teacher_drift.pdf    whether a shorter segment changes the answer
    state_teacher_memory.pdf   peak GPU memory, which turned out not to bind

Separate rather than a multi-panel figure because they answer different
questions and belong in different parts of the write-up: the cost figure argues
a scheduling decision, the drift figure is a finding about the teacher itself.

WHAT THE SWEEP FOUND, 2026-09-11, Tesla T4, batch 3 trials = 6 examples, AMP:
  baseline 0.669 s/step against the recorded 0.674 -- the profile is sound
  the full 13-window term costs +53 %, i.e. 16.0 h against a 12 h cap
  L_state moves 0.2542 -> 0.0623 across segment lengths, a factor of FOUR

That last line is the important one. Shortening the segment does not buy the
same measurement more cheaply, it produces a DIFFERENT measurement -- so the
cheap settings are a different teacher, not a saving, and w_state would need
re-deriving for each. The lever is the epoch count instead: the 10.5 h baseline
is 16 epochs, and no run has ever selected past epoch 6.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Journal style, matching notebooks/eda_data_construction.ipynb so these drop
# into the report without being redrawn.
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})
TEXTWIDTH_IN = 6.3
HALF = TEXTWIDTH_IN * 0.49
INK, ACCENT, ACCENT2 = "#222222", "#0B6E99", "#C1440E"

SESSION_CAP_HOURS = 12.0
BASELINE_HOURS_16_EPOCHS = 10.5


def save(figure, path):
    figure.savefig(path)
    figure.savefig(path.with_suffix(".png"))
    plt.close(figure)
    print(f"  {path.name}")


def cost_figure(rows, baseline, out_dir, epochs_alternative=10):
    """Hours against windows, with the cap and the shorter-run alternative."""
    windows = [r["n_windows"] for r in rows]
    hours = [BASELINE_HOURS_16_EPOCHS * r["seconds_per_step"] / baseline
             for r in rows]
    shorter = [h * epochs_alternative / 16 for h in hours]

    figure, axis = plt.subplots(figsize=(HALF, 2.3))
    axis.plot(windows, hours, "o-", color=ACCENT2, lw=1.3, ms=4,
              label="16 epochs")
    axis.plot(windows, shorter, "s-", color=ACCENT, lw=1.3, ms=3.5,
              label=f"{epochs_alternative} epochs")

    axis.axhline(SESSION_CAP_HOURS, color=INK, lw=0.9, ls="--")
    axis.text(windows[0], SESSION_CAP_HOURS + 0.25, "session cap",
              fontsize=7, color=INK, va="bottom")
    axis.axhline(BASELINE_HOURS_16_EPOCHS, color=INK, lw=0.6, ls=":", alpha=0.5)
    axis.text(windows[-1], BASELINE_HOURS_16_EPOCHS - 0.25, "no teacher",
              fontsize=7, color=INK, ha="right", va="top", alpha=0.7)

    axis.set_xlabel("windows scored")
    axis.set_ylabel("run length (h)")
    axis.set_xticks(windows)
    axis.legend(frameon=False, loc="upper left")
    save(figure, out_dir / "state_teacher_cost.pdf")


def drift_figure(rows, out_dir):
    """L_state against windows. The finding, so it gets its own figure."""
    windows = [r["n_windows"] for r in rows]
    values = [r["L_state"] for r in rows]

    figure, axis = plt.subplots(figsize=(HALF, 2.3))
    axis.plot(windows, values, "o-", color=ACCENT, lw=1.3, ms=4)

    # What "no drift" would have looked like: the full-length value, flat.
    axis.axhline(values[-1], color=INK, lw=0.9, ls="--")
    axis.text(windows[0], values[-1] - 0.012, "full-length value",
              fontsize=7, color=INK, va="top")
    axis.annotate(f"x{values[0] / values[-1]:.1f}",
                  xy=(windows[0], values[0]),
                  xytext=(windows[0] + 1.8, values[0]),
                  fontsize=8, color=ACCENT2, va="center",
                  arrowprops=dict(arrowstyle="-", color=ACCENT2, lw=0.7))

    axis.set_xlabel("windows scored")
    axis.set_ylabel(r"$L_{\mathrm{state}}$ (nats)")
    axis.set_xticks(windows)
    axis.set_ylim(0, max(values) * 1.15)
    save(figure, out_dir / "state_teacher_drift.pdf")


def memory_figure(rows, baseline_gb, out_dir, card_gb=14.6):
    """Peak memory, which turned out not to be the binding constraint."""
    windows = [r["n_windows"] for r in rows]
    memory = [r["peak_gb"] for r in rows]

    figure, axis = plt.subplots(figsize=(HALF, 2.3))
    axis.plot(windows, memory, "o-", color=ACCENT, lw=1.3, ms=4)
    axis.axhline(card_gb, color=INK, lw=0.9, ls="--")
    axis.text(windows[0], card_gb - 0.4, "T4 memory", fontsize=7, color=INK,
              va="top")
    axis.axhline(baseline_gb, color=INK, lw=0.6, ls=":", alpha=0.5)
    axis.text(windows[-1], baseline_gb - 0.4, "no teacher", fontsize=7,
              color=INK, ha="right", va="top", alpha=0.7)

    axis.set_xlabel("windows scored")
    axis.set_ylabel("peak GPU memory (GB)")
    axis.set_xticks(windows)
    axis.set_ylim(0, card_gb * 1.08)
    save(figure, out_dir / "state_teacher_memory.pdf")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", required=True,
                        help="results.json from profile_state_teacher.py")
    parser.add_argument("--epochs-alternative", type=int, default=10,
                        help="the shorter run drawn beside the 16-epoch line. "
                             "10 because no run has selected past epoch 6.")
    parser.add_argument("--out", default=None)
    arguments = parser.parse_args()

    results = json.loads(Path(arguments.results).read_text())
    rows = [r for r in results["rows"] if r["n_windows"] > 0]
    baseline = results["baseline_seconds_per_step"]
    baseline_gb = next(r["peak_gb"] for r in results["rows"]
                       if r["n_windows"] == 0)
    out_dir = Path(arguments.out or Path(arguments.results).parent)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{results.get('gpu', 'unknown GPU')}   "
          f"batch {results['batch_examples']} examples   "
          f"amp {results.get('amp')}   baseline {baseline:.3f} s/step")
    print(f"writing to {out_dir}")
    cost_figure(rows, baseline, out_dir, arguments.epochs_alternative)
    drift_figure(rows, out_dir)
    memory_figure(rows, baseline_gb, out_dir)

    full = rows[-1]
    hours = BASELINE_HOURS_16_EPOCHS * full["seconds_per_step"] / baseline
    drift = rows[0]["L_state"] / full["L_state"]
    print(f"\nfull {full['n_windows']} windows: +{full['overhead_pct']:.0f} %, "
          f"{hours:.1f} h at 16 epochs, "
          f"{hours * arguments.epochs_alternative / 16:.1f} h at "
          f"{arguments.epochs_alternative}")
    print(f"drift across segment lengths: x{drift:.1f} -- subsampling is not a "
          f"saving, it is a different teacher")


if __name__ == "__main__":
    main()
