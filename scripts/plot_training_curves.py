"""Training curves for the report: baseline and extension, three rows per model.

    ../tse_venv/bin/python scripts/plot_training_curves.py \
        --out report/figures/training_curves.pdf

Rows share y, columns share x (never a twin y-axis -- the rows have different units):

  top     target-present loss, the first bracket of the total loss,
          L_pres + w_m L_MR + w_g L_gain -- exactly what `select_on:
          present_branch` ranks on. Training dashed, validation solid.
  middle  L_abs, output level on target-absent crops relative to the mixture
          (dB), with the silence bar `select_abs_max`. Epochs failing it shaded.
  bottom  word error rate on sir0_val `both` (n=103): one diamond per live-judge
          run, labelled with the epoch's mean; the extension's in-training
          Whisper probe (40 clips) as a grey line. The baseline had no probe.

w_m, w_g and the silence bar are read from each run's own config copy, never
typed in here. The baseline's chosen epoch is recomputed (pass the bar, then
lowest target-present loss) and checked against results.json `best_val`. The
extension's is given: it was chosen on the judge, not by training
(decisions-m2.md 2026-09-26).

Judge results are FOUND, not listed by value: any results.json matching a run's
patterns is plotted, after checking it is sir0_val / both / 103 trials. So
re-running this after a new judge run adds it -- the baseline epochs 7 and 15
queued 2026-10-03 appear without editing this file.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import yaml  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# House style, copied from scripts/plot_state_teacher_cost.py.
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 140, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01, "pdf.fonttype": 42,
})
TEXTWIDTH_IN = 6.3
INK, ACCENT, ACCENT2 = "#222222", "#0B6E99", "#C1440E"
MUTED, SHADE = "#6B6B6B", "#ECECEC"
MINUS = "−"

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/results"

# The figure's inputs. `judge` maps epoch -> glob patterns under RES.
RUNS = {
    "Baseline": {
        "history": ["2026-09-04-train-sir0-10000"],
        "chosen": None,                       # recomputed from the run
        "lowest_marker": False,               # lowest validation loss IS the pick
        "resume": None,
        "judge": {
            6: ["2026-09-06-evaluate-10000-judge", "2026-09-27-eval-baseline-judge-r*"],
            7: ["*-eval-10000-e7-judge*"],
            15: ["*-eval-10000-e15-judge*"],
        },
    },
    "Extension": {
        "history": ["2026-09-24-train-sir0-cuecontext-wer",
                    "2026-09-24-train-sir0-cuecontext-wer-resume"],
        "chosen": 21,                         # judge pick, decisions-m2 2026-09-26
        "lowest_marker": True,
        "resume": 13.5,                       # second Kaggle session starts at 14
        "judge": {e: [f"*-eval-cuecontext-wer-e{e}-judge",
                      f"*-eval-cuecontext-wer-e{e}-judge-r*"]
                  for e in (13, 15, 18, 21, 27)},
    },
}


def run_config(run_dir):
    for path in sorted(run_dir.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        if isinstance(config, dict) and "loss" in config and "training" in config:
            return config
    raise FileNotFoundError(f"no training config in {run_dir}")


def present_branch(row, prefix, config):
    """Same sum as selection_score(mode='present_branch') in scripts/train.py."""
    w_m = float(config["loss"]["w_m"])
    w_g = float(config["loss"].get("w_g", 0.0))
    return (float(row[prefix + "L_pres"]) + w_m * float(row[prefix + "L_MR"])
            + w_g * float(row[prefix + "L_gain"]))


def find_key(tree, key):
    if isinstance(tree, dict):
        if key in tree:
            return tree[key]
        for value in tree.values():
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def recompute_choice(rows, config, run_dir):
    bar = float(config["training"]["select_abs_max"])
    eligible = [r for r in rows if float(r["val_L_abs"]) <= bar]
    best = min(eligible, key=lambda r: present_branch(r, "val_", config))
    recorded = find_key(json.loads((run_dir / "results.json").read_text()), "best_val")
    if recorded is not None and abs(present_branch(best, "val_", config) - float(recorded)) > 1e-6:
        raise SystemExit(f"recomputed pick epoch {best['epoch']} does not match "
                         f"{run_dir.name} best_val {recorded}")
    return int(best["epoch"])


def judge_runs(patterns):
    """LCF-WER of every matching judge run, checked to be the headline set."""
    values = []
    for d in sorted({p for pattern in patterns for p in RES.glob(pattern)}):
        results = json.loads((d / "results.json").read_text())
        if (results["split"], results["condition"], results["n_trials"]) != ("sir0_val", "both", 103):
            raise SystemExit(f"{d.name} is not sir0_val / both / 103")
        values.append(float(results["systems"]["estimate"]["lcf_wer"]))
    return values


def load(name, spec):
    run_dirs = [RES / d for d in spec["history"]]
    rows = [r for d in run_dirs for r in csv.DictReader((d / "history.csv").open())]
    configs = [run_config(d) for d in run_dirs]
    for c in configs[1:]:
        if (c["loss"], c["training"]["select_abs_max"]) != (configs[0]["loss"], configs[0]["training"]["select_abs_max"]):
            raise SystemExit(f"{name}: sessions disagree on the loss config")
    config = configs[0]
    chosen = spec["chosen"] if spec["chosen"] is not None else recompute_choice(rows, config, run_dirs[0])
    judged = {e: v for e, p in spec["judge"].items() if (v := judge_runs(p))}
    return rows, config, chosen, judged


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None, help="also write a PNG here")
    args = parser.parse_args()

    widths = []
    loaded = {}
    for name, spec in RUNS.items():
        loaded[name] = load(name, spec)
        widths.append(len(loaded[name][0]))

    fig, axes = plt.subplots(3, len(RUNS), figsize=(TEXTWIDTH_IN, 5.6), sharey="row",
                             sharex="col", gridspec_kw={"hspace": 0.15, "wspace": 0.08,
                                                        "width_ratios": widths})
    for col, (name, spec) in enumerate(RUNS.items()):
        rows, config, chosen, judged = loaded[name]
        bar = float(config["training"]["select_abs_max"])
        ep = [int(r["epoch"]) for r in rows]
        val_pb = [present_branch(r, "val_", config) for r in rows]
        absent_val = [float(r["val_L_abs"]) for r in rows]
        top, mid, bot = axes[:, col]
        lowest = ep[min(range(len(ep)), key=lambda i: val_pb[i])] if spec["lowest_marker"] else None

        for ax in (top, mid, bot):
            for e, a in zip(ep, absent_val):
                if a > bar:
                    ax.axvspan(e - 0.5, e + 0.5, color=SHADE, lw=0, zorder=0)
            ax.axvline(chosen, color=INK, lw=0.7, zorder=2)
            if spec["resume"] is not None:
                ax.axvline(spec["resume"], color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
            if lowest is not None:
                ax.axvline(lowest, color=ACCENT, lw=0.7, ls="--", zorder=2)

        top.plot(ep, [present_branch(r, "train_", config) for r in rows], "--",
                 color=ACCENT, lw=1.1, zorder=3)
        top.plot(ep, val_pb, "o-", color=ACCENT, lw=1.3, ms=2.6, zorder=3)
        top.set_title(name)
        top.text(chosen + 0.3, 0.98, f"chosen ({chosen})", transform=top.get_xaxis_transform(),
                 ha="left", va="top", fontsize=7)
        if lowest is not None:
            i = ep.index(lowest)
            top.plot(lowest, val_pb[i], "o", ms=7, mfc="none", mec=ACCENT, mew=1.0, zorder=4)
            mid.text(lowest + 0.4, 0.97, f"lowest validation\nloss ({lowest})", color=ACCENT,
                     transform=mid.get_xaxis_transform(), ha="left", va="top", fontsize=7)

        mid.plot(ep, [float(r["train_L_abs"]) for r in rows], "--", color=ACCENT2, lw=1.1, zorder=3)
        mid.plot(ep, absent_val, "s-", color=ACCENT2, lw=1.3, ms=2.6, zorder=3)
        mid.axhline(bar, color=INK, lw=0.7, ls=":", zorder=1)
        if spec["resume"] is not None:
            mid.text(spec["resume"] + 0.3, 0.03, "session 2", transform=mid.get_xaxis_transform(),
                     ha="left", va="bottom", fontsize=7, color=MUTED)
        if col == len(RUNS) - 1:
            mid.text(ep[-1] + 0.5, bar + 0.3, f"silence bar ({bar:g} dB)".replace("-", MINUS),
                     ha="right", va="bottom", fontsize=7, color=MUTED)

        if "val_content_wer" in rows[0]:
            bot.plot(ep, [float(r["val_content_wer"]) for r in rows], "-", color=MUTED,
                     lw=0.9, alpha=0.8, zorder=2)
        else:
            bot.text(0.97, 0.97, "Whisper not run during\nbaseline training",
                     transform=bot.transAxes, ha="right", va="top", fontsize=7, color=MUTED)
        epochs_judged = sorted(judged)
        for e in epochs_judged:
            values = judged[e]
            bot.plot([e] * len(values), values, "D", color=ACCENT2, ms=4.5, mew=0,
                     alpha=0.9, zorder=4)
            later = [k for k in epochs_judged if k > e]
            left = e == ep[-1] or (later and later[0] - e < 3)
            bot.text(e + (-0.6 if left else 0.6), mean(values), f"{mean(values):.1f}",
                     ha="right" if left else "left", va="center", fontsize=7, color=INK)

        bot.set_xlabel("Epoch")
        bot.xaxis.set_major_locator(MultipleLocator(2 if len(ep) <= 16 else 4))
        bot.xaxis.set_minor_locator(MultipleLocator(1))
        bot.set_xlim(ep[0] - 0.5, ep[-1] + 0.5)
        print(f"{name}: chosen {chosen}, judged "
              + ", ".join(f"e{e} {[round(v, 2) for v in judged[e]]}" for e in epochs_judged))

    axes[2, 0].set_ylim(15, 80)
    axes[0, 0].set_ylabel("Target-present loss\n(lower is better)")
    axes[1, 0].set_ylabel("Level when target\nabsent (dB, lower is quieter)")
    axes[2, 0].set_ylabel("Word error rate\n(%, lower is better)")
    fig.legend(handles=[Line2D([], [], color=INK, ls="-", marker="o", ms=2.6, lw=1.3),
                        Line2D([], [], color=INK, ls="--", lw=1.1),
                        Line2D([], [], color=MUTED, lw=0.9),
                        Line2D([], [], color=ACCENT2, ls="none", marker="D", ms=4.5, mew=0)],
               labels=["Validation", "Training", "Whisper in training (40 clips)",
                       "Gemini, one per run (103 trials)"],
               loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=4, frameon=False,
               handlelength=2.2, columnspacing=1.4)
    fig.align_ylabels(axes[:, 0])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.preview is not None:
        fig.savefig(args.preview)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
