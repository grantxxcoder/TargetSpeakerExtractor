"""Training curves for the report: the baseline and the extension on shared axes.

    ../tse_venv/bin/python scripts/plot_training_curves.py \
        --out report/figures/training_curves.pdf [--layout terms-wer|full|loss-wer|sisdr-wer]

`--layout` picks the panels (LAYOUTS). The default, `terms-wer`, is the report
figure (chosen 2026-10-08): the target-present loss split into its three
weighted terms, L_pres, w_m L_MR and w_g L_gain, side by side, which add up to
the loss; then word error below. `full` is the earlier three-row figure (the
loss, then L_abs with the silence bar, then word error); `loss-wer` keeps the
loss alone and `sisdr-wer` shows L_pres as SI-SDR (dB).

Every panel shares the epoch axis (never a twin y-axis -- the panels have
different units). Colour is the run (baseline orange, extension blue); line
style is the split (training dashed, validation solid). Epochs where either run
fails the silence bar `select_abs_max` are shaded in every panel.

The word-error panel is sir0_val `both` (n=103): one filled diamond per
live-judge run, labelled with the epoch's mean; each run's in-training Whisper
probe (40 clips) as a thin line. Whisper scores of saved epochs are not drawn
(removed 2026-10-08: the figure reports the judge).

The baseline is the matched-protocol run (decisions-m2.md 2026-10-07): the
baseline model trained under the extension's protocol, so the two runs differ in
the model only. The original baseline run (2026-09-04, 16 epochs, chosen on the
loss) is no longer drawn.

w_m, w_g and the silence bar are read from each run's own config copy, never
typed in here; the runs must agree on the bar. Chosen epochs are given, not
recomputed: both are picked on the judge (decisions-m2.md 2026-09-26 and
2026-10-07). None = not picked yet.

Judge results are FOUND, not listed by value: any results.json matching a run's
patterns is plotted, after checking it is sir0_val / both / 103 trials. So
re-running this after a new evaluation adds it without editing this file.
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
    "mathtext.fontset": "stix",   # Times-like maths, to match the serif text
})
TEXTWIDTH_IN = 6.3
INK, ACCENT, ACCENT2 = "#222222", "#0B6E99", "#C1440E"
MUTED, SHADE = "#6B6B6B", "#ECECEC"
MINUS = "−"

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "experiments/results"

JUDGED_EPOCHS = (13, 15, 18, 21, 27)   # e21's judged set; the matched run reuses it
RESUME = 13.5                          # both runs' second Kaggle session starts at 14

# The figure's inputs. `judge` maps epoch -> glob patterns under RES.
# Drawn in this order; on an epoch both runs scored, the first is nudged left.
RUNS = {
    "Baseline": {
        "label": "Baseline, matched protocol",
        "colour": ACCENT2,
        "history": ["2026-10-07-train-sir0-baseline-matched-s1"],   # session 1 of 2
        "chosen": None,                       # picked on the judge after session 2
        "lowest_marker": False,
        "judge": {e: [f"*-eval-baseline-matched-e{e}-judge",
                      f"*-eval-baseline-matched-e{e}-judge-r*"] for e in JUDGED_EPOCHS},
    },
    "Extension": {
        "label": "Extension",
        "colour": ACCENT,
        "history": ["2026-09-24-train-sir0-cuecontext-wer",
                    "2026-09-24-train-sir0-cuecontext-wer-resume"],
        "chosen": 21,                         # judge pick, decisions-m2 2026-09-26
        "lowest_marker": True,
        "judge": {e: [f"*-eval-cuecontext-wer-e{e}-judge",
                      f"*-eval-cuecontext-wer-e{e}-judge-r*"] for e in JUDGED_EPOCHS},
    },
}


# Loss panels. `f(row, prefix, config)` reads one history row; prefix is "train_"
# or "val_". Every panel draws both runs, training dashed and validation solid.
PANELS = {
    "present": {"ylabel": "Target-present loss\n(lower is better)", "marker": "o",
                "f": lambda r, p, c: present_branch(r, p, c)},
    "absent": {"ylabel": "Level when target\nabsent (dB, lower is quieter)", "marker": "s",
               "f": lambda r, p, c: float(r[p + "L_abs"])},
    # L_pres is the floored SI-SDR negated (CARTSE eq 1, Li et al. 2026), so its
    # negative is the SI-SDR in dB.
    "sisdr": {"ylabel": "Separation, SI-SDR\n(dB, higher is better)", "marker": "o",
              "f": lambda r, p, c: -float(r[p + "L_pres"])},
    # The three terms of the target-present loss, each with its weight, so the
    # three panels add up to the "present" panel.
    "pres": {"title": r"Separation, $L_\mathrm{pres}$", "marker": "o",
             "f": lambda r, p, c: float(r[p + "L_pres"])},
    "mr": {"title": r"Spectrum, $w_\mathrm{m} L_\mathrm{MR}$", "marker": "o",
           "f": lambda r, p, c: float(c["loss"]["w_m"]) * float(r[p + "L_MR"])},
    "gain": {"title": r"Level, $w_\mathrm{g} L_\mathrm{gain}$", "marker": "o",
             "f": lambda r, p, c: float(c["loss"].get("w_g", 0.0)) * float(r[p + "L_gain"])},
}

# Figure layouts: rows top to bottom, "wer" = the word-error panel. `notes` is
# the panel carrying the session-2 and lowest-loss labels, `chosen` the one
# carrying the chosen-epoch label. "terms-wer" is the report figure (2026-10-08);
# "full" is the earlier three-row one.
LAYOUTS = {
    "full": {"rows": [["present"], ["absent"], ["wer"]], "notes": "absent",
             "chosen": "present", "height": 5.6},
    "loss-wer": {"rows": [["present"], ["wer"]], "notes": "wer", "chosen": "present",
                 "height": 4.0},
    "sisdr-wer": {"rows": [["sisdr"], ["wer"]], "notes": "wer", "chosen": "wer",
                  "height": 4.0},
    "terms-wer": {"rows": [["pres", "mr", "gain"], ["wer"]], "notes": "wer", "chosen": "wer",
                  "height": 4.4, "ylabel": "Loss term\n(lower is better)"},
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


def scored_runs(patterns):
    """LCF-WER of every matching run, checked to be the headline set."""
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
    judged = {e: v for e, p in spec["judge"].items() if (v := scored_runs(p))}
    return rows, configs[0], judged


def separate(texts, ax):
    """Nudge overlapping labels upward until none overlap. Vertical only, so each
    label keeps its place beside its own diamond."""
    renderer = ax.figure.canvas.get_renderer()
    y0, y1 = ax.get_ylim()
    per_px = (y1 - y0) / ax.get_window_extent(renderer).height
    for _ in range(100):
        boxes = [t.get_window_extent(renderer) for t in texts]
        order = sorted(range(len(texts)), key=lambda k: boxes[k].y0)
        clash = next(((a, b) for i, a in enumerate(order) for b in order[i + 1:]
                      if boxes[a].overlaps(boxes[b])), None)
        if clash is None:
            return
        a, b = clash
        x, y = texts[b].get_position()
        texts[b].set_position((x, y + (boxes[a].y1 - boxes[b].y0 + 2) * per_px))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", type=Path, default=None, help="also write a PNG here")
    parser.add_argument("--layout", choices=sorted(LAYOUTS), default="terms-wer",
                        help="which panels to draw (LAYOUTS); default is the report figure")
    args = parser.parse_args()
    layout = LAYOUTS[args.layout]
    rows_spec = layout["rows"]

    loaded = {name: load(name, spec) for name, spec in RUNS.items()}
    bars = {float(config["training"]["select_abs_max"]) for _, config, _ in loaded.values()}
    if len(bars) != 1:
        raise SystemExit(f"runs disagree on the silence bar: {sorted(bars)}")
    bar = bars.pop()
    last_ep = max(int(rows[-1]["epoch"]) for rows, *_ in loaded.values())
    failing = sorted({int(r["epoch"]) for rows, *_ in loaded.values() for r in rows
                      if float(r["val_L_abs"]) > bar})

    # One gridspec row per layout row; a row holding several panels is split
    # across the width. Every panel shares the epoch axis.
    fig = plt.figure(figsize=(TEXTWIDTH_IN, layout["height"]))
    split = any(len(names) > 1 for names in rows_spec)
    grid = fig.add_gridspec(len(rows_spec), 1, hspace=0.4 if split else 0.15)
    axes, first = {}, None
    for i, names in enumerate(rows_spec):
        cells = grid[i].subgridspec(1, len(names), wspace=0.35)
        for k, panel in enumerate(names):
            axes[panel] = fig.add_subplot(cells[0, k], sharex=first)
            first = first or axes[panel]
    wer = axes["wer"]
    loss_panels = [p for names in rows_spec for p in names if p != "wer"]
    notes = axes[layout["notes"]]
    notes_on_top = layout["notes"] != "wer"   # the word-error panel is empty at the bottom

    for ax in axes.values():
        for e in failing:
            ax.axvspan(e - 0.5, e + 0.5, color=SHADE, lw=0, zorder=0)
        ax.axvline(RESUME, color=MUTED, lw=0.6, ls=(0, (1, 2)), zorder=1)
    notes.text(RESUME + (0.3 if notes_on_top else -0.3), 0.03, "session 2",
               transform=notes.get_xaxis_transform(), ha="left" if notes_on_top else "right",
               va="bottom", fontsize=7, color=MUTED)
    if "absent" in axes:
        axes["absent"].axhline(bar, color=INK, lw=0.7, ls=":", zorder=1)
        axes["absent"].text(last_ep + 0.5, bar + 0.3,
                            f"silence bar ({bar:g} dB)".replace("-", MINUS),
                            ha="right", va="bottom", fontsize=7, color=MUTED)

    marks = {}   # epoch -> [(run index, name, values)], for dodging and labels
    chosen_labels = []
    for i, (name, spec) in enumerate(RUNS.items()):
        rows, config, judged = loaded[name]
        c = spec["colour"]
        ep = [int(r["epoch"]) for r in rows]

        for p in loss_panels:
            f = PANELS[p]["f"]
            axes[p].plot(ep, [f(r, "train_", config) for r in rows], "--", color=c, lw=1.1,
                         zorder=3)
            axes[p].plot(ep, [f(r, "val_", config) for r in rows], PANELS[p]["marker"] + "-",
                         color=c, lw=1.3, ms=2.6, zorder=3)
        if "val_content_wer" in rows[0]:
            wer.plot(ep, [float(r["val_content_wer"]) for r in rows], "-", color=c, lw=0.8,
                     alpha=0.55, zorder=2)

        chosen = spec["chosen"]
        if chosen is not None:
            for ax in axes.values():
                ax.axvline(chosen, color=c, lw=0.8, zorder=2)
            chosen_labels.append((chosen, f"{name.lower()} chosen ({chosen})"))
        if spec["lowest_marker"]:
            # "Lowest validation loss" is always the target-present loss, whatever
            # the panels show; the circle sits on the first loss panel.
            val_pb = [present_branch(r, "val_", config) for r in rows]
            j = min(range(len(ep)), key=lambda k: val_pb[k])
            for ax in axes.values():
                ax.axvline(ep[j], color=c, lw=0.7, ls="--", zorder=2)
            p0 = loss_panels[0]
            axes[p0].plot(ep[j], PANELS[p0]["f"](rows[j], "val_", config), "o", ms=7,
                          mfc="none", mec=c, mew=1.0, zorder=4)
            notes.text(ep[j] + 0.4, 0.97 if notes_on_top else 0.03,
                       f"{name.lower()} lowest\nvalidation loss ({ep[j]})",
                       transform=notes.get_xaxis_transform(), ha="left",
                       va="top" if notes_on_top else "bottom", fontsize=7, color=INK)

        for e, values in judged.items():
            marks.setdefault(e, []).append((i, name, values))
        print(f"{name}: epochs {ep[0]}-{ep[-1]}, chosen {chosen}, judged "
              + ", ".join(f"e{e} {[round(v, 2) for v in judged[e]]}" for e in sorted(judged)))

    # With two chosen lines the earlier one is labelled on its left, so the two
    # labels share the top line without touching. In the word-error panel a lone
    # label also goes left: the in-training line peaks top right.
    chosen_ax = axes[layout["chosen"]]
    chosen_labels.sort()
    for k, (x, text) in enumerate(chosen_labels):
        left = (len(chosen_labels) > 1 and k == 0) or (chosen_ax is wer and len(chosen_labels) == 1)
        chosen_ax.text(x + (-0.3 if left else 0.3), 0.98, text,
                       transform=chosen_ax.get_xaxis_transform(),
                       ha="right" if left else "left", va="top", fontsize=7, color=INK)

    # Diamonds. Where both runs scored an epoch they are nudged apart and each run
    # labels on its own side (first run left, second right), so a label always
    # sits beside its own diamond. A lone run labels on the right unless the next
    # scored epoch is closer than 3 or this is the last epoch. Overlaps left after
    # that are separated vertically by `separate`.
    scored = sorted(marks)
    texts = []
    right_edge = last_ep + 0.5
    for e in scored:
        runs_here = sorted({m[0] for m in marks[e]})
        shared = len(runs_here) > 1
        dodge = {r: (k - (len(runs_here) - 1) / 2) * 0.4 for k, r in enumerate(runs_here)}
        later = [k for k in scored if k > e]
        crowded = e == last_ep or bool(later and later[0] - e < 3)
        for run, name, values in marks[e]:
            c = RUNS[name]["colour"]
            x = e + dodge[run]
            wer.plot([x] * len(values), values, "D", color=c, ms=4.5, mew=0, alpha=0.9,
                     zorder=4)
            left = dodge[run] < 0 if shared else crowded
            if not left:
                right_edge = max(right_edge, x + 1.6)
            texts.append(wer.text(x + (-0.45 if left else 0.45), mean(values),
                                  f"{mean(values):.1f}", ha="right" if left else "left",
                                  va="center", fontsize=7, zorder=6, color=INK,
                                  bbox={"boxstyle": "square,pad=0.1", "fc": "white",
                                        "ec": "none"}))

    wer.set_xlabel("Epoch")
    wer.xaxis.set_major_locator(MultipleLocator(2 if last_ep < 16 else 4))
    wer.xaxis.set_minor_locator(MultipleLocator(1))
    wer.set_xlim(-0.5, right_edge)
    wer.set_ylim(15, 80)
    separate(texts, wer)
    for p in loss_panels:
        if "ylabel" in PANELS[p]:
            axes[p].set_ylabel(PANELS[p]["ylabel"])
        if "title" in PANELS[p]:
            axes[p].set_title(PANELS[p]["title"], fontsize=8)
    if "ylabel" in layout:
        axes[rows_spec[0][0]].set_ylabel(layout["ylabel"])
    wer.set_ylabel("Word error rate\n(%, lower is better)")
    for names in rows_spec[:-1]:
        if len(names) == 1:                    # a split row keeps its own epoch ticks
            axes[names[0]].tick_params(labelbottom=False)

    # Legend below the word-error panel. The report layout keeps its hand-set
    # position; the others sit just under the "Epoch" label.
    if args.layout == "full":                  # kept as first published
        legend_y = 0.035
    else:
        fig.canvas.draw()                      # lay out tick labels before measuring
        label = wer.xaxis.label.get_window_extent(fig.canvas.get_renderer())
        legend_y = fig.transFigure.inverted().transform((0, label.y0))[1] - 0.01
    # Three columns of two, filled top to bottom: runs | line styles | word error.
    fig.legend(handles=[*(Line2D([], [], color=spec["colour"], ls="-", marker="o", ms=2.6, lw=1.3)
                          for spec in RUNS.values()),
                        Line2D([], [], color=INK, ls="-", lw=1.3),
                        Line2D([], [], color=INK, ls="--", lw=1.1),
                        Line2D([], [], color=MUTED, lw=0.8),
                        Line2D([], [], color=MUTED, ls="none", marker="D", ms=4.5, mew=0)],
               labels=[*(spec["label"] for spec in RUNS.values()),
                       "Validation", "Training",
                       "Whisper in training (40 clips)", "Gemini, one per run (103 trials)"],
               loc="upper center", bbox_to_anchor=(0.5, legend_y), ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.4)
    fig.align_ylabels([axes[names[0]] for names in rows_spec])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    if args.preview is not None:
        fig.savefig(args.preview)
    plt.close(fig)
    print(args.out)


if __name__ == "__main__":
    main()
