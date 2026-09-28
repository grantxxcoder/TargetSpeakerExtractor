#!/usr/bin/env python3
"""Which epoch would each selection rule have kept? Re-read from logged history.

    python3 scripts/reselect_epochs.py

WHY THIS EXISTS. 2026-09-23 found that `present_branch` kept item 1a's epoch 9,
which scores WORSE THAN DOING NOTHING on content (ASR LCF-WER 65.63 against a
65.22 floor), and rejected epoch 15, which beats the floor by 12.5 points on the
same listener. This script asks the same question of every run on disk: does the
selection rule agree with itself across modes, and does any mode's pick change
if the volume term is removed?

NO CHECKPOINT, NO GPU, NO INFERENCE. Every number is recombined from the
`history.csv` a run already wrote, so this cannot fail on a machine without
torch and costs seconds.

The modes are `scripts/train.py:selection_score`'s, re-implemented here against
the logged columns rather than the live val_loss dict. They are kept in step by
`tests/test_reselect_modes.py`, which asserts the two agree.
"""

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "results"

# Fallbacks only. Every run archives its own config in the results dir and that
# is what is read; these are used when a run predates the archiving convention.
DEFAULT_W_M, DEFAULT_W_G = 9.62, 1.69


def read_config_weights(run_dir):
    """w_m, w_g, select_abs_max, select_on from the run's ARCHIVED config.

    The archived copy, not experiments/configs/, because the live file drifts:
    2026-09-21's checked-in config no longer reproduces the run it names.
    """
    w_m, w_g, bar, mode = DEFAULT_W_M, DEFAULT_W_G, None, "present_branch"
    for yaml_path in sorted(run_dir.glob("*.yaml")):
        if yaml_path.name == "meta.yaml":
            continue
        for line in yaml_path.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("w_m:"):
                w_m = float(stripped.split(":", 1)[1].split("#")[0])
            elif stripped.startswith("w_g:"):
                w_g = float(stripped.split(":", 1)[1].split("#")[0])
            elif stripped.startswith("select_abs_max:"):
                raw = stripped.split(":", 1)[1].split("#")[0].strip()
                bar = None if raw in ("null", "none", "~", "") else float(raw)
            elif stripped.startswith("select_on:"):
                mode = stripped.split(":", 1)[1].split("#")[0].strip()
    return w_m, w_g, bar, mode


def score(row, mode, w_m, w_g):
    """Mirror of train.py:selection_score, over logged val_* columns.

    Lower is better in every mode. `present_no_gain` and `spectrum` are the two
    modes train.py does not have; they are here to answer whether removing the
    volume term changes any past decision.
    """
    pres = float(row["val_L_pres"])
    mr = float(row["val_L_MR"])
    gain = float(row.get("val_L_gain") or "nan")
    if mode == "total":
        return float(row["val_total"])
    if mode == "present_branch":
        return pres + w_m * mr + (w_g * gain if not math.isnan(gain) else 0.0)
    if mode == "present_no_gain":
        return pres + w_m * mr
    if mode == "separation":
        return pres
    if mode == "spectrum":
        return mr
    raise ValueError(mode)


MODES = ["total", "present_branch", "present_no_gain", "separation", "spectrum"]


def pick(rows, mode, w_m, w_g, bar):
    """Best eligible epoch. The silence bar is a CONSTRAINT, as in train.py."""
    best_epoch, best_score = None, float("inf")
    for row in rows:
        if bar is not None and float(row["val_L_abs"]) > bar:
            continue
        value = score(row, mode, w_m, w_g)
        if math.isnan(value):
            continue
        if value < best_score:
            best_epoch, best_score = int(row["epoch"]), value
    return best_epoch


def main():
    runs = sorted(RESULTS.glob("*/history.csv"))
    print(f"{'run':34s} {'cfg':>4s} " + " ".join(f"{m[:9]:>9s}" for m in MODES)
          + "  n_ep  bar")
    disagree = []
    for history_path in runs:
        run_dir = history_path.parent
        rows = [r for r in csv.DictReader(open(history_path)) if r.get("val_L_pres")]
        if not rows:
            continue
        w_m, w_g, bar, cfg_mode = read_config_weights(run_dir)
        picks = {m: pick(rows, m, w_m, w_g, bar) for m in MODES}
        shipped = picks.get(cfg_mode)
        cells = " ".join(f"{('-' if picks[m] is None else picks[m]):>9}" for m in MODES)
        print(f"{run_dir.name:34s} {cfg_mode[:4]:>4s} {cells}  {len(rows):4d}  {bar}")
        if picks["present_branch"] != picks["present_no_gain"]:
            disagree.append((run_dir.name, picks["present_branch"],
                             picks["present_no_gain"]))
    print()
    print("Runs where DROPPING THE VOLUME TERM changes the kept epoch:")
    if not disagree:
        print("  none -- w_g is not what decides any past selection.")
    for name, a, b in disagree:
        print(f"  {name}: present_branch keeps {a}, without L_gain keeps {b}")


if __name__ == "__main__":
    main()
