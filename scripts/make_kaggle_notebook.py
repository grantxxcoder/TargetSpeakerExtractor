#!/usr/bin/env python3
"""Emit notebooks/kaggle_train_mid.ipynb.

Generated rather than hand-edited because the cell sources are long and the
nbformat `source` convention is easy to get wrong: every element must carry its
own trailing newline, and Jupyter concatenates them with NO separator. Get that
wrong and the whole cell collapses onto one line -- every statement a syntax
error -- while a validator that joins with "\n" still reports it as fine. That
bug shipped once; _lines() and the check at the bottom exist to stop it again.

    python scripts/make_kaggle_notebook.py
"""
import json, pathlib

def _lines(src):
    """nbformat convention: every element of `source` carries its own trailing
    newline, except optionally the last. Plain .split("\n") drops them, and
    Jupyter concatenates source with no separator -- so the whole cell collapses
    onto one line and every statement becomes a syntax error. Validate with
    "".join(source), never "\n".join(source), or the bug is invisible."""
    lines = src.strip("\n").split("\n")
    return [l + "\n" for l in lines[:-1]] + [lines[-1]]

def md(src):   return {"cell_type": "markdown", "metadata": {}, "source": _lines(src)}
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {},
                       "outputs": [], "source": _lines(src)}

cells = []

cells.append(md(r"""
# Train BSRNN + TF-Map — `mid` split, Kaggle GPU

Everything needed to train. Nothing else: no eval harness, no live-model metric,
no data generation, no analysis.

## Before you run

1. **Settings -> Accelerator -> GPU** (T4 or P100).
2. **Add data ->** both datasets: `tse-mid-audio` (from `kaggle_data.zip`,
   2.7 GB, upload once) and `tse-code` (from `kaggle_code.zip`, small, re-upload
   whenever the code changes). Separate so a one-line fix never costs a 2.7 GB
   upload. Their paths are hardcoded in `DATA_DIR` / `CODE_DIR` below — if you
   rename a dataset, copy the new path from the right-hand **Input** panel.
3. Set `EPOCHS` in the next cell. Run with `EPOCHS = 2` first to get a measured
   seconds/epoch, then decide — Kaggle kills a GPU session at 12 h and
   `history.csv` is only written when training *finishes*. The checkpoint is
   saved on every val improvement, so a kill costs the curve, not the weights.

## If the session dies before it finishes

`history.csv` is only written when training completes, but the training cell
prints one CSV row per epoch as it goes. Select that block (header included),
paste it into `history.csv`, and the curve is intact. The same rows are also
appended live to `/kaggle/working/results/history_live.csv`.

The checkpoint is saved on every val improvement, so the weights survive too.

## To resume in a later session

Save Version, then add this notebook's own output as a dataset input and set
`RESUME_FROM` to the `model_mid.pt` inside it. `train.py` refuses to resume
across a config change, so do not edit the knobs between sessions.
""".strip()))

cells.append(code(r'''
# ============================== KNOBS ==============================
EPOCHS      = 2       # start at 2 to measure, then raise. 12 h GPU cap.
BATCH_SIZE  = 12      # CEILING, not a promise. 12 OOMs on a 14.6 GiB T4; the
                      # probe below steps down until one fwd+bwd+step fits and
                      # writes the winner into the config that trains.
BATCH_FLOOR = 2       # give up below this
NUM_WORKERS = 4       # 0 starves the GPU: 3 windowed wav reads per crop
RESUME_FROM = None    # e.g. "/kaggle/input/prev-run/model_mid.pt"

# The two Kaggle dataset mount points. Change only if you rename the datasets.
DATA_DIR = "/kaggle/input/datasets/grantbooysen/tse-mid-audio"
CODE_DIR = "/kaggle/input/datasets/grantbooysen/tse-code"
# ===================================================================

WORK = "/kaggle/working"
REPO = f"{WORK}/repo"          # writable copy of the code; /kaggle/input is not
OUT  = f"{WORK}/models"
RES  = f"{WORK}/results"
'''))

cells.append(code(r'''
# --- environment + input paths -------------------------------------------
import os, sys, shutil, subprocess, json
from pathlib import Path

# Set BEFORE torch initialises CUDA or it has no effect. The OOM traceback
# reported 4.40 GiB "reserved but unallocated" -- that is fragmentation, not
# real demand, and this is the allocator setting the error message itself
# recommends. Inherited by the probe and training subprocesses.
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device: {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
else:
    print("  NO GPU -- set Settings -> Accelerator -> GPU. Refusing to train on CPU here.")

# soundfile is the only dep Kaggle sometimes lacks; everything else is preinstalled
try:
    import soundfile
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "soundfile"], check=True)
    import soundfile
print(f"soundfile {soundfile.__version__}")

DATA = Path(DATA_DIR)
CODE = Path(CODE_DIR)
print(f"data: {DATA}\ncode: {CODE}")

# Fail here, with the exact missing path, rather than three cells later with
# something obscure. If a path is wrong, the Kaggle right-hand Input panel shows
# the real one -- copy it into the knobs cell above.
for base, rel in [(DATA, "data/manifests/mid_train.csv"),
                  (DATA, "data/manifests/mid_val.csv"),
                  (CODE, "scripts/train.py"),
                  (CODE, "experiments/configs/bsrnn_baseline.yaml"),
                  (CODE, "src/models/bsrnn.py")]:
    if not (base / rel).exists():
        raise SystemExit(f"missing: {base / rel}\n"
                         f"  fix DATA_DIR / CODE_DIR in the knobs cell above.")
print("  contents verified")
'''))

cells.append(code(r'''
# --- stage the code somewhere writable, then write the derived config -----
# /kaggle/input is read-only and src.run_log writes repo_root/docs/run_times.md,
# so the code cannot run in place.
import yaml

if Path(REPO).exists():
    shutil.rmtree(REPO)
# NO ignore= here. shutil.ignore_patterns("data") matches ANY directory named
# `data` at any depth, so it silently dropped src/data/ along with the
# top-level data/ it was meant for -- and the code bundle stopped containing
# audio at all once the bundles were split. Symptom was
# ModuleNotFoundError: No module named 'src.data', three cells later.
shutil.copytree(CODE, REPO)
Path(OUT).mkdir(parents=True, exist_ok=True)
Path(RES).mkdir(parents=True, exist_ok=True)

# Import from the staged copy exactly as train.py will, before any GPU time is
# spent. A staging bug is invisible until the training subprocess dies; this
# turns it into one obvious line here.
chk = subprocess.run(
    [sys.executable, "-c", "import sys; sys.path.insert(0, '.'); "
     "import src.data.dataset_loader, src.models.bsrnn, src.models.losses, "
     "src.models.stft, src.models.bands, src.models.modules, "
     "src.models.conditioning, src.run_log"],
    cwd=REPO, capture_output=True, text=True)
if chk.returncode:
    raise SystemExit(f"staged code at {REPO} does not import:\n{chk.stderr}")
print("  staged code imports OK")

cfg_path = Path(REPO) / "experiments/configs/bsrnn_baseline.yaml"
cfg = yaml.safe_load(cfg_path.read_text())
cfg["data"]["batch_size"]  = BATCH_SIZE
cfg["data"]["num_workers"] = NUM_WORKERS
# Written back so meta.yaml records the config that actually trained, and so a
# resume compares equal: train.py raises if the checkpoint's config differs.
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))

print(f"code staged -> {REPO}")
print(f"  batch_size={cfg['data']['batch_size']}  num_workers={cfg['data']['num_workers']}  "
      f"seed={cfg['seed']}  chunk_s={cfg['data']['chunk_s']}")
print(f"  loss: w={cfg['loss']['w']} w_m={cfg['loss']['w_m']} "
      f"tau_pres={cfg['loss']['tau_pres']} tau_abs={cfg['loss']['tau_abs']}")

if RESUME_FROM:
    dst = Path(OUT) / "model_mid.pt"
    shutil.copy2(RESUME_FROM, dst)
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    print(f"resume: copied checkpoint from epoch {ck['epoch']}, best_val {ck['best_val']:.4f}")
    if ck.get("config") != cfg:
        raise SystemExit("checkpoint config != this config; train.py will refuse. "
                         "Restore the knobs used for that checkpoint.")
'''))

cells.append(code(r'''
# --- find the largest batch size that actually fits ----------------------
# Measured, not guessed: one real forward + backward + optimizer step per
# candidate. AdamW allocates its state on the first step(), so a probe that
# stops at backward() understates peak memory.
#
# Each candidate runs in its OWN subprocess. A caught OutOfMemoryError leaves
# the allocator in a poor state and the failed graph can stay reachable, so
# retrying in-process measures the wrong thing; process exit is the only
# reliable free.
PROBE = Path(REPO) / "_probe_batch.py"
PROBE.write_text("""
import sys, yaml, torch
from pathlib import Path
sys.path.insert(0, ".")
from scripts.train import get_data_loaders, build_model, build_loss_fn, unpack
B, data = int(sys.argv[1]), Path(sys.argv[2])
cfg = yaml.safe_load(open("experiments/configs/bsrnn_baseline.yaml"))
cfg["data"]["batch_size"] = B
torch.manual_seed(int(cfg["seed"]))
dev = torch.device("cuda")
tr, _ = get_data_loaders("mid", data / "manifests", data, cfg)
m = build_model(cfg).to(dev); m.train()
L = build_loss_fn(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
b = next(iter(tr)); x, s, e, a = unpack(b, dev)
loss, _ = L(s, m(x, e), x, a)
loss.backward(); opt.step()
torch.cuda.synchronize()
print(f"OK {B} {torch.cuda.max_memory_allocated() / 2**30:.2f} "
      f"{torch.cuda.max_memory_reserved() / 2**30:.2f}")
""")

import yaml
cfg_path = Path(REPO) / "experiments/configs/bsrnn_baseline.yaml"
cfg = yaml.safe_load(cfg_path.read_text())

if RESUME_FROM:
    # The checkpoint's config is authoritative -- train.py refuses to resume
    # across a config change, so probing could only pick a losing fight.
    ck = torch.load(RESUME_FROM, map_location="cpu", weights_only=False)
    chosen = ck["config"]["data"]["batch_size"]
    print(f"resuming: batch_size {chosen} taken from the checkpoint, probe skipped")
elif not torch.cuda.is_available():
    chosen = cfg["data"]["batch_size"]
    print(f"no CUDA: leaving batch_size at {chosen}, probe skipped")
else:
    cands = [b for b in [BATCH_SIZE, 10, 8, 6, 5, 4, 3, 2]
             if BATCH_FLOOR <= b <= BATCH_SIZE]
    cands = sorted(set(cands), reverse=True)
    chosen = None
    for B in cands:
        r = subprocess.run([sys.executable, "_probe_batch.py", str(B), str(DATA / "data")],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.startswith("OK"):
            _, b_ok, peak, res = r.stdout.split()
            print(f"  batch {B:2d}: FITS   peak allocated {peak} GiB, reserved {res} GiB")
            chosen = B
            break
        if "OutOfMemoryError" in r.stderr or "out of memory" in r.stderr.lower():
            print(f"  batch {B:2d}: OOM")
            continue
        print(r.stderr[-1500:])
        raise SystemExit(f"probe failed at batch {B} for a reason other than OOM")
    if chosen is None:
        raise SystemExit(f"nothing fits down to BATCH_FLOOR={BATCH_FLOOR}. "
                         "Reduce data.chunk_s or the model width.")

cfg["data"]["batch_size"]  = chosen
cfg["data"]["num_workers"] = NUM_WORKERS
# Rewritten so meta.yaml records the batch size that ACTUALLY trained, not the
# ceiling that was asked for.
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"\ntraining will use batch_size={chosen}, num_workers={NUM_WORKERS}")
if chosen != BATCH_SIZE:
    print(f"  NOTE: below the requested {BATCH_SIZE}. loss.w was calibrated against "
          f"an absent rate assuming batch 12; at batch {chosen} a larger share of "
          f"batches contain no absent crop, so L_abs is a noisier estimate.")
'''))

cells.append(code(r'''
# --- train ---------------------------------------------------------------
# Audio is read straight from the read-only data input; only outputs land
# in /kaggle/working.
#
# train.py prints one CSV row per epoch on STDOUT, with the same columns as
# history.csv. tqdm's progress bars go to STDERR. They are kept apart on
# purpose: the stdout block below is a valid history.csv, so if this session
# dies you can select it, paste it into a .csv, and lose nothing. history.csv
# itself is only written when training *finishes*.
#
# -u as well as train.py's own flush=True: stdout is a pipe here, and a killed
# session must not lose rows to a buffer.
cmd = [sys.executable, "-u", "scripts/train.py",
       "--split", "mid",
       "--epochs", str(EPOCHS),
       "--config", "experiments/configs/bsrnn_baseline.yaml",
       "--data-root", str(DATA / "data"),
       "--manifest-dir", str(DATA / "data/manifests"),
       "--outdir", OUT,
       "--results-dir", RES]
if RESUME_FROM:
    cmd.append("--resume")
print(" ".join(cmd), flush=True)

# Tee stdout: print each line live AND append it to disk, so the curve survives
# both ways. stderr is inherited, so the bars render normally and never mix into
# the copyable block.
live = Path(RES) / "history_live.csv"
import re
CSV_ROW = re.compile(r"^(epoch|\d+),")
with open(live, "a") as fh:
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                            stderr=None, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="")
        if CSV_ROW.match(line):
            fh.write(line)
            fh.flush()
    rc = proc.wait()

print(f"\nexit {rc}   (rows also appended to {live})")
if rc != 0:
    raise SystemExit(f"training failed with exit {rc}")
'''))

cells.append(code(r'''
# --- what came out -------------------------------------------------------
import pandas as pd

ck = Path(OUT) / "model_mid.pt"
print(f"checkpoint: {ck}  ({ck.stat().st_size/1e6:.0f} MB)" if ck.exists() else "NO CHECKPOINT")

hist = Path(RES) / "history.csv"
live = Path(RES) / "history_live.csv"
# history.csv only exists if training finished; history_live.csv exists either way
src = hist if hist.exists() else live
if src.exists():
    h = pd.read_csv(src)
    h = h[h.epoch != "epoch"].astype({c: float for c in h.columns})  # drop repeated headers from a resume
    print(f"read {src.name} ({len(h)} epochs)")
    cols = ["epoch", "train_total", "val_total", "val_L_pres", "val_L_MR", "val_L_abs", "lr"]
    print(h[cols].to_string(index=False, float_format=lambda v: f"{v:9.4f}"))
    best = h.loc[h.val_total.idxmin()]
    print(f"\nbest val_total {best.val_total:.4f} at epoch {int(best.epoch)}")
    if not hist.exists():
        print("NOTE: read from history_live.csv -- training did not finish, "
              "so meta.yaml and loss_plot.png were never written.")
    # The thing this run exists to answer. L_MR rising while val_total falls is
    # the enrolment-blind mute of 2026-08-25 reappearing at 940 speakers.
    d = h.val_L_MR.iloc[-1] - h.val_L_MR.iloc[0]
    print(f"val_L_MR {h.val_L_MR.iloc[0]:.4f} -> {h.val_L_MR.iloc[-1]:.4f}  ({d:+.4f})  "
          + ("STILL RISING: mute not escaped" if d > 0 else "falling: mute escaped"))
else:
    print("no history at all -- training died before finishing epoch 1. "
          "The CSV rows printed by the training cell are the only record; "
          "copy them into a .csv if you need them.")

rt = Path(REPO) / "docs/run_times.md"
if rt.exists():
    rows = [l for l in rt.read_text().splitlines() if l.startswith("| 20")]
    print("\nmeasured wall time (copy this row into the repo's docs/run_times.md):")
    for r in rows[:1]:
        print(" ", r)
print("\nDownload /kaggle/working/models/model_mid.pt and /kaggle/working/results/ "
      "before the session ends, or Save Version to keep them.")
'''))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = pathlib.Path("notebooks/kaggle_train_mid.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1) + "\n")
# Validate the way Jupyter reads it: "".join, never "\n".join.
import ast
for i, c in enumerate(nb["cells"]):
    if c["cell_type"] == "code":
        ast.parse("".join(c["source"]))          # raises if the cell is broken
    assert all(l.endswith("\n") for l in c["source"][:-1]), \
        f"cell {i}: source elements must carry trailing newlines"
print(f"wrote {out} ({len(cells)} cells); all code cells parse via \"\".join")
