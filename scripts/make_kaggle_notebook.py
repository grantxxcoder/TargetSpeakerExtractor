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
2. **Add data ->** two datasets: the audio for the split you are training
   (from `kaggle_data_<split>.zip`, ~15 GB for the 4,976-trial sir0 split as of
   2026-08-31, upload once per split) and `tse-code`
   (from `kaggle_code.zip`, small, re-upload whenever the code changes). Separate
   so a one-line fix never costs a 2.9 GB upload. Set `SPLIT`, `DATA_DIR` and
   `CODE_DIR` below to match — copy the paths from the right-hand **Input** panel.
3. Set `EPOCHS` in the next cell. Run with `EPOCHS = 2` first to get a measured
   seconds/epoch, then decide — Kaggle kills a GPU session at 12 h and
   `history.csv` is only written when training *finishes*. The checkpoint is
   saved on every val improvement, so a kill costs the curve, not the weights.

## Do not lose the results

Launch with **Save & Run All (Commit)**, not interactively. Kaggle then runs this
headless and commits `/kaggle/working` permanently, so nothing depends on you
downloading in time. An interactive session discards `/kaggle/working` when it
times out -- that is how the 2026-08-25 results were nearly lost.

The second-to-last cell also zips everything small into one file and prints the
history as text, so even a lost filesystem leaves the numbers in this notebook's
saved output.

## If the session dies before it finishes

`history.csv` is only written when training completes, but the training cell
prints one CSV row per epoch as it goes. Select that block (header included),
paste it into `history.csv`, and the curve is intact. The same rows are also
appended live to `/kaggle/working/results/history_live.csv`.

The checkpoint is saved on every val improvement, so the weights survive too.

## To resume in a later session

Save Version, then add this notebook's own output as a dataset input and set
`RESUME_FROM` to the `model_<split>.pt` inside it. `train.py` refuses to resume
across a config change, so do not edit the knobs between sessions.
""".strip()))

cells.append(code(r'''
# ============================== KNOBS ==============================
SPLIT       = "sir0"  # "mid" = 90% target-louder (control) | "sir0" = symmetric
EPOCHS      = 25      # 12 h GPU cap. sir0_train is 4,976 trials as of
                      # 2026-08-31 (was 1,989), so ~1,360 s/epoch PROJECTED from
                      # the measured 523-568 at 1,989 -- about 29 epochs fit a
                      # session. 25 leaves headroom and patience 10 will stop it
                      # earlier if it turns. The previous best epoch (14 at 1,989
                      # trials) was ~18,600 optimiser steps, which at 4,976
                      # trials lands near epoch 6, so 25 is ~4x past it.
                      # Run EPOCHS = 2 first for a measured s/epoch.
BATCH_SIZE  = 12      # CEILING, not a promise. 12 OOMs on a 14.6 GiB T4; the
                      # probe below steps down until one fwd+bwd+step fits and
                      # writes the winner into the config that trains.
BATCH_FLOOR = 2       # give up below this
NUM_WORKERS = 4       # 0 starves the GPU: 3 windowed wav reads per crop
RESUME_FROM = None    # e.g. "/kaggle/input/prev-run/models/model_sir0.pt"

# The two Kaggle dataset mount points. Change only if you rename the datasets.
DATA_DIR     = "/kaggle/input/tse-audio-s0-v3"   # the dataset holding the audio
DATA_DIR_NEW = None    # SECOND audio dataset, or None. Set this only when the
                       # trials are split across two datasets because the first
                       # half was already uploaded (make_kaggle_bundle.py
                       # --new-only). The merge cell below symlinks both into one
                       # tree; the FULL manifest is read from DATA_DIR_NEW, which
                       # is the one carrying every row. 2026-09-03.
CODE_DIR     = "/kaggle/input/tse-code-v3"
# The paths above are GUESSES until you look: Kaggle has mounted datasets at both
# /kaggle/input/<slug> and /kaggle/input/datasets/<user>/<slug>. Run `!ls
# /kaggle/input` and copy what it actually prints -- a wrong path fails in the
# next cell with the exact missing file, which is the cheapest place to find out.
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
DATA_NEW = Path(DATA_DIR_NEW) if DATA_DIR_NEW else None
CODE = Path(CODE_DIR)
print(f"data: {DATA}" + (f"\ndata (new half): {DATA_NEW}" if DATA_NEW else "")
      + f"\ncode: {CODE}")

# Fail here, with the exact missing path, rather than three cells later with
# something obscure. If a path is wrong, the Kaggle right-hand Input panel shows
# the real one -- copy it into the knobs cell above.
for base, rel in [(CODE, "scripts/train.py"),
                  (CODE, "experiments/configs/bsrnn_baseline.yaml"),
                  (CODE, "src/models/bsrnn.py")]:
    if not (base / rel).exists():
        raise SystemExit(f"missing: {base / rel}\n"
                         f"  fix CODE_DIR in the knobs cell above.")
print("  code contents verified (data is checked in the merge cell below)")
'''))

cells.append(md(r"""
## Resolve the data root

One audio dataset, or two symlinked into one tree. Two happens when the first
half of the trials was already on Kaggle and only the new half was uploaded.
"""))

cells.append(code(r'''
# --- resolve the data root, merging two datasets if needed ----------------
#
# WHY THIS EXISTS. train.py takes ONE --data-root, but the trials can be spread
# across two Kaggle datasets: the first N were uploaded earlier, and re-sending
# them is pure wall-clock -- 30.3 GB at the measured 0.40 MB/s is ~21 h against
# ~10.5 h for the new half alone. So when DATA_DIR_NEW is set, build a single
# tree of SYMLINKS pointing into both mounts. 2026-09-03.
#
# Linked at the TRIAL-DIRECTORY level, not per file: ~10k links rather than
# ~130k, seconds instead of minutes, and /kaggle/working holds only links so it
# costs essentially no quota.
#
# The MANIFEST is read from DATA_DIR_NEW, and that is not arbitrary. The older
# dataset still carries the SHORT manifest it was built with (4,976 rows). Read
# that one and the run trains on half the data while reporting success -- no
# error, just a quietly wrong experiment. The new dataset carries every row.
MERGED = Path(WORK) / "merged"

if DATA_NEW is None:
    DATA_ROOT    = DATA / "data"
    MANIFEST_DIR = DATA / "data/manifests"
    print(f"single dataset -> {DATA_ROOT}")
else:
    from collections import Counter
    made = Counter()
    for sub in (f"{SPLIT}_train", f"{SPLIT}_val"):
        dst = MERGED / "data/rendered" / sub
        dst.mkdir(parents=True, exist_ok=True)
        # DATA first, DATA_NEW second; first writer wins on a duplicate trial_id.
        # Duplicates are EXPECTED: val is staged into both halves on purpose, so
        # that the new dataset can verify itself without the old one.
        for src_root in (DATA, DATA_NEW):
            src = src_root / "data/rendered" / sub
            if not src.is_dir():
                continue
            for tid in os.listdir(src):
                link = dst / tid
                if not link.exists():
                    os.symlink(src / tid, link)
                    made[sub] += 1
    # Link the manifests in too, so MERGED is a COMPLETE data root rather than
    # a rendered-only tree. The batch-size probe derives its manifest directory
    # as <data-root>/manifests, so a partial merge would break it in a way that
    # looks like an OOM rather than a path bug.
    mdst = MERGED / "data/manifests"
    mdst.mkdir(parents=True, exist_ok=True)
    for f in sorted((DATA_NEW / "data/manifests").iterdir()):
        link = mdst / f.name
        if not link.exists():
            os.symlink(f, link)
    DATA_ROOT    = MERGED / "data"
    MANIFEST_DIR = mdst
    print(f"merged two datasets -> {DATA_ROOT}")
    for sub, n in sorted(made.items()):
        print(f"  {sub}: {n} trial symlinks")

# Check what the run will ACTUALLY read, after resolution -- not the knob paths.
# Counting manifest rows against trials on disk is what catches a half-merged
# tree, which is the one new failure this cell can introduce.
import csv as _csv
for stem in (f"{SPLIT}_train", f"{SPLIT}_val"):
    man = MANIFEST_DIR / f"{stem}.csv"
    if not man.exists():
        raise SystemExit(f"missing manifest: {man}\n  fix the knobs cell above.")
    with open(man) as fh:
        ids = [r["trial_id"] for r in _csv.DictReader(fh)]
    missing = [t for t in ids if not (DATA_ROOT / "rendered" / stem / t).exists()]
    print(f"  {stem}: {len(ids)} rows, {len(ids) - len(missing)} present on disk")
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(ids)} trials in {stem} are NOT on disk, "
            f"first missing: {missing[0]}\n"
            f"  Split across two datasets? Set DATA_DIR_NEW.\n"
            f"  Already set? The older dataset holds the first half -- if it was "
            f"deleted, this run cannot proceed and the full set must be re-uploaded.")
print("  data verified")
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
    dst = Path(OUT) / f"model_{SPLIT}.pt"
    shutil.copy2(RESUME_FROM, dst)
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    print(f"resume: copied checkpoint from epoch {ck['epoch']}, best_val {ck['best_val']:.4f}")
    if ck.get("config") != cfg:
        raise SystemExit("checkpoint config != this config; train.py will refuse. "
                         "Restore the knobs used for that checkpoint.")
'''))

cells.append(code(r'''
# --- find the largest batch size that actually fits ----------------------
# Measured, not guessed: real forward + backward + optimizer step per candidate.
# AdamW allocates its state on the first step() that APPLIES, so a probe that
# stops at backward() -- or whose step is skipped -- understates peak memory.
#
# PROBES IN THE PRECISION THAT TRAINS. Until 2026-09-04 this ran fp32 while
# train.py trained under AMP, so it found the fp32 ceiling (batch 3, measured)
# and wrote it into an fp16 run that fits 6 -- capping every Kaggle run at half
# the batch it could hold. Same class as the fp32-warmup bug fixed in
# profile_step.py on 2026-08-28; that fix never reached here.
# decisions-pending.md E8.
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
from scripts.train import (get_data_loaders, build_model, build_loss_fn, unpack,
                           amp_ctx)
B, data, split = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
cfg = yaml.safe_load(open("experiments/configs/bsrnn_baseline.yaml"))
cfg["data"]["batch_size"] = B
torch.manual_seed(int(cfg["seed"]))
dev = torch.device("cuda")
# amp_ctx is IMPORTED, not reimplemented -- its docstring says one definition so
# train, val and the diagnostic cannot drift into different precisions, and this
# probe is subject to exactly that requirement.
use_amp = bool(cfg["training"].get("amp", False))
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
tr, _ = get_data_loaders(split, data / "manifests", data, cfg)
m = build_model(cfg).to(dev); m.train()
L = build_loss_fn(cfg)
opt = torch.optim.AdamW(m.parameters(), lr=float(cfg["training"]["lr"]),
                        weight_decay=float(cfg["training"]["weight_decay"]))
grad_clip = float(cfg["training"]["grad_clip"])

# LOOP UNTIL A STEP ACTUALLY APPLIES. GradScaler starts at a high scale and
# skips the step whenever the gradients hold inf/NaN, which is normal on the
# first steps while it calibrates. A skipped step leaves opt.state empty, so
# AdamW's exp_avg/exp_avg_sq are never allocated and the peak comes back ~2
# param-copies light. Detect it directly rather than assuming a step count.
applied = False
for i, b in enumerate(tr):
    x, s, e, a = unpack(b, dev)
    opt.zero_grad()
    # Mirrors train.py's step exactly: autocast the FORWARD ONLY, .float()
    # before the loss (LossBSRNN's 1e-12 epsilons underflow in fp16), then
    # scale / unscale / clip / step / update.
    with amp_ctx(use_amp):
        out = m(x, e)
    loss, _ = L(s, out.float(), x, a)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(m.parameters(), grad_clip)
    scaler.step(opt)
    scaler.update()
    if opt.state:
        applied = True
        break
    if i >= 4:
        break
torch.cuda.synchronize()
if not applied:
    raise SystemExit(f"PROBE-INCONCLUSIVE {B}: GradScaler skipped every step, "
                     f"so AdamW state was never allocated and the peak is an "
                     f"underestimate. Do not trust this batch size.")
print(f"OK {B} {torch.cuda.max_memory_allocated() / 2**30:.2f} "
      f"{torch.cuda.max_memory_reserved() / 2**30:.2f} "
      f"{'amp' if use_amp else 'fp32'}")
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
        r = subprocess.run([sys.executable, "_probe_batch.py", str(B), str(DATA_ROOT), SPLIT],
                           cwd=REPO, capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.startswith("OK"):
            # 5 fields since 2026-09-04: the last one is the precision probed,
            # and it MUST read `amp` whenever training.amp is on. A probe in the
            # wrong precision measures a run that never happens -- that is how
            # every Kaggle run got capped at the fp32 ceiling. E8.
            _, b_ok, peak, res, prec = r.stdout.split()
            print(f"  batch {B:2d}: FITS   peak allocated {peak} GiB, "
                  f"reserved {res} GiB, probed in {prec}")
            if prec != ("amp" if cfg["training"].get("amp") else "fp32"):
                raise SystemExit(f"probe ran in {prec} but training.amp="
                                 f"{cfg['training'].get('amp')}; the ceiling it "
                                 f"found is for the wrong precision")
            chosen = B
            break
        if "PROBE-INCONCLUSIVE" in r.stderr:
            print(r.stderr[-500:])
            raise SystemExit(f"probe inconclusive at batch {B}: see above")
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
# history.csv. The human-readable per-epoch term breakdown goes to STDERR
# (it replaced tqdm's progress bars on 2026-08-31, which emitted one line per
# batch and buried everything else). They are kept apart on purpose: the stdout
# block below is a valid history.csv, so if this session dies you can select it,
# paste it into a .csv, and lose nothing. history.csv itself is only written
# when training *finishes*.
#
# Read the `gap(val-train)` column in the stderr breakdown, not `total`: both
# totals fell the whole way through the 2026-08-29 run while held-out
# separation collapsed below pass-through.
#
# -u as well as train.py's own flush=True: stdout is a pipe here, and a killed
# session must not lose rows to a buffer.
cmd = [sys.executable, "-u", "scripts/train.py",
       "--split", SPLIT,
       "--epochs", str(EPOCHS),
       "--config", "experiments/configs/bsrnn_baseline.yaml",
       "--data-root", str(DATA_ROOT),
       "--manifest-dir", str(MANIFEST_DIR),
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
# --- SAVE EVERYTHING, before anything can be lost ------------------------
# The reliable way to keep Kaggle output is "Save & Run All (Commit)": Kaggle
# then runs this headless and commits /kaggle/working permanently. An
# INTERACTIVE session throws /kaggle/working away when it times out, which is
# how the 10-epoch results were nearly lost on 2026-08-25.
#
# Belt and braces regardless of how it was launched:
#   1. train.py now writes model_<split>_last.pt EVERY epoch, so the newest
#      weights survive a kill even if the best ones are stale.
#   2. ALL-<split>-e<n>.zip holds the results AND the checkpoints, so
#      recovery is a single download.
#   3. the history is printed as text below, so even a lost filesystem leaves
#      the numbers in this notebook's saved output.
import shutil, zipfile, glob
from pathlib import Path

STAMP = f"{SPLIT}-e{EPOCHS}"
bundle = Path(WORK) / f"results-{STAMP}.zip"
with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(Path(RES).rglob("*")):
        if f.is_file():
            z.write(f, f"results/{f.name}")
    cfgp = Path(REPO) / "experiments/configs/bsrnn_baseline.yaml"
    if cfgp.exists():
        z.write(cfgp, "bsrnn_baseline.yaml")      # the config that actually ran
    rt = Path(REPO) / "docs/run_times.md"
    if rt.exists():
        z.write(rt, "run_times.md")
print(f"small artefacts -> {bundle}  ({bundle.stat().st_size/1e6:.1f} MB)")

# ONE archive with the checkpoints in it too, so recovery is a single download.
# ZIP_STORED for the .pt files: they are already-compressed tensors, so
# deflating them costs minutes and saves almost nothing.
full = Path(WORK) / f"ALL-{STAMP}.zip"
with zipfile.ZipFile(full, "w", zipfile.ZIP_STORED) as z:
    z.write(bundle, bundle.name)
    for pt in sorted(glob.glob(f"{OUT}/*.pt")):
        z.write(pt, f"models/{Path(pt).name}")
        print(f"  + {Path(pt).name}  ({Path(pt).stat().st_size/1e6:.0f} MB)")
print(f"EVERYTHING       -> {full}  ({full.stat().st_size/1e6:.0f} MB)")

# Click-to-download, so nothing has to be hunted for in the file browser.
try:
    from IPython.display import FileLink, display
    print("download this one and you have everything:")
    display(FileLink(str(full.relative_to("/kaggle/working"))))
    print("or separately:")
    display(FileLink(str(bundle.relative_to("/kaggle/working"))))
    for pt in sorted(glob.glob(f"{OUT}/*.pt")):
        display(FileLink(str(Path(pt).relative_to("/kaggle/working"))))
except Exception as exc:
    print(f"(no download links in this environment: {exc})")

# The history as TEXT. If the filesystem is lost this block is still in the
# notebook's saved output, and it is a valid history.csv on its own.
for name in ("history.csv", "history_live.csv"):
    f = Path(RES) / name
    if f.exists():
        print(f"\n----- {name} -----")
        print(f.read_text().strip())
        break
'''))

cells.append(code(r'''
# --- what came out -------------------------------------------------------
import pandas as pd

ck = Path(OUT) / f"model_{SPLIT}.pt"
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
print(f"\nDownload {full} -- it holds the results and every checkpoint. "
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
