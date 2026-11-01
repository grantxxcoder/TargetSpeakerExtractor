"""Writes the Kaggle notebook that trains one config from the code bundle of scripts/make_kaggle_bundle.py.

    ../tse_venv/bin/python scripts/make_kaggle_notebook.py --config experiments/configs/bsrnn_cue_context.yaml --epochs 6
    ../tse_venv/bin/python scripts/make_kaggle_notebook.py --script scripts/train_turn_down.py --config experiments/configs/turn_down_head.yaml --epochs 6
    ../tse_venv/bin/python scripts/make_kaggle_notebook.py --config experiments/configs/bsrnn_cue_context_wide.yaml --epochs 14 --resume

The notebook copies the bundle's repo/ and ecapa_pretrained/ into /kaggle/working, installs the pinned speechbrain and probe packages, runs the training script on the GPU, and zips the checkpoints and results into one download.
The script must take --config, --epochs, --data-root, --manifest-dir, --outdir and --resume, as scripts/train.py and scripts/train_turn_down.py both do.
It refuses to run without a GPU, so it can never train on a CPU by accident.
The config is set once, in the first code cell, and the notebook warns when it differs from the config the bundle was checked with.
With --resume the notebook finds the one model_sir0.pt attached under /kaggle/input (an earlier session's output or an uploaded dataset), copies only that file, and continues the run. --epochs is then the total, not the number of new epochs.
Rebuilt by Claude Code on 2026-10-28 at the author's request. The version deleted in db74c05 passed train.py flags the 2026-10-23 refactor removed (--split, --results-dir).
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Train on Kaggle

Written by `scripts/make_kaggle_notebook.py`. Before running:

1. Build the code bundle: `../tse_venv/bin/python scripts/make_kaggle_bundle.py --config <config>`, then upload `kaggle_bundle/kaggle_code.zip` as a new version of the code dataset.
2. Attach the code dataset and `tse-audio-s0-v3`. To resume, also attach the earlier session's output (or a dataset holding its `model_sir0.pt`). Settings: Accelerator GPU T4, Internet on (pip installs and the probe's Whisper model).
3. Run all, then download `ALL-*.zip` from the output panel."""

SETUP = """import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise RuntimeError("No GPU. Set Settings, Accelerator, GPU T4 before training.")
print(f"torch {torch.__version__} on {torch.cuda.get_device_name(0)}")

# The bundle is read-only under /kaggle/input, so the repo and the ECAPA snapshot are copied side by side into the working folder.
code = Path(CODE_DIR)
work = Path(WORK)
repo = work / "repo"
for name in ["repo", "ecapa_pretrained"]:
    if (work / name).exists():
        shutil.rmtree(work / name)
    shutil.copytree(code / name, work / name)

manifest = json.loads((code / "bundle_manifest.json").read_text())
# The repo copy has no git history, so the training script reads the bundle's commit from here.
shutil.copy(code / "bundle_manifest.json", repo / "bundle_manifest.json")
print(f"bundle built {manifest['date']} from commit {manifest['git_commit']}, checked with {manifest['script']} and {manifest['config']}")
if manifest["config"] != CONFIG or manifest["script"] != SCRIPT:
    print(f"NOTE: the bundle was checked with {manifest['script']} and {manifest['config']}, but this run uses {SCRIPT} and {CONFIG}")
if not (repo / CONFIG).exists() or not (repo / SCRIPT).exists():
    raise RuntimeError(f"{SCRIPT} or {CONFIG} is not in the bundle.")

# Pinned as in requirements.txt. speechbrain runs the ECAPA encoder; the other three run the per-epoch content probe.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain==1.1.1", "faster-whisper==1.2.1", "jiwer==4.0.0", "whisper-normalizer==0.1.15", "soundfile"], check=True)

models = work / "models"
models.mkdir(exist_ok=True)
if RESUME and RESUME_FROM:
    found = [Path(RESUME_FROM)]
elif RESUME:
    # --resume reads models/model_sir0.pt, so find the one attached copy of it. The audio and code datasets are skipped, since walking 27 GB of audio is slow.
    # Links are followed, since Kaggle may mount inputs through them, and each real folder is visited once so a linked copy is not counted twice.
    skip = {os.path.realpath(DATA_DIR), os.path.realpath(CODE_DIR)}
    visited = set()
    found = []
    for folder, subfolders, files in os.walk("/kaggle/input", followlinks=True):
        real_folder = os.path.realpath(folder)
        if real_folder in visited:
            subfolders[:] = []
            continue
        visited.add(real_folder)
        kept = []
        for name in subfolders:
            if os.path.realpath(os.path.join(folder, name)) not in skip:
                kept.append(name)
        subfolders[:] = kept
        if "model_sir0.pt" in files:
            found.append(Path(folder) / "model_sir0.pt")
    if len(found) != 1:
        # Show what is attached, three levels deep, so the right path can be put in RESUME_FROM.
        for folder, subfolders, files in os.walk("/kaggle/input", followlinks=True):
            depth = folder.count("/") - 2
            if os.path.realpath(folder) in skip or depth > 3:
                subfolders[:] = []
                continue
            print(folder, sorted(files)[:5])
        raise RuntimeError(f"RESUME needs exactly one model_sir0.pt under /kaggle/input, found {len(found)}: {found}. Attach the earlier session's output, or set RESUME_FROM to the file.")
if RESUME:
    if not found[0].exists():
        raise RuntimeError(f"RESUME_FROM {found[0]} does not exist.")
    # Only this file is copied, so the earlier session's per-epoch checkpoints are not zipped again with this session's.
    shutil.copy(found[0], models / "model_sir0.pt")
    print(f"resuming from {found[0]}")"""

TRAIN = """# A resume without its checkpoint would fail inside train.py, so stop here with the reason instead.
if RESUME and not (models / "model_sir0.pt").exists():
    raise RuntimeError("RESUME is on but models/model_sir0.pt is missing, so the setup cell did not finish. Read its output above.")
command = [sys.executable, "-u", SCRIPT, "--config", CONFIG, "--epochs", str(EPOCHS), "--data-root", f"{DATA_DIR}/data", "--manifest-dir", f"{DATA_DIR}/data/manifests", "--outdir", str(models)]
if RESUME:
    command.append("--resume")
print(" ".join(command), flush=True)
process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
return_code = process.wait()
print(f"exit {return_code}")"""

SAVE = """import zipfile

# One download holds the checkpoints, the results folder the script wrote, the config that ran, the run-time log and the bundle's manifest.
archive_path = work / f"ALL-{Path(CONFIG).stem}-e{EPOCHS}.zip"
results = repo / "experiments/results"
with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_STORED) as archive:
    for path in sorted(models.glob("*.pt")):
        archive.write(path, f"models/{path.name}")
    if results.exists():
        for path in sorted(results.rglob("*")):
            if path.is_file():
                archive.write(path, f"results/{path.relative_to(results)}")
    archive.write(repo / CONFIG, Path(CONFIG).name)
    archive.write(repo / "docs/run_times.md", "run_times.md")
    # The bundle's manifest holds the commit the code came from, which the run's own logs cannot read without git.
    archive.write(repo / "bundle_manifest.json", "bundle_manifest.json")
print(f"{archive_path}: {archive_path.stat().st_size / 1e6:.0f} MB")"""


# this splits a cell's text into the list of lines a notebook file stores
def source_lines(text):
    lines = text.split("\n")
    stored = []
    for index, line in enumerate(lines):
        if index < len(lines) - 1:
            stored.append(line + "\n")
        else:
            stored.append(line)
    return stored


# this makes one markdown cell
def markdown_cell(text):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines(text)}


# this makes one code cell
def code_cell(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source_lines(text)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", default="scripts/train.py")
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--out", default="notebooks/kaggle_train.ipynb")
    parser.add_argument("--code-dir", default="/kaggle/input/datasets/grantbooysen/tse-code-v3", help="where Kaggle mounts the code dataset")
    parser.add_argument("--data-dir", default="/kaggle/input/datasets/grantbooysen/tse-audio-s0-v3", help="where Kaggle mounts the audio dataset")
    parser.add_argument("--resume", action="store_true", help="continue a run from the model_sir0.pt attached under /kaggle/input; --epochs is then the total")
    arguments = parser.parse_args()
    if not (REPO_ROOT / arguments.config).exists():
        raise SystemExit(f"No config at {arguments.config}.")
    if not (REPO_ROOT / arguments.script).exists():
        raise SystemExit(f"No training script at {arguments.script}.")

    knobs = f"""# The run, set once. RESUME continues an earlier session from its attached model_sir0.pt, and EPOCHS is then the total.
# RESUME_FROM is that file's path, only needed if the setup cell cannot find it by itself.
SCRIPT = "{arguments.script}"
CONFIG = "{arguments.config}"
EPOCHS = {arguments.epochs}
RESUME = {arguments.resume}
RESUME_FROM = None
DATA_DIR = "{arguments.data_dir}"
CODE_DIR = "{arguments.code_dir}"
WORK = "/kaggle/working\""""
    notebook = {"cells": [markdown_cell(INTRO), code_cell(knobs), code_cell(SETUP), code_cell(TRAIN), code_cell(SAVE)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    output_path = REPO_ROOT / arguments.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(notebook, handle, indent=1)
    if arguments.resume:
        print(f"wrote {arguments.out}: resumes {arguments.script} on {arguments.config} up to {arguments.epochs} epochs in total")
    else:
        print(f"wrote {arguments.out}: runs {arguments.script} on {arguments.config} for {arguments.epochs} epochs")


if __name__ == "__main__":
    main()
