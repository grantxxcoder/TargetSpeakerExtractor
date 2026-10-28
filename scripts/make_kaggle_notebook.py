"""Writes the Kaggle notebook that trains one config from the code bundle of scripts/make_kaggle_bundle.py.

    ../tse_venv/bin/python scripts/make_kaggle_notebook.py --config experiments/configs/bsrnn_cue_context.yaml --epochs 6

The notebook copies the bundle's repo/ and ecapa_pretrained/ into /kaggle/working, installs the pinned speechbrain and probe packages, runs scripts/train.py on the GPU, and zips the checkpoints and results into one download.
It refuses to run without a GPU, so it can never train on a CPU by accident.
The config is set once, in the first code cell, and the notebook warns when it differs from the config the bundle was checked with.
Rebuilt by Claude Code on 2026-10-28 at the author's request. The version deleted in db74c05 passed train.py flags the 2026-10-23 refactor removed (--split, --results-dir).
"""

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

INTRO = """# Train on Kaggle

Written by `scripts/make_kaggle_notebook.py`. Before running:

1. Build the code bundle: `../tse_venv/bin/python scripts/make_kaggle_bundle.py --config <config>`, then upload `kaggle_bundle/kaggle_code.zip` as a new version of the code dataset.
2. Attach the code dataset and `tse-audio-s0-v3`. Settings: Accelerator GPU T4, Internet on (pip installs and the probe's Whisper model).
3. Run all, then download `ALL-*.zip` from the output panel."""

SETUP = """import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

if not torch.cuda.is_available():
    raise SystemExit("No GPU. Set Settings, Accelerator, GPU T4 before training.")
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
print(f"bundle built {manifest['date']} from commit {manifest['git_commit']}, checked with {manifest['config']}")
if manifest["config"] != CONFIG:
    print(f"NOTE: the bundle was checked with {manifest['config']}, but this run trains {CONFIG}")
if not (repo / CONFIG).exists():
    raise SystemExit(f"{CONFIG} is not in the bundle.")

# Pinned as in requirements.txt. speechbrain runs the ECAPA encoder; the other three run the per-epoch content probe.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "speechbrain==1.1.1", "faster-whisper==1.2.1", "jiwer==4.0.0", "whisper-normalizer==0.1.15", "soundfile"], check=True)

models = work / "models"
models.mkdir(exist_ok=True)
if RESUME_DIR:
    shutil.copy(Path(RESUME_DIR) / "models/model_sir0.pt", models / "model_sir0.pt")
    print(f"resuming from {RESUME_DIR}")"""

TRAIN = """command = [sys.executable, "-u", "scripts/train.py", "--config", CONFIG, "--epochs", str(EPOCHS), "--data-root", f"{DATA_DIR}/data", "--manifest-dir", f"{DATA_DIR}/data/manifests", "--outdir", str(models)]
if RESUME_DIR:
    command.append("--resume")
print(" ".join(command), flush=True)
process = subprocess.Popen(command, cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
return_code = process.wait()
print(f"exit {return_code}")"""

SAVE = """import zipfile

# One download holds the checkpoints, the results folder train.py wrote, the config that ran and the run-time log.
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
    parser.add_argument("--config", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--out", default="notebooks/kaggle_train.ipynb")
    parser.add_argument("--code-dir", default="/kaggle/input/tse-code-v3", help="where Kaggle mounts the code dataset")
    parser.add_argument("--data-dir", default="/kaggle/input/tse-audio-s0-v3", help="where Kaggle mounts the audio dataset")
    arguments = parser.parse_args()
    if not (REPO_ROOT / arguments.config).exists():
        raise SystemExit(f"No config at {arguments.config}.")

    knobs = f"""# The run, set once. RESUME_DIR is an attached earlier output holding models/model_sir0.pt, to continue that run with --resume.
CONFIG = "{arguments.config}"
EPOCHS = {arguments.epochs}
RESUME_DIR = None
DATA_DIR = "{arguments.data_dir}"
CODE_DIR = "{arguments.code_dir}"
WORK = "/kaggle/working\""""
    notebook = {"cells": [markdown_cell(INTRO), code_cell(knobs), code_cell(SETUP), code_cell(TRAIN), code_cell(SAVE)], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    output_path = REPO_ROOT / arguments.out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as handle:
        json.dump(notebook, handle, indent=1)
    print(f"wrote {arguments.out}: trains {arguments.config} for {arguments.epochs} epochs")


if __name__ == "__main__":
    main()
