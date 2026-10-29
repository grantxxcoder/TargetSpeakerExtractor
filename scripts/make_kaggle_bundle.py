"""Stages what a Kaggle GPU session needs to run scripts/train.py, checks the staged copy, and zips it as one code dataset.

    ../tse_venv/bin/python scripts/make_kaggle_bundle.py --config experiments/configs/bsrnn_cue_context.yaml
    ../tse_venv/bin/python scripts/make_kaggle_bundle.py --config <config> --init-weights models/model_sir0_cuecontext-wer-e21.pt
    ../tse_venv/bin/python scripts/make_kaggle_bundle.py --script scripts/train_turn_down.py --config experiments/configs/turn_down_head.yaml --init-weights models/model_sir0_cuecontext-wer-e21.pt

The audio is not staged. It is already on Kaggle as the dataset tse-audio-s0-v3 (data/rendered and data/manifests of sir0_train and sir0_val), and only the code changes between runs.
The code is the import closure of the training script (scripts/train.py unless --script says otherwise), found by reading the import lines, so a module added later cannot be left out. Every config and docs/run_times.md (src/run_log.py appends to it) go with it.
The ECAPA snapshot sits beside the repo in the bundle (ecapa_pretrained/ next to repo/), because src/models/context_encoder.py looks for it one level above the repo.
Before zipping, the staged copy is checked in its own process: train.py is imported from the staged folder and two validation batches go through the model and the loss with no gradients. Any other script is run with its own --check, which must do the same. Nothing is trained.
Rebuilt by Claude Code on 2026-10-28 at the author's request. The version deleted in db74c05 called train.py functions that the 2026-10-23 refactor removed, so it could not be restored as it was.
"""

import argparse
import ast
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIRECTORY = REPO_ROOT / "kaggle_bundle"
STAGE_DIRECTORY = BUNDLE_DIRECTORY / "code"
ZIP_PATH = BUNDLE_DIRECTORY / "kaggle_code.zip"
ECAPA_SOURCE = REPO_ROOT.parent / "ecapa_pretrained"
DEFAULT_SCRIPT = "scripts/train.py"

# The check that runs inside the staged repo. It is a separate process with the staged folder as its working directory, so nothing can quietly import from the real repo.
VERIFY_CODE = """
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, ".")
from scripts.train import build_loss_settings, build_model, build_speaker_encoder, get_data_loaders, speaker_embedding, unpack
from src.models.bsrnn import extract
from src.models.losses import bsrnn_loss

config_path, data_root, manifest_dir, weights_path = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
config = yaml.safe_load(open(config_path))
# No worker processes for a two-batch check.
config["data"]["num_workers"] = 0
torch.manual_seed(int(config["seed"]))
train_loader, val_loader = get_data_loaders(manifest_dir, data_root, config)
model = build_model(config)
model.eval()
if weights_path != "none":
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    print(f"starting weights load into this config's model: {weights_path}")
encoder = build_speaker_encoder(config, torch.device("cpu"))
if encoder is not None:
    print("ECAPA snapshot found beside the staged repo")
settings = build_loss_settings(config)
parameter_count = 0
for parameter in model.parameters():
    parameter_count += parameter.numel()
print(f"train trials {len(train_loader.dataset)}, val trials {len(val_loader.dataset)}, parameters {parameter_count:,}")
with torch.no_grad():
    for index, batch in enumerate(val_loader):
        mixture, target, enrollment, crop_absent = unpack(batch, "cpu")
        output = extract(model, mixture, enrollment, enrol_embedding=speaker_embedding(encoder, enrollment))
        total, parts = bsrnn_loss(target, output, mixture, crop_absent, settings)
        print(f"val batch {index}: loss {float(total):.4f}, finite {bool(torch.isfinite(total))}")
        if not torch.isfinite(total):
            raise SystemExit("the loss is not finite")
        if index >= 1:
            break
"""


# this gives the repo files that one python file imports, read from its import lines without running it
def local_imports(path):
    tree = ast.parse(path.read_text())
    module_names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            module_names.append(node.module)
            # "from src.x import y" can also name the module src/x/y.py.
            for alias in node.names:
                module_names.append(f"{node.module}.{alias.name}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_names.append(alias.name)
    found = []
    for module_name in module_names:
        if not (module_name.startswith("src.") or module_name.startswith("scripts.")):
            continue
        module_file = REPO_ROOT / (module_name.replace(".", "/") + ".py")
        package_file = REPO_ROOT / module_name.replace(".", "/") / "__init__.py"
        if module_file.exists():
            found.append(module_file)
        elif package_file.exists():
            found.append(package_file)
    return found


# this follows the imports from train.py until no new file turns up, and adds the __init__.py of every package on the way
def import_closure(start_file):
    to_visit = [start_file]
    seen = set()
    while to_visit:
        path = to_visit.pop()
        if path in seen:
            continue
        seen.add(path)
        for imported in local_imports(path):
            if imported not in seen:
                to_visit.append(imported)
    with_packages = set(seen)
    for path in seen:
        parent = path.parent
        while parent != REPO_ROOT:
            init_file = parent / "__init__.py"
            if init_file.exists():
                with_packages.add(init_file)
            parent = parent.parent
    return sorted(with_packages)


# this gives the md5 of a file, so the notebook and a later reader can tell which weights went up
def md5_of(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        block = handle.read(1 << 20)
        while block:
            digest.update(block)
            block = handle.read(1 << 20)
    return digest.hexdigest()


# this gives the commit the bundle was built from, marked dirty when the working tree has changes
def git_commit():
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()
    if status:
        return commit + "-dirty"
    return commit


# this copies the code, every config, the run-time log and any starting weights into the staged repo, at the same relative paths
def stage_repo(code_files, weights_path):
    repo_directory = STAGE_DIRECTORY / "repo"
    files = list(code_files)
    for config_file in sorted((REPO_ROOT / "experiments/configs").glob("*.yaml")):
        files.append(config_file)
    files.append(REPO_ROOT / "docs/run_times.md")
    if weights_path is not None:
        files.append(weights_path)
    relative_paths = []
    for path in files:
        relative = path.relative_to(REPO_ROOT)
        (repo_directory / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, repo_directory / relative)
        relative_paths.append(str(relative))
    return repo_directory, relative_paths


# this checks the staged copy in its own process, against the local audio, which has the same layout as the Kaggle dataset
def verify(repo_directory, script_relative, config_relative, weights_relative):
    weights_argument = "none"
    if weights_relative is not None:
        weights_argument = weights_relative
    command = [sys.executable, "-c", VERIFY_CODE, config_relative, str(REPO_ROOT / "data"), str(REPO_ROOT / "data/manifests"), weights_argument]
    # Any script other than train.py checks itself with --check, which must run validation batches without gradients.
    if script_relative != DEFAULT_SCRIPT:
        command = [sys.executable, script_relative, "--check", "--config", config_relative, "--data-root", str(REPO_ROOT / "data"), "--manifest-dir", str(REPO_ROOT / "data/manifests")]
    result = subprocess.run(command, cwd=repo_directory, capture_output=True, text=True)
    for line in result.stdout.strip().splitlines():
        print(f"  {line}")
    if result.returncode != 0:
        print(result.stderr[-3000:], file=sys.stderr)
        raise SystemExit("The staged copy failed its check, so do not upload this bundle.")


# this zips the staged folder, so the upload holds repo/ and ecapa_pretrained/ side by side
def zip_stage():
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE_DIRECTORY.rglob("*")):
            # The check leaves compiled caches behind in the staged copy; they are not part of the upload.
            if path.is_file() and "__pycache__" not in path.parts:
                archive.write(path, path.relative_to(STAGE_DIRECTORY))
    return ZIP_PATH.stat().st_size


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--script", default=DEFAULT_SCRIPT, help="the training script the Kaggle run will call")
    parser.add_argument("--config", required=True, help="the config the Kaggle run will train, checked here before upload")
    parser.add_argument("--init-weights", default=None, help="starting weights to ship, for example models/model_sir0_cuecontext-wer-e21.pt")
    parser.add_argument("--no-zip", action="store_true", help="stage and check only")
    arguments = parser.parse_args()

    config_path = REPO_ROOT / arguments.config
    if not config_path.exists():
        raise SystemExit(f"No config at {config_path}.")
    script_path = REPO_ROOT / arguments.script
    if not script_path.exists():
        raise SystemExit(f"No training script at {script_path}.")
    weights_path = None
    if arguments.init_weights:
        weights_path = REPO_ROOT / arguments.init_weights
        if not weights_path.exists():
            raise SystemExit(f"No weights at {weights_path}.")
    if not (ECAPA_SOURCE / "embedding_model.ckpt").exists():
        raise SystemExit(f"No ECAPA snapshot at {ECAPA_SOURCE}.")

    if STAGE_DIRECTORY.exists():
        shutil.rmtree(STAGE_DIRECTORY)
    code_files = import_closure(script_path)
    repo_directory, staged_files = stage_repo(code_files, weights_path)
    shutil.copytree(ECAPA_SOURCE, STAGE_DIRECTORY / "ecapa_pretrained")
    # Generated by Claude Code
    print(f"staged {len(code_files)} code files, {len(staged_files) - len(code_files)} other files and the ECAPA snapshot in {STAGE_DIRECTORY}")
    for path in code_files:
        print(f"  {path.relative_to(REPO_ROOT)}")

    weights_relative = None
    if weights_path is not None:
        weights_relative = str(weights_path.relative_to(REPO_ROOT))
    print("checking the staged copy (no gradients, nothing trained)")
    verify(repo_directory, arguments.script, arguments.config, weights_relative)

    ecapa_files = []
    for path in sorted((STAGE_DIRECTORY / "ecapa_pretrained").iterdir()):
        ecapa_files.append(path.name)
    manifest = {"date": date.today().isoformat(), "git_commit": git_commit(), "script": arguments.script, "config": arguments.config, "init_weights": weights_relative, "files": staged_files, "ecapa": ecapa_files}
    if weights_path is not None:
        manifest["init_weights_md5"] = md5_of(weights_path)
    with open(STAGE_DIRECTORY / "bundle_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)

    if arguments.no_zip:
        print("staged and checked, not zipped (--no-zip)")
        return
    size = zip_stage()
    print(f"{ZIP_PATH}: {size / 1e6:.0f} MB, from commit {manifest['git_commit']}")


if __name__ == "__main__":
    main()
