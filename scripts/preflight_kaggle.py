#!/usr/bin/env python3
"""What arm will the uploaded bundle + notebook actually train? Answered WITHOUT
running anything.

    ../tse_venv/bin/python scripts/preflight_kaggle.py

WHY THIS EXISTS. decisions-pending.md E8: the config was once hardcoded in six
places, so a run could probe one arm, train another and archive a third while
every step printed "OK". The notebook now declares CONFIG once, but the thing
that actually trains is the copy of that config INSIDE THE ZIP, and the zip is
built at a different moment from the notebook. This reads both, from the
artefacts themselves, and says what will happen.

Reads: kaggle_bundle/kaggle_code.zip and notebooks/kaggle_train_mid.ipynb.
Never imports the model, so it cannot be fooled by the local working tree.
"""

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
ZIP = REPO / "kaggle_bundle/kaggle_code.zip"
NB = REPO / "notebooks/kaggle_train_mid.ipynb"

# model keys that DEFINE the arm. Anything here differing between two runs means
# they are different experiments, whatever the file is called.
ARM_KEYS = ("tfmap_parts", "tfmap_part_scales", "context_embedding",
            "tfmap_inject", "state_head", "tfmap_scale")


def notebook_knobs():
    source = "".join("".join(c["source"]) for c in json.loads(NB.read_text())["cells"])
    out = {}
    for line in source.split("\n"):
        for key in ("SPLIT", "EPOCHS", "BATCH_SIZE", "CONFIG", "DATA_DIR", "CODE"):
            if line.startswith(key) and "=" in line:
                out.setdefault(key, line.split("=", 1)[1].split("#")[0].strip())
    return out


def main():
    if not ZIP.exists():
        sys.exit(f"no bundle at {ZIP} -- run scripts/make_kaggle_bundle.py")
    knobs = notebook_knobs()
    config_rel = knobs.get("CONFIG", "").strip('"\'')

    with zipfile.ZipFile(ZIP) as z:
        names = set(z.namelist())
        stamp = next((z.read(n).decode().strip() for n in names
                      if n.endswith("bundle_commit.txt")), "(none)")
        member = next((n for n in names if n.endswith(config_rel)), None)
        if member is None:
            sys.exit(f"\n  THE NOTEBOOK ASKS FOR {config_rel}\n"
                     f"  AND IT IS NOT IN THE ZIP. The run would fail, or worse,\n"
                     f"  fall back to something else. Re-run make_kaggle_bundle.py.")
        cfg = yaml.safe_load(z.read(member))
        staged = sorted(n.split("configs/")[-1] for n in names if "/configs/" in n)

    head = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO,
                          capture_output=True, text=True).stdout.strip()

    print(f"\n  bundle   {ZIP.name}  stamped {stamp[:12]}"
          f"{'  <- MATCHES HEAD' if stamp.startswith(head) else f'  <- HEAD IS {head}'}")
    if stamp.endswith("-dirty"):
        print("           STAMP IS -dirty: built from uncommitted changes, so the")
        print("           commit does NOT identify the code that will run.")
    print(f"  notebook {NB.name}")
    for key in ("SPLIT", "EPOCHS", "BATCH_SIZE", "CONFIG"):
        print(f"    {key:<11}{knobs.get(key, '(not found)')}")

    # THE CONTENT PROBE. Checked here because a run whose scheduler and
    # selector read `content_wer` will die at the end of epoch 0 if the probe
    # is off, and a run with the probe on but the old selector silently wastes
    # the ASR time. decisions-m2.md 2026-09-23.
    probe = (cfg.get("content_probe") or {})
    train_cfg = cfg.get("training", {})
    reads_wer = {train_cfg.get("select_on"), train_cfg.get("lr_schedule_on")} & {"content_wer"}
    print("\n  THE CONTENT PROBE (validation WER in the loop):")
    print(f"    enabled             {probe.get('enabled', False)}")
    if probe.get("enabled"):
        print(f"    split / n_trials    {probe.get('split')} / {probe.get('n_trials')}")
        print(f"    every_n_epochs      {probe.get('every_n_epochs', 1)}")
        print(f"    asr_device          {probe.get('asr_device', 'cpu')}")
    print(f"    select_on           {train_cfg.get('select_on')}")
    print(f"    lr_schedule_on      {train_cfg.get('lr_schedule_on')}")
    print(f"    keep_stride         {train_cfg.get('keep_stride', 0)}"
          f"   keep_top_k {train_cfg.get('keep_top_k')}")
    if reads_wer and not probe.get("enabled"):
        sys.exit("\n  THE SCHEDULER OR SELECTOR READS `content_wer` AND THE PROBE IS OFF.\n"
                 "  train.py raises at the end of epoch 0. Set content_probe.enabled true.")
    if reads_wer and int(probe.get("every_n_epochs", 1)) != 1:
        sys.exit("\n  `content_wer` IS READ EVERY EPOCH BUT THE PROBE DOES NOT RUN EVERY EPOCH.\n"
                 "  Set content_probe.every_n_epochs to 1.")
    if probe.get("enabled") and probe.get("split") in ("sir0_privval", "eval_private"):
        sys.exit(f"\n  THE PROBE WOULD STEER TRAINING ON {probe.get('split')}, A REPORTED HOLDOUT.")
    if probe.get("enabled") and train_cfg.get("keep_stride", 0) in (0, None):
        print("    NOTE keep_stride is 0, so only the top-k by score and _last.pt")
        print("         survive. 1a's best content epoch ranked 10th of 14.")

    print(f"\n  THE ARM, read from the config INSIDE THE ZIP ({member}):")
    for key in ARM_KEYS:
        value = cfg["model"].get(key, "(absent)")
        print(f"    {key:<20}{value}")
    print(f"    {'batch_size':<20}{cfg['data'].get('batch_size')}  (data.batch_size; "
          f"the notebook's BATCH_SIZE is a CEILING that overrides it)")
    print(f"    {'lr_schedule_on':<20}{cfg['training'].get('lr_schedule_on', '(absent => total)')}")

    context = bool(cfg["model"].get("context_embedding", False))
    parts = bool(cfg["model"].get("tfmap_parts", False))
    arm = ("1c -- cue parts PLUS the frozen ECAPA identity anchor" if context
           else "1a -- cue parts, no identity anchor" if parts
           else "baseline / other -- neither cue parts nor the anchor")
    print(f"\n  => THIS WILL TRAIN: {arm}")

    if context:
        needed = ["src/models/context_encoder.py", "ecapa_pretrained/embedding_model.ckpt"]
        for want in needed:
            ok = any(n.endswith(want) for n in names)
            print(f"     {'OK ' if ok else 'MISSING'}  {want}")
        print("     expect this line in the log, or the arm is NOT on:")
        print('       context encoder: {"backbone": "ECAPA-TDNN", ...}')
    print(f"\n  configs staged: {', '.join(staged)}\n")


if __name__ == "__main__":
    main()
