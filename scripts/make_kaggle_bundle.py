#!/usr/bin/env python3
"""Stage what a Kaggle GPU session needs to train `--split mid`, and nothing else.

WHY TWO BUNDLES
---------------
data/rendered/train is 19,938 trials / ~25 GB. `mid` needs 2,000 of them plus
val's 200 -- ~2.7 GB, small enough for one Kaggle dataset. The code is ~150 KB.

They are emitted as SEPARATE zips because they change at completely different
rates: the audio never changes once rendered, the code changes every time a bug
is found. One combined bundle would mean re-uploading 2.7 GB to fix a one-line
typo. Upload the data zip once; re-upload the code zip as often as you like.

scripts/train.py takes --data-root and --manifest-dir, so the two never need to
sit in the same directory -- the notebook runs code from one Kaggle input and
reads audio from the other.

Deliberately NOT included: the eval harness, the live-model metric, the data
generators, the tests, the literature. None are on the training path.

USAGE
-----
    python scripts/make_kaggle_bundle.py                # both zips
    python scripts/make_kaggle_bundle.py --code-only    # after a code change
    python scripts/make_kaggle_bundle.py --no-zip       # stage, do not zip

Data staging is idempotent: files already present with the right size are
skipped, so re-running after a code change costs seconds, not minutes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]

# Resolved by walking scripts/train.py's imports, not guessed. verify() imports
# from the staged copy, so a new transitive import fails here rather than after
# an upload.
CODE = [
    "scripts/train.py",
    "src/run_log.py",
    "src/data/dataset_loader.py",
    "src/models/bands.py",
    "src/models/bsrnn.py",
    "src/models/conditioning.py",
    "src/models/losses.py",
    "src/models/modules.py",
    "src/models/stft.py",
    "experiments/configs/bsrnn_baseline.yaml",
    "docs/run_times.md",   # src.run_log appends here; give it a real file
]

# Which (manifest stem, rendered audio dir) pairs a split needs. Must match
# SPLIT_MANIFESTS in scripts/train.py -- for `mid` the manifest and the audio
# directory differ, because it is a row-subset over train/val audio; for `sir0`
# they are the same, because those are freshly generated trials.
SPLIT_FILES = {
    "mid":  [("mid_train", "train"), ("mid_val", "val")],
    "sir0": [("sir0_train", "sir0_train"), ("sir0_val", "sir0_val")],
}

# the only files TrialDataset opens, plus meta.json for provenance
TRIAL_FILES = ["mixture.wav", "target.wav", "enrollment.wav", "meta.json"]


def git_commit() -> str:
    """Repo commit, or a clear marker. Stamped into the bundle so a Kaggle run,
    which has no .git, can still record what code produced it."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def copy_if_changed(src: Path, dst: Path) -> bool:
    """True if it copied. Size-only comparison: these files are write-once
    renders, so a matching size means matching content, and stat() is ~1000x
    cheaper than hashing 2.7 GB on every re-run."""
    if dst.exists() and dst.stat().st_size == src.stat().st_size:
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def stage_code(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)          # small; always rebuild clean
    for rel in CODE:
        src = REPO / rel
        if not src.exists():
            sys.exit(f"missing {rel} -- run from the repo root")
        dst = out / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    # Stamp the repo commit into the bundle. There is no .git inside it, so
    # without this every Kaggle run logs "UNKNOWN-not-a-git-checkout" and the
    # result cannot be tied to a code version -- which CLAUDE.md requires.
    (out / "docs").mkdir(parents=True, exist_ok=True)
    (out / "docs/bundle_commit.txt").write_text(git_commit() + "\n")
    print(f"  code: {len(CODE)} files, stamped commit {git_commit()[:12]}")


def stage_data(out: Path, split: str) -> None:
    copied = skipped = 0
    for stem, audio_dir in SPLIT_FILES[split]:
        man = REPO / "data/manifests" / f"{stem}.csv"
        if not man.exists():
            sys.exit(f"missing {man} -- build it with "
                     f"scripts/make_subset_manifest.py --subset {stem}")
        for f in [man, man.with_suffix(".meta.yaml")]:
            if f.exists():
                copy_if_changed(f, out / "data/manifests" / f.name)

        ids = pd.read_csv(man)["trial_id"].tolist()
        for tid in ids:
            for fn in TRIAL_FILES:
                s = REPO / "data/rendered" / audio_dir / tid / fn
                if not s.exists():
                    sys.exit(f"missing rendered file {s}")
                if copy_if_changed(s, out / "data/rendered" / audio_dir / tid / fn):
                    copied += 1
                else:
                    skipped += 1
        print(f"  {stem}: {len(ids)} trials from rendered/{audio_dir}")
    print(f"  audio files: {copied} copied, {skipped} already current")


def verify(code_dir: Path, data_dir: Path, split: str) -> None:
    """Import and run one batch through the loss FROM THE STAGED COPIES.

    Own process, cwd=code_dir, so nothing can silently resolve against the real
    repo. This is the step that catches a missing transitive import before a
    2.7 GB upload rather than after it.
    """
    src = f'''
import sys, yaml, torch
from pathlib import Path
sys.path.insert(0, ".")
from scripts.train import get_data_loaders, build_loss_fn, build_model, unpack
cfg = yaml.safe_load(open("experiments/configs/bsrnn_baseline.yaml"))

# SEEDED, and seeded once at the top. Without this the number below moved on
# every run -- the model is randomly initialised and the train loader has
# shuffle=True, so each run scored different random weights on a different
# random batch (measured swing: 13.5 to 24.5). One seed pins both, because the
# loader's shuffle draws from this same RNG stream. CLAUDE.md: set and log a
# seed for every run. An unreproducible number cannot detect a regression --
# which is the only reason this check exists.
torch.manual_seed(int(cfg["seed"]))

data = Path(r"{data_dir.resolve()}") / "data"
tr, va = get_data_loaders("{split}", data / "manifests", data, cfg)
assert len(tr.dataset) and len(va.dataset), "empty dataset"
L, m = build_loss_fn(cfg), build_model(cfg); m.eval()
print(f"train={{len(tr.dataset)}} val={{len(va.dataset)}} "
      f"params={{sum(p.numel() for p in m.parameters()):,}} "
      f"num_workers={{cfg['data'].get('num_workers')}} seed={{cfg['seed']}}")

# Walk batches until BOTH halves of the objective have been exercised. One batch
# is not enough: at batch 3 and the measured 0.297 absent rate, ~34% of batches
# contain no absent crop, so L_abs comes back nan and a bug in the push-to-
# silence branch would pass this check unnoticed.
seen, shown = {{"present": 0, "absent": 0}}, []
with torch.no_grad():
    for i, b in enumerate(tr):
        x, s, e, a = unpack(b, "cpu")
        loss, parts = L(s, m(x, e), x, a)
        seen["present"] += parts["n_present"]; seen["absent"] += parts["n_absent"]
        shown.append((i, float(loss), parts))
        if seen["present"] and seen["absent"]:
            break
        if i >= 15:
            break
for i, tot, parts in shown:
    print(f"  batch {{i}}: total={{tot:8.4f}} L_pres={{parts['L_pres']:8.4f}} "
          f"L_MR={{parts['L_MR']:.4f}} L_abs={{parts['L_abs']:9.4f}} "
          f"n_pres={{parts['n_present']}} n_abs={{parts['n_absent']}}")
assert seen["present"] and seen["absent"], f"both halves not exercised: {{seen}}"
# untrained model: a large POSITIVE total is the expected result, not a red flag
print(f"  both halves exercised over {{len(shown)}} batch(es): "
      f"{{seen['present']}} present + {{seen['absent']}} absent crops")
'''
    r = subprocess.run([sys.executable, "-c", src], cwd=code_dir,
                       capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        sys.exit("bundle verification FAILED -- do not upload this")
    for line in r.stdout.strip().splitlines():
        print(f"  verified in place: {line}" if not line.startswith("  ") else f"  {line.strip()}")


def zip_dir(d: Path, z: Path, compress: bool) -> None:
    mode = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    with zipfile.ZipFile(z, "w", mode) as zf:
        for f in sorted(d.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(d))
    print(f"  {z} -> {z.stat().st_size / 1e6:.0f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="mid", choices=sorted(SPLIT_FILES),
                    help="which split's audio to bundle")
    ap.add_argument("--out", default="kaggle_bundle")
    ap.add_argument("--code-only", action="store_true",
                    help="skip the audio entirely; use after a code change")
    ap.add_argument("--no-zip", action="store_true")
    args = ap.parse_args()

    root = Path(args.out)
    code_dir, data_dir = root / "code", root / f"data_bundle_{args.split}"

    print(f"bundling split '{args.split}'")
    print(f"staging code -> {code_dir}")
    stage_code(code_dir)

    if not args.code_only:
        print(f"staging data -> {data_dir}")
        stage_data(data_dir, args.split)

    if data_dir.exists():
        verify(code_dir, data_dir, args.split)
    else:
        print("  (no staged data; skipping verification)")

    if not args.no_zip:
        # wav is already PCM: DEFLATE buys almost nothing on the audio and costs
        # minutes. The code zip is text, so compress that one.
        zip_dir(code_dir, root / "kaggle_code.zip", compress=True)
        if not args.code_only:
            # named per split: a sir0 bundle must not silently overwrite the mid
            # one, since `mid` is the control arm and has to stay reproducible
            zip_dir(data_dir, root / f"kaggle_data_{args.split}.zip", compress=False)

        print("\nOn Kaggle, add BOTH as datasets (data once, code whenever it changes):")
        print(f"  {root}/kaggle_data_{args.split}.zip   -> dataset with data/manifests + data/rendered")
        print(f"  {root}/kaggle_code.zip   -> dataset with scripts/ + src/ + experiments/")
        print("Then run notebooks/kaggle_train_mid.ipynb.")


if __name__ == "__main__":
    main()
