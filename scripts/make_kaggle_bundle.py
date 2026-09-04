#!/usr/bin/env python3
"""Stage what a Kaggle GPU session needs to train one split, and nothing else.

    python scripts/make_kaggle_bundle.py --split sir0              # both zips
    python scripts/make_kaggle_bundle.py --split sir0 --code-only  # after a code change
    python scripts/make_kaggle_bundle.py --split sir0 --no-zip     # stage only
    python scripts/make_kaggle_bundle.py --split sir0 --new-only 4976   # see below

TWO ZIPS because they change at different rates: audio never changes once
rendered, code changes constantly. One combined bundle would mean re-uploading
~2.7 GB to fix a typo. train.py takes --data-root and --manifest-dir, so the two
need not sit together.

NOTE --split defaults to `mid`, and --code-only still verifies against whatever
data bundle is staged for that split. Pass the split you actually train.

Not included: the eval harness, the metric, the generators, tests, literature --
none are on the training path. Data staging is idempotent (size-compared).

--new-only N: stage only TRAIN rows N onward, because the first N are already
uploaded and re-sending them is pure wall-clock. Added 2026-09-03, when the
sir0 manifest grew 4,976 -> 9,955 rows and the full zip came to 30.3 GB: at the
measured 0.40 MB/s upload that is ~21 h against ~10.5 h for the new half.

SAFE ONLY BECAUSE THE MANIFEST IS ADDITIVE. build_manifest.py appended rows
rather than regenerating them, so rows 0..N-1 still describe the same audio.
The script PROVES that rather than trusting it: --prefix-manifest takes the
manifest already uploaded and asserts that header + first N rows of the current
one are byte-identical to it. If they are not, the old upload is not reusable
and the run aborts -- the alternative is a dataset whose first half describes
different audio than its manifest claims, which trains without error and is
silently wrong.

The VAL split is staged in full regardless: it is ~200 trials against ~5,000,
so the saving is not worth a second class of partial-bundle bug, and it keeps
the new dataset able to verify itself.

The two halves land as two Kaggle datasets. train.py takes ONE --data-root, so
the notebook symlinks both mounts into one tree -- at the TRIAL-DIRECTORY level,
so it is ~10k symlinks and not ~130k. See scripts/make_kaggle_notebook.py.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
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
    # profile_step.py imports build_model/build_loss_fn/get_data_loaders from
    # train.py and nothing else new, so it adds no transitive dependency. Shipped
    # because notebooks/kaggle_profile_step.ipynb needs it on the GPU box -- the
    # AMP and batch-ceiling numbers cannot be measured on the laptop.
    "scripts/profile_step.py",
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

# the only files TrialDataset opens, plus meta.json for provenance.
# The interferer pair is required by data.both_directions (2026-08-26): without
# them the bundle uploads fine and training dies on the first batch with a
# LibsndfileError, several GB and one Kaggle session too late.
TRIAL_FILES = ["mixture.wav", "target.wav", "enrollment.wav",
               "interferer.wav", "interferer_enrollment.wav", "meta.json"]

# The enrollment bank (scripts/render_enrollment_bank.py, 2026-08-30), DISCOVERED
# rather than listed: it holds as many files as the K it was rendered with, and
# data.enrollment_variants picks among them at train time. A hardcoded list would
# fail the same way TRIAL_FILES was about to -- 9 of the 15 files in a banked
# trial are bank files, so a K=4 sir0_train would upload 2.55 GB short, look
# complete, and die at TrialDataset construction one Kaggle session later.
#
# Optional on purpose: an evaluation split has no bank and must not need one.
# `random_crop=False` forces enrollment_variants to 1, so val reads
# enrollment.wav and a bank there would be dead weight in the upload.
BANK_PATTERNS = ["enrollment_v*.wav", "interferer_enrollment_v*.wav",
                 "enrollment_bank.json"]


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


def stage_data(out: Path, split: str, new_only: int = 0) -> None:
    copied = skipped = 0
    # SPLIT_FILES lists train first, val second; --new-only trims the train
    # rows only. The manifest itself is always staged WHOLE -- the merged run
    # needs all 9,955 rows, and the older dataset carries a stale short copy
    # that the notebook must override.
    for pair_idx, (stem, audio_dir) in enumerate(SPLIT_FILES[split]):
        man = REPO / "data/manifests" / f"{stem}.csv"
        if not man.exists():
            sys.exit(f"missing {man} -- build it with "
                     f"scripts/make_subset_manifest.py --subset {stem}")
        for f in [man, man.with_suffix(".meta.yaml")]:
            if f.exists():
                copy_if_changed(f, out / "data/manifests" / f.name)

        ids = pd.read_csv(man)["trial_id"].tolist()
        n_total = len(ids)
        if new_only and pair_idx == 0:
            if new_only >= n_total:
                sys.exit(f"--new-only {new_only} but {stem} has only {n_total} rows")
            ids = ids[new_only:]
            print(f"  {stem}: --new-only {new_only} -> staging rows "
                  f"{new_only}..{n_total - 1} ({len(ids)} of {n_total} trials); "
                  f"the first {new_only} must already be on Kaggle")
        banked = 0
        for tid in ids:
            trial_dir = REPO / "data/rendered" / audio_dir / tid
            names = list(TRIAL_FILES)
            bank = sorted({f.name for pat in BANK_PATTERNS
                           for f in trial_dir.glob(pat)})
            names += bank
            if bank:
                banked += 1
            for fn in names:
                s = trial_dir / fn
                if not s.exists():
                    sys.exit(f"missing rendered file {s}")
                if copy_if_changed(s, out / "data/rendered" / audio_dir / tid / fn):
                    copied += 1
                else:
                    skipped += 1
        # All-or-nothing per split. A bank on SOME trials is worse than none:
        # the loader falls back to v00 wherever files are missing, so the
        # augmentation would silently apply to part of the set and the run would
        # not be the arm its config claims.
        if banked and banked != len(ids):
            sys.exit(f"{stem}: {banked} of {len(ids)} trials have an enrollment "
                     f"bank. Finish it (scripts/render_enrollment_bank.py "
                     f"--split {stem}) or delete the partial one -- a partial "
                     f"bank makes the run un-attributable.")
        note = f", enrollment bank on all {banked}" if banked else ""
        print(f"  {stem}: {len(ids)} trials from rendered/{audio_dir}{note}")
    print(f"  audio files: {copied} copied, {skipped} already current")


def check_additive(split: str, new_only: int, prefix_manifest: Path) -> None:
    """PROVE the first `new_only` rows are unchanged before skipping them.

    The whole saving rests on the uploaded half still describing the same audio.
    If build_manifest.py had regenerated rather than appended, every row would
    name different levels, positions and onsets -- and the resulting dataset
    would train happily while being silently wrong. That is worth an assertion.

    Byte comparison, not a row count: a same-length row with a different SIR is
    exactly the failure this exists to catch.
    """
    stem = SPLIT_FILES[split][0][0]
    man = REPO / "data/manifests" / f"{stem}.csv"
    with man.open("rb") as fh:
        head = b"".join([fh.readline() for _ in range(new_only + 1)])   # +1 header
    print(f"  --new-only {new_only}: header + first {new_only} rows = {len(head):,} bytes")
    if not prefix_manifest:
        print("  NOT VERIFIED -- no --prefix-manifest given. Compare that byte count")
        print("  against the uploaded manifest's size before trusting this bundle.")
        return
    ref = prefix_manifest.read_bytes()
    if head != ref:
        sys.exit(f"--prefix-manifest MISMATCH: {prefix_manifest} is {len(ref):,} bytes, "
                 f"the current manifest's first {new_only} rows are {len(head):,}. The "
                 f"uploaded half does NOT describe the same trials -- re-upload in full.")
    print(f"  verified byte-identical to {prefix_manifest}")


def write_verify_manifests(td: Path, split: str, new_only: int) -> None:
    """Manifests trimmed to what --new-only actually staged, for verify() only.

    Never zipped and never uploaded: the staged bundle keeps the FULL manifest,
    because the merged run needs all of it.
    """
    (train_stem, _), (val_stem, _) = SPLIT_FILES[split]
    df = pd.read_csv(REPO / "data/manifests" / f"{train_stem}.csv")
    df.iloc[new_only:].to_csv(td / f"{train_stem}.csv", index=False)
    shutil.copy2(REPO / "data/manifests" / f"{val_stem}.csv", td / f"{val_stem}.csv")


def verify(code_dir: Path, data_dir: Path, split: str, man_dir: Path = None) -> None:
    """Import and run one batch through the loss FROM THE STAGED COPIES.

    Own process, cwd=code_dir, so nothing can silently resolve against the real
    repo. This is the step that catches a missing transitive import before a
    2.7 GB upload rather than after it.

    `man_dir` overrides where the manifests are read from. Under --new-only the
    staged bundle holds the FULL manifest but only the tail of the audio, so
    verifying against it would die on the first un-staged trial -- a false
    alarm that would look exactly like a real missing-file bug. main() passes a
    temporary manifest filtered to the staged rows instead, so the check still
    does its job: one real batch, through the real loss, from the staged copy.
    """
    src = f'''
import sys, yaml, torch
from pathlib import Path
sys.path.insert(0, ".")
from scripts.train import get_data_loaders, build_loss_fn, build_model, unpack
cfg = yaml.safe_load(open("experiments/configs/bsrnn_baseline.yaml"))

# SEEDED once at the top: random init + shuffle=True made this number swing
# 13.5 to 24.5 between runs, and an unreproducible number detects no regression.
torch.manual_seed(int(cfg["seed"]))

data = Path(r"{data_dir.resolve()}") / "data"
mans = Path(r"{(man_dir or (data_dir / 'data' / 'manifests')).resolve()}")
tr, va = get_data_loaders("{split}", mans, data, cfg)
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
    ap.add_argument("--new-only", type=int, default=0, metavar="N",
                    help="stage only TRAIN rows N onward; the first N are already "
                         "uploaded. N = the row count of the uploaded manifest.")
    ap.add_argument("--prefix-manifest", type=Path, default=None, metavar="CSV",
                    help="the manifest already uploaded. Asserts header + first N "
                         "rows of the current manifest are byte-identical to it. "
                         "Strongly advised with --new-only.")
    args = ap.parse_args()

    if args.prefix_manifest and not args.new_only:
        sys.exit("--prefix-manifest only means something with --new-only")

    root = Path(args.out)
    code_dir, data_dir = root / "code", root / f"data_bundle_{args.split}"

    print(f"bundling split '{args.split}'")
    print(f"staging code -> {code_dir}")
    stage_code(code_dir)

    if not args.code_only:
        if args.new_only:
            check_additive(args.split, args.new_only, args.prefix_manifest)
        print(f"staging data -> {data_dir}")
        stage_data(data_dir, args.split, args.new_only)

    if data_dir.exists():
        if args.new_only:
            # Verify against a manifest trimmed to what is actually staged.
            with tempfile.TemporaryDirectory() as td:
                write_verify_manifests(Path(td), args.split, args.new_only)
                verify(code_dir, data_dir, args.split, Path(td))
        else:
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
        if args.new_only:
            print(f"\n  THIS IS A PARTIAL BUNDLE: train rows {args.new_only} onward only.")
            print("  Upload it as a SECOND dataset and keep the first one -- deleting")
            print("  it makes this bundle unusable. In the notebook set DATA_DIR to")
            print("  the OLD dataset and DATA_DIR_NEW to this one; the merge cell")
            print("  symlinks both into /kaggle/working/merged and reads the full")
            print("  manifest from the NEW one.")
        print("Then run notebooks/kaggle_train_mid.ipynb.")


if __name__ == "__main__":
    main()
