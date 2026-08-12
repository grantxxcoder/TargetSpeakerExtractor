#!/usr/bin/env python3
"""Generate the speaker-disjoint split assignment for the constructed set.

Corpus: LibriSpeech (Panayotov, Chen, Povey & Khudanpur, ICASSP 2015).

WHY THIS EXISTS
---------------
Every trial in the constructed set is built from LibriSpeech speakers. If a
speaker appears in both training and evaluation, the model has heard that
voice before and the eval number is inflated — silently, with no error and
no obvious symptom. Speech splits must be *speaker*-disjoint, not merely
file-disjoint (docs/data/definitions.md:315).

This script pins the assignment ONCE, deterministically, before any data is
generated, and writes it to experiments/configs/splits.yaml. Everything
downstream reads that file. Nothing else is allowed to decide which speaker
goes where.

Run it once. Commit the output. Do not re-run with a different seed after
generating data, because that silently redefines what "eval" means.

HOW THE ASSIGNMENT WORKS
------------------------
  train         train-clean-100 + train-clean-360   (1172 speakers)
  val           dev-clean                           (40 speakers)
  eval_public   half of test-clean                  (20 speakers)
  eval_private  the other half of test-clean        (20 speakers)

test-clean is split in half because docs/data/metric-definitions.md:198-200
requires a private split held back from publication, so that headline
numbers cannot be overfitted to trials anyone can see.

The public/private halves are stratified by sex, so the two halves stay
comparable and a difference between them means something.

Two "smoke" subsets are also emitted for laptop prototyping. These are
SUBSETS of train and val, not peers of them — they exist so you can run the
pipeline end to end on a machine without a GPU.

USAGE
-----
    python scripts/make_splits.py
    python scripts/make_splits.py --librispeech-root data/librispeech/LibriSpeech
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Fixed for the life of the project. Changing this invalidates every result
# produced before the change. If you must change it, log it in decisions.md.
SEED = 42

TRAIN_SUBSETS = ("train-clean-100", "train-clean-360")
VAL_SUBSETS = ("dev-clean",)
EVAL_SUBSETS = ("test-clean",)

SMOKE_TRAIN_SPEAKERS = 20
SMOKE_VAL_SPEAKERS = 5


def parse_speakers(speakers_txt: Path) -> list[dict]:
    """Parse LibriSpeech SPEAKERS.TXT.

    Format is pipe-separated with ';'-prefixed comment lines:
        ;ID  |SEX| SUBSET           |MINUTES| NAME
        14   | F | train-clean-360  | 25.03 | Kristin LeMoine
    """
    rows = []
    for line in speakers_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 4:
            continue
        rows.append({"id": parts[0], "sex": parts[1], "subset": parts[2]})
    if not rows:
        sys.exit(f"ERROR: parsed zero speakers from {speakers_txt}. Wrong file?")
    return rows


def stratified_halves(speakers: list[dict], seed: int) -> tuple[list[str], list[str]]:
    """Split speakers into two halves, balanced by sex, deterministically.

    Deterministic without importing `random`: speakers are ordered by a hash
    of (seed, speaker id), then alternately dealt into the two halves within
    each sex group. Same inputs always give the same output, on any machine,
    on any Python version — which `random.shuffle` does not guarantee across
    versions.
    """
    by_sex: dict[str, list[str]] = defaultdict(list)
    for s in speakers:
        by_sex[s["sex"]].append(s["id"])

    a: list[str] = []
    b: list[str] = []
    # The counter runs ACROSS sex groups, not within them. If it reset per
    # group, every odd-sized group would hand its extra speaker to `a`, and
    # two odd groups would give a 21/19 split instead of 20/20.
    n = 0
    for sex in sorted(by_sex):
        ordered = sorted(
            by_sex[sex],
            key=lambda sid: hashlib.sha256(f"{seed}:{sid}".encode()).hexdigest(),
        )
        for sid in ordered:
            (a if n % 2 == 0 else b).append(sid)
            n += 1
    return sorted(a, key=int), sorted(b, key=int)


def deterministic_sample(ids: list[str], n: int, seed: int, tag: str) -> list[str]:
    """Pick n ids reproducibly, by hash order. Same rationale as above."""
    ordered = sorted(
        ids, key=lambda sid: hashlib.sha256(f"{seed}:{tag}:{sid}".encode()).hexdigest()
    )
    return sorted(ordered[:n], key=int)


def git_commit() -> str:
    """Current commit hash, or a clear marker if unavailable.

    CLAUDE.md requires the commit hash alongside every generated artefact.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, check=True, timeout=10,
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def yaml_list(ids: list[str], indent: int = 4) -> str:
    pad = " " * indent
    return "\n".join(f"{pad}- \"{i}\"" for i in ids)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--librispeech-root",
        default="data/librispeech/LibriSpeech",
        help="Directory containing SPEAKERS.TXT",
    )
    ap.add_argument("--out", default="experiments/configs/splits.yaml")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    speakers_txt = Path(args.librispeech_root) / "SPEAKERS.TXT"
    if not speakers_txt.exists():
        sys.exit(
            f"ERROR: {speakers_txt} not found.\n"
            "Finish docs/data/data-setup.md step 3 first, or pass --librispeech-root."
        )

    rows = parse_speakers(speakers_txt)
    src_md5 = hashlib.md5(speakers_txt.read_bytes()).hexdigest()

    def in_subsets(names: tuple[str, ...]) -> list[dict]:
        return [r for r in rows if r["subset"] in names]

    train = sorted((r["id"] for r in in_subsets(TRAIN_SUBSETS)), key=int)
    val = sorted((r["id"] for r in in_subsets(VAL_SUBSETS)), key=int)
    eval_rows = in_subsets(EVAL_SUBSETS)
    eval_public, eval_private = stratified_halves(eval_rows, args.seed)

    smoke_train = deterministic_sample(train, SMOKE_TRAIN_SPEAKERS, args.seed, "smoke_train")
    smoke_val = deterministic_sample(val, SMOKE_VAL_SPEAKERS, args.seed, "smoke_val")

    # --- Assertions. These are the point of the script. -------------------
    # Peer splits must be mutually disjoint. Assert, never eyeball.
    peers = {
        "train": set(train),
        "val": set(val),
        "eval_public": set(eval_public),
        "eval_private": set(eval_private),
    }
    names = sorted(peers)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            overlap = peers[a] & peers[b]
            assert not overlap, f"SPEAKER LEAK: {a} ∩ {b} = {sorted(overlap)}"

    # Smoke sets are deliberate subsets, not peers. Verify that relationship
    # holds, so they can never accidentally become a fifth and sixth split.
    assert set(smoke_train) <= peers["train"], "smoke_train escaped train"
    assert set(smoke_val) <= peers["val"], "smoke_val escaped val"

    # The eval halves must be equal-sized and sex-balanced, because the whole
    # point of stratifying is that a public-vs-private difference means
    # something. Assert it rather than trusting the dealing logic.
    assert len(eval_public) == len(eval_private), (
        f"eval halves unequal: {len(eval_public)} vs {len(eval_private)}"
    )
    sex_of = {r["id"]: r["sex"] for r in eval_rows}
    for sex in {r["sex"] for r in eval_rows}:
        n_pub = sum(1 for s in eval_public if sex_of[s] == sex)
        n_prv = sum(1 for s in eval_private if sex_of[s] == sex)
        assert abs(n_pub - n_prv) <= 1, f"sex {sex} unbalanced: {n_pub} vs {n_prv}"

    # SPEAKERS.TXT lists all 2,484 LibriSpeech speakers regardless of which
    # archives you extracted, so this can only catch a corrupted or edited
    # SPEAKERS.TXT — it cannot detect a failed extraction. Audio presence is
    # checked separately below.
    expected = {"train": 1172, "val": 40, "eval_public": 20, "eval_private": 20}
    actual = {k: len(v) for k, v in peers.items()}
    if actual != expected:
        print(
            f"WARNING: speaker counts {actual} != expected {expected}.\n"
            "SPEAKERS.TXT does not match the published LibriSpeech release.",
            file=sys.stderr,
        )

    # Real extraction check: does each assigned speaker have audio on disk?
    root = Path(args.librispeech_root)
    subset_of = {r["id"]: r["subset"] for r in rows}
    missing = [
        sid
        for split in peers.values()
        for sid in split
        if not (root / subset_of[sid] / sid).is_dir()
    ]
    if missing:
        print(
            f"WARNING: {len(missing)} assigned speakers have no directory on "
            f"disk (e.g. {missing[:5]}).\n"
            "An archive probably failed to extract, or extracted to a nested "
            "path. Check docs/data/data-setup.md step 3d.",
            file=sys.stderr,
        )

    # Resolve once. Calling git_commit() again after writing the YAML would
    # report "-dirty" (the new untracked file), giving one artefact two
    # different provenance strings.
    commit = git_commit()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"""# Speaker-disjoint split assignment for the constructed set.
#
# GENERATED FILE — do not hand-edit. Regenerate with:
#     python scripts/make_splits.py
#
# Pinned before any data was generated. Changing this file after mixtures
# exist silently redefines what "eval" means and invalidates every prior
# result. See docs/data/data-setup.md step 6.

meta:
  generated: "{date.today().isoformat()}"
  generator: "scripts/make_splits.py"
  seed: {args.seed}
  git_commit: "{commit}"
  source: "LibriSpeech SPEAKERS.TXT"
  source_md5: "{src_md5}"
  citation: "Panayotov et al., ICASSP 2015"

counts:
  train: {len(train)}
  val: {len(val)}
  eval_public: {len(eval_public)}
  eval_private: {len(eval_private)}

# train-clean-100 + train-clean-360
train:
{yaml_list(train)}

# dev-clean
val:
{yaml_list(val)}

# test-clean, first half, sex-stratified. Publishable.
eval_public:
{yaml_list(eval_public)}

# test-clean, second half, sex-stratified. HELD BACK — headline numbers only.
# Required by docs/data/metric-definitions.md:198-200.
eval_private:
{yaml_list(eval_private)}

# Laptop prototyping subsets. SUBSETS of train/val above, not separate
# splits. Use for smoke tests on a machine without a GPU.
smoke_train:
{yaml_list(smoke_train)}

smoke_val:
{yaml_list(smoke_val)}
""",
        encoding="utf-8",
    )

    print(f"Wrote {out}")
    for k, v in actual.items():
        print(f"  {k:<13} {v:>5} speakers")
    print(f"  smoke_train   {len(smoke_train):>5} speakers (subset of train)")
    print(f"  smoke_val     {len(smoke_val):>5} speakers (subset of val)")
    print(f"  seed={args.seed}  commit={commit}")
    print("Disjointness asserted. Commit this file before generating data.")


if __name__ == "__main__":
    main()
