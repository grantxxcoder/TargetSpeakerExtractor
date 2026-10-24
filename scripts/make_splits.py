"""Generate the speaker-disjoint split assignment from LibriSpeech (Panayotov, Chen, Povey & Khudanpur, ICASSP 2015).

    python scripts/make_splits.py

A speaker heard in training and again in evaluation inflates the evaluation number with no visible symptom, so the splits must be disjoint by speaker, not only by file.
This pins the assignment once, before any data exists, into experiments/configs/splits.yaml, which everything downstream reads.
train is train-clean-100 plus train-clean-360 (1,172 speakers), val is dev-clean (40), and test-clean is halved into eval_public and eval_private (20 each).
The halves are balanced by sex and by enrolment guard tier, so a difference between them means something (B10, decisions-m0.md 2026-08-13).
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

# Fixed for the life of the project, because changing it redefines what "eval" means.
SEED = 42
LIBRISPEECH_ROOT = Path("data/librispeech/LibriSpeech")
OUT_PATH = Path("experiments/configs/splits.yaml")

TRAIN_SUBSETS = ("train-clean-100", "train-clean-360")
VAL_SUBSETS = ("dev-clean",)
EVAL_SUBSETS = ("test-clean",)


# this reads every speaker's id, sex and subset from SPEAKERS.TXT
def parse_speakers(speakers_txt):
    rows = []
    for line in speakers_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        parts = []
        for part in line.split("|"):
            parts.append(part.strip())
        if len(parts) < 4:
            continue
        rows.append({"id": parts[0], "sex": parts[1], "subset": parts[2]})
    return rows


# this reads {chapter id: (speaker id, book id)} from CHAPTERS.TXT
def parse_chapters(chapters_txt):
    chapters = {}
    for line in chapters_txt.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        parts = []
        for part in line.split("|"):
            parts.append(part.strip())
        if len(parts) < 6:
            continue
        chapters[parts[0]] = (parts[1], parts[5])
    return chapters


# this gives the strongest B10 guard tier a speaker can support: another book, another chapter of the same book, or only another utterance
def guard_tier(speaker_id, chapters):
    books = set()
    chapter_count = 0
    for chapter_speaker, chapter_book in chapters.values():
        if chapter_speaker == speaker_id:
            books.add(chapter_book)
            chapter_count += 1
    if len(books) >= 2:
        return "book"
    if chapter_count >= 2:
        return "chapter"
    return "utterance"


# this orders speaker ids by a hash of the seed and the id, which reproduces on any machine and any Python version, unlike random.shuffle
def hash_order(speaker_ids, seed):
    return sorted(speaker_ids, key=lambda speaker_id: hashlib.sha256(f"{seed}:{speaker_id}".encode()).hexdigest())


# this deals each stratum's speakers alternately into two halves, where bit i of mask says which half stratum i starts in
def deal(strata, ordered, mask):
    first_half = []
    second_half = []
    for stratum_index, stratum in enumerate(strata):
        starts_second = (mask >> stratum_index) & 1
        for position, speaker_id in enumerate(ordered[stratum]):
            if (position + starts_second) % 2:
                second_half.append(speaker_id)
            else:
                first_half.append(speaker_id)
    return first_half, second_half


# this counts how many of the speakers carry a given label
def count_label(speaker_ids, label_of, label):
    count = 0
    for speaker_id in speaker_ids:
        if label_of[speaker_id] == label:
            count += 1
    return count


# this checks that two halves are the same size and within one speaker of each other on every sex and every tier
def is_balanced(first_half, second_half, sex_of, tier_of):
    if len(first_half) != len(second_half):
        return False
    for label_of in (sex_of, tier_of):
        for label in sorted(set(label_of.values())):
            if abs(count_label(first_half, label_of, label) - count_label(second_half, label_of, label)) > 1:
                return False
    return True


# this splits speakers into two halves balanced by sex and guard tier
def stratified_halves(speakers, seed, tier_of):
    """Alternating within a stratum balances that stratum, but the odd speaker of each stratum must be steered, so the start side of each stratum is searched in a fixed order and the first balanced choice is taken."""
    by_stratum = defaultdict(list)
    sex_of = {}
    for speaker in speakers:
        by_stratum[(speaker["sex"], tier_of[speaker["id"]])].append(speaker["id"])
        sex_of[speaker["id"]] = speaker["sex"]
    strata = sorted(by_stratum)
    ordered = {}
    for stratum in strata:
        ordered[stratum] = hash_order(by_stratum[stratum], seed)
    for mask in range(1 << len(strata)):
        first_half, second_half = deal(strata, ordered, mask)
        if is_balanced(first_half, second_half, sex_of, tier_of):
            return sorted(first_half, key=int), sorted(second_half, key=int)
    raise RuntimeError(f"No split balances sex and guard tier across {len(strata)} strata.")


# this writes speaker ids as an indented YAML list
def yaml_list(speaker_ids):
    lines = []
    for speaker_id in speaker_ids:
        lines.append(f"    - \"{speaker_id}\"")
    return "\n".join(lines)


# this gives the sorted ids of the speakers in the named subsets
def speakers_in(rows, subsets):
    speaker_ids = []
    for row in rows:
        if row["subset"] in subsets:
            speaker_ids.append(row["id"])
    return sorted(speaker_ids, key=int)


# this is where the split assignment starts
def main():
    # It takes no options, but parsing still answers --help instead of starting the job.
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    speakers_txt = LIBRISPEECH_ROOT / "SPEAKERS.TXT"
    rows = parse_speakers(speakers_txt)
    source_md5 = hashlib.md5(speakers_txt.read_bytes()).hexdigest()

    train = speakers_in(rows, TRAIN_SUBSETS)
    val = speakers_in(rows, VAL_SUBSETS)
    eval_rows = []
    for row in rows:
        if row["subset"] in EVAL_SUBSETS:
            eval_rows.append(row)

    chapters = parse_chapters(LIBRISPEECH_ROOT / "CHAPTERS.TXT")
    tier_of = {}
    for row in eval_rows:
        tier_of[row["id"]] = guard_tier(row["id"], chapters)
    eval_public, eval_private = stratified_halves(eval_rows, SEED, tier_of)

    # The whole point of the script: no speaker may sit in two splits.
    splits = {"train": set(train), "val": set(val), "eval_public": set(eval_public), "eval_private": set(eval_private)}
    names = sorted(splits)
    for first_index, first_name in enumerate(names):
        for second_name in names[first_index + 1:]:
            shared = splits[first_name] & splits[second_name]
            if shared:
                sys.exit(f"Speaker leak between {first_name} and {second_name}: {sorted(shared)}")

    OUT_PATH.write_text(f"""# Speaker-disjoint split assignment for the constructed set.
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
  seed: {SEED}
  source: "LibriSpeech SPEAKERS.TXT"
  source_md5: "{source_md5}"
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
""", encoding="utf-8")

    # Generated by Claude Code
    print(f"Wrote {OUT_PATH}")
    for name in ("train", "val", "eval_public", "eval_private"):
        print(f"  {name:<13} {len(splits[name]):>5} speakers")


if __name__ == "__main__":
    main()
