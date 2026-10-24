"""Render K enrolment clips per trial, so training can change the cue every epoch (decisions-m2.md 2026-08-30).

    python scripts/render_enrollment_bank.py --split sir0_train --variants 3

The single enrollment.wav was the same 5 s waveform on every epoch, which made "this waveform means this voice" a lookup the model memorised.
A bank of different utterances by the same speaker removes that lookup. Each variant goes through the same render.render_enrollment as enrollment.wav, so it is dry (A4), the manifest's length (B3),
guarded by B10's book, chapter and utterance tiers, levelled to the trial's target_loudness_lufs, and EQ'd when the manifest says so, with its own curve.
Variant 00 reproduces enrollment.wav exactly. Mixtures, targets and interferers are untouched.
"""

import argparse
import csv
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyloudnorm
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_manifest import pick_enrollment, read_books, read_speakers
from scripts.render_trials import flac_paths, read_index
from src.data import render
from src.run_log import timed

CONFIG_PATH = "experiments/configs/generator.yaml"
MANIFEST_DIR = Path("data/manifests")
RENDERED_DIR = Path("data/rendered")
INDEX_DIR = Path("data/index")
WORKERS = 8

# "" is the target's enrolment and "interferer_" is the other direction's. Both are built identically.
PREFIXES = ("", "interferer_")

# What every worker process needs, set once when the worker starts.
worker_state = {}


# this gives every utterance by the enrolled speaker that is long enough to be an enrolment
def candidate_pool(row, prefix, by_speaker, enrollment_length):
    """On a phantom trial the enrolled speaker is not in the mixture at all, and the right answer is silence."""
    if prefix == "":
        speaker = row["target_speaker"]
    else:
        speaker = row["interferer_enrollment_speaker"]
    candidates = []
    for utterance in by_speaker[speaker]:
        if float(utterance["duration"]) >= enrollment_length:
            candidates.append(utterance)
    return speaker, candidates


# this gives (chapter key, utterances used in the mixture) for the guard tiers, or (None, empty) when the speaker has no audio in the mixture
def mixture_context(row, prefix):
    if prefix == "":
        speaker, chapter, used = row["target_speaker"], row["target_chapter"], row["target_utts"]
    else:
        # A phantom's speaker says nothing in the mixture, so there is no content to leak from.
        if int(row["interferer_enrollment_phantom"]):
            return None, set()
        speaker, chapter, used = row["interferer_speaker"], row["interferer_chapter"], row["interferer_utts"]
    if not chapter:
        return None, set()
    used_utterances = set()
    for utterance in used.split("|"):
        if utterance:
            used_utterances.add(utterance)
    return (speaker, chapter), used_utterances


# this gives the manifest's own enrolment first, then up to n - 1 different alternatives
def draw_variants(row, prefix, by_speaker, book, n_variants):
    """Each alternative is drawn through pick_enrollment with the drawn ones removed, so the bank falls back tier by tier. A speaker with too few candidates gets a shorter bank rather than repeats."""
    enrollment_length = float(row[f"{prefix}enrollment_length_s"])
    speaker, pool = candidate_pool(row, prefix, by_speaker, enrollment_length)
    chapter, used = mixture_context(row, prefix)

    first = {"utt": row[f"{prefix}enrollment_utt"], "offset_s": float(row[f"{prefix}enrollment_offset_s"]), "tier": row[f"{prefix}enrollment_guard"], "duration": None}
    variants = [first]
    taken = {first["utt"]}

    # Seeded from the trial id, so the bank follows from the manifest alone, whatever the worker count.
    rng = np.random.default_rng(render.trial_seed(f"{row['trial_id']}#bank#{prefix}"))
    for draw_index in range(n_variants - 1):
        remaining = []
        for utterance in pool:
            if utterance["utt"] not in taken:
                remaining.append(utterance)
        if not remaining:
            break
        chosen = pick_enrollment(rng, remaining, chapter, used, book)
        if chosen is None:
            break
        tier, utterance = chosen
        span = float(utterance["duration"]) - enrollment_length
        variants.append({"utt": utterance["utt"], "offset_s": round(float(rng.uniform(0, max(span, 0.0))), 4), "tier": tier, "duration": float(utterance["duration"])})
        taken.add(utterance["utt"])
    return speaker, variants


# this renders one variant through the same enrolment renderer as enrollment.wav
def render_variant(row, prefix, variant, variant_index, sample_rate, audio_paths, meter):
    """The EQ is seeded from the trial id, so variants after the first add a "#v{k}" suffix, and variant 0 keeps the bare id and matches enrollment.wav byte for byte."""
    synthetic = dict(row)
    if variant_index > 0:
        synthetic["trial_id"] = f"{row['trial_id']}#v{variant_index}"
    synthetic[f"{prefix}enrollment_utt"] = variant["utt"]
    synthetic[f"{prefix}enrollment_offset_s"] = variant["offset_s"]
    return render.render_enrollment(synthetic, sample_rate, audio_paths, meter, float(row["target_loudness_lufs"]), prefix)


# this stores what the workers share, once per worker process
def start_worker(sample_rate, audio_paths, by_speaker, book, out_dir, n_variants):
    worker_state["sample_rate"] = sample_rate
    worker_state["audio_paths"] = audio_paths
    worker_state["by_speaker"] = by_speaker
    worker_state["book"] = book
    worker_state["out_dir"] = out_dir
    worker_state["n_variants"] = n_variants
    worker_state["meter"] = pyloudnorm.Meter(sample_rate)


# this renders one trial's bank, returning (trial id, status, note)
def render_one(row):
    trial_dir = worker_state["out_dir"] / row["trial_id"]
    n_variants = worker_state["n_variants"]
    if not trial_dir.is_dir():
        return row["trial_id"], "failed", "trial not rendered, run render_trials.py first"
    if (trial_dir / "enrollment_bank.json").exists():
        return row["trial_id"], "skipped", ""

    record = {"n_requested": n_variants, "variants": {}}
    try:
        for prefix in PREFIXES:
            speaker, variants = draw_variants(row, prefix, worker_state["by_speaker"], worker_state["book"], n_variants)
            entries = []
            for variant_index, variant in enumerate(variants):
                audio, bands = render_variant(row, prefix, variant, variant_index, worker_state["sample_rate"], worker_state["audio_paths"], worker_state["meter"])
                sf.write(trial_dir / f"{prefix}enrollment_v{variant_index:02d}.wav", audio, worker_state["sample_rate"], subtype="PCM_16")
                entry = dict(variant)
                entry["eq_bands"] = bands
                entries.append(entry)
            if prefix == "":
                direction = "target"
            else:
                direction = prefix
            record["variants"][direction] = {"speaker": speaker, "n": len(entries), "entries": entries}
    except Exception as error:
        return row["trial_id"], "failed", f"{type(error).__name__}: {error}"

    # Written last, because its presence is what marks the bank complete.
    (trial_dir / "enrollment_bank.json").write_text(json.dumps(record, indent=1))
    shortest = n_variants
    for direction in record["variants"]:
        shortest = min(shortest, record["variants"][direction]["n"])
    if shortest == n_variants:
        return row["trial_id"], "rendered", ""
    return row["trial_id"], "rendered", f"only {shortest} variants"


# this is where the bank render starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    parser.add_argument("--variants", type=int, required=True, help="bank size K, counting the original enrolment")
    args = parser.parse_args()

    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    sample_rate = config["sample_rate"]
    librispeech_root = Path(config["paths"]["librispeech"])

    with (MANIFEST_DIR / f"{args.split}.csv").open() as handle:
        rows = list(csv.DictReader(handle))

    # Every utterance by every speaker in the split, not only the ones the mixtures used. That surplus is the bank.
    index = read_index(INDEX_DIR / f"utterances_{args.split}.csv")
    audio_paths = flac_paths(librispeech_root, index, read_speakers(librispeech_root))
    by_speaker = {}
    for utterance, row in index.items():
        by_speaker.setdefault(row["speaker"], []).append({"utt": utterance, "speaker": row["speaker"], "chapter": row["chapter"], "duration": row["duration"]})
    for utterances in by_speaker.values():
        utterances.sort(key=lambda utterance: utterance["utt"])
    book = read_books(librispeech_root)

    out_dir = RENDERED_DIR / args.split
    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    failures = []
    short_banks = []
    with timed(f"scripts/render_enrollment_bank.py --split {args.split} --variants {args.variants}", scope=lambda: f"{counts['rendered']:,} trials x {args.variants} variants", rate=f"{WORKERS} workers, 16 kHz PCM_16"):
        with Pool(WORKERS, initializer=start_worker, initargs=(sample_rate, audio_paths, by_speaker, book, out_dir, args.variants)) as pool:
            for trial_id, status, note in pool.imap_unordered(render_one, rows, chunksize=4):
                counts[status] += 1
                if status == "failed":
                    failures.append((trial_id, note))
                elif note:
                    short_banks.append((trial_id, note))

    (out_dir / "enrollment_bank.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/render_enrollment_bank.py",
        "split": args.split,
        "config": CONFIG_PATH,
        "variants": args.variants,
        "n_trials": len(rows),
        "n_rendered": counts["rendered"],
        "n_skipped": counts["skipped"],
        "n_failed": counts["failed"],
        "n_short": len(short_banks),
    }, sort_keys=False))

    # Generated by Claude Code
    print(f"Wrote banks under {out_dir}  ({counts['rendered']} rendered, {counts['skipped']} skipped, {counts['failed']} failed)")
    if short_banks:
        print(f"  {len(short_banks)} trial(s) got fewer than {args.variants} variants, e.g. {short_banks[0][0]} ({short_banks[0][1]})")
    if failures:
        for trial_id, note in failures[:5]:
            print(f"  FAILED {trial_id}  {note}", file=sys.stderr)
        sys.exit(f"{len(failures)} trial(s) failed")


if __name__ == "__main__":
    main()
