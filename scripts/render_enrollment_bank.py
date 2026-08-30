#!/usr/bin/env python3
"""Render K alternative enrollment clips per trial, so the cue varies per epoch.

    python scripts/render_enrollment_bank.py --split sir0_train --variants 4

WHY THIS EXISTS
---------------
`enrollment.wav` is rendered once and read in full on every epoch, so for a
given trial the model saw the SAME 5 s waveform ~20 times in the 2026-08-29 run.
That makes "this exact waveform -> this exact voice" a memorisable lookup over
1,989 entries, and memorising it is enough to drive TRAINING separation to
5.51 dB while held-out separation falls to -0.17 dB. Validation speakers are
disjoint (`splits.yaml`), so the table is worth nothing there.

A bank of distinct enrollment UTTERANCES -- different sentences, different
recording sessions, each with its own EQ curve -- removes the fixed waveform to
memorise. Random cropping of the mixture does not do this: it resamples the same
scene, it does not vary the identity cue. decisions-m1.md 2026-08-30.

WHAT IS PRESERVED, DELIBERATELY
-------------------------------
Every guarantee the original enrollment carries is applied per variant, by
calling the SAME `render.render_enrollment` rather than a parallel copy:

  A4  dry, no room convolution
  B3  the manifest's own `enrollment_length_s`
  B8/B10  the book -> chapter -> utterance guard tier, via `pick_enrollment`
  level   each variant levelled to the trial's `target_loudness_lufs`, so which
          variant is in play cannot be read off the loudness
  EQ      the manifest's per-trial eq on/off flag, with a fresh curve per variant

**Variant 00 reproduces `enrollment.wav` exactly** -- same utterance, same
offset, same EQ seed. So `enrollment_variants: 1` and a bank of 1 are the same
data, which is what makes this a clean ablation arm rather than a confound.

Additive: mixtures, targets and interferers are untouched, so this does not
invalidate existing rendered audio, manifests or checkpoints.

COST
----
(K-1) x 2 x ~160 kB per trial. K=4 on sir0_train (1,989 trials) is ~1.9 GB,
about 45 % on top of the split. Raise K only if the Kaggle upload can carry it.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_manifest import pick_enrollment, read_books  # noqa: E402
from scripts.render_trials import (flac_paths, git_commit, read_index,  # noqa: E402
                                   read_speaker_subsets)
from src.data import render  # noqa: E402
from src.run_log import timed  # noqa: E402

import numpy as np  # noqa: E402
import pyloudnorm  # noqa: E402

# "" = the target's enrollment, "interferer_" = the other direction's. The two
# must be built identically or how the enrollment was made becomes a cue for
# which direction is being asked (render.py, render_enrollment).
PREFIXES = ("", "interferer_")
_ctx = {}


def candidate_pool(row, prefix, by_speaker, enroll_len):
    """Every utterance by the enrolled speaker long enough to be an enrollment.

    The enrolled speaker is NOT always the one in the mixture: on a phantom
    interferer trial (`target_only` / `noise_only`) a speaker who is genuinely
    absent is enrolled, and the right answer is silence.
    """
    speaker = (row["target_speaker"] if prefix == ""
               else row["interferer_enrollment_speaker"])
    if not speaker:
        return None, []
    return speaker, [u for u in by_speaker.get(speaker, [])
                     if float(u["duration"]) >= enroll_len]


def mixture_context(row, prefix):
    """(chapter_key, utterances_used_in_the_mixture) for the guard tiers.

    `None` chapter means this speaker contributes no audio to the mixture, so
    there is no content to leak from and no tier is being applied -- the phantom
    case, and the absent-target case.
    """
    if prefix == "":
        speaker, chapter = row["target_speaker"], row["target_chapter"]
        used = row["target_utts"]
    else:
        # A phantom's enrolled speaker is not the mixture's interferer, so its
        # own chapter is irrelevant: nothing of theirs is in the audio.
        if int(row.get("interferer_enrollment_phantom", 0)):
            return None, set()
        speaker, chapter = row["interferer_speaker"], row["interferer_chapter"]
        used = row["interferer_utts"]
    if not chapter:
        return None, set()
    return (speaker, chapter), {u for u in used.split("|") if u}


def draw_variants(row, prefix, by_speaker, book, n_variants):
    """The manifest's own enrollment first, then n-1 DISTINCT alternatives.

    Alternatives are drawn one at a time through `pick_enrollment`, with the
    already-drawn utterances removed from the pool each time, so the strongest
    guard tier available is re-evaluated at every draw and the bank degrades
    tier by tier rather than failing. A speaker with too few candidates gets a
    shorter bank; the caller records the shortfall rather than padding with
    repeats, because a silently repeated variant would weaken the augmentation
    exactly where the speaker is rarest.
    """
    enroll_len = float(row[f"{prefix}enrollment_length_s"])
    speaker, pool = candidate_pool(row, prefix, by_speaker, enroll_len)
    chapter, used = mixture_context(row, prefix)

    first = {"utt": row[f"{prefix}enrollment_utt"],
             "offset_s": float(row[f"{prefix}enrollment_offset_s"]),
             "tier": row[f"{prefix}enrollment_guard"],
             "duration": None}
    out = [first]
    taken = {first["utt"]}

    # Offsets and tier draws come from the trial id, not an ambient RNG, so the
    # bank is reproducible from the manifest alone and independent of worker
    # count or which subset of trials is being rendered -- the same rule
    # render.trial_seed() exists for.
    rng = np.random.default_rng(render.trial_seed(f"{row['trial_id']}#bank#{prefix}"))
    for _ in range(n_variants - 1):
        remaining = [u for u in pool if u["utt"] not in taken]
        if not remaining:
            break
        chosen = pick_enrollment(rng, remaining, chapter, used, book)
        if chosen is None:
            break
        tier, utt = chosen
        span = float(utt["duration"]) - enroll_len
        out.append({"utt": utt["utt"],
                    "offset_s": round(float(rng.uniform(0, max(span, 0.0))), 4),
                    "tier": tier,
                    "duration": float(utt["duration"])})
        taken.add(utt["utt"])
    return speaker, out


def render_variant(row, prefix, variant, k, cfg, flac_of, meter):
    """One variant's audio, through the production enrollment renderer.

    A synthetic row is handed to `render.render_enrollment` rather than
    duplicating its body: levelling, EQ and the dry (A4) guarantee then cannot
    drift between the bank and `enrollment.wav`.

    The trial id carries a `#v{k}` suffix for k>0 ONLY, because the EQ curve is
    seeded from it. k=0 keeps the bare id and so reproduces `enrollment.wav`
    byte for byte.
    """
    synthetic = dict(row)
    synthetic["trial_id"] = row["trial_id"] if k == 0 else f"{row['trial_id']}#v{k}"
    synthetic[f"{prefix}enrollment_utt"] = variant["utt"]
    synthetic[f"{prefix}enrollment_offset_s"] = variant["offset_s"]
    audio, bands = render.render_enrollment(
        synthetic, cfg, flac_of, meter, float(row["target_loudness_lufs"]),
        prefix=prefix)
    return audio, bands


def _init(cfg, flac_of, by_speaker, book, out_dir, n_variants, force):
    _ctx.update(cfg=cfg, flac_of=flac_of, by_speaker=by_speaker, book=book,
                out_dir=out_dir, n_variants=n_variants, force=force,
                meter=pyloudnorm.Meter(cfg["sample_rate"]))


def bank_path(trial_dir, prefix, k):
    return trial_dir / f"{prefix}enrollment_v{k:02d}.wav"


def _work(row):
    """Render one trial's bank. Returns (trial_id, status, note)."""
    trial_dir = _ctx["out_dir"] / row["trial_id"]
    n = _ctx["n_variants"]
    if not trial_dir.is_dir():
        return row["trial_id"], "failed", "trial not rendered -- run render_trials.py first"
    if not _ctx["force"] and (trial_dir / "enrollment_bank.json").exists():
        return row["trial_id"], "skipped", ""

    record = {"n_requested": n, "variants": {}}
    try:
        for prefix in PREFIXES:
            speaker, variants = draw_variants(
                row, prefix, _ctx["by_speaker"], _ctx["book"], n)
            entries = []
            for k, v in enumerate(variants):
                audio, bands = render_variant(row, prefix, v, k, _ctx["cfg"],
                                              _ctx["flac_of"], _ctx["meter"])
                sf.write(bank_path(trial_dir, prefix, k), audio,
                         _ctx["cfg"]["sample_rate"], subtype="PCM_16")
                entries.append({**v, "eq_bands": bands})
            record["variants"][prefix or "target"] = {
                "speaker": speaker, "n": len(entries), "entries": entries}
    except Exception as e:                                    # noqa: BLE001
        return row["trial_id"], "failed", f"{type(e).__name__}: {e}"

    # Written last: its presence is what marks the bank complete, so an
    # interrupted run re-does the trial instead of half-skipping it.
    (trial_dir / "enrollment_bank.json").write_text(json.dumps(record, indent=1))
    short = min(v["n"] for v in record["variants"].values())
    return row["trial_id"], "rendered", "" if short == n else f"only {short} variants"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, help="manifest name, e.g. sir0_train")
    ap.add_argument("--variants", type=int, default=4,
                    help="bank size K including the original. 1 is a no-op.")
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--manifest-dir", default="data/manifests")
    ap.add_argument("--out-dir", default="data/rendered")
    ap.add_argument("--limit", type=int, default=None,
                    help="first N trials only -- use this to TIME the job first")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.variants < 1:
        raise SystemExit("--variants must be >= 1")

    config = yaml.safe_load(Path(args.config).read_text())
    cfg = {"sample_rate": config["sample_rate"]}
    ls_root = Path(config["paths"]["librispeech"])

    manifest = Path(args.manifest_dir) / f"{args.split}.csv"
    with manifest.open() as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # The split's own utterance index: every utterance by every speaker in the
    # split, not only the ones the mixtures used. That surplus IS the bank.
    index = read_index(Path("data/index") / f"utterances_{args.split}.csv")
    flac_of = flac_paths(ls_root, index, read_speaker_subsets(ls_root))
    by_speaker = {}
    for utt, r in index.items():
        by_speaker.setdefault(r["speaker"], []).append(
            {"utt": utt, "speaker": r["speaker"], "chapter": r["chapter"],
             "duration": r["duration"]})
    for v in by_speaker.values():
        v.sort(key=lambda u: u["utt"])
    book = read_books(ls_root)

    out_dir = Path(args.out_dir) / args.split
    if not out_dir.is_dir():
        raise SystemExit(f"{out_dir} does not exist -- render the split first.")

    counts = {"rendered": 0, "skipped": 0, "failed": 0}
    failures, short_banks = [], []
    with timed(f"scripts/render_enrollment_bank.py --split {args.split} "
               f"--variants {args.variants}",
               scope=lambda: f"{counts['rendered']:,} trials x {args.variants} variants",
               rate=f"{args.workers} workers, 16 kHz PCM_16"):
        with Pool(args.workers, initializer=_init,
                  initargs=(cfg, flac_of, by_speaker, book, out_dir,
                            args.variants, args.force)) as pool:
            for i, (tid, status, note) in enumerate(
                    pool.imap_unordered(_work, rows, chunksize=4), 1):
                counts[status] += 1
                if status == "failed":
                    failures.append((tid, note))
                elif note:
                    short_banks.append((tid, note))
                if i % 25 == 0 or i == len(rows):
                    print(f"\r  {i}/{len(rows)}  rendered {counts['rendered']}  "
                          f"skipped {counts['skipped']}  failed {counts['failed']}",
                          end="", file=sys.stderr, flush=True)
    print(file=sys.stderr)

    (out_dir / "enrollment_bank.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/render_enrollment_bank.py",
        "split": args.split,
        "git_commit": git_commit(),
        "config": args.config,
        "variants": args.variants,
        "n_trials": len(rows),
        "n_rendered": counts["rendered"],
        "n_skipped": counts["skipped"],
        "n_failed": counts["failed"],
        "n_short": len(short_banks),
        "partial": bool(args.limit),
    }, sort_keys=False))

    if failures:
        for t, n in failures[:5]:
            print(f"  FAILED {t}  {n}", file=sys.stderr)
    if short_banks:
        # Not an error: a speaker with few long utterances legitimately supports
        # a smaller bank. Reported because it is a per-speaker weakening of the
        # augmentation and belongs in the write-up, not in a silent fallback.
        print(f"  {len(short_banks)} trial(s) got fewer than {args.variants} "
              f"variants, e.g. {short_banks[0][0]} ({short_banks[0][1]})",
              file=sys.stderr)

    print(f"Wrote banks under {out_dir}  ({counts['rendered']} rendered, "
          f"{counts['skipped']} skipped, {counts['failed']} failed)")
    if counts["failed"]:
        sys.exit(f"{counts['failed']} trial(s) failed")


if __name__ == "__main__":
    main()
