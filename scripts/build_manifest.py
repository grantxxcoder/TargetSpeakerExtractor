#!/usr/bin/env python3
"""Build the trial manifest for one split: one row per trial, every random
draw decided here. Reads file headers only, never audio samples.

    python scripts/build_manifest.py --split val

Transcripts are not copied into the manifest. They live in the utterance
index, keyed by utterance id.

B2 PR2 (decisions-m0.md 2026-08-15) split one quantity into two; confusing them
is how this file breaks:

  FOOTPRINT  timeline the audio occupies, silence included. Drives PLACEMENT.
  SPEECH     how much of that is detected voice (data/index/vad_segments.csv).
             Drives target_activity, interferer_activity, overlap_achieved,
             interrupted.

They differ ~14 % (LibriSpeech is ~86 % speech). Using footprint for both
overstated overlap by ~25 %.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import vad  # noqa: E402
from src.data.sampling import (  # noqa: E402
    draw, draw_regime, resolve, split_config,
)
from src.run_log import timed  # noqa: E402

COLUMNS = [
    "trial_id", "split",
    "target_speaker", "target_chapter", "target_utts", "target_onsets_s",
    # B2 PR2: *_speech_s is DETECTED SPEECH, *_footprint_s is how much timeline
    # the audio occupies. They differ by ~14 % and mean different things -- see
    # the module docstring. *_activity is speech / mixture_length_s.
    "target_speech_s", "target_footprint_s", "target_activity",
    "interferer_speaker", "interferer_chapter", "interferer_utts",
    "interferer_onsets_s", "interferer_speech_s", "interferer_footprint_s",
    "interferer_activity",
    "enrollment_utt", "enrollment_offset_s", "enrollment_length_s", "enrollment_eq",
    # B10: which of the three guard tiers supplied the enrollment clip.
    "enrollment_guard",
    # SECOND training direction. Every trial carries one; where nobody
    # interferes the enrolled speaker is a PHANTOM and the answer is silence. A
    # trial is a whole trial -- no special-casing a missing field downstream.
    # interferer_enrollment_speaker is deliberately separate from
    # interferer_speaker ("who interferes, empty if nobody"): overloading it
    # would change what every existing manifest means. decisions-m1.md 2026-08-26.
    "interferer_enrollment_speaker", "interferer_enrollment_utt",
    "interferer_enrollment_offset_s", "interferer_enrollment_length_s",
    "interferer_enrollment_eq", "interferer_enrollment_guard",
    "interferer_enrollment_phantom",
    "noise_clip", "noise_offset_s",
    "mixture_length_s", "sir_db", "snr_db", "target_loudness_lufs",
    "overlap_requested", "overlap_achieved",
    "t60_s", "room_l", "room_w", "room_h",
    "mic_x", "mic_y", "mic_z",
    "target_x", "target_y", "target_z",
    "interferer_x", "interferer_y", "interferer_z",
    # B9: which of the four trial types this is. `target_absent` is kept as a
    # derived convenience -- it is true for interferer_only and noise_only.
    "target_absent", "condition",
    "same_gender",
    # B13's interruption condition, derived from the onsets (decisions-m0.md
    # 2026-08-14).
    "interrupted",
    # B12: provenance of the bands this trial was drawn from, never a stratum.
    "regime",
]


def rngs(trial_id, n):
    """Independent RNG streams for one trial, derived from its id alone.

    blake2b rather than hash(): Python salts string hashing per process, so
    hash() would give different audio on every run.
    """
    digest = hashlib.blake2b(trial_id.encode(), digest_size=8).digest()
    seeds = np.random.SeedSequence(int.from_bytes(digest, "big")).spawn(n)
    return [np.random.default_rng(s) for s in seeds]


# --- corpus metadata ------------------------------------------------------

def read_speakers(root):
    """{speaker_id: (sex, subset)} from SPEAKERS.TXT."""
    out = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = (f[1], f[2])
    return out


def read_books(root):
    """{chapter_id: book_id} from CHAPTERS.TXT."""
    out = {}
    for line in (root / "CHAPTERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        f = [p.strip() for p in line.split("|")]
        out[f[0]] = f[5]
    return out


def index_utterances(root, speakers, subset_of, cache):
    """One row per LibriSpeech utterance: id, speaker, chapter, duration, text."""
    if cache.exists():
        with cache.open() as f:
            rows = list(csv.DictReader(f))
        # The cache is keyed only by split NAME, so a splits.yaml change (B10
        # redrew the eval pools) leaves a cache describing the previous set of
        # speakers. Reading it would silently build the manifest from the wrong
        # people and every disjointness check downstream would still pass,
        # because they all read this same file. Verify and rebuild instead.
        cached = {r["speaker"] for r in rows}
        if cached == set(speakers):
            return rows
        print(f"  {cache.name}: cached speakers do not match splits.yaml "
              f"({len(cached)} vs {len(set(speakers))}); rebuilding index",
              file=sys.stderr)

    rows = []
    for i, sid in enumerate(speakers, 1):
        print(f"\rindexing speakers {i}/{len(speakers)}", end="", file=sys.stderr)
        for chapter in sorted((root / subset_of[sid] / sid).iterdir()):
            trans = {}
            for line in (chapter / f"{sid}-{chapter.name}.trans.txt").read_text().splitlines():
                utt, text = line.split(" ", 1)
                trans[utt] = text
            for flac in sorted(chapter.glob("*.flac")):
                rows.append({
                    "utt": flac.stem,
                    "speaker": sid,
                    "chapter": chapter.name,
                    "duration": f"{sf.info(flac).duration:.3f}",
                    "text": trans[flac.stem],
                })
    print(file=sys.stderr)

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt", "speaker", "chapter", "duration", "text"])
        w.writeheader()
        w.writerows(rows)
    return rows


def load_vad_index(path, utterances):
    """`{utt: [(start_s, end_s), ...]}` for every utterance in this split.

    B2 PR2. Built once by `scripts/build_vad_index.py`; this only reads it. The
    index covers the whole corpus, so it is filtered down to the split's own
    utterances rather than held in full.

    Required, not optional. Silently falling back to file boundaries is exactly
    the ~25 % overlap overstatement B2 exists to remove, and it would leave no
    trace in the manifest.
    """
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Run scripts/build_vad_index.py first: overlap "
            "cannot be measured from speech without it.")

    wanted = {u["utt"] for u in utterances}
    segs = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            if r["utt"] in wanted:
                segs[r["utt"]] = vad.parse_segments(r["segments"])

    missing = wanted - segs.keys()
    if missing:
        raise SystemExit(
            f"{path} is stale: {len(missing)} utterances in this split were never "
            f"indexed (e.g. {sorted(missing)[0]}). Re-run scripts/build_vad_index.py.")
    return segs


def drop_silent(utterances, segs_of):
    """Utterances the detector found no speech in at all.

    Measured 2026-08-15: exactly 1 of 137,876. It is dropped rather than
    special-cased -- as a target it would contribute zero activity while still
    consuming window, and as an enrollment clip it would be 5 s of silence
    against B10/A1's >=5 s of voice.
    """
    kept = [u for u in utterances if segs_of[u["utt"]]]
    return kept, len(utterances) - len(kept)


def index_noise(root, noise_split, cache):
    """One row per WHAM! noise clip: file name and duration."""
    if cache.exists():
        with cache.open() as f:
            return list(csv.DictReader(f))

    rows = [{"clip": p.name, "duration": f"{sf.info(p).duration:.3f}"}
            for p in sorted((root / noise_split).glob("*.flac"))]
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "duration"])
        w.writeheader()
        w.writerows(rows)
    return rows


def reject_speech_clips(noise, screen_csv, max_speech_run_s):
    """Drop noise beds holding a run of speech long enough to become words.

    B2 `noise_speech_rejection`, decisions-m0.md 2026-08-16. WHAM! was recorded in
    real cafes and bars, so some beds contain audible background talkers. Their
    words reach the mixture but appear in no transcript, so the metric would
    score them as speech the model invented, and they add an unlabelled third
    talker to a task CLAUDE.md declares two-speaker.

    Rejection is by the longest UNBROKEN run of detected speech, not by the
    total: half a second in one piece can be a word, whereas the same half
    second spread over five 100 ms blips is a detector twitching at clatter.

    The screening report is scripts/screen_noise_speech.py's output. It is
    required, not optional -- a missing or stale report would silently let
    contaminated beds through, which is the exact failure this prevents.
    """
    if not screen_csv.exists():
        raise SystemExit(
            f"{screen_csv} not found. Run scripts/screen_noise_speech.py first: "
            "the noise pool cannot be filtered without it.")
    with screen_csv.open() as f:
        longest = {r["clip"]: float(r["max_segment_s"]) for r in csv.DictReader(f)}

    missing = [c["clip"] for c in noise if c["clip"] not in longest]
    if missing:
        raise SystemExit(
            f"{screen_csv} is stale: {len(missing)} clips in the pool were never "
            f"screened (e.g. {missing[0]}). Re-run scripts/screen_noise_speech.py.")

    kept = [c for c in noise if longest[c["clip"]] < max_speech_run_s]
    if not kept:
        raise SystemExit(f"{screen_csv}: every clip rejected at "
                         f"max_speech_run_s={max_speech_run_s}.")
    return kept


# --- sampling ------------------------------------------------------------

def sample_room(rng, cfg):
    """Room dimensions, T60 and mic position. Redraws until the T60 is
    physically reachable in the room (small T60 in a big room is not).

    Reachability is Sabine's equation, T60 = 0.161V / (S*alpha), for room volume
    V, total surface area S and average absorption alpha -- Sabine (1922),
    "Collected Papers on Acoustics". `pra.inverse_sabine` solves it for alpha and
    raises when the answer would exceed 1, i.e. when the wanted T60 is shorter
    than perfectly absorbing walls could achieve.

    The whole room is redrawn on failure, not just the T60. Holding the room and
    resampling T60 alone would sample from a range truncated by room size, so big
    rooms would systematically get longer reverb and the T60 distribution would
    skew upward. Verified uniform in section 4 of the manifest notebook."""
    for _ in range(50):
        dims = [draw(rng, cfg["room_length_m"]),
                draw(rng, cfg["room_width_m"]),
                draw(rng, cfg["room_height_m"])]
        t60 = draw(rng, cfg["t60_s"])
        try:
            pra.inverse_sabine(t60, dims)
        except ValueError:
            continue
        mic = [rng.uniform(0.35 * dims[0], 0.65 * dims[0]),
               rng.uniform(0.35 * dims[1], 0.65 * dims[1]),
               draw(rng, cfg["mic_height_m"])]
        return dims, t60, mic
    raise RuntimeError("no feasible room after 50 draws")


def place_source(rng, cfg, dims, mic):
    """A point at a sampled distance and random angle from the mic, kept at
    least 0.3 m off every wall."""
    for _ in range(50):
        d = draw(rng, cfg["source_distance_m"])
        az = rng.uniform(0, 2 * np.pi)
        x, y = mic[0] + d * np.cos(az), mic[1] + d * np.sin(az)
        if 0.3 < x < dims[0] - 0.3 and 0.3 < y < dims[1] - 0.3:
            return [x, y, draw(rng, cfg["source_height_m"])]
    raise RuntimeError("no feasible source position after 50 draws")


def pick_run(rng, chapters, by_chapter, wanted_s, cap_s, max_n, segs_of,
             footprint_cap_s):
    """A run of consecutive utterances from one chapter whose DETECTED SPEECH
    totals between wanted_s and cap_s. Consecutive and single-chapter so the
    joined audio is continuous reading and the joined transcript is exact.

    Returns `(chapter, run, footprint_s, speech_s)`.

    B2 PR2 changed the quantity being accumulated from file duration to speech.
    `footprint_cap_s` is new and is not the same limit: selecting on speech lets
    a run reach the wanted amount while its audio overruns the window, because
    the silence between and inside utterances is carried along with it. `lay_out`
    would then be handed a negative amount of leftover time and scatter negative
    gaps. Capping the footprint at the window length is what prevents that.
    """
    for _ in range(20):
        chapter = chapters[rng.integers(len(chapters))]
        utts = by_chapter[chapter]
        run, speech, footprint = [], 0.0, 0.0
        for u in utts[rng.integers(len(utts)):][:max_n]:
            s = vad.total_speech(segs_of[u["utt"]])
            d = float(u["duration"])
            if speech + s > cap_s or footprint + d > footprint_cap_s:
                break
            run.append(u)
            speech += s
            footprint += d
            if speech >= wanted_s:
                return chapter, run, footprint, speech
    return None


def lay_out(rng, run, span_s):
    """Onsets for a run of utterances inside span_s, with the leftover time
    scattered as gaps before, between and after them.

    Deliberately still FOOTPRINT-based after B2 PR2: an audio file occupies its
    whole duration on the timeline whether or not anyone is speaking during it.
    Placing by speech would overlap the actual audio.
    """
    durations = [float(u["duration"]) for u in run]
    gaps = rng.dirichlet(np.ones(len(run) + 1)) * (span_s - sum(durations))
    onsets, t = [], gaps[0]
    for d, gap in zip(durations, gaps[1:]):
        onsets.append(t)
        t += d + gap
    return onsets


def segments_for(run, segs_of):
    """The per-utterance speech segments of a run, in run order."""
    return [segs_of[u["utt"]] for u in run]


def block_spans(run, segs_of):
    """Speech spans of a contiguous run laid out from time zero.

    The interferer is placed as one unbroken block, so its internal geometry is
    fixed and `best_onset` only slides it. Computing it once per candidate onset
    instead would repeat the same work 400 times.
    """
    onsets, t = [], 0.0
    for u in run:
        onsets.append(t)
        t += float(u["duration"])
    return vad.spans_of(segments_for(run, segs_of), onsets)


def best_onset(target_spans, block, footprint_s, length, wanted_s):
    """Slide the interferer block across the window and take the onset whose
    speech overlap with the target comes closest to wanted_s. Searching beats
    solving it in closed form because both sides have gaps in them.

    B2 PR2: `block` is now the interferer's speech spans at time zero rather than
    one solid rectangle, so the gaps inside the interferer count as not-talking
    too. `footprint_s` still bounds the slide, because it is the audio that has
    to fit in the window.
    """
    candidates = np.linspace(0.0, max(0.0, length - footprint_s), 400)
    overlaps = [vad.shared_seconds(target_spans, vad.shift(block, c))
                for c in candidates]
    i = int(np.argmin([abs(o - wanted_s) for o in overlaps]))
    return float(candidates[i]), overlaps[i]


# B9's four trial types. Order is fixed: the cumulative walk below maps one
# uniform onto it, so reordering would silently change every trial's type.
#
#   both            target and interferer, overlapping           50 %
#   target_only     target speaks, nobody interrupts             25 %
#   interferer_only target silent, one other voice               20 %
#   noise_only      nobody speaks, noise bed only                 5 %
#
# The two zero-overlap types are deliberately equal in size (25 % present,
# 25 % absent), which puts P(target absent | no overlap) at exactly 0.50 so
# "did two voices overlap?" carries no information about absence.
# decisions-m0.md 2026-08-13 (B9).
CONDITIONS = ("both", "target_only", "interferer_only", "noise_only")


def draw_condition(rng, comp):
    """Which of B9's four types this trial is. Consumes exactly one uniform."""
    total = sum(float(comp[c]) for c in CONDITIONS)
    if not 0.999 <= total <= 1.001:
        raise ValueError(f"composition must sum to 1.0, got {total}: {comp}")
    u = rng.random() * total
    acc = 0.0
    for name in CONDITIONS:
        acc += float(comp[name])
        if u < acc:
            return name
    return CONDITIONS[-1]


def utt_index(utt_id):
    """The trailing sequence number of a LibriSpeech utterance id."""
    return int(utt_id.rsplit("-", 1)[1])


def pick_interferer_enrollment(rng, interferer, target, speakers, sex,
                               same_gender, by_speaker, book,
                               interferer_chapter, interferer_run, enroll_len):
    """The enrollment for the SECOND training direction, and whether it is a phantom.

    Returns (speaker, utterance, guard_tier, is_phantom) or None.

    WHY. Every mixture trains twice, asked for each speaker; an
    enrollment-ignoring model must answer both the same and so cannot fit both.
    Measured 2026-08-26: with one direction the model ignored the enrollment
    entirely (8 % output movement on a swap), and neither the loss schedule nor
    removing the loudness shortcut changed that. decisions-m1.md 2026-08-26.

    PHANTOM. On `target_only`/`noise_only` nobody interferes, so a speaker who is
    genuinely NOT in the audio is enrolled and the answer is silence -- the purest
    target-absent example there is. Skipping those trials instead would drop the
    absent rate ~28 % -> ~16 % and invalidate the w the loss was calibrated on.
    Drawn under the SAME same-gender constraint as a real interferer, or the
    enrolled speaker's sex would predict whether anyone is interfering.
    """
    phantom = interferer is None
    if phantom:
        pool = [s for s in speakers
                if s != target and (sex[s] == sex[target]) == same_gender]
        if not pool:
            return None
        speaker = str(rng.choice(pool))
    else:
        speaker = interferer

    candidates = [u for u in by_speaker[speaker]
                  if float(u["duration"]) >= enroll_len]
    if not candidates:
        return None

    # Same three-tier guard as the target's enrollment, via the same function:
    # the two directions must be constructed identically, or how the enrollment
    # was made becomes a cue for which direction is being asked.
    used = {u["utt"] for u in interferer_run}
    chosen = pick_enrollment(rng, candidates, interferer_chapter, used, book)
    if chosen is None:
        return None
    guard, utt = chosen
    return speaker, utt, guard, phantom


def pick_enrollment(rng, candidates, target_chapter, used_utts, book):
    """B10: the enrollment clip, and which guard tier supplied it.

    Falls back book -> chapter -> utterance, taking the first tier with any
    candidate, and records which one fired. B8 required `book` with no
    fallback, which meant a speaker who read only one book could never be a
    present target -- 60.2 % of `train` speakers, and an AUC 0.795 label leak
    (decisions-m0.md 2026-08-13, B10). Falling back keeps every speaker and makes
    the content-leak cost measurable instead of assumed.

    Returns (tier, utterance) or None if even the weakest tier is empty.
    """
    if target_chapter is None:
        # No mixture content exists to leak from, so no tier is being applied.
        # Record the strongest tier this speaker COULD support rather than a
        # sentinel: a value that appeared only on absent trials would itself
        # separate absent from present, which is the class of giveaway this
        # whole decision exists to remove.
        books = {book[u["chapter"]] for u in candidates}
        chapters = {u["chapter"] for u in candidates}
        tier = "book" if len(books) > 1 else "chapter" if len(chapters) > 1 else "utterance"
        return tier, candidates[rng.integers(len(candidates))]

    target_book = book[target_chapter[1]]
    by_tier = [
        ("book", [u for u in candidates if book[u["chapter"]] != target_book]),
        ("chapter", [u for u in candidates
                     if book[u["chapter"]] == target_book
                     and u["chapter"] != target_chapter[1]]),
        ("utterance", [u for u in candidates
                       if u["chapter"] == target_chapter[1]
                       and u["utt"] not in used_utts]),
    ]
    for tier, pool in by_tier:
        if not pool:
            continue
        if tier == "utterance":
            # Same chapter is the weakest case, so take the utterances
            # furthest in index from the ones the mixture used. LibriSpeech
            # numbers utterances sequentially within a chapter, so distance in
            # index is a free proxy for distance in the narrative (B10).
            used = [utt_index(u) for u in used_utts]
            pool = sorted(pool, key=lambda u: (-min(abs(utt_index(u["utt"]) - i)
                                                    for i in used), u["utt"]))
            return tier, pool[0]
        return tier, pool[rng.integers(len(pool))]
    return None


# B2 PR2 removed the local is_interrupted() and shared_seconds() in favour of
# src/data/vad.py's, which take detected speech instead of file boundaries. The
# interruption test itself is unchanged in form -- what changed is that both the
# target's spans and the interferer's onsets are now real speech. Option A
# (`first_only=True`): one onset per interferer utterance, the moment they begin
# that turn. decisions-m0.md 2026-08-15 Part 3.


def build_trial(trial_id, split, sampling_cfg, speakers, sex, book, by_speaker,
                by_chapter, chapters_of, noise, segs_of):
    # Dedicated streams: spawn(N) yields the same first children as spawn(N-1),
    # so each addition is purely additive -- the same seed still yields
    # byte-identical mixtures and only new columns appear. `reg` (5th, regime)
    # and `ienr` (6th, 2026-08-26, interferer enrollment). Drawing either from
    # `pick`/`aug` would shift every later draw and change the IDENTITY of every
    # trial in every existing manifest, so mid, sir0 and the anchors would stop
    # being comparable to anything built afterwards.
    pick, level, room, aug, reg, ienr = rngs(trial_id, 6)

    # One regime per trial, then every band comes from it. None when the split
    # declares no regimes (the eval splits), in which case nothing is consumed
    # from `reg` and cfg is the defaults unchanged.
    regime = draw_regime(reg, sampling_cfg)
    cfg = resolve(sampling_cfg, regime)

    condition = draw_condition(aug, cfg["composition"])
    has_target = condition in ("both", "target_only")
    has_interferer = condition in ("both", "interferer_only")
    absent = not has_target

    length = draw(level, cfg["mixture_length_s"])
    enroll_len = draw(level, cfg["enrollment_length_s"])
    same_gender = pick.random() < cfg["same_gender_fraction"]

    # target_activity_ratio now varies instead of sitting at 0.75 (B9). It is
    # the binding constraint on overlap: you cannot overlap more of the window
    # than you speak in, so the overlap band is clipped to it below rather than
    # left to fail the tolerance check and reject the trial.
    t_act_wanted = draw(level, cfg["target_activity_ratio"])
    o_lo, o_hi = cfg["overlap_ratio"]
    o_hi = min(o_hi, t_act_wanted)
    o_lo = min(o_lo, o_hi)
    overlap_requested = draw(level, [o_lo, o_hi]) if condition == "both" else 0.0
    max_n = cfg["max_utterances_per_source"]

    target_chapter, target_run = None, []
    target_speech, target_footprint = 0.0, 0.0
    target_spans, target_onsets = [], []
    interferer_chapter, interferer_run = None, []
    interferer_speech, interferer_footprint = 0.0, 0.0
    interferer_onsets, i0, overlap_achieved = [], 0.0, 0.0
    interferer_spans = []
    interferer = None
    enrollment, guard = None, ""
    i_enrol_spk, i_enrollment, i_guard, i_phantom = None, None, "", False

    for _ in range(20):
        target = str(pick.choice(speakers))

        if has_interferer:
            pool = [s for s in speakers
                    if s != target and (sex[s] == sex[target]) == same_gender]
            if not pool:
                continue
            interferer = str(pick.choice(pool))
        else:
            interferer = None

        long_enough = [u for u in by_speaker[target]
                       if float(u["duration"]) >= enroll_len]
        if not long_enough:
            continue

        if has_target:
            # t_wanted is now SPEECH seconds, not audio seconds. The footprint
            # cap is what keeps the audio inside the window (see pick_run).
            t_wanted = t_act_wanted * length
            found = pick_run(pick, chapters_of[target], by_chapter, t_wanted,
                             t_wanted + cfg["activity_tolerance"] * length, max_n,
                             segs_of, length)
            if found is None:
                continue
            target_chapter, target_run, target_footprint, target_speech = found
            target_onsets = lay_out(level, target_run, length)
            target_spans = vad.spans_of(segments_for(target_run, segs_of),
                                        target_onsets)
            used_utts = {u["utt"] for u in target_run}
        else:
            target_chapter, target_run = None, []
            target_speech, target_footprint = 0.0, 0.0
            target_onsets, target_spans, used_utts = [], [], set()

        # B10's three-tier guard, in place of B8's book-or-redraw.
        chosen = pick_enrollment(pick, long_enough, target_chapter, used_utts, book)
        if chosen is None:
            continue
        guard, enrollment = chosen

        if not has_interferer:
            interferer_chapter, interferer_run = None, []
            interferer_speech, interferer_footprint = 0.0, 0.0
            interferer_onsets, i0, overlap_achieved = [], 0.0, 0.0
            interferer_spans = []
            # Nobody interferes, so the second direction enrolls a PHANTOM: a
            # speaker who is genuinely absent from this audio. The correct
            # answer is silence, which makes it the cleanest possible
            # target-absent example.
            picked = pick_interferer_enrollment(
                ienr, None, target, speakers, sex, same_gender, by_speaker,
                book, None, [], enroll_len)
            if picked is None:
                continue
            i_enrol_spk, i_enrollment, i_guard, i_phantom = picked
            break

        if has_target:
            # The interferer must talk for at least the requested overlap, and
            # not so much that the two cannot avoid each other. Both bounds
            # follow from  max(0, t + i - 1) <= overlap <= min(t, i).
            t_act = target_speech / length
            lo, hi = overlap_requested, min(1.0, 1.0 - t_act + overlap_requested)
            if lo > hi:
                continue
            i_act = level.uniform(lo, hi)
        else:
            # The interferer's activity is drawn from the same distribution a
            # present trial would have produced. Pinning it made absent trials
            # identifiable without listening: their interferer always talked
            # 0.75-0.85 of the window, where present-trial interferers span
            # roughly 0.2-0.9. decisions-m0.md 2026-08-11. The shadow target
            # activity now comes from the same varying band as a real one (B9).
            shadow_t_act = draw(level, cfg["target_activity_ratio"])
            s_lo, s_hi = cfg["overlap_ratio"]
            s_hi = min(s_hi, shadow_t_act)
            s_lo = min(s_lo, s_hi)
            shadow_overlap = draw(level, [s_lo, s_hi])
            shadow_t_act += level.uniform(0.0, cfg["activity_tolerance"])
            i_act = level.uniform(shadow_overlap,
                                  min(1.0, 1.0 - shadow_t_act + shadow_overlap))

        i_chapters = [c for c in chapters_of[interferer]
                      if target_chapter is None or book[c[1]] != book[target_chapter[1]]]
        if not i_chapters:
            continue
        found = pick_run(pick, i_chapters, by_chapter, i_act * length,
                         min((i_act + cfg["activity_tolerance"]) * length, length),
                         max_n, segs_of, length)
        if found is None:
            continue
        interferer_chapter, interferer_run, interferer_footprint, interferer_speech = found

        # One contiguous block, so sliding it is the single free variable that
        # sets the overlap. The slide range is bounded by the FOOTPRINT -- that
        # is what has to fit -- while the overlap it achieves is measured from
        # the speech inside it.
        block = block_spans(interferer_run, segs_of)
        if not has_target:
            i0 = level.uniform(0, length - interferer_footprint)
            overlap_achieved = 0.0
        else:
            i0, shared = best_onset(target_spans, block, interferer_footprint,
                                    length, overlap_requested * length)
            overlap_achieved = shared / length
        if abs(overlap_achieved - overlap_requested) > cfg["overlap_tolerance"]:
            continue

        interferer_spans = vad.shift(block, i0)
        interferer_onsets, t = [], i0
        for u in interferer_run:
            interferer_onsets.append(t)
            t += float(u["duration"])

        picked = pick_interferer_enrollment(
            ienr, interferer, target, speakers, sex, same_gender, by_speaker,
            book, interferer_chapter, interferer_run, enroll_len)
        if picked is None:
            continue
        i_enrol_spk, i_enrollment, i_guard, i_phantom = picked
        break
    else:
        return None

    clip = noise[int(pick.integers(len(noise)))]
    noise_offset = pick.uniform(0, float(clip["duration"]))

    dims, t60, mic = sample_room(room, cfg)
    target_pos = place_source(room, cfg, dims, mic)
    # A position is drawn for the interferer even when it does not speak, for
    # the same reason an absent trial keeps a target position: a trial is a
    # whole trial, and the renderer should never have to special-case a
    # missing coordinate.
    interferer_pos = place_source(room, cfg, dims, mic)

    enroll_offset = pick.uniform(0, float(enrollment["duration"]) - enroll_len)
    i_enroll_offset = ienr.uniform(0, float(i_enrollment["duration"]) - enroll_len)

    assert interferer is None or target != interferer
    assert enrollment["speaker"] == target
    assert float(enrollment["duration"]) >= enroll_len >= 5.0
    assert target_onsets == sorted(target_onsets)
    assert all(o >= 0 for o in target_onsets + interferer_onsets)
    # B10 requires this explicitly: in the first two tiers it is automatic, in
    # the third it is the only thing separating enrollment from mixture.
    assert enrollment["utt"] not in {u["utt"] for u in target_run}
    # the same three guarantees for the second direction
    assert i_enrollment["speaker"] == i_enrol_spk
    assert float(i_enrollment["duration"]) >= enroll_len >= 5.0
    assert i_enrollment["utt"] not in {u["utt"] for u in interferer_run}
    # a phantom must not be the target, or "the other speaker" is the same
    # speaker and the two directions stop being different questions
    assert i_enrol_spk != target
    # phantom exactly when nobody interferes
    assert i_phantom == (interferer is None)
    if interferer_run:
        assert interferer_onsets[-1] + float(interferer_run[-1]["duration"]) <= length + 1e-6
    if target_chapter is not None:
        assert book[interferer_chapter[1]] != book[target_chapter[1]] \
            if interferer_chapter else True
        assert target_onsets[-1] + float(target_run[-1]["duration"]) <= length + 1e-6

    return {
        "trial_id": trial_id,
        "split": split,
        "target_speaker": target,
        "target_chapter": "" if not has_target else target_chapter[1],
        "target_utts": "|".join(u["utt"] for u in target_run),
        "target_onsets_s": "|".join(f"{o:.4f}" for o in target_onsets),
        "target_speech_s": round(target_speech, 3),
        "target_footprint_s": round(target_footprint, 3),
        "target_activity": round(target_speech / length, 4),
        "interferer_speaker": interferer or "",
        "interferer_chapter": interferer_chapter[1] if interferer_chapter else "",
        "interferer_utts": "|".join(u["utt"] for u in interferer_run),
        "interferer_onsets_s": "|".join(f"{o:.4f}" for o in interferer_onsets),
        "interferer_speech_s": round(interferer_speech, 3),
        "interferer_footprint_s": round(interferer_footprint, 3),
        "interferer_activity": round(interferer_speech / length, 4),
        "interferer_enrollment_speaker": i_enrol_spk,
        "interferer_enrollment_utt": i_enrollment["utt"],
        "interferer_enrollment_offset_s": round(i_enroll_offset, 4),
        "interferer_enrollment_length_s": round(enroll_len, 3),
        "interferer_enrollment_eq": int(ienr.random() < cfg["enrollment_eq_prob"]),
        "interferer_enrollment_guard": i_guard,
        "interferer_enrollment_phantom": int(i_phantom),
        "enrollment_utt": enrollment["utt"],
        "enrollment_offset_s": round(enroll_offset, 4),
        "enrollment_length_s": round(enroll_len, 3),
        "enrollment_eq": int(aug.random() < cfg["enrollment_eq_prob"]),
        "enrollment_guard": guard,
        "noise_clip": clip["clip"],
        "noise_offset_s": round(noise_offset, 4),
        "mixture_length_s": round(length, 3),
        # SIR is the target-to-interferer ratio, so it only exists when both
        # of them do -- not merely when the target does.
        "sir_db": round(draw(level, cfg["sir_db"]), 2) if condition == "both" else "",
        "snr_db": round(draw(level, cfg["snr_db"]), 2),
        "target_loudness_lufs": round(draw(level, cfg["target_loudness_lufs"]), 2),
        "overlap_requested": round(overlap_requested, 4),
        "overlap_achieved": round(overlap_achieved, 4),
        "t60_s": round(t60, 4),
        "room_l": round(dims[0], 3), "room_w": round(dims[1], 3), "room_h": round(dims[2], 3),
        "mic_x": round(mic[0], 3), "mic_y": round(mic[1], 3), "mic_z": round(mic[2], 3),
        "target_x": round(target_pos[0], 3), "target_y": round(target_pos[1], 3),
        "target_z": round(target_pos[2], 3),
        "interferer_x": round(interferer_pos[0], 3),
        "interferer_y": round(interferer_pos[1], 3),
        "interferer_z": round(interferer_pos[2], 3),
        "target_absent": int(absent),
        "condition": condition,
        "same_gender": "" if interferer is None else int(sex[interferer] == sex[target]),
        "interrupted": int(vad.is_interrupted(
            target_spans,
            vad.onsets_of(segments_for(interferer_run, segs_of), interferer_onsets,
                          first_only=True))),
        "regime": regime or "none",
    }


# --- entry point ---------------------------------------------------------

def git_commit():
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True,
                               text=True, check=True, timeout=10).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:
        return "UNKNOWN-not-a-git-checkout"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--config", default="experiments/configs/generator.yaml")
    ap.add_argument("--out-dir", default="data/manifests")
    args = ap.parse_args()

    config = yaml.safe_load(Path(args.config).read_text())
    config_md5 = hashlib.md5(Path(args.config).read_bytes()).hexdigest()

    if args.split not in config["splits"]:
        sys.exit(f"unknown split '{args.split}'. Known: {sorted(config['splits'])}")
    sampling_cfg, cfg = split_config(config, args.split)

    ls_root = Path(config["paths"]["librispeech"])
    splits = yaml.safe_load(Path(config["paths"]["splits"]).read_text())
    # `speakers_from` lets a new split borrow an existing split's speaker list.
    # splits.yaml is a GENERATED file, pinned before any data was made, and
    # hand-editing it silently redefines what "eval" means -- so a split that
    # only varies acoustics (sir0, which changes sir_db and nothing else) must
    # NOT need its own entry there. Speaker-disjointness is inherited from
    # whichever split is borrowed, so it cannot be broken by borrowing.
    speaker_split = cfg.get("speakers_from", args.split)
    if speaker_split not in splits:
        sys.exit(f"speakers_from '{speaker_split}' is not in "
                 f"{config['paths']['splits']}. Known: "
                 f"{sorted(k for k in splits if k != 'meta' and k != 'counts')}")
    speakers = [str(s) for s in splits[speaker_split]]
    if speaker_split != args.split:
        print(f"  speakers borrowed from '{speaker_split}' "
              f"({len(speakers)} speakers)", file=sys.stderr)

    meta = read_speakers(ls_root)
    sex = {s: meta[s][0] for s in speakers}
    book = read_books(ls_root)
    subset_of = {s: meta[s][1] for s in speakers}

    index_dir = Path("data/index")
    utterances = index_utterances(ls_root, speakers, subset_of,
                                 index_dir / f"utterances_{args.split}.csv")
    noise = index_noise(Path(config["paths"]["wham_noise"]), cfg["noise_split"],
                        index_dir / f"noise_{cfg['noise_split']}.csv")
    n_noise_screened = len(noise)
    max_speech_run_s = config["noise_speech_rejection"]["max_speech_run_s"]
    noise = reject_speech_clips(
        noise, index_dir / f"noise_speech_{cfg['noise_split']}.csv",
        max_speech_run_s)

    # B2 PR2: every activity and overlap figure below is measured against this.
    segs_of = load_vad_index(index_dir / "vad_segments.csv", utterances)
    vad_meta_path = index_dir / "vad_segments.meta.yaml"
    vad_meta = yaml.safe_load(vad_meta_path.read_text()) if vad_meta_path.exists() else {}
    utterances, n_silent = drop_silent(utterances, segs_of)
    if n_silent:
        print(f"  dropped {n_silent} utterance(s) with no detected speech",
              file=sys.stderr)

    by_speaker = defaultdict(list)
    by_chapter = defaultdict(list)
    for u in utterances:
        by_speaker[u["speaker"]].append(u)
        by_chapter[(u["speaker"], u["chapter"])].append(u)
    for key in by_chapter:
        by_chapter[key].sort(key=lambda u: u["utt"])
    chapters_of = defaultdict(list)
    for spk, chapter in sorted(by_chapter):
        chapters_of[spk].append((spk, chapter))

    rows, failed_ids = [], []
    with timed(f"scripts/build_manifest.py --split {args.split}",
               scope=lambda: f"{len(rows):,} trials",
               rate="headers only, no audio"):
        for i in range(cfg["n_trials"]):
            trial_id = f"{args.split}-{config['seed']}-{i:06d}"
            row = build_trial(trial_id, args.split, sampling_cfg, speakers, sex,
                              book, by_speaker, by_chapter, chapters_of, noise,
                              segs_of)
            if row is None:
                failed_ids.append(trial_id)
            else:
                rows.append(row)
    failed = len(failed_ids)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{args.split}.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)

    # Unsatisfiable trials are dropped, and the ones that fail are the hard ones
    # -- tight overlap, low interferer activity, speakers with little material.
    # Their absence therefore biases the split. Record which ids were lost so the
    # bias can be characterised instead of guessed at.
    failed_file = out_dir / f"{args.split}.failed.txt"
    if failed_ids:
        failed_file.write_text("\n".join(failed_ids) + "\n")
    elif failed_file.exists():
        failed_file.unlink()

    (out_dir / f"{args.split}.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/build_manifest.py",
        "split": args.split,
        "seed": config["seed"],
        "config": args.config,
        "config_md5": config_md5,
        "git_commit": git_commit(),
        "n_trials": len(rows),
        "n_requested": cfg["n_trials"],
        "n_failed": failed,
        "failed_ids_file": failed_file.name if failed_ids else None,
        # Which bands produced this split, recorded here because the config can
        # change under a manifest that is not itself in git.
        # Which noise beds this split could draw from, after B2's speech
        # rejection. Recorded because the cutoff is a config value and the
        # manifest is not in git. decisions-m0.md 2026-08-16.
        "noise_split": cfg["noise_split"],
        "noise_clips_screened": n_noise_screened,
        "noise_clips_kept": len(noise),
        "noise_max_speech_run_s": max_speech_run_s,
        # B2 PR2: which detector pass defined "overlap" for this manifest. The
        # vad: block is what these numbers mean; quoting an overlap figure
        # without it is quoting a number whose definition is unstated.
        "vad_settings": vad.vad_config(config),
        "vad_index_built": vad_meta.get("generated"),
        "vad_detector_version": vad_meta.get("detector_version"),
        "n_utterances_silent_dropped": n_silent,
        "regimes": sampling_cfg["regimes"],
        "regime_mix": {r: sum(1 for x in rows if x["regime"] == r)
                       for r in sorted({x["regime"] for x in rows})},
    }, sort_keys=False))

    print(f"Wrote {out}  ({len(rows)} trials, {failed} unsatisfiable)")
    dropped = n_noise_screened - len(noise)
    print(f"  noise pool     {len(noise)} clips  "
          f"({dropped} dropped for speech >= {max_speech_run_s}s)")
    absent = sum(r["target_absent"] for r in rows)
    present = [r for r in rows if not r["target_absent"]]
    paired = [r for r in rows if r["condition"] == "both"]
    for name in CONDITIONS:
        share = sum(1 for r in rows if r["condition"] == name) / len(rows)
        want = cfg["composition"][name]
        print(f"  condition {name:<16} {share:.3f}  (asked {want:.2f})")
    print(f"  target absent  {absent / len(rows):.3f}  "
          f"(asked {cfg['target_absent_fraction']:.2f})")
    for r in sorted({x["regime"] for x in rows}):
        share = sum(1 for x in rows if x["regime"] == r) / len(rows)
        print(f"  regime {r:<8} {share:.2f}")
    # same_gender and interruption only exist where there is an interferer to
    # compare against or be interrupted by.
    if paired:
        print(f"  same gender    {sum(r['same_gender'] for r in paired) / len(paired):.2f}"
              f"  (of the {len(paired)} paired trials)")
        print(f"  interrupted    {sum(r['interrupted'] for r in paired) / len(paired):.2f}"
              f"  (of the {len(paired)} paired trials)")
    # The shortcut B9 exists to close: given no overlap at all, how often is
    # the target actually absent? 0.50 is a coin flip and therefore useless.
    quiet = [r for r in rows if r["overlap_achieved"] == 0.0]
    if quiet:
        p = sum(r["target_absent"] for r in quiet) / len(quiet)
        print(f"  P(absent | no overlap)  {p:.3f}   over {len(quiet)} trials "
              f"(0.50 = no information)")
    for name in ("overlap_requested", "overlap_achieved", "target_activity",
                 "interferer_activity"):
        v = [r[name] for r in present]
        if v:
            print(f"  {name:<20} mean {np.mean(v):.2f}  "
                  f"min {min(v):.2f}  max {max(v):.2f}")
    n_utts = [len(r["target_utts"].split("|")) for r in present]
    if n_utts:
        print(f"  utterances per target  mean {np.mean(n_utts):.1f}  max {max(n_utts)}")


if __name__ == "__main__":
    main()
