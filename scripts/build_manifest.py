"""Build the trial manifest for one split: one row per trial, with every random draw decided here. It reads file headers, never audio.

    python scripts/build_manifest.py --split sir0_train

Two quantities must not be confused (B2, decisions-m0.md 2026-08-15). The footprint is how much of the timeline the audio occupies, silence included, and it decides placement.
The speech is how much of it is detected voice (data/index/vad_segments.csv), and it decides activity, overlap and interruption. LibriSpeech is about 86 % speech, so using the footprint for both overstated overlap by about 25 %.
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
import soundfile as sf
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import vad
from src.data.sampling import draw, draw_regime, resolve, split_config
from src.run_log import timed

CONFIG_PATH = "experiments/configs/generator.yaml"
MANIFEST_DIR = Path("data/manifests")
INDEX_DIR = Path("data/index")

COLUMNS = [
    "trial_id", "split",
    "target_speaker", "target_chapter", "target_utts", "target_onsets_s",
    # The speech columns are detected voice and the footprint columns are timeline taken, and activity is speech over the mixture length.
    "target_speech_s", "target_footprint_s", "target_activity",
    "interferer_speaker", "interferer_chapter", "interferer_utts",
    "interferer_onsets_s", "interferer_speech_s", "interferer_footprint_s",
    "interferer_activity",
    "enrollment_utt", "enrollment_offset_s", "enrollment_length_s", "enrollment_eq",
    # Which of B10's three guard tiers supplied the enrolment.
    "enrollment_guard",
    # The second training direction (decisions-m2.md 2026-08-26). Where nobody interferes, the enrolled speaker is a phantom and the answer is silence.
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
    # B9's trial type. target_absent is true for interferer_only and noise_only.
    "target_absent", "condition",
    "same_gender",
    # B13's interruption condition (decisions-m0.md 2026-08-14).
    "interrupted",
    # B12's regime, recorded as provenance and never used as a reporting stratum.
    "regime",
]

# B9's four trial types (decisions-m0.md 2026-08-13). The order is fixed, because one uniform draw walks through it.
# The two zero-overlap types are the same size, so "did two voices overlap?" says nothing about whether the target is absent.
CONDITIONS = ("both", "target_only", "interferer_only", "noise_only")


# this gives a trial its own independent random streams, seeded from its id. blake2b is used because Python's hash() changes between processes
def trial_rngs(trial_id, count):
    digest = hashlib.blake2b(trial_id.encode(), digest_size=8).digest()
    seeds = np.random.SeedSequence(int.from_bytes(digest, "big")).spawn(count)
    generators = []
    for seed in seeds:
        generators.append(np.random.default_rng(seed))
    return generators


# this splits one line of a LibriSpeech metadata file into its stripped fields
def split_fields(line):
    fields = []
    for part in line.split("|"):
        fields.append(part.strip())
    return fields


# this reads {speaker id: (sex, subset)} from LibriSpeech's SPEAKERS.TXT
def read_speakers(root):
    speakers = {}
    for line in (root / "SPEAKERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        fields = split_fields(line)
        speakers[fields[0]] = (fields[1], fields[2])
    return speakers


# this reads {chapter id: book id} from LibriSpeech's CHAPTERS.TXT
def read_books(root):
    books = {}
    for line in (root / "CHAPTERS.TXT").read_text(errors="replace").splitlines():
        if line.lstrip().startswith(";") or not line.strip():
            continue
        fields = split_fields(line)
        books[fields[0]] = fields[5]
    return books


# this indexes every utterance of the split's speakers (id, speaker, chapter, duration, text), or reads the cached index
def index_utterances(root, speakers, subset_of, cache):
    if cache.exists():
        with cache.open() as handle:
            return list(csv.DictReader(handle))
    rows = []
    for speaker in speakers:
        for chapter in sorted((root / subset_of[speaker] / speaker).iterdir()):
            transcripts = {}
            for line in (chapter / f"{speaker}-{chapter.name}.trans.txt").read_text().splitlines():
                utterance, text = line.split(" ", 1)
                transcripts[utterance] = text
            for flac in sorted(chapter.glob("*.flac")):
                rows.append({"utt": flac.stem, "speaker": speaker, "chapter": chapter.name, "duration": f"{sf.info(flac).duration:.3f}", "text": transcripts[flac.stem]})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["utt", "speaker", "chapter", "duration", "text"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


# this reads the speech segments of the split's utterances from the VAD index that scripts/build_vad_index.py built for the whole corpus
def load_vad_index(path, utterances):
    wanted = set()
    for utterance in utterances:
        wanted.add(utterance["utt"])
    segments_of = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["utt"] in wanted:
                segments_of[row["utt"]] = vad.parse_segments(row["segments"])
    return segments_of


# this drops utterances with no detected speech at all, which was 1 of 137,876 (decisions-m0.md 2026-08-15)
def drop_silent(utterances, segments_of):
    kept = []
    for utterance in utterances:
        if segments_of[utterance["utt"]]:
            kept.append(utterance)
    return kept, len(utterances) - len(kept)


# this indexes the WHAM! noise clips of one pool (name and duration), or reads the cached index
def index_noise(root, noise_split, cache):
    if cache.exists():
        with cache.open() as handle:
            return list(csv.DictReader(handle))
    rows = []
    for path in sorted((root / noise_split).glob("*.flac")):
        rows.append({"clip": path.name, "duration": f"{sf.info(path).duration:.3f}"})
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["clip", "duration"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


# this drops noise beds whose longest unbroken run of speech could carry words (decisions-m0.md 2026-08-16)
def reject_speech_clips(noise, screen_csv, max_speech_run_s):
    """Background talkers in a cafe bed would put words into the mixture that no transcript holds, and add a third talker to a two-speaker task.

    The longest run is used rather than the total, because half a second in one piece can be a word, while five 100 ms blips are clatter.
    """
    longest_run = {}
    with screen_csv.open() as handle:
        for row in csv.DictReader(handle):
            longest_run[row["clip"]] = float(row["max_segment_s"])
    kept = []
    for clip in noise:
        if longest_run[clip["clip"]] < max_speech_run_s:
            kept.append(clip)
    return kept


# this draws the room size, T60 and mic position, redrawing the whole room until Sabine's equation can reach the T60 in it
def sample_room(rng, settings):
    """Sabine's equation, T60 = 0.161 V / (S alpha), from Sabine (1922). pra.inverse_sabine raises when alpha would exceed 1.

    The whole room is redrawn, not just the T60, or big rooms would systematically get longer reverb.
    """
    for attempt in range(50):
        dims = [draw(rng, settings["room_length_m"]), draw(rng, settings["room_width_m"]), draw(rng, settings["room_height_m"])]
        t60 = draw(rng, settings["t60_s"])
        try:
            pra.inverse_sabine(t60, dims)
        except ValueError:
            continue
        mic = [rng.uniform(0.35 * dims[0], 0.65 * dims[0]), rng.uniform(0.35 * dims[1], 0.65 * dims[1]), draw(rng, settings["mic_height_m"])]
        return dims, t60, mic
    raise RuntimeError("No feasible room after 50 draws.")


# this places a source at a sampled distance and random angle from the mic, at least 0.3 m from every wall
def place_source(rng, settings, dims, mic):
    for attempt in range(50):
        distance = draw(rng, settings["source_distance_m"])
        angle = rng.uniform(0, 2 * np.pi)
        x, y = mic[0] + distance * np.cos(angle), mic[1] + distance * np.sin(angle)
        if 0.3 < x < dims[0] - 0.3 and 0.3 < y < dims[1] - 0.3:
            return [x, y, draw(rng, settings["source_height_m"])]
    raise RuntimeError("No feasible source position after 50 draws.")


# this picks a run of consecutive utterances from one chapter whose detected speech lands between wanted_s and cap_s
def pick_run(rng, chapters, by_chapter, wanted_s, cap_s, max_utterances, segments_of, footprint_cap_s):
    """Consecutive and from one chapter, so the audio is continuous reading and the joined transcript is exact. It returns (chapter, run, footprint seconds, speech seconds), or None.

    The footprint is capped at the window too, because selecting on speech alone lets the silence carried along overrun the window.
    """
    for attempt in range(20):
        chapter = chapters[rng.integers(len(chapters))]
        utterances = by_chapter[chapter]
        run = []
        speech = 0.0
        footprint = 0.0
        for utterance in utterances[rng.integers(len(utterances)):][:max_utterances]:
            utterance_speech = vad.total_speech(segments_of[utterance["utt"]])
            duration = float(utterance["duration"])
            if speech + utterance_speech > cap_s or footprint + duration > footprint_cap_s:
                break
            run.append(utterance)
            speech += utterance_speech
            footprint += duration
            if speech >= wanted_s:
                return chapter, run, footprint, speech
    return None


# this gives onsets for a run inside the window, with the leftover time scattered as gaps before, between and after the utterances
def lay_out(rng, run, span_s):
    """Placement uses the footprint, because a file occupies its whole duration whether or not anyone is speaking."""
    durations = []
    for utterance in run:
        durations.append(float(utterance["duration"]))
    gaps = rng.dirichlet(np.ones(len(run) + 1)) * (span_s - sum(durations))
    onsets = []
    time = gaps[0]
    for duration, gap in zip(durations, gaps[1:]):
        onsets.append(time)
        time += duration + gap
    return onsets


# this gives the speech segments of each utterance in a run, in run order
def segments_for(run, segments_of):
    segment_lists = []
    for utterance in run:
        segment_lists.append(segments_of[utterance["utt"]])
    return segment_lists


# this gives the speech spans of a run laid end to end from time zero, so the interferer block can be slid as one piece
def block_spans(run, segments_of):
    onsets = []
    time = 0.0
    for utterance in run:
        onsets.append(time)
        time += float(utterance["duration"])
    return vad.spans_of(segments_for(run, segments_of), onsets)


# this slides the interferer block across the window and keeps the onset whose speech overlap with the target is closest to wanted_s
def best_onset(target_spans, block, footprint_s, length, wanted_s):
    """A search, because both sides have gaps in them. The footprint bounds the slide, because the audio is what has to fit."""
    candidates = np.linspace(0.0, max(0.0, length - footprint_s), 400)
    overlaps = []
    for candidate in candidates:
        overlaps.append(vad.shared_seconds(target_spans, vad.shift(block, candidate)))
    distances = []
    for overlap in overlaps:
        distances.append(abs(overlap - wanted_s))
    best = int(np.argmin(distances))
    return float(candidates[best]), overlaps[best]


# this draws which of B9's four types the trial is, from exactly one uniform draw
def draw_condition(rng, composition):
    shares = []
    for name in CONDITIONS:
        shares.append(float(composition[name]))
    draw_point = rng.random() * sum(shares)
    cumulative = 0.0
    for name in CONDITIONS:
        cumulative += float(composition[name])
        if draw_point < cumulative:
            return name
    return CONDITIONS[-1]


# this gives the trailing sequence number of a LibriSpeech utterance id
def utt_index(utterance_id):
    return int(utterance_id.rsplit("-", 1)[1])


# this gives how far in index an utterance is from the nearest one the mixture used
def distance_from_used(utterance, used_indices):
    distances = []
    for used_index in used_indices:
        distances.append(abs(utt_index(utterance["utt"]) - used_index))
    return min(distances)


# this picks the enrolment for the second training direction, and whether its speaker is a phantom
def pick_interferer_enrollment(rng, interferer, target, speakers, sex, same_gender, by_speaker, book, interferer_chapter, interferer_run, enrollment_length):
    """Each mixture trains twice, asked for each speaker, so a model that ignores the enrolment cannot fit both (decisions-m2.md 2026-08-26).

    Where nobody interferes, a speaker who is not in the audio is enrolled and the answer is silence. The phantom is drawn under the same same-gender rule as a real interferer, or its sex would give the condition away.
    It returns (speaker, utterance, guard tier, is phantom), or None.
    """
    phantom = interferer is None
    if phantom:
        pool = []
        for speaker in speakers:
            if speaker != target and (sex[speaker] == sex[target]) == same_gender:
                pool.append(speaker)
        if not pool:
            return None
        enrolled_speaker = str(rng.choice(pool))
    else:
        enrolled_speaker = interferer

    candidates = []
    for utterance in by_speaker[enrolled_speaker]:
        if float(utterance["duration"]) >= enrollment_length:
            candidates.append(utterance)
    if not candidates:
        return None

    # The same guard as the target's enrolment, through the same function, so the two directions are built identically.
    used = set()
    for utterance in interferer_run:
        used.add(utterance["utt"])
    chosen = pick_enrollment(rng, candidates, interferer_chapter, used, book)
    if chosen is None:
        return None
    guard, utterance = chosen
    return enrolled_speaker, utterance, guard, phantom


# this picks the enrolment clip under B10's guard, falling back from another book to another chapter to another utterance, and records the tier used
def pick_enrollment(rng, candidates, target_chapter, used_utterances, book):
    """B8 demanded another book with no fallback, which shut out 60.2 % of train speakers as targets and leaked the label (decisions-m0.md 2026-08-13, B10). It returns (tier, utterance), or None."""
    if target_chapter is None:
        # No mixture content exists to leak from. The strongest tier this speaker could support is recorded, so the tier never marks absent trials.
        books_read = set()
        chapters_read = set()
        for utterance in candidates:
            books_read.add(book[utterance["chapter"]])
            chapters_read.add(utterance["chapter"])
        if len(books_read) > 1:
            tier = "book"
        elif len(chapters_read) > 1:
            tier = "chapter"
        else:
            tier = "utterance"
        return tier, candidates[rng.integers(len(candidates))]

    target_book = book[target_chapter[1]]
    other_book = []
    same_book_other_chapter = []
    same_chapter = []
    for utterance in candidates:
        if book[utterance["chapter"]] != target_book:
            other_book.append(utterance)
        if book[utterance["chapter"]] == target_book and utterance["chapter"] != target_chapter[1]:
            same_book_other_chapter.append(utterance)
        if utterance["chapter"] == target_chapter[1] and utterance["utt"] not in used_utterances:
            same_chapter.append(utterance)
    if other_book:
        return "book", other_book[rng.integers(len(other_book))]
    if same_book_other_chapter:
        return "chapter", same_book_other_chapter[rng.integers(len(same_book_other_chapter))]
    if same_chapter:
        # The weakest tier, so take the utterance furthest in index from the ones the mixture used, which is a free proxy for distance in the story.
        used_indices = []
        for utterance_id in used_utterances:
            used_indices.append(utt_index(utterance_id))
        ranked = sorted(same_chapter, key=lambda utterance: (-distance_from_used(utterance, used_indices), utterance["utt"]))
        return "utterance", ranked[0]
    return None


# this draws one whole trial, or returns None when 20 attempts cannot satisfy it
def build_trial(trial_id, split, settings, regimes, speakers, sex, book, by_speaker, by_chapter, chapters_of, noise, segments_of):
    # Each concern has its own stream. Streams were only ever appended, because spawn(N) gives the same first children as spawn(N - 1), so older draws never shift.
    pick_rng, level_rng, room_rng, augment_rng, regime_rng, interferer_enrollment_rng = trial_rngs(trial_id, 6)

    regime = draw_regime(regime_rng, regimes)
    params = resolve(settings, regimes, regime)

    condition = draw_condition(augment_rng, params["composition"])
    has_target = condition in ("both", "target_only")
    has_interferer = condition in ("both", "interferer_only")

    length = draw(level_rng, params["mixture_length_s"])
    enrollment_length = draw(level_rng, params["enrollment_length_s"])
    same_gender = pick_rng.random() < params["same_gender_fraction"]

    # Overlap cannot exceed the target's activity, so its range is clipped to the activity drawn (B9).
    wanted_target_activity = draw(level_rng, params["target_activity_ratio"])
    overlap_low, overlap_high = params["overlap_ratio"]
    overlap_high = min(overlap_high, wanted_target_activity)
    overlap_low = min(overlap_low, overlap_high)
    if condition == "both":
        overlap_requested = draw(level_rng, [overlap_low, overlap_high])
    else:
        overlap_requested = 0.0
    max_utterances = params["max_utterances_per_source"]

    target_chapter, target_run = None, []
    target_speech, target_footprint = 0.0, 0.0
    target_spans, target_onsets = [], []
    interferer_chapter, interferer_run = None, []
    interferer_speech, interferer_footprint = 0.0, 0.0
    interferer_onsets, interferer_start, overlap_achieved = [], 0.0, 0.0
    interferer = None
    enrollment, guard = None, ""
    interferer_enrollment_speaker, interferer_enrollment, interferer_guard, interferer_phantom = None, None, "", False

    for attempt in range(20):
        target = str(pick_rng.choice(speakers))

        if has_interferer:
            pool = []
            for speaker in speakers:
                if speaker != target and (sex[speaker] == sex[target]) == same_gender:
                    pool.append(speaker)
            if not pool:
                continue
            interferer = str(pick_rng.choice(pool))
        else:
            interferer = None

        long_enough = []
        for utterance in by_speaker[target]:
            if float(utterance["duration"]) >= enrollment_length:
                long_enough.append(utterance)
        if not long_enough:
            continue

        if has_target:
            # The wanted amount is seconds of speech, and the footprint cap keeps the audio inside the window.
            wanted_speech = wanted_target_activity * length
            found = pick_run(pick_rng, chapters_of[target], by_chapter, wanted_speech, wanted_speech + params["activity_tolerance"] * length, max_utterances, segments_of, length)
            if found is None:
                continue
            target_chapter, target_run, target_footprint, target_speech = found
            target_onsets = lay_out(level_rng, target_run, length)
            target_spans = vad.spans_of(segments_for(target_run, segments_of), target_onsets)
            used_utterances = set()
            for utterance in target_run:
                used_utterances.add(utterance["utt"])
        else:
            target_chapter, target_run = None, []
            target_speech, target_footprint = 0.0, 0.0
            target_onsets, target_spans, used_utterances = [], [], set()

        chosen = pick_enrollment(pick_rng, long_enough, target_chapter, used_utterances, book)
        if chosen is None:
            continue
        guard, enrollment = chosen

        if not has_interferer:
            interferer_chapter, interferer_run = None, []
            interferer_speech, interferer_footprint = 0.0, 0.0
            interferer_onsets, interferer_start, overlap_achieved = [], 0.0, 0.0
            # Nobody interferes, so the second direction enrols a phantom who is absent from this audio.
            picked = pick_interferer_enrollment(interferer_enrollment_rng, None, target, speakers, sex, same_gender, by_speaker, book, None, [], enrollment_length)
            if picked is None:
                continue
            interferer_enrollment_speaker, interferer_enrollment, interferer_guard, interferer_phantom = picked
            break

        if has_target:
            # The interferer talks at least the requested overlap, and little enough that the two can avoid each other: max(0, t + i - 1) <= overlap <= min(t, i).
            target_activity = target_speech / length
            activity_low, activity_high = overlap_requested, min(1.0, 1.0 - target_activity + overlap_requested)
            if activity_low > activity_high:
                continue
            interferer_activity = level_rng.uniform(activity_low, activity_high)
        else:
            # The interferer's activity comes from the distribution a present trial would have produced, or absent trials could be spotted without listening (decisions-m0.md 2026-08-11).
            shadow_target_activity = draw(level_rng, params["target_activity_ratio"])
            shadow_low, shadow_high = params["overlap_ratio"]
            shadow_high = min(shadow_high, shadow_target_activity)
            shadow_low = min(shadow_low, shadow_high)
            shadow_overlap = draw(level_rng, [shadow_low, shadow_high])
            shadow_target_activity += level_rng.uniform(0.0, params["activity_tolerance"])
            interferer_activity = level_rng.uniform(shadow_overlap, min(1.0, 1.0 - shadow_target_activity + shadow_overlap))

        interferer_chapters = []
        for chapter in chapters_of[interferer]:
            if target_chapter is None or book[chapter[1]] != book[target_chapter[1]]:
                interferer_chapters.append(chapter)
        if not interferer_chapters:
            continue
        found = pick_run(pick_rng, interferer_chapters, by_chapter, interferer_activity * length, min((interferer_activity + params["activity_tolerance"]) * length, length), max_utterances, segments_of, length)
        if found is None:
            continue
        interferer_chapter, interferer_run, interferer_footprint, interferer_speech = found

        # The interferer is one contiguous block, so its onset is the one free variable that sets the overlap.
        block = block_spans(interferer_run, segments_of)
        if not has_target:
            interferer_start = level_rng.uniform(0, length - interferer_footprint)
            overlap_achieved = 0.0
        else:
            interferer_start, shared = best_onset(target_spans, block, interferer_footprint, length, overlap_requested * length)
            overlap_achieved = shared / length
        if abs(overlap_achieved - overlap_requested) > params["overlap_tolerance"]:
            continue

        interferer_onsets = []
        time = interferer_start
        for utterance in interferer_run:
            interferer_onsets.append(time)
            time += float(utterance["duration"])

        picked = pick_interferer_enrollment(interferer_enrollment_rng, interferer, target, speakers, sex, same_gender, by_speaker, book, interferer_chapter, interferer_run, enrollment_length)
        if picked is None:
            continue
        interferer_enrollment_speaker, interferer_enrollment, interferer_guard, interferer_phantom = picked
        break
    else:
        # Every attempt failed.
        return None

    clip = noise[int(pick_rng.integers(len(noise)))]
    noise_offset = pick_rng.uniform(0, float(clip["duration"]))

    dims, t60, mic = sample_room(room_rng, params)
    target_position = place_source(room_rng, params, dims, mic)
    # The interferer gets a position even when silent, so the renderer never has to handle a missing coordinate.
    interferer_position = place_source(room_rng, params, dims, mic)

    enrollment_offset = pick_rng.uniform(0, float(enrollment["duration"]) - enrollment_length)
    interferer_enrollment_offset = interferer_enrollment_rng.uniform(0, float(interferer_enrollment["duration"]) - enrollment_length)

    # B10 requires that neither enrolment is an utterance its speaker says in the mixture, and the enrolled other speaker is never the target.
    for utterance in target_run:
        if utterance["utt"] == enrollment["utt"]:
            raise RuntimeError(f"{trial_id}: the target's enrolment is part of the mixture.")
    for utterance in interferer_run:
        if utterance["utt"] == interferer_enrollment["utt"]:
            raise RuntimeError(f"{trial_id}: the interferer's enrolment is part of the mixture.")
    if interferer_enrollment_speaker == target:
        raise RuntimeError(f"{trial_id}: the second direction enrols the target.")

    # These draws keep the order they had when the row was built in one expression.
    interferer_enrollment_eq = int(interferer_enrollment_rng.random() < params["enrollment_eq_prob"])
    enrollment_eq = int(augment_rng.random() < params["enrollment_eq_prob"])
    # SIR is the target-to-interferer ratio, so it only exists when both speak.
    if condition == "both":
        sir_db = round(draw(level_rng, params["sir_db"]), 2)
    else:
        sir_db = ""
    snr_db = round(draw(level_rng, params["snr_db"]), 2)
    target_loudness_lufs = round(draw(level_rng, params["target_loudness_lufs"]), 2)

    target_utterance_ids = []
    for utterance in target_run:
        target_utterance_ids.append(utterance["utt"])
    target_onset_texts = []
    for onset in target_onsets:
        target_onset_texts.append(f"{onset:.4f}")
    interferer_utterance_ids = []
    for utterance in interferer_run:
        interferer_utterance_ids.append(utterance["utt"])
    interferer_onset_texts = []
    for onset in interferer_onsets:
        interferer_onset_texts.append(f"{onset:.4f}")

    if interferer is None:
        same_gender_value = ""
    else:
        same_gender_value = int(sex[interferer] == sex[target])
    interrupted = vad.is_interrupted(target_spans, vad.onsets_of(segments_for(interferer_run, segments_of), interferer_onsets))

    return {
        "trial_id": trial_id,
        "split": split,
        "target_speaker": target,
        "target_chapter": target_chapter[1] if has_target else "",
        "target_utts": "|".join(target_utterance_ids),
        "target_onsets_s": "|".join(target_onset_texts),
        "target_speech_s": round(target_speech, 3),
        "target_footprint_s": round(target_footprint, 3),
        "target_activity": round(target_speech / length, 4),
        "interferer_speaker": interferer or "",
        "interferer_chapter": interferer_chapter[1] if interferer_chapter else "",
        "interferer_utts": "|".join(interferer_utterance_ids),
        "interferer_onsets_s": "|".join(interferer_onset_texts),
        "interferer_speech_s": round(interferer_speech, 3),
        "interferer_footprint_s": round(interferer_footprint, 3),
        "interferer_activity": round(interferer_speech / length, 4),
        "interferer_enrollment_speaker": interferer_enrollment_speaker,
        "interferer_enrollment_utt": interferer_enrollment["utt"],
        "interferer_enrollment_offset_s": round(interferer_enrollment_offset, 4),
        "interferer_enrollment_length_s": round(enrollment_length, 3),
        "interferer_enrollment_eq": interferer_enrollment_eq,
        "interferer_enrollment_guard": interferer_guard,
        "interferer_enrollment_phantom": int(interferer_phantom),
        "enrollment_utt": enrollment["utt"],
        "enrollment_offset_s": round(enrollment_offset, 4),
        "enrollment_length_s": round(enrollment_length, 3),
        "enrollment_eq": enrollment_eq,
        "enrollment_guard": guard,
        "noise_clip": clip["clip"],
        "noise_offset_s": round(noise_offset, 4),
        "mixture_length_s": round(length, 3),
        "sir_db": sir_db,
        "snr_db": snr_db,
        "target_loudness_lufs": target_loudness_lufs,
        "overlap_requested": round(overlap_requested, 4),
        "overlap_achieved": round(overlap_achieved, 4),
        "t60_s": round(t60, 4),
        "room_l": round(dims[0], 3),
        "room_w": round(dims[1], 3),
        "room_h": round(dims[2], 3),
        "mic_x": round(mic[0], 3),
        "mic_y": round(mic[1], 3),
        "mic_z": round(mic[2], 3),
        "target_x": round(target_position[0], 3),
        "target_y": round(target_position[1], 3),
        "target_z": round(target_position[2], 3),
        "interferer_x": round(interferer_position[0], 3),
        "interferer_y": round(interferer_position[1], 3),
        "interferer_z": round(interferer_position[2], 3),
        "target_absent": int(not has_target),
        "condition": condition,
        "same_gender": same_gender_value,
        "interrupted": int(interrupted),
        "regime": regime or "none",
    }


# Generated by Claude Code
def print_summary(rows, failed, noise_kept, noise_dropped, max_speech_run_s, composition):
    print(f"Wrote {len(rows)} trials, {failed} unsatisfiable")
    print(f"  noise pool     {noise_kept} clips  ({noise_dropped} dropped for speech >= {max_speech_run_s}s)")
    for name in CONDITIONS:
        count = 0
        for row in rows:
            if row["condition"] == name:
                count += 1
        print(f"  condition {name:<16} {count / len(rows):.3f}  (asked {composition[name]:.2f})")
    absent_count = 0
    for row in rows:
        absent_count += row["target_absent"]
    print(f"  target absent  {absent_count / len(rows):.3f}  (asked {composition['interferer_only'] + composition['noise_only']:.2f})")
    quiet = []
    for row in rows:
        if row["overlap_achieved"] == 0.0:
            quiet.append(row["target_absent"])
    if quiet:
        print(f"  P(absent | no overlap)  {sum(quiet) / len(quiet):.3f}   over {len(quiet)} trials (0.50 = no information)")


# this is where the manifest build starts
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", required=True)
    args = parser.parse_args()

    config = yaml.safe_load(Path(CONFIG_PATH).read_text())
    config_md5 = hashlib.md5(Path(CONFIG_PATH).read_bytes()).hexdigest()
    settings, regimes = split_config(config, args.split)

    librispeech_root = Path(config["paths"]["librispeech"])
    splits = yaml.safe_load(Path(config["paths"]["splits"]).read_text())
    # A split that only changes acoustics, such as sir0, borrows another split's speakers, so the pinned splits.yaml never needs a hand edit.
    speaker_split = settings.get("speakers_from", args.split)
    speakers = []
    for speaker in splits[speaker_split]:
        speakers.append(str(speaker))

    speaker_metadata = read_speakers(librispeech_root)
    sex = {}
    subset_of = {}
    for speaker in speakers:
        sex[speaker] = speaker_metadata[speaker][0]
        subset_of[speaker] = speaker_metadata[speaker][1]
    book = read_books(librispeech_root)

    utterances = index_utterances(librispeech_root, speakers, subset_of, INDEX_DIR / f"utterances_{args.split}.csv")
    noise = index_noise(Path(config["paths"]["wham_noise"]), settings["noise_split"], INDEX_DIR / f"noise_{settings['noise_split']}.csv")
    noise_screened = len(noise)
    max_speech_run_s = config["noise_speech_rejection"]["max_speech_run_s"]
    noise = reject_speech_clips(noise, INDEX_DIR / f"noise_speech_{settings['noise_split']}.csv", max_speech_run_s)

    # Every activity and overlap figure is measured against the detected speech.
    segments_of = load_vad_index(INDEX_DIR / "vad_segments.csv", utterances)
    vad_meta = yaml.safe_load((INDEX_DIR / "vad_segments.meta.yaml").read_text())
    utterances, silent_dropped = drop_silent(utterances, segments_of)

    by_speaker = defaultdict(list)
    by_chapter = defaultdict(list)
    for utterance in utterances:
        by_speaker[utterance["speaker"]].append(utterance)
        by_chapter[(utterance["speaker"], utterance["chapter"])].append(utterance)
    for key in by_chapter:
        by_chapter[key].sort(key=lambda utterance: utterance["utt"])
    chapters_of = defaultdict(list)
    for speaker, chapter in sorted(by_chapter):
        chapters_of[speaker].append((speaker, chapter))

    rows = []
    failed_ids = []
    with timed(f"scripts/build_manifest.py --split {args.split}", scope=lambda: f"{len(rows):,} trials", rate=lambda: "headers only, no audio"):
        for trial_index in range(settings["n_trials"]):
            trial_id = f"{args.split}-{config['seed']}-{trial_index:06d}"
            row = build_trial(trial_id, args.split, settings, regimes, speakers, sex, book, by_speaker, by_chapter, chapters_of, noise, segments_of)
            if row is None:
                failed_ids.append(trial_id)
            else:
                rows.append(row)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    with (MANIFEST_DIR / f"{args.split}.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # The trials that fail are the hard ones, so their ids are kept and the bias they leave can be described.
    failed_file = MANIFEST_DIR / f"{args.split}.failed.txt"
    if failed_ids:
        failed_file.write_text("\n".join(failed_ids) + "\n")
    elif failed_file.exists():
        failed_file.unlink()

    regime_mix = {}
    for row in rows:
        regime_mix[row["regime"]] = regime_mix.get(row["regime"], 0) + 1
    sorted_regime_mix = {}
    for regime in sorted(regime_mix):
        sorted_regime_mix[regime] = regime_mix[regime]

    (MANIFEST_DIR / f"{args.split}.meta.yaml").write_text(yaml.safe_dump({
        "generated": date.today().isoformat(),
        "generator": "scripts/build_manifest.py",
        "split": args.split,
        "seed": config["seed"],
        "config": CONFIG_PATH,
        "config_md5": config_md5,
        "n_trials": len(rows),
        "n_requested": settings["n_trials"],
        "n_failed": len(failed_ids),
        "failed_ids_file": failed_file.name if failed_ids else None,
        "noise_split": settings["noise_split"],
        "noise_clips_screened": noise_screened,
        "noise_clips_kept": len(noise),
        "noise_max_speech_run_s": max_speech_run_s,
        # The vad block is what "overlap" means for this manifest.
        "vad_settings": vad.vad_config(config),
        "vad_index_built": vad_meta.get("generated"),
        "vad_detector_version": vad_meta.get("detector_version"),
        "n_utterances_silent_dropped": silent_dropped,
        "regimes": regimes,
        "regime_mix": sorted_regime_mix,
    }, sort_keys=False))

    print_summary(rows, len(failed_ids), len(noise), noise_screened - len(noise), max_speech_run_s, settings["composition"])


if __name__ == "__main__":
    main()
