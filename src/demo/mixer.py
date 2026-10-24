"""Builds a two-speaker trial from any audio, the way the dataset was built.

It reuses src/data/render.py, so a demo mixture is levelled and reverberated as the training trials were: BS.1770 loudness (A3), the target and the interferer each through their own image-source RIR in one shoebox room,
WHAM! noise at an SNR relative to the target, the shared 0.95 peak guard (A6), and a dry enrolment levelled to the target's loudness (A4).
It differs from render_trial in four stated ways: the target recording is used as it is rather than laid out from VAD onsets, the interferer is looped to cover it, the room is drawn here from generator.yaml's ranges, and the enrolment gets no EQ.
"""

import csv
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from src.data.render import convolve_to, gain_to, impulse_responses, wrap_noise

# Browser recordings start with a click and a second of room tone, so frames this far below the loudest are trimmed from both ends.
TRIM_BELOW_DB = 40.0
TRIM_FRAME_S = 0.02


# this drops the leading and trailing frames that are much quieter than the loudest
def trim_silence(audio, sample_rate):
    frame_length = int(sample_rate * TRIM_FRAME_S)
    if len(audio) < 2 * frame_length:
        return audio
    frames = audio[:len(audio) // frame_length * frame_length].reshape(-1, frame_length)
    level_db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    loud_frames = np.flatnonzero(level_db > level_db.max() - TRIM_BELOW_DB)
    return audio[loud_frames[0] * frame_length:(loud_frames[-1] + 1) * frame_length]


# this draws a value uniformly between the two ends of a configured range
def draw_between(generator, bounds):
    return generator.uniform(bounds[0], bounds[1])


# this draws a room, its T60, the microphone and the two source positions from generator.yaml's ranges
def sample_room(generator, config):
    dimensions = [draw_between(generator, config["room_length_m"]), draw_between(generator, config["room_width_m"]), draw_between(generator, config["room_height_m"])]
    margin = config["wall_margin_m"]
    for attempt in range(1000):
        microphone = [generator.uniform(margin, dimensions[0] - margin), generator.uniform(margin, dimensions[1] - margin), draw_between(generator, config["mic_height_m"])]
        sources = []
        for source_index in range(2):
            distance = draw_between(generator, config["source_distance_m"])
            angle = generator.uniform(0, 2 * np.pi)
            sources.append([microphone[0] + distance * np.cos(angle), microphone[1] + distance * np.sin(angle), draw_between(generator, config["source_height_m"])])
        inside = True
        for source in sources:
            if not (margin <= source[0] <= dimensions[0] - margin and margin <= source[1] <= dimensions[1] - margin):
                inside = False
        if inside:
            return dimensions, float(draw_between(generator, config["t60_s"])), microphone, sources
    raise RuntimeError("could not place two sources in the room")


# this repeats the audio to fill samples offset to length of a track that long
def loop_to(audio, length, offset):
    track = np.zeros(length)
    if offset < length:
        repeats = (length - offset + len(audio) - 1) // len(audio)
        track[offset:] = np.tile(audio, repeats)[:length - offset]
    return track


# this builds the mixture and returns its stems, plus a record of every draw so the run can be reproduced
def build_mixture(target, interferer, config, generator, sir_db, snr_db, reverb, noise_dir):
    """snr_db of None means no noise. The target stem is the reference as it sits in the mixture, so it goes through its own room when reverb is on (A1)."""
    sample_rate = config["sample_rate"]
    meter = pyln.Meter(sample_rate)
    target_lufs = config["target_lufs"]
    length = len(target)
    dry_interferer = loop_to(interferer, length, int(round(config["interferer_offset_s"] * sample_rate)))
    meta = {"sir_db": sir_db, "snr_db": snr_db, "reverb": reverb, "target_lufs": target_lufs, "interferer_offset_s": config["interferer_offset_s"]}

    if reverb:
        dimensions, t60, microphone, sources = sample_room(generator, config)
        # The output is padded so the reverb tail can decay (A5).
        output_length = length + int(round(t60 * sample_rate))
        target_rir, interferer_rir = impulse_responses(dimensions, t60, microphone, sources, sample_rate)
        wet_target = convolve_to(np.pad(target, (0, output_length - length)), target_rir, output_length)
        wet_interferer = convolve_to(np.pad(dry_interferer, (0, output_length - length)), interferer_rir, output_length)
        meta["room"] = {"dims_m": dimensions, "t60_s": t60, "mic": microphone, "sources": sources}
    else:
        output_length = length
        wet_target = target.astype(np.float64)
        wet_interferer = dry_interferer

    target_stem = wet_target * gain_to(wet_target, target_lufs, meter)
    interferer_stem = wet_interferer * gain_to(wet_interferer, target_lufs - sir_db, meter)
    noise = np.zeros(output_length)
    if snr_db is not None:
        clips = sorted(Path(noise_dir).glob("*.*"))
        clip = clips[generator.integers(len(clips))]
        offset = float(generator.uniform(0, sf.info(clip).duration))
        noise = wrap_noise(clip, offset, output_length, sample_rate)
        noise = noise * gain_to(noise, target_lufs - snr_db, meter)
        meta["noise"] = {"clip": str(clip), "offset_s": offset}

    mixture = target_stem + interferer_stem + noise
    # One gain for all three stems, so the mixture never clips (A6).
    peak = max(np.abs(mixture).max(), np.abs(target_stem).max(), np.abs(interferer_stem).max())
    common_gain = 1.0
    if peak > config["peak"]:
        common_gain = config["peak"] / peak
    meta["common_gain"] = float(common_gain)
    return {"mixture": (mixture * common_gain).astype(np.float32), "target": (target_stem * common_gain).astype(np.float32), "interferer": (interferer_stem * common_gain).astype(np.float32), "meta": meta}


# this keeps at most enrollment_max_s from the middle of the enrolment, dry, levelled to the target's loudness (A4)
def level_enrollment(enrollment, config):
    sample_rate = config["sample_rate"]
    max_length = int(config["enrollment_max_s"] * sample_rate)
    if len(enrollment) > max_length:
        start = (len(enrollment) - max_length) // 2
        enrollment = enrollment[start:start + max_length]
    levelled = enrollment * gain_to(enrollment.astype(np.float64), config["target_lufs"], pyln.Meter(sample_rate))
    peak = np.abs(levelled).max()
    peak_gain = 1.0
    if peak > config["peak"]:
        peak_gain = config["peak"] / peak
    return (levelled * peak_gain).astype(np.float32)


# The rest serves "use example voices": LibriSpeech speakers from one public split.

# this maps each given LibriSpeech speaker to "F" or "M", from SPEAKERS.TXT
def speaker_sexes(librispeech_root, split_speakers):
    all_sexes = {}
    with open(Path(librispeech_root).parent / "SPEAKERS.TXT") as handle:
        for row in csv.reader(handle, delimiter="|"):
            if row and not row[0].startswith(";"):
                all_sexes[row[0].strip()] = row[1].strip()
    sexes = {}
    for speaker in split_speakers:
        sexes[speaker] = all_sexes[speaker]
    return sexes


# this lists one speaker's utterances as (path, seconds, text)
def utterances(librispeech_root, speaker):
    found = []
    for transcript_file in sorted(Path(librispeech_root, speaker).glob("*/*.trans.txt")):
        for line in transcript_file.read_text().splitlines():
            utterance_id, text = line.split(" ", 1)
            path = transcript_file.parent / f"{utterance_id}.flac"
            found.append((path, sf.info(path).duration, text))
    return found


# this shuffles one speaker's utterances and joins them until they cover the given length. build_mixture loops the audio if it is still short
def draw_utterances(generator, librispeech_root, speaker, seconds):
    candidates = utterances(librispeech_root, speaker)
    generator.shuffle(candidates)
    chosen = []
    total_seconds = 0.0
    for candidate in candidates:
        chosen.append(candidate)
        total_seconds += candidate[1]
        if total_seconds >= seconds:
            break
    pieces = []
    utterance_ids = []
    for candidate in chosen:
        pieces.append(sf.read(candidate[0], dtype="float32")[0])
        utterance_ids.append(candidate[0].stem)
    return np.concatenate(pieces), utterance_ids


# this picks a target utterance, a different utterance of the same speaker as the enrolment, and an interferer
def pick_example(generator, librispeech_root, split_speakers, target_seconds, enrollment_s, interferer_speaker, exclude):
    """The interferer is interferer_speaker when one is given, else a random opposite-sex speaker, because the demo should show the mechanism rather than the hardest case.
    Speakers in exclude are never the target, so the target can never be one of the selectable interferer voices.
    """
    sexes = speaker_sexes(librispeech_root, split_speakers)
    order = list(sexes)
    generator.shuffle(order)
    for speaker in order:
        if speaker in exclude or speaker == interferer_speaker:
            continue
        speaker_utterances = utterances(librispeech_root, speaker)
        targets = []
        for candidate in speaker_utterances:
            if target_seconds[0] <= candidate[1] <= target_seconds[1]:
                targets.append(candidate)
        if not targets:
            continue
        target = targets[generator.integers(len(targets))]
        enrolments = []
        for candidate in speaker_utterances:
            if candidate[0] != target[0] and candidate[1] >= enrollment_s:
                enrolments.append(candidate)
        if not enrolments:
            continue
        enrolment = enrolments[generator.integers(len(enrolments))]
        if interferer_speaker is None:
            opposite_sex = []
            for other in order:
                if sexes[other] != sexes[speaker]:
                    opposite_sex.append(other)
            chosen_interferer = opposite_sex[generator.integers(len(opposite_sex))]
        else:
            chosen_interferer = interferer_speaker
        interferer_audio, interferer_ids = draw_utterances(generator, librispeech_root, chosen_interferer, target[1])
        return {
            "target": sf.read(target[0], dtype="float32")[0],
            "enrollment": sf.read(enrolment[0], dtype="float32")[0],
            "interferer": interferer_audio,
            "reference_text": target[2],
            "ids": {"target": target[0].stem, "enrollment": enrolment[0].stem, "interferer": interferer_ids, "target_speaker": f"{speaker} ({sexes[speaker]})", "interferer_speaker": f"{chosen_interferer} ({sexes[chosen_interferer]})"},
        }
    raise RuntimeError("no speaker has an utterance in the requested length range")
