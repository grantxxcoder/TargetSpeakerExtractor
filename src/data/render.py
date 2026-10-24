"""Turns one manifest row into audio. scripts/render_trials.py writes it to disk.

The renderer draws nothing at random except the enrolment EQ, which is seeded from the trial id, so the same row always gives bit-identical audio.
The decisions it follows are all in decisions-m0.md:
A1 (2026-08-13) the reference is the target through its own room, alone. A2 (2026-08-11) the noise bed loops.
A3 (2026-08-12) every level is BS.1770-4 loudness, never RMS. A4 (2026-08-12) the enrolment has no room.
A5 (2026-08-13) the audio runs one T60 past the mixture length so the reverb tail is kept. A6 (2026-08-13) clipping is fixed with one gain on every stem.
On a target-absent trial the interferer sets the level instead (2026-08-11).
The wall absorption comes from Sabine's equation (Sabine, 1922) and the impulse responses from the image-source method (Allen & Berkley, 1979).
"""

from hashlib import sha1

import numpy as np
import pyloudnorm as pyln
import pyroomacoustics as pra
import soundfile as sf
from scipy.signal import fftconvolve, sosfilt

# BS.1770 measures loudness in 400 ms blocks, so a shorter stem cannot be measured.
MIN_LOUDNESS_S = 0.4

# A6's ceiling. It is below 1.0 so int16 rounding cannot push a sample over full scale.
CLIP_CEILING = 0.95

# The enrolment EQ has three peaking bands.
EQ_BANDS = 3


# this gives the BS.1770-4 integrated loudness of a signal in LUFS (A3)
def loudness(signal, meter):
    if len(signal) < MIN_LOUDNESS_S * meter.rate:
        raise ValueError(f"The stem is {len(signal) / meter.rate:.3f} s, below BS.1770's {MIN_LOUDNESS_S} s block size.")
    return meter.integrated_loudness(signal)


# this gives the linear gain that puts a signal at the wanted loudness
def gain_to(signal, target_lufs, meter):
    """It returns the gain rather than applying it, because A6 needs every stem unclipped first, and the target and its copy in the mixture must take the same gain."""
    current_lufs = loudness(signal, meter)
    if not np.isfinite(current_lufs):
        raise ValueError("The stem is silent, so its loudness is -inf.")
    return float(10 ** ((target_lufs - current_lufs) / 20.0))


# this lays each utterance into a silent track at its recorded onset
def lay_track(utterance_paths, onsets_s, n_samples, sample_rate):
    track = np.zeros(n_samples, dtype=np.float64)
    for path, onset in zip(utterance_paths, onsets_s):
        audio, file_sample_rate = sf.read(path, dtype="float64", always_2d=False)
        if file_sample_rate != sample_rate:
            raise ValueError(f"{path} is {file_sample_rate} Hz, but {sample_rate} Hz was expected.")
        start = int(round(onset * sample_rate))
        end = start + len(audio)
        if end > n_samples:
            raise ValueError(f"{path} runs {(end - n_samples) / sample_rate:.3f} s past the window.")
        track[start:end] += audio
    return track


# this takes n_samples of a noise clip from the offset onwards, looping the clip when it runs out (A2)
def wrap_noise(path, offset_s, n_samples, sample_rate):
    clip, file_sample_rate = sf.read(path, dtype="float64", always_2d=False)
    if file_sample_rate != sample_rate:
        raise ValueError(f"{path} is {file_sample_rate} Hz, but {sample_rate} Hz was expected.")
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    start = int(round(offset_s * sample_rate)) % len(clip)
    repeats = int(np.ceil((start + n_samples) / len(clip)))
    return np.tile(clip, repeats)[start:start + n_samples]


# this computes one room impulse response per source, with both sources in one room so the image-source model runs once
def impulse_responses(dims, t60, mic, sources, sample_rate):
    absorption, max_order = pra.inverse_sabine(t60, dims)
    room = pra.ShoeBox(dims, fs=sample_rate, max_order=max_order, materials=pra.Material(absorption))
    room.add_microphone(np.array(mic))
    for source in sources:
        room.add_source(np.array(source))
    room.compute_rir()
    responses = []
    for source_index in range(len(sources)):
        responses.append(np.asarray(room.rir[0][source_index], dtype=np.float64))
    return responses


# this convolves a dry track with a room and trims it back to the window, which already holds the tail because of A5
def convolve_to(track, rir, n_samples):
    return fftconvolve(track, rir)[:n_samples]


# this draws a random peaking EQ that keeps the RMS, standing in for a different microphone
def eq_curve(rng, sample_rate):
    """CARTSE's channel-gap augmentation (Li & Seki, 2026). The three bands, the +-6 dB range and Q of 1.0 are this renderer's choice, and each curve is recorded per trial."""
    bands = []
    sections = []
    for band_index in range(EQ_BANDS):
        centre_hz = float(np.exp(rng.uniform(np.log(120.0), np.log(0.4 * sample_rate))))
        gain_db = float(rng.uniform(-6.0, 6.0))
        quality = 1.0
        bands.append({"f0_hz": round(centre_hz, 2), "gain_db": round(gain_db, 2), "q": quality})
        sections.append(peaking_sos(centre_hz, gain_db, quality, sample_rate))
    return np.concatenate(sections, axis=0), bands


# this builds one peaking-EQ biquad as a second-order section, from the Audio EQ Cookbook (Bristow-Johnson)
def peaking_sos(centre_hz, gain_db, quality, sample_rate):
    amplitude = 10 ** (gain_db / 40.0)
    angular_frequency = 2 * np.pi * centre_hz / sample_rate
    alpha = np.sin(angular_frequency) / (2 * quality)
    numerator = [1 + alpha * amplitude, -2 * np.cos(angular_frequency), 1 - alpha * amplitude]
    denominator = [1 + alpha / amplitude, -2 * np.cos(angular_frequency), 1 - alpha / amplitude]
    return np.array([[numerator[0] / denominator[0], numerator[1] / denominator[0], numerator[2] / denominator[0], 1.0, denominator[1] / denominator[0], denominator[2] / denominator[0]]])


# this filters a signal and then restores its RMS, so the EQ changes the colour and not the level
def apply_eq(signal, sections):
    rms_before = float(np.sqrt(np.mean(signal ** 2)))
    filtered = sosfilt(sections, signal)
    rms_after = float(np.sqrt(np.mean(filtered ** 2)))
    if rms_after > 0:
        return filtered * (rms_before / rms_after)
    return filtered


# this splits a "|" separated manifest field into its non-empty parts
def split_field(text):
    parts = []
    for part in text.split("|"):
        if part:
            parts.append(part)
    return parts


# this renders one manifest row to its stems in memory, plus the trial's meta
def render_trial(row, sample_rate, audio_paths, texts, noise_root, noise_split):
    """The levels are set in order: the anchor to target_loudness_lufs, the interferer to anchor minus sir_db, the noise to anchor minus snr_db, then A6's one common gain if the sum would clip.

    The anchor is the target, or the interferer when the target is absent, so absent trials cannot be spotted by loudness.
    """
    length_s = float(row["mixture_length_s"])
    t60 = float(row["t60_s"])
    # A5: run past the last word so the reverb tail inside the reference is not cut off.
    n_samples = int(round((length_s + t60) * sample_rate))

    meter = pyln.Meter(sample_rate)
    target_lufs = float(row["target_loudness_lufs"])

    target_utterances = split_field(row["target_utts"])
    interferer_utterances = split_field(row["interferer_utts"])
    target_onsets = []
    for onset in split_field(row["target_onsets_s"]):
        target_onsets.append(float(onset))
    interferer_onsets = []
    for onset in split_field(row["interferer_onsets_s"]):
        interferer_onsets.append(float(onset))

    target_paths = []
    for utterance in target_utterances:
        target_paths.append(audio_paths[utterance])
    interferer_paths = []
    for utterance in interferer_utterances:
        interferer_paths.append(audio_paths[utterance])
    dry_target = lay_track(target_paths, target_onsets, n_samples, sample_rate)
    dry_interferer = lay_track(interferer_paths, interferer_onsets, n_samples, sample_rate)

    dims = [float(row["room_l"]), float(row["room_w"]), float(row["room_h"])]
    mic = [float(row["mic_x"]), float(row["mic_y"]), float(row["mic_z"])]
    target_position = [float(row["target_x"]), float(row["target_y"]), float(row["target_z"])]
    interferer_position = [float(row["interferer_x"]), float(row["interferer_y"]), float(row["interferer_z"])]
    target_rir, interferer_rir = impulse_responses(dims, t60, mic, [target_position, interferer_position], sample_rate)

    if target_utterances:
        wet_target = convolve_to(dry_target, target_rir, n_samples)
    else:
        wet_target = np.zeros(n_samples)
    if interferer_utterances:
        wet_interferer = convolve_to(dry_interferer, interferer_rir, n_samples)
    else:
        wet_interferer = np.zeros(n_samples)

    # The anchor first, then the interferer relative to it.
    if target_utterances:
        target_gain = gain_to(wet_target, target_lufs, meter)
        if interferer_utterances:
            interferer_gain = gain_to(wet_interferer, target_lufs - float(row["sir_db"]), meter)
        else:
            interferer_gain = 0.0
    elif interferer_utterances:
        # The target is absent, so the interferer is the anchor.
        target_gain = 0.0
        interferer_gain = gain_to(wet_interferer, target_lufs, meter)
    else:
        # Nobody speaks, so the noise sits where it would have sat had someone spoken at target_loudness_lufs.
        target_gain = 0.0
        interferer_gain = 0.0

    # The noise is always set relative to the anchor's level.
    noise = wrap_noise(noise_root / noise_split / row["noise_clip"], float(row["noise_offset_s"]), n_samples, sample_rate)
    noise_gain = gain_to(noise, target_lufs - float(row["snr_db"]), meter)

    target = wet_target * target_gain
    interferer = wet_interferer * interferer_gain
    noise = noise * noise_gain
    mixture = target + interferer + noise

    # A6: one gain on every stem, so the SIR and SNR survive the clip fix and mixture still equals target plus interferer plus noise.
    peak = max(float(np.abs(mixture).max()), float(np.abs(target).max()), float(np.abs(interferer).max()))
    if peak > CLIP_CEILING:
        common_gain = CLIP_CEILING / peak
    else:
        common_gain = 1.0
    mixture *= common_gain
    target *= common_gain
    interferer *= common_gain

    enrollment, eq_bands = render_enrollment(row, sample_rate, audio_paths, meter, target_lufs, "")
    # The interferer's enrolment, for the second training direction. Its EQ is seeded from a different string, so the two enrolments never share a curve.
    interferer_enrollment, interferer_eq_bands = render_enrollment(row, sample_rate, audio_paths, meter, target_lufs, "interferer_")

    target_texts = []
    for utterance in target_utterances:
        target_texts.append(texts[utterance])
    interferer_texts = []
    for utterance in interferer_utterances:
        interferer_texts.append(texts[utterance])

    meta = {
        "trial_id": row["trial_id"],
        "sample_rate": sample_rate,
        "n_samples": n_samples,
        "mixture_length_s": length_s,
        "tail_pad_s": round(t60, 4),
        "gain_target": round(target_gain, 6),
        "gain_interferer": round(interferer_gain, 6),
        "gain_noise": round(noise_gain, 6),
        "peak_before_clip_guard": round(peak, 6),
        "common_gain": round(common_gain, 6),
        "clipped": common_gain < 1.0,
        "rir_len_target": len(target_rir),
        "rir_len_interferer": len(interferer_rir),
        # Both transcripts are kept, because the leakage measure needs the interferer's words too.
        "target_text": " ".join(target_texts),
        "interferer_text": " ".join(interferer_texts),
        "enrollment_eq": bool(int(row["enrollment_eq"])),
        "enrollment_eq_bands": eq_bands,
        "interferer_enrollment_eq": bool(int(row["interferer_enrollment_eq"])),
        "interferer_enrollment_eq_bands": interferer_eq_bands,
    }
    stems = {"mixture": mixture, "target": target, "enrollment": enrollment, "interferer": interferer, "interferer_enrollment": interferer_enrollment}
    return stems, meta


# this renders an enrolment clip: dry with no room (A4), with the EQ when the manifest asks for it, levelled to target_loudness_lufs
def render_enrollment(row, sample_rate, audio_paths, meter, target_lufs, prefix):
    """The prefix picks whose: "" for the target and "interferer_" for the other speaker. One function builds both, so how an enrolment was made cannot become a cue for which direction is asked."""
    path = audio_paths[row[f"{prefix}enrollment_utt"]]
    start = int(round(float(row[f"{prefix}enrollment_offset_s"]) * sample_rate))
    n_samples = int(round(float(row[f"{prefix}enrollment_length_s"]) * sample_rate))
    audio, file_sample_rate = sf.read(path, dtype="float64", start=start, frames=n_samples, always_2d=False)
    if file_sample_rate != sample_rate:
        raise ValueError(f"{path} is {file_sample_rate} Hz, but {sample_rate} Hz was expected.")

    bands = []
    if int(row[f"{prefix}enrollment_eq"]):
        # Seeded from the trial id and the prefix, so the curve follows from the manifest alone and the two enrolments in a trial get different curves.
        rng = np.random.default_rng(trial_seed(row["trial_id"] + prefix))
        sections, bands = eq_curve(rng, sample_rate)
        audio = apply_eq(audio, sections)

    return audio * gain_to(audio, target_lufs, meter), bands


# this turns a trial id into a stable 32-bit seed. Python's hash() changes between processes, so it cannot be used
def trial_seed(trial_id):
    return int.from_bytes(sha1(trial_id.encode()).digest()[:4], "little")
