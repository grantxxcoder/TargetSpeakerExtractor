"""Manifest row -> audio. Pure functions; `scripts/render_trials.py` is the writer.

Kept a pure function of (row, config, corpus paths) so a PyTorch `Dataset` can call
`render_trial` directly if B7's on-the-fly path is ever switched back on
(decisions.md 2026-08-15). Nothing here touches disk except reading source audio.

Every level, length and position comes from the manifest row. The renderer draws
NOTHING: given the same row and the same corpora it produces bit-identical audio,
which is what makes a logged seed mean something.

The decisions this file implements, all in docs/decisions/decisions.md:

  A1  2026-08-13  the reference is the target convolved with its OWN room, no
                  interferer and no noise -- "what the mic heard from that person".
                  Reverb is NOT removed; that is a stated thesis limitation.
  A2  2026-08-11  the noise bed WRAPS. `noise_offset_s` is a phase into a looped
                  stream, not a slice index. A naive slice silently zero-pads.
  A3  2026-08-12  every level is BS.1770-4 integrated loudness via `pyloudnorm`,
                  never RMS. RMS would make `sir_db` mean something different in
                  every trial, because it does not gate silence.
  A4  2026-08-12  the enrollment carries NO room. Convolving it would let the model
                  match on room acoustics instead of on voice.
  A5  2026-08-13  the output runs `t60_s` past `mixture_length_s`, so the reverb
                  tail that A1 puts inside the reference is not cut off.
  A6  2026-08-13  clipping is fixed by scaling every stem by ONE common factor, so
                  the SIR and SNR the trial was built to have are preserved.

  2026-08-11      in target-absent trials the INTERFERER is the level anchor.

Reference: Sabine (1922) via `pra.inverse_sabine` for the wall absorption, and the
image-source method (Allen & Berkley, 1979) for the impulse responses.
"""

from __future__ import annotations

from hashlib import sha1

import numpy as np
import pyroomacoustics as pra
import pyloudnorm as pyln
import soundfile as sf
from scipy.signal import fftconvolve, sosfilt

# BS.1770 gating uses 400 ms blocks, so `pyloudnorm` raises below that. Every stem
# here is a whole mixture (>=15 s) so this is a tripwire, not a real constraint.
MIN_LOUDNESS_S = 0.4

# A6. Not 1.0: leaves headroom so int16 rounding cannot push a sample over.
CLIP_CEILING = 0.95


# --- level ----------------------------------------------------------------

def loudness(x, meter):
    """BS.1770-4 integrated loudness in LUFS (A3)."""
    if len(x) < MIN_LOUDNESS_S * meter.rate:
        raise ValueError(f"stem is {len(x) / meter.rate:.3f} s, below BS.1770's "
                         f"{MIN_LOUDNESS_S} s block size")
    return meter.integrated_loudness(x)


def gain_to(x, target_lufs, meter):
    """The linear gain that puts `x` at `target_lufs`.

    Returned rather than applied, because A6 needs the un-clipped signals first
    and the target stem has to be scaled by exactly the same amount as the copy
    of it that goes into the mixture.
    """
    current = loudness(x, meter)
    if not np.isfinite(current):
        raise ValueError("silent stem: loudness is -inf. Under the 2026-08-11 "
                         "anchor rule this is only reachable through a bug.")
    return float(10 ** ((target_lufs - current) / 20.0))


# --- sources --------------------------------------------------------------

def lay_track(utt_paths, onsets_s, n_samples, sr):
    """Dry track: each utterance's audio dropped in at its recorded onset.

    Onsets are footprint positions from `build_manifest.lay_out`, so the files
    never overlap each other and the gaps between them are real silence.
    """
    track = np.zeros(n_samples, dtype=np.float64)
    for path, onset in zip(utt_paths, onsets_s):
        audio, file_sr = sf.read(path, dtype="float64", always_2d=False)
        if file_sr != sr:
            raise ValueError(f"{path} is {file_sr} Hz, expected {sr}")
        start = int(round(onset * sr))
        end = start + len(audio)
        if end > n_samples:
            raise ValueError(
                f"{path} runs {(end - n_samples) / sr:.3f} s past the window. "
                "build_manifest's footprint cap should make this unreachable.")
        track[start:end] += audio
    return track


def wrap_noise(path, offset_s, n_samples, sr):
    """`n_samples` of noise starting at `offset_s`, looping the clip (A2).

    WHAM! clips have a median of 10 s against 15-20 s mixtures, and the offset is
    drawn over the whole clip, so most trials cross the end at least once --
    1.9 seams on average, measured 2026-08-11 and recorded as a known artefact.
    """
    clip, file_sr = sf.read(path, dtype="float64", always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path} is {file_sr} Hz, expected {sr}")
    if clip.ndim > 1:
        clip = clip.mean(axis=1)
    start = int(round(offset_s * sr)) % len(clip)
    reps = int(np.ceil((start + n_samples) / len(clip)))
    return np.tile(clip, reps)[start:start + n_samples]


# --- room -----------------------------------------------------------------

def impulse_responses(dims, t60, mic, sources, sr):
    """One RIR per source, from the room the manifest already recorded.

    Both sources go in a single `ShoeBox` so the image-source model runs once
    rather than twice -- the dominant cost in the whole renderer.

    The room is fully determined by manifest columns, so no RNG is involved and
    re-rendering a trial reproduces its acoustics exactly.
    """
    absorption, max_order = pra.inverse_sabine(t60, dims)
    room = pra.ShoeBox(dims, fs=sr, max_order=max_order,
                       materials=pra.Material(absorption))
    room.add_microphone(np.array(mic))
    for src in sources:
        room.add_source(np.array(src))
    room.compute_rir()
    return [np.asarray(room.rir[0][i], dtype=np.float64)
            for i in range(len(sources))]


def convolve_to(track, rir, n_samples):
    """Wet signal, trimmed back to the padded window length.

    Trimming is safe because A5 pads by `t60_s`: the last utterance ends before
    `mixture_length_s`, so its tail decays inside the pad rather than being cut.
    """
    return fftconvolve(track, rir)[:n_samples]


# --- enrollment -----------------------------------------------------------

def eq_curve(rng, sr, n_bands=3):
    """Random RMS-preserving peaking EQ, standing in for a different microphone.

    CARTSE "channel-gap" augmentation (Li & Seki, 2026), via
    data-construction-parameters.md `enrollment_eq_augmentation`, which specifies
    "random RMS-preserving EQ curves" without fixing the curve. The three bands,
    the +/-6 dB range and Q=1.0 are this renderer's choice and are recorded per
    trial so the exact filter is reproducible and reportable.
    """
    bands = []
    sos = []
    for _ in range(n_bands):
        f0 = float(np.exp(rng.uniform(np.log(120.0), np.log(0.4 * sr))))
        gain_db = float(rng.uniform(-6.0, 6.0))
        q = 1.0
        bands.append({"f0_hz": round(f0, 2), "gain_db": round(gain_db, 2), "q": q})
        sos.append(_peaking_sos(f0, gain_db, q, sr))
    return np.concatenate(sos, axis=0), bands


def _peaking_sos(f0, gain_db, q, sr):
    """One peaking-EQ biquad as a second-order section.

    Audio EQ Cookbook (Bristow-Johnson), the standard RBJ formulation.
    """
    a = 10 ** (gain_db / 40.0)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    a_ = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return np.array([[b[0] / a_[0], b[1] / a_[0], b[2] / a_[0],
                      1.0, a_[1] / a_[0], a_[2] / a_[0]]])


def apply_eq(x, sos):
    """Filter, then restore the original RMS so EQ changes colour, not level."""
    before = float(np.sqrt(np.mean(x ** 2)))
    y = sosfilt(sos, x)
    after = float(np.sqrt(np.mean(y ** 2)))
    return y * (before / after) if after > 0 else y


# --- the whole trial ------------------------------------------------------

def render_trial(row, cfg, flac_of, texts, noise_root, noise_split):
    """One manifest row -> the stems, in memory. Pure: no disk writes, no RNG
    beyond the enrollment EQ, which is seeded from the trial id.

    Returns `(stems, meta)`. `stems` holds float64 arrays at `cfg["sample_rate"]`:

        mixture      what the model hears
        target       A1's reference: the target through its own room, alone
        enrollment   dry, no room (A4)

    Level chain, in the order it has to happen:

      1. anchor -> `target_loudness_lufs`. The anchor is the target, or the
         INTERFERER when the target is absent (2026-08-11), so absent trials sit
         at the same volume as present ones and cannot be spotted by loudness.
      2. interferer -> anchor level minus `sir_db`.
      3. noise -> anchor level minus `snr_db`.
      4. sum, then A6's single common gain if anything would clip.

    Gains are computed on the un-clipped signals and applied once at the end, so
    the target stem and the copy of it inside the mixture are scaled identically.
    """
    sr = cfg["sample_rate"]
    length_s = float(row["mixture_length_s"])
    t60 = float(row["t60_s"])
    # A5: keep running past the last word so the reverb tail inside the
    # reference is not truncated.
    n = int(round((length_s + t60) * sr))

    meter = pyln.Meter(sr)
    target_lufs = float(row["target_loudness_lufs"])

    t_utts = [u for u in row["target_utts"].split("|") if u]
    i_utts = [u for u in row["interferer_utts"].split("|") if u]
    t_onsets = [float(x) for x in row["target_onsets_s"].split("|") if x]
    i_onsets = [float(x) for x in row["interferer_onsets_s"].split("|") if x]

    dry_t = lay_track([flac_of[u] for u in t_utts], t_onsets, n, sr)
    dry_i = lay_track([flac_of[u] for u in i_utts], i_onsets, n, sr)

    # Both RIRs from one image-source solve. Positions are recorded per trial,
    # which is also what lets the A1 dereverberation ablation be rendered later
    # without re-drawing rooms (milestones.md M0).
    dims = [float(row["room_l"]), float(row["room_w"]), float(row["room_h"])]
    mic = [float(row["mic_x"]), float(row["mic_y"]), float(row["mic_z"])]
    src_t = [float(row["target_x"]), float(row["target_y"]), float(row["target_z"])]
    src_i = [float(row["interferer_x"]), float(row["interferer_y"]),
             float(row["interferer_z"])]
    rir_t, rir_i = impulse_responses(dims, t60, mic, [src_t, src_i], sr)

    wet_t = convolve_to(dry_t, rir_t, n) if t_utts else np.zeros(n)
    wet_i = convolve_to(dry_i, rir_i, n) if i_utts else np.zeros(n)

    # 1-2. Anchor, then the interferer relative to it.
    if t_utts:
        g_t = gain_to(wet_t, target_lufs, meter)
        g_i = (gain_to(wet_i, target_lufs - float(row["sir_db"]), meter)
               if i_utts else 0.0)
    elif i_utts:
        # Target absent: the interferer anchors, so `snr_db` and
        # `target_loudness_lufs` keep the meanings the manifest recorded.
        g_t = 0.0
        g_i = gain_to(wet_i, target_lufs, meter)
    else:
        # noise_only: nobody speaks, so there is no anchor signal. The noise is
        # placed where it would have sat had someone spoken at
        # `target_loudness_lufs`, so this condition is not identifiable by
        # level alone -- the same argument as the 2026-08-11 anchor decision.
        g_t = g_i = 0.0

    # 3. Noise, always relative to the anchor's level rather than to whichever
    #    signal happens to exist.
    noise = wrap_noise(noise_root / noise_split / row["noise_clip"],
                       float(row["noise_offset_s"]), n, sr)
    g_n = gain_to(noise, target_lufs - float(row["snr_db"]), meter)

    target = wet_t * g_t
    interferer = wet_i * g_i
    noise = noise * g_n
    mixture = target + interferer + noise

    # 4. A6: one factor across every stem, so SIR and SNR survive the fix.
    peak = max(float(np.abs(a).max()) for a in (mixture, target, interferer))
    common_gain = CLIP_CEILING / peak if peak > CLIP_CEILING else 1.0
    mixture *= common_gain
    target *= common_gain

    enrollment, eq_bands = render_enrollment(row, cfg, flac_of, meter,
                                             target_lufs)

    meta = {
        "trial_id": row["trial_id"],
        "sample_rate": sr,
        "n_samples": n,
        "mixture_length_s": length_s,
        "tail_pad_s": round(t60, 4),
        "gain_target": round(g_t, 6),
        "gain_interferer": round(g_i, 6),
        "gain_noise": round(g_n, 6),
        "peak_before_clip_guard": round(peak, 6),
        "common_gain": round(common_gain, 6),
        "clipped": common_gain < 1.0,
        "rir_len_target": len(rir_t),
        "rir_len_interferer": len(rir_i),
        # d in metric-definitions.md section 2: without both transcripts ICR is
        # not computable, and adding them later costs a re-render.
        "target_text": " ".join(texts[u] for u in t_utts),
        "interferer_text": " ".join(texts[u] for u in i_utts),
        "enrollment_eq": bool(int(row["enrollment_eq"])),
        "enrollment_eq_bands": eq_bands,
    }
    return {"mixture": mixture, "target": target, "enrollment": enrollment}, meta


def render_enrollment(row, cfg, flac_of, meter, target_lufs):
    """The conditioning clip: dry, no room (A4), optionally EQ'd.

    Levelled to `target_loudness_lufs` like the anchor. Not covered by any
    decision -- see the renderer entry in decisions.md 2026-08-16 -- but leaving
    it at LibriSpeech's native level would put a spread of loudness on the
    conditioning path for no reason, and level is a cue we close everywhere else.
    """
    sr = cfg["sample_rate"]
    path = flac_of[row["enrollment_utt"]]
    start = int(round(float(row["enrollment_offset_s"]) * sr))
    n = int(round(float(row["enrollment_length_s"]) * sr))
    audio, file_sr = sf.read(path, dtype="float64", start=start, frames=n,
                             always_2d=False)
    if file_sr != sr:
        raise ValueError(f"{path} is {file_sr} Hz, expected {sr}")

    bands = []
    if int(row["enrollment_eq"]):
        # Seeded from the trial id so the curve is reproducible from the
        # manifest alone, and independent of render order or worker count.
        rng = np.random.default_rng(trial_seed(row["trial_id"]))
        sos, bands = eq_curve(rng, sr)
        audio = apply_eq(audio, sos)

    return audio * gain_to(audio, target_lufs, meter), bands


def trial_seed(trial_id):
    """A stable 32-bit seed from the trial id.

    Derived from the id rather than a counter so the enrollment EQ is
    reproducible from the manifest alone -- independent of render order, worker
    count, and which subset of trials is being rendered. Python's `hash()` is
    salted per process and would not survive that.
    """
    return int.from_bytes(sha1(trial_id.encode()).digest()[:4], "little")
