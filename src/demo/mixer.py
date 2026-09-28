"""Build a two-speaker trial from arbitrary audio, the way the dataset was built.

Reuses src/data/render.py so a demo mixture is levelled and reverberated as the
training trials were: BS.1770 loudness (A3), target and interferer each through
their own image-source RIR in one shoebox room, WHAM! noise at an SNR relative
to the target, the shared 0.95 peak guard (A6), and a dry enrolment levelled to
the target's loudness (A4).

DIFFERENCES FROM render_trial, stated: the target recording is used as-is
rather than laid out from VAD onsets; the interferer is looped to cover it; the
room is drawn here from generator.yaml's ranges rather than read from a
manifest row; no enrolment EQ.
"""

import csv
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from src.data.render import convolve_to, gain_to, impulse_responses, wrap_noise


def trim_silence(x, sr, top_db=40.0, frame_s=0.02):
    """Drop leading/trailing frames more than `top_db` below the loudest.
    Browser recordings start with a click and a second of room tone."""
    n = int(sr * frame_s)
    if len(x) < 2 * n:
        return x
    frames = x[:len(x) // n * n].reshape(-1, n)
    rms_db = 10 * np.log10((frames ** 2).mean(axis=1) + 1e-12)
    keep = np.flatnonzero(rms_db > rms_db.max() - top_db)
    return x[keep[0] * n:(keep[-1] + 1) * n]


def sample_room(rng, cfg):
    """dims, t60, mic, [target_pos, interferer_pos] from generator.yaml's ranges."""
    dims = [rng.uniform(*cfg["room_length_m"]), rng.uniform(*cfg["room_width_m"]),
            rng.uniform(*cfg["room_height_m"])]
    m = cfg["wall_margin_m"]
    for _ in range(1000):
        mic = [rng.uniform(m, dims[0] - m), rng.uniform(m, dims[1] - m),
               rng.uniform(*cfg["mic_height_m"])]
        sources = []
        for _ in range(2):
            d, a = rng.uniform(*cfg["source_distance_m"]), rng.uniform(0, 2 * np.pi)
            sources.append([mic[0] + d * np.cos(a), mic[1] + d * np.sin(a),
                            rng.uniform(*cfg["source_height_m"])])
        if all(m <= s[0] <= dims[0] - m and m <= s[1] <= dims[1] - m for s in sources):
            return dims, float(rng.uniform(*cfg["t60_s"])), mic, sources
    raise RuntimeError("could not place two sources in the room")


def loop_to(x, n, offset):
    """`x` repeated to fill samples [offset, n) of a length-n track."""
    track = np.zeros(n)
    if offset < n:
        reps = -(-(n - offset) // len(x))
        track[offset:] = np.tile(x, reps)[:n - offset]
    return track


def build_mixture(target, interferer, cfg, rng, sir_db, snr_db, reverb, noise_dir):
    """float arrays at cfg['sample_rate'] -> dict of stems + what was drawn.

    `snr_db` None = no noise. Returns mixture, target (the reference as it sits
    in the mixture: through its own room when reverb is on, A1), interferer,
    and `meta` recording every draw so the record can reproduce it.
    """
    sr = cfg["sample_rate"]
    meter = pyln.Meter(sr)
    lufs = cfg["target_lufs"]
    n = len(target)
    dry_i = loop_to(interferer, n, int(round(cfg["interferer_offset_s"] * sr)))
    meta = {"sir_db": sir_db, "snr_db": snr_db, "reverb": reverb,
            "target_lufs": lufs, "interferer_offset_s": cfg["interferer_offset_s"]}

    if reverb:
        dims, t60, mic, sources = sample_room(rng, cfg)
        n_out = n + int(round(t60 * sr))          # A5: pad so the tail decays
        rir_t, rir_i = impulse_responses(dims, t60, mic, sources, sr)
        pad = lambda x: np.pad(x, (0, n_out - n))  # noqa: E731
        wet_t, wet_i = convolve_to(pad(target), rir_t, n_out), convolve_to(pad(dry_i), rir_i, n_out)
        meta["room"] = {"dims_m": dims, "t60_s": t60, "mic": mic, "sources": sources}
    else:
        n_out, wet_t, wet_i = n, target.astype(np.float64), dry_i

    t = wet_t * gain_to(wet_t, lufs, meter)
    i = wet_i * gain_to(wet_i, lufs - sir_db, meter)
    noise = np.zeros(n_out)
    if snr_db is not None:
        clips = sorted(Path(noise_dir).glob("*.*"))
        clip = clips[rng.integers(len(clips))]
        offset = float(rng.uniform(0, sf.info(clip).duration))
        noise = wrap_noise(clip, offset, n_out, sr)
        noise = noise * gain_to(noise, lufs - snr_db, meter)
        meta["noise"] = {"clip": str(clip), "offset_s": offset}

    mixture = t + i + noise
    peak = max(np.abs(mixture).max(), np.abs(t).max(), np.abs(i).max())
    g = cfg["peak"] / peak if peak > cfg["peak"] else 1.0      # A6, one gain for all
    meta["common_gain"] = float(g)
    f32 = lambda x: (x * g).astype(np.float32)  # noqa: E731
    return {"mixture": f32(mixture), "target": f32(t), "interferer": f32(i),
            "meta": meta}


def level_enrollment(enrollment, cfg):
    """Dry, at most enrollment_max_s from the middle, at the target's LUFS (A4)."""
    sr = cfg["sample_rate"]
    n = int(cfg["enrollment_max_s"] * sr)
    if len(enrollment) > n:
        start = (len(enrollment) - n) // 2
        enrollment = enrollment[start:start + n]
    out = enrollment * gain_to(enrollment.astype(np.float64), cfg["target_lufs"],
                               pyln.Meter(sr))
    peak = np.abs(out).max()
    return (out * (cfg["peak"] / peak if peak > cfg["peak"] else 1.0)).astype(np.float32)


# --- "use example voices": LibriSpeech speakers from ONE public split ---------

def _speakers(root, split_ids):
    sexes = {}
    with open(Path(root).parent / "SPEAKERS.TXT") as fh:
        for row in csv.reader(fh, delimiter="|"):
            if row and not row[0].startswith(";"):
                sexes[row[0].strip()] = row[1].strip()
    return {s: sexes[s] for s in split_ids}


def _utterances(root, speaker):
    """[(path, seconds, text)] for one speaker."""
    out = []
    for trans in sorted(Path(root, speaker).glob("*/*.trans.txt")):
        for line in trans.read_text().splitlines():
            utt, text = line.split(" ", 1)
            path = trans.parent / f"{utt}.flac"
            out.append((path, sf.info(path).duration, text))
    return out


def pick_example(rng, root, split_ids, target_seconds, enrollment_s):
    """Target utterance in `target_seconds`, a DIFFERENT utterance of the same
    speaker for enrolment, and an opposite-sex speaker as interferer.

    Opposite sex because the demo should show the mechanism, not the hardest
    case; same-sex is harder (decisions-m2.md, "lands right" row: 65.8 %
    same-gender vs 87.2 % cross-gender for this checkpoint).
    """
    speakers = _speakers(root, split_ids)
    order = list(speakers)
    rng.shuffle(order)
    for spk in order:
        utts = _utterances(root, spk)
        targets = [u for u in utts if target_seconds[0] <= u[1] <= target_seconds[1]]
        if not targets:
            continue
        tgt = targets[rng.integers(len(targets))]
        enrols = [u for u in utts if u[0] != tgt[0] and u[1] >= enrollment_s]
        if not enrols:
            continue
        enr = enrols[rng.integers(len(enrols))]
        others = [s for s in order if speakers[s] != speakers[spk]]
        itf_spk = others[rng.integers(len(others))]
        itf_utts = _utterances(root, itf_spk)
        rng.shuffle(itf_utts)
        itf, total = [], 0.0
        for u in itf_utts:                       # enough to cover the target
            itf.append(u)
            total += u[1]
            if total >= tgt[1]:
                break
        read = lambda p: sf.read(p, dtype="float32")[0]  # noqa: E731
        return {
            "target": read(tgt[0]),
            "enrollment": read(enr[0]),
            "interferer": np.concatenate([read(u[0]) for u in itf]),
            "reference_text": tgt[2],
            "ids": {"target": tgt[0].stem, "enrollment": enr[0].stem,
                    "interferer": [u[0].stem for u in itf],
                    "target_speaker": f"{spk} ({speakers[spk]})",
                    "interferer_speaker": f"{itf_spk} ({speakers[itf_spk]})"},
        }
    raise RuntimeError("no speaker has an utterance in the requested length range")
