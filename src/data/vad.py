"""Voice-activity detection over the utterance corpus (B2).

Detector: Silero VAD — Silero Team (2021), "Silero VAD: pre-trained enterprise-grade
Voice Activity Detector", https://github.com/snakers4/silero-vad. Used unmodified.

WHY THIS EXISTS
---------------
`scripts/build_manifest.py` currently treats a LibriSpeech utterance as speech from
the first sample to the last, because the file duration is all it has. It is not: a
read sentence carries silence before the first word, pauses between clauses, and
silence after the last. Measured over 2,000 utterances (decisions-m0.md 2026-08-15),
**86.0 % of a file is speech**, with a near-constant **0.331 s of leading silence**
and 0.129 s of trailing silence.

Measuring overlap from file boundaries therefore overstates it by ~25 %, and the
error varies per trial (mean 0.071, max 0.274) rather than sitting at a constant
offset — so it cannot be corrected with a multiplier, and it puts individual trials
into the wrong B13 overlap bucket.

This module produces the speech map. It changes no audio: mixtures still contain
the pauses, because that is what speech sounds like. Only the measurement changes.

WHAT IS AND IS NOT HERE
-----------------------
Everything below the `detect()` boundary is pure interval arithmetic with no torch
dependency, so it is unit-tested directly. `detect()` and `load_model()` are the
only parts that touch the model.

Nothing here is wired into `scripts/build_manifest.py` yet — that is PR2, which
rebuilds the manifests. This is PR1: build the map, verify it, change nothing.
The same split was used for B12 (sampling.py PR1 / wire-in PR2) and worked.

    from src.data import vad
    segs = vad.detect(wav, vad.load_model(), cfg)   # [(start_s, end_s), ...]
    vad.total_speech(segs)                          # seconds of actual speech
    vad.shared_seconds(a, b)                        # genuine simultaneous speech
"""

from __future__ import annotations

SEG_SEP = "|"      # between segments, matching the manifest's existing convention
BOUND_SEP = ":"    # within a segment. NOT "-": utterance ids already use hyphens,
                   # so "-" would make a segment string ambiguous to eyeball.

# The four knobs, and the only ones this project varies. Defaults here are Silero's
# own; the values actually used come from generator.yaml and must be passed in.
VAD_KEYS = ("threshold", "min_silence_duration_ms",
            "min_speech_duration_ms", "speech_pad_ms")

_model = None


def load_model():
    """The Silero model, loaded once per process.

    From the pip package, never `torch.hub.load`: hub fetches from GitHub at call
    time, so a rebuild months later could silently pick up different weights and
    move every overlap number in the thesis. The pip package pins them.
    """
    global _model
    if _model is None:
        from silero_vad import load_silero_vad
        _model = load_silero_vad()
    return _model


def model_version():
    """Installed silero-vad version, recorded alongside every cache we write."""
    from importlib.metadata import version
    return version("silero-vad")


def vad_config(config):
    """The validated `vad:` block from generator.yaml.

    Every knob must be stated explicitly. Falling back to Silero's defaults would
    let the definition of "overlap" change without appearing in any diff, and that
    definition is quoted in the thesis next to every overlap figure.
    """
    cfg = config.get("vad")
    if not cfg:
        raise KeyError("generator.yaml has no `vad:` block; B2 requires one")
    missing = [k for k in VAD_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"vad config is missing {missing}")

    out = {k: cfg[k] for k in VAD_KEYS}
    if not 0.0 < out["threshold"] < 1.0:
        raise ValueError(f"vad threshold must be in (0, 1), got {out['threshold']}")
    for k in VAD_KEYS[1:]:
        if out[k] < 0:
            raise ValueError(f"vad {k} must be >= 0, got {out[k]}")

    # A silent weights upgrade would move every overlap number without touching a
    # tracked file. Same discipline as pinning the judge model id.
    want = cfg.get("expected_model_version")
    if want and want != model_version():
        raise RuntimeError(
            f"silero-vad is {model_version()}, config expects {want}. Every "
            f"overlap figure depends on these weights -- either pin the install "
            f"back, or bump expected_model_version AND rebuild the VAD index and "
            f"the manifests, recording the change in decisions-m0.md.")
    return out


def detect(wav, model, cfg, sample_rate=16000):
    """Speech segments in one waveform, as [(start_s, end_s), ...].

    Sample-accurate: Silero's `return_seconds=True` also rounds to `time_resolution`
    decimal places, which defaults to ONE (0.1 s). Measured on 600 utterances, that
    rounding shifts mean speech/duration by only +0.0013, but it moves individual
    utterances by up to 0.042 -- and per-trial precision is exactly what overlap
    bucketing needs. So we take sample indices and divide ourselves.
    """
    from silero_vad import get_speech_timestamps
    ts = get_speech_timestamps(
        wav, model,
        sampling_rate=sample_rate,
        threshold=cfg["threshold"],
        min_silence_duration_ms=cfg["min_silence_duration_ms"],
        min_speech_duration_ms=cfg["min_speech_duration_ms"],
        speech_pad_ms=cfg["speech_pad_ms"],
        return_seconds=False,
    )
    return [(s["start"] / sample_rate, s["end"] / sample_rate) for s in ts]


# --- pure interval arithmetic (no torch, no audio) ------------------------

def format_segments(segs):
    """[(0.12, 1.83), ...] -> "0.1200:1.8300|...". Four decimals is 0.1 ms at
    16 kHz, finer than any boundary the detector can resolve."""
    return SEG_SEP.join(f"{a:.4f}{BOUND_SEP}{b:.4f}" for a, b in segs)


def parse_segments(text):
    """Inverse of format_segments. Empty string -> [] (an utterance with no
    detected speech, which is rare but real and must not crash a rebuild)."""
    if not text:
        return []
    out = []
    for part in text.split(SEG_SEP):
        a, b = part.split(BOUND_SEP)
        out.append((float(a), float(b)))
    return out


def shift(segs, offset):
    """Segments moved from utterance-local time onto the mixture timeline."""
    return [(a + offset, b + offset) for a, b in segs]


def spans_of(segment_lists, onsets):
    """Every speech span of one speaker on the mixture timeline.

    Replaces build_manifest.spans(), which returned one rectangle per utterance
    running the full file duration. Same call shape, real intervals.
    """
    out = []
    for segs, onset in zip(segment_lists, onsets):
        out.extend(shift(segs, onset))
    return out


def total_speech(segs):
    """Seconds of speech. Assumes segments are disjoint, which Silero guarantees."""
    return sum(b - a for a, b in segs)


def shared_seconds(a, b):
    """Seconds where both span-sets are speaking.

    Identical to build_manifest.shared_seconds; PR2 makes that module import this
    one rather than keep a second copy. Left duplicated for now only because PR1
    must not touch build_manifest.py.
    """
    return sum(max(0.0, min(x2, y2) - max(x1, y1)) for x1, x2 in a for y1, y2 in b)


def onsets_of(segment_lists, onsets, first_only=True):
    """The moments this speaker STARTS talking, on the mixture timeline.

    `first_only=True` (B13 option A, chosen 2026-08-15) takes the first speech
    onset of each utterance: the moment they begin that turn. It is the minimal
    correction to the existing definition, which used each utterance's FILE onset
    -- same number of events, corrected by the ~0.331 s of leading silence.

    `first_only=False` (option B) takes every speech-segment onset, so an
    interferer resuming after their own breath pause counts as starting to talk.
    That reading raised the measured interruption rate from 0.570 to 0.725 purely
    by widening the definition, which is why it was not chosen.
    """
    out = []
    for segs, onset in zip(segment_lists, onsets):
        if not segs:
            continue
        picked = segs[:1] if first_only else segs
        out.extend(a + onset for a, _ in picked)
    return out


def is_interrupted(target_spans, interferer_onsets):
    """B13's interruption condition: the interferer begins a turn while the target
    is genuinely mid-utterance.

    Strictly inside, so beginning exactly as the target stops is turn-taking.
    Unchanged in form from build_manifest.is_interrupted -- what changes is that
    both arguments now come from detected speech rather than file boundaries.
    decisions-m0.md 2026-08-14 (definition), 2026-08-15 (option A).
    """
    return any(t0 < o < t1 for o in interferer_onsets for t0, t1 in target_spans)
