"""Finds where the speech is inside each recording (B2, decisions-m0.md 2026-08-15).

The detector is Silero VAD, used unmodified (Silero Team, 2021, "Silero VAD: pre-trained enterprise-grade Voice Activity Detector", https://github.com/snakers4/silero-vad).
A LibriSpeech file is only about 86 % speech, so overlap and activity are measured from the detected speech rather than from the file's start and end, which overstated overlap by about 25 %.
The audio itself is never changed. Mixtures keep their pauses, and only the measurement uses this speech map.
"""

from importlib.metadata import version

from silero_vad import get_speech_timestamps, load_silero_vad

# A segment list is written as "start:end|start:end". The colon is used inside a segment because utterance ids already contain hyphens.
SEGMENT_SEPARATOR = "|"
BOUND_SEPARATOR = ":"

# The Silero model, loaded once per process and kept here.
model_cache = {}


# this loads the Silero model from the pip package, which pins the weights, rather than from torch.hub, which fetches whatever is on GitHub that day
def load_model():
    if "silero" not in model_cache:
        model_cache["silero"] = load_silero_vad()
    return model_cache["silero"]


# this gives the installed silero-vad version, which is recorded beside every cache built with it
def model_version():
    return version("silero-vad")


# this reads the vad block of generator.yaml, which defines what counts as speech everywhere in the project
def vad_config(config):
    """The installed weights must be the pinned version, because a silent upgrade would move every overlap figure."""
    settings = config["vad"]
    expected_version = settings.get("expected_model_version")
    if expected_version and expected_version != model_version():
        raise RuntimeError(f"silero-vad is {model_version()}, but generator.yaml expects {expected_version}. Every overlap figure depends on these weights, so pin the install back or rebuild the VAD index and the manifests.")
    return {"threshold": settings["threshold"], "min_silence_duration_ms": settings["min_silence_duration_ms"], "min_speech_duration_ms": settings["min_speech_duration_ms"], "speech_pad_ms": settings["speech_pad_ms"]}


# this finds the speech segments in one waveform, as a list of (start seconds, end seconds)
def detect(waveform, model, settings, sample_rate):
    """Silero's own seconds output rounds to 0.1 s, which moves single utterances by up to 0.042, so the sample indices are divided here instead."""
    timestamps = get_speech_timestamps(waveform, model, sampling_rate=sample_rate, threshold=settings["threshold"], min_silence_duration_ms=settings["min_silence_duration_ms"], min_speech_duration_ms=settings["min_speech_duration_ms"], speech_pad_ms=settings["speech_pad_ms"], return_seconds=False)
    segments = []
    for timestamp in timestamps:
        segments.append((timestamp["start"] / sample_rate, timestamp["end"] / sample_rate))
    return segments


# this writes segments as text with four decimals, which is 0.1 ms and finer than the detector can resolve
def format_segments(segments):
    parts = []
    for start, end in segments:
        parts.append(f"{start:.4f}{BOUND_SEPARATOR}{end:.4f}")
    return SEGMENT_SEPARATOR.join(parts)


# this reads segments back from text. An empty string is an utterance with no detected speech
def parse_segments(text):
    if not text:
        return []
    segments = []
    for part in text.split(SEGMENT_SEPARATOR):
        start, end = part.split(BOUND_SEPARATOR)
        segments.append((float(start), float(end)))
    return segments


# this moves segments from an utterance's own time onto the mixture timeline
def shift(segments, offset):
    shifted = []
    for start, end in segments:
        shifted.append((start + offset, end + offset))
    return shifted


# this gives every speech span of one speaker on the mixture timeline, from each utterance's segments and its onset
def spans_of(segment_lists, onsets):
    spans = []
    for segments, onset in zip(segment_lists, onsets):
        spans.extend(shift(segments, onset))
    return spans


# this gives the seconds of speech in a list of segments, which Silero guarantees do not overlap
def total_speech(segments):
    lengths = []
    for start, end in segments:
        lengths.append(end - start)
    return sum(lengths)


# this gives the seconds where both speakers are talking at once
def shared_seconds(first_spans, second_spans):
    overlaps = []
    for first_start, first_end in first_spans:
        for second_start, second_end in second_spans:
            overlaps.append(max(0.0, min(first_end, second_end) - max(first_start, second_start)))
    return sum(overlaps)


# this gives the moment each utterance's first speech starts, on the mixture timeline
def onsets_of(segment_lists, onsets):
    """Only the first speech onset of each utterance counts as starting a turn (B13 option A, decisions-m0.md 2026-08-15)."""
    turn_onsets = []
    for segments, onset in zip(segment_lists, onsets):
        if segments:
            turn_onsets.append(segments[0][0] + onset)
    return turn_onsets


# this decides whether the interferer starts a turn while the target is mid-utterance
def is_interrupted(target_spans, interferer_onsets):
    """Strictly inside a span, so starting exactly as the target stops is turn-taking (decisions-m0.md 2026-08-14)."""
    for onset in interferer_onsets:
        for span_start, span_end in target_spans:
            if span_start < onset < span_end:
                return True
    return False
