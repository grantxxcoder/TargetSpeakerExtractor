"""The validation word error rate computed inside the training loop, which chose both final checkpoints and stepped their learning rate (decisions-m2.md 2026-09-23).

A fixed set of whole validation clips is extracted and transcribed by the offline ASR (faster-whisper small.en, int8, greedy, on the CPU), then scored with the same count_errors as the reported metric.
It is not a loss, nothing backpropagates through it, and it runs on a subset, so it is a training diagnostic, not the reported number.
Word error rate jumps around from epoch to epoch, so the scheduler steps on an exponential moving average of it rather than on the raw value.
"""

import json

import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel

from src.estimates.runner import read_trials
from src.live_model_metric.lcf_wer import count_errors

# The same listener as evaluate.py's default. Changing one without the other would make the training signal and the reported score two different instruments.
ASR_MODEL_SIZE = "small.en"
SAMPLE_RATE = 16000


# this reads a wav file as one mono channel
def read_mono(path):
    audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


# this picks the first limit "both" trials in manifest order that have a mixture, an enrolment and a non-empty target script
def load_probe_trials(manifest_csv, audio_root, limit):
    """The same clips at every epoch of every run, or the curve would measure which trials were drawn. Trials whose target never speaks are skipped, because their rate is 0/0 (decisions-m0.md 2026-08-13)."""
    kept = []
    for trial in read_trials(manifest_csv, audio_root, None, "both"):
        meta_path = trial["directory"] / "meta.json"
        mixture = trial["directory"] / "mixture.wav"
        enrolment = trial["directory"] / "enrollment.wav"
        if not (meta_path.exists() and mixture.exists() and enrolment.exists()):
            continue
        text = json.loads(meta_path.read_text()).get("target_text", "")
        if not str(text).strip():
            continue
        kept.append((trial["trial_id"], mixture, enrolment, str(text)))
        if len(kept) >= limit:
            break
    return kept


# this makes a probe over the given trials. ema_span sets how many epochs the moving average smooths over
def make_content_probe(trials, ema_span):
    if not trials:
        raise ValueError("The content probe got no trials.")
    return {
        "trials": trials,
        "ema_span": max(1, int(ema_span)),
        "asr": None,
        "history": [],     # the raw word error rate of every scored epoch
        "smoothed": [],    # its moving average, which the scheduler and the checkpoint choice read
    }


# this transcribes one clip with the probe's ASR, loading the model on first use
def probe_transcribe(probe, audio):
    if probe["asr"] is None:
        probe["asr"] = WhisperModel(ASR_MODEL_SIZE, device="cpu", compute_type="int8")
    # English, greedy, and each clip on its own, exactly as evaluate.transcribe decodes.
    segments, info = probe["asr"].transcribe(np.asarray(audio, dtype="float32"), language="en", beam_size=1, temperature=0.0, condition_on_previous_text=False)
    texts = []
    for segment in segments:
        texts.append(segment.text.strip())
    return " ".join(texts).strip()


# this scores an extractor on the probe's clips and returns the corpus word error rate, lower is better. extract takes (mixture, enrolment, sample rate) and returns the estimate
def probe_score(probe, extract):
    """A corpus rate, the total errors over the total reference words, the same form compute_lcf_wer reports."""
    errors = 0
    reference_words = 0
    for trial_id, mixture_path, enrolment_path, text in probe["trials"]:
        estimate = extract(read_mono(mixture_path), read_mono(enrolment_path), SAMPLE_RATE)
        counts = count_errors(text, probe_transcribe(probe, estimate))
        if counts["reference_word_count"] == 0:
            continue
        errors += counts["substitutions"] + counts["deletions"] + counts["insertions"]
        reference_words += counts["reference_word_count"]
    if reference_words == 0:
        return float("nan")
    wer = 100.0 * errors / reference_words
    probe["history"].append(wer)
    probe["smoothed"].append(ema(probe["history"], probe["ema_span"]))
    return wer


# this rebuilds the probe's history from an earlier run when training resumes, so the moving average carries on rather than restarting
def probe_warm_start(probe, history):
    """Epochs where the probe did not run are stored as NaN and are dropped."""
    probe["history"] = []
    for value in history:
        if value == value:
            probe["history"].append(float(value))
    probe["smoothed"] = []
    for index in range(1, len(probe["history"]) + 1):
        probe["smoothed"].append(ema(probe["history"][:index], probe["ema_span"]))
    return probe


# this is the exponential moving average of a history, read at its last point, with alpha = 2 / (span + 1)
def ema(history, span):
    alpha = 2.0 / (span + 1.0)
    value = history[0]
    for point in history[1:]:
        value = alpha * point + (1 - alpha) * value
    return value
