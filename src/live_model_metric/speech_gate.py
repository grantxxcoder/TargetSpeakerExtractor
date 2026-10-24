"""Does a clip contain speech at all? This is decided before any listener hears it.

Fed digital silence, the judge still reports 17 to 42 invented words, and small.en says "you" (measured 2026-09-02). No prompt wording fixed it.
Whether a clip holds speech is a signal question, so a voice-activity detector answers it instead, the established fix ("Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio", arXiv:2501.11378).
The same rule gates the judge and the offline ASR, or every difference between them on a silent clip would measure the gate rather than the listeners.

For the rendered clips the answer is known from how the trial was built, so no detector runs:

    condition          target.wav   interferer.wav   mixture.wav
    both                  speech        speech          speech
    target_only           speech        silent          speech
    interferer_only       silent        speech          speech
    noise_only            silent        silent          noise, no speech

Only estimate.wav, the model's output, needs Silero VAD. Recorded in decisions-m4.md 2026-09-02.
A decision is a plain dictionary: has_speech, reason, speech_s, and fired, which is True when the clip was blocked.
"""

import csv
from datetime import date
from pathlib import Path

import soundfile
import yaml

from src.data import vad

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_LOG = REPO_ROOT / "experiments/results/speech_gate.csv"
GENERATOR_CONFIG = REPO_ROOT / "experiments/configs/generator.yaml"

LOG_FIELDS = ["run_date", "split", "trial_id", "file", "condition", "listener", "has_speech", "reason", "speech_s"]

# The least detected speech for an estimate to count as speech. 0.10 s is about one short syllable, and Silero cannot resolve anything shorter.
MIN_SPEECH_S = 0.10

# Which rendered stems carry speech in each condition, checked against their measured RMS on sir0_val (2026-09-02).
TARGET_SPEAKS = {"both", "target_only"}
INTERFERER_SPEAKS = {"both", "interferer_only"}


# this builds one gate decision. fired is True when the gate blocked the clip
def gate_decision(has_speech, reason, speech_s=None):
    return {"has_speech": has_speech, "reason": reason, "speech_s": speech_s, "fired": not has_speech}


# this decides whether a clip holds speech: from the trial's condition for rendered clips, and from the VAD for an estimate
def decide(audio_path, condition, vad_detect):
    """vad_detect gives the seconds of speech in a file. Without one, an estimate is passed through rather than blocked, because blocking on a missing detector would invent a measurement."""
    stem = Path(audio_path).stem.lower()
    if condition:
        if stem == "target":
            return gate_decision(condition in TARGET_SPEAKS, f"construction:{condition}:target")
        if stem == "interferer":
            return gate_decision(condition in INTERFERER_SPEAKS, f"construction:{condition}:interferer")
        if stem == "mixture":
            # noise_only has real energy but no speech, which makes it a more realistic test of invention than silence.
            return gate_decision(condition != "noise_only", f"construction:{condition}:mixture")
    if stem == "estimate":
        if vad_detect is None:
            return gate_decision(True, "estimate:no-vad-supplied:passed-through")
        speech_s = float(vad_detect(audio_path))
        if speech_s >= MIN_SPEECH_S:
            return gate_decision(True, "vad:speech", speech_s)
        return gate_decision(False, "vad:no-speech", speech_s)
    # An unknown clip with no condition is never blocked.
    return gate_decision(True, "unknown-clip:passed-through")


# this appends one decision to the gate log. Every decision is logged, not only the blocks, so the denominator can always be recovered
def log_decision(decision, audio_path, listener, split, condition):
    """A block on a trial where the target spoke is not a measurement error: it is a finding that the system destroyed the speech, so it must be in the log."""
    path = Path(audio_path)
    exists = GATE_LOG.exists()
    GATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(GATE_LOG, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        speech_s = "" if decision["speech_s"] is None else f"{decision['speech_s']:.3f}"
        writer.writerow({"run_date": date.today().isoformat(), "split": split or "", "trial_id": path.parent.name, "file": path.name, "condition": condition or "", "listener": listener, "has_speech": int(decision["has_speech"]), "reason": decision["reason"], "speech_s": speech_s})


# this gives a function that looks up a clip's condition from the split manifest, by its trial folder
def condition_lookup(split, manifest_dir, repo_root):
    root = Path(repo_root or REPO_ROOT)
    manifest = root / manifest_dir / f"{split}.csv"
    conditions = {}
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            conditions[row["trial_id"]] = row["condition"]

    def lookup(path):
        return conditions.get(Path(path).parent.name)
    return lookup


# this gives a function that measures the seconds of speech in a file with Silero VAD, using the same settings the renderer used
def vad_seconds_fn():
    """The settings come from the vad block of generator.yaml, so the gate and the overlap measurements agree on what speech is. The model loads once."""
    with open(GENERATOR_CONFIG) as handle:
        config = yaml.safe_load(handle)
    model = vad.load_model()
    vad_settings = vad.vad_config(config)

    def seconds(audio_path):
        waveform, sample_rate = soundfile.read(str(audio_path), dtype="float32")
        segments = vad.detect(waveform, model, vad_settings, sample_rate=sample_rate)
        return vad.total_speech(segments)
    return seconds
