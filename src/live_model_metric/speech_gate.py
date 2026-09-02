"""Does this clip contain speech at all? Decided BEFORE any listener sees it.

WHY THIS EXISTS
---------------
Measured 2026-09-02: fed digital silence (RMS exactly 0), `gemini-3.7-flash`
returns status=speech and fabricates 17-42 words. It survived three prompt
variants, including one that said "Do not hallucinate", and one response came
back in French on an English-only pipeline. That is unconditioned generation
from the language prior, not misheard audio, and no wording fixes it. The
offline ASR has the same failure in milder form: `small.en` emits "you" on
silence in 8 of 8 absent trials.

Established mitigation is to pre-filter with a voice-activity detector rather
than to ask the model: Koenecke et al. / WhisperX-style VAD gating, evaluated
against confidence thresholding and pattern matching in "Investigation of
Whisper ASR Hallucinations Induced by Non-Speech Audio" (arXiv:2501.11378),
which finds VAD and confidence thresholding both give meaningful reductions and
that no single method eliminates the problem. Confidence thresholding is not
available to us: logprobs are missing from the Interactions API.

THE ARGUMENT, not just the workaround. "Did the system emit speech?" is a
SIGNAL question. It never needed a language model. The judge is needed for what
was SAID -- and at that it is excellent (0.0 % on clean targets, byte-identical
five times out of five). So the gate is the right instrument for that question,
not a patch over a broken one.

SYMMETRY IS MANDATORY. The same rule must gate the judge AND the offline ASR.
Gate one and not the other and every judge-vs-ASR difference on a speech-free
clip measures the gate rather than the listeners. `gated()` exists to make the
symmetric use the natural one.

WHAT DECIDES WHAT
-----------------
For the rendered anchors the answer is known by CONSTRUCTION -- no VAD needed,
and no new moving part in the instrument. Measured on sir0_val, 2026-09-02:

    condition          target.wav   interferer.wav   mixture.wav
    both                  speech        speech          speech
    target_only           speech        RMS 0           speech
    interferer_only       RMS 0         speech          speech
    noise_only            RMS 0         RMS 0           noise, NO speech

`estimate.wav` is the only clip whose content is not determined by how the
trial was built, so it is the only one that needs the VAD -- and it is also
where the question matters, because a muting extractor is a real failure mode
(the 2026-08-25 collapse-to-silence run).

This changes metric-definitions.md 3.1, which sends every clip to the judge.
Recorded in decisions-m4.md 2026-09-02.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_LOG = REPO_ROOT / "experiments/results/speech_gate.csv"

LOG_FIELDS = ["run_date", "split", "trial_id", "file", "condition", "listener",
              "has_speech", "reason", "speech_s"]

# Minimum detected speech for an estimate to count as containing speech. 0.10 s
# is roughly one short syllable: below that there is nothing a listener could
# transcribe, and Silero's own frame resolution makes anything smaller noise.
DEFAULT_MIN_SPEECH_S = 0.10

# Which stems carry speech, per condition, BY CONSTRUCTION. Verified against
# measured RMS on sir0_val, 2026-09-02 -- see the module docstring.
_TARGET_SPEAKS = {"both", "target_only"}
_INTERFERER_SPEAKS = {"both", "interferer_only"}


@dataclass
class GateDecision:
    has_speech: bool
    reason: str
    speech_s: float = None

    @property
    def fired(self):
        """True when the gate BLOCKED the clip, i.e. found no speech."""
        return not self.has_speech


def decide(audio_path, condition=None, min_speech_s=DEFAULT_MIN_SPEECH_S,
           vad_detect=None):
    """Does this clip contain speech?

    `condition` is the manifest's condition for the trial. When it is known and
    the clip is a rendered anchor, the answer comes from construction and no VAD
    runs. `vad_detect` is a callable (path) -> seconds of detected speech, used
    only for estimate.wav; omit it and an estimate is passed through untouched
    rather than silently blocked, because blocking on a missing detector would
    fabricate a measurement.
    """
    stem = Path(audio_path).stem.lower()

    if condition:
        if stem == "target":
            speaks = condition in _TARGET_SPEAKS
            return GateDecision(speaks, f"construction:{condition}:target")
        if stem == "interferer":
            speaks = condition in _INTERFERER_SPEAKS
            return GateDecision(speaks, f"construction:{condition}:interferer")
        if stem == "mixture":
            # Any speaker at all. noise_only has real energy but no speech,
            # which makes it a more realistic hallucination probe than silence.
            speaks = condition != "noise_only"
            return GateDecision(speaks, f"construction:{condition}:mixture")

    if stem == "estimate":
        if vad_detect is None:
            return GateDecision(True, "estimate:no-vad-supplied:passed-through")
        speech_s = float(vad_detect(audio_path))
        return GateDecision(speech_s >= min_speech_s,
                            f"vad:{'speech' if speech_s >= min_speech_s else 'no-speech'}",
                            speech_s)

    # Unknown clip and no condition: never block on ignorance.
    return GateDecision(True, "unknown-clip:passed-through")


def log_decision(decision, audio_path, listener, split=None, condition=None,
                 log_path=None):
    """Append one row. EVERY decision is logged, not only the blocks, so the
    denominator is recoverable and a gate that never fires is visible as such.

    A gate firing on a target-PRESENT trial is not a measurement error -- it is
    a finding that the system destroyed the speech. It has to be auditable.
    """
    log_path = Path(log_path or GATE_LOG)
    path = Path(audio_path)
    exists = log_path.exists()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({
            "run_date": date.today().isoformat(),
            "split": split or "",
            "trial_id": path.parent.name,
            "file": path.name,
            "condition": condition or "",
            "listener": listener,
            "has_speech": int(decision.has_speech),
            "reason": decision.reason,
            "speech_s": "" if decision.speech_s is None else f"{decision.speech_s:.3f}",
        })


def gated(transcribe_one, listener, condition_of=None, split=None,
          min_speech_s=DEFAULT_MIN_SPEECH_S, vad_detect=None, log_path=None,
          verbose=False):
    """Wrap a `transcribe_one` so speech-free clips are answered locally.

    Returns a callable with the same (path) -> text signature, so it drops into
    lcf_wer / icr / nrr exactly like the ungated one. Wrap BOTH listeners with
    this or the comparison between them is not a comparison.

        judge_gated = gated(judge, "judge", condition_of=lookup)
        asr_gated   = gated(asr,   "small.en", condition_of=lookup)

    A blocked clip returns "" -- the empty hypothesis, which jiwer scores as
    all-deletions. That is metric-definitions.md 3.1's stated treatment of a
    listener that reported nothing, so no new scoring rule is introduced.
    """
    def inner(audio_path):
        condition = condition_of(audio_path) if condition_of else None
        decision = decide(audio_path, condition=condition,
                          min_speech_s=min_speech_s, vad_detect=vad_detect)
        log_decision(decision, audio_path, listener, split=split,
                     condition=condition, log_path=log_path)
        if decision.fired:
            if verbose:
                print(f"    gate blocked {Path(audio_path).parent.name}/"
                      f"{Path(audio_path).name} ({decision.reason})", flush=True)
            return ""
        return transcribe_one(audio_path)
    return inner


def condition_lookup(split, manifest_dir="data/manifests", repo_root=None):
    """(path) -> condition, read from the split manifest by trial id."""
    root = Path(repo_root or REPO_ROOT)
    manifest = root / manifest_dir / f"{split}.csv"
    with open(manifest, newline="") as handle:
        conditions = {r["trial_id"]: r["condition"] for r in csv.DictReader(handle)}
    return lambda path: conditions.get(Path(path).parent.name)


def summarise(log_path=None):
    """Rows blocked / total, per listener and reason. For the write-up."""
    log_path = Path(log_path or GATE_LOG)
    if not log_path.exists():
        return {}
    out = {}
    with open(log_path, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["listener"], row["reason"])
            total, blocked = out.get(key, (0, 0))
            out[key] = (total + 1, blocked + (row["has_speech"] == "0"))
    return out


GENERATOR_CONFIG = REPO_ROOT / "experiments/configs/generator.yaml"


def vad_seconds_fn(config_path=None):
    """Silero-backed (path) -> seconds of detected speech, for estimates.

    Built lazily and returned as a closure so the model loads once. Silero VAD
    6.2.1, pinned under B2 and already used to measure overlap_ratio from
    detected speech rather than from file boundaries.

    THE SAME `vad:` BLOCK AS THE RENDERER, read from generator.yaml. B2 requires
    every knob to be stated explicitly rather than defaulted, because the
    definition of "speech" is quoted in the thesis next to every figure that
    depends on it -- and the gate's verdict is now one of those figures. Using
    Silero's defaults here would let the gate and the overlap measurements
    disagree about what speech is, silently.
    """
    import soundfile
    import yaml

    from src.data import vad

    path = Path(config_path or GENERATOR_CONFIG)
    with open(path) as handle:
        config = yaml.safe_load(handle)

    model = vad.load_model()
    cfg = vad.vad_config(config)

    def seconds(audio_path):
        wav, sample_rate = soundfile.read(str(audio_path), dtype="float32")
        segments = vad.detect(wav, model, cfg, sample_rate=sample_rate)
        return vad.total_speech(segments)

    return seconds
