"""Validation WORD ERROR RATE, computed inside the training loop.

WHY THIS EXISTS. Until 2026-09-23 nothing in training could see content. The
scheduler, the selector and the early stop all watched signal-domain proxies,
and item 1a proved those proxies do not rank epochs the way content does: the
kept epoch 9 reads WORSE THAN DOING NOTHING (ASR LCF-WER 65.63 against a 65.22
floor) while epoch 15, ranked tenth of fourteen by `present_branch`, reads 52.77.
A run cannot be steered by a number it never computes. decisions-m2.md
2026-09-23.

WHAT IT IS. A fixed, small set of whole validation clips, extracted and then
transcribed by the SAME listener the evaluation harness uses -- faster-whisper
`small.en`, int8, greedy, `condition_on_previous_text=False` -- and scored with
the SAME `count_errors` as `metric-definitions.md` 3.1. Default device is cpu,
which makes the probe byte-identical to the reported instrument. Moving it to
cuda is faster and is NOT the same instrument, so the device is recorded.

WHAT IT IS NOT.

- **Not a loss.** Nothing here is differentiable and nothing backpropagates. It
  is a number the scheduler and the selector read, exactly as they read
  `val_L_pres` today.
- **Not the judge.** The judge cannot go in a training loop: rate limits, cost,
  and a measured test-retest band of +-4.16 points at k=1 on an n=103 aggregate
  (`2026-09-22-judge-retest`), which is wider than most epoch-to-epoch moves.
  The offline ASR correlates r = 0.825 per trial with the judge
  (`2026-09-15-judge-predictability`), which is enough to RANK epochs and not
  enough to calibrate them.
- **Not the reported number.** It runs on a subset, so quote it as a training
  diagnostic and re-score the chosen checkpoint through `scripts/evaluate.py`.

WER IS NOT SMOOTH, AND THAT IS MEASURED, NOT FEARED. The 2026-09-01 mix-back
sweep moved the signal monotonically across five settings while LCF-WER went
65.2, 63.4, 69.6, 67.2, 59.1 -- non-monotonic over a provably linear change,
because a transcriber's behaviour is not a smooth function of audio quality.
A scheduler stepping on a jumpy number cuts the learning rate on noise, so this
module also exposes an exponential moving average and `trend()`. Step the
scheduler on the SMOOTHED value; read the raw one.
"""

import math
from pathlib import Path

import numpy as np
import soundfile as sf

from src.estimates.runner import read_trials
from src.live_model_metric.lcf_wer import count_errors, normalise_text

# Identical to src/live_model_metric/evaluate.py. Changing either without the
# other silently makes the training signal and the reported score two different
# instruments.
ASR_MODEL_SIZE = "small.en"
ASR_DECODE = dict(language="en", beam_size=1, temperature=0.0,
                  condition_on_previous_text=False)


def _read_mono(path):
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    return audio.mean(axis=1) if audio.ndim > 1 else audio


def words_spoken(trial_directory):
    """Words BOTH speakers said in this clip, after the scorer's normaliser."""
    import json
    meta = json.loads((Path(trial_directory) / "meta.json").read_text())
    return sum(len(normalise_text(meta.get(key) or "").split())
               for key in ("target_text", "interferer_text"))


def clip_errors(counts, spoken=None):
    """S + D + I for one clip, optionally capped at the words spoken in it.

    THE LOOP GUARD, added 2026-09-24. `small.en` sometimes falls into a
    repetition loop and writes many times more words than anyone said -- on
    sir0_val-42-000098 it transcribed the target correctly, then repeated one
    clause seven more times (166 words for a 55-word target). One such clip
    moved the 40-clip score by ~12 points, and the same checkpoint on the same
    40 clips read 53.66 on Kaggle and 66.19 locally because the loop fired in
    one place and not the other. decisions-m2.md 2026-09-24.

    WHY THIS CAP AND NOT ANOTHER. The error count is an edit distance, so it
    never exceeds max(reference words, transcript words). The reference is part
    of what was spoken, so a transcript no longer than everything spoken --
    however wrong, including one that is entirely the OTHER speaker -- can
    never reach the cap. It binds ONLY when the transcript holds more words
    than both speakers said together, i.e. on words nobody said. Leakage is
    still charged in full.

    `spoken=None` is the uncapped rule: identical to metric-definitions.md 3.1
    and to the reported LCF-WER.
    """
    errors = counts.substitutions + counts.deletions + counts.insertions
    return errors if spoken is None else min(errors, spoken)


def load_probe_trials(manifest_csv, audio_root, limit=40, condition="both"):
    """A fixed subset of whole clips that have something to transcribe.

    Manifest order, not a random sample: the probe must score the SAME clips at
    every epoch of every run, or the curve measures which trials were drawn.
    Trials whose target never speaks are dropped -- the rate is 0/0 there, not
    perfect (B4, decisions-m0.md 2026-08-13).
    """
    trials = read_trials(manifest_csv, audio_root, limit=None, condition=condition)
    kept = []
    for trial in trials:
        meta_path = trial.directory / "meta.json"
        mixture = trial.directory / "mixture.wav"
        enrolment = trial.directory / "enrollment.wav"
        if not (meta_path.exists() and mixture.exists() and enrolment.exists()):
            continue
        import json
        text = json.loads(meta_path.read_text()).get("target_text", "")
        if not str(text).strip():
            continue
        kept.append((trial.trial_id, mixture, enrolment, str(text)))
        if limit is not None and len(kept) >= limit:
            break
    return kept


class ContentProbe:
    """Scores an extractor on content. `extract` is runner.py's Extractor
    contract: (mixture_1d, enrolment_1d, sample_rate) -> estimate_1d."""

    def __init__(self, trials, sample_rate=16000, model_size=ASR_MODEL_SIZE,
                 device="cpu", compute_type="int8", ema_span=3,
                 transcriber=None, cap_errors_at_spoken=False):
        if not trials:
            raise ValueError("content probe got no trials")
        self.trials = trials
        # OFF by default, so a run started before 2026-09-24 -- or resumed from
        # one -- keeps the rule its history was scored under. A curve must never
        # mix the two rules; the config comparison refuses a resume across it.
        self.cap_errors_at_spoken = bool(cap_errors_at_spoken)
        self._spoken = ({trial_id: words_spoken(Path(mixture).parent)
                         for trial_id, mixture, _enrolment, _text in trials}
                        if self.cap_errors_at_spoken else {})
        self.capped = []                     # (clips capped, errors removed) per epoch
        self.sample_rate = sample_rate
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.ema_span = max(1, int(ema_span))
        self._asr = None
        self._transcriber = transcriber      # injected in tests
        self.history = []                    # raw WER per scored epoch
        self.smoothed = []

    def describe(self):
        return (f"faster-whisper {self.model_size} {self.compute_type} "
                f"{self.device} greedy, {len(self.trials)} clips"
                + (", errors capped at words spoken" if self.cap_errors_at_spoken else ""))

    def _transcribe(self, audio):
        if self._transcriber is not None:
            return self._transcriber(audio)
        if self._asr is None:
            from faster_whisper import WhisperModel
            self._asr = WhisperModel(self.model_size, device=self.device,
                                     compute_type=self.compute_type)
        segments, _ = self._asr.transcribe(np.asarray(audio, dtype="float32"),
                                           **ASR_DECODE)
        return " ".join(s.text.strip() for s in segments).strip()

    def score(self, extract):
        """Corpus LCF-WER over the probe set. Lower is better.

        CORPUS rate -- total errors over total reference words -- matching
        compute_lcf_wer, not a mean of per-trial rates. The two differ whenever
        utterance lengths differ, and the harness reports the corpus form.
        """
        errors = reference_words = 0
        n_capped = removed = 0
        for trial_id, mixture_path, enrolment_path, text in self.trials:
            mixture = _read_mono(mixture_path)
            enrolment = _read_mono(enrolment_path)
            estimate = extract(mixture, enrolment, self.sample_rate)
            counts = count_errors(text, self._transcribe(estimate))
            if counts.reference_word_count == 0:
                continue
            raw = clip_errors(counts)
            kept = clip_errors(counts, self._spoken.get(trial_id))
            if kept < raw:
                n_capped += 1
                removed += raw - kept
            errors += kept
            reference_words += counts.reference_word_count
        if reference_words == 0:
            return float("nan")
        wer = 100.0 * errors / reference_words
        self.history.append(wer)
        self.smoothed.append(self._ema())
        self.capped.append((n_capped, removed))
        return wer

    def warm_start(self, history):
        """Adopt an earlier run's per-epoch WERs, for a resume.

        A 12 h Kaggle session caps a 2,575 s/epoch run at ~16 epochs, so going
        further means resuming -- and the EMA the scheduler reads is a function
        of every epoch so far. Restarting it at the session boundary hands
        ReduceLROnPlateau a step change that is an artefact of where the session
        ended, not of the model. NaNs are dropped: epochs where the probe did
        not run carry no information about the level.
        """
        self.history = [float(v) for v in history if v == v]
        self.smoothed = []
        for index in range(1, len(self.history) + 1):
            kept, self.history = self.history, self.history[:index]
            self.smoothed.append(self._ema())
            self.history = kept
        return self

    def _ema(self):
        alpha = 2.0 / (self.ema_span + 1.0)
        value = self.history[0]
        for point in self.history[1:]:
            value = alpha * point + (1 - alpha) * value
        return value

    def trend(self, window=4):
        """Is the run still improving? Least-squares slope of the last `window`
        SMOOTHED points, in WER points per scored epoch.

        Negative means still getting better, so there is headroom and the run is
        worth continuing. This is the question item 1a could not answer: its
        content was still improving at the last epoch, and nothing in the loop
        was watching.
        """
        points = self.smoothed[-window:]
        if len(points) < 2:
            return float("nan")
        n = len(points)
        xs = list(range(n))
        mean_x = sum(xs) / n
        mean_y = sum(points) / n
        denominator = sum((x - mean_x) ** 2 for x in xs)
        if denominator == 0:
            return float("nan")
        return sum((x - mean_x) * (y - mean_y)
                   for x, y in zip(xs, points)) / denominator

    def verdict(self, window=4, flat=0.25):
        """One line for the epoch report. `flat` is the slope below which the
        curve is called level, in WER points per epoch."""
        slope = self.trend(window)
        if math.isnan(slope):
            return "content: too few points to read a trend"
        best = min(self.history)
        latest = self.history[-1]
        if slope < -flat:
            state = "STILL IMPROVING -- headroom, keep training"
        elif slope > flat:
            state = "GETTING WORSE -- past the useful epochs"
        else:
            state = "FLAT -- no headroom visible"
        # A capped clip is a listener loop, not the model: say so on the line
        # the epoch is read from, so a jump is never mistaken for a regression.
        loop = ""
        if self.capped and self.capped[-1][0]:
            loop = (f" [loop guard: {self.capped[-1][0]} clip(s), "
                    f"{self.capped[-1][1]} invented words not counted]")
        return (f"content: WER {latest:.2f} (best {best:.2f}, "
                f"slope {slope:+.2f}/epoch) {state}{loop}")
