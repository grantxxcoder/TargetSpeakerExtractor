"""The in-loop validation WER probe (src/live_model_metric/content_probe.py).

The transcriber is injected, so these run without faster-whisper and without a
GPU. The audio and the reference texts are the REAL rendered sir0_val trials, so
file reading, text lookup and the corpus-rate arithmetic are all exercised.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.live_model_metric.content_probe import (  # noqa: E402
    ContentProbe, load_probe_trials)

MANIFEST = ROOT / "data/manifests/sir0_val.csv"
AUDIO = ROOT / "data/rendered/sir0_val"
pytestmark = pytest.mark.skipif(
    not (MANIFEST.exists() and AUDIO.exists()),
    reason="rendered sir0_val is not on this machine")


def probe_trials(n=4):
    return load_probe_trials(MANIFEST, AUDIO, limit=n, condition="both")


def passthrough(mixture, enrolment, sample_rate):
    return mixture


def test_probe_trials_are_fixed_and_have_text():
    first, second = probe_trials(4), probe_trials(4)
    assert [t[0] for t in first] == [t[0] for t in second], "probe set must not vary"
    assert len(first) == 4
    for _trial_id, mixture, enrolment, text in first:
        assert mixture.exists() and enrolment.exists()
        assert text.strip(), "a target-silent trial reached the probe"


def test_a_perfect_listener_scores_zero():
    trials = probe_trials(3)
    texts = iter([t[3] for t in trials])
    probe = ContentProbe(trials, transcriber=lambda audio: next(texts))
    assert probe.score(passthrough) == pytest.approx(0.0)


def test_silence_scores_one_hundred_not_zero():
    """A muting extractor recovers nothing, so every reference word deletes."""
    trials = probe_trials(3)
    probe = ContentProbe(trials, transcriber=lambda audio: "")
    assert probe.score(passthrough) == pytest.approx(100.0)


def test_it_is_a_corpus_rate_not_a_mean_of_rates():
    """Long utterances must carry more weight, as compute_lcf_wer does."""
    trials = probe_trials(2)
    lengths = [len(t[3].split()) for t in trials]
    if lengths[0] == lengths[1]:
        pytest.skip("need two different reference lengths to tell the two apart")
    # Perfect on the first trial, nothing on the second.
    answers = iter([trials[0][3], ""])
    probe = ContentProbe(trials, transcriber=lambda audio: next(answers))
    corpus = probe.score(passthrough)
    mean_of_rates = (0.0 + 100.0) / 2
    expected = 100.0 * lengths[1] / (lengths[0] + lengths[1])
    assert corpus == pytest.approx(expected)
    assert corpus != pytest.approx(mean_of_rates)


def test_trend_is_negative_while_improving():
    probe = ContentProbe(probe_trials(1), transcriber=lambda audio: "")
    probe.history = [70.0, 65.0, 60.0, 55.0]
    probe.smoothed = probe.history[:]
    assert probe.trend() < 0
    assert "STILL IMPROVING" in probe.verdict()


def test_trend_is_flat_when_it_stops_moving():
    probe = ContentProbe(probe_trials(1), transcriber=lambda audio: "")
    probe.history = [55.0, 55.05, 54.95, 55.0]
    probe.smoothed = probe.history[:]
    assert abs(probe.trend()) < 0.25
    assert "FLAT" in probe.verdict()


def test_trend_flags_a_run_going_backwards():
    probe = ContentProbe(probe_trials(1), transcriber=lambda audio: "")
    probe.history = [55.0, 58.0, 61.0, 64.0]
    probe.smoothed = probe.history[:]
    assert probe.trend() > 0
    assert "GETTING WORSE" in probe.verdict()


def test_smoothing_damps_a_single_spike():
    """WER is non-monotonic under smooth signal changes (2026-09-01 mix-back
    sweep), so the scheduler must not see one bad epoch as a plateau."""
    trials = probe_trials(1)
    answers = iter([trials[0][3], "", trials[0][3]])
    probe = ContentProbe(trials, ema_span=3, transcriber=lambda audio: next(answers))
    for _ in range(3):
        probe.score(passthrough)
    raw_swing = max(probe.history) - min(probe.history)
    smoothed_swing = max(probe.smoothed) - min(probe.smoothed)
    assert smoothed_swing < raw_swing


def test_warm_start_rebuilds_the_ema_across_a_resume():
    """A 12 h Kaggle cap puts 1c at ~14 epochs, so going further means resuming.
    A probe that restarts its EMA hands ReduceLROnPlateau a step change that is
    an artefact of where the session ended."""
    warm = ContentProbe(probe_trials(1), ema_span=3, transcriber=lambda a: "")
    warm.warm_start([70.0, 65.0, 60.0, 55.0])
    assert warm.history == [70.0, 65.0, 60.0, 55.0]
    assert len(warm.smoothed) == 4
    assert warm.smoothed[0] == pytest.approx(70.0)
    # strictly falling, and below the raw value it is tracking
    assert warm.smoothed[-1] < warm.smoothed[0]
    assert warm.trend() < 0
    assert "STILL IMPROVING" in warm.verdict()


def test_warm_start_drops_epochs_where_the_probe_did_not_run():
    """NaN means the probe was skipped that epoch, not that WER was zero."""
    warm = ContentProbe(probe_trials(1), transcriber=lambda a: "")
    warm.warm_start([70.0, float("nan"), 60.0])
    assert warm.history == [70.0, 60.0]


def test_warm_start_then_score_continues_the_series():
    trials = probe_trials(1)
    warm = ContentProbe(trials, ema_span=3, transcriber=lambda a: "")
    warm.warm_start([70.0, 65.0, 60.0])
    before = len(warm.history)
    warm.score(passthrough)              # a muting listener -> 100.0
    assert len(warm.history) == before + 1
    assert warm.history[-1] == pytest.approx(100.0)
    # the EMA absorbs the spike rather than jumping to it
    assert warm.smoothed[-1] < 100.0


def test_an_empty_warm_start_is_a_cold_probe():
    warm = ContentProbe(probe_trials(1), transcriber=lambda a: "")
    warm.warm_start([])
    assert warm.history == [] and warm.smoothed == []
