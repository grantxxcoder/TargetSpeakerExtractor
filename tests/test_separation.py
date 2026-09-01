"""Tests for the SDR / SIR / SAR decomposition (src/live_model_metric/separation.py).

Each test names the property it protects. The decomposition is only meaningful
if the three parts really are what they claim to be, so most of these check
signals whose correct answer is known by construction.
"""

import numpy as np
import pytest

from src.live_model_metric.separation import (CEILING_DB, TAU, decompose,
                                              improvement_over_mixture)

SAMPLES = 16000


def sources(seed=42):
    """Three uncorrelated sources plus the mixture they sum to."""
    rng = np.random.default_rng(seed)
    target = rng.standard_normal(SAMPLES)
    interferer = rng.standard_normal(SAMPLES)
    noise = 0.3 * rng.standard_normal(SAMPLES)
    return target, interferer, noise, target + interferer + noise


# --- the property the whole decomposition rests on -----------------------

def test_the_mixture_contains_no_artefact():
    """THE key correctness check. The mixture IS exactly the sum of the sources,
    so no scaled combination of them can leave anything unexplained. If this
    fails, the projection is wrong and every artefact figure is meaningless."""
    target, interferer, noise, mixture = sources()
    scores = decompose(mixture, target, interferer, noise)
    assert scores.signal_to_artefact_db == pytest.approx(CEILING_DB, abs=1e-6)


def test_the_mixture_does_show_interference():
    """The same signal that has zero artefact must still be penalised for
    containing the interferer -- otherwise the two piles are not separated."""
    target, interferer, noise, mixture = sources()
    scores = decompose(mixture, target, interferer, noise)
    assert scores.signal_to_interference_db < 5.0


def test_a_perfect_estimate_saturates_all_three():
    target, interferer, noise, _ = sources()
    scores = decompose(target, target, interferer, noise)
    for value in (scores.signal_to_distortion_db, scores.signal_to_interference_db,
                  scores.signal_to_artefact_db):
        assert value == pytest.approx(CEILING_DB, abs=1e-6)


def test_pure_invention_is_all_artefact():
    """A signal unrelated to anything in the room cannot be explained by the
    sources, so it must land in the artefact pile, not the interference pile."""
    target, interferer, noise, _ = sources()
    invented = np.random.default_rng(7).standard_normal(SAMPLES)
    scores = decompose(invented, target, interferer, noise)
    assert scores.signal_to_artefact_db < -20.0


def test_the_wrong_speaker_is_interference_not_artefact():
    """The interferer WAS in the room. Reporting it is a failure to remove
    something, not a failure to avoid inventing something -- so it must hit SIR
    and leave SAR clean. This is the distinction plain subtraction cannot make."""
    target, interferer, noise, _ = sources()
    scores = decompose(interferer, target, interferer, noise)
    assert scores.signal_to_interference_db < -20.0
    assert scores.signal_to_artefact_db == pytest.approx(CEILING_DB, abs=1e-6)


def test_partial_suppression_is_counted_as_interference_not_artefact():
    """An estimate that attenuates the interferer to 30 % rather than removing
    it has that residue explained by the interferer source, so it is
    interference. Subtracting the sources outright would miscount it."""
    target, interferer, noise, _ = sources()
    scores = decompose(target + 0.3 * interferer, target, interferer, noise)
    assert 5.0 < scores.signal_to_interference_db < 20.0
    assert scores.signal_to_artefact_db == pytest.approx(CEILING_DB, abs=1e-6)


def test_suppression_and_artefact_are_separated():
    """The diagnostic case: good suppression AND real artefact at once. SDR is
    the net and hides which moved, which is the whole reason for the split."""
    target, interferer, noise, _ = sources()
    invented = 0.2 * np.random.default_rng(9).standard_normal(SAMPLES)
    scores = decompose(target + 0.05 * interferer + invented,
                       target, interferer, noise)
    assert scores.signal_to_interference_db > 15.0          # suppressed well
    assert scores.signal_to_artefact_db < 20.0              # but invented plenty
    assert (scores.signal_to_distortion_db
            < scores.signal_to_interference_db)             # SDR hides the win


# --- scale invariance ---------------------------------------------------

def test_all_three_scores_are_scale_invariant():
    """The training objective is scale-invariant, so these must be too or they
    are not commensurable with it. An absolute epsilon in the denominator broke
    this: SAR swung 17 dB under a x7.5 gain because the floor stopped scaling
    with the signal. TAU floors the denominator RELATIVE to the numerator."""
    target, interferer, noise, _ = sources()
    estimate = (target + 0.05 * interferer
                + 0.2 * np.random.default_rng(9).standard_normal(SAMPLES))
    reference = decompose(estimate, target, interferer, noise)
    for gain in (7.5, 0.01, 1000.0):
        scaled = decompose(gain * estimate, target, interferer, noise)
        assert scaled.signal_to_distortion_db == pytest.approx(
            reference.signal_to_distortion_db, abs=1e-9)
        assert scaled.signal_to_interference_db == pytest.approx(
            reference.signal_to_interference_db, abs=1e-9)
        assert scaled.signal_to_artefact_db == pytest.approx(
            reference.signal_to_artefact_db, abs=1e-9)


def test_nothing_can_exceed_the_ceiling():
    """TAU caps every score at 10*log10(1/TAU). Without a cap, an artefact-free
    estimate returns a number governed by floating-point noise, which then
    poisons any average taken over trials."""
    assert CEILING_DB == pytest.approx(10.0 * np.log10(1.0 / TAU))
    target, interferer, noise, _ = sources()
    for estimate in (target, 1e6 * target, 1e-6 * target):
        scores = decompose(estimate, target, interferer, noise)
        assert scores.signal_to_artefact_db <= CEILING_DB + 1e-9
        assert scores.signal_to_interference_db <= CEILING_DB + 1e-9


# --- mechanics ----------------------------------------------------------

def test_signals_of_different_lengths_are_truncated_not_rejected():
    """The renderer pads the tail by the room's decay time, and a whole-clip
    estimate can differ from its sources by a few samples."""
    target, interferer, noise, _ = sources()
    scores = decompose(target[:SAMPLES - 37], target, interferer, noise)
    assert scores.signal_to_artefact_db == pytest.approx(CEILING_DB, abs=1e-6)


def test_improvement_is_reported_relative_to_doing_nothing():
    """An absolute SIR depends on the trial's own signal-to-interference ratio,
    which varies by construction. What a system can be judged on is what it
    added, so the mixture is the reference point."""
    target, interferer, noise, mixture = sources()
    unchanged = improvement_over_mixture(mixture, mixture, target, interferer, noise)
    assert unchanged.signal_to_distortion_db == pytest.approx(0.0, abs=1e-9)
    assert unchanged.signal_to_interference_db == pytest.approx(0.0, abs=1e-9)
    assert unchanged.signal_to_artefact_db == pytest.approx(0.0, abs=1e-9)

    better = improvement_over_mixture(target, mixture, target, interferer, noise)
    assert better.signal_to_interference_db > 20.0


def test_removing_the_interferer_costs_artefact_when_the_estimate_is_imperfect():
    """The trade the metric exists to expose: as suppression gets more
    aggressive SIR rises and SAR falls, and only the split shows both."""
    target, interferer, noise, mixture = sources()
    invented = np.random.default_rng(11).standard_normal(SAMPLES)
    gentle = decompose(target + 0.5 * interferer + 0.02 * invented,
                       target, interferer, noise)
    aggressive = decompose(target + 0.02 * interferer + 0.30 * invented,
                           target, interferer, noise)
    assert aggressive.signal_to_interference_db > gentle.signal_to_interference_db
    assert aggressive.signal_to_artefact_db < gentle.signal_to_artefact_db
