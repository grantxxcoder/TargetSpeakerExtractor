"""FR, the fabrication rate. docs/data/metric-definitions.md 3.3b."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.live_model_metric.fabrication import (  # noqa: E402
    HEADLINE_K, compute_fr, invented_content, measure_fabrication,
)

TARGET = "the quick brown fox jumped over the lazy dog"
INTERFERER = "a sailor telephoned his mother from harbour"


def test_echoing_the_target_invents_nothing():
    assert invented_content(TARGET, TARGET, INTERFERER) == set()


def test_echoing_the_interferer_invents_nothing():
    """Leakage is ICR's measurement, not FR's. The words WERE said."""
    assert invented_content(INTERFERER, TARGET, INTERFERER) == set()


def test_a_word_neither_speaker_said_is_invented():
    assert invented_content("the fox jumped into a helicopter", TARGET, INTERFERER) \
        == {"helicopter"}


def test_stopwords_do_not_count_as_invented():
    """"into" and "through" are NLTK stopwords; only content words can be
    fabricated, so a response made entirely of function words invents nothing."""
    assert invented_content("into through about", TARGET, INTERFERER) == set()


def test_target_only_trial_has_no_interferer_text():
    assert invented_content("fox helicopter", TARGET, "") == {"helicopter"}


def test_absent_target_trial_has_no_target_text():
    assert invented_content("sailor helicopter", "", INTERFERER) == {"helicopter"}


def test_fraction_denominator_is_the_response_not_the_scripts():
    f = measure_fabrication("fox helicopter", TARGET, INTERFERER)
    assert f.invented_count == 1 and f.response_count == 2
    assert f.invented_fraction == pytest.approx(0.5)


def test_every_trial_is_eligible_at_every_k():
    """Unlike ICR, no per-k eligibility: anything can be invented, so the
    denominator is the same at every k."""
    r = compute_fr(["helicopter"] * 4, [TARGET] * 4, [INTERFERER] * 4)
    assert r.trials_scored == 4
    assert r.fr_at_k[1] == pytest.approx(100.0)
    assert r.fr_at_k[2] == pytest.approx(0.0)     # only one invented word each


def test_k_threshold_fires_on_the_second_invented_word():
    r = compute_fr(["helicopter submarine"], [TARGET], [INTERFERER])
    assert r.fr_at_k[2] == pytest.approx(100.0)
    assert r.headline == r.fr_at_k[HEADLINE_K]


def test_an_empty_response_is_excluded_not_scored_clean():
    """Scoring silence as 0 % invented would let a muting system lower its
    fabrication rate by saying nothing. That is NRR's measurement."""
    r = compute_fr(["", "helicopter submarine"], [TARGET] * 2, [INTERFERER] * 2)
    assert r.trials_empty == 1
    assert r.trials_scored == 1
    assert r.fr_at_k[2] == pytest.approx(100.0)   # not 50 %


def test_a_perfect_transcript_scores_zero():
    r = compute_fr([TARGET], [TARGET], [INTERFERER])
    assert r.fr_at_k[1] == pytest.approx(0.0)
    assert r.mean_invented_percent == pytest.approx(0.0)


def test_short_responses_are_kept_out_of_the_mean_fraction():
    """One invented word in a one-word response is not 100 % fabrication."""
    r = compute_fr(["helicopter"], [TARGET], [INTERFERER])
    assert r.trials_for_fraction == 0
    assert r.mean_invented_percent is None
    assert r.fr_at_k[1] == pytest.approx(100.0)   # the @k reading still works


def test_per_trial_count_is_immune_to_response_length():
    """The length trap that the percentage falls into, guarded.

    Two systems inventing the same words, one terser. `invented_per_trial` must
    call them equal; `mean_invented_percent` must not be used to compare them.
    Measured 2026-09-03: baseline and WeSep both invent ~1.8 words per trial but
    read 10.4 % and 13.8 % purely because WeSep says less.
    """
    wordy = "helicopter submarine fox jumped over lazy dog quick brown"
    terse = "helicopter submarine fox jumped dog"   # >=5 words, so the % is defined
    a = compute_fr([wordy], [TARGET], [INTERFERER])
    b = compute_fr([terse], [TARGET], [INTERFERER])
    assert a.invented_per_trial == b.invented_per_trial == pytest.approx(2.0)
    assert b.mean_invented_percent > a.mean_invented_percent


def test_per_trial_count_ignores_empty_responses_in_its_denominator():
    r = compute_fr(["", "helicopter submarine"], [TARGET] * 2, [INTERFERER] * 2)
    assert r.invented_per_trial == pytest.approx(2.0)   # not 1.0


def test_mismatched_lengths_are_refused():
    with pytest.raises(ValueError, match="parallel"):
        compute_fr(["a", "b"], [TARGET], [INTERFERER])


def test_no_trials_leaves_the_result_undefined_not_zero():
    r = compute_fr([], [], [])
    assert r.trials_scored == 0
    assert r.headline is None
    assert "undefined" in str(r)


def test_normalisation_is_shared_with_icr():
    """FR must not be able to disagree with ICR about what a word is."""
    assert invented_content("HELICOPTER.", TARGET, INTERFERER) == {"helicopter"}
