"""Tests for ICR (src/live_model_metric/icr.py).

Each test names the property it protects. These are the claims the metric rests
on, so they have to be defensible in writing, not just green.
"""

import pytest

from src.live_model_metric.icr import (HEADLINE_K, compute_icr, content_words,
                                       interferer_exclusive_content,
                                       load_stopwords, measure_leakage)

TARGET = "the pencil outline shows psyche entering heaven"
INTERFERER = "the pencil outline shows heredity and all our faults"


# --- the two anchors -------------------------------------------------------

def test_reporting_the_target_leaks_nothing():
    """The correct answer must score exactly 0, not merely low. Every other
    number is read against this."""
    result = compute_icr([TARGET], [TARGET], [INTERFERER])
    assert result.headline == 0.0


def test_reporting_the_interferer_leaks_everything():
    """The opposite failure must saturate, or the metric has no range."""
    result = compute_icr([INTERFERER], [TARGET], [INTERFERER])
    assert result.headline == 100.0


# --- the exclusion rule ----------------------------------------------------

def test_words_both_speakers_said_are_not_counted_as_leakage():
    """Only interferer-EXCLUSIVE content counts. Otherwise a system is punished
    for correctly reporting words the target genuinely said."""
    available = interferer_exclusive_content(TARGET, INTERFERER)
    assert available == {"heredity", "faults"}
    assert "pencil" not in available and "outline" not in available


def test_a_correct_response_containing_shared_words_still_scores_zero():
    leakage = measure_leakage("pencil outline shows psyche", TARGET, INTERFERER)
    assert leakage.leaked_count == 0


# --- the denominator, which is easy to get wrong ---------------------------

def test_the_fraction_denominator_is_available_words_not_interferer_length():
    """leaked / AVAILABLE, never leaked / len(interferer_text).

    The interferer text here is 10 words but only 4 are exclusive content, so
    the two denominators differ by 2.5x. Dividing by utterance length would make
    the score depend on how many function words the interferer happened to use.
    """
    interferer = "the heredity of all our faults and the horse sense"
    assert len(interferer.split()) == 10
    leakage = measure_leakage("heredity faults", target_text="",
                              interferer_text=interferer)
    assert leakage.available_count == 4          # heredity faults horse sense
    assert leakage.leaked_count == 2
    assert leakage.leaked_fraction == pytest.approx(0.5)     # 2/4, not 2/10


# --- per-k eligibility, the correctness point behind ICR@k -----------------

def test_a_trial_that_cannot_reach_k_is_excluded_from_that_k():
    """A trial with 1 exclusive word available is structurally incapable of
    scoring @2. Leaving it in that denominator would drag the rate down for a
    reason about the trial, not the system."""
    responses   = ["alpha beta gamma", "alpha"]
    targets     = ["alpha beta",       "alpha"]
    interferers = ["alpha beta gamma", "delta epsilon zeta eta theta"]
    result = compute_icr(responses, targets, interferers)

    # trial 1 has 1 available and leaks it; trial 2 has 5 available and leaks none
    assert result.eligible_at_k[1] == 2
    assert result.icr_at_k[1] == 50.0
    # at k=2 trial 1 drops out of the denominator entirely
    assert result.eligible_at_k[2] == 1
    assert result.icr_at_k[2] == 0.0


def test_eligible_counts_are_reported_for_every_k():
    """Each ICR@k has a different denominator, so n_k must travel with it or the
    curve cannot be read."""
    result = compute_icr([TARGET], [TARGET], [INTERFERER])
    assert set(result.eligible_at_k) == {1, 2, 3, 5}


# --- ineligible trials -----------------------------------------------------

def test_a_trial_with_no_exclusive_interferer_content_is_excluded_not_scored_clean():
    """It cannot evidence leakage either way. Scoring it 0 would dilute the rate
    with trials that could never have fired."""
    result = compute_icr(["anything at all"], [TARGET], [TARGET])
    assert result.trials_ineligible == 1
    assert result.eligible_at_k[1] == 0
    assert result.headline is None


def test_noise_only_trials_are_ineligible():
    """Nobody speaks, so there is no interferer content to leak."""
    result = compute_icr(["you"], [""], [""])
    assert result.trials_ineligible == 1


# --- absent trials, where ICR is sharpest ---------------------------------

def test_when_the_target_is_absent_all_interferer_content_is_exclusive():
    """B4 excludes absent trials from LCF-WER because `t` is empty and word
    error rate against nothing is undefined. ICR's reference is `d`, which is
    NOT empty -- so these trials are scoreable, and they are the cleanest case:
    the right answer is silence, so any interferer content is unambiguous."""
    available = interferer_exclusive_content("", INTERFERER)
    assert available == content_words(INTERFERER)
    result = compute_icr([INTERFERER], [""], [INTERFERER])
    assert result.headline == 100.0


def test_silence_on_an_absent_trial_leaks_nothing():
    result = compute_icr([""], [""], [INTERFERER])
    assert result.headline == 0.0


# --- the stopword snapshot ------------------------------------------------

def test_stopwords_are_split_after_normalising():
    """45 of the 198 snapshot entries are apostrophe forms, and the normaliser
    expands them into two words ("don't" -> "do not"). Stored whole, such an
    entry is a set member no single token can ever equal, so it would silently
    filter nothing. This is the bug this test exists to prevent recurring."""
    stopwords = load_stopwords()
    assert not [w for w in stopwords if " " in w], "multi-word stopword member"
    assert {"do", "not", "will", "are", "is"} <= stopwords


def test_content_words_drops_function_words():
    assert content_words("the pencil is on a table") == {"pencil", "table"}


def test_content_words_are_a_set_not_a_count():
    """ICR asks whether the interferer's words are PRESENT, not how often they
    were repeated -- a judge repeating one leaked word is one leak."""
    assert content_words("heredity heredity heredity") == {"heredity"}


# --- the continuous measure ----------------------------------------------

def test_the_mean_fraction_ignores_trials_with_too_little_available_content():
    """A fraction over a tiny denominator is not a fraction: 1 of 1 is 100 %,
    which is noise. Only trials with >=5 available words contribute."""
    four_available = compute_icr(["alpha"], [""], ["alpha beta gamma delta"])
    assert four_available.trials_for_fraction == 0
    assert four_available.mean_leaked_percent is None

    five_available = compute_icr(["alpha"], [""], ["alpha beta gamma delta epsilon"])
    assert five_available.trials_for_fraction == 1
    assert five_available.mean_leaked_percent == pytest.approx(20.0)   # 1/5


def test_headline_is_k_two():
    """One shared content word is coincidence at the rate English repeats nouns.
    Two is signal."""
    assert HEADLINE_K == 2
    one_leak = compute_icr(["heredity"], [TARGET], [INTERFERER])
    assert one_leak.icr_at_k[1] == 100.0
    assert one_leak.icr_at_k[2] == 0.0
    assert one_leak.headline == 0.0


# --- input validation ----------------------------------------------------

def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError, match="must be parallel"):
        compute_icr(["a", "b"], [TARGET], [INTERFERER])
