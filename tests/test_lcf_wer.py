"""Tests for LCF-WER (src/live_model_metric/lcf_wer.py).

Each test names the property it protects. These are the claims the primary
metric rests on, so they have to be defensible in writing, not just green.
"""

import pytest

from src.live_model_metric.lcf_wer import (compute_lcf_wer, count_errors,
                                           normalise_text, score_audio_files,
                                           transcribe_audio_files)

TARGET = "the pencil outline by edward burne jones shows psyche entering heaven"
TARGET_WORD_COUNT = 11


# --- the anchor every other number is read against ------------------------

def test_a_perfect_response_scores_exactly_zero():
    """Handed the reference itself, LCF-WER must be exactly 0 -- not "small".

    Every reported number is relative to this. If the normaliser or the jiwer
    wiring drifts, this is the test that catches it.
    """
    assert compute_lcf_wer([TARGET], [TARGET]).word_error_rate == 0.0


def test_saying_nothing_scores_exactly_one_hundred_percent_as_deletions():
    """A listener that said nothing recovered nothing.

    Exactly 100.0, not 100.00000000000001: the rate is computed as
    edits * 100 / n rather than edits * (100 / n), because the latter is
    inexact when edits == n.
    """
    result = compute_lcf_wer(reference_texts=[TARGET], hypothesis_texts=[""])
    assert result.word_error_rate == 100.0
    assert result.deletion_rate == 100.0
    assert result.substitution_rate == 0.0
    assert result.insertion_rate == 0.0


# --- the corpus-level convention -----------------------------------------

def test_wer_is_corpus_level_not_the_mean_of_per_trial_rates():
    """Total edits / total reference words, so a long trial outweighs a short one.

    Measured consequence on real data: the mean of per-trial rates is 71.0 %
    against a corpus rate of 57.4 % on eval_public `both` -- a 13.6 point
    inflation, driven by short utterances against a talkative interferer.
    Reference lengths there run 7 to 63 words.
    """
    long_reference = " ".join(["alpha"] * 20)
    result = compute_lcf_wer(
        reference_texts=[long_reference, "beta"],
        hypothesis_texts=["alpha " * 20, "wrong"],
    )
    assert result.reference_word_count == 21          # 20 + 1, not "2 trials"
    assert result.word_error_rate == pytest.approx(100 / 21, abs=0.05)
    assert result.word_error_rate < 10                # the mean-of-rates would be 50


def test_wer_can_exceed_one_hundred_percent():
    """Insertions are not bounded by the reference length.

    Diagnostic, not a bug: `tiny.en` was disqualified as the offline ASR at a
    123 % floor because it inserted more words than were spoken, which makes a
    metric un-rankable. A real trial reached 729 % -- a 7-word reference against
    a listener that returned 51 words.
    """
    result = compute_lcf_wer(["alpha"], ["alpha beta gamma delta epsilon"])
    assert result.word_error_rate > 100.0
    assert result.insertion_rate > 100.0


# --- the error decomposition --------------------------------------------

def test_the_three_error_rates_sum_exactly_to_the_headline():
    """That identity is what makes the breakdown a DECOMPOSITION rather than
    three loosely related numbers.

    It is also why the rates are stored unrounded: three independent roundings
    drift by up to 0.015, which broke this identity in an earlier draft.
    """
    result = compute_lcf_wer([TARGET], ["the pencil outline by someone else entirely"])
    assert (result.substitution_rate + result.deletion_rate
            + result.insertion_rate) == pytest.approx(result.word_error_rate)


def test_deletions_mean_suppression_and_insertions_mean_leakage():
    """The two failures this project exists to tell apart must be
    distinguishable in the breakdown, not just in the total.

    Measured on the floor at n=230: insertions 30.8, substitutions 23.2,
    deletions 3.5 -- doing nothing means the listener hears too MUCH, not too
    little.
    """
    suppressed = compute_lcf_wer([TARGET], ["the pencil outline"])
    leaked = compute_lcf_wer([TARGET], [TARGET + " heredity is the cause of all our faults"])

    assert suppressed.deletion_rate > suppressed.insertion_rate
    assert leaked.insertion_rate > leaked.deletion_rate
    assert suppressed.word_error_rate > 0 and leaked.word_error_rate > 0


def test_count_errors_returns_counts_not_rates():
    """So the caller can sum them into a corpus rate. Summing per-trial RATES
    instead is the mistake test_wer_is_corpus_level guards against."""
    counts = count_errors(TARGET, "the pencil outline")
    assert counts.reference_word_count == TARGET_WORD_COUNT
    assert counts.deletions == 8
    assert counts.total_errors == 8


# --- B4: trials with no reference text -----------------------------------

def test_a_trial_with_no_reference_text_is_excluded_and_counted():
    """B4, decisions-m0.md 2026-08-13. Where the target never speaks there is no
    reference, so the rate is 0/0 -- undefined, not perfect. Folding those in as
    0 % would reward a system for saying nothing when nothing was said."""
    result = compute_lcf_wer(["", TARGET], ["anything at all", TARGET])
    assert result.trials_scored == 1
    assert result.trials_without_reference == 1
    assert result.word_error_rate == 0.0        # from the one scorable trial


def test_no_scorable_trials_refuses_to_invent_a_number():
    result = compute_lcf_wer([""], ["something"])
    assert result.trials_scored == 0
    assert result.word_error_rate is None
    assert "undefined" in str(result)


def test_count_errors_on_an_empty_reference_reports_zero_words():
    """It refuses to invent a rate; the caller decides what to do."""
    counts = count_errors("", "something came out")
    assert counts.reference_word_count == 0
    assert counts.total_errors == 0


# --- normalisation is part of the instrument ----------------------------

def test_normalisation_removes_differences_the_system_is_not_responsible_for():
    """Casing, punctuation and contractions must not count as errors.

    LibriSpeech ground truth is uppercase and unpunctuated; Whisper output is
    mixed-case and punctuated. Counting that difference would measure the corpus
    formatting, not the extractor.
    """
    assert compute_lcf_wer(["it is only a pencil outline"],
                           ["It's ONLY a pencil, outline."]).word_error_rate == 0.0


def test_both_sides_are_normalised_by_the_same_function():
    """Normalising the two sides differently is the classic way to flatter your
    own system. count_errors normalises internally so a caller cannot."""
    assert count_errors("TWENTY FIVE PENCILS.", "twenty five pencils").total_errors == 0


def test_missing_values_are_scored_not_skipped():
    """A missing response is a real outcome -- the judge said nothing. Skipping
    it would quietly delete the worst trials from the average."""
    assert normalise_text(None) == ""
    assert normalise_text(float("nan")) == ""
    assert compute_lcf_wer([TARGET], [None]).word_error_rate == 100.0


# --- the swappable listener ---------------------------------------------

def test_audio_enters_only_through_the_transcribe_callable():
    """The swap point: the same scoring serves the offline ASR today and the
    live judge later. Nothing below transcription knows about audio."""
    fake_audio = ["a.wav", "b.wav"]
    transcripts = {"a.wav": TARGET, "b.wav": ""}

    assert transcribe_audio_files(fake_audio, transcripts.__getitem__) == [TARGET, ""]

    result = score_audio_files(fake_audio, [TARGET, TARGET], transcripts.__getitem__)
    assert result.trials_scored == 2
    assert result.word_error_rate == 50.0        # one perfect, one all-deletions


# --- input validation ---------------------------------------------------

def test_mismatched_input_lengths_raise():
    with pytest.raises(ValueError, match="references but"):
        compute_lcf_wer([TARGET], [TARGET, TARGET])
