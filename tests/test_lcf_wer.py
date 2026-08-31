"""Tests for LCF-WER (src/live_model_metric/lcf_wer.py).

Each test states the property it protects, because these are the claims the
metric rests on and they have to be defensible in writing, not just green.
"""

import pytest

from src.live_model_metric import (edit_counts, headroom_captured, lcf_wer,
                                   normalise)

T = "the pencil outline by edward burne jones shows psyche entering heaven"


# --- the anchor every other number is read against ------------------------

def test_a_perfect_response_scores_exactly_zero():
    """Handed the reference itself, LCF-WER must be exactly 0 -- not "small".

    Every reported number is relative to this. If normalisation or the WER
    wiring drifts, this is the test that catches it.
    """
    assert lcf_wer([T], [T]).wer == 0.0


def test_saying_nothing_scores_one_hundred_percent_as_deletions():
    """A model that said nothing recovered nothing. All error is DELETION,
    which is what identifies suppression rather than mishearing."""
    r = lcf_wer([""], [T])
    assert r.wer == 100.0
    assert r.del_rate == 100.0
    assert r.sub_rate == 0.0 and r.ins_rate == 0.0


# --- the corpus-level convention -----------------------------------------

def test_wer_is_corpus_level_not_the_mean_of_per_trial_rates():
    """Total edits / total reference words, so a long trial outweighs a short
    one. Averaging per-trial rates would score these two trials equally and let
    a 1-word utterance swing the headline."""
    long_ref = " ".join(["alpha"] * 20)
    r = lcf_wer(["alpha " * 20, "wrong"], [long_ref, "beta"])
    assert r.ref_words == 21                       # 20 + 1, not 2 trials
    assert r.wer == pytest.approx(100 / 21, abs=0.05)


def test_wer_can_exceed_one_hundred_percent():
    """Insertions are not bounded by reference length. This is diagnostic, not a
    bug: it is why tiny.en was disqualified at a 123 % floor -- it invented more
    words than were spoken, which makes a metric un-rankable."""
    r = lcf_wer(["alpha beta gamma delta epsilon"], ["alpha"])
    assert r.wer > 100.0
    assert r.ins_rate > 100.0


def test_the_three_error_rates_sum_to_the_headline():
    """The decomposition must account for the whole number, or it is not a
    decomposition."""
    r = lcf_wer(["the pencil outline by someone else entirely"], [T])
    assert r.sub_rate + r.del_rate + r.ins_rate == pytest.approx(r.wer, abs=0.01)


# --- what the error types mean ------------------------------------------

def test_deletions_identify_suppression_insertions_identify_leakage():
    """The two failures this project exists to tell apart must be
    distinguishable in the breakdown, not just in the total."""
    suppressed = lcf_wer(["the pencil outline"], [T])       # dropped most words
    leaked = lcf_wer([T + " and heredity is the cause of all our faults"], [T])
    assert suppressed.del_rate > suppressed.ins_rate
    assert leaked.ins_rate > leaked.del_rate
    # Both are penalised -- neither failure is free.
    assert suppressed.wer > 0 and leaked.wer > 0


# --- B4: absent trials ---------------------------------------------------

def test_absent_trials_are_excluded_and_counted_not_scored_as_zero():
    """B4, decisions-m0.md 2026-08-13. No reference text means WER is 0/0,
    undefined. Scoring them as 0 % would reward saying nothing when nothing was
    said, and dilute the headline with trials the metric cannot judge."""
    r = lcf_wer(["anything at all", T], ["", T], target_absent=[True, False])
    assert r.n_scored == 1
    assert r.n_excluded_absent == 1
    assert r.wer == 0.0                    # from the one present trial only


def test_a_present_trial_with_no_reference_text_is_flagged_not_silently_dropped():
    """That combination is a data bug. It must be visible, because a silent drop
    changes the denominator without telling anyone."""
    r = lcf_wer(["something"], [""], target_absent=[False])
    assert r.n_excluded_no_ref == 1
    assert r.n_scored == 0
    assert r.wer is None                   # refuses to invent a number


# --- normalisation is part of the instrument ----------------------------

def test_normalisation_removes_differences_the_system_is_not_responsible_for():
    """Casing, punctuation and contractions must not count as errors."""
    assert lcf_wer(["It's ONLY a pencil, outline."],
                   ["it is only a pencil outline"]).wer == 0.0


def test_both_sides_are_normalised_by_the_same_function():
    """Normalising the two sides differently is the classic way to flatter your
    own system. edit_counts normalises internally so a caller cannot."""
    e = edit_counts("TWENTY FIVE PENCILS.", "twenty five pencils")
    assert e.total == 0


def test_missing_values_are_scored_not_skipped():
    """A missing response is a real outcome -- the judge said nothing. Skipping
    it would quietly delete the worst trials from the average."""
    assert normalise(None) == "" and normalise(float("nan")) == ""
    assert lcf_wer([None], [T]).wer == 100.0


# --- reading the number against the anchors -----------------------------

def test_headroom_puts_a_system_on_the_floor_to_ceiling_scale():
    """metric-definitions.md 3.4. Measured anchors, eval_public `both`, n=230:
    floor 57.4 %, ceiling 6.1 %."""
    assert headroom_captured(57.4, 57.4, 6.1) == pytest.approx(0.0)   # doing nothing
    assert headroom_captured(6.1, 57.4, 6.1) == pytest.approx(1.0)    # clean audio
    assert headroom_captured(31.75, 57.4, 6.1) == pytest.approx(0.5, abs=0.01)


def test_a_system_worse_than_doing_nothing_reports_a_negative_number():
    """This has happened here -- the epoch-24 checkpoint fell below
    pass-through. Clipping it at zero at measurement time would hide a real
    regression; clip at the plot, not in the metric."""
    assert headroom_captured(70.0, 57.4, 6.1) < 0


def test_no_headroom_is_an_error_not_a_division_by_zero():
    with pytest.raises(ValueError, match="no headroom"):
        headroom_captured(30.0, 20.0, 20.0)
