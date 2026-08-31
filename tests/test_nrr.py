"""Tests for NRR (src/live_model_metric/nrr.py)."""

import pytest

from src.live_model_metric.nrr import (compute_nrr, non_response_reason,
                                       silence_compliance,
                                       system_attributable_nrr)

TARGET = "the pencil outline by edward burne jones shows psyche entering heaven"


# --- the tripwire it exists to be -----------------------------------------

def test_a_system_that_outputs_silence_scores_one_hundred_percent():
    """The degenerate strategy. Such a system scores 0 % on ICR -- a perfect
    score for doing nothing."""
    assert compute_nrr([""] * 4, [TARGET] * 4).nrr == 100.0


def test_a_working_system_scores_zero():
    assert compute_nrr([TARGET], [TARGET]).nrr == 0.0


def test_a_partial_report_is_not_a_non_response():
    """NRR is deliberately coarse: it fires only when NOTHING came back. How
    much was lost is LCF-WER's job, specifically its deletion rate."""
    assert compute_nrr(["the pencil"], [TARGET]).nrr == 0.0


# --- detection is emptiness, one rule for judge and transcriber alike ----

def test_an_empty_response_is_the_signal():
    """The prompt instructs the judge to return nothing when it cannot identify
    speech. The offline ASR already does this naturally on silence, so ONE rule
    serves both."""
    assert non_response_reason("") == "silence"
    assert non_response_reason("   ") == "silence"
    assert non_response_reason(None) == "silence"


@pytest.mark.parametrize("artefact", ["you", "You.", "thank you", "bye"])
def test_known_transcriber_artefacts_count_as_non_responses(artefact):
    """Measured: small.en emits "you" on digital silence, 8 of 8 absent trials.
    A transcriber cannot be instructed, so it needs this list."""
    assert non_response_reason(artefact) == "artefact"


# --- the false-positive class this design avoids ------------------------

@pytest.mark.parametrize("spoken", [
    "nothing would satisfy him that could not stand cross examination",
    "there was no man sir his troubled blue eyes glanced at each of them",
    "to morrow is the examination i cannot allow the examination to be held",
    "onward said a distant voice no sound broke the stillness of the night",
])
def test_speech_containing_refusal_words_is_a_real_response(spoken):
    """Real ground-truth texts from eval_public. A pattern-matching detector
    flagged 7.6 % of trials as refusals because the SPEAKERS say these words.
    Reporting them correctly must score as a response."""
    assert non_response_reason(spoken) is None
    assert compute_nrr([spoken], [spoken]).nrr == 0.0


def test_a_prose_refusal_is_NOT_detected_and_that_is_deliberate():
    """A judge that ignores the instruction and declines in prose scores as a
    bad transcription, not a non-response. Catching it needs the pattern list
    whose false-positive rate exceeds the metric's whole range. Non-compliance
    is handled by disqualifying the judge -- see silence_compliance."""
    assert non_response_reason("I'm sorry, I cannot make out any speech") is None


# --- B4: absent trials --------------------------------------------------

def test_absent_trials_are_excluded_not_counted_as_non_responses():
    """Where the target never speaks, reporting nothing is CORRECT. An earlier
    draft scored 32.8 % NRR on CLEAN TARGET AUDIO because the 145 absent trials
    were counted as failures. This is that bug, pinned."""
    result = compute_nrr(["", TARGET], ["", TARGET])
    assert result.trials_excluded_absent == 1
    assert result.trials_scored == 1
    assert result.nrr == 0.0


def test_exclusion_happens_inside_the_metric_not_in_the_caller():
    """target_texts is required precisely so this cannot be forgotten."""
    with pytest.raises(TypeError):
        compute_nrr([""])


# --- the breakdown -----------------------------------------------------

def test_reasons_are_broken_out():
    result = compute_nrr(["", "you", TARGET], [TARGET] * 3)
    assert result.nrr == pytest.approx(66.67, abs=0.01)
    assert result.counts_by_reason == {"silence": 1, "artefact": 1}


# --- interpretation ----------------------------------------------------

def test_only_nrr_above_the_ceiling_is_attributable_to_the_system():
    """NRR on clean target audio IS the judge's own defect rate."""
    assert system_attributable_nrr(12.0, 10.0) == pytest.approx(2.0)
    assert system_attributable_nrr(10.0, 10.0) == pytest.approx(0.0)


def test_silence_compliance_is_a_judge_selection_measure():
    """Run on silent input. A judge that keeps talking when there is nothing to
    hear cannot be scored on NRR at all."""
    assert silence_compliance(["", "you", "I definitely hear a voice"]) == pytest.approx(66.67, abs=0.01)
    assert silence_compliance([]) is None


# --- input validation --------------------------------------------------

def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="responses but"):
        compute_nrr(["a", "b"], [TARGET])
