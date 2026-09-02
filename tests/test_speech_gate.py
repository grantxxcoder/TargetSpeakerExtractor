"""Tests for the speech gate (src/live_model_metric/speech_gate.py).

No network, no VAD model. What is tested is the decision rule and the symmetry
guarantee, because a gate applied to one listener and not the other turns every
judge-vs-ASR difference on a speech-free clip into an artefact of the gate.
"""

import csv

import pytest

from src.live_model_metric.speech_gate import (GateDecision, condition_lookup,
                                               decide, gated, log_decision,
                                               summarise)


# --- construction decides the anchors, no VAD needed ----------------------

@pytest.mark.parametrize("condition,target,interferer,mixture", [
    # verified against measured RMS on sir0_val, 2026-09-02
    ("both",            True,  True,  True),
    ("target_only",     True,  False, True),
    ("interferer_only", False, True,  True),
    # real noise energy but NOT ONE WORD of speech -- a more realistic
    # hallucination probe than digital silence
    ("noise_only",      False, False, False),
])
def test_construction_rules_match_the_rendered_audio(condition, target,
                                                     interferer, mixture):
    assert decide("t1/target.wav", condition).has_speech is target
    assert decide("t1/interferer.wav", condition).has_speech is interferer
    assert decide("t1/mixture.wav", condition).has_speech is mixture


def test_construction_needs_no_vad():
    """If an anchor ever consulted the VAD it would add a moving part to the
    instrument for information already known by construction."""
    def explode(_path):
        raise AssertionError("VAD must not run for an anchor")
    d = decide("t1/target.wav", "interferer_only", vad_detect=explode)
    assert d.fired
    assert d.reason.startswith("construction:")


def test_the_reason_names_the_condition():
    """The log has to say WHY, or a blocked row cannot be audited later."""
    assert decide("t1/mixture.wav", "noise_only").reason == \
        "construction:noise_only:mixture"


# --- the estimate is the only clip the VAD decides -----------------------

def test_estimate_with_speech_passes():
    d = decide("t1/estimate.wav", "both", vad_detect=lambda _p: 3.2)
    assert d.has_speech and d.speech_s == 3.2 and d.reason == "vad:speech"


def test_estimate_that_muted_is_blocked():
    """The mute detector. A degenerate extractor is a real failure mode --
    the 2026-08-25 run collapsed to silence -- and NRR cannot catch it because
    the judge hallucinates on silence."""
    d = decide("t1/estimate.wav", "both", vad_detect=lambda _p: 0.0)
    assert d.fired and d.reason == "vad:no-speech"


def test_estimate_just_under_threshold_is_blocked():
    assert decide("t1/estimate.wav", "both", min_speech_s=0.10,
                  vad_detect=lambda _p: 0.09).fired
    assert decide("t1/estimate.wav", "both", min_speech_s=0.10,
                  vad_detect=lambda _p: 0.10).has_speech


def test_missing_vad_passes_the_estimate_through():
    """Blocking on a missing detector would fabricate a measurement. Failing
    open is the honest default; the reason string records that it happened."""
    d = decide("t1/estimate.wav", "both")
    assert d.has_speech
    assert "no-vad-supplied" in d.reason


def test_unknown_clip_without_condition_is_never_blocked():
    assert decide("t1/something.wav").has_speech


# --- symmetry: the whole point -------------------------------------------

def test_both_listeners_get_the_identical_verdict(tmp_path):
    """Gate one listener and not the other and the comparison is void."""
    log = tmp_path / "gate.csv"
    calls = {"judge": 0, "asr": 0}

    def judge(_p):
        calls["judge"] += 1
        return "invented words from silence"

    def asr(_p):
        calls["asr"] += 1
        return "you"

    lookup = lambda _p: "noise_only"
    jg = gated(judge, "judge", condition_of=lookup, log_path=log)
    ag = gated(asr, "small.en", condition_of=lookup, log_path=log)

    assert jg("t1/mixture.wav") == ""
    assert ag("t1/mixture.wav") == ""
    assert calls == {"judge": 0, "asr": 0}, "neither listener should be called"


def test_a_passing_clip_reaches_the_listener(tmp_path):
    log = tmp_path / "gate.csv"
    g = gated(lambda _p: "the words", "judge",
              condition_of=lambda _p: "both", log_path=log)
    assert g("t1/mixture.wav") == "the words"


def test_blocked_clip_returns_empty_string_not_a_sentinel(tmp_path):
    """An empty hypothesis is scored as all-deletions by jiwer, which is
    metric-definitions.md 3.1's stated treatment of a listener that reported
    nothing -- so the gate introduces no new scoring rule."""
    g = gated(lambda _p: "x", "judge", condition_of=lambda _p: "noise_only",
              log_path=tmp_path / "gate.csv")
    assert g("t1/mixture.wav") == ""


# --- logging: every decision, not only the blocks ------------------------

def test_every_decision_is_logged_so_the_denominator_survives(tmp_path):
    log = tmp_path / "gate.csv"
    g = gated(lambda _p: "words", "judge", condition_of=lambda _p: "both",
              split="sir0_val", log_path=log)
    g("t1/mixture.wav")
    g("t2/mixture.wav")
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == 2
    assert all(r["has_speech"] == "1" for r in rows)
    assert {r["trial_id"] for r in rows} == {"t1", "t2"}
    assert rows[0]["split"] == "sir0_val" and rows[0]["listener"] == "judge"


def test_a_gate_firing_on_target_present_audio_is_recorded_as_such(tmp_path):
    """Not a measurement error -- a finding that the extractor destroyed the
    speech. It must be visible in the log with its condition attached."""
    log = tmp_path / "gate.csv"
    g = gated(lambda _p: "x", "judge", condition_of=lambda _p: "both",
              vad_detect=lambda _p: 0.0, log_path=log)
    g("t1/estimate.wav")
    row = list(csv.DictReader(open(log)))[0]
    assert row["has_speech"] == "0"
    assert row["condition"] == "both"          # target WAS present
    assert row["reason"] == "vad:no-speech"
    assert row["speech_s"] == "0.000"


def test_summarise_counts_blocked_over_total(tmp_path):
    log = tmp_path / "gate.csv"
    for condition in ("both", "noise_only", "noise_only"):
        gated(lambda _p: "x", "judge", condition_of=lambda _p: condition,
              log_path=log)("t1/mixture.wav")
    counts = summarise(log)
    assert counts[("judge", "construction:both:mixture")] == (1, 0)
    assert counts[("judge", "construction:noise_only:mixture")] == (2, 2)


def test_summarise_of_a_missing_log_is_empty(tmp_path):
    assert summarise(tmp_path / "nope.csv") == {}


def test_condition_lookup_reads_the_real_manifest():
    lookup = condition_lookup("sir0_val")
    assert lookup("data/rendered/sir0_val/sir0_val-42-000002/mixture.wav") == "both"
    assert lookup("data/rendered/sir0_val/sir0_val-42-000000/target.wav") == \
        "interferer_only"


def test_fired_is_the_inverse_of_has_speech():
    assert GateDecision(False, "r").fired is True
    assert GateDecision(True, "r").fired is False
