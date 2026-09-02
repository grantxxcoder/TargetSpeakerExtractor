"""Tests for the live judge adapter (src/live_model_metric/judge.py).

No network. Everything here is the logic that decides what gets SPENT and what
gets REUSED -- the parts that must be right before a multi-day quota run, since
a cache bug either wastes calls or silently serves the wrong answer.
"""

import csv

import pytest

from src.live_model_metric.judge import (Judge, QuotaExhausted, _is_quota_error,
                                         cache_key, load_cache, prompt_sha)


# --- the structured status, which is why the prompt has no in-band sentinel --

def test_no_speech_yields_empty_text_not_a_sentinel():
    """The whole reason for a status field. whisper-normalizer maps '####' to
    '', so an in-band marker cannot survive to be counted -- the status has to
    carry it instead."""
    assert Judge._parse('{"status":"no_speech","transcript":""}') == ("no_speech", "")


def test_status_wins_over_junk_left_in_the_transcript():
    """If the judge sets no_speech but also fills the transcript, the status is
    authoritative. Otherwise a stray sentinel would be scored as a real word."""
    assert Judge._parse('{"status":"no_speech","transcript":"####"}') == ("no_speech", "")


def test_speech_is_returned_verbatim_and_stripped():
    assert Judge._parse('{"status":"speech","transcript":" hello there "}') == \
        ("speech", "hello there")


# --- a judge that wanders off-prompt is a FINDING, not a crash --------------

def test_unparsed_response_is_recorded_rather_than_raised():
    """J2: a judge that chats instead of reporting lands in NRR and is
    indistinguishable from a degenerate extractor. So it must be visible in the
    data, not swallowed by an exception mid-run."""
    status, text = Judge._parse("Sure! Here is the transcription: hello")
    assert status == "unparsed"
    assert "hello" in text


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_empty_response_is_its_own_status(empty):
    assert Judge._parse(empty) == ("empty_response", "")


# --- the cache key: what makes a call reusable ------------------------------

def test_prompt_change_changes_the_key(tmp_path):
    """The prompt is part of the instrument. Editing it must invalidate cached
    answers rather than silently reuse them."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    a = cache_key(audio, "m", "sha-aaa", 0)
    b = cache_key(audio, "m", "sha-bbb", 0)
    assert a != b


def test_repeat_and_backend_and_model_all_change_the_key(tmp_path):
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    base = cache_key(audio, "m", "sha", 0, "aistudio")
    assert base != cache_key(audio, "m", "sha", 1, "aistudio")   # k repeats differ
    assert base != cache_key(audio, "m", "sha", 0, "vertex")     # surfaces differ
    assert base != cache_key(audio, "other", "sha", 0, "aistudio")


def test_prompt_sha_is_stable_and_short():
    assert prompt_sha() == prompt_sha()
    assert len(prompt_sha()) == 12


# --- resumability: the cache is the ledger ---------------------------------

def test_cached_key_is_served_without_a_client(tmp_path):
    """A cache hit must not touch the network -- there is no client configured
    here, so this would raise if it tried."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    cache = tmp_path / "judge.csv"
    judge = Judge(model_id="m", cache_path=cache)
    key = cache_key(audio, "m", judge.sha, 0, "aistudio")
    with open(cache, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["key", "model", "backend",
                                                    "prompt_sha", "trial_id", "file",
                                                    "repeat", "status", "text",
                                                    "run_date"])
        writer.writeheader()
        writer.writerow({"key": key, "model": "m", "backend": "aistudio",
                         "prompt_sha": judge.sha, "trial_id": "t1",
                         "file": "mixture.wav", "repeat": 0, "status": "speech",
                         "text": "cached answer", "run_date": "2026-09-02"})
    judge = Judge(model_id="m", cache_path=cache)
    assert judge.judge(audio) == ("speech", "cached answer")
    assert judge.calls_made == 0
    assert judge.cache_hits == 1


def test_allow_new_false_refuses_to_spend_a_call(tmp_path):
    """A silent multi-day quota burn is worse than an error."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    judge = Judge(cache_path=tmp_path / "j.csv", allow_new=False)
    with pytest.raises(RuntimeError, match="not in the judge cache"):
        judge.judge(audio)


def test_missing_cache_file_is_an_empty_cache(tmp_path):
    assert load_cache(tmp_path / "nope.csv") == {}


# --- quota: a daily cap is not a failure -----------------------------------

@pytest.mark.parametrize("message,throttled,daily", [
    ("429 RESOURCE_EXHAUSTED quota per day exceeded", True, True),
    ("Quota exceeded for requests per day", True, True),
    ("429 rate limit exceeded", True, False),
    ("500 internal error", False, False),
])
def test_quota_error_classification(message, throttled, daily):
    """Per-minute throttling is retried; the daily cap stops the run so it can
    resume tomorrow. Misclassifying the second as the first would burn the
    retry budget against a wall."""
    assert _is_quota_error(Exception(message)) == (throttled, daily)


def test_bad_backend_is_rejected_at_construction():
    with pytest.raises(ValueError, match="backend must be"):
        Judge(backend="openai")


def test_quota_exhausted_is_not_retried(tmp_path, monkeypatch):
    """QuotaExhausted must propagate immediately -- retrying a spent daily cap
    just sleeps through the backoff for nothing."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    judge = Judge(cache_path=tmp_path / "j.csv", requests_per_minute=0,
                  backoff_initial_s=0.0)
    calls = {"n": 0}

    def boom(_path):
        calls["n"] += 1
        raise Exception("429 RESOURCE_EXHAUSTED: quota per day exceeded")

    monkeypatch.setattr(judge, "_call_once", boom)
    with pytest.raises(QuotaExhausted):
        judge.judge(audio)
    assert calls["n"] == 1, "daily quota must not be retried"
