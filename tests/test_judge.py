"""Tests for the live judge adapter (src/live_model_metric/judge.py).

No network. Everything here is the logic that decides what gets SPENT and what
gets REUSED -- the parts that must be right before a multi-day quota run, since
a cache bug either wastes calls or silently serves the wrong answer.
"""

import csv

import pytest

from pathlib import Path

from src.live_model_metric.judge import (CACHE_FIELDS, Judge,
                                         NewCallLimitReached, QuotaExhausted,
                                         _is_quota_error, cache_key,
                                         load_cache, load_once_index,
                                         prompt_sha, runs_once)


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


def test_repeat_and_backend_and_model_all_change_the_estimate_key(tmp_path):
    """estimate.wav only -- a run-once clip deliberately ignores repeat."""
    audio = tmp_path / "t1" / "estimate.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    base = cache_key(audio, "m", "sha", 0, "aistudio")
    assert base != cache_key(audio, "m", "sha", 1, "aistudio")   # k repeats differ
    assert base != cache_key(audio, "m", "sha", 0, "vertex")     # surfaces differ
    assert base != cache_key(audio, "other", "sha", 0, "aistudio")


def test_run_once_key_ignores_repeat_but_not_the_instrument(tmp_path):
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"x" * 16)
    base = cache_key(audio, "m", "sha", 0, "aistudio")
    assert base == cache_key(audio, "m", "sha", 7, "aistudio")   # repeat ignored
    assert base != cache_key(audio, "m", "sha", 0, "vertex")     # surface matters
    assert base != cache_key(audio, "m", "other-sha", 0, "aistudio")
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


# --- run-once vs re-runnable ----------------------------------------------

def _write_cache(path, key, judge, filename="mixture.wav", text="cached answer"):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CACHE_FIELDS)
        writer.writeheader()
        writer.writerow({"key": key, "model": judge.model_id,
                         "backend": judge.backend, "prompt_sha": judge.sha,
                         "trial_id": "t1", "file": filename, "repeat": 0,
                         "status": "speech", "text": text,
                         "run_date": "2026-09-02"})


@pytest.mark.parametrize("name,once", [
    ("mixture.wav", True), ("target.wav", True), ("interferer.wav", True),
    ("estimate.wav", False),
])
def test_which_clips_run_once(name, once):
    assert runs_once(Path("t1") / name) is once


@pytest.mark.parametrize("name", ["mixture.wav", "target.wav", "interferer.wav"])
def test_run_once_clip_ignores_repeat(tmp_path, name):
    """The rule: an unchangeable clip is judged ONCE, whatever the repeat index.
    No model can alter the rendered mixture, so a second answer buys nothing."""
    audio = tmp_path / "t1" / name
    audio.parent.mkdir(exist_ok=True)
    audio.write_bytes(b"audio")
    cache = tmp_path / "j.csv"
    j0 = Judge(model_id="m", cache_path=cache, repeat=0)
    _write_cache(cache, cache_key(audio, "m", j0.sha, 0, "aistudio"), j0,
                 filename=name)

    for repeat in (0, 1, 2):
        j = Judge(model_id="m", cache_path=cache, repeat=repeat)
        assert j.cached(audio) is not None, f"repeat={repeat} must reuse the answer"
        assert j.judge(audio)[1] == "cached answer"
        assert j.calls_made == 0


def test_repeat_run_once_is_opt_in_and_scoped(tmp_path):
    """The spread study must be able to buy the same answer twice, and that
    ability must not leak into normal use -- otherwise the run-once guarantee
    is only a default rather than a guarantee."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    cache = tmp_path / "j.csv"
    j0 = Judge(model_id="m", cache_path=cache)
    _write_cache(cache, cache_key(audio, "m", j0.sha, 0, "aistudio"), j0)

    for repeat in (0, 1, 2):
        normal = Judge(model_id="m", cache_path=cache, repeat=repeat)
        assert normal.cached(audio) is not None, "run-once must still hold"

    # Opted in: repeat 0 matches the stored repeat-keyed row only if one exists.
    # The stored row used the once key, so every repeat is a miss here.
    for repeat in (0, 1, 2):
        spread = Judge(model_id="m", cache_path=cache, repeat=repeat,
                       repeat_run_once=True)
        assert spread.cached(audio) is None


def test_estimate_may_be_judged_again_per_repeat(tmp_path):
    """estimate.wav is model output, so k repeats are legitimate work."""
    est = tmp_path / "t1" / "estimate.wav"
    est.parent.mkdir()
    est.write_bytes(b"model output")
    keys = {cache_key(est, "m", "sha", r, "aistudio") for r in (0, 1, 2)}
    assert len(keys) == 3


def test_a_retrained_estimate_at_the_same_path_is_not_served_a_stale_answer(tmp_path):
    """A new checkpoint overwrites estimate.wav in place. Serving model A's
    answer for model B's audio would silently corrupt the results table, which
    is why the estimate key keeps a content hash."""
    est = tmp_path / "t1" / "estimate.wav"
    est.parent.mkdir()
    est.write_bytes(b"checkpoint A output")
    key_a = cache_key(est, "m", "sha", 0, "aistudio")
    est.write_bytes(b"checkpoint B output")
    assert cache_key(est, "m", "sha", 0, "aistudio") != key_a


def test_run_once_answer_found_whatever_key_format_wrote_it(tmp_path):
    """The index is built from the CSV columns, so answers already paid for are
    never orphaned by a change to the key format."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"already paid for")
    cache = tmp_path / "j.csv"
    judge = Judge(model_id="m", cache_path=cache)
    _write_cache(cache, "some|entirely|different|legacy|key|format", judge,
                 text="paid answer")

    judge = Judge(model_id="m", cache_path=cache)
    assert judge.judge(audio) == ("speech", "paid answer")
    assert judge.calls_made == 0


def test_a_changed_prompt_is_a_different_instrument(tmp_path):
    """Run-once means once per INSTRUMENT. A new prompt is a new measurement,
    so it must not be served the old prompt's answer."""
    audio = tmp_path / "t1" / "mixture.wav"
    audio.parent.mkdir()
    audio.write_bytes(b"audio")
    cache = tmp_path / "j.csv"
    judge = Judge(model_id="m", cache_path=cache)
    _write_cache(cache, cache_key(audio, "m", judge.sha, 0, "aistudio"), judge)

    judge = Judge(model_id="m", cache_path=cache)
    judge.sha = "differentsha"                 # as if the prompt file changed
    judge._once = load_once_index(cache)
    assert judge.cached(audio) is None


def test_preflight_counts_without_calling(tmp_path):
    paid = tmp_path / "t1" / "mixture.wav"
    paid.parent.mkdir()
    paid.write_bytes(b"paid")
    unpaid = tmp_path / "t2" / "mixture.wav"
    unpaid.parent.mkdir()
    unpaid.write_bytes(b"unpaid")
    cache = tmp_path / "j.csv"
    judge = Judge(model_id="m", cache_path=cache)
    _write_cache(cache, cache_key(paid, "m", judge.sha, 0, "aistudio"), judge)

    judge = Judge(model_id="m", cache_path=cache)
    assert judge.preflight([paid, unpaid, None]) == (1, 1)
    assert judge.calls_made == 0


def test_max_new_calls_refuses_rather_than_billing_through(tmp_path, monkeypatch):
    a = tmp_path / "t1" / "estimate.wav"
    a.parent.mkdir()
    a.write_bytes(b"x")
    b = tmp_path / "t2" / "estimate.wav"
    b.parent.mkdir()
    b.write_bytes(b"y")

    judge = Judge(cache_path=tmp_path / "j.csv", requests_per_minute=0,
                  max_new_calls=1)
    monkeypatch.setattr(judge, "_call_once", lambda _p: ("speech", "answer"))
    judge.judge(a)
    assert judge.calls_made == 1
    with pytest.raises(NewCallLimitReached):
        judge.judge(b)
    assert judge.calls_made == 1


# --- quota: a daily cap is not a failure -----------------------------------

@pytest.mark.parametrize("message,throttled,daily", [
    ("429 RESOURCE_EXHAUSTED quota per day exceeded", True, True),
    ("Quota exceeded for requests per day", True, True),
    ("429 rate limit exceeded", True, False),
    ("500 internal error", False, False),
    # A quota whose LIMIT IS 0 means the model is not available on this tier.
    # It is a permanent refusal wearing a 429, so it must NOT be retried --
    # this shape burned all five attempts on the first real smoke call.
    ("429 RESOURCE_EXHAUSTED: Quota exceeded, limit: 0", False, False),
    ('429 RESOURCE_EXHAUSTED quota_limit_value: "0"', False, False),
    # Observed 2026-09-02 on the first real call: a prepay project at zero
    # balance. Carries a 429 but no wait fixes it, so it must not be retried.
    ("Error code: 429 - Your prepayment credits are depleted. Please go to "
     "AI Studio to manage your project and billing.", False, False),
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


# --- a safety filter is a measurement event, not an error ------------------

REAL_FILTER_ERROR = (
    "Error code: 400 - {'error': {'message': \"Input blocked: This request was "
    "blocked by Gemini's filters. They can occasionally trigger by mistake on "
    "safe coding, security, or biology-related queries.\", "
    "'code': 'content_blocked'}}")


def test_the_real_filter_error_is_recognised():
    """Verbatim from a run on 2026-09-02, on an ESTIMATE clip."""
    from src.live_model_metric.judge import _is_content_blocked
    assert _is_content_blocked(Exception(REAL_FILTER_ERROR))
    assert _is_quota_error(Exception(REAL_FILTER_ERROR)) == (False, False)


def test_a_transient_filter_block_is_retried_and_not_recorded(tmp_path, monkeypatch):
    """THE FILTER IS NON-DETERMINISTIC. Measured 2026-09-02: a refused estimate
    passed on the re-run. Caching that refusal would permanently score the trial
    as a non-response because of a coin flip."""
    est = tmp_path / "t1" / "estimate.wav"
    est.parent.mkdir()
    est.write_bytes(b"artefact-laden audio")
    judge = Judge(cache_path=tmp_path / "j.csv", requests_per_minute=0,
                  backoff_initial_s=0.0, verbose=False, filter_retries=2)
    tries = {"n": 0}

    def blocked_once(_path):
        tries["n"] += 1
        if tries["n"] == 1:
            raise Exception(REAL_FILTER_ERROR)
        return ("speech", "the words after all")

    monkeypatch.setattr(judge, "_call_once", blocked_once)
    assert judge.judge(est) == ("speech", "the words after all")
    assert tries["n"] == 2, "the refusal must be retried"
    assert judge.filter_transients == 1
    assert judge.filter_blocks == 0


def test_a_persistent_filter_block_is_recorded(tmp_path, monkeypatch):
    """Refused every attempt -> a finding about this clip, not a coin flip."""
    est = tmp_path / "t1" / "estimate.wav"
    est.parent.mkdir()
    est.write_bytes(b"artefact-laden audio")
    cache = tmp_path / "j.csv"
    judge = Judge(cache_path=cache, requests_per_minute=0, backoff_initial_s=0.0,
                  verbose=False, filter_retries=2)
    tries = {"n": 0}

    def always_blocked(_path):
        tries["n"] += 1
        raise Exception(REAL_FILTER_ERROR)

    monkeypatch.setattr(judge, "_call_once", always_blocked)
    assert judge.judge(est) == ("blocked_by_filter", "")
    assert tries["n"] == 3, "one attempt plus filter_retries=2"
    assert judge.filter_blocks == 1

    again = Judge(cache_path=cache, requests_per_minute=0, verbose=False)
    assert again.judge(est) == ("blocked_by_filter", "")
    assert again.calls_made == 0


def test_filter_retries_do_not_consume_the_throttle_budget(tmp_path, monkeypatch):
    """A filter retry and a 429 retry are different budgets; sharing them would
    make a filter block eat the backoff needed for genuine throttling."""
    est = tmp_path / "t1" / "estimate.wav"
    est.parent.mkdir()
    est.write_bytes(b"x")
    judge = Judge(cache_path=tmp_path / "j.csv", requests_per_minute=0,
                  backoff_initial_s=0.0, verbose=False, max_retries=2,
                  filter_retries=2)
    seq = ["filter", "filter", "429", "ok"]
    tries = {"n": 0}

    def mixed(_path):
        kind = seq[tries["n"]]
        tries["n"] += 1
        if kind == "filter":
            raise Exception(REAL_FILTER_ERROR)
        if kind == "429":
            raise Exception("429 rate limit exceeded")
        return ("speech", "made it")

    monkeypatch.setattr(judge, "_call_once", mixed)
    assert judge.judge(est) == ("speech", "made it")
    assert tries["n"] == 4

