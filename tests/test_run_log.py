"""Unit tests for src/run_log.py.

The property that matters most: this must never take down a job. A render that
succeeded after two hours and then died writing a markdown row would be worse
than having no log at all, so every failure path is tested.
"""

import pytest

from src import run_log

HEADER = f"""# Run times

| date | command | scope | wall | rate |
|---|---|---|---|---|
{run_log.MARKER}
| 2026-08-14 | `old.py` | 1 trial | 90 s | |
"""


@pytest.fixture
def log(tmp_path):
    p = tmp_path / "run_times.md"
    p.write_text(HEADER)
    return p


# --- formatting -----------------------------------------------------------

@pytest.mark.parametrize("seconds,want", [
    (58, "58 s"), (89, "89 s"), (90, "2 min"), (600, "10 min"),
    (3599, "60 min"), (3600, "1.0 h"), (9000, "2.5 h"),
])
def test_human_units(seconds, want):
    assert run_log.human(seconds) == want


def test_record_resolves_path_at_call_time(tmp_path, monkeypatch):
    """Regression. `path=RUN_TIMES` as a default argument bound at import time,
    so patching RUN_TIMES did nothing and `timed()` wrote into the real
    docs/run_times.md during tests. It did, four times, before this existed."""
    p = tmp_path / "run_times.md"
    p.write_text(HEADER)
    monkeypatch.setattr(run_log, "RUN_TIMES", p)
    assert run_log.record("x.py", "a", 100) is True
    assert "`x.py`" in p.read_text()


# --- the threshold --------------------------------------------------------

def test_under_a_minute_writes_nothing(log):
    """The CLAUDE.md rule is 'over a minute'. Short runs must not accumulate,
    or the file stops being worth reading."""
    assert run_log.record("fast.py", "10 trials", 59, path=log) is False
    assert log.read_text() == HEADER


def test_over_a_minute_writes_a_row(log):
    assert run_log.record("slow.py", "20,000 trials", 3600, path=log) is True
    assert "| `slow.py` | 20,000 trials | 1.0 h |" in log.read_text()


def test_row_goes_directly_below_the_marker_newest_first(log):
    run_log.record("first.py", "a", 100, path=log)
    run_log.record("second.py", "b", 100, path=log)
    lines = [l for l in log.read_text().splitlines() if l.startswith("|")]
    order = [l for l in lines if "`" in l]
    assert "second.py" in order[0] and "first.py" in order[1] \
        and "old.py" in order[2]


def test_header_and_existing_rows_survive(log):
    run_log.record("new.py", "a", 100, path=log)
    text = log.read_text()
    assert text.startswith("# Run times")
    assert "| date | command | scope | wall | rate |" in text
    assert "`old.py`" in text


# --- never fatal ----------------------------------------------------------

def test_missing_file_does_not_raise(tmp_path):
    assert run_log.record("x.py", "a", 100, path=tmp_path / "nope.md") is False


def test_missing_marker_appends_instead_of_losing_the_row(tmp_path):
    p = tmp_path / "run_times.md"
    p.write_text("# Run times\n\nno marker here\n")
    assert run_log.record("x.py", "a", 100, path=p) is True
    assert "`x.py`" in p.read_text()


def test_env_var_disables(log, monkeypatch):
    monkeypatch.setenv("RUN_LOG", "0")
    assert run_log.record("x.py", "a", 100, path=log) is False
    assert log.read_text() == HEADER


# --- the context manager --------------------------------------------------

def test_timed_logs_on_success(log, monkeypatch):
    monkeypatch.setattr(run_log, "RUN_TIMES", log)
    monkeypatch.setattr(run_log, "MIN_SECONDS", 0)
    with run_log.timed("job.py", scope=lambda: "5 things"):
        pass
    assert "`job.py` | 5 things" in log.read_text()


def test_timed_marks_a_failed_job_and_reraises(log, monkeypatch):
    """A render that dies 40 minutes in is exactly what you want recorded."""
    monkeypatch.setattr(run_log, "RUN_TIMES", log)
    monkeypatch.setattr(run_log, "MIN_SECONDS", 0)
    with pytest.raises(ValueError):
        with run_log.timed("job.py", scope=lambda: "5 things"):
            raise ValueError("boom")
    assert "**(failed)**" in log.read_text()


def test_timed_survives_a_broken_scope_callable(log, monkeypatch):
    """The scope callable runs after the work is done. A bug in it must not
    turn a successful two-hour job into a crash."""
    monkeypatch.setattr(run_log, "RUN_TIMES", log)
    monkeypatch.setattr(run_log, "MIN_SECONDS", 0)
    with run_log.timed("job.py", scope=lambda: 1 / 0):
        pass
    assert "`job.py` | ?" in log.read_text()


def test_timed_accepts_plain_strings_too(log, monkeypatch):
    monkeypatch.setattr(run_log, "RUN_TIMES", log)
    monkeypatch.setattr(run_log, "MIN_SECONDS", 0)
    with run_log.timed("job.py", scope="literal", rate="8 workers"):
        pass
    assert "| literal | 0 s | 8 workers |" in log.read_text()


# --- the shipped file -----------------------------------------------------

def test_shipped_run_times_has_the_marker():
    """Without it every row silently appends to the bottom of the file, below
    the projections table, where nobody looks."""
    assert run_log.MARKER in run_log.RUN_TIMES.read_text()
