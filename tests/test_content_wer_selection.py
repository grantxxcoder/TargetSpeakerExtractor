"""Selecting and scheduling on validation WORD ERROR RATE (scripts/train.py).

Item 1a kept epoch 9 -- worse than doing nothing on content -- and rejected
epoch 15, the best content result in the project. These pin the replacement:
the scheduler and the selector can now read the thing the project measures.
decisions-m2.md 2026-09-23.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.train import (  # noqa: E402
    VAL_DIAGNOSTICS, build_content_probe, history_header, history_row,
    lr_schedule_metric, selection_score)


def cfg(mode, **training):
    return {"training": {"select_on": mode, **training},
            "loss": {"w_m": 9.62, "w_g": 1.69}}


def test_content_wer_mode_reads_the_smoothed_value():
    val = {"content_wer": 52.77, "content_wer_ema": 55.10,
           "L_pres": -3.2, "L_MR": 0.28, "L_gain": 4.1, "total": 1.0}
    assert selection_score(val, cfg("content_wer")) == pytest.approx(55.10)


def test_it_falls_back_to_raw_when_no_ema_is_present():
    val = {"content_wer": 52.77, "L_pres": -3.2, "L_MR": 0.28,
           "L_gain": 4.1, "total": 1.0}
    assert selection_score(val, cfg("content_wer")) == pytest.approx(52.77)


def test_missing_wer_is_an_error_not_a_silent_fallback():
    """Selecting on something the config did not ask for is the bug class this
    whole change exists to remove."""
    val = {"L_pres": -3.2, "L_MR": 0.28, "L_gain": 4.1, "total": 1.0}
    with pytest.raises(ValueError, match="content_wer"):
        selection_score(val, cfg("content_wer"))


def test_nan_wer_is_also_an_error():
    val = {"content_wer": float("nan"), "content_wer_ema": float("nan"),
           "L_pres": -3.2, "L_MR": 0.28, "L_gain": 4.1, "total": 1.0}
    with pytest.raises(ValueError):
        selection_score(val, cfg("content_wer"))


def test_the_scheduler_can_step_on_it_too():
    val = {"content_wer": 52.77, "content_wer_ema": 55.10,
           "L_pres": -3.2, "L_MR": 0.28, "L_gain": 4.1, "total": 1.0}
    config = {"training": {"lr_schedule_on": "content_wer"},
              "loss": {"w_m": 9.62, "w_g": 1.69}}
    assert lr_schedule_metric(val, config) == pytest.approx(55.10)


def test_lower_wer_wins():
    """Lower is better, like every other selection mode."""
    better = {"content_wer_ema": 45.2, "content_wer": 45.2}
    worse = {"content_wer_ema": 65.6, "content_wer": 65.6}
    assert selection_score(better, cfg("content_wer")) < \
           selection_score(worse, cfg("content_wer"))


def test_the_old_modes_are_untouched():
    val = {"L_pres": -4.102, "L_MR": 0.29844, "L_gain": 3.28383, "total": 1.0}
    assert selection_score(val, cfg("present_branch")) == pytest.approx(4.319, abs=1e-3)
    assert selection_score(val, cfg("separation")) == pytest.approx(-4.102)
    assert selection_score(val, cfg("total")) == pytest.approx(1.0)


def test_unknown_mode_names_the_new_one():
    with pytest.raises(ValueError, match="content_wer"):
        selection_score({"total": 1.0}, cfg("nonsense"))


def test_history_carries_the_new_columns():
    assert "content_wer" in VAL_DIAGNOSTICS
    assert "content_wer_ema" in VAL_DIAGNOSTICS
    header = history_header()
    assert "val_content_wer" in header and "val_content_wer_ema" in header


def test_a_row_without_the_probe_is_still_writable():
    """Every run before 2026-09-23 has no probe; those rows must still build."""
    fields = {"total": 1.0, "L_pres": -1.0, "L_MR": 0.2, "L_gain": 3.0,
              "L_abs": -11.0, "L_state": float("nan"), "L_struct": float("nan"),
              "n_present": 261, "n_absent": 139}
    va = {**fields, "epoch": 3, "lr": 0.00025, "w": 0.458}
    row = history_row(fields, va)
    assert len(row) == len(history_header())


def test_probe_is_off_unless_asked_for():
    assert build_content_probe({}, verbose=False) is None
    assert build_content_probe({"content_probe": {"enabled": False}},
                               verbose=False) is None


@pytest.mark.parametrize("split", ["sir0_privval", "eval_private"])
def test_the_probe_refuses_to_steer_on_a_holdout(split):
    with pytest.raises(ValueError, match="holdout"):
        build_content_probe({"content_probe": {"enabled": True, "split": split}},
                            verbose=False)
