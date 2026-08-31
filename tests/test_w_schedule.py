"""The absent-branch warmup schedule. decisions-m2.md 2026-08-25.

Pure arithmetic, so it is cheap to pin exactly -- and worth pinning, because an
off-by-one here changes which epochs the model can earn a reward for silence in,
which is the whole point of the schedule. A schedule that reached full w one
epoch early, or never reached it at all, would look like a training result.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import w_at_epoch  # noqa: E402


def cfg(**sched):
    return {"loss": {"w": 0.458, **({"w_schedule": sched} if sched else {})}}


def test_no_schedule_is_constant_w():
    """An unscheduled config must behave exactly as before the feature existed."""
    c = cfg()
    assert [w_at_epoch(c, e) for e in range(5)] == [0.458] * 5


def test_warmup_holds_at_w_start_then_ramps_then_holds_at_w():
    c = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    got = [round(w_at_epoch(c, e), 6) for e in range(10)]
    #      epochs 0-3 warmup      epochs 4-6 ramp                epochs 7+ full
    assert got[:4] == [0.0] * 4
    assert got[4:7] == [round(0.458 / 3, 6), round(2 * 0.458 / 3, 6), 0.458]
    assert got[7:] == [0.458] * 3


def test_last_ramp_epoch_reaches_w_exactly():
    """The +1 in the ramp fraction. Without it the schedule stops one step short
    and the model never trains at the w every reported number is computed at."""
    c = cfg(w_start=0.0, warmup_epochs=2, ramp_epochs=5)
    assert w_at_epoch(c, 2 + 5 - 1) == pytest.approx(0.458)


def test_zero_ramp_steps_straight_to_w():
    c = cfg(w_start=0.0, warmup_epochs=3, ramp_epochs=0)
    assert [w_at_epoch(c, e) for e in range(5)] == [0.0, 0.0, 0.0, 0.458, 0.458]


def test_zero_warmup_ramps_from_epoch_zero():
    c = cfg(w_start=0.0, warmup_epochs=0, ramp_epochs=2)
    assert w_at_epoch(c, 0) == pytest.approx(0.229)
    assert w_at_epoch(c, 1) == pytest.approx(0.458)


def test_nonzero_w_start_is_honoured():
    """w_start need not be 0 -- a partial warmup is a legitimate arm."""
    c = cfg(w_start=0.1, warmup_epochs=2, ramp_epochs=2)
    assert w_at_epoch(c, 0) == pytest.approx(0.1)
    assert w_at_epoch(c, 2) == pytest.approx(0.1 + 0.5 * (0.458 - 0.1))
    assert w_at_epoch(c, 3) == pytest.approx(0.458)


def test_w_stays_in_the_convex_range():
    """(1 - w) must never go negative: that trains the model to DESTROY the
    target while the curve still descends. Same failure build_loss_fn asserts on."""
    c = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    assert all(0.0 <= w_at_epoch(c, e) <= 1.0 for e in range(50))


def test_out_of_range_w_start_is_rejected():
    with pytest.raises(AssertionError):
        w_at_epoch(cfg(w_start=1.5, warmup_epochs=1, ramp_epochs=1), 0)


def test_resume_epoch_numbering_is_absolute():
    """w depends on the ABSOLUTE epoch, so resuming at epoch 8 of a 4+3 schedule
    trains at full w -- it does not restart the warmup and re-teach silence."""
    c = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    assert w_at_epoch(c, 8) == pytest.approx(0.458)
