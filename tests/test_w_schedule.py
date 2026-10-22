"""The absent-branch warmup schedule, in optimiser steps. decisions-m2.md 2026-08-25 and 2026-09-03.

Pure arithmetic, so it is cheap to pin exactly -- and worth pinning, because an
off-by-one here changes which steps the model can earn a reward for silence in,
which is the whole point of the schedule. A schedule that reached full w early,
or never reached it at all, would look like a training result.
"""

import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train import schedule_in_steps, w_at_step  # noqa: E402


def cfg(**sched):
    return {"loss": {"w": 0.458, **({"w_schedule": sched} if sched else {})}}


def test_no_schedule_is_constant_w():
    c = cfg()
    assert [w_at_step(c, t) for t in range(5)] == [0.458] * 5
    assert schedule_in_steps(c) is None


def test_step_schedule_shape():
    """warmup flat, ramp linear, full w from warmup + ramp onwards."""
    c = cfg(w_start=0.0, warmup_steps=4, ramp_steps=3)
    got = [round(w_at_step(c, t), 6) for t in range(9)]
    assert got == [0.0, 0.0, 0.0, 0.0,
                   round(0.458 / 3, 6), round(2 * 0.458 / 3, 6), 0.458,
                   0.458, 0.458]


def test_last_ramp_step_reaches_w_exactly():
    """The +1 in the ramp fraction. Without it the schedule stops one step short
    and the model never trains at the w every reported number is computed at."""
    c = cfg(w_start=0.0, warmup_steps=2, ramp_steps=5)
    assert w_at_step(c, 2 + 5 - 1) == pytest.approx(0.458)


def test_ramp_of_zero_steps_jumps_straight_to_full_w():
    c = cfg(w_start=0.0, warmup_steps=100, ramp_steps=0)
    assert w_at_step(c, 99) == 0.0
    assert w_at_step(c, 100) == pytest.approx(0.458)


def test_zero_warmup_ramps_from_step_zero():
    c = cfg(w_start=0.0, warmup_steps=0, ramp_steps=2)
    assert w_at_step(c, 0) == pytest.approx(0.229)
    assert w_at_step(c, 1) == pytest.approx(0.458)


def test_nonzero_w_start_is_honoured():
    c = cfg(w_start=0.1, warmup_steps=2, ramp_steps=2)
    assert w_at_step(c, 0) == pytest.approx(0.1)
    assert w_at_step(c, 2) == pytest.approx(0.1 + 0.5 * (0.458 - 0.1))
    assert w_at_step(c, 3) == pytest.approx(0.458)


def test_w_stays_in_the_convex_range():
    """(1 - w) must never go negative: that trains the model to DESTROY the
    target while the curve still descends."""
    c = cfg(w_start=0.0, warmup_steps=4, ramp_steps=3)
    assert all(0.0 <= w_at_step(c, t) <= 1.0 for t in range(50))


def test_out_of_range_w_start_is_rejected():
    with pytest.raises(AssertionError):
        w_at_step(cfg(w_start=1.5, warmup_steps=1, ramp_steps=1), 0)


def test_resume_step_numbering_is_absolute():
    """w depends on the ABSOLUTE step, so a resume past the ramp trains at full
    w -- it does not restart the warmup and re-teach silence."""
    c = cfg(w_start=0.0, warmup_steps=4, ramp_steps=3)
    assert w_at_step(c, 8) == pytest.approx(0.458)


def test_epoch_units_are_refused():
    """The epoch-indexed form was removed 2026-10-22. Reading it as "no schedule"
    would silently train a different objective."""
    with pytest.raises(AssertionError, match="removed"):
        w_at_step(cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3), 0)


@pytest.mark.parametrize("name", ["bsrnn_baseline_matched", "bsrnn_cue_context"])
def test_final_configs_resolve_to_the_09_01_schedule(name):
    """Warmup 4 and ramp 3 epochs at the 4,976-trial run's 1,658 steps/epoch."""
    loaded = yaml.safe_load((ROOT / f"experiments/configs/{name}.yaml").read_text())
    assert schedule_in_steps(loaded) == (4 * 1658, 3 * 1658, 0.0, 0.458)
