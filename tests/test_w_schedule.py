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
from scripts.train import (  # noqa: E402
    schedule_in_steps,
    w_at_epoch,
    w_at_step,
)


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


# ---------------------------------------------------------------------------
# STEP-INDEXED SCHEDULE, 2026-09-03.
#
# The schedule used to be indexed in epochs, so its length in gradient steps
# moved with the size of the training set -- 11,606 steps at 4,976 trials,
# 23,212 at 9,955. These tests pin the two properties that fix: the curve is
# unchanged in shape, and it no longer depends on steps_per_epoch.
# decisions-m2.md 2026-09-03.
# ---------------------------------------------------------------------------


def cfg_steps(**sched):
    return {"loss": {"w": 0.458, **({"w_schedule": sched} if sched else {})}}


def test_step_schedule_shape():
    """warmup flat, ramp linear, full w from warmup + ramp onwards."""
    c = cfg_steps(w_start=0.0, warmup_steps=4, ramp_steps=3)
    got = [round(w_at_step(c, t, steps_per_epoch=10), 6) for t in range(9)]
    assert got == [0.0, 0.0, 0.0, 0.0,
                   round(0.458 / 3, 6), round(2 * 0.458 / 3, 6), 0.458,
                   0.458, 0.458]


def test_step_schedule_matches_epoch_schedule_at_epoch_ends():
    """THE EQUIVALENCE, and its one honest limit.

    The epoch-indexed schedule is a STAIRCASE: it holds w flat for a whole
    epoch, then jumps. The step-indexed one is a CONTINUOUS ramp over the same
    span. They cannot agree everywhere, and they agree exactly at the LAST STEP
    of each epoch -- which is what pins that the two forms are the same curve
    rather than a rescaled one.

    Consequence, stated because it is a real difference from the 2026-09-01
    baseline: within the ramp the mean weight is now ~0.5 w rather than the
    staircase's ~0.67 w, so slightly LESS absent-branch pressure is applied
    during the ramp. The endpoints -- where warmup ends and where full w is
    reached -- are identical. decisions-m2.md 2026-09-03.
    """
    n = 1658                      # steps/epoch at the 4,976-trial baseline
    legacy = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    stepwise = cfg_steps(w_start=0.0, warmup_steps=4 * n, ramp_steps=3 * n)
    for e in range(10):
        last_step_of_epoch = (e + 1) * n - 1
        assert w_at_step(stepwise, last_step_of_epoch, n) == pytest.approx(
            w_at_epoch(legacy, e)), f"epoch {e}"


def test_step_and_epoch_forms_share_their_endpoints():
    """Warmup ends and full w arrives at the SAME step under both forms.

    These two boundaries are the schedule's whole job -- when silence stops
    being free, and when the objective is finally the one being reported.
    """
    n = 1658
    legacy = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    stepwise = cfg_steps(w_start=0.0, warmup_steps=4 * n, ramp_steps=3 * n)
    assert schedule_in_steps(legacy, n)[:2] == schedule_in_steps(stepwise, n)[:2]
    # last warmup step is still w_start under both
    assert w_at_step(stepwise, 4 * n - 1, n) == 0.0
    # and full w from the end of the ramp onwards
    assert w_at_step(stepwise, 7 * n - 1, n) == pytest.approx(0.458)
    assert w_at_step(stepwise, 7 * n, n) == pytest.approx(0.458)


def test_ramp_is_continuous_not_a_staircase():
    """The step form must actually move WITHIN an epoch, or nothing changed."""
    n = 1658
    stepwise = cfg_steps(w_start=0.0, warmup_steps=4 * n, ramp_steps=3 * n)
    inside = [w_at_step(stepwise, 4 * n + k, n) for k in (0, n // 2, n - 1)]
    assert inside[0] < inside[1] < inside[2], inside


def test_step_schedule_is_invariant_to_dataset_size():
    """THE WHOLE POINT. The same step gets the same w whatever an epoch costs.

    The epoch-indexed form fails this: at 2x the data its warmup is 2x as long
    in the unit the optimiser actually moves in.
    """
    c = cfg_steps(w_start=0.0, warmup_steps=6632, ramp_steps=4974)
    for t in (0, 6631, 6632, 9000, 11605, 11606, 50000):
        assert w_at_step(c, t, 1658) == w_at_step(c, t, 3318)


def test_epoch_schedule_is_not_invariant_to_dataset_size():
    """The bug, pinned so nobody 'simplifies' the epoch path back into the loop."""
    c = cfg(w_start=0.0, warmup_epochs=4, ramp_epochs=3)
    assert schedule_in_steps(c, 1658) == (6632, 4974, 0.0, 0.458)
    assert schedule_in_steps(c, 3318) == (13272, 9954, 0.0, 0.458)


def test_baseline_config_resolves_to_the_09_01_schedule():
    """The shipped config must reproduce the baseline run's schedule exactly."""
    import yaml
    cfg_path = Path(__file__).resolve().parents[1] / "experiments/configs/bsrnn_baseline.yaml"
    loaded = yaml.safe_load(cfg_path.read_text())
    warmup, ramp, w_start, w_final = schedule_in_steps(loaded, steps_per_epoch=1658)
    # 4 and 3 epochs at the 4,976-trial baseline's 1,658 steps/epoch.
    assert (warmup, ramp) == (4 * 1658, 3 * 1658)
    assert w_start == 0.0
    # Invariant to the data size it is now run at.
    assert schedule_in_steps(loaded, steps_per_epoch=3318)[:2] == (warmup, ramp)


def test_mixing_units_is_refused():
    """Silently honouring one unit and ignoring the other is the failure mode."""
    c = cfg_steps(w_start=0.0, warmup_steps=100, ramp_epochs=3)
    with pytest.raises(AssertionError, match="mixes units"):
        w_at_step(c, 0, steps_per_epoch=10)


def test_no_schedule_is_constant_w_by_step():
    c = cfg_steps()
    assert [w_at_step(c, t, 10) for t in range(5)] == [0.458] * 5
    assert schedule_in_steps(c, 10) is None


def test_legacy_entry_point_refuses_a_step_schedule():
    """w_at_epoch() reading warmup_steps as epochs would be silently wrong."""
    c = cfg_steps(w_start=0.0, warmup_steps=6632, ramp_steps=4974)
    with pytest.raises(AssertionError, match="legacy epoch-indexed"):
        w_at_epoch(c, 0)


def test_ramp_of_zero_steps_jumps_straight_to_full_w():
    c = cfg_steps(w_start=0.0, warmup_steps=100, ramp_steps=0)
    assert w_at_step(c, 99, 10) == 0.0
    assert w_at_step(c, 100, 10) == pytest.approx(0.458)
