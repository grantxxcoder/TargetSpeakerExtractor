"""What `ReduceLROnPlateau` is allowed to watch. decisions-m2.md 2026-09-21.

WHY THESE TESTS EXIST
---------------------
The scheduler stepped on `val_loss["total"]` for three weeks after selection
stopped doing so. `total` contains `L_abs`, and `L_abs` rewards silence, so it
keeps falling while the model mutes itself -- which is exactly why
`selection_score` refuses to rank on it. The 14.73 M run halved its learning
rate at epoch idx 13 on that number.

The failure is silent: a schedule watching the wrong quantity produces a run
that looks fine and a checkpoint chosen under a learning rate that dropped for
the wrong reason. So these tests pin the PROPERTY that matters -- a model going
quiet must not be able to look like progress -- not just the plumbing.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_mod", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)


def cfg(mode=None, w=0.458, w_m=9.62, w_g=1.69):
    t = {"select_on": "present_branch"}
    if mode is not None:
        t["lr_schedule_on"] = mode
    return {"training": t, "loss": {"w": w, "w_m": w_m, "w_g": w_g}}


def row(L_pres=-2.9, L_MR=0.18, L_gain=3.4, L_abs=-11.0):
    w, w_m, w_g = 0.458, 9.62, 1.69
    total = (1 - w) * (L_pres + w_m * L_MR + w_g * L_gain) + w * L_abs
    return {"total": total, "L_pres": L_pres, "L_MR": L_MR,
            "L_gain": L_gain, "L_abs": L_abs}


def test_defaults_to_total_so_old_configs_reproduce():
    """A config with no lr_schedule_on key must behave exactly as before."""
    r = row()
    assert train_mod.lr_schedule_metric(r, cfg()) == pytest.approx(r["total"])


def test_present_branch_mode_excludes_the_silence_term():
    r = row()
    got = train_mod.lr_schedule_metric(r, cfg("present_branch"))
    assert got == pytest.approx(-2.9 + 9.62 * 0.18 + 1.69 * 3.4)
    assert got != pytest.approx(r["total"])


def test_THE_BUG_going_quiet_improves_total_but_not_present_branch():
    """The whole reason this exists.

    Two epochs identical on every present-crop term; the second is merely
    quieter on crops where the target never speaks. `total` calls that an
    improvement and would keep the learning rate up. `present_branch` sees
    no change at all.
    """
    before, after = row(L_abs=-11.0), row(L_abs=-14.0)   # same everything else

    assert after["total"] < before["total"], "premise: total rewards silence"

    t_before = train_mod.lr_schedule_metric(before, cfg("total"))
    t_after = train_mod.lr_schedule_metric(after, cfg("total"))
    assert t_after < t_before, "total mode inherits the flaw, by design"

    p_before = train_mod.lr_schedule_metric(before, cfg("present_branch"))
    p_after = train_mod.lr_schedule_metric(after, cfg("present_branch"))
    assert p_after == pytest.approx(p_before), "present_branch must be unmoved"


def test_separation_mode_is_reachable_but_ignores_reconstruction():
    """Available, and NOT recommended: L_pres alone cannot see output level or
    spectral match, so a model that separates better while reconstructing worse
    looks like progress. Measured and rejected 2026-08-30."""
    good_level, bad_level = row(L_gain=3.4), row(L_gain=9.9)
    s1 = train_mod.lr_schedule_metric(good_level, cfg("separation"))
    s2 = train_mod.lr_schedule_metric(bad_level, cfg("separation"))
    assert s1 == pytest.approx(s2), "separation mode is blind to L_gain"

    p1 = train_mod.lr_schedule_metric(good_level, cfg("present_branch"))
    p2 = train_mod.lr_schedule_metric(bad_level, cfg("present_branch"))
    assert p2 > p1, "present_branch penalises the wrong output level"


def test_schedule_and_selector_cannot_drift_apart():
    """The arithmetic is selection_score's, reused. If someone re-derives it
    here the two will disagree again, which is the bug this replaced."""
    r = row()
    for mode in ("total", "present_branch", "separation"):
        assert train_mod.lr_schedule_metric(r, cfg(mode)) == pytest.approx(
            train_mod.selection_score(r, {"training": {"select_on": mode},
                                          "loss": {"w_m": 9.62, "w_g": 1.69}}))
