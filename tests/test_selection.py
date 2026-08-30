"""Unit tests for model selection (scripts/train.py).

Selection is not the loss and is never backpropagated -- validation runs under
no_grad. It only decides which epoch's weights are kept and when to stop. That
made it easy to get wrong silently: before 2026-08-30 the training total was
reused as the ranking rule, and because `w` = 0.458 makes the absent branch
dominate that total, it kept a checkpoint at 1.13 dB held-out separation from a
run whose epoch 10 reached 2.36 dB.

Each test pins one property of the replacement.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import selection_eligible, selection_score  # noqa: E402

CFG = {"loss": {"w_m": 9.62, "w_g": 1.69},
       "training": {"select_on": "present_branch", "select_abs_max": -10.0}}


def row(L_pres, L_MR=0.19, L_gain=3.7, L_abs=-11.0, total=None):
    w = 0.458
    pres = L_pres + CFG["loss"]["w_m"]*L_MR + CFG["loss"]["w_g"]*L_gain
    return {"L_pres": L_pres, "L_MR": L_MR, "L_gain": L_gain, "L_abs": L_abs,
            "total": (1-w)*pres + w*L_abs if total is None else total}


def cfg(**over):
    c = {"loss": dict(CFG["loss"]), "training": dict(CFG["training"])}
    c["training"].update(over)
    return c


# --- the bug this exists to prevent ---------------------------------------

def test_absent_branch_cannot_outvote_separation():
    """THE REGRESSION TEST. Two epochs: one separates a full dB better, the
    other is merely quieter on absent crops. `val_total` prefers the quiet one;
    the selection score must prefer the separator."""
    good_sep = row(L_pres=-2.35, L_abs=-10.2)
    just_quiet = row(L_pres=-1.13, L_abs=-12.3)
    assert just_quiet["total"] < good_sep["total"]          # the old rule fails
    assert selection_score(good_sep, CFG) < selection_score(just_quiet, CFG)


def test_present_branch_ignores_L_abs_entirely():
    """The absent branch is handled by the eligibility bar, not by being
    reweighted into the score -- so L_abs must not shift the score at all."""
    a, b = row(-2.0, L_abs=-9.9), row(-2.0, L_abs=-25.0)
    assert selection_score(a, CFG) == selection_score(b, CFG)


def test_present_branch_is_the_loss_present_half():
    r = row(-2.0, L_MR=0.2, L_gain=3.0)
    assert selection_score(r, CFG) == pytest.approx(-2.0 + 9.62*0.2 + 1.69*3.0)


# --- the silence bar -------------------------------------------------------

def test_bar_excludes_a_model_that_will_not_shut_up():
    """Epoch 4-5 of the real runs sat at L_abs -5.1 to -6.3 and had the best
    raw separation. Keeping one would mean a model that talks over silence on
    the quarter of trials with no target."""
    assert not selection_eligible(row(-2.4, L_abs=-5.1), CFG)
    assert selection_eligible(row(-2.4, L_abs=-10.2), CFG)


def test_bar_is_inclusive_at_the_threshold():
    assert selection_eligible(row(-2.0, L_abs=-10.0), CFG)


def test_null_bar_disables_the_constraint():
    assert selection_eligible(row(-2.0, L_abs=0.0), cfg(select_abs_max=None))


# --- the other modes stay available and honest -----------------------------

def test_total_mode_reproduces_the_old_behaviour():
    """Kept so runs before 2026-08-30 remain reproducible."""
    r = row(-2.0)
    assert selection_score(r, cfg(select_on="total")) == r["total"]


def test_separation_mode_is_L_pres_alone():
    r = row(-2.0)
    assert selection_score(r, cfg(select_on="separation")) == -2.0


def test_unknown_mode_fails_loudly():
    """A typo in the config must not silently fall back to a different rule --
    that is exactly how the original bug survived unnoticed."""
    with pytest.raises(ValueError, match="select_on"):
        selection_score(row(-2.0), cfg(select_on="val_total"))


def test_missing_w_g_defaults_to_zero():
    """Configs predating L_gain have no w_g and must still score."""
    c = {"loss": {"w_m": 9.62}, "training": {"select_on": "present_branch"}}
    r = {"L_pres": -2.0, "L_MR": 0.2, "L_abs": -11.0, "total": -1.0}
    assert selection_score(r, c) == pytest.approx(-2.0 + 9.62*0.2)
