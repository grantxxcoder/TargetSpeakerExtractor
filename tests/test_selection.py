"""Unit tests for the silence bar in model selection (scripts/train.py).

Selection is not the loss and is never backpropagated -- validation runs under
no_grad. It only decides which epoch's weights are kept and when to stop. The
ranking is the validation word error rate (tests/test_content_wer_selection.py);
this file pins the bar an epoch must clear before it may be ranked at all.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train import selection_eligible  # noqa: E402

CFG = {"training": {"select_on": "content_wer", "select_abs_max": -10.0}}


def row(L_abs):
    return {"L_abs": L_abs}


def cfg(**over):
    return {"training": {**CFG["training"], **over}}


def test_bar_excludes_a_model_that_will_not_shut_up():
    """Epoch 4-5 of the real runs sat at L_abs -5.1 to -6.3 and had the best
    raw separation. Keeping one would mean a model that talks over silence on
    the quarter of trials with no target."""
    assert not selection_eligible(row(-5.1), CFG)
    assert selection_eligible(row(-10.2), CFG)


def test_bar_is_inclusive_at_the_threshold():
    assert selection_eligible(row(-10.0), CFG)


def test_null_bar_disables_the_constraint():
    assert selection_eligible(row(0.0), cfg(select_abs_max=None))
