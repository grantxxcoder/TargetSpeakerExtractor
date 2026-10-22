"""The holdout guards: one shared list, and every guard reads it.

decisions-m8.md 2026-10-12: until then three scripts kept their own copies and
none listed ami_eval or eval_public.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.holdouts import HOLDOUT_SPLITS  # noqa: E402


def test_the_four_reported_holdouts_are_listed():
    assert HOLDOUT_SPLITS == {"sir0_privval", "eval_private", "eval_public", "ami_eval"}


def test_train_uses_the_shared_list_not_a_copy():
    from scripts import train
    assert train.HOLDOUT_SPLITS is HOLDOUT_SPLITS


@pytest.mark.parametrize("split", ["sir0pub", "sir0priv", "ami"])
def test_test_splits_are_render_only_because_their_val_half_is_a_holdout(split):
    from scripts.train import RENDER_ONLY_SPLITS, SPLIT_MANIFESTS, get_data_loaders
    assert SPLIT_MANIFESTS[split][1][0] in HOLDOUT_SPLITS
    assert split in RENDER_ONLY_SPLITS
    with pytest.raises(ValueError, match="rendering estimates only"):
        get_data_loaders(split, None, None, None)


def test_no_trainable_split_validates_on_a_holdout():
    from scripts.train import RENDER_ONLY_SPLITS, SPLIT_MANIFESTS
    for name, (_, (val_manifest, _)) in SPLIT_MANIFESTS.items():
        if name not in RENDER_ONLY_SPLITS:
            assert val_manifest not in HOLDOUT_SPLITS, name


def test_no_split_trains_on_a_holdout():
    from scripts.train import SPLIT_MANIFESTS
    for (train, audio), _ in SPLIT_MANIFESTS.values():
        assert train not in HOLDOUT_SPLITS and audio not in HOLDOUT_SPLITS
