"""The holdout guards: one shared list, and every guard reads it.

decisions-m8.md 2026-10-12: until then three scripts kept their own copies and
none listed ami_eval or eval_public.
"""
import subprocess
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


def test_sir0ext_is_render_only_because_its_val_half_is_a_holdout():
    from scripts.train import RENDER_ONLY_SPLITS, SPLIT_MANIFESTS, get_data_loaders
    assert SPLIT_MANIFESTS["sir0ext"][1][0] in HOLDOUT_SPLITS
    assert "sir0ext" in RENDER_ONLY_SPLITS
    with pytest.raises(ValueError, match="rendering estimates only"):
        get_data_loaders("sir0ext", None, None, None)


def test_no_trainable_split_validates_on_a_holdout():
    from scripts.train import RENDER_ONLY_SPLITS, SPLIT_MANIFESTS
    for name, (_, (val_manifest, _)) in SPLIT_MANIFESTS.items():
        if name not in RENDER_ONLY_SPLITS:
            assert val_manifest not in HOLDOUT_SPLITS, name


@pytest.mark.parametrize("split", sorted(HOLDOUT_SPLITS))
def test_select_by_wer_refuses_every_holdout(split):
    run = subprocess.run(
        [sys.executable, str(ROOT / "scripts/select_by_wer.py"), "--config", "x.yaml",
         "--checkpoints", "nothing-*.pt", "--split", split, "--dry-run"],
        capture_output=True, text=True, cwd=ROOT)
    assert run.returncode != 0
    assert "holdout" in run.stderr


def test_m8_splits_train_on_ami_and_never_on_ami_eval():
    """decisions-m8.md 2026-10-12: amift/amimix train on ami_train and validate
    on sir0_val; amitr only renders ami_train for the Gemini check."""
    from scripts.train import RENDER_ONLY_SPLITS, SPLIT_MANIFESTS, get_data_loaders
    assert SPLIT_MANIFESTS["amift"] == (("amift_train", "ami_train"), ("sir0_val", "sir0_val"))
    assert SPLIT_MANIFESTS["amimix"] == (("amimix_train", "ami_train"), ("sir0_val", "sir0_val"))
    for split in ("amitr", "amitrchk"):
        assert split in RENDER_ONLY_SPLITS
        with pytest.raises(ValueError, match="rendering estimates only"):
            get_data_loaders(split, None, None, None)
    for (train, audio), _ in SPLIT_MANIFESTS.values():
        assert train != "ami_eval" and audio != "ami_eval"
