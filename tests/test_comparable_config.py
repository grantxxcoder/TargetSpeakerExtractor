"""The resume check must ignore where the data was mounted, and nothing else.

Every checkpoint since 2026-09-23 carries content_probe.data_root /
manifest_dir (filled from the CLI by train.py's main), and the config file does
not, so a plain != refused the first resume of the content_wer run.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train import comparable_config  # noqa: E402


def _cfg(**probe):
    return {"seed": 42, "content_probe": {"enabled": True, "n_trials": 40, **probe}}


def test_mount_paths_do_not_count():
    saved = _cfg(data_root="/kaggle/input/x/data", manifest_dir="/kaggle/input/x/data/manifests")
    assert comparable_config(saved) == comparable_config(_cfg())


def test_a_real_setting_still_counts():
    assert comparable_config(_cfg(n_trials=80)) != comparable_config(_cfg())


def test_does_not_mutate_its_input():
    saved = _cfg(data_root="/kaggle/input/x/data")
    comparable_config(saved)
    assert saved["content_probe"]["data_root"] == "/kaggle/input/x/data"


def test_config_without_a_probe_block():
    assert comparable_config({"seed": 42}) == {"seed": 42}
    assert comparable_config(None) == {}
