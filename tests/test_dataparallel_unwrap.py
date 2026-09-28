"""`nn.DataParallel` must not change what a checkpoint looks like.
decisions-m2.md 2026-09-21, decisions-pending.md E7.

WHY THESE TESTS EXIST
---------------------
E7's one real trap. `DataParallel` forwards `__call__` but NOT attribute
access, and it prefixes every `state_dict()` key with `module.`. So a run that
saved from the wrapper would write checkpoints that:

  - `--resume` cannot load (it builds an unwrapped BSRNN_TFMAP),
  - `make_estimates.py` cannot load, for the same reason,
  - are shaped differently from every checkpoint already on disk.

The failure is silent at save time and only appears hours later when something
tries to read the file, which is the worst possible place to find it. It also
would not show up on the laptop, where `device_count()` is 0 and the wrapper is
never applied -- so only a Kaggle run would hit it.

These tests pin the CONTRACT (a checkpoint is always unwrapped, and attribute
access always reaches the real module), not the wiring, so an edit that reaches
for `model.state_dict()` again fails here rather than on the second T4.

`DataParallel` needs no GPU to construct when given an empty device list, so
these run anywhere.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "train_mod", Path(__file__).resolve().parents[1] / "scripts" / "train.py")
train_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_mod)


class Tiny(nn.Module):
    """Stands in for BSRNN_TFMAP: one parameter and one non-forwarded attribute."""

    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)
        self.band_widths = [2, 2]

    def forward(self, x):
        return self.lin(x)


def test_unwrap_returns_the_module_itself_when_not_wrapped():
    m = Tiny()
    assert train_mod.unwrap(m) is m


def test_unwrap_reaches_through_dataparallel():
    m = Tiny()
    assert train_mod.unwrap(nn.DataParallel(m)) is m


def test_dataparallel_really_does_prefix_state_dict_keys():
    """The premise. If torch ever stops doing this the tests below are vacuous,
    so assert the behaviour we are defending against actually exists."""
    m = Tiny()
    assert all(k.startswith("module.") for k in nn.DataParallel(m).state_dict())
    assert not any(k.startswith("module.") for k in m.state_dict())


def test_checkpoint_saved_through_unwrap_loads_into_a_bare_model(tmp_path):
    """The whole point: a two-card run's checkpoint must load on one card."""
    trained = Tiny()
    with torch.no_grad():
        trained.lin.weight.fill_(0.5)
    wrapped = nn.DataParallel(trained)

    path = tmp_path / "ckpt.pt"
    torch.save({"model": train_mod.unwrap(wrapped).state_dict()}, path)

    fresh = Tiny()
    fresh.load_state_dict(torch.load(path, weights_only=True)["model"])
    assert torch.allclose(fresh.lin.weight, torch.full((4, 4), 0.5))


def test_saving_the_wrapper_directly_is_what_breaks_resume(tmp_path):
    """Negative control -- pins WHY unwrap() is needed, not just that it works."""
    wrapped = nn.DataParallel(Tiny())
    path = tmp_path / "bad.pt"
    torch.save({"model": wrapped.state_dict()}, path)

    with pytest.raises(RuntimeError, match="module"):
        Tiny().load_state_dict(torch.load(path, weights_only=True)["model"])


def test_attribute_access_needs_unwrap():
    """`log_results` reads model.band_widths and `oracle_mask_and_mag` reads
    model.stft. Neither is forwarded."""
    wrapped = nn.DataParallel(Tiny())
    with pytest.raises(AttributeError):
        _ = wrapped.band_widths
    assert train_mod.unwrap(wrapped).band_widths == [2, 2]


def test_n_hidden_defaults_to_one_when_the_key_is_absent():
    """Absent key => 1 => the architecture every run before 2026-09-21 trained.
    A default of 2 would silently change what an old config means."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(open(root / "experiments/configs/bsrnn_baseline.yaml"))
    assert "n_hidden" not in base["model"]["mask"]

    ref = yaml.safe_load(open(root / "experiments/configs/bsrnn_wesep_ref.yaml"))
    assert ref["model"]["mask"]["n_hidden"] == 2

    # Assert on PARAMETERS, not on an attribute: Estimator consumes n_hidden in
    # a loop and does not keep it, so only the built shape shows whether the
    # config key reached the model. mlp_hidden is 384 in both configs, so the
    # estimator can differ for no other reason.
    n_base = sum(p.numel() for p in train_mod.build_model(base).estimator.parameters())
    n_ref = sum(p.numel() for p in train_mod.build_model(ref).estimator.parameters())
    assert n_base == 2_187_014, n_base          # the shipped baseline, unchanged
    assert n_ref == 6_917_894, n_ref            # wesep depth, decisions-m1.md ladder


def test_w_schedule_holds_steps_times_batch_constant():
    """decisions-m2.md 2026-09-21. The warmup is indexed in STEPS, so doubling
    the batch without halving the step counts would cover 2x the audio."""
    import yaml
    root = Path(__file__).resolve().parents[1]
    base = yaml.safe_load(open(root / "experiments/configs/bsrnn_baseline.yaml"))
    ref = yaml.safe_load(open(root / "experiments/configs/bsrnn_wesep_ref.yaml"))

    for key in ("warmup_steps", "ramp_steps"):
        assert (base["loss"]["w_schedule"][key] * base["data"]["batch_size"]
                == ref["loss"]["w_schedule"][key] * ref["data"]["batch_size"]), key
