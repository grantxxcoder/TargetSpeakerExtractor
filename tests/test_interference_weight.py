"""Unit tests for D10, the interference-weighted present term (2026-10-02).

L_pres counts every unit of error energy once. With `w_interf`, the part of the
error that the OTHER speaker's clean stem explains counts w_interf times. Every
failure of that change is silent -- a run that trains the plain objective while
calling itself the arm, a weight that also punishes the target, a NaN on trials
with no other speaker -- so each test pins one property the decision entry relies
on:

  * w_interf = 1 is e21's objective exactly, so the control is a real control
  * the interference part is orthogonal to the target and an exact share of the
    error, so the weighted denominator can never go negative
  * a silent other stem gives exactly zero and a finite gradient
  * leakage costs more than other error of the same energy -- the whole point
  * scale invariance survives the change
  * an arm handed no stem refuses to train rather than training the control
  * the loader pairs each direction with the right other stem
  * train.py refuses configurations in which the term would silently vanish
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402
from src.models.losses import LossBSRNN  # noqa: E402
from scripts.train import (  # noqa: E402
    build_loss_fn, history_row, init_weights_record, other_kwargs, HISTORY_FIELDS)

T = 16000


def _loss(w_interf=1.0):
    return LossBSRNN(wm=9.62, w=0.458, wg=1.69, w_interf=w_interf)


def _signals(seed=0, batch=3):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(batch, T, generator=g)
    other = torch.randn(batch, T, generator=g)
    noise = torch.randn(batch, T, generator=g)
    return target, other, noise


# ---------------------------------------------------------------- the loss --

def test_unit_weight_is_the_plain_term_exactly():
    target, other, noise = _signals()
    output = target + 0.4 * other + 0.2 * noise
    loss = _loss()
    plain = loss._loss_target_present(target, output)
    with_stem = loss._loss_target_present(target, output, s_other=other, w_interf=1.0)
    assert torch.equal(plain, with_stem)


def test_unit_weight_total_is_unchanged_by_passing_the_stem():
    target, other, noise = _signals()
    output = target + 0.4 * other + 0.2 * noise
    absent = torch.tensor([False, False, True])
    mixture = target + other + noise
    target = target.clone()
    target[2] = 0.0                     # an absent crop, as the loader would give
    loss = _loss()
    a, _ = loss(target, output, mixture, absent)
    b, parts = loss(target, output, mixture, absent, s_other=other)
    assert torch.equal(a, b)
    assert parts["L_pres_w"] == pytest.approx(parts["L_pres"])
    assert 0.0 <= parts["interf_share"] <= 1.0


def test_interference_part_is_orthogonal_to_target_and_an_exact_share():
    target, other, noise = _signals(1)
    output = 0.8 * target + 0.5 * other + 0.3 * noise
    loss = _loss()
    e_i = loss._interference_part(target, output, other)
    alpha = (output * target).sum(-1, keepdim=True) / target.pow(2).sum(-1, keepdim=True)
    error = output - alpha * target
    # Orthogonal to the target, so the target is never what gets punished.
    cos = (e_i * target).sum(-1) / (e_i.norm(dim=-1) * target.norm(dim=-1))
    assert cos.abs().max() < 1e-4
    # Pythagoras: |error|^2 = |e_i|^2 + |error - e_i|^2.
    lhs = error.pow(2).sum(-1)
    rhs = e_i.pow(2).sum(-1) + (error - e_i).pow(2).sum(-1)
    torch.testing.assert_close(lhs, rhs, rtol=1e-4, atol=1e-3)


def test_silent_other_stem_gives_zero_and_a_finite_gradient():
    target, _, noise = _signals(2)
    other = torch.zeros_like(target)
    output = (target + 0.3 * noise).requires_grad_(True)
    loss = _loss(2.0)
    e_i = loss._interference_part(target, output, other)
    assert torch.equal(e_i, torch.zeros_like(e_i))
    weighted = loss._loss_target_present(target, output, s_other=other, w_interf=2.0)
    torch.testing.assert_close(weighted, loss._loss_target_present(target, output.detach()))
    weighted.sum().backward()
    assert torch.isfinite(output.grad).all()


def test_leakage_costs_more_than_other_error_of_the_same_energy():
    """The worked example: same error energy, one leaky output, one hissy one.
    The plain term cannot tell them apart; the weighted one must prefer hiss."""
    target, other, noise = _signals(3, batch=1)
    loss = _loss()
    # Remove the target's direction from both, so each output's error IS the
    # added component and the two errors have exactly equal energy.
    def perp(x, ref):
        return x - (x * ref).sum(-1, keepdim=True) / ref.pow(2).sum(-1, keepdim=True) * ref
    other_perp, noise_perp = perp(other, target), perp(noise, target)
    noise_perp = perp(noise_perp, other_perp)                 # no other-speaker part
    noise_perp = noise_perp * other_perp.norm() / noise_perp.norm()
    leaky, hissy = target + 0.3 * other_perp, target + 0.3 * noise_perp

    plain = [float(loss._loss_target_present(target, y)) for y in (leaky, hissy)]
    assert plain[0] == pytest.approx(plain[1], abs=1e-3)
    weighted = [float(loss._loss_target_present(target, y, s_other=other, w_interf=2.0))
                for y in (leaky, hissy)]
    assert weighted[0] > weighted[1] + 1.0          # ~3 dB worse at w_interf = 2
    assert weighted[1] == pytest.approx(plain[1], abs=1e-3)


def test_weighted_term_stays_scale_invariant():
    target, other, noise = _signals(4)
    output = target + 0.4 * other + 0.2 * noise
    loss = _loss()
    ref = loss._loss_target_present(target, output, s_other=other, w_interf=2.0)
    for gain in (0.1, 3.0, 50.0):
        torch.testing.assert_close(
            loss._loss_target_present(target, gain * output, s_other=other, w_interf=2.0),
            ref, rtol=1e-4, atol=1e-3)


def test_arm_without_a_stem_refuses_to_train():
    target, other, noise = _signals()
    loss = _loss(2.0)
    with pytest.raises(ValueError, match="w_interf"):
        loss(target, target + noise, target + other + noise, torch.zeros(3, dtype=torch.bool))


# ---------------------------------------------------------------- the data --

def _write(path, value, n):
    sf.write(path, np.full(n, value, dtype=np.float32), 16000, subtype="FLOAT")


@pytest.fixture
def data_root(tmp_path):
    """Two trials whose stems are distinct constants, so every stem is
    identifiable from its value: target 0.20, interferer 0.05, noise 0.01."""
    n = 8000 * 4
    rows = []
    for tid, cond in (("u-000", "both"), ("u-001", "target_only")):
        d = tmp_path / "rendered" / "unit" / tid
        d.mkdir(parents=True)
        itf = 0.05 if cond == "both" else 0.0
        _write(d / "target.wav", 0.20, n)
        _write(d / "interferer.wav", itf, n)
        _write(d / "mixture.wav", 0.20 + itf + 0.01, n)
        for e in ("enrollment", "interferer_enrollment"):
            _write(d / f"{e}.wav", 0.3, n)
        rows.append({"trial_id": tid, "condition": cond, "target_absent": 0,
                     "sir_db": 0.0, "snr_db": 10.0, "overlap_achieved": 0.5,
                     "regime": "base", "same_gender": 0.0,
                     "interferer_enrollment_phantom": 0})
    pd.DataFrame(rows).to_csv(tmp_path / "unit.csv", index=False)
    return tmp_path


def _dataset(root, **kw):
    return TrialDataset(manifest_csv=root / "unit.csv", data_root=root, split="unit",
                        chunk_s=0.5, sample_rate=16000, seed=42, random_crop=False, **kw)


def test_each_direction_carries_the_other_speakers_stem(data_root):
    target_dir, interferer_dir = _dataset(data_root, both_directions=True)[0]
    assert target_dir["direction"] == "target"
    assert torch.allclose(target_dir["other"], torch.full_like(target_dir["other"], 0.05))
    assert interferer_dir["direction"] == "interferer"
    assert torch.allclose(interferer_dir["other"], torch.full_like(interferer_dir["other"], 0.20))


def test_other_stem_is_zero_when_the_interferer_is_not_loaded(data_root):
    (example,) = _dataset(data_root)[0]
    assert torch.equal(example["other"], torch.zeros_like(example["target"]))


def test_target_only_trial_has_a_silent_other_stem(data_root):
    target_dir, _ = _dataset(data_root, both_directions=True)[1]
    assert torch.equal(target_dir["other"], torch.zeros_like(target_dir["other"]))


# ---------------------------------------------------------------- train.py --

def _config(w_interf, both_directions=True, remix_gains=False, **loss):
    return {"data": {"sample_rate": 16000, "both_directions": both_directions,
                     "remix_gains": remix_gains},
            "loss": {"w": 0.458, "w_m": 9.62, "w_g": 1.69, "tau_pres": 0.001,
                     "tau_abs": 0.01, "p": 0.3, "windows_ms": [8, 16, 32, 64],
                     "w_interf": w_interf, **loss}}


def test_build_loss_fn_passes_the_weight_and_defaults_to_one():
    assert build_loss_fn(_config(2.0)).w_interf == 2.0
    config = _config(1.0)
    del config["loss"]["w_interf"]
    assert build_loss_fn(config).w_interf == 1.0


def test_build_loss_fn_refuses_a_weight_with_no_stem_on_disk():
    with pytest.raises(AssertionError, match="both_directions"):
        build_loss_fn(_config(2.0, both_directions=False, remix_gains=False))
    build_loss_fn(_config(1.0, both_directions=False))          # the default is fine
    build_loss_fn(_config(2.0, both_directions=False, remix_gains=True))


def test_build_loss_fn_refuses_non_positive_weights():
    with pytest.raises(AssertionError):
        build_loss_fn(_config(0.0))


def test_other_kwargs_only_for_losses_that_take_the_stem():
    batch = {"other": torch.zeros(2, 4)}
    assert set(other_kwargs(_loss(), batch, "cpu")) == {"s_other"}

    class NoStem:
        ACCEPTS_OTHER = False
    assert other_kwargs(NoStem(), batch, "cpu") == {}
    assert other_kwargs(_loss(), {}, "cpu") == {}


def test_history_row_tolerates_rows_without_the_new_columns():
    old = {k: 0.0 for k in HISTORY_FIELDS if k not in ("L_pres_w", "interf_share")}
    row = history_row(old, {**old, "epoch": 3, "lr": 1e-4})
    assert len(row) == 1 + 2 * len(HISTORY_FIELDS) + 2 + 4
    assert math.isnan(row[1 + HISTORY_FIELDS.index("interf_share")])


def test_init_weights_record(tmp_path):
    assert init_weights_record({"training": {}}) is None
    path = tmp_path / "w.pt"
    path.write_bytes(b"abc")
    record = init_weights_record({"training": {"init_weights": str(path)}})
    assert record == {"path": str(path), "md5": "900150983cd24fb0d6963f7d28e17f72"}
    with pytest.raises(FileNotFoundError):
        init_weights_record({"training": {"init_weights": str(tmp_path / "missing.pt")}})
