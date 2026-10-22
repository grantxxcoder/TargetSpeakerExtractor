"""LossBSRNNPresence (src/models/losses_presence.py), eq. (12) of
docs/extra/variational-presence-gate.md.

What these pin: the total is exactly base + beta * KL + lam * BCE; unlabelled
crops are left out of the BCE; gradient from both new terms reaches the gate's
logits; and a call without the gate's inputs fails loudly instead of quietly
training without them.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.losses import LossBSRNN  # noqa: E402
from src.models.losses_presence import LossBSRNNPresence  # noqa: E402
from src.models.presence_gate import markov_kl, markov_prior, supervised_bce  # noqa: E402

B, N, T = 4, 16000, 126
D_ON, D_OFF = 596.0, 490.0


def batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, N, generator=g)
    target[3] = 0.0                                   # one target-absent crop
    mixture = target + 0.5 * torch.randn(B, N, generator=g)
    output = target + 0.1 * torch.randn(B, N, generator=g)
    absent = torch.tensor([False, False, False, True])
    logits = torch.randn(B, T, generator=g, requires_grad=True)
    v = (torch.rand(B, T, generator=g) > 0.5).float()
    v[3] = 0.0
    v_valid = torch.tensor([True, True, False, True])
    return target, output, mixture, absent, logits, v, v_valid


def make(beta=0.1, lam=0.5):
    return LossBSRNNPresence(wm=6.6, w=0.458, beta=beta, lam=lam, d_on=D_ON, d_off=D_OFF)


def test_total_is_base_plus_the_two_terms():
    target, output, mixture, absent, logits, v, v_valid = batch()
    total, parts = make()(target, output, mixture, absent, logits, v, v_valid)
    base, _ = LossBSRNN(wm=6.6, w=0.458)(target, output, mixture, absent)
    a11, a01, rho = markov_prior(D_ON, D_OFF)
    kl = markov_kl(torch.sigmoid(logits), a11, a01, rho).mean()
    bce = supervised_bce(logits[v_valid], v[v_valid]).mean()
    assert total.item() == pytest.approx((base + 0.1 * kl + 0.5 * bce).item(), rel=1e-6)
    assert parts["L_kl"] == pytest.approx(kl.item(), rel=1e-6)
    assert parts["L_bce"] == pytest.approx(bce.item(), rel=1e-6)


def test_unlabelled_crop_is_left_out_of_the_bce():
    target, output, mixture, absent, logits, v, v_valid = batch()
    _, with_flag = make()(target, output, mixture, absent, logits, v, v_valid)
    v_changed = v.clone()
    v_changed[2] = 1.0 - v_changed[2]                 # crop 2 is unlabelled
    _, changed = make()(target, output, mixture, absent, logits, v_changed, v_valid)
    assert with_flag["L_bce"] == changed["L_bce"]


def test_no_labelled_crop_gives_nan_bce_not_a_crash():
    target, output, mixture, absent, logits, v, _ = batch()
    total, parts = make()(target, output, mixture, absent, logits, v,
                          torch.zeros(B, dtype=torch.bool))
    assert math.isnan(parts["L_bce"]) and math.isfinite(total.item())


def test_both_terms_send_gradient_to_the_gate():
    for beta, lam in ((1.0, 0.0), (0.0, 1.0)):
        target, output, mixture, absent, logits, v, v_valid = batch()
        total, _ = make(beta, lam)(target, output, mixture, absent, logits, v, v_valid)
        total.backward()
        assert logits.grad.abs().sum() > 0


def test_missing_gate_inputs_fail_loudly():
    target, output, mixture, absent, *_ = batch()
    with pytest.raises(RuntimeError):
        make()(target, output, mixture, absent)
