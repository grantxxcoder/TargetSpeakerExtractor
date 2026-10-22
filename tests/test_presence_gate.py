"""Tests for src/models/presence_gate.py. Equation numbers: the note,
docs/extra/variational-presence-gate.md.

The brute-force test is the one that matters: it computes KL(q || p) from its
definition (13) by enumerating all 2^T on/off patterns, and checks the closed
form (16) against it. If the derivation (including the mean-field step) were
wrong, this would fail.
"""

import itertools
import math
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.presence_gate import (PresenceGate, apply_gate,  # noqa: E402
                                      bernoulli_kl, markov_kl, markov_prior,
                                      objective, supervised_bce)


def brute_force_kl(pi, a11, a01, rho):
    """(13) by enumeration: sum_z q(z) [log q(z) - log p(z)]."""
    A = {(1, 1): a11, (1, 0): 1 - a11, (0, 1): a01, (0, 0): 1 - a01}
    total = 0.0
    for z in itertools.product((0, 1), repeat=len(pi)):
        q = math.prod(p if zt else 1 - p for p, zt in zip(pi, z))
        log_p = math.log(rho if z[0] else 1 - rho)
        log_p += sum(math.log(A[(z[t - 1], z[t])]) for t in range(1, len(z)))
        total += q * (math.log(q) - log_p)
    return total


def test_prior_recovers_the_run_lengths_and_rho():
    a11, a01, rho = markov_prior(d_on=596.0, d_off=490.0)
    assert 1 / (1 - a11) == pytest.approx(596.0)     # (7): E[D] = 1/(1-A11)
    assert 1 / a01 == pytest.approx(490.0)
    assert rho == pytest.approx(596.0 / (596.0 + 490.0))  # check under (9)


def test_prior_rejects_runs_of_one_frame_or_less():
    with pytest.raises(ValueError):
        markov_prior(d_on=1.0, d_off=50.0)


def test_bernoulli_kl_is_zero_only_when_equal():
    a = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64)
    assert torch.allclose(bernoulli_kl(a, 0.5)[1], torch.tensor(0.0, dtype=torch.float64))
    assert (bernoulli_kl(a, 0.5)[[0, 2]] > 0).all()


@pytest.mark.parametrize("seed", range(5))
def test_closed_form_matches_brute_force(seed):
    g = torch.Generator().manual_seed(seed)
    T = 6
    pi = torch.rand(T, generator=g, dtype=torch.float64) * 0.98 + 0.01
    a11, a01, rho = 0.9, 0.2, 0.3
    closed = markov_kl(pi, a11, a01, rho).item()
    assert closed == pytest.approx(brute_force_kl(pi.tolist(), a11, a01, rho),
                                   rel=1e-9)


def test_uniform_prior_reduces_to_eq_22():
    pi = torch.tensor([[0.2, 0.7, 0.5, 0.99]], dtype=torch.float64)
    expected = (math.log(2) + pi * pi.log() + (1 - pi) * (1 - pi).log()).sum(-1)
    assert torch.allclose(markov_kl(pi, 0.5, 0.5, 0.5), expected)


def test_batched_shape():
    pi = torch.rand(3, 500)
    assert markov_kl(pi, 0.998, 0.002, 0.55).shape == (3,)


def test_gate_has_129_parameters():
    assert sum(p.numel() for p in PresenceGate(128).parameters()) == 129


def test_gate_is_band_mean_then_layer_norm_then_linear():
    """Eq. (2): pi_t = sigmoid(w^T hat h_t + b), hat h_t = (h_t - mu_t) / sqrt(s_t^2 + eps), h_t the band mean."""
    torch.manual_seed(0)
    gate = PresenceGate(4)
    u = torch.randn(1, 32, 4, 5)
    h = u.mean(1)
    h_hat = (h - h.mean(1, keepdim=True)) / torch.sqrt(h.var(1, unbiased=False, keepdim=True) + gate.eps)
    w, b = gate.linear.weight.view(-1), gate.linear.bias
    expected = torch.einsum("n,bnt->bt", w, h_hat) + b
    assert torch.allclose(gate(u), expected, atol=1e-5)


def test_gate_ignores_the_feature_scale():
    """e21's features are ~300x larger than at init (2026-10-14); after LN the gate must not see the difference."""
    torch.manual_seed(0)
    gate = PresenceGate(128)
    u = 10.0 * torch.randn(2, 32, 128, 10)
    assert torch.allclose(gate(u), gate(300.0 * u), atol=1e-4)


def test_gate_is_not_saturated_at_e21_scale():
    """Before LN, logits on e21's features were +10 to +56 and pi was pinned at 1 on 51-100 % of frames."""
    torch.manual_seed(0)
    gate = PresenceGate(128)
    logits = gate(600.0 * torch.randn(2, 32, 128, 50))
    pi = torch.sigmoid(logits)
    assert logits.abs().max() < 5.0
    assert not ((pi < 1e-6) | (pi > 1 - 1e-6)).any()


def test_gate_is_causal():
    gate = PresenceGate(8)
    u = torch.randn(2, 32, 8, 10)
    later = u.clone()
    later[..., 6:] = torch.randn(2, 32, 8, 4)
    assert gate(u).shape == (2, 10)
    assert torch.equal(gate(u)[:, :6], gate(later)[:, :6])


def test_gate_construction_leaves_global_rng_alone():
    torch.manual_seed(0)
    a = torch.rand(3)
    torch.manual_seed(0)
    PresenceGate(128)
    assert torch.equal(a, torch.rand(3))


def test_apply_gate_scales_each_frame():
    spec = torch.randn(2, 257, 10, dtype=torch.complex64)
    assert torch.equal(apply_gate(spec, torch.ones(2, 10)), spec)
    assert apply_gate(spec, torch.zeros(2, 10)).abs().max() == 0
    g = torch.ones(2, 10)
    g[:, 3] = 0.5
    assert torch.allclose(apply_gate(spec, g)[..., 3], 0.5 * spec[..., 3])


def test_supervised_bce_matches_the_formula():
    logits = torch.tensor([[0.3, -1.2, 2.0]])
    v = torch.tensor([[1, 0, 1]])
    pi = torch.sigmoid(logits)
    expected = -(v * pi.log() + (1 - v) * (1 - pi).log()).sum(-1)
    assert torch.allclose(supervised_bce(logits, v), expected)


def test_objective_is_eq_12_and_beta_zero_is_eq_23():
    loss = torch.tensor([1.0, 3.0])
    kl = torch.tensor([10.0, 20.0])
    bce = torch.tensor([4.0, 6.0])
    assert objective(loss, kl, bce, beta=0.1, lam=0.5).item() == pytest.approx(2.0 + 1.5 + 2.5)
    assert objective(loss, kl, bce, beta=0.0, lam=0.5).item() == pytest.approx(2.0 + 2.5)
