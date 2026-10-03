"""Unit tests for the artefact-weighted present term (AB-SDR, 2026-10-03).

L_pres splits its error into the BSS_EVAL parts (Vincent et al. 2006): the other
speaker, the noise, and the artefact that no source explains. `w_artif` counts
the artefact part w_artif times (Ochiai et al. 2024). Each test pins one property
the 2x2 design in decisions-m2.md 2026-10-03 relies on:

  * w_artif = 1 is the D10 term exactly, at either w_interf, so the two runs
    already trained remain the 2x2's other two cells
  * the three parts are mutually orthogonal and sum to the error, so the
    weighted denominator is an exact re-weighting and never negative
  * an output built only from the three sources has no artefact
  * silent stems give zero parts and finite gradients
  * an artefact costs more than noise of the same energy -- the whole point
  * the energy weight is Ochiai's alpha squared, as the configs claim
  * the noise recovered inside __call__ is the true noise
  * scale invariance survives, and an arm with no stem refuses to train
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.losses import LossBSRNN  # noqa: E402
from scripts.train import build_loss_fn  # noqa: E402

T = 16000


def _loss(w_interf=1.0, w_artif=1.0):
    return LossBSRNN(wm=9.62, w=0.458, wg=1.69, w_interf=w_interf, w_artif=w_artif)


def _signals(seed=0, batch=3):
    g = torch.Generator().manual_seed(seed)
    return tuple(torch.randn(batch, T, generator=g) for _ in range(4))   # s, d, n, artefact


def _energy(x):
    return x.pow(2).sum(-1)


def _parts(loss, s, y, d, n):
    alpha = (y * s).sum(-1, keepdim=True) / _energy(s).unsqueeze(-1)
    return (y - alpha * s, loss._interference_part(s, y, d),
            loss._noise_part(s, y, d, n), loss._artefact_part(s, y, d, n))


# ------------------------------------------------------------ the control --

@pytest.mark.parametrize("w_interf", [1.0, 2.0])
def test_unit_artefact_weight_is_the_d10_term_exactly(w_interf):
    s, d, n, a = _signals()
    y = s + 0.4 * d + 0.2 * n + 0.3 * a
    loss = _loss()
    d10 = loss._loss_target_present(s, y, s_other=d, w_interf=w_interf)
    both = loss._loss_target_present(s, y, s_other=d, w_interf=w_interf, s_noise=n, w_artif=1.0)
    assert torch.equal(d10, both)


# -------------------------------------------------------- the decomposition --

def test_three_parts_are_orthogonal_and_sum_to_the_error():
    s, d, n, a = _signals(1)
    y = 0.8 * s + 0.5 * d + 0.3 * n + 0.4 * a
    error, e_i, e_n, e_a = _parts(_loss(), s, y, d, n)
    for u, v in ((e_i, e_n), (e_i, e_a), (e_n, e_a), (e_i, s), (e_n, s), (e_a, s)):
        cos = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        assert cos.abs().max() < 1e-4
    torch.testing.assert_close(e_i + e_n + e_a, error, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(_energy(error), _energy(e_i) + _energy(e_n) + _energy(e_a),
                               rtol=1e-4, atol=1e-2)


def test_artefact_is_orthogonal_to_every_source():
    s, d, n, a = _signals(2)
    y = s + 0.5 * d + 0.5 * n + 0.5 * a
    e_a = _loss()._artefact_part(s, y, d, n)
    for ref in (s, d, n):
        cos = (e_a * ref).sum(-1) / (e_a.norm(dim=-1) * ref.norm(dim=-1))
        assert cos.abs().max() < 1e-4


def test_output_made_only_of_the_sources_has_no_artefact():
    s, d, n, _ = _signals(3)
    y = 0.9 * s + 0.6 * d - 0.4 * n
    e_a = _loss()._artefact_part(s, y, d, n)
    assert (_energy(e_a) / _energy(y)).max() < 1e-8


def test_silent_noise_and_other_stems_give_zero_parts_and_finite_gradients():
    s, _, _, a = _signals(4)
    silent = torch.zeros_like(s)
    y = (s + 0.3 * a).requires_grad_(True)
    loss = _loss(w_artif=4.0)
    assert torch.equal(loss._noise_part(s, y, silent, silent), torch.zeros_like(s))
    # Nothing but the target to explain the error: all of it is artefact.
    error, _, _, e_a = _parts(loss, s, y.detach(), silent, silent)
    torch.testing.assert_close(e_a, error)
    out = loss._loss_target_present(s, y, s_other=silent, s_noise=silent, w_artif=4.0)
    out.sum().backward()
    assert torch.isfinite(y.grad).all()


# ------------------------------------------------------------ the weighting --

def test_artefact_costs_more_than_noise_of_the_same_energy():
    """Same error energy, one output with an artefact, one with residual noise.
    The plain term cannot tell them apart; the weighted one must prefer noise."""
    s, d, n, a = _signals(5, batch=1)
    loss = _loss()
    _, _, n_part, a_part = _parts(loss, s, s + n + a, d, n)   # pure noise / artefact directions
    a_part = a_part * n_part.norm() / a_part.norm()
    noisy, broken = s + 0.3 * n_part, s + 0.3 * a_part
    plain = [float(loss._loss_target_present(s, y)) for y in (noisy, broken)]
    assert plain[0] == pytest.approx(plain[1], abs=1e-3)
    weighted = [float(loss._loss_target_present(s, y, s_other=d, s_noise=n, w_artif=4.0))
                for y in (noisy, broken)]
    assert weighted[1] > weighted[0] + 3.0          # ~6 dB worse at w_artif = 4
    assert weighted[0] == pytest.approx(plain[0], abs=1e-3)


def test_energy_weight_is_ochiais_alpha_squared():
    """Ochiai et al. 2024 eq. 18 puts alpha INSIDE the norm. By orthogonality
    that is w_artif = alpha^2 here -- the claim the configs rest on."""
    s, d, n, a = _signals(6)
    y = s + 0.4 * d + 0.3 * n + 0.5 * a
    loss = _loss()
    alpha_ochiai, tau = 2.0, 0.001
    error, e_i, e_n, e_a = _parts(loss, s, y, d, n)
    s_proj = y - error
    ab_sdr = -10 * torch.log10(_energy(s_proj)
                               / (_energy(e_i + e_n + alpha_ochiai * e_a) + tau * _energy(s_proj)))
    ours = loss._loss_target_present(s, y, tau, s_other=d, s_noise=n, w_artif=alpha_ochiai ** 2)
    torch.testing.assert_close(ours, ab_sdr, rtol=1e-4, atol=1e-3)


def test_both_weights_add_their_own_share():
    s, d, n, a = _signals(7)
    y = s + 0.4 * d + 0.3 * n + 0.5 * a
    loss = _loss()
    error, e_i, _, e_a = _parts(loss, s, y, d, n)
    s_proj = y - error
    expected = -10 * torch.log10(
        _energy(s_proj) / (_energy(error) + 1.0 * _energy(e_i) + 3.0 * _energy(e_a)
                           + 0.001 * _energy(s_proj)))
    got = loss._loss_target_present(s, y, s_other=d, w_interf=2.0, s_noise=n, w_artif=4.0)
    torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-3)


def test_weighted_term_stays_scale_invariant():
    s, d, n, a = _signals(8)
    y = s + 0.4 * d + 0.2 * n + 0.3 * a
    loss = _loss()
    ref = loss._loss_target_present(s, y, s_other=d, w_interf=2.0, s_noise=n, w_artif=4.0)
    for gain in (0.1, 3.0, 50.0):
        torch.testing.assert_close(
            loss._loss_target_present(s, gain * y, s_other=d, w_interf=2.0, s_noise=n, w_artif=4.0),
            ref, rtol=1e-4, atol=1e-3)


# ----------------------------------------------------------------- __call__ --

def test_call_recovers_the_true_noise_from_the_mixture():
    s, d, n, a = _signals(9)
    y = s + 0.3 * d + 0.2 * n + 0.4 * a
    absent = torch.zeros(3, dtype=torch.bool)
    loss = _loss(w_artif=4.0)
    total, parts = loss(s, y, s + d + n, absent, s_other=d)
    direct = loss._loss_target_present(s, y, loss.tau_pres, d, 1.0, n, 4.0).mean()
    assert parts["L_pres_w"] == pytest.approx(float(direct), abs=1e-4)
    share = float(loss._artefact_share(s, y, d, n).mean())
    assert parts["artif_share"] == pytest.approx(share, abs=1e-5)
    assert 0.0 < parts["artif_share"] < 1.0


def test_unit_weights_log_the_share_without_changing_the_total():
    s, d, n, a = _signals(10)
    y = s + 0.3 * d + 0.2 * n + 0.4 * a
    absent = torch.tensor([False, False, True])
    s = s.clone()
    s[2] = 0.0
    loss = _loss()
    plain, _ = loss(s, y, s + d + n, absent)
    with_stem, parts = loss(s, y, s + d + n, absent, s_other=d)
    assert torch.equal(plain, with_stem)
    assert 0.0 < parts["artif_share"] < 1.0


def test_artefact_arm_without_a_stem_refuses_to_train():
    s, d, n, _ = _signals()
    with pytest.raises(ValueError, match="w_artif"):
        _loss(w_artif=4.0)(s, s + n, s + d + n, torch.zeros(3, dtype=torch.bool))


# ---------------------------------------------------------------- train.py --

def _config(w_artif, both_directions=True, remix_gains=False, w_interf=1.0):
    return {"data": {"sample_rate": 16000, "both_directions": both_directions,
                     "remix_gains": remix_gains},
            "loss": {"w": 0.458, "w_m": 9.62, "w_g": 1.69, "tau_pres": 0.001,
                     "tau_abs": 0.01, "p": 0.3, "windows_ms": [8, 16, 32, 64],
                     "w_interf": w_interf, "w_artif": w_artif}}


def test_build_loss_fn_passes_both_weights_and_defaults_to_one():
    built = build_loss_fn(_config(4.0, w_interf=2.0))
    assert (built.w_interf, built.w_artif) == (2.0, 4.0)
    config = _config(1.0)
    del config["loss"]["w_artif"]
    assert build_loss_fn(config).w_artif == 1.0


def test_build_loss_fn_refuses_an_artefact_weight_with_no_stem_on_disk():
    with pytest.raises(AssertionError, match="w_artif"):
        build_loss_fn(_config(4.0, both_directions=False, remix_gains=False))
    build_loss_fn(_config(1.0, both_directions=False))
    build_loss_fn(_config(4.0, both_directions=False, remix_gains=True))


def test_build_loss_fn_refuses_non_positive_artefact_weights():
    with pytest.raises(AssertionError):
        build_loss_fn(_config(0.0))
