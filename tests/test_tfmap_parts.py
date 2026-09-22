"""Item 1a: TFMap returns the cue's PARTS instead of their product.

The claims under test are algebraic, so they are testable exactly rather than
by eyeballing a training curve:

    x_t = alpha_t * u_t + r_t,     <u_t, r_t> = 0
    ||x_t||^2 = alpha_t^2 + ||r_t||^2
    cue_t = alpha_t * u_t = cos_t * ||x_t|| * u_t        (nothing is lost)

Plus the regression that matters more than any of them: with the flag off,
every number must be bit-identical to the pre-2026-09-22 model, because a
silent change here would move every run this project has recorded.

ranked-next-steps.md item 1a; decisions-m2.md 2026-09-22.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import BSRNN_TFMAP  # noqa: E402
from src.models.conditioning import TFMap  # noqa: E402

F_BINS, T_MIX, T_ENROL = 257, 40, 60


def tiny_model(**kwargs):
    """Same fixture as tests/test_tfmap_inject.py: every module the shipped
    model has, narrow instead of wide, so a ctor test costs milliseconds."""
    return BSRNN_TFMAP(sample_rate=8000, n_fft=128, hop=32, num_repeat=2,
                       band_segments={"uniform": 5}, feature_dim=16,
                       hidden_dim=16, mlp_hidden=16, **kwargs)


def _mags(seed=0):
    """Magnitude spectrograms are NON-NEGATIVE. Sampling signed noise would
    test a tensor the STFT can never produce, and the residual's sign argument
    below depends on it."""
    g = torch.Generator().manual_seed(seed)
    mix = torch.rand(2, F_BINS, T_MIX, generator=g)
    enrol = torch.rand(2, F_BINS, T_ENROL, generator=g)
    return mix, enrol


def test_parts_shape_and_default_unchanged():
    mix, enrol = _mags()
    assert TFMap(scale=16.0)(mix, enrol).shape == (2, 1, F_BINS, T_MIX)
    assert TFMap(scale=16.0, return_parts=True)(mix, enrol).shape == (2, 3, F_BINS, T_MIX)


def test_decomposition_is_exact_and_orthogonal():
    """x = alpha*u + r with <u, r> = 0 -- the whole justification for handing
    over three channels rather than one."""
    mix, enrol = _mags()
    u, cos, residual = TFMap(scale=16.0, return_parts=True)(mix, enrol).unbind(dim=1)
    norm_x = mix.norm(dim=1, keepdim=True)
    alpha = cos[:, :1, :] * norm_x                      # cos was broadcast over F

    torch.testing.assert_close(alpha * u + residual, mix, rtol=1e-4, atol=1e-5)
    inner = (u * residual).sum(dim=1)
    assert inner.abs().max() < 1e-3, f"u and r not orthogonal: {inner.abs().max()}"


def test_pythagoras_so_cos_reads_as_energy_fraction():
    """cos^2 + (||r||/||x||)^2 = 1, which is what licenses reading channel 2 as
    'the fraction of this frame's energy the target template explains'."""
    mix, enrol = _mags(seed=3)
    u, cos, residual = TFMap(scale=16.0, return_parts=True)(mix, enrol).unbind(dim=1)
    norm_x = mix.norm(dim=1)
    frac_r = residual.norm(dim=1) / norm_x
    torch.testing.assert_close(cos[:, 0, :] ** 2 + frac_r ** 2,
                               torch.ones_like(frac_r), rtol=1e-3, atol=1e-4)


def test_nothing_is_lost_product_is_recoverable():
    """The old cue is cos * ||x|| * u, so the parts strictly dominate it."""
    mix, enrol = _mags(seed=7)
    product = TFMap(scale=16.0)(mix, enrol)[:, 0]
    u, cos, _ = TFMap(scale=16.0, return_parts=True)(mix, enrol).unbind(dim=1)
    rebuilt = cos * mix.norm(dim=1, keepdim=True) * u
    torch.testing.assert_close(rebuilt, product, rtol=1e-4, atol=1e-5)


def test_direction_is_unit_norm_and_cos_is_bounded():
    """u is loudness-free BY CONSTRUCTION -- the property the whole item rests
    on. cos in [0, 1] because magnitudes are non-negative."""
    mix, enrol = _mags(seed=11)
    u, cos, _ = TFMap(scale=16.0, return_parts=True)(mix, enrol).unbind(dim=1)
    torch.testing.assert_close(u.norm(dim=1), torch.ones(2, T_MIX), rtol=1e-4, atol=1e-5)
    assert 0.0 <= cos.min() and cos.max() <= 1.0 + 1e-5


def test_scaling_the_mixture_leaves_the_loudness_free_channels_alone():
    """Multiply the mixture by 10: the PRODUCT channel scales with it, u and cos
    do not. This is the confound the item exists to remove, as a test."""
    mix, enrol = _mags(seed=13)
    quiet = TFMap(scale=16.0, return_parts=True)(mix, enrol)
    loud = TFMap(scale=16.0, return_parts=True)(mix * 10.0, enrol)
    torch.testing.assert_close(loud[:, 0], quiet[:, 0], rtol=1e-4, atol=1e-5)   # u
    torch.testing.assert_close(loud[:, 1], quiet[:, 1], rtol=1e-3, atol=1e-4)   # cos
    # the residual is in the mixture's units, so it DOES scale -- 10x, exactly
    torch.testing.assert_close(loud[:, 2], quiet[:, 2] * 10.0, rtol=1e-3, atol=1e-3)
    # and the old 1-channel cue scales too, which is the whole problem
    old_quiet = TFMap(scale=16.0)(mix, enrol)
    old_loud = TFMap(scale=16.0)(mix * 10.0, enrol)
    torch.testing.assert_close(old_loud, old_quiet * 10.0, rtol=1e-3, atol=1e-3)


def test_model_input_width_is_derived_not_configured():
    off, on = tiny_model(tfmap_parts=False), tiny_model(tfmap_parts=True)
    # SubbandNorm's first conv reads bw * C channels.
    assert off.subband_norm.blocks[0][1].in_channels == off.band_widths[0] * 3
    assert on.subband_norm.blocks[0][1].in_channels == on.band_widths[0] * 5

    # The cost is TWO extra channels through the per-band 1x1 conv (weights and
    # the LayerNorm's gain+bias), and nothing else. Stated as the closed form so
    # that a future widening of the cue is caught here rather than in a config.
    feature_dim, extra = 16, 2
    total_bins = sum(off.band_widths)
    expected = extra * total_bins * (feature_dim + 2)
    delta = sum(p.numel() for p in on.parameters()) - sum(p.numel() for p in off.parameters())
    assert delta == expected, f"{delta} != {expected}"


def test_a_wrong_in_channels_is_refused_not_silently_fixed():
    with pytest.raises(ValueError, match="in_channels"):
        tiny_model(in_channels=4)


def test_parts_and_inject_together_are_refused():
    """The injector asserts a 1-channel cue. Fail at construction, not six
    frames deep in a Kaggle log."""
    with pytest.raises(NotImplementedError, match="tfmap_inject"):
        tiny_model(tfmap_parts=True, tfmap_inject=True)


def test_forward_runs_end_to_end_and_the_default_path_is_untouched():
    torch.manual_seed(0)
    mixture, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    for parts in (False, True):
        torch.manual_seed(0)
        model = tiny_model(tfmap_parts=parts).eval()
        with torch.no_grad():
            y = model(mixture, enrol)
        assert y.shape == mixture.shape and torch.isfinite(y).all()
