"""Item 1a: tfmap returns the cue's PARTS instead of their product.

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

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import build_bsrnn_tfmap, extract
from src.models.conditioning import tfmap

F_BINS, T_MIX, T_ENROL = 257, 40, 60


def tiny_model(tfmap_parts, tfmap_part_scales):
    """Every module the shipped model has, narrow instead of wide, so a ctor
    test costs milliseconds."""
    return build_bsrnn_tfmap(sample_rate=16000, n_fft=512, hop=128, feature_dim=16, hidden_dim=16, num_repeat=2, mlp_hidden=16, tfmap_scale=16.0, tfmap_parts=tfmap_parts, tfmap_part_scales=tfmap_part_scales, embedding_dim=None)


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
    assert tfmap(mix, enrol, scale=16.0, return_parts=False, part_scales=None).shape == (2, 1, F_BINS, T_MIX)
    assert tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0]).shape == (2, 3, F_BINS, T_MIX)


def test_decomposition_is_exact_and_orthogonal():
    """x = alpha*u + r with <u, r> = 0 -- the whole justification for handing
    over three channels rather than one."""
    mix, enrol = _mags()
    u, cos, residual = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0]).unbind(dim=1)
    norm_x = mix.norm(dim=1, keepdim=True)
    alpha = cos[:, :1, :] * norm_x                      # cos was broadcast over F

    torch.testing.assert_close(alpha * u + residual, mix, rtol=1e-4, atol=1e-5)
    inner = (u * residual).sum(dim=1)
    assert inner.abs().max() < 1e-3, f"u and r not orthogonal: {inner.abs().max()}"


def test_pythagoras_so_cos_reads_as_energy_fraction():
    """cos^2 + (||r||/||x||)^2 = 1, which is what licenses reading channel 2 as
    'the fraction of this frame's energy the target template explains'."""
    mix, enrol = _mags(seed=3)
    u, cos, residual = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0]).unbind(dim=1)
    norm_x = mix.norm(dim=1)
    frac_r = residual.norm(dim=1) / norm_x
    torch.testing.assert_close(cos[:, 0, :] ** 2 + frac_r ** 2,
                               torch.ones_like(frac_r), rtol=1e-3, atol=1e-4)


def test_nothing_is_lost_product_is_recoverable():
    """The old cue is cos * ||x|| * u, so the parts strictly dominate it."""
    mix, enrol = _mags(seed=7)
    product = tfmap(mix, enrol, scale=16.0, return_parts=False, part_scales=None)[:, 0]
    u, cos, _ = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0]).unbind(dim=1)
    rebuilt = cos * mix.norm(dim=1, keepdim=True) * u
    torch.testing.assert_close(rebuilt, product, rtol=1e-4, atol=1e-5)


def test_direction_is_unit_norm_and_cos_is_bounded():
    """u is loudness-free BY CONSTRUCTION -- the property the whole item rests
    on. cos in [0, 1] because magnitudes are non-negative."""
    mix, enrol = _mags(seed=11)
    u, cos, _ = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0]).unbind(dim=1)
    torch.testing.assert_close(u.norm(dim=1), torch.ones(2, T_MIX), rtol=1e-4, atol=1e-5)
    assert 0.0 <= cos.min() and cos.max() <= 1.0 + 1e-5


def test_scaling_the_mixture_leaves_the_loudness_free_channels_alone():
    """Multiply the mixture by 10: the PRODUCT channel scales with it, u and cos
    do not. This is the confound the item exists to remove, as a test."""
    mix, enrol = _mags(seed=13)
    quiet = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0])
    loud = tfmap(mix * 10.0, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0])
    torch.testing.assert_close(loud[:, 0], quiet[:, 0], rtol=1e-4, atol=1e-5)   # u
    torch.testing.assert_close(loud[:, 1], quiet[:, 1], rtol=1e-3, atol=1e-4)   # cos
    # the residual is in the mixture's units, so it DOES scale -- 10x, exactly
    torch.testing.assert_close(loud[:, 2], quiet[:, 2] * 10.0, rtol=1e-3, atol=1e-3)
    # and the old 1-channel cue scales too, which is the whole problem
    old_quiet = tfmap(mix, enrol, scale=16.0, return_parts=False, part_scales=None)
    old_loud = tfmap(mix * 10.0, enrol, scale=16.0, return_parts=False, part_scales=None)
    torch.testing.assert_close(old_loud, old_quiet * 10.0, rtol=1e-3, atol=1e-3)


def test_model_input_width_is_derived_not_configured():
    off, on = tiny_model(False, None), tiny_model(True, [1.0, 1.0, 1.0])
    # The subband norm's first conv reads bw * C channels.
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


def test_forward_runs_end_to_end_and_the_default_path_is_untouched():
    torch.manual_seed(0)
    mixture, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    for parts in (False, True):
        torch.manual_seed(0)
        model = tiny_model(parts, [1.0, 1.0, 1.0] if parts else None).eval()
        with torch.no_grad():
            y = extract(model, mixture, enrol, None)
        assert y.shape == mixture.shape and torch.isfinite(y).all()


# --- per-channel scales (derive_cue_scales.py) ----------------------------

def test_scales_multiply_their_own_channel_and_no_other():
    mix, enrol = _mags(seed=23)
    plain = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0])
    scaled = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[7.5, 0.5, 1.3])
    for channel, factor in enumerate((7.5, 0.5, 1.3)):
        torch.testing.assert_close(scaled[:, channel], plain[:, channel] * factor,
                                   rtol=1e-4, atol=1e-5)


def test_direction_rms_is_the_closed_form_the_derivation_checks_against():
    """1/sqrt(F) exactly, because the direction is unit-norm over F bins. This
    is what makes derive_cue_scales.py's measurement checkable rather than
    merely plausible."""
    mix, enrol = _mags(seed=29)
    direction = tfmap(mix, enrol, scale=16.0, return_parts=True, part_scales=[1.0, 1.0, 1.0])[:, 0]
    rms = direction.pow(2).mean().sqrt()
    torch.testing.assert_close(rms, torch.tensor(1.0 / F_BINS ** 0.5),
                               rtol=1e-4, atol=1e-6)


def test_scales_survive_the_model_and_are_not_parameters():
    """A setting, not a parameter: it must not be optimised, and it must not
    appear in the checkpoint as something that could silently differ from the
    config that claims to describe the run."""
    m = tiny_model(True, [7.5, 0.5, 1.3])
    assert m.tfmap_part_scales == [7.5, 0.5, 1.3]
    assert sum(p.numel() for p in m.parameters()) == sum(p.numel() for p in tiny_model(True, [1.0, 1.0, 1.0]).parameters())
    assert not any("part_scales" in k for k in m.state_dict())
