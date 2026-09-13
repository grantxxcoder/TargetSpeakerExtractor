"""Estimator.residual_scale -- the inference-time ablation of the residual branch.

WHY THIS EXISTS. The output is S = M (x) X + R. The mask is MULTIPLIED, so it
can only scale energy the mixture already has; R is ADDED from an unbounded
Conv1d and can place energy in bins the recording never contained. That is the
physical operation behind an invented word, and invented words are 41.5 % of our
wrong content words. residual_scale makes R removable at inference so the
hypothesis costs CPU hours instead of a training run.

The first test is the important one: the default must be arithmetically the old
path, or every published number silently changes.
"""
import torch

from src.models.modules import Estimator


def _estimator(seed=0, residual_branch=True):
    torch.manual_seed(seed)
    return Estimator(band_widths=[4, 4, 8], feature_dim=6, mlp_hidden=12,
                     n_hidden=1, causal=True,
                     residual_branch=residual_branch).eval()


def _inputs(bands=(4, 4, 8), B=2, T=7, N=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    feats = torch.randn(B, len(bands), N, T, generator=g)
    mix = [torch.complex(torch.randn(B, bw, T, generator=g),
                         torch.randn(B, bw, T, generator=g)) for bw in bands]
    return feats, mix


def test_default_is_bit_identical_to_the_untouched_model():
    """residual_scale defaults to 1.0 and must change nothing at all."""
    est = _estimator()
    feats, mix = _inputs()
    with torch.no_grad():
        before = est(feats, mix)
    assert est.residual_scale == 1.0
    with torch.no_grad():
        after = est(feats, mix)
    assert torch.equal(before, after)


def test_scale_zero_equals_a_model_built_without_the_branch():
    """R deleted at inference == the masked mixture alone, exactly."""
    est = _estimator()
    feats, mix = _inputs()
    est.residual_scale = 0.0
    with torch.no_grad():
        ablated = est(feats, mix)

    est.residual_scale = 1.0
    est.capture_parts = True
    with torch.no_grad():
        est(feats, mix)
    masked_only = est.last_parts["masked"]
    assert torch.allclose(ablated, masked_only, atol=1e-6)


def test_scale_is_linear_in_the_residual():
    """Half the scale removes half of R, so output moves by exactly R/2."""
    est = _estimator()
    feats, mix = _inputs()
    est.capture_parts = True
    with torch.no_grad():
        full = est(feats, mix)
    residual = est.last_parts["residual"]
    est.residual_scale = 0.5
    with torch.no_grad():
        half = est(feats, mix)
    assert torch.allclose(full - half, residual * 0.5, atol=1e-6)


def test_capture_parts_reconstructs_the_output():
    """masked + residual == the returned tensor, so the split is the real one."""
    est = _estimator()
    feats, mix = _inputs()
    est.capture_parts = True
    with torch.no_grad():
        out = est(feats, mix)
    parts = est.last_parts
    assert torch.allclose(parts["masked"] + parts["residual"], out, atol=1e-6)
    assert parts["mixture"].shape == out.shape


def test_residual_can_place_energy_where_the_mixture_is_silent():
    """The claim the ablation rests on, asserted rather than argued.

    In a bin where the mixture is exactly zero the mask contributes exactly
    zero, so any output there came from R.
    """
    est = _estimator()
    feats, mix = _inputs()
    silent = [torch.zeros_like(m) for m in mix]
    est.capture_parts = True
    with torch.no_grad():
        out = est(feats, silent)
    assert torch.allclose(est.last_parts["masked"],
                          torch.zeros_like(out), atol=1e-12)
    assert out.abs().max() > 0, "R contributed nothing to a silent mixture"


def test_no_branch_means_no_residual_captured():
    est = _estimator(residual_branch=False)
    feats, mix = _inputs()
    est.capture_parts = True
    with torch.no_grad():
        out = est(feats, mix)
    assert est.last_parts["residual"] is None
    assert torch.allclose(est.last_parts["masked"], out, atol=1e-6)
