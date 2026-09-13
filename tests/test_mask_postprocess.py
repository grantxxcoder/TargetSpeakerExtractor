"""Inference-time mask post-processing: the floor and the frequency hysteresis.

Both exist to test hypotheses on an ALREADY-TRAINED checkpoint rather than by
spending a training session, so the property that matters most is that they
change nothing when switched off. A silent drift here would rewrite every
published number without touching a weight.

The Estimator was restructured on 2026-09-12 to gather all bands before applying
the mask, because frequency-adjacency cannot be computed inside a per-band loop.
`_legacy_forward` is kept solely so that equivalence can be ASSERTED here rather
than argued in a commit message.
"""

import numpy as np
import pytest
import torch

from src.models.bands import band_plan
from src.models.modules import Estimator, apply_hysteresis


def make(feature_dim=16, n_frames=9, batch=2, seed=0):
    torch.manual_seed(seed)
    widths = band_plan(16000, 512, "wesep_16k")
    estimator = Estimator(widths, feature_dim, 16, 1)
    feats = torch.randn(batch, len(widths), feature_dim, n_frames)
    mix = [torch.complex(torch.randn(batch, w, n_frames),
                         torch.randn(batch, w, n_frames)) for w in widths]
    return estimator, feats, mix


# --- the refactor must be exactly the old arithmetic ----------------------

def test_gathered_forward_is_bit_identical_to_the_band_loop():
    estimator, feats, mix = make()
    assert torch.equal(estimator(feats, mix), estimator._legacy_forward(feats, mix))


def test_still_identical_with_a_floor():
    estimator, feats, mix = make()
    estimator.mask_floor = 0.1
    assert torch.equal(estimator(feats, mix), estimator._legacy_forward(feats, mix))


def test_both_post_processors_are_off_by_default():
    estimator, _, _ = make()
    assert estimator.mask_floor == 0.0
    assert estimator.mask_hysteresis is None


# --- the floor ------------------------------------------------------------

def test_a_floor_changes_the_output_and_zero_does_not():
    estimator, feats, mix = make()
    plain = estimator(feats, mix)
    estimator.mask_floor = 0.0
    assert torch.equal(estimator(feats, mix), plain)
    estimator.mask_floor = 0.3
    assert not torch.equal(estimator(feats, mix), plain)


def test_the_floor_tops_up_towards_pass_through_not_along_the_mask():
    """In a bin the model wants to kill, the complex mask sits near the origin
    and its phase is noise. Scaling that up would inject an arbitrary phase
    rotation, so the top-up is added to the REAL part only."""
    mr = torch.zeros(1, 4, 3)
    mi = torch.zeros(1, 4, 3)
    floor = 0.25
    magnitude = (mr.pow(2) + mi.pow(2) + 1e-12).sqrt()
    topped = mr + (floor - magnitude).clamp_min(0.0)
    assert torch.allclose(topped, torch.full_like(topped, floor), atol=1e-6)
    assert torch.equal(mi, torch.zeros_like(mi))


# --- the hysteresis -------------------------------------------------------

def test_down_of_one_is_a_no_op():
    """The control arm. Nothing survives-or-dies when the penalty is 1.0, so the
    mask must come back untouched -- otherwise the level restoration is
    introducing a change of its own."""
    torch.manual_seed(0)
    mr, mi = torch.rand(2, 32, 7) + 0.1, torch.rand(2, 32, 7) * 0.1
    out_r, out_i = apply_hysteresis(mr, mi, 1.5, 0.5, 1.0)
    assert torch.allclose(out_r, mr, atol=1e-5)
    assert torch.allclose(out_i, mi, atol=1e-5)


def test_the_frame_ENERGY_is_preserved_not_its_mean():
    """The invariant has to be RMS, and this is the test that was wrong first.

    It asserted the MEAN magnitude until 2026-09-12. Concentrating the same mean
    into fewer bins multiplies the RMS, and the audio level follows the RMS: on
    the real mask that produced an output +7.38 dB above the mixture, 12 dB
    louder than the baseline, and the evaluation of it measured distortion rather
    than the idea (65 % of clips returned no transcript). A test asserting the
    wrong invariant is how a bug reaches a results table."""
    torch.manual_seed(0)
    mr, mi = torch.rand(2, 32, 7) + 0.05, torch.rand(2, 32, 7) * 0.1
    rms = lambda a: a.pow(2).mean(dim=1).sqrt()          # noqa: E731
    before = rms((mr.pow(2) + mi.pow(2)).sqrt())
    out_r, out_i = apply_hysteresis(mr, mi, 1.4, 0.6, 0.0)
    after = rms((out_r.pow(2) + out_i.pow(2)).sqrt())
    assert torch.allclose(before, after, rtol=1e-4)


def test_a_frame_where_nothing_survives_is_left_alone():
    """Otherwise the level restoration divides by ~0 and the frame explodes.
    A mask flat enough to have no bin above the seed threshold is exactly the
    case our real mask is closest to."""
    flat = torch.full((1, 16, 3), 0.5)
    out_r, _ = apply_hysteresis(flat, torch.zeros_like(flat), 5.0, 4.0, 0.0)
    assert torch.allclose(out_r, flat, atol=1e-6)
    assert torch.isfinite(out_r).all()


def test_it_adds_variation_across_frequency():
    """The entire point. MEASURED on the real checkpoint: 1.5/0.5/0.0 raises the
    mask's frequency variation from 0.0238 to 0.0596 against the ideal mask's
    0.1725."""
    torch.manual_seed(0)
    mr = torch.rand(1, 64, 12) + 0.5      # smooth-ish, like the real mask
    mi = torch.zeros_like(mr)
    rough = lambda a: (a[:, 1:] - a[:, :-1]).abs().mean()   # noqa: E731
    out_r, _ = apply_hysteresis(mr, mi, 1.3, 0.7, 0.0)
    assert rough((out_r.pow(2)).sqrt()) > rough(mr)


def test_an_isolated_weak_bin_dies_but_a_connected_one_survives():
    """The hysteresis rule itself, and the only thing distinguishing it from a
    plain threshold. Both test bins sit between the two thresholds; one touches
    a seed and one does not."""
    mr = torch.full((1, 7, 1), 0.5)
    mr[0, 0, 0] = 3.0          # seed
    mr[0, 1, 0] = 1.1          # weak, ADJACENT to the seed -> survives
    mr[0, 4, 0] = 0.05         # below lo -> breaks the run
    mr[0, 5, 0] = 1.1          # weak, isolated from any seed -> dies
    mi = torch.zeros_like(mr)
    out_r, _ = apply_hysteresis(mr, mi, 2.0, 0.6, 0.0)
    assert out_r[0, 1, 0] > 0, "a weak bin touching a seed must survive"
    assert out_r[0, 5, 0] == 0, "a weak bin with no path to a seed must die"


def test_frames_are_independent_so_it_is_causal_in_time():
    """Growth runs across frequency only. Changing a LATER frame must not change
    an earlier one, or the post-processor has quietly spent latency."""
    torch.manual_seed(0)
    mr = torch.rand(1, 16, 5) + 0.2
    mi = torch.zeros_like(mr)
    a, _ = apply_hysteresis(mr, mi, 1.3, 0.7, 0.0)
    edited = mr.clone()
    edited[0, :, 4] = 5.0
    b, _ = apply_hysteresis(edited, mi, 1.3, 0.7, 0.0)
    assert torch.allclose(a[0, :, :4], b[0, :, :4], atol=1e-6)


def test_the_estimator_applies_it():
    estimator, feats, mix = make()
    plain = estimator(feats, mix)
    estimator.mask_hysteresis = (1.5, 0.5, 0.0)
    assert not torch.equal(estimator(feats, mix), plain)
    estimator.mask_hysteresis = None
    assert torch.equal(estimator(feats, mix), plain)
