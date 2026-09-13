"""D17, the mask-structure term. decisions-pending.md 2026-09-13.

The first test is the one that protects every published number: at w_struct 0
and with no mask supplied, the objective must be arithmetically what it was.

The term supervises the mask's FREQUENCY SHAPE -- the per-frame mean is removed
from both sides -- because the model's per-frame gain is already roughly right
and three existing terms argue about level. What is measured is which bins get
more than their frame's average and which get less.
"""
import math

import torch

from src.models.losses import LossBSRNN


def _loss(**kw):
    return LossBSRNN(wm=9.62, w=0.458, wg=1.69, sample_rate=16000, **kw)


def _batch(B=2, T=4000, seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, T, generator=g) * 0.1
    output = target + torch.randn(B, T, generator=g) * 0.01
    mixture = target + torch.randn(B, T, generator=g) * 0.05
    absent = torch.zeros(B, dtype=torch.bool)
    return target, output, mixture, absent


def _grids(B=2, F=16, T=20, seed=1):
    g = torch.Generator().manual_seed(seed)
    mask = torch.rand(B, F, T, generator=g)
    oracle = torch.rand(B, F, T, generator=g)
    mag = torch.rand(B, F, T, generator=g) + 0.5     # no near-silent bins
    return mask, oracle, mag


def test_absent_term_reproduces_the_old_objective_exactly():
    """No mask supplied -> the pre-2026-09-13 objective, bit for bit."""
    t, o, x, a = _batch()
    base = _loss()
    total_before, parts_before = base(t, o, x, a)
    total_after, parts_after = base(t, o, x, a, mask=None, oracle_mask=None)
    assert torch.equal(total_before, total_after)
    assert math.isnan(parts_after["L_struct"])
    for k in ("L_pres", "L_MR", "L_gain"):
        assert parts_before[k] == parts_after[k]


def test_logged_at_zero_weight_but_contributes_no_gradient():
    """The arrangement derive_w_struct.py depends on: computed, logged, inert."""
    t, o, x, a = _batch()
    mask, oracle, mag = _grids()
    off = _loss(w_struct=0.0)
    total_off, parts_off = off(t, o, x, a, mask=mask, oracle_mask=oracle,
                               mixture_mag=mag)
    plain_total, _ = off(t, o, x, a)
    assert not math.isnan(parts_off["L_struct"]), "term must be LOGGED at w 0"
    assert torch.allclose(total_off, plain_total, atol=1e-6), \
        "term must contribute NO gradient at w 0"


def test_weight_moves_the_total():
    t, o, x, a = _batch()
    mask, oracle, mag = _grids()
    lo, _ = _loss(w_struct=0.0)(t, o, x, a, mask=mask, oracle_mask=oracle,
                                mixture_mag=mag)
    hi, parts = _loss(w_struct=1.0)(t, o, x, a, mask=mask, oracle_mask=oracle,
                                     mixture_mag=mag)
    assert hi > lo
    # (1 - w) * w_struct * L_struct is the documented contribution.
    assert torch.allclose(hi - lo, torch.tensor((1 - 0.458) * parts["L_struct"]),
                          atol=1e-5)


def test_a_perfect_mask_costs_nothing():
    t, o, x, a = _batch()
    mask, _, mag = _grids()
    loss = _loss(w_struct=1.0)
    value = loss._loss_mask_shape(mask, mask.clone(), mag)
    assert torch.allclose(value, torch.zeros_like(value), atol=1e-6)


def test_a_constant_offset_costs_nothing():
    """THE DESIGN CLAIM. Adding a constant to every bin of a frame changes the
    frame's LEVEL and not its SHAPE, and the term must be blind to it -- that is
    the whole reason the per-frame mean is removed."""
    mask, oracle, mag = _grids()
    loss = _loss(w_struct=1.0)
    offset = torch.rand(mask.shape[0], 1, mask.shape[2]) * 3.0
    a = loss._loss_mask_shape(mask, oracle, mag)
    b = loss._loss_mask_shape(mask + offset, oracle, mag)
    assert torch.allclose(a, b, atol=1e-5)


def test_a_flat_mask_is_penalised_against_a_structured_oracle():
    """The failure mode the term exists to catch: a volume knob."""
    B, F, T = 2, 16, 20
    mag = torch.ones(B, F, T)
    knob = torch.ones(B, F, T) * 0.5                       # same gain everywhere
    g = torch.Generator().manual_seed(3)
    structured = torch.rand(B, F, T, generator=g)
    loss = _loss(w_struct=1.0)
    flat_cost = loss._loss_mask_shape(knob, structured, mag)
    good_cost = loss._loss_mask_shape(structured, structured, mag)
    assert (flat_cost > good_cost).all()
    assert (flat_cost > 0).all()


def test_silent_bins_are_excluded():
    """Below struct_floor_db the ideal mask is noise over noise. Cells there must
    not be able to drive the term."""
    B, F, T = 1, 8, 5
    mag = torch.ones(B, F, T)
    mag[:, :4] = 1e-6                       # 4 bins far below the peak
    mask = torch.zeros(B, F, T)
    oracle = torch.zeros(B, F, T)
    loss = _loss(w_struct=1.0, struct_floor_db=-40.0)
    quiet_only = oracle.clone()
    quiet_only[:, :4] = 5.0                 # huge disagreement, silent bins only
    assert torch.allclose(loss._loss_mask_shape(mask, quiet_only, mag),
                          torch.zeros(B), atol=1e-6)
    # The loud bins must VARY, or both shapes are flat once the frame mean is
    # removed and zero cost is the correct answer -- which is what the first
    # draft of this test got wrong.
    loud_only = oracle.clone()
    loud_only[:, 4:] = torch.tensor([0.0, 1.0, 2.0, 3.0]).view(1, 4, 1)
    assert (loss._loss_mask_shape(mask, loud_only, mag) > 0).all()


def test_gradient_reaches_the_mask():
    """A detached mask would train nothing while the curve looked healthy."""
    mask, oracle, mag = _grids()
    mask.requires_grad_(True)
    loss = _loss(w_struct=1.0)
    loss._loss_mask_shape(mask, oracle, mag).sum().backward()
    assert mask.grad is not None and mask.grad.abs().sum() > 0
