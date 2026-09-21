"""`lookahead_shift` must actually run, and shift by exactly k frames.
decisions-m1.md 2026-08-18 (the lookahead knob), fixed 2026-09-21.

WHY THIS TEST EXISTS
--------------------
The function passed a 2-element pad spec to `F.pad(..., mode="replicate")` on a
4-D tensor, which raises NotImplementedError. So every non-zero
`lookahead_frames` crashed, and the latency ablation the config has advertised
since 2026-08-18 could never have run. Nothing caught it because every training
run to date used `lookahead_frames: 0`, the one value that short-circuits before
the bug.

These tests pin both halves: that k > 0 runs at all, and that it shifts by the
right amount in the right direction -- a shift of the wrong sign would leak the
FUTURE into a model sold as causal, which is worse than a crash because it
would not announce itself.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.modules import lookahead_shift  # noqa: E402

B, K, N, T = 2, 3, 4, 10


def ramp():
    """h[..., t] == t, so a shift is readable straight off the values."""
    return torch.arange(T, dtype=torch.float32).expand(B, K, N, T).clone()


def test_k_zero_is_identity():
    h = ramp()
    assert torch.equal(lookahead_shift(h, 0), h)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_nonzero_k_runs_and_keeps_shape(k):
    """The regression: this raised NotImplementedError for every k > 0."""
    h = ramp()
    out = lookahead_shift(h, k)
    assert out.shape == h.shape


@pytest.mark.parametrize("k", [1, 2, 5])
def test_shifts_forward_by_exactly_k(k):
    """h'[..., t] == h[..., t + k] for every t that still has a source frame."""
    out = lookahead_shift(ramp(), k)
    for t in range(T - k):
        assert torch.allclose(out[..., t], torch.full((B, K, N), float(t + k))), t


@pytest.mark.parametrize("k", [1, 2, 5])
def test_tail_is_edge_replicated(k):
    """The last k frames have no future to read, so they repeat the final frame."""
    out = lookahead_shift(ramp(), k)
    for t in range(T - k, T):
        assert torch.allclose(out[..., t], torch.full((B, K, N), float(T - 1))), t


def test_direction_is_future_not_past():
    """A sign error here would be silent and would break causality claims."""
    out = lookahead_shift(ramp(), 1)
    assert out[0, 0, 0, 0].item() == 1.0, "must read frame t+1, not t-1"
