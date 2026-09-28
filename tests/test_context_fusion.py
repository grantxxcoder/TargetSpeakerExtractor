"""Item 1c: a frozen speaker embedding modulates the separator's features.

NO ECAPA AND NO SPEECHBRAIN ANYWHERE IN THIS FILE. The encoder lives outside
the model precisely so the fusion can be tested with a synthetic 192-vector in
milliseconds, against a 9-minute suite. If a test here ever needs the snapshot,
the separation of concerns has broken.

decisions-m2.md 2026-09-22; ranked-next-steps.md item 1c.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import BSRNN_TFMAP, BSRNN_TFMAP_CONTEXT  # noqa: E402
from src.models.conditioning import ContextFusion  # noqa: E402

D, N = 192, 16


def tiny(cls=BSRNN_TFMAP_CONTEXT, **kw):
    return cls(sample_rate=8000, n_fft=128, hop=32, num_repeat=2,
               band_segments={"uniform": 5}, feature_dim=N,
               hidden_dim=N, mlp_hidden=N, tfmap_parts=True, **kw)


def unit(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(batch, D, generator=g), dim=-1)


# --- the fusion in isolation ---------------------------------------------

def test_zero_init_is_exactly_the_identity():
    """At initialisation the arm must be BIT-IDENTICAL to its 1a control, so a
    difference can never be credited to a different starting point."""
    f = ContextFusion(D, N)
    z = torch.randn(2, 4, N, 7)
    torch.testing.assert_close(f(z, unit()), z, rtol=0, atol=0)


def test_it_still_learns_from_a_zero_start():
    """Zero init must not be a dead start: dL/dW = (dL/dz') * z * e^T is
    non-zero wherever z is."""
    f = ContextFusion(D, N)
    z = torch.randn(2, 4, N, 7)
    f(z, unit()).pow(2).sum().backward()
    assert f.project.weight.grad.abs().max() > 0


def test_gamma_varies_with_the_speaker_and_nothing_else():
    """gamma depends on WHO, not on when, where, or how loud -- the property
    that makes it un-confoundable by the per-frame loudness the cue rides."""
    f = ContextFusion(D, N)
    torch.nn.init.normal_(f.project.weight, std=0.1)
    a, b = unit(1, seed=1), unit(1, seed=2)
    assert not torch.allclose(f.gamma(a), f.gamma(b))
    torch.testing.assert_close(f.gamma(a), f.gamma(a))          # deterministic
    assert f.gamma(a).shape == (1, N)                           # one per FEATURE


def test_the_same_gain_is_applied_to_every_band_and_frame():
    f = ContextFusion(D, N)
    torch.nn.init.normal_(f.project.weight, std=0.1)
    e = unit(1)
    z = torch.ones(1, 4, N, 7)
    out = f(z, e)
    expected = (1.0 + f.gamma(e))[0]                            # (N,)
    for k in range(4):
        for t in range(7):
            torch.testing.assert_close(out[0, k, :, t], expected)


def test_parameter_count_is_the_projection_and_nothing_else():
    f = ContextFusion(D, 128)
    assert f.n_parameters == 128 * D + 128 == 24_704


def test_batch_mismatch_is_refused():
    f = ContextFusion(D, N)
    with pytest.raises(ValueError, match="batch mismatch"):
        f(torch.randn(3, 4, N, 7), unit(2))


# --- the model ------------------------------------------------------------

def test_the_embedding_is_required_not_optional():
    """THE POINT OF THE SUBCLASS. A call site that forgets the embedding must
    crash, never run unconditioned and still score."""
    m = tiny().eval()
    with pytest.raises(TypeError):
        m(torch.randn(2, 8000), torch.randn(2, 8000))


def test_the_base_class_is_untouched():
    """Every pre-2026-09-22 call signature still works and ignores context."""
    m = BSRNN_TFMAP(sample_rate=8000, n_fft=128, hop=32, num_repeat=2,
                    band_segments={"uniform": 5}, feature_dim=N, hidden_dim=N,
                    mlp_hidden=N).eval()
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    with torch.no_grad():
        torch.testing.assert_close(m(mix, enrol), m(mix, enrol, context=None))


def test_at_init_the_arm_equals_its_1a_control_end_to_end():
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    torch.manual_seed(0); ctrl = tiny(cls=BSRNN_TFMAP).eval()
    torch.manual_seed(0); arm = tiny().eval()
    with torch.no_grad():
        torch.testing.assert_close(arm(mix, enrol, enrol_embedding=unit()),
                                   ctrl(mix, enrol), rtol=1e-5, atol=1e-6)


def test_a_different_speaker_changes_the_output_once_gamma_is_nonzero():
    torch.manual_seed(0)
    m = tiny().eval()
    torch.nn.init.normal_(m.context.project.weight, std=0.1)
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    with torch.no_grad():
        a = m(mix, enrol, enrol_embedding=unit(2, seed=1))
        b = m(mix, enrol, enrol_embedding=unit(2, seed=2))
    assert not torch.allclose(a, b), "the identity anchor is doing nothing"


def test_the_encoder_is_not_a_submodule():
    """It must be absent from state_dict, parameters and DataParallel's
    per-forward replication -- the whole reason it lives outside."""
    m = tiny()
    keys = " ".join(m.state_dict())
    assert "ecapa" not in keys.lower() and "embedding_model" not in keys
    added = m.context.n_parameters
    torch.manual_seed(0); ctrl = tiny(cls=BSRNN_TFMAP)
    assert (sum(p.numel() for p in m.parameters())
            - sum(p.numel() for p in ctrl.parameters())) == added


def test_gradients_reach_the_projection_through_the_whole_model():
    torch.manual_seed(0)
    m = tiny()
    m(torch.randn(2, 8000), torch.randn(2, 8000),
      enrol_embedding=unit()).pow(2).sum().backward()
    assert m.context.project.weight.grad.abs().max() > 0


# --- the evaluation path --------------------------------------------------

def test_the_embedding_is_reusable_across_many_forwards():
    """THE LATENCY CLAIM, as a test. The whole design rests on the embedding
    being computed ONCE per utterance and reused for every chunk, so reusing one
    across forwards must give exactly the same answer as passing it each time.
    If this ever fails, measure_rtf.py's placement of the embedding outside the
    timing loop becomes a lie rather than an optimisation."""
    torch.manual_seed(0)
    m = tiny().eval()
    torch.nn.init.normal_(m.context.project.weight, std=0.1)
    e = unit(1)
    chunks = [torch.randn(1, 2000) for _ in range(3)]
    enrol = torch.randn(1, 8000)
    with torch.no_grad():
        once = [m(c, enrol, enrol_embedding=e) for c in chunks]
        again = [m(c, enrol, enrol_embedding=e) for c in chunks]
    for a, b in zip(once, again):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_context_kwargs_is_a_noop_without_an_encoder():
    """Every baseline and 1a call site goes through this helper; it must leave
    them byte-identical."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import context_kwargs
    assert context_kwargs(None, torch.randn(2, 8000)) == {}


def test_build_context_encoder_returns_none_off_the_arm():
    """No speechbrain import, no snapshot, on any config that is not 1c."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import build_context_encoder
    assert build_context_encoder({"model": {}}) is None
    assert build_context_encoder({"model": {"context_embedding": False}}) is None
