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
from src.models.bsrnn import build_bsrnn_tfmap, extract
from src.models.conditioning import build_context_fusion, context_fusion_forward

D, N = 192, 16


def tiny(embedding_dim=D):
    """A narrow model with the cue split into its parts. embedding_dim=None gives the same model without the fusion, its control."""
    return build_bsrnn_tfmap(sample_rate=16000, n_fft=512, hop=128, feature_dim=N, hidden_dim=N, num_repeat=2, mlp_hidden=N, tfmap_scale=16.0, tfmap_parts=True, tfmap_part_scales=[1.0, 1.0, 1.0], embedding_dim=embedding_dim)


def unit(batch=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.functional.normalize(torch.randn(batch, D, generator=g), dim=-1)


# --- the fusion in isolation ---------------------------------------------

def test_zero_init_is_exactly_the_identity():
    """At initialisation the arm must be BIT-IDENTICAL to its 1a control, so a
    difference can never be credited to a different starting point."""
    f = build_context_fusion(D, N)
    z = torch.randn(2, 4, N, 7)
    torch.testing.assert_close(context_fusion_forward(f, z, unit()), z, rtol=0, atol=0)


def test_it_still_learns_from_a_zero_start():
    """Zero init must not be a dead start: dL/dW = (dL/dz') * z * e^T is
    non-zero wherever z is."""
    f = build_context_fusion(D, N)
    z = torch.randn(2, 4, N, 7)
    context_fusion_forward(f, z, unit()).pow(2).sum().backward()
    assert f.project.weight.grad.abs().max() > 0


def test_gamma_varies_with_the_speaker_and_nothing_else():
    """gamma depends on WHO, not on when, where, or how loud -- the property
    that makes it un-confoundable by the per-frame loudness the cue rides."""
    f = build_context_fusion(D, N)
    torch.nn.init.normal_(f.project.weight, std=0.1)
    a, b = unit(1, seed=1), unit(1, seed=2)
    assert not torch.allclose(f.project(a), f.project(b))
    torch.testing.assert_close(f.project(a), f.project(a))  # deterministic
    assert f.project(a).shape == (1, N)                    # one per FEATURE


def test_the_same_gain_is_applied_to_every_band_and_frame():
    f = build_context_fusion(D, N)
    torch.nn.init.normal_(f.project.weight, std=0.1)
    e = unit(1)
    z = torch.ones(1, 4, N, 7)
    out = context_fusion_forward(f, z, e)
    expected = (1.0 + f.project(e))[0]                   # (N,)
    for k in range(4):
        for t in range(7):
            torch.testing.assert_close(out[0, k, :, t], expected)


def test_parameter_count_is_the_projection_and_nothing_else():
    f = build_context_fusion(D, 128)
    assert sum(p.numel() for p in f.parameters()) == 128 * D + 128 == 24_704


def test_batch_mismatch_is_refused():
    f = build_context_fusion(D, N)
    with pytest.raises(ValueError, match="batch of"):
        context_fusion_forward(f, torch.randn(3, 4, N, 7), unit(2))


# --- the model ------------------------------------------------------------

def test_the_embedding_is_required_not_optional():
    """THE POINT OF fuse_context. A call site that forgets the embedding must
    crash, never run unconditioned and still score."""
    m = tiny().eval()
    with pytest.raises(ValueError, match="needs the enrolled speaker's embedding"):
        extract(m, torch.randn(2, 8000), torch.randn(2, 8000), None)


def test_the_baseline_runs_without_an_embedding_and_refuses_one():
    """The pre-2026-09-22 call still works, and handing the baseline an
    embedding is an error rather than silently ignored."""
    m = build_bsrnn_tfmap(sample_rate=16000, n_fft=512, hop=128, feature_dim=N, hidden_dim=N, num_repeat=2, mlp_hidden=N, tfmap_scale=16.0, tfmap_parts=False, tfmap_part_scales=None, embedding_dim=None).eval()
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    with torch.no_grad():
        assert extract(m, mix, enrol, None).shape == mix.shape
        with pytest.raises(ValueError, match="baseline"):
            extract(m, mix, enrol, enrol_embedding=unit())


def test_at_init_the_arm_equals_its_1a_control_end_to_end():
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    torch.manual_seed(0); ctrl = tiny(embedding_dim=None).eval()
    torch.manual_seed(0); arm = tiny().eval()
    with torch.no_grad():
        torch.testing.assert_close(extract(arm, mix, enrol, enrol_embedding=unit()), extract(ctrl, mix, enrol, None), rtol=1e-5, atol=1e-6)


def test_a_different_speaker_changes_the_output_once_gamma_is_nonzero():
    torch.manual_seed(0)
    m = tiny().eval()
    torch.nn.init.normal_(m.context.project.weight, std=0.1)
    mix, enrol = torch.randn(2, 8000), torch.randn(2, 8000)
    with torch.no_grad():
        a = extract(m, mix, enrol, enrol_embedding=unit(2, seed=1))
        b = extract(m, mix, enrol, enrol_embedding=unit(2, seed=2))
    assert not torch.allclose(a, b), "the identity anchor is doing nothing"


def test_the_encoder_is_not_a_submodule():
    """It must be absent from state_dict and parameters -- the reason it
    lives outside."""
    m = tiny()
    keys = " ".join(m.state_dict())
    assert "ecapa" not in keys.lower() and "embedding_model" not in keys
    added = sum(p.numel() for p in m.context.parameters())
    torch.manual_seed(0); ctrl = tiny(embedding_dim=None)
    assert (sum(p.numel() for p in m.parameters())
            - sum(p.numel() for p in ctrl.parameters())) == added


def test_gradients_reach_the_projection_through_the_whole_model():
    torch.manual_seed(0)
    m = tiny()
    extract(m, torch.randn(2, 8000), torch.randn(2, 8000), enrol_embedding=unit()).pow(2).sum().backward()
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
        once = [extract(m, c, enrol, enrol_embedding=e) for c in chunks]
        again = [extract(m, c, enrol, enrol_embedding=e) for c in chunks]
    for a, b in zip(once, again):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_speaker_embedding_is_none_without_an_encoder():
    """Every baseline and 1a call site goes through this helper; it must hand
    the model no embedding."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import speaker_embedding
    assert speaker_embedding(None, torch.randn(2, 8000)) is None


def test_build_speaker_encoder_returns_none_off_the_arm():
    """No snapshot is loaded on any config that is not 1c."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import build_speaker_encoder
    assert build_speaker_encoder({"model": {}}, "cpu") is None
    assert build_speaker_encoder({"model": {"context_embedding": False}}, "cpu") is None
