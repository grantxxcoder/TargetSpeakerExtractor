"""The presence gate's "postfilter" mode (src/models/bsrnn.py, losses_presence.py; decisions-m2.md 2026-10-18).

What the fix rests on, each pinned here rather than assumed:
  * the extractor's own (ungated) output is bit-identical to a gate-less model with the same weights, so the
    extractor is e21 when seeded from e21 and gate-on vs e21 is a paired comparison;
  * no gradient of the gated loss, the KL or the BCE reaches the separator (the memorisation channel is closed);
  * the gated loss does reach the gate's 129 weights;
  * the gain is floor + (1 - floor) * pi, the same in train() and eval() (no draw, no tau);
  * the loss total is base(gated) + base(ungated) + beta * KL + lam * BCE, with the extractor's terms logged;
  * "joint" mode (arm A as trained 2026-10-15..18) is untouched: two outputs, a draw in training.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.bsrnn import BSRNN_TFMAP  # noqa: E402
from src.models.losses import LossBSRNN  # noqa: E402
from src.models.losses_presence import LossBSRNNPresence  # noqa: E402
from src.models.presence_gate import gain_from_logits, markov_kl, supervised_bce  # noqa: E402

SR, N_FFT, HOP = 8000, 128, 32
B, N = 2, 4000


def tiny(**kw):
    """One block wide, five uniform bands: every module the shipped model has."""
    return BSRNN_TFMAP(sample_rate=SR, n_fft=N_FFT, hop=HOP, num_repeat=1, band_segments={"uniform": 5},
                       feature_dim=16, hidden_dim=16, mlp_hidden=16, **kw)


def inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    target = torch.randn(B, N, generator=g)
    mixture = target + 0.5 * torch.randn(B, N, generator=g)
    enrol = torch.randn(B, N, generator=g)
    return mixture, target, enrol


def n_frames(model, n):
    return model.stft(torch.zeros(1, n)).shape[-1]


def test_ungated_output_is_the_gateless_model_bit_for_bit():
    torch.manual_seed(0)
    plain = tiny()
    gated = tiny(presence_gate=True, presence_mode="postfilter", presence_floor=0.05)
    missing, unexpected = gated.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("presence_gate.") for k in missing)
    mixture, _, enrol = inputs()
    plain.eval(), gated.eval()
    with torch.no_grad():
        y_plain = plain(mixture, enrol)
        y_gated, logits, y_ungated = gated(mixture, enrol, return_gate=True)
    assert torch.equal(y_plain, y_ungated)
    assert logits.shape == (B, n_frames(gated, N))
    assert not torch.equal(y_plain, y_gated)   # the gate did something


def test_gated_output_is_the_floored_gain_times_the_ungated_spectrum():
    torch.manual_seed(0)
    floor = 0.05
    model = tiny(presence_gate=True, presence_mode="postfilter", presence_floor=floor).eval()
    with torch.no_grad():
        model.presence_gate.linear.weight.zero_()
        model.presence_gate.linear.bias.fill_(2.0)        # every frame: pi = sigmoid(2)
    mixture, _, enrol = inputs()
    with torch.no_grad():
        y_gated, logits, y_ungated = model(mixture, enrol, return_gate=True)
    gain = floor + (1 - floor) * torch.sigmoid(torch.tensor(2.0))
    assert torch.allclose(logits, torch.full_like(logits, 2.0), atol=1e-5)
    # A constant gain commutes with the linear inverse STFT.
    assert torch.allclose(y_gated, gain * y_ungated, atol=1e-5)
    assert torch.allclose(gain_from_logits(torch.tensor(0.0), 0.0), torch.tensor(0.5))
    assert torch.allclose(gain_from_logits(torch.tensor(-1e4), 0.05), torch.tensor(0.05))


def test_train_and_eval_give_the_same_output():
    """No draw in postfilter mode: what trains is what ships."""
    torch.manual_seed(0)
    model = tiny(presence_gate=True, presence_mode="postfilter")
    mixture, _, enrol = inputs()
    with torch.no_grad():
        model.train()
        a = model(mixture, enrol)
        b = model(mixture, enrol)
        model.eval()
        c = model(mixture, enrol)
    assert torch.equal(a, b) and torch.equal(a, c)


def _grads(model, loss):
    model.zero_grad()
    loss.backward()
    gate = {id(p) for p in model.presence_gate.parameters()}
    ext = [p.grad for p in model.parameters() if id(p) not in gate and p.grad is not None]
    gt = [p.grad for p in model.presence_gate.parameters() if p.grad is not None]
    return ext, gt


def test_no_gate_term_reaches_the_separator_and_the_gated_loss_reaches_the_gate():
    torch.manual_seed(0)
    model = tiny(presence_gate=True, presence_mode="postfilter").train()
    mixture, target, enrol = inputs()
    base = LossBSRNN(wm=1.0, w=0.3, sample_rate=SR, windows=(8, 16))
    absent = torch.tensor([False, False])

    # 1. The gated loss alone: the gate learns, the extractor does not.
    y_gated, logits, _ = model(mixture, enrol, return_gate=True)
    loss_gated, _ = base(target, y_gated, mixture, absent)
    ext, gt = _grads(model, loss_gated)
    assert gt and all(g.abs().sum() > 0 for g in gt)
    assert all(g.abs().sum() == 0 for g in ext), "the gated loss moved the separator"

    # 2. KL and BCE alone: same.
    _, logits, _ = model(mixture, enrol, return_gate=True)
    pi = torch.sigmoid(logits.float())
    v = (torch.rand_like(pi) > 0.5).float()
    aux = markov_kl(pi, 0.998, 0.002, 0.55).mean() + supervised_bce(logits, v).mean()
    ext, gt = _grads(model, aux)
    assert gt and all(g.abs().sum() > 0 for g in gt)
    assert all(g.abs().sum() == 0 for g in ext), "a gate term moved the separator"

    # 3. The ungated loss alone: the extractor learns, the gate does not.
    _, _, y_ungated = model(mixture, enrol, return_gate=True)
    loss_ungated, _ = base(target, y_ungated, mixture, absent)
    ext, gt = _grads(model, loss_ungated)
    assert any(g.abs().sum() > 0 for g in ext)
    assert all(g.abs().sum() == 0 for g in gt)


def test_loss_total_is_both_base_terms_plus_the_gate_terms():
    torch.manual_seed(0)
    T = 10
    target = torch.randn(B, N)
    mixture = target + 0.3 * torch.randn(B, N)
    gated, ungated = target + 0.2 * torch.randn(B, N), target + 0.1 * torch.randn(B, N)
    absent = torch.tensor([False, False])
    logits = torch.randn(B, T, requires_grad=True)
    v, v_valid = (torch.rand(B, T) > 0.5).float(), torch.tensor([True, True])
    kw = dict(wm=1.0, w=0.3, sample_rate=SR, windows=(8, 16))
    beta, lam = 0.01, 0.02
    loss = LossBSRNNPresence(beta=beta, lam=lam, d_on=596.0, d_off=490.0, **kw)
    base = LossBSRNN(**kw)
    total, parts = loss(target, gated, mixture, absent, gate_logits=logits, v=v, v_valid=v_valid,
                        s_output_ungated=ungated)
    b_g, p_g = base(target, gated, mixture, absent)
    b_u, p_u = base(target, ungated, mixture, absent)
    pi = torch.sigmoid(logits)
    expected = b_g + b_u + beta * markov_kl(pi, loss.a11, loss.a01, loss.rho).mean() + lam * supervised_bce(logits, v).mean()
    assert torch.allclose(total, expected, atol=1e-5)
    assert parts["L_pres"] == pytest.approx(p_g["L_pres"])            # plain names: the system (gated) output
    assert parts["L_pres_ungated"] == pytest.approx(p_u["L_pres"])    # the extractor alone
    assert parts["L_pres"] != pytest.approx(parts["L_pres_ungated"])
    # Without the ungated output the loss is eq. (12) as before, and logs no *_ungated key.
    total2, parts2 = loss(target, gated, mixture, absent, gate_logits=logits, v=v, v_valid=v_valid)
    assert torch.allclose(total2, expected - b_u, atol=1e-5)
    assert "L_pres_ungated" not in parts2


def test_joint_mode_is_unchanged_and_floor_is_refused_there():
    torch.manual_seed(0)
    model = tiny(presence_gate=True)        # default: joint
    assert model.presence_mode == "joint"
    model.gate_tau = 1.0
    mixture, _, enrol = inputs()
    model.train()
    out = model(mixture, enrol, return_gate=True)
    assert len(out) == 2
    with pytest.raises(ValueError):
        tiny(presence_gate=True, presence_mode="joint", presence_floor=0.05)
    with pytest.raises(ValueError):
        tiny(presence_gate=True, presence_mode="other")


def test_train_helpers_strip_gate_keys_and_split_outputs():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from train import model_block_without_gate, split_gate_output  # noqa: E402
    e21 = {"model": {"bands": {"plan": "wesep_16k"}, "stft": {"n_fft": 512, "hop": 128}}}
    gated = {"model": {**e21["model"], "presence_gate": True, "presence_mode": "postfilter", "presence_floor": 0.05}}
    assert model_block_without_gate(e21) == model_block_without_gate(gated)
    assert model_block_without_gate({"model": {**e21["model"], "lookahead_frames": 2}}) != model_block_without_gate(e21)
    a, b, c = torch.zeros(1), torch.ones(1), torch.full((1,), 2.0)
    assert split_gate_output((a, b)) == (a, b, None)
    assert split_gate_output((a, b, c)) == (a, b, c)
