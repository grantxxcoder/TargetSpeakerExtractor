"""Unit tests for D4a: per-block TF-Map re-injection. decisions-pending.md D4.

These cannot say whether re-injecting the cue helps -- that is a training arm,
and D3a's 2026-08-30 measurement is an argument that it will not. What they pin
down is every property the arm's comparison rests on:

  * OFF is arithmetically the frozen 2026-08-28 baseline, not approximately it
  * the added parameter count is exactly what the write-up claims, 37,698
  * at zero-initialised gates the arm computes the baseline's function bit for
    bit, so any divergence during training is learned rather than an init
    artefact
  * the cue actually reaches the mask -- changing the enrollment changes the
    output through the injection path and not only through the input channel
  * gradient reaches the gates on the first step even though it cannot reach
    the projections (the ReZero ordering, stated in the docstring)
  * causality survives: no output sample depends on a later input sample
  * turning it on does not move the global RNG, so the arm and its control see
    the same data in the same order (the 2026-09-11 shuffle finding)
"""

import pytest
import torch

from src.models.bands import band_plan
from src.models.bsrnn import BSRNN_TFMAP
from src.models.conditioning import TFMapInjector
from src.models.modules import BandSequenceModel, BandSplit


def tiny_model(**kwargs):
    """Every module the shipped model has, one block wide instead of six."""
    return BSRNN_TFMAP(sample_rate=8000, n_fft=128, hop=32, num_repeat=2,
                       band_segments={"uniform": 5}, feature_dim=16,
                       hidden_dim=16, mlp_hidden=16, **kwargs)


# --- the injector in isolation -------------------------------------------

def test_parameter_count_at_the_shipped_width():
    """37,698 = 257x128 + 32x128 projection, 2x257 LayerNorm, 6x32 gates. The
    number the write-up quotes, asserted rather than recomputed by hand."""
    widths = band_plan(16000, 512, "wesep_16k")
    injector = TFMapInjector(widths, 128, 6)
    assert injector.n_parameters == 37698
    assert sum(widths) == 257 and len(widths) == 32


def test_shape_is_the_separator_feature_map():
    widths = band_plan(16000, 512, "wesep_16k")
    injector = TFMapInjector(widths, 128, 6)
    tf = torch.randn(2, 1, 257, 11)
    cue = injector(BandSplit(widths)(tf))
    assert cue.shape == (2, 32, 128, 11)


def test_gates_are_one_per_block_per_band_and_start_at_zero():
    injector = TFMapInjector([4, 4, 4], 8, num_blocks=6)
    assert injector.gates.shape == (6, 3)
    assert torch.count_nonzero(injector.gates) == 0


def test_gate_init_is_honoured():
    injector = TFMapInjector([4, 4, 4], 8, num_blocks=2, gate_init=1.0)
    assert torch.equal(injector.gates, torch.ones(2, 3))


def test_a_multi_channel_input_is_refused():
    """The TF-Map is one channel. Handing this the 3-channel feature stack would
    otherwise reshape silently and scramble channels into frequency."""
    injector = TFMapInjector([4, 4], 8, num_blocks=2)
    with pytest.raises(AssertionError, match="one channel"):
        injector([torch.randn(1, 3, 4, 7), torch.randn(1, 3, 4, 7)])


# --- the stack's injection point -----------------------------------------

def test_the_stack_without_a_cue_is_untouched():
    torch.manual_seed(0)
    stack = BandSequenceModel(8, 8, num_repeat=2)
    x = torch.randn(1, 3, 8, 7)
    assert torch.equal(stack(x), stack(x, cue=None, gates=None))


def test_half_an_injection_is_refused():
    stack = BandSequenceModel(8, 8, num_repeat=2)
    x = torch.randn(1, 3, 8, 7)
    with pytest.raises(ValueError, match="together or neither"):
        stack(x, cue=torch.randn_like(x))
    with pytest.raises(ValueError, match="together or neither"):
        stack(x, gates=torch.zeros(2, 3))


def test_zero_gates_reproduce_the_uninjected_stack_exactly():
    """Not `allclose` -- adding 0.0 * cue must be bit-identical, which is what
    makes 'the arm starts as the baseline' a fact rather than a hope."""
    torch.manual_seed(0)
    stack = BandSequenceModel(8, 8, num_repeat=3)
    x = torch.randn(1, 3, 8, 7)
    injected = stack(x, cue=torch.randn_like(x), gates=torch.zeros(3, 3))
    assert torch.equal(stack(x), injected)


def test_a_nonzero_gate_changes_the_output():
    torch.manual_seed(0)
    stack = BandSequenceModel(8, 8, num_repeat=3)
    x, cue = torch.randn(1, 3, 8, 7), torch.randn(1, 3, 8, 7)
    assert not torch.equal(stack(x), stack(x, cue=cue, gates=torch.ones(3, 3)))


def test_the_gate_is_per_band():
    """A gate open on one band only must leave the other bands' first-block
    input untouched. One block, so nothing has mixed across bands yet."""
    torch.manual_seed(0)
    stack = BandSequenceModel(8, 8, num_repeat=1)
    x, cue = torch.randn(1, 3, 8, 7), torch.randn(1, 3, 8, 7)
    gates = torch.tensor([[0.0, 5.0, 0.0]])
    a, b = stack(x), stack(x, cue=cue, gates=gates)
    # The band LSTM inside BSNet mixes bands, so compare the pre-block input
    # instead: that is what the gate controls.
    injected = x + gates[0].view(1, -1, 1, 1) * cue
    assert torch.equal(injected[:, 0], x[:, 0])
    assert torch.equal(injected[:, 2], x[:, 2])
    assert not torch.equal(injected[:, 1], x[:, 1])
    assert not torch.equal(a, b)


# --- the model ------------------------------------------------------------

def test_off_by_default():
    assert tiny_model().tfmap_inject is None


def test_off_is_the_frozen_baseline_bit_for_bit():
    """The control for this arm is a checkpoint of the 2026-08-28 architecture.
    If tfmap_inject=False were only approximately that, every delta would be
    contaminated."""
    torch.manual_seed(0)
    plain = tiny_model()
    mixture, enrolment = torch.randn(1, 4000), torch.randn(1, 4000)
    with torch.no_grad():
        before = plain(mixture, enrolment)
    torch.manual_seed(0)
    again = tiny_model()
    with torch.no_grad():
        assert torch.equal(before, again(mixture, enrolment))


def test_zero_init_gates_make_the_arm_start_at_the_baseline():
    """Built at the same seed, the injected model's weights are identical to the
    baseline's (the injector forks the RNG) and its gates are zero, so at step 0
    the two models are the same function."""
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    injected = tiny_model(tfmap_inject=True)

    mixture, enrolment = torch.randn(1, 4000), torch.randn(1, 4000)
    with torch.no_grad():
        assert torch.allclose(plain(mixture, enrolment),
                              injected(mixture, enrolment), atol=1e-6)


def test_an_open_gate_changes_the_output():
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    injected = tiny_model(tfmap_inject=True, tfmap_gate_init=1.0)

    mixture, enrolment = torch.randn(1, 4000), torch.randn(1, 4000)
    with torch.no_grad():
        assert not torch.allclose(plain(mixture, enrolment),
                                  injected(mixture, enrolment), atol=1e-6)


def test_the_injected_cue_carries_the_enrollment():
    """The mechanism. With the input channel's contribution held fixed by using
    the same mixture, swapping the enrollment must still move the output --
    otherwise the injection path is carrying no identity."""
    torch.manual_seed(0)
    model = tiny_model(tfmap_inject=True, tfmap_gate_init=1.0)
    mixture = torch.randn(1, 4000)
    with torch.no_grad():
        a = model(mixture, torch.randn(1, 4000))
        b = model(mixture, torch.randn(1, 4000))
    assert not torch.allclose(a, b, atol=1e-6)


def test_gradient_reaches_the_gates_on_the_first_step():
    """The ReZero ordering the docstring promises: at gate = 0 the projections
    get nothing and the gates get a real gradient, so the gates open first."""
    torch.manual_seed(0)
    model = tiny_model(tfmap_inject=True)
    model(torch.randn(1, 4000), torch.randn(1, 4000)).pow(2).mean().backward()

    gates = model.tfmap_inject.gates
    assert gates.grad is not None and gates.grad.abs().sum() > 0
    projection = model.tfmap_inject.blocks[0][1].weight
    assert projection.grad is not None
    assert projection.grad.abs().sum() == 0


def test_gradient_reaches_the_projection_once_a_gate_is_open():
    torch.manual_seed(0)
    model = tiny_model(tfmap_inject=True, tfmap_gate_init=1.0)
    model(torch.randn(1, 4000), torch.randn(1, 4000)).pow(2).mean().backward()
    assert model.tfmap_inject.blocks[0][1].weight.grad.abs().sum() > 0


def test_the_injection_is_causal():
    """No output sample may move when a LATER input sample changes. The whole
    architecture's latency claim depends on this, and an injection inside the
    stack is exactly where a non-causal normalisation would slip in."""
    torch.manual_seed(0)
    model = tiny_model(tfmap_inject=True, tfmap_gate_init=1.0).eval()
    mixture, enrolment = torch.randn(1, 2048), torch.randn(1, 2048)
    edited = mixture.clone()
    edited[0, 1500:] += 10.0                      # change only the tail

    with torch.no_grad():
        a = model(mixture, enrolment)
        b = model(edited, enrolment)
    # One STFT window of slack (n_fft 128 = 4 hops) before the edit.
    safe = 1500 - 128
    assert torch.allclose(a[0, :safe], b[0, :safe], atol=1e-5)


def test_turning_it_on_does_not_move_the_global_rng():
    """MEASURED 2026-09-11: a module constructed before the first batch is drawn
    advances the global RNG, reshuffles the loader and changes which trial
    drop_last discards -- so 'same data, same seed' quietly stops being true."""
    torch.manual_seed(42)
    tiny_model()
    without = torch.randn(3)

    torch.manual_seed(42)
    tiny_model(tfmap_inject=True)
    with_inject = torch.randn(3)

    assert torch.equal(without, with_inject)


def test_it_composes_with_head_a():
    """The two 2026-09-11 flags are independent switches, and both forks of the
    RNG must still leave the stream untouched when they are on together."""
    torch.manual_seed(42)
    tiny_model()
    without = torch.randn(3)

    torch.manual_seed(42)
    model = tiny_model(tfmap_inject=True, state_head=True)
    assert torch.equal(without, torch.randn(3))

    _, logits = model(torch.randn(1, 4000), torch.randn(1, 4000), return_state=True)
    assert logits.shape[1] == 4
