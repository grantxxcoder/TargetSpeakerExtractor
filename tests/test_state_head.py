"""Unit tests for head A: src/models/state_head.py and its loss. D14 piece A.

What these can and cannot check. They cannot say whether head A helps -- that is
a training arm. What they CAN pin down is every property the arm's claim rests
on, each of which would otherwise be an assumption in a write-up:

  * the parameter count is 516, the number D14 records
  * the head changes no audio, so "no capacity confound" is true rather than
    intended
  * gradient does reach the separator (the entire mechanism) and stops at the
    detach flag (the diagnostic control)
  * turning the head on does not disturb the global RNG, so the arm and its
    control draw the same data in the same order -- the failure measured on
    2026-09-11 and the reason the head forks the RNG at construction
  * an old checkpoint still loads strict into a head-equipped model once the
    head is stripped, and into a headless one untouched
  * the class weights are inverse frequency, average 1, and survive an empty
    class rather than becoming inf
"""

import numpy as np
import pytest
import torch

from src.models.bsrnn import BSRNN_TFMAP
from src.models import state_head as sh
from src.models.losses_state_head import LossBSRNNStateHead


def tiny_model(**kwargs):
    """A small but structurally real BSRNN_TFMAP: every module the shipped one
    has, one block wide instead of six. Five uniform bands keep it fast; nothing
    under test depends on the rate or the plan, only on the (B, K, N, T) shape
    the separator emits."""
    return BSRNN_TFMAP(sample_rate=8000, n_fft=128, hop=32, num_repeat=1,
                       band_segments={"uniform": 5}, feature_dim=16,
                       hidden_dim=16, mlp_hidden=16, **kwargs)


# --- the head itself ------------------------------------------------------

def test_parameter_count_is_516_at_the_shipped_width():
    """The number D14 records. 4 x 128 weights + 4 biases."""
    assert sh.AuxStateHead(feature_dim=128).n_parameters == 516


def test_shape_is_one_logit_set_per_frame():
    head = sh.AuxStateHead(feature_dim=16)
    logits = head(torch.randn(2, 5, 16, 37))     # (B, K, N, T)
    assert logits.shape == (2, 4, 37)


def test_head_pools_bands_by_the_mean():
    """Not a style check: the loss's alignment assertion and the probe script
    both assume this exact reduction, so a change to sum or max must fail here
    rather than surface as a scale mismatch later."""
    head = sh.AuxStateHead(feature_dim=16)
    z = torch.randn(1, 5, 16, 9)
    expected = head.classifier(z.mean(dim=1))
    assert torch.allclose(head(z), expected)


def test_rejects_the_wrong_rank():
    with pytest.raises(AssertionError):
        sh.AuxStateHead(feature_dim=16)(torch.randn(2, 16, 37))


# --- class weights --------------------------------------------------------

def test_weights_are_inverse_frequency():
    w = sh.inverse_frequency_weights([100, 100, 100, 100])
    assert torch.allclose(w, torch.ones(4))


def test_rarer_class_gets_the_larger_weight():
    """`both` is the minority at frame level (16.9 %), which is the whole reason
    the weighting exists."""
    w = sh.inverse_frequency_weights([307, 324, 200, 169])
    # Order is the state code: none 30.7 %, target 32.4 %, interferer 20.0 %,
    # both 16.9 %. Rarest gets the most weight: both > interferer > none > target.
    assert w[3] > w[2] > w[0] > w[1] > 0
    assert w.mean().item() == pytest.approx(1.0, abs=1e-5)


def test_an_absent_class_gets_zero_not_infinity():
    w = sh.inverse_frequency_weights([10, 10, 10, 0])
    assert w[3].item() == 0.0
    assert torch.isfinite(w).all()


def test_chance_costs_ln4_with_normalised_weights():
    """The readability claim in the docstring: above ~1.386 nats the head has
    learnt nothing, without needing a baseline run to say so."""
    weight = sh.inverse_frequency_weights([307, 324, 200, 169])
    logits = torch.zeros(4, 4, 50)                      # uniform, i.e. chance
    labels = torch.randint(0, 4, (4, 50))
    loss = sh.state_cross_entropy(logits, labels, weight=weight)
    assert loss.item() == pytest.approx(np.log(4), abs=1e-5)


def test_ignore_index_drops_unlabelled_frames():
    logits = torch.zeros(1, 4, 3)
    labels = torch.tensor([[0, -1, -1]])
    assert sh.state_cross_entropy(logits, labels).item() == pytest.approx(np.log(4), abs=1e-5)


# --- per-class recall -----------------------------------------------------

def test_recall_is_per_class_and_nan_for_an_absent_state():
    logits = torch.full((1, 4, 4), -10.0)
    logits[0, 0, :] = 10.0                     # always predicts `none`
    labels = torch.tensor([[0, 0, 1, 1]])
    recall = sh.per_class_recall(logits, labels)
    assert recall[0].item() == 1.0             # none: perfect
    assert recall[1].item() == 0.0             # target: never predicted
    assert torch.isnan(recall[2:]).all()       # absent from the batch


# --- the model wiring -----------------------------------------------------

def test_off_by_default_and_the_return_type_is_unchanged():
    model = tiny_model()
    assert model.state_head is None
    out = model(torch.randn(1, 4000), torch.randn(1, 4000))
    assert isinstance(out, torch.Tensor)


def test_asking_for_state_without_a_head_raises():
    """Silence here would train a baseline and report it as head A."""
    model = tiny_model()
    with pytest.raises(RuntimeError, match="without head A"):
        model(torch.randn(1, 4000), torch.randn(1, 4000), return_state=True)


def test_the_head_changes_no_audio():
    """The claim the whole arm rests on: same weights, same output samples,
    head or no head. Built twice at the same seed, so the separator's weights
    are identical and only the branch differs."""
    torch.manual_seed(0)
    plain = tiny_model()
    torch.manual_seed(0)
    headed = tiny_model(state_head=True)

    mixture, enrolment = torch.randn(1, 4000), torch.randn(1, 4000)
    with torch.no_grad():
        a = plain(mixture, enrolment)
        b, logits = headed(mixture, enrolment, return_state=True)
    assert torch.allclose(a, b, atol=0, rtol=0)
    assert logits.shape[1] == 4


def test_logits_have_one_frame_per_output_frame():
    model = tiny_model(state_head=True)
    _, logits = model(torch.randn(2, 4000), torch.randn(2, 4000), return_state=True)
    z_frames = model.stft(torch.randn(1, 4000)).shape[-1]
    assert logits.shape == (2, 4, z_frames)


def test_gradient_from_the_head_reaches_the_separator():
    """The mechanism. If this fails the head is an ornament."""
    model = tiny_model(state_head=True)
    _, logits = model(torch.randn(1, 4000), torch.randn(1, 4000), return_state=True)
    sh.state_cross_entropy(logits, torch.zeros(1, logits.shape[-1], dtype=torch.long)).backward()
    grads = [p.grad for p in model.separator.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)


def test_detach_stops_the_gradient_at_the_head():
    """The diagnostic arm: the head still learns to READ the features but
    applies no pressure, which is what separates 'the loss changed the
    features' from 'they were always like this'."""
    model = tiny_model(state_head=True, state_head_detach=True)
    _, logits = model(torch.randn(1, 4000), torch.randn(1, 4000), return_state=True)
    sh.state_cross_entropy(logits, torch.zeros(1, logits.shape[-1], dtype=torch.long)).backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for p in model.separator.parameters())
    assert model.state_head.classifier.weight.grad.abs().sum() > 0


def test_turning_the_head_on_does_not_move_the_global_rng():
    """MEASURED 2026-09-11: constructing extra modules before the first batch is
    drawn advances the global RNG, changes the dataloader shuffle and changes
    which trial drop_last discards -- so the arm and its control stop seeing the
    same data. The head forks the RNG so that cannot happen here."""
    torch.manual_seed(42)
    tiny_model()
    without = torch.randn(3)

    torch.manual_seed(42)
    tiny_model(state_head=True)
    with_head = torch.randn(3)

    assert torch.equal(without, with_head)


def test_the_head_still_initialises_to_something_nonconstant():
    """The RNG fork must not collapse the head to a fixed or zero tensor."""
    head = sh.AuxStateHead(feature_dim=16)
    assert head.classifier.weight.std() > 0


# --- checkpoint compatibility --------------------------------------------

def test_a_stripped_checkpoint_loads_strict_into_the_baseline():
    """An eval script builds the baseline architecture. A head-A checkpoint must
    load into it without strict=False, which would also swallow a genuinely
    missing separator weight."""
    torch.manual_seed(0)
    headed = tiny_model(state_head=True)
    torch.manual_seed(0)
    plain = tiny_model()
    plain.load_state_dict(sh.drop_state_head(headed.state_dict()), strict=True)


def test_an_old_checkpoint_still_loads_into_a_headless_model():
    torch.manual_seed(0)
    a = tiny_model()
    torch.manual_seed(0)
    b = tiny_model()
    b.load_state_dict(a.state_dict(), strict=True)


# --- the loss subclass ----------------------------------------------------

def loss_inputs(batch=4, samples=800):
    target = torch.randn(batch, samples)
    output = target + 0.1 * torch.randn(batch, samples)
    mixture = target + torch.randn(batch, samples)
    absent = torch.tensor([False, False, True, True])
    target[absent] = 0.0
    return target, output, mixture, absent


def test_the_term_adds_on_top_of_an_unchanged_base():
    """The subclass must not perturb the M2 objective. At w_state_head = 0 the
    total is the parent's, to the last bit."""
    from src.models.losses import LossBSRNN

    torch.manual_seed(0)
    args = dict(wm=9.62, w=0.458, wg=1.69)
    base = LossBSRNN(**args)
    head = LossBSRNNStateHead(**args, w_state_head=0.0,
                              class_counts=[307, 324, 200, 169])
    head.state_logits = torch.randn(4, 4, 25)
    head.state_labels = torch.randint(0, 4, (4, 25))

    inputs = loss_inputs()
    base_total, _ = base(*inputs)
    head_total, parts = head(*inputs)
    assert float(head_total) == pytest.approx(float(base_total), abs=1e-9)
    assert "L_head" in parts and parts["L_head"] > 0


def test_the_weighted_term_is_added_exactly_once():
    args = dict(wm=9.62, w=0.458, wg=1.69, class_counts=[307, 324, 200, 169])
    inputs = loss_inputs()
    logits, labels = torch.randn(4, 4, 25), torch.randint(0, 4, (4, 25))

    off = LossBSRNNStateHead(**args, w_state_head=0.0)
    on = LossBSRNNStateHead(**args, w_state_head=0.5)
    for fn in (off, on):
        fn.state_logits, fn.state_labels = logits, labels
    off_total, off_parts = off(*inputs)
    on_total, _ = on(*inputs)
    assert float(on_total) == pytest.approx(float(off_total) + 0.5 * off_parts["L_head"],
                                            abs=1e-6)


def test_a_forgotten_assignment_raises_rather_than_scoring_zero():
    fn = LossBSRNNStateHead(wm=9.62, w=0.458, w_state_head=0.1)
    with pytest.raises(RuntimeError, match="never set"):
        fn(*loss_inputs())


def test_misaligned_labels_are_refused():
    """A frame-count mismatch means the head and the loader disagree about
    framing -- a silent accuracy cost that would be blamed on the idea."""
    fn = LossBSRNNStateHead(wm=9.62, w=0.458, w_state_head=0.1)
    fn.state_logits = torch.randn(4, 4, 25)
    fn.state_labels = torch.randint(0, 4, (4, 24))
    with pytest.raises(AssertionError, match="do not line up"):
        fn(*loss_inputs())


def test_per_class_recall_is_logged_alongside_the_term():
    fn = LossBSRNNStateHead(wm=9.62, w=0.458, w_state_head=0.1,
                            class_counts=[307, 324, 200, 169])
    fn.state_logits = torch.randn(4, 4, 25)
    fn.state_labels = torch.randint(0, 4, (4, 25))
    _, parts = fn(*loss_inputs())
    assert all(f"head_recall_{s}" in parts for s in range(4))
    assert "head_bal_acc" in parts
