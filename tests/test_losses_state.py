"""Unit tests for src/models/losses_state.py -- the frozen-teacher term (D14).

A loss function's failure mode is silence, and this term has a worse one than
most: it can compute a believable number, log a curve that moves, and have
EXACTLY ZERO effect on any weight. That happens whenever the teacher's forward
is wrapped in `no_grad` or `inference_mode` instead of being frozen with
`requires_grad = False`.

Nothing else catches it. Shapes pass. The loss value is a real BCE of real
probabilities. The training curve still moves, because the extractor changes for
other reasons. `w_state = 0` still reproduces the baseline, trivially. The ONLY
observable difference is whether a gradient exists -- so that is what these
tests assert.

This project has already had this bug once, on paper: D9's original plan was to
add WER to the loss, and the log records why it does nothing -- "a term that does
not vary with the weights has zero gradient, so backprop would produce identical
updates and only the printed loss would change". Same failure, different route.

The teacher is loaded once per session: it is 21 M frozen parameters and a
hash check, so a fixture rather than a per-test construction.
"""

import numpy as np
import pytest
import torch

from src.models.losses import LossBSRNN
from src.models.losses_state import LossBSRNNState
from src.models.state_teacher import (NON_TARGET_AUDIBLE, StateTeacher,
                                      required_answers)

CHECKPOINT = "models/state_detector_notebook.pt"
ECAPA_DIR = "../ecapa_pretrained"
SAMPLE_RATE = 16000
CHUNK = 64128                      # 4.008 s, the extractor's own chunk length


@pytest.fixture(scope="module")
def teacher():
    return StateTeacher(CHECKPOINT, ECAPA_DIR, device="cpu")


@pytest.fixture(scope="module")
def batch():
    """Two crops: one present, one target-absent. Seeded, so a failure is
    reproducible rather than occasional."""
    generator = torch.Generator().manual_seed(42)
    mixture = torch.randn(2, CHUNK, generator=generator) * 0.05
    target = torch.randn(2, CHUNK, generator=generator) * 0.05
    target[1] = 0.0                                  # the absent crop
    crop_absent = torch.tensor([False, True])
    enrolment = torch.randn(2, 5 * SAMPLE_RATE, generator=generator) * 0.05
    return mixture, target, enrolment, crop_absent


def build(teacher, w_state):
    loss_fn = LossBSRNNState(wm=1.0, w=0.3, wg=1.69, teacher=teacher,
                             w_state=w_state)
    return loss_fn


# --- the test this file exists for ---------------------------------------

def test_gradient_reaches_the_waveform(teacher, batch):
    """THE test. A frozen teacher must still pass gradient to its input.

    Fails if the teacher's forward is wrapped in no_grad or inference_mode,
    which is the one bug that would otherwise be invisible for a whole run.
    """
    mixture, target, enrolment, crop_absent = batch
    estimate = (mixture.clone() * 0.5).requires_grad_(True)

    loss_fn = build(teacher, w_state=1.0)
    loss_fn.enrolment_embedding = teacher.embed_enrolment(enrolment)
    loss, parts = loss_fn(target, estimate, mixture, crop_absent)
    loss.backward()

    assert estimate.grad is not None, "no gradient at all -- the graph is severed"
    assert torch.isfinite(estimate.grad).all(), "gradient holds inf or NaN"
    assert estimate.grad.abs().sum() > 0, (
        "the gradient is exactly zero. The teacher's forward is almost "
        "certainly under no_grad/inference_mode: freeze with "
        "requires_grad=False instead.")


def test_the_state_term_is_what_carries_the_gradient(teacher, batch):
    """The gradient above must come from L_state, not only from L_pres.

    Without this, test_gradient_reaches_the_waveform would pass on a completely
    dead state term, because the base class's terms are differentiable anyway.
    """
    mixture, target, enrolment, crop_absent = batch
    embedding = teacher.embed_enrolment(enrolment)

    gradients = {}
    for w_state in (0.0, 1.0):
        estimate = (mixture.clone() * 0.5).requires_grad_(True)
        loss_fn = build(teacher, w_state=w_state)
        loss_fn.enrolment_embedding = embedding
        loss, _ = loss_fn(target, estimate, mixture, crop_absent)
        loss.backward()
        gradients[w_state] = estimate.grad.clone()

    difference = (gradients[1.0] - gradients[0.0]).abs().sum()
    assert difference > 0, (
        "turning w_state from 0 to 1 changed no gradient, so the term "
        "contributes nothing to the update")


def test_teacher_parameters_receive_no_gradient(teacher, batch):
    """Frozen means frozen. If the teacher trains, it and the extractor can
    satisfy the objective by agreeing with each other while the audio does not
    change -- the collusion the freeze exists to prevent."""
    mixture, target, enrolment, crop_absent = batch
    estimate = (mixture.clone() * 0.5).requires_grad_(True)

    loss_fn = build(teacher, w_state=1.0)
    loss_fn.enrolment_embedding = teacher.embed_enrolment(enrolment)
    loss, _ = loss_fn(target, estimate, mixture, crop_absent)
    loss.backward()

    for name, parameter in teacher.named_parameters():
        assert not parameter.requires_grad, f"{name} is trainable"
        assert parameter.grad is None, f"{name} accumulated a gradient"


# --- the term's arithmetic ------------------------------------------------

def test_w_state_zero_reproduces_the_base_objective(teacher, batch):
    """At w_state = 0 the total must equal LossBSRNN's exactly.

    Not a tolerance: the same terms in the same order should give the same
    float. Any difference means the subclass changed the base arithmetic.
    """
    mixture, target, enrolment, crop_absent = batch

    base = LossBSRNN(wm=1.0, w=0.3, wg=1.69)
    base_total, base_parts = base(target, mixture * 0.5, mixture, crop_absent)

    loss_fn = build(teacher, w_state=0.0)
    loss_fn.enrolment_embedding = teacher.embed_enrolment(enrolment)
    total, parts = loss_fn(target, mixture * 0.5, mixture, crop_absent)

    assert float(total) == float(base_total)
    for key in ("L_pres", "L_MR", "L_gain", "L_abs"):
        assert parts[key] == base_parts[key] or (
            np.isnan(parts[key]) and np.isnan(base_parts[key]))


def test_forgetting_the_enrolment_raises(teacher, batch):
    """A silently wrong enrolment would make this term reward extracting the
    WRONG speaker, so the omission must be loud."""
    mixture, target, _, crop_absent = batch
    loss_fn = build(teacher, w_state=1.0)
    with pytest.raises(RuntimeError, match="enrolment_embedding"):
        loss_fn(target, mixture * 0.5, mixture, crop_absent)


def test_silence_scores_better_than_the_mixture(teacher, batch):
    """The term must prefer an output with no second voice in it.

    A directional sanity check on the whole chain -- teacher, column index,
    sign of the BCE target. Silence contains no non-target voice; the raw
    two-speaker mixture does. If the column index or the target were flipped,
    this inverts.
    """
    mixture, target, enrolment, crop_absent = batch
    loss_fn = build(teacher, w_state=1.0)
    loss_fn.enrolment_embedding = teacher.embed_enrolment(enrolment)

    with torch.no_grad():
        on_silence = loss_fn._loss_non_target_audible(
            torch.zeros_like(mixture)).mean()
        on_mixture = loss_fn._loss_non_target_audible(mixture).mean()

    assert on_silence < on_mixture, (
        f"silence scored {float(on_silence):.4f} against the mixture's "
        f"{float(on_mixture):.4f} -- the column index or the BCE target is "
        f"inverted")


def test_the_term_is_logged_unweighted(teacher, batch):
    """parts['L_state'] must be the raw term, so a break-even weight can be
    derived from it the way scripts/derive_w_g.py derives wg."""
    mixture, target, enrolment, crop_absent = batch
    embedding = teacher.embed_enrolment(enrolment)

    reported = {}
    for w_state in (0.0, 2.0):
        loss_fn = build(teacher, w_state=w_state)
        loss_fn.enrolment_embedding = embedding
        with torch.no_grad():
            _, parts = loss_fn(target, mixture * 0.5, mixture, crop_absent)
        reported[w_state] = parts["L_state"]

    assert reported[0.0] == pytest.approx(reported[2.0], rel=1e-6)


def test_required_answers_never_asks_for_a_non_target(teacher):
    """No input state may require a non-target to be audible in the output. If
    one could, the loss would be rewarding leakage."""
    target_was_speaking = torch.tensor([[True, False, True, False]])
    answers = required_answers(target_was_speaking)
    assert answers.shape == (1, 4, 2)
    assert float(answers[..., NON_TARGET_AUDIBLE].abs().sum()) == 0.0
    assert answers[0, :, 0].tolist() == [1.0, 0.0, 1.0, 0.0]
