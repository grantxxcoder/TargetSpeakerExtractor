"""The M2 objective plus one speaker-state term. decisions-pending.md D14.

A SUBCLASS, deliberately, and in its own file: `losses.py` is untouched, so
every run before this term existed stays reproducible by construction rather
than by a weight set to zero. `build_loss()` in scripts/train.py returns the
base class unless the config asks for `w_state`, so an old config cannot reach
this code path at all.

WHAT THE TERM ADDS
------------------
A frozen teacher listens to the extractor's output and answers, per ~1 s
window, "is a non-target voice audible here?". The required answer is ZERO,
everywhere, unconditionally -- there is no situation in which a second voice
belongs in the output. So this half of the teacher needs no labels and no
branch on `crop_absent`.

The teacher's other output, "is the target audible", is NOT used here. It
requires per-window labels from `data/index/state_*.csv` and therefore a loader
change; it is the second increment, not the first.

WHY THIS IS NOT REDUNDANT WITH L_pres
-------------------------------------
`L_pres` maximises SI-SDR to the target, whose denominator sums interferer
leakage, residual noise and invented artefact into one undifferentiated error
(verified orthogonal to six decimals, decisions-m3.md 2026-09-01). Per unit of
energy they cost exactly the same. For THIS project they are not equally bad:
the interferer's words get transcribed as the target's by a live judge, which
is the failure ICR was defined to catch. This term prices that specifically.

WHAT THE TEACHER IS, AND WHAT IT IS NOT
---------------------------------------
Frozen ECAPA-TDNN (Desplanques et al., Interspeech 2020) plus a small
bidirectional readout, fitted once offline on the rendered stems. MEASURED
2026-09-11 on held-out trials: "is a non-target audible" balanced accuracy
0.802, recall 0.795, precision 0.679.

**A noisy proxy, not a detector.** It misses about one leak in five and a third
of the penalties it hands out are undeserved. That is usable because the loss
never thresholds -- it uses the raw probability, so a wrong judgement is a small
push the wrong way, outnumbered by the right ones. Noise costs STRENGTH, not
direction, which is what `w_state` compensates for.

It is also **better at sustained leakage than at brief leakage**: part of the
readout's accuracy comes from temporal smoothing across windows, because speaker
states come in runs. An isolated window of leakage may be smoothed away.

The judge never appears here in any form, and the teacher is not WeSep: its
checkpoint carries an ECAPA encoder, but one trained jointly with its separator,
and WeSep is this project's comparison baseline.

THE FAILURE MODE TO WATCH
-------------------------
Reward-model overoptimisation. A frozen learned scorer optimised against gets
gamed off-distribution eventually, and this teacher has never heard masked audio
with spectral holes -- only real mixtures and synthetic partial suppressions.
Its signature is `L_state` falling while an independent measure stalls, which has
happened twice in this project already (2026-08-25, total loss falling into a
mute; 2026-09-04, conditioning diagnostics peaking on an epoch below
pass-through).

Mitigations belong to the caller and are not optional:
  - NEVER select checkpoints on this term. See train.selection_score().
  - keep `w_state` modest; a proxy that dominates is a proxy you cannot audit.
  - log the teacher's accuracy on the extractor's own output each epoch. We own
    the ground-truth stems for every output, so calibration drift is directly
    observable, and it is the only warning available.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from src.models.losses import LossBSRNN
from src.models.state_teacher import NON_TARGET_AUDIBLE


class LossBSRNNState(LossBSRNN):
    """LossBSRNN plus `w_state * L_state`.

    At `w_state = 0.0` this is arithmetically the base class, but prefer
    constructing the base class directly in that case -- an exactly reproduced
    baseline should not depend on a multiplication by zero.
    """

    def __init__(self, *args, teacher=None, w_state=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if teacher is None:
            raise ValueError(
                "LossBSRNNState needs a frozen StateTeacher. Construct "
                "LossBSRNN instead if the state term is not wanted.")
        self.teacher = teacher
        self.w_state = w_state
        # Which windows of the output to score. None = all of them, which is
        # what a normal run uses.
        #
        # A SHORTER CONTIGUOUS SEGMENT IS THE ONLY SUBSAMPLING THAT SAVES
        # ANYTHING HERE. The term is a mean over windows, so a random subset
        # would be an unbiased estimate of it -- the mini-batch argument. But
        # the teacher's head is a BiLSTM over the window sequence, so a loss on
        # 4 windows still needs all 13 embeddings to feed the recurrence, and
        # the backward flows through it to all 13 regardless. Shortening the
        # SEQUENCE shortens both. The cost is that the head then sees less
        # context than the 13 windows it was fitted on, so
        # scripts/profile_state_teacher.py reports L_state at each length: a
        # setting that is cheap AND changes what the teacher says is not a
        # saving.
        self.window_starts = None
        # this is needed because the teacher needs to know who the target is, and the loss function is the only place that has access to the enrolment embedding. 
        self.enrolment_embedding = None

    def _loss_non_target_audible(self, s_output):
        """L_state. (B, T) -> (B,)

        Lower is better, 0 when the teacher hears no second voice in any
        window. Unbounded above, but bounded in practice by the sigmoid: a
        window the teacher is certain about contributes about 7 nats at p =
        0.999.

        Binary cross-entropy against a target of ZERO. On logits, not
        probabilities: `binary_cross_entropy_with_logits` fuses the sigmoid and
        the log, which is both numerically stable and the only correct way --
        sigmoid followed by plain BCE double-applies nothing but does overflow
        for confident windows. Essentially we want this value to ALWAYS be 0 against the teacher's output, so the label to aim for is 0. We should always not hear the non-target speaker. We are lucky that the teacher is very good at identifying between when the identity of the speaker is present and when it is not.
        """
        if self.enrolment_embedding is None:
            raise RuntimeError(
                "loss_fn.enrolment_embedding was never set. The teacher cannot "
                "tell who the target is without it, and a silently wrong "
                "enrolment would make this term reward extracting the wrong "
                "speaker. Set it each step, beside loss_fn.w.")


        logits = self.teacher(s_output, self.enrolment_embedding,
                              window_starts=self.window_starts)
        logits = logits[..., NON_TARGET_AUDIBLE]

        return F.binary_cross_entropy_with_logits(
            logits, torch.zeros_like(logits), reduction="none").mean(dim=-1)

    def __call__(self, s_target, s_output, x_input, crop_absent):
        """The M2 objective plus the state term.

            L = base + w_state * mean_all_crops[ L_state ]

        `base` is the parent's `(1 - w) * present + w * absent` split, unchanged.
        The state term sits OUTSIDE that split, and deliberately: "no second
        voice in the output" holds on every crop, present or absent, so
        splitting it by `crop_absent` would only shrink the number of examples
        it is averaged over. This method is used to add the loss terms together and return the total loss and a dictionary of the individual loss components.
        """
        total, parts = super().__call__(s_target, s_output, x_input, crop_absent)

        loss_state = self._loss_non_target_audible(s_output).mean()
        total = total + self.w_state * loss_state

        # Logged unweighted, so it is readable at w_state = 0 and so a break-even
        # weight can be derived from it the way scripts/derive_w_g.py derives wg.
        parts["L_state"] = float(loss_state.detach())
        parts["total"] = float(total.detach())
        return total, parts
