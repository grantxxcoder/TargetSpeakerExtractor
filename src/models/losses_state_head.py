"""The M2 objective plus head A's auxiliary state cross-entropy. D14 piece A.

A SUBCLASS in its own file, for the same reason `losses_state.py` is one:
`losses.py` is untouched, so every run predating this term reproduces by
construction rather than by a weight set to zero. `build_loss_fn()` returns the
base class unless the config asks for `w_state_head`, so an old config cannot
reach this code path at all.

WHAT THE TERM ADDS, IN PLAIN WORDS
-----------------------------------
A 516-parameter classifier reads the separator's own hidden features and is
asked, every 8 ms frame, who is audible in the MIXTURE: nobody, the target, the
other speaker, or both. It is scored against the free labels built from the
rendered stems. The classifier's answer is never used to change the audio --
the pressure is entirely indirect. To answer well the features must encode who
is talking, and the mask is built from those same features.

HOW IT DIFFERS FROM `L_state` (head B, losses_state.py). Both are about speaker
state and they are NOT the same term and must not be run together:

  head A   supervised, honest labels, reads INTERNAL features, ~8 ms frames,
           trains from scratch alongside the extractor, adds 516 params in
           training and zero at inference
  head B   a FROZEN teacher, reads the OUTPUT AUDIO, ~1 s windows, unsupervised
           in the sense that the required answer is always "no second voice",
           adds nothing anywhere

D14 says one variable per run: A alone, then B alone. Combining them gives one
number attributable to neither, which is the 2026-08-25 mistake.

WHY IT CANNOT COLLUDE
---------------------
Head B needs freezing because a jointly-trained scorer can satisfy a shared
objective by agreeing with the extractor while the audio does not change. Head A
has no such escape: its labels come from the rendered stems and are fixed, so
the only way to lower this term is to make the features genuinely predictive.
That is why head A can be trained jointly and head B cannot.

THE FAILURE MODE TO WATCH IS THE OPPOSITE ONE
---------------------------------------------
Not gaming -- shortcuts. A frame classifier is an easy place to hide a gender
cue (the extractor already leans on gender, 56.1 % vs 44.4 %, 2026-08-30) or to
degenerate into a voice-activity detector that never uses the enrolment at all.
Both of D14's controls therefore belong in the epoch log, not in a post-hoc
analysis: per-class recall split same- vs cross-gender, and per-class recall
with a stranger's enrolment. `state_head.per_class_recall` returns the
breakdown; pooled accuracy would conceal both, because `both` is only 16.9 % of
frames and a head that never predicts it still scores 83 %.

Also watch that this term does not buy its accuracy by deleting the target: the
same one-sidedness recorded for head B applies here in reverse -- nothing in a
classification loss keeps the target audible. Deletion rate and enrolment
sensitivity are the numbers that say whether the arm is working, not this term.
"""

from __future__ import annotations

import torch

from src.models.losses import LossBSRNN
from src.models.state_head import (
    inverse_frequency_weights,
    per_class_recall,
    state_cross_entropy,
)


class LossBSRNNStateHead(LossBSRNN):
    """LossBSRNN plus `w_state_head * L_head`.

    At `w_state_head = 0.0` this is arithmetically the base class, but prefer
    constructing the base class directly in that case -- an exactly reproduced
    baseline should not depend on a multiplication by zero.
    """

    def __init__(self, *args, w_state_head=0.0, class_counts=None,
                 ignore_index=-1, **kwargs):
        super().__init__(*args, **kwargs)
        self.w_state_head = w_state_head
        self.ignore_index = ignore_index
        # DERIVED FROM THE MEASURED HISTOGRAM, not chosen -- the same discipline
        # as w_g = 1.69 and w from the 0.297 absent rate. None means unweighted,
        # which is a deliberate ablation arm and NOT a sensible default: `both`
        # is the minority class at frame level, so an unweighted head predicts
        # `target` and `none` and scores well while being blind to the one state
        # a gate would care about.
        self.class_weight = (None if class_counts is None
                             else inverse_frequency_weights(class_counts))

        # Set per step by the training loop, beside loss_fn.w. Attributes rather
        # than call arguments because __call__'s signature is shared with the
        # base class and with losses_state.py, and changing it would touch every
        # caller of every objective. Same pattern, same reason.
        self.state_logits = None    # (B, 4, T) from model(..., return_state=True)
        self.state_labels = None    # (B, T)    int, from the loader

    def _loss_state_head(self):
        """L_head. -> scalar.

        Lower is better. With mean-1 class weights a head at chance sits at
        ln(4) = 1.386 nats, so the term is readable without a baseline: above
        ~1.39 the head has learnt nothing.
        """
        if self.state_logits is None or self.state_labels is None:
            raise RuntimeError(
                "loss_fn.state_logits / state_labels were never set. Call the "
                "model with return_state=True and assign both each step. A "
                "silently skipped term would train a baseline and be reported "
                "as head A having had no effect.")

        logits, labels = self.state_logits, self.state_labels
        # The head emits one logit per STFT frame and the loader builds one
        # label per STFT frame from the same hop, so a mismatch means the two
        # were computed with different framing -- a misalignment that would
        # silently cost accuracy and be blamed on the idea.
        assert logits.shape[0] == labels.shape[0] and logits.shape[-1] == labels.shape[-1], (
            f"state logits {tuple(logits.shape)} do not line up with labels "
            f"{tuple(labels.shape)}; expected (B, 4, T) against (B, T)")

        weight = (None if self.class_weight is None
                  else self.class_weight.to(logits.device))
        return state_cross_entropy(logits, labels, weight=weight,
                                   ignore_index=self.ignore_index)

    def __call__(self, s_target, s_output, x_input, crop_absent):
        """The M2 objective plus the state-head term.

            L = base + w_state_head * L_head

        `base` is the parent's `(1 - w) * present + w * absent` split, unchanged.
        The head term sits OUTSIDE that split deliberately: the question "who is
        audible in this frame" is well posed on every crop, and the crops where
        the target is absent are the informative ones for the `none` and
        `interferer` states. Splitting it by `crop_absent` would only shrink the
        number of examples it is averaged over.
        """
        total, parts = super().__call__(s_target, s_output, x_input, crop_absent)

        loss_head = self._loss_state_head()
        total = total + self.w_state_head * loss_head

        # Logged UNWEIGHTED so it is readable at w_state_head = 0 and so a
        # break-even weight can be derived from it the way derive_w_g.py does.
        parts["L_head"] = float(loss_head.detach())
        # Per-class recall, not pooled accuracy, and logged every step because
        # D14's two controls are read off exactly these four numbers. NaN where
        # a state has no frames in the batch, which is a gap in the curve rather
        # than a zero.
        recalls = per_class_recall(self.state_logits.detach(), self.state_labels,
                                   ignore_index=self.ignore_index)
        for state, value in enumerate(recalls.tolist()):
            parts[f"head_recall_{state}"] = value
        present = recalls[~torch.isnan(recalls)]
        parts["head_bal_acc"] = (float(present.mean()) if present.numel()
                                 else float("nan"))
        parts["total"] = float(total.detach())
        return total, parts
