"""Head A: an auxiliary per-frame speaker-state classifier on the separator.

decisions-pending.md D14, piece A. NOT the teacher -- `state_teacher.StateHead`
is head B's frozen readout over ECAPA embeddings and lives in a different file
on purpose. This one is 516 parameters bolted onto the extractor's own features
and trained from scratch alongside it.

WHAT IT IS FOR, IN PLAIN WORDS
------------------------------
The extractor currently has no internal notion of who is talking at any instant.
The only presence signal anywhere in the system is `crop_absent`, one bit per
4 s crop. Head A asks the separator's own features, every 8 ms frame, "who is
audible right now: nobody, the target, the other speaker, or both?" and
penalises wrong answers. It changes no audio. The point is the PRESSURE: to
answer, the features must carry the answer, and the mask is built from those
same features.

    (B, K, N, T) separator features
        -> mean over the K bands          (B, N, T)
        -> 1x1 conv, N -> 4               (B, 4, T)
        -> cross-entropy against the four-state label

516 parameters at N=128 (4 x 128 weights + 4 biases). It is deleted at
inference unless D14's piece C is built, so the shipped model is byte-identical
in architecture, parameter count and latency to the baseline. **This arm has no
capacity confound in the audio path.**

WHY THE LABEL IS THE HONEST STATE, NOT THE REQUIRED OUTPUT STATE
----------------------------------------------------------------
`state_labels.REQUIRED_OUTPUT_STATE` maps `both -> target only`: what a correct
OUTPUT should contain. That mapping belongs to head B, which scores the output
audio. Head A reads the separator's INTERNAL features, and those features must
represent what is actually in the mixture -- a model that has correctly noticed
"both speakers are here, and I am about to suppress one" should be rewarded for
noticing, not punished for it. Training head A on the required-output mapping
would ask the features to forget the interferer they need to represent in order
to remove it.

WHY MEAN-POOL OVER BANDS -- ANSWERED 2026-09-11, KEEP IT
--------------------------------------------------------
It is what D14 specifies, and the question of whether it throws away state
information is now measured rather than assumed.
`scripts/probe_state_features.py` fitted both probes on the frozen
`model_sir0_10000-e6.pt` over 100 `sir0_val` trials, trial-disjoint split:

    linear, mean-pooled          516 params    61.0 %   <- keep
    MLP, mean-pooled          17,028 params    62.0 %
    linear, band-resolved     16,388 params    59.0 %   <- WORSE
    chance                                     25.0 %

Dropping the pool COSTS 2 points, and 33x the parameters buys 1. The pooling
discards nothing this task can use, and the argument that capacity belongs on
the band axis -- that the pool hides which frequencies the second voice
occupies -- is false. 516 stays.

THE NUMBER HEAD A HAS TO BEAT
-----------------------------
Those same probes are the null hypothesis for this whole arm. They read state
off a separator that was NEVER trained to encode it, so they measure what the
features carry for free:

    balanced         61.0 %
    none             93.1 %      target only      72.0 %
    interferer only  39.6 %      both             39.4 %

**If head A trains and its own head lands near 61 / 39, the auxiliary loss
changed nothing and the arm is negative.** The pressure has to show up as
movement on `both` and `interferer only`, which are the two states D13's gate
would consume and the two that are currently near useless. `detach_features=True`
is the online version of the same control: it reads the features without
pushing on them, so the gap between a detached run and the arm is the auxiliary
loss's actual effect.

AND THE SHORTCUT IS REAL, NOT HYPOTHETICAL
------------------------------------------
The same probe split by gender: **53.4 % same-gender against 66.5 %
cross-gender**, a 13-point gap on 18,054 vs 42,126 frames. On the case that
matters -- two speakers of the same gender -- the features are close to useless
beyond detecting silence, and a large part of the pooled 61 % is the model
noticing a male and a female voice. Head A trained on this is trained to lean on
that harder. **Never report head A's accuracy pooled over gender.**

TWO CONTROLS, WITHOUT WHICH ANY ACCURACY FIGURE IS MEANINGLESS (D14)
--------------------------------------------------------------------
1. Ablate the enrolment. If accuracy holds with a stranger's cue, the head is a
   voice-activity detector and the "who" half is unearned.
2. Split accuracy same- vs cross-gender. The extractor already leans on gender
   (56.1 % vs 44.4 %, 2026-08-30) and a frame classifier is an easier place to
   hide it.
`per_class_recall` exists so both can be logged per epoch rather than
reconstructed afterwards.

AND REPORT PER CLASS, NEVER POOLED. Measured on sir0_train: `both` is only
16.9 % of frames (`state_labels.histogram`), so a head that never predicts
`both` still scores 83 % pooled while being useless for gating.

Provenance:
  the state set     frame-level {non-speech, target, non-target} conditioned on
                    a speaker embedding is personal VAD, Ding et al., ICASSP
                    2020; per-frame per-speaker activity from speaker profiles
                    is TS-VAD, Medennikov et al., Interspeech 2020.
                    BORROWED WITH A DIFFERENCE: there the per-frame state is the
                    SYSTEM OUTPUT and the whole point; here it is a training-only
                    auxiliary on a separator's hidden features, discarded before
                    deployment, and it has four states rather than three because
                    `both` is separated from `target`.
  the auxiliary     multi-task learning as a regulariser / representation
                    pressure: Caruana, "Multitask Learning", Machine Learning
                    1997.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.state_labels import N_STATES, STATE_NAMES  # noqa: F401  (re-export)

# Frame shares MEASURED on sir0_train, 9,955 trials, 8 ms frames, tail 1.0 x t60
# (src/data/state_labels.histogram). Order is the state code: none, target,
# interferer, both. Kept here as a DEFAULT ONLY -- prefer passing the histogram
# of the split actually being trained on, which build_state_labels.py writes.
MEASURED_FRAME_SHARE_SIR0_TRAIN = (0.307, 0.324, 0.200, 0.169)


def inverse_frequency_weights(counts, normalise=True):
    """Class weights for the cross-entropy. Length-4 float tensor.

    D14: "derive it, do not pick it". `both` is the MINORITY class at frame
    level (16.9 %), not the majority the trial-level 50 % `both` rate suggests,
    so an unweighted head will simply predict `target` and `none` and score
    well while being blind to the one state gating cares about.

    `normalise=True` rescales so the weights average 1. That is not cosmetic:
    without it the term's magnitude moves whenever the histogram moves, and any
    weight derived against it (the way w_g and w_state were derived) would stop
    meaning what it meant. With it, a head at chance costs about ln(4) = 1.386
    nats regardless of which split produced the histogram.

    A class with zero frames gets weight 0 rather than infinity. That happens on
    a small debug subset and must not become a NaN.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    total = counts.sum()
    weights = torch.where(counts > 0, total / counts.clamp(min=1.0),
                          torch.zeros_like(counts))
    if normalise:
        present = weights > 0
        if bool(present.any()):
            weights = weights / weights[present].mean()
    return weights.to(torch.float32)


class AuxStateHead(nn.Module):
    """Mean-pool over bands, then a 1x1 conv to four per-frame logits.

    516 parameters at feature_dim=128, and `n_parameters` asserts it rather than
    trusting the arithmetic -- D14 carried an underived "~10 k" for three days.

    IT CONSTRUCTS UNDER A FORKED RNG, AND THAT IS LOAD-BEARING. Measured
    2026-09-11 (D14): building extra modules before the first batch is drawn
    advances the global RNG, which changes the dataloader's shuffle, which
    changes WHICH trial `drop_last` discards -- the arm and its control then see
    different data in a different order, and the config header's claim of "same
    data, same seed" becomes false. `torch.random.fork_rng` makes this head's
    initialisation deterministic AND invisible to everything constructed after
    it, so turning head A on changes the model and nothing else.
    """

    def __init__(self, feature_dim=128, n_states=N_STATES, detach_features=False,
                 init_seed=20260911, isolate_rng=True):
        super().__init__()
        self.detach_features = detach_features
        # DIAGNOSTIC MODE, not the arm. With detach_features=True the head still
        # learns to read the features but no gradient flows back into the
        # separator, so it measures how decodable state ALREADY is, online,
        # without applying any pressure. That is the training-time version of
        # scripts/probe_state_features.py and it is the correct control for
        # "did the auxiliary loss change the features, or were they always like
        # this?". The arm itself runs with detach_features=False.
        if isolate_rng:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                self.classifier = nn.Conv1d(feature_dim, n_states, 1)
        else:
            self.classifier = nn.Conv1d(feature_dim, n_states, 1)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, z):
        """(B, K, N, T) separator features -> (B, n_states, T) logits.

        No softmax. Every consumer here (`state_cross_entropy`,
        `per_class_recall`) wants logits, and D14's piece C would want them too
        so the gate sees an unsaturated quantity.
        """
        assert z.dim() == 4, f"expected (B, K, N, T), got {tuple(z.shape)}"
        if self.detach_features:
            z = z.detach()
        return self.classifier(z.mean(dim=1))


def state_cross_entropy(logits, labels, weight=None, ignore_index=-1):
    """L_state_head. (B, C, T) logits + (B, T) int labels -> scalar.

    Lower is better. A head at chance with mean-1 weights sits at ln(4) = 1.386
    nats; 0 is a perfect, confident classifier.

    POOLED OVER THE WHOLE BATCH, not per example then averaged. The unit here is
    the frame, every crop carries the same number of them, and a per-example
    weighted mean would renormalise the class weights inside each crop --
    silently undoing the balancing on any crop that happens to contain one state.

    fp32 REGARDLESS OF AMP, matching train.py's rule that the model forward runs
    in half precision and the loss does not. A cross-entropy on fp16 logits
    saturates where the head is confident, which is exactly where the gradient
    should be small but not zero.

    `ignore_index` covers frames with no label. Nothing produces them today --
    crops are fixed length -- but a variable-length or padded batch later must
    not quietly train on padding.
    """
    return F.cross_entropy(logits.float(), labels.long(), weight=weight,
                           ignore_index=ignore_index, reduction="mean")


@torch.no_grad()
def per_class_recall(logits, labels, n_states=N_STATES, ignore_index=-1):
    """Recall for each state. (n_states,) float tensor, NaN for absent classes.

    The number to log, and the reason it is per class: `both` is 16.9 % of
    frames, so pooled accuracy hides a head that never predicts it. Balanced
    accuracy is the mean of the non-NaN entries -- computed by the caller, so
    the per-class breakdown reaches the log rather than only its average.
    """
    predicted = logits.argmax(dim=1)
    labels = labels.long()
    valid = labels != ignore_index
    out = logits.new_full((n_states,), float("nan"))
    for state in range(n_states):
        mask = valid & (labels == state)
        if bool(mask.any()):
            out[state] = (predicted[mask] == state).float().mean()
    return out


def drop_state_head(state_dict):
    """Strip the head's weights from a checkpoint. Returns a new dict.

    Head A is training-only: the deployed extractor never calls it. An eval
    script that builds the baseline architecture and loads a head-A checkpoint
    with `strict=True` would otherwise fail on unexpected keys, and the
    temptation would be to set `strict=False` -- which would also swallow a
    genuinely missing separator weight. Strip explicitly instead.
    """
    return {k: v for k, v in state_dict.items() if not k.startswith("state_head.")}
