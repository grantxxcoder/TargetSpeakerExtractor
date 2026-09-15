"""Per-frame speaker-state labels: who is audible in each frame of a trial.

Four states, and they are only well defined because a trial holds at most two
voices (decisions-m0.md 2026-08-14). A third talker would break the set.

    0  NONE         neither speaker audible (noise only, or silence)
    1  TARGET       the target alone
    2  INTERFERER   the other speaker alone
    3  BOTH         both at once

WHERE THE LABELS COME FROM, AND WHY NOTHING IS DETECTED HERE
------------------------------------------------------------
Nothing is measured from the rendered audio. The manifest already records which
utterances went into a trial (`target_utts`) and where each was placed
(`target_onsets_s`), and `data/index/vad_segments.csv` already records where the
speech is inside each utterance (B2, Silero 6.2.1 pinned). Shifting the second by
the first reconstructs the speech spans on the mixture timeline exactly.

That matters for consistency, not just for cost: `overlap_achieved` in the
manifest was computed from these very spans, as
`shared_seconds(target, interferer) / mixture_length_s`
(`scripts/build_manifest.py`). Re-detecting activity with an energy gate would
introduce a SECOND definition of "speaking", so a trial could sit in one B13
overlap bucket by the manifest and a different one by these labels. Composing
from the same source keeps one definition, and lets the reconstruction be checked
against four manifest columns instead of trusted.

THE ONE THING THE MANIFEST CANNOT GIVE: THE REVERBERATION TAIL
--------------------------------------------------------------
The spans above are DRY -- they describe the source utterances before
convolution with the room. A1 makes the reference the full reverberant target
("what the mic heard", decisions-m0.md 2026-08-13), so while a decay tail is
still ringing the correct output IS that tail, and the frame must count as
target-active. Labelling from dry activity alone would teach a gate to suppress
exactly the tail the reference contains.

`extend_tails` therefore lengthens every span by `tail_t60_mult * t60_s`, reusing
A5's existing convention (the renderer already pads the mixture tail by `t60_s`,
decisions-m0.md 2026-08-13) rather than inventing a level threshold. The
multiplier is a config knob and the class histogram is reported at 0.0 and 1.0 so
its effect is measured rather than assumed.

Note the effect is smaller than it first looks: the VAD index was built with
`min_silence_duration_ms: 250`, so gaps under 250 ms are already bridged and
counted as speech. The extension mostly moves trailing edges, not mid-sentence
pauses.

Pure interval and array arithmetic -- no audio, no torch, no I/O. Unit-tested
directly, the same split `src/data/vad.py` uses.
"""

from __future__ import annotations

import numpy as np

from src.data import vad

NONE, TARGET, INTERFERER, BOTH = 0, 1, 2, 3
STATE_NAMES = ("none", "target", "interferer", "both")
N_STATES = 4


def extend_tails(spans, tail_s):
    """Lengthen every span by `tail_s` seconds, then merge what now overlaps.

    Merging is required, not cosmetic: without it two spans widened into each
    other would be counted twice by `vad.total_speech`.
    """
    if tail_s <= 0.0:
        return vad.merge(spans)
    return vad.merge([(a, b + tail_s) for a, b in spans])


def clip_spans(spans, length_s):
    """Drop what falls outside [0, length_s] and truncate what straddles it.

    Tail extension can push a span past the end of the rendered audio, and a
    label beyond the last sample would silently mis-align every frame index
    derived from it.
    """
    out = []
    for a, b in spans:
        a, b = max(0.0, a), min(length_s, b)
        if b > a:
            out.append((a, b))
    return out


def build_spans(utts, onsets_s, segments_by_utt, t60_s, tail_t60_mult, length_s):
    """One speaker's audible spans on the rendered timeline.

    `utts` and `onsets_s` are the manifest's pipe-separated lists, already split.
    `segments_by_utt` maps an utterance id to its parsed VAD segments.
    An utterance with no detected speech maps to [] -- rare (1 in 137,876) but
    real, and it must not crash a rebuild.
    """
    per_utt = [segments_by_utt.get(u, []) for u in utts]
    spans = vad.spans_of(per_utt, onsets_s)
    spans = extend_tails(spans, tail_t60_mult * t60_s)
    return clip_spans(spans, length_s)


def spans_to_mask(spans, n_frames, hop_s, offset_s=0.0):
    """Boolean per-frame activity. (n_frames,) of bool.

    Frame `t` is taken to cover `[offset_s + t*hop_s, offset_s + (t+1)*hop_s)` --
    its own hop step, which tiles the timeline with no gaps and no overlap, so
    every instant belongs to exactly one frame. The alternative (the analysis
    WINDOW, 32 ms at n_fft 512) overlaps its neighbours and would smear each
    label by a frame in both directions.

    A frame counts as active if any part of it is covered. That is deliberate for
    a gating label: the cost of missing target speech (the judge never hears the
    word) is worse than the cost of a frame of margin.

    EPS EXISTS FOR A REAL BUG, not for tidiness. A boundary landing exactly on a
    frame edge does not divide exactly in binary: at offset_s = 1.00, hop 0.01
    and b = 1.02, `(b - offset_s) / hop_s` is 2.0000000000000018, and `ceil`
    then claims a third frame. Crop offsets are drawn per example by the data
    loader, so without the snap roughly every second crop would carry a label
    one frame wider than the audio it describes -- invisible in any total, and
    exactly the kind of drift that would show up later as the teacher and the
    extractor disagreeing about frame alignment. 1e-6 of a frame is far below
    any boundary the detector resolves and far above double-precision residue.
    """
    mask = np.zeros(n_frames, dtype=bool)
    if not spans or n_frames == 0:
        return mask
    eps = 1e-6
    for a, b in spans:
        lo = int(np.floor((a - offset_s) / hop_s + eps))
        hi = int(np.ceil((b - offset_s) / hop_s - eps))
        lo, hi = max(0, lo), min(n_frames, hi)
        if hi > lo:
            mask[lo:hi] = True
    return mask


def states_from_masks(target_active, interferer_active):
    """Two boolean masks -> the four-state code per frame. (n_frames,) of int8.

    TARGET is bit 0 and INTERFERER is bit 1, so BOTH falls out as 3 and the
    encoding is order-independent.
    """
    return (target_active.astype(np.int8)
            + 2 * interferer_active.astype(np.int8))


def states_at_frames(target_spans, interferer_spans, n_frames, hop_s,
                     offset_s=0.0):
    """The whole way from spans to state codes. (n_frames,) of int8."""
    return states_from_masks(
        spans_to_mask(target_spans, n_frames, hop_s, offset_s),
        spans_to_mask(interferer_spans, n_frames, hop_s, offset_s),
    )


def histogram(states):
    """Frame counts per state, as a length-4 array. Used to derive the class
    weights for the detector's cross-entropy (inverse frequency).

    MEASURED on sir0_train, 9,955 trials, 8 ms frames, tail 1.0x t60:
    target 32.4 %, none 30.7 %, interferer 20.0 %, **both only 16.9 %**.

    Note what that corrects. B9 renders ~50 % `both` TRIALS, so the intuition is
    that `both` dominates -- it does not. Mean overlap inside a `both` trial is
    0.279, so a `both` trial is mostly not both-speaking, and at frame level
    `both` is the MINORITY class. The two routes agree: 4930/9955 x 0.279 =
    13.8 % predicted against 13.6 % measured on the dry labels.

    Consequence for reading any accuracy figure: a detector that never predicts
    `both` still scores 83 % pooled. Report per-class, never pooled."""
    return np.bincount(np.asarray(states).ravel(), minlength=N_STATES)


# --- what the extractor's output is REQUIRED to look like -----------------

# The extraction task, written in the state domain. The label above says what IS
# happening; this says what SHOULD still be audible once the model has done its
# job. Used by the target-state loss (decisions-pending.md D14), never by the
# detector's own training, which is always supervised on the honest label.
#
#   target only  -> target only     nothing to remove
#   both         -> target only     the interferer must go
#   interferer   -> none            the answer is silence
#   none         -> none            the answer is silence
REQUIRED_OUTPUT_STATE = np.array([NONE, TARGET, NONE, TARGET], dtype=np.int8)


def required_output_states(states):
    """Map observed input states to the states a correct output would show."""
    return REQUIRED_OUTPUT_STATE[np.asarray(states)]
