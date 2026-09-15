"""Unit tests for src/data/state_labels.py and vad.merge (D14, PR1).

The label BUILD is verified against reality elsewhere and more strongly:
scripts/build_state_labels.py rebuilds each trial's spans and checks them
against four manifest columns that build_manifest.py derived from the same
source, so the script is tested against the code that actually built the
dataset. Nothing here can do that.

What IS tested here is the arithmetic that check cannot reach: the tail
extension (build_manifest.py had none, so no manifest column constrains it),
the clipping, and the rasterising from spans to per-frame state codes. A bug in
the rasteriser would misalign every label by a frame or two without changing any
total, so no aggregate check would notice.
"""

import numpy as np
import pytest

from src.data import state_labels as sl
from src.data import vad


# --- vad.merge ------------------------------------------------------------

def test_merge_empty():
    assert vad.merge([]) == []


def test_merge_leaves_disjoint_alone():
    assert vad.merge([(0.0, 1.0), (2.0, 3.0)]) == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_joins_overlapping():
    assert vad.merge([(0.0, 1.5), (1.0, 2.0)]) == [(0.0, 2.0)]


def test_merge_joins_touching():
    """Touching counts as contiguous: a tail that ends exactly where the next
    utterance starts leaves no silent instant between them."""
    assert vad.merge([(0.0, 1.0), (1.0, 2.0)]) == [(0.0, 2.0)]


def test_merge_sorts_and_absorbs_nested():
    assert vad.merge([(2.0, 3.0), (0.0, 5.0), (1.0, 1.5)]) == [(0.0, 5.0)]


def test_merge_keeps_total_speech_honest():
    """The reason merge exists. Widened spans that run into each other would be
    double-counted by total_speech, inflating every activity figure."""
    widened = [(0.0, 1.2), (1.0, 2.0)]
    assert vad.total_speech(widened) == pytest.approx(2.2)      # wrong
    assert vad.total_speech(vad.merge(widened)) == pytest.approx(2.0)


# --- tail extension -------------------------------------------------------

def test_extend_tails_zero_still_merges():
    """tail_t60_mult = 0.0 is the dry ablation arm. It must still return merged,
    sorted spans so the two arms differ only in the tail."""
    assert sl.extend_tails([(1.0, 2.0), (0.0, 1.0)], 0.0) == [(0.0, 2.0)]


def test_extend_tails_lengthens_the_end_only():
    """A reverberation tail follows the speech; it does not precede it."""
    assert sl.extend_tails([(1.0, 2.0)], 0.5) == [(1.0, 2.5)]


def test_extend_tails_bridges_a_short_gap():
    assert sl.extend_tails([(0.0, 1.0), (1.2, 2.0)], 0.3) == [(0.0, 2.3)]


def test_extend_tails_leaves_a_long_gap_alone():
    assert sl.extend_tails([(0.0, 1.0), (2.0, 3.0)], 0.3) == [(0.0, 1.3), (2.0, 3.3)]


# --- clipping -------------------------------------------------------------

def test_clip_truncates_a_tail_past_the_end():
    """Tail extension can push a span past the last rendered sample. A label
    beyond the audio would misalign every frame index derived from it."""
    assert sl.clip_spans([(1.0, 5.0)], 4.0) == [(1.0, 4.0)]


def test_clip_drops_a_span_entirely_outside():
    assert sl.clip_spans([(5.0, 6.0)], 4.0) == []


def test_clip_drops_a_zero_length_result():
    assert sl.clip_spans([(4.0, 4.5)], 4.0) == []


# --- rasterising ----------------------------------------------------------

def test_mask_tiles_frames_by_hop():
    """Frame t covers [t*hop, (t+1)*hop). One 0.02 s span at 0.01 hop touches
    frames 0 and 1 and nothing else."""
    m = sl.spans_to_mask([(0.0, 0.02)], n_frames=4, hop_s=0.01)
    assert m.tolist() == [True, True, False, False]


def test_mask_counts_partial_coverage_as_active():
    """Deliberate: for a gating label, missing target speech costs the judge a
    word, so a frame of margin is the cheaper error."""
    m = sl.spans_to_mask([(0.015, 0.016)], n_frames=4, hop_s=0.01)
    assert m.tolist() == [False, True, False, False]


def test_mask_respects_a_crop_offset():
    m = sl.spans_to_mask([(1.00, 1.02)], n_frames=4, hop_s=0.01, offset_s=1.00)
    assert m.tolist() == [True, True, False, False]


def test_mask_is_empty_with_no_spans():
    assert not sl.spans_to_mask([], n_frames=5, hop_s=0.01).any()


def test_mask_never_runs_past_the_frame_count():
    m = sl.spans_to_mask([(0.0, 10.0)], n_frames=3, hop_s=0.01)
    assert m.tolist() == [True, True, True]


# --- the four states ------------------------------------------------------

def test_state_codes():
    t = np.array([False, True, False, True])
    i = np.array([False, False, True, True])
    assert sl.states_from_masks(t, i).tolist() == [
        sl.NONE, sl.TARGET, sl.INTERFERER, sl.BOTH]


def test_state_encoding_is_a_bitfield():
    """TARGET is bit 0 and INTERFERER is bit 1, so BOTH falls out as 3 and the
    encoding does not depend on which speaker is considered first."""
    assert (sl.TARGET, sl.INTERFERER, sl.BOTH) == (1, 2, 3)


def test_states_at_frames_end_to_end():
    states = sl.states_at_frames(
        target_spans=[(0.0, 0.02)], interferer_spans=[(0.01, 0.03)],
        n_frames=4, hop_s=0.01)
    assert states.tolist() == [sl.TARGET, sl.BOTH, sl.INTERFERER, sl.NONE]


def test_histogram_has_a_bin_per_state_even_when_unused():
    """Length 4 always: an absent-target split can contain no BOTH frames at
    all, and a shorter histogram would silently break the class weights."""
    h = sl.histogram(np.array([sl.NONE, sl.NONE, sl.TARGET]))
    assert h.tolist() == [2, 1, 0, 0]


# --- what a correct output must look like ---------------------------------

def test_required_output_state_mapping():
    """The extraction task in the state domain: the interferer must go, and
    target-absent frames must go quiet."""
    observed = np.array([sl.NONE, sl.TARGET, sl.INTERFERER, sl.BOTH])
    assert sl.required_output_states(observed).tolist() == [
        sl.NONE, sl.TARGET, sl.NONE, sl.TARGET]


def test_required_output_never_asks_for_the_interferer():
    """No input state may require INTERFERER or BOTH in the output. If it could,
    the loss would be rewarding leakage."""
    every = np.arange(sl.N_STATES)
    assert set(sl.required_output_states(every).tolist()) <= {sl.NONE, sl.TARGET}
