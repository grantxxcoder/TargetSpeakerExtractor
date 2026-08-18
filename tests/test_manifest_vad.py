"""Unit tests for B2 PR2 -- the footprint/speech split in build_manifest.py.

decisions-m0.md 2026-08-15. The single defect this PR can introduce is using one
quantity where the other belongs, so every test here pins which one a function
is supposed to be reading:

    FOOTPRINT  timeline occupied by audio, silence included -- placement
    SPEECH     detected voice inside that audio        -- measurement

The fixtures use utterances that are deliberately only half speech, so the two
quantities can never be confused by coincidence.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manifest import (  # noqa: E402
    best_onset, block_spans, drop_silent, lay_out, load_vad_index, pick_run,
)
from src.data import vad  # noqa: E402


def utt(name, duration):
    return {"utt": name, "speaker": "1", "chapter": "c", "duration": f"{duration}"}


# Four 4 s utterances, each holding exactly 2 s of speech in the middle.
# speech/footprint = 0.5, so a bug reading the wrong one is a factor-of-two miss.
RUN = [utt(f"u{i}", 4.0) for i in range(4)]
SEGS = {f"u{i}": [(1.0, 3.0)] for i in range(4)}


# --- pick_run: selects on SPEECH, capped on FOOTPRINT ----------------------

def test_pick_run_accumulates_speech_not_duration():
    """4 s of speech needs 2 utterances here, but 4 would fit by duration."""
    got = pick_run(np.random.default_rng(0), [("1", "c")], {("1", "c"): RUN},
                   wanted_s=4.0, cap_s=5.0, max_n=6, segs_of=SEGS,
                   footprint_cap_s=100.0)
    chapter, run, footprint, speech = got
    assert speech == pytest.approx(4.0)
    assert footprint == pytest.approx(8.0)   # 2 files x 4 s
    assert len(run) == 2


def test_pick_run_respects_the_footprint_cap():
    """The cap that stops lay_out being handed negative leftover time.

    6 s of speech would need 3 utterances = 12 s of audio, but only 9 s of
    window is on offer, so the run cannot be built and the caller must redraw.
    """
    got = pick_run(np.random.default_rng(0), [("1", "c")], {("1", "c"): RUN},
                   wanted_s=6.0, cap_s=7.0, max_n=6, segs_of=SEGS,
                   footprint_cap_s=9.0)
    assert got is None


def test_pick_run_footprint_always_fits_the_window():
    """The property the cap exists for, over many draws."""
    for seed in range(50):
        got = pick_run(np.random.default_rng(seed), [("1", "c")],
                       {("1", "c"): RUN}, wanted_s=2.0, cap_s=6.0, max_n=6,
                       segs_of=SEGS, footprint_cap_s=10.0)
        if got is not None:
            assert got[2] <= 10.0


def test_pick_run_returns_none_when_speech_is_unreachable():
    """20 s of speech does not exist in 4 half-silent utterances."""
    got = pick_run(np.random.default_rng(0), [("1", "c")], {("1", "c"): RUN},
                   wanted_s=20.0, cap_s=25.0, max_n=6, segs_of=SEGS,
                   footprint_cap_s=1000.0)
    assert got is None


# --- lay_out stays footprint-based ----------------------------------------

def test_lay_out_never_overlaps_the_audio():
    """Placing by speech would let files collide in their silent parts."""
    onsets = lay_out(np.random.default_rng(1), RUN[:2], span_s=20.0)
    assert onsets[1] >= onsets[0] + 4.0 - 1e-9
    assert onsets[-1] + 4.0 <= 20.0 + 1e-9


# --- block_spans: the interferer's internal geometry ----------------------

def test_block_spans_are_speech_not_rectangles():
    spans = block_spans(RUN[:2], SEGS)
    # Second utterance starts at 4.0, so its speech is 5.0-7.0, not 4.0-8.0.
    assert spans == [(1.0, 3.0), (5.0, 7.0)]
    assert vad.total_speech(spans) == pytest.approx(4.0)


# --- best_onset: overlap measured through the gaps ------------------------

def test_best_onset_counts_only_speech_overlap():
    """A solid-rectangle interferer would score 4 s here; a real one scores 2."""
    target_spans = [(0.0, 10.0)]          # target talks throughout
    block = [(1.0, 3.0), (5.0, 7.0)]      # 8 s footprint, 4 s speech
    onset, shared = best_onset(target_spans, block, footprint_s=8.0,
                               length=10.0, wanted_s=4.0)
    assert shared == pytest.approx(4.0)   # both segments land inside
    assert 0.0 <= onset <= 2.0


def test_best_onset_can_place_a_gap_over_the_target():
    """The interferer's own pause is not talking, so it can hide a target span."""
    target_spans = [(4.0, 6.0)]           # target speaks only in the middle
    block = [(0.0, 2.0), (6.0, 8.0)]      # 8 s footprint, silent 2-6
    onset, shared = best_onset(target_spans, block, footprint_s=8.0,
                               length=12.0, wanted_s=0.0)
    assert shared == pytest.approx(0.0, abs=1e-6)


def test_best_onset_never_slides_audio_past_the_window():
    block = [(0.5, 1.5)]
    for wanted in (0.0, 1.0, 5.0):
        onset, _ = best_onset([(0.0, 3.0)], block, footprint_s=4.0,
                              length=10.0, wanted_s=wanted)
        assert 0.0 <= onset <= 6.0 + 1e-9


# --- index loading and its guards -----------------------------------------

def write_index(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["utt", "duration", "speech_s", "segments"])
        w.writeheader()
        for name, segs in rows:
            w.writerow({"utt": name, "duration": "4.0",
                        "speech_s": vad.total_speech(segs),
                        "segments": vad.format_segments(segs)})


def test_load_vad_index_round_trips(tmp_path):
    path = tmp_path / "v.csv"
    write_index(path, [("u0", [(1.0, 3.0)]), ("u1", [(0.5, 1.0), (2.0, 3.5)])])
    got = load_vad_index(path, [utt("u0", 4.0), utt("u1", 4.0)])
    assert got["u1"] == [(0.5, 1.0), (2.0, 3.5)]


def test_load_vad_index_ignores_utterances_from_other_splits(tmp_path):
    path = tmp_path / "v.csv"
    write_index(path, [("u0", [(1.0, 3.0)]), ("other", [(0.0, 1.0)])])
    got = load_vad_index(path, [utt("u0", 4.0)])
    assert set(got) == {"u0"}


def test_missing_index_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="build_vad_index"):
        load_vad_index(tmp_path / "absent.csv", [utt("u0", 4.0)])


def test_stale_index_is_fatal(tmp_path):
    """An unindexed utterance must stop the build, not fall back to durations."""
    path = tmp_path / "v.csv"
    write_index(path, [("u0", [(1.0, 3.0)])])
    with pytest.raises(SystemExit, match="stale"):
        load_vad_index(path, [utt("u0", 4.0), utt("u1", 4.0)])


def test_drop_silent_removes_only_speechless_utterances():
    kept, n = drop_silent([utt("u0", 4.0), utt("q", 4.0)],
                          {"u0": [(1.0, 3.0)], "q": []})
    assert [u["utt"] for u in kept] == ["u0"]
    assert n == 1
