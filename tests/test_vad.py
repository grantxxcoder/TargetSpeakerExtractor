"""Unit tests for src/data/vad.py (B2, PR1).

The model itself is not tested here -- Silero is a third-party detector used
unmodified, and its behaviour is characterised empirically in
experiments/results/2026-08-15-vad-impact/ instead.

What IS tested is everything the manifest rebuild will depend on: the interval
arithmetic, the cache round-trip, and the two interruption definitions. A bug in
any of those silently mislabels 21,000 trials, and nothing downstream would catch
it -- the manifests are not in git, so there is no diff to notice.
"""

import pytest
import yaml

from src.data import vad


# --- cache round-trip -----------------------------------------------------

def test_format_parse_round_trip():
    segs = [(0.12, 1.83), (2.05, 4.4), (5.0, 5.0001)]
    assert vad.parse_segments(vad.format_segments(segs)) == pytest.approx(
        [(0.12, 1.83), (2.05, 4.4), (5.0, 5.0001)])


def test_empty_segments_round_trip():
    """An utterance with no detected speech is rare but real. It must survive the
    cache as [] rather than crashing a 137k-row rebuild."""
    assert vad.format_segments([]) == ""
    assert vad.parse_segments("") == []


def test_separator_is_not_a_hyphen():
    """Utterance ids are hyphenated (`1272-128104-0000`), so a hyphen boundary
    separator would make a segment string ambiguous to read in the CSV."""
    assert vad.BOUND_SEP != "-"
    assert vad.BOUND_SEP not in vad.SEG_SEP


def test_format_precision_is_finer_than_a_sample():
    """Four decimals is 0.1 ms; a sample at 16 kHz is 62.5 us apart in the worst
    case, so rounding must not merge or reorder boundaries at trial scale."""
    segs = [(1.00005, 1.00015)]
    a, b = vad.parse_segments(vad.format_segments(segs))[0]
    assert b >= a


# --- interval arithmetic --------------------------------------------------

def test_total_speech_ignores_the_gaps():
    assert vad.total_speech([(0.0, 1.0), (2.0, 4.0)]) == pytest.approx(3.0)


def test_total_speech_of_silence_is_zero():
    assert vad.total_speech([]) == 0.0


def test_shift_moves_onto_the_mixture_timeline():
    assert vad.shift([(0.5, 1.5)], 10.0) == [(10.5, 11.5)]


def test_spans_of_concatenates_every_utterance():
    """The whole point: one speaker's utterances become MANY spans, not one
    rectangle per file."""
    per_utt = [[(0.3, 1.0), (1.4, 2.0)], [(0.2, 0.9)]]
    onsets = [0.0, 5.0]
    assert vad.spans_of(per_utt, onsets) == [(0.3, 1.0), (1.4, 2.0), (5.2, 5.9)]


def test_shared_seconds_counts_only_real_coincidence():
    a = [(0.0, 2.0), (5.0, 7.0)]
    b = [(1.0, 6.0)]
    assert vad.shared_seconds(a, b) == pytest.approx(1.0 + 1.0)


def test_shared_seconds_is_zero_when_they_take_turns():
    """The 2.8 % case from 2026-08-15: two files overlap on the timeline, but the
    shared interval is one speaker's trailing silence and the other's leading
    silence, so no genuine overlap exists."""
    target = [(0.0, 7.5)]          # file ran 0-8 s, stopped talking at 7.5
    interferer = [(7.8, 15.0)]     # file began at 7 s, first word at 7.8
    assert vad.shared_seconds(target, interferer) == 0.0


def test_shared_seconds_touching_intervals_do_not_overlap():
    assert vad.shared_seconds([(0.0, 1.0)], [(1.0, 2.0)]) == 0.0


def test_shared_seconds_is_symmetric():
    a, b = [(0.0, 3.0), (4.0, 5.0)], [(2.0, 4.5)]
    assert vad.shared_seconds(a, b) == vad.shared_seconds(b, a)


# --- the two interruption definitions (option A chosen 2026-08-15) --------

def test_onsets_first_only_takes_one_per_utterance():
    """Option A: one turn-start per utterance, corrected for leading silence.
    The minimal correction to the old file-onset definition."""
    per_utt = [[(0.3, 1.0), (1.4, 2.0)], [(0.2, 0.9), (1.1, 1.5)]]
    onsets = [0.0, 10.0]
    assert vad.onsets_of(per_utt, onsets, first_only=True) == [0.3, 10.2]


def test_onsets_all_takes_every_resumption():
    """Option B, rejected: counts a breath pause as a new turn. This is why the
    measured rate rose 0.570 -> 0.725 -- a wider definition, not new truth."""
    per_utt = [[(0.3, 1.0), (1.4, 2.0)], [(0.2, 0.9), (1.1, 1.5)]]
    onsets = [0.0, 10.0]
    assert vad.onsets_of(per_utt, onsets, first_only=False) == [0.3, 1.4, 10.2, 11.1]


def test_option_a_never_reports_more_onsets_than_option_b():
    per_utt = [[(0.1, 0.5), (0.7, 1.0)], [(0.0, 0.2)]]
    onsets = [0.0, 3.0]
    a = vad.onsets_of(per_utt, onsets, first_only=True)
    b = vad.onsets_of(per_utt, onsets, first_only=False)
    assert len(a) <= len(b)
    assert set(a) <= set(b)


def test_onsets_skip_utterances_with_no_speech():
    """A silent utterance contributes no turn-start. Without this guard it would
    index [0] on an empty list and abort the rebuild."""
    assert vad.onsets_of([[], [(0.5, 1.0)]], [0.0, 4.0], first_only=True) == [4.5]


def test_interrupted_requires_strictly_inside():
    """Beginning exactly as the target stops is turn-taking, not interruption.
    decisions.md 2026-08-14."""
    target = [(0.0, 5.0)]
    assert vad.is_interrupted(target, [5.0]) is False
    assert vad.is_interrupted(target, [4.999]) is True


def test_interrupted_ignores_onsets_inside_a_target_pause():
    """The correction that matters. The interferer starts at 3.0 s, which is
    inside the target's FILE but inside a pause in their speech, so under the old
    file-boundary test this counted as an interruption and it should not."""
    target_speech = [(0.0, 2.5), (3.5, 6.0)]   # pause 2.5-3.5
    assert vad.is_interrupted(target_speech, [3.0]) is False


def test_interrupted_is_false_when_the_target_is_silent():
    assert vad.is_interrupted([], [1.0, 2.0]) is False


# --- config validation ----------------------------------------------------

def good_config(**over):
    cfg = {"threshold": 0.5, "min_silence_duration_ms": 250,
           "min_speech_duration_ms": 100, "speech_pad_ms": 30}
    cfg.update(over)
    return {"vad": cfg}


def test_vad_config_returns_the_four_knobs():
    assert vad.vad_config(good_config()) == {
        "threshold": 0.5, "min_silence_duration_ms": 250,
        "min_speech_duration_ms": 100, "speech_pad_ms": 30}


def test_missing_vad_block_is_an_error():
    with pytest.raises(KeyError, match="no `vad:` block"):
        vad.vad_config({})


def test_partial_vad_block_is_an_error():
    """No silent fallback to Silero's defaults: the definition of overlap must
    never change without appearing in a diff."""
    with pytest.raises(KeyError, match="missing"):
        vad.vad_config({"vad": {"threshold": 0.5}})


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_threshold_out_of_range_is_an_error(bad):
    with pytest.raises(ValueError, match="threshold"):
        vad.vad_config(good_config(threshold=bad))


def test_negative_duration_is_an_error():
    with pytest.raises(ValueError, match="min_silence_duration_ms"):
        vad.vad_config(good_config(min_silence_duration_ms=-1))


def test_wrong_model_version_is_an_error():
    with pytest.raises(RuntimeError, match="expected_model_version"):
        vad.vad_config(good_config(expected_model_version="0.0.1-not-installed"))


# --- the shipped config ---------------------------------------------------

def test_generator_yaml_vad_block_is_valid():
    """The config that actually builds the index. If this fails, every manifest
    built after it is measured against settings nobody reviewed."""
    config = yaml.safe_load(
        open("experiments/configs/generator.yaml").read())
    cfg = vad.vad_config(config)
    assert cfg["min_silence_duration_ms"] == 250, \
        "250 ms is the value decisions.md 2026-08-15 records and justifies"


def test_build_manifest_keeps_no_private_copies():
    """PR1 duplicated shared_seconds and is_interrupted while build_manifest.py
    was off limits, and this test kept the two copies identical. PR2 deleted the
    copies, so it now guards the other direction: a re-introduced local version
    would be one that measures file boundaries instead of speech, which is the
    entire bug B2 exists to remove and would leave no trace in the manifest.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "bm", "scripts/build_manifest.py")
    bm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bm)
    assert not hasattr(bm, "shared_seconds"), "use vad.shared_seconds"
    assert not hasattr(bm, "is_interrupted"), "use vad.is_interrupted"
    assert not hasattr(bm, "spans"), "use vad.spans_of"
