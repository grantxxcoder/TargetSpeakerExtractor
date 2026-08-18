"""Unit tests for reject_speech_clips() in scripts/build_manifest.py.

B2 `noise_speech_rejection`, decisions-m0.md 2026-08-16.

The guards matter as much as the filter here. A missing or stale screening index
must stop the build, because the whole point of the parameter is that a bed with
an unlabelled talker in it never reaches a mixture -- and silently skipping the
filter looks identical to passing it.
"""

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_manifest import reject_speech_clips  # noqa: E402


def write_screen(path, rows):
    """Minimal stand-in for screen_noise_speech.py's output."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["clip", "duration", "speech_s",
                                          "max_segment_s", "n_segments", "segments"])
        w.writeheader()
        for clip, longest in rows:
            w.writerow({"clip": clip, "duration": "10.0", "speech_s": longest,
                        "max_segment_s": longest, "n_segments": 1, "segments": ""})


def pool(*names):
    return [{"clip": n, "duration": "10.0"} for n in names]


def test_drops_at_and_above_threshold(tmp_path):
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 0.0), ("b.flac", 0.49),
                          ("c.flac", 0.5), ("d.flac", 3.0)])
    kept = reject_speech_clips(pool("a.flac", "b.flac", "c.flac", "d.flac"),
                               screen, 0.5)
    # 0.5 is rejected: the cutoff is "reaches", not "exceeds".
    assert [c["clip"] for c in kept] == ["a.flac", "b.flac"]


def test_threshold_comes_from_the_caller(tmp_path):
    """Nothing about 0.5 is baked in -- it is a config value."""
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 0.3), ("b.flac", 0.8)])
    clips = pool("a.flac", "b.flac")
    assert len(reject_speech_clips(clips, screen, 1.0)) == 2
    assert [c["clip"] for c in reject_speech_clips(clips, screen, 0.5)] == ["a.flac"]


def test_clean_pool_is_untouched(tmp_path):
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 0.0), ("b.flac", 0.0)])
    kept = reject_speech_clips(pool("a.flac", "b.flac"), screen, 0.5)
    assert len(kept) == 2


def test_missing_index_is_fatal(tmp_path):
    with pytest.raises(SystemExit, match="screen_noise_speech"):
        reject_speech_clips(pool("a.flac"), tmp_path / "absent.csv", 0.5)


def test_stale_index_is_fatal(tmp_path):
    """A clip in the pool that was never screened must not pass unchecked."""
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 0.0)])
    with pytest.raises(SystemExit, match="stale"):
        reject_speech_clips(pool("a.flac", "unscreened.flac"), screen, 0.5)


def test_rejecting_everything_is_fatal(tmp_path):
    """An empty pool would fail far away, inside trial sampling. Fail here."""
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 2.0), ("b.flac", 2.0)])
    with pytest.raises(SystemExit, match="every clip rejected"):
        reject_speech_clips(pool("a.flac", "b.flac"), screen, 0.5)


def test_rows_are_passed_through_unchanged(tmp_path):
    """The filter selects; it must not rewrite the index rows it keeps."""
    screen = tmp_path / "s.csv"
    write_screen(screen, [("a.flac", 0.0)])
    original = pool("a.flac")
    kept = reject_speech_clips(original, screen, 0.5)
    assert kept[0] is original[0]
