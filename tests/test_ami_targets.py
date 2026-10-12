"""Headset -> training target (src/data/ami_targets.py) and the range fetcher's
header parsing (src/data/ami_fetch.py). decisions-m8.md 2026-10-12.

Synthetic signals with a planted answer: a known delay, a known gain, known
speech intervals. Each test pins one property the M8 target depends on.
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.ami_fetch import parse_wav_header  # noqa: E402
from src.data.ami_targets import (activity_mask, estimate_lag, faded,  # noqa: E402
                                  gcc_phat_lag, process_headset)

SR = 16000
CFG = {"align": {"max_lag_ms": 50, "min_solo_s": 1.0, "min_peak": 0.2}, "rescale": "least_squares",
       "gate": {"pad_s": 0.2, "merge_gap_s": 0.3, "fade_ms": 10}}


def speech(n, seed):
    return np.random.default_rng(seed).standard_normal(n).astype(np.float32)


def test_gcc_phat_finds_a_planted_delay_in_both_directions():
    x = speech(SR * 2, 0)
    for lag in (37, -12, 0):
        delayed = np.roll(x, lag)                     # reference hears it `lag` later
        assert gcc_phat_lag(delayed, x, 800)[0] == lag


def test_process_headset_undoes_delay_and_level_and_gates_outside_speech():
    n, margin, lag, gain = SR * 6, 1600, 64, 0.05     # 4 ms late, -26 dB
    voice = speech(n + 2 * margin, 1)
    own = [(0.5, 2.5), (3.5, 5.0)]                    # seconds, window-relative
    keep_ext = np.concatenate([np.zeros(margin, bool),
                               activity_mask(own, 0.0, n), np.zeros(margin, bool)])
    voice *= keep_ext                                 # the speaker talks only inside `own`
    headset_ext = voice + 0.001 * speech(n + 2 * margin, 2)
    # Table mic: the same voice, `lag` samples later, at `gain`.
    mixture = gain * voice[margin - lag:margin - lag + n]
    est, peak = estimate_lag(mixture, headset_ext, margin, own, [], CFG)
    assert est == lag and peak > 0.2
    out, info = process_headset(mixture, headset_ext, margin, own, [], CFG, est)

    assert info["lag_ms"] == pytest.approx(1000 * lag / SR)
    assert info["gain_db"] == pytest.approx(20 * np.log10(gain), abs=0.1)
    inside = activity_mask(own, 0.0, n, pad_s=0.0)
    # Inside speech (away from the fades) the target matches the table mic.
    core = inside & ~activity_mask([(a, a + 0.05) for a, _ in own] + [(b - 0.05, b) for _, b in own], 0.0, n)
    assert np.corrcoef(out[core], mixture[core])[0, 1] > 0.99
    # Outside the padded speech the target is EXACTLY zero: the gate, not quiet.
    kept = activity_mask(own, 0.0, n, pad_s=0.2, merge_gap_s=0.3)
    assert np.all(out[~kept] == 0.0)
    assert info["gated_frac"] == pytest.approx(1 - kept.mean(), abs=1e-3)


def test_bleed_while_only_the_other_person_talks_is_removed():
    n, margin = SR * 4, 1600
    own, other = [(0.2, 1.5)], [(2.2, 3.8)]
    me = speech(n + 2 * margin, 3) * np.concatenate(
        [np.zeros(margin, bool), activity_mask(own, 0.0, n), np.zeros(margin, bool)])
    them = speech(n + 2 * margin, 4) * np.concatenate(
        [np.zeros(margin, bool), activity_mask(other, 0.0, n), np.zeros(margin, bool)])
    headset_ext = me + 0.09 * them                    # ~21 dB bleed, as on AMI
    mixture = 0.05 * (me + them)[margin:margin + n]
    lag, _ = estimate_lag(mixture, headset_ext, margin, own, other, CFG)
    out, _ = process_headset(mixture, headset_ext, margin, own, other, CFG, lag)
    assert np.all(out[activity_mask(other, 0.0, n)] == 0.0)


def test_fades_are_inside_kept_spans_and_short_spans_still_fade():
    mask = np.zeros(1000, bool)
    mask[100:600] = True
    mask[800:810] = True                              # shorter than two fades
    g = faded(mask, 160)
    assert np.all(g[~mask] == 0.0)
    assert g[100] < 0.01 and g[350] == 1.0 and g[599] < 0.01
    assert 0 < g[804] <= 1.0 and g[800] < g[804]


def test_activity_mask_pads_and_merges():
    m = activity_mask([(1.0, 1.2), (1.4, 1.6)], 0.0, SR * 2, pad_s=0.0, merge_gap_s=0.3)
    assert m[int(1.3 * SR)]                           # 0.2 s gap < 0.3 s: kept
    m = activity_mask([(1.0, 1.2)], 0.0, SR * 2, pad_s=0.2)
    assert m[int(0.85 * SR)] and not m[int(0.75 * SR)]


def _header(channels, rate=SR, bits=16, fmt_tag=1):
    fmt = struct.pack("<HHIIHH", fmt_tag, channels, rate, rate * channels * bits // 8,
                      channels * bits // 8, bits)
    return (b"RIFF" + struct.pack("<I", 0) + b"WAVE" + b"fmt " + struct.pack("<I", 16) + fmt
            + b"data" + struct.pack("<I", 1000))


def test_header_parser_accepts_mono_and_stereo_and_reports_channels():
    assert parse_wav_header(_header(1)) == (44, 1000, 1)
    assert parse_wav_header(_header(2)) == (44, 1000, 2)


@pytest.mark.parametrize("bad", [dict(channels=1, rate=8000), dict(channels=1, bits=24),
                                 dict(channels=4), dict(channels=1, fmt_tag=3)])
def test_header_parser_refuses_any_other_format(bad):
    with pytest.raises(ValueError):
        parse_wav_header(_header(**bad))
