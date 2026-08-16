"""Unit tests for src/data/render.py.

The renderer's failure mode is silence: wrong audio still plays, still has the
right length, and still passes every check that does not measure it. So each test
pins one decision from decisions.md to a property that would break if the code
drifted.

Nothing here reads the corpora. Signals are synthetic so the expected answer is
known exactly rather than approximately.
"""

import sys
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import render  # noqa: E402

SR = 16000


def tone(seconds, freq=220.0, amp=0.2, sr=SR):
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t)


@pytest.fixture
def meter():
    return pyln.Meter(SR)


# --- A2: the noise bed wraps ----------------------------------------------

def test_wrap_noise_loops_a_short_clip(tmp_path):
    """A naive slice would zero-pad here. The 2026-08-11 entry calls that out
    explicitly: WHAM! clips are shorter than the mixtures, so wrapping is
    mandatory rather than an optimisation."""
    clip = np.arange(1000, dtype=np.float64) / 1000.0
    path = tmp_path / "n.wav"
    sf.write(path, clip, SR)

    got = render.wrap_noise(path, offset_s=0.0, n_samples=2500, sr=SR)
    assert len(got) == 2500
    assert np.abs(got).min() > 0 or got[0] == 0.0     # never silent padding
    np.testing.assert_allclose(got[:1000], clip, atol=1e-4)
    np.testing.assert_allclose(got[1000:2000], clip, atol=1e-4)


def test_wrap_noise_starts_at_the_offset(tmp_path):
    clip = np.arange(1000, dtype=np.float64) / 1000.0
    path = tmp_path / "n.wav"
    sf.write(path, clip, SR)
    offset_samples = 400
    got = render.wrap_noise(path, offset_s=offset_samples / SR,
                            n_samples=300, sr=SR)
    np.testing.assert_allclose(got, clip[400:700], atol=1e-4)


def test_wrap_noise_offset_past_the_end_wraps_round(tmp_path):
    """`noise_offset_s` is a phase, not an index, so an offset beyond the clip
    is legal and must wrap rather than read nothing."""
    clip = np.arange(1000, dtype=np.float64) / 1000.0
    path = tmp_path / "n.wav"
    sf.write(path, clip, SR)
    got = render.wrap_noise(path, offset_s=1200 / SR, n_samples=100, sr=SR)
    np.testing.assert_allclose(got, clip[200:300], atol=1e-4)


# --- A3: levels are BS.1770, and gains are exact --------------------------

def test_gain_to_hits_the_requested_loudness(meter):
    x = tone(3.0)
    g = render.gain_to(x, -25.0, meter)
    assert meter.integrated_loudness(x * g) == pytest.approx(-25.0, abs=0.01)


def test_gain_to_rejects_a_silent_stem(meter):
    """-inf is only reachable through a bug under the 2026-08-11 anchor rule,
    so it must raise rather than produce inf gain."""
    with pytest.raises(ValueError, match="silent stem"):
        render.gain_to(np.zeros(SR * 3), -25.0, meter)


def test_loudness_rejects_a_stem_below_the_block_size(meter):
    with pytest.raises(ValueError, match="BS.1770"):
        render.loudness(tone(0.2), meter)


def test_loudness_is_gated_so_silence_does_not_change_it(meter):
    """The whole reason A3 chose BS.1770 over RMS. Padding speech with silence
    must barely move loudness; under RMS it moved 7.5 dB."""
    speech = tone(3.0)
    padded = np.concatenate([speech, np.zeros(14 * SR)])
    assert abs(render.loudness(speech, meter)
               - render.loudness(padded, meter)) < 1.0


# --- placement -------------------------------------------------------------

def test_lay_track_places_audio_at_its_onset(tmp_path):
    a = tone(0.5, amp=0.3)
    path = tmp_path / "a.wav"
    sf.write(path, a, SR)
    track = render.lay_track([path], [1.0], n_samples=SR * 3, sr=SR)
    assert np.abs(track[:SR]).max() == 0.0
    assert np.abs(track[SR:SR + len(a)]).max() > 0.2
    assert np.abs(track[SR + len(a):]).max() == 0.0


def test_lay_track_refuses_audio_past_the_window(tmp_path):
    """build_manifest's footprint cap should make this unreachable; if it ever
    happens the render must stop rather than silently truncate a word."""
    path = tmp_path / "a.wav"
    sf.write(path, tone(2.0), SR)
    with pytest.raises(ValueError, match="past the window"):
        render.lay_track([path], [1.5], n_samples=SR * 3, sr=SR)


def test_lay_track_rejects_a_sample_rate_mismatch(tmp_path):
    path = tmp_path / "a.wav"
    sf.write(path, tone(0.5), 8000)
    with pytest.raises(ValueError, match="expected"):
        render.lay_track([path], [0.0], n_samples=SR * 3, sr=SR)


# --- A5: the tail survives -------------------------------------------------

def test_convolve_keeps_the_reverb_tail_inside_the_pad():
    """A1 puts the tail in the reference, so A5's pad has to be long enough to
    hold it. Speech ending at 1 s with a 0.5 s tail must still be decaying at
    1.2 s and must not be cut at the window edge."""
    n = int(1.5 * SR)
    dry = np.zeros(n)
    dry[:SR] = tone(1.0)
    rir = np.exp(-np.arange(int(0.5 * SR)) / (0.1 * SR))
    wet = render.convolve_to(dry, rir, n)
    assert len(wet) == n
    assert np.abs(wet[int(1.2 * SR):int(1.3 * SR)]).max() > 0


def test_convolve_trims_to_the_window():
    dry = tone(1.0)
    rir = np.ones(1000)
    assert len(render.convolve_to(dry, rir, 500)) == 500


# --- enrollment EQ ---------------------------------------------------------

def test_eq_preserves_rms():
    """`enrollment_eq_augmentation` is specified as RMS-preserving: it changes
    the colour of the voice, never its level."""
    x = tone(2.0, freq=300) + tone(2.0, freq=1500, amp=0.1)
    sos, _ = render.eq_curve(np.random.default_rng(0), SR)
    y = render.apply_eq(x, sos)
    assert np.sqrt(np.mean(y ** 2)) == pytest.approx(np.sqrt(np.mean(x ** 2)),
                                                     rel=1e-6)


def test_eq_actually_changes_the_signal():
    x = tone(2.0, freq=300) + tone(2.0, freq=1500, amp=0.1)
    sos, bands = render.eq_curve(np.random.default_rng(0), SR)
    assert len(bands) == 3
    assert not np.allclose(render.apply_eq(x, sos), x)


def test_eq_bands_are_recorded_and_reproducible():
    """The curve has to be re-derivable, since it is a per-trial experimental
    variable rather than throwaway augmentation."""
    a = render.eq_curve(np.random.default_rng(7), SR)[1]
    b = render.eq_curve(np.random.default_rng(7), SR)[1]
    assert a == b
    assert all(120.0 <= x["f0_hz"] <= 0.4 * SR for x in a)
    assert all(-6.0 <= x["gain_db"] <= 6.0 for x in a)


def test_trial_seed_is_stable_across_processes():
    """Python's hash() is salted per process, so the seed must not use it --
    otherwise the EQ curve changes between render runs and the audio stops
    being reproducible from the manifest."""
    assert render.trial_seed("train-42-000123") == render.trial_seed("train-42-000123")
    assert render.trial_seed("train-42-000123") != render.trial_seed("train-42-000124")
    assert 0 <= render.trial_seed("x") < 2 ** 32


# --- room ------------------------------------------------------------------

def test_impulse_responses_are_deterministic():
    """No RNG in the acoustics: the manifest's room columns fully determine the
    RIR, which is what lets a trial be re-rendered identically."""
    args = ([5.0, 4.0, 2.8], 0.4, [2.5, 2.0, 1.4], [[3.5, 2.5, 1.6]], SR)
    np.testing.assert_array_equal(render.impulse_responses(*args)[0],
                                  render.impulse_responses(*args)[0])


def test_two_sources_get_different_rirs():
    rirs = render.impulse_responses([5.0, 4.0, 2.8], 0.4, [2.5, 2.0, 1.4],
                                    [[3.5, 2.5, 1.6], [1.5, 3.0, 1.6]], SR)
    assert len(rirs) == 2
    assert not np.array_equal(rirs[0][:100], rirs[1][:100])
