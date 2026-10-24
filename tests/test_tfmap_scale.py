"""The TF-Map cue at different logit scales (decisions-m2.md 2026-08-25).

Without a large enough scale the softmax averages every enrolment frame, and the cue becomes one static fingerprint that says nothing about when the target speaks.
These tests check the real cue: it varies over time, it changes with the speaker, and it stays finite.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.conditioning import tfmap

F_BINS, TE, TX = 257, 628, 40      # 257 bins at n_fft=512; 628 frames = 5 s enrollment


def spectra(n_frames, seed):
    """Non-negative frames standing in for a magnitude spectrogram.

    CALIBRATED, not arbitrary. The only property these tests depend on is how
    far apart the cosines between frames end up, because that is what the logit
    scale acts on. Real speech measured 2026-08-25: cosines 0.055..0.980, a
    spread of 0.925. `rand ** 8` gives 0.051..1.000, spread 0.949 -- the heavy
    tail makes each frame dominated by a few bins, like a harmonic spectrum.

    Plain `rand` does NOT work: 257 uniform positive values all normalise to
    nearly the same direction, giving a spread of only 0.286, so even a x16
    scale cannot separate them and every test below reads a false negative. The
    first draft of this file used `rand + 0.05` and failed for exactly that
    reason -- the data was wrong, not the code.

    +1e-6 so no frame is all-zero: that normalises to zero and the cosine is
    then undefined rather than small.
    """
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, F_BINS, n_frames, generator=g) ** 8 + 1e-6


def test_scaling_makes_the_cue_vary_over_time():
    """The point of the fix. A near-uniform weighting returns almost the same
    spectral shape at every moment, so the cue cannot say WHEN the target speaks."""
    mix, enroll = spectra(TX, 10), spectra(TE, 11)

    def time_variation(scale):
        tf = tfmap(mix, enroll, scale=scale, return_parts=False, part_scales=None)[:, 0]          # (1, F, Tx)
        u = tf / tf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return float((u - u.mean(-1, keepdim=True)).norm() / u.norm())

    assert time_variation(32.0) > 3 * time_variation(1.0)


def test_cue_still_responds_to_a_different_speaker():
    """Sharpening must not trade speaker information away for time resolution."""
    mix = spectra(TX, 12)
    a, b = spectra(TE, 13), spectra(TE, 14)
    for scale in (1.0, 16.0, 32.0):
        tf_a = tfmap(mix, a, scale=scale, return_parts=False, part_scales=None)
        tf_b = tfmap(mix, b, scale=scale, return_parts=False, part_scales=None)
        rel = float(((tf_a - tf_b).pow(2).sum() / tf_a.pow(2).sum()).sqrt())
        assert rel > 0.05, f"scale {scale}: cue barely moves between speakers ({rel:.3f})"


def test_shape_and_finiteness_are_unchanged_by_scaling():
    """A large scale exponentiates large logits; NaN/inf here would be silent."""
    mix, enroll = spectra(TX, 15), spectra(TE, 16)
    for scale in (0.0, 1.0, 16.0, 64.0):
        tf = tfmap(mix, enroll, scale=scale, return_parts=False, part_scales=None)
        assert tf.shape == (1, 1, F_BINS, TX)
        assert torch.isfinite(tf).all(), f"non-finite cue at scale {scale}"
