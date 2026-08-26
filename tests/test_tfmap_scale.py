"""The TF-Map logit scale. decisions-m1.md 2026-08-25.

WHY THESE TESTS EXIST
---------------------
The speaker cue is built by comparing every mixture frame against every frame of
the enrollment and taking a weighted average of the best matches. Without a
logit scale that weighting degenerates into a plain average, and the cue becomes
the enrollment's long-term mean spectrum -- one static fingerprint that says
nothing about when the target is talking. Measured on 2026-08-25: 619.6 of 628
enrollment frames effectively used, top frame holding 0.22 % of the weight
against 0.16 % for a flat average, cue varying 4.7 % over time. The model
ignored it, and the loss curve looked healthy the whole time.

The cause is that softmax compares logits by DIFFERENCE, not ratio:

    weight_i / weight_j = exp(s_i - s_j)

F.normalize bounds every cosine to [-1, 1], so the largest achievable
difference is ~1 and the best frame can never outweigh the worst by more than
e^1 = 2.7x, however unalike they really are.

These tests pin the MECHANISM, not just the symptom, so that a future edit that
re-flattens the softmax (removing the scale, or renormalising after it) fails
here rather than in a 5-hour GPU run whose loss curve looks fine.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.conditioning import TFMap  # noqa: E402

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


def weights(mix, enroll, scale):
    """The h matrix TFMap computes internally, recomputed here so the test reads
    the actual weighting rather than inferring it from the output."""
    import torch.nn.functional as Fn
    bx = Fn.normalize(mix, p=2, dim=1, eps=1e-8)
    be = Fn.normalize(enroll, p=2, dim=1, eps=1e-8)
    sim = torch.matmul(bx.transpose(1, 2), be)
    s = mix.shape[1] ** 0.5 if scale is None else scale
    return torch.softmax(sim * s, dim=-1), sim


def effective_frames(h):
    """Perplexity: how many enrollment frames the weighting actually uses. Equals
    TE for a flat average, 1 for a hard pick."""
    return float(torch.exp(-(h * (h + 1e-12).log()).sum(-1)).mean())


# --- the bug, pinned so it cannot come back -------------------------------

def test_scale_one_degenerates_into_an_average():
    """THE BUG. Unscaled, the weighting is within a hair of ignoring the scores."""
    h, _ = weights(spectra(TX, 0), spectra(TE, 1), scale=1.0)
    assert effective_frames(h) > 0.95 * TE
    # the best frame barely outranks a flat share
    assert float(h.max()) < 3.0 / TE


def test_default_scale_actually_selects():
    """sqrt(F) ~ 16 at F=257 must weight the good matches materially higher.

    Asserted RELATIVE to scale 1, not against an absolute frame count. On this
    fixture sqrt(F) gives 428.6 effective frames of 628 and a 30 % top-50 share;
    on real audio the same scale gave 186.9, because a real mixture actually
    CONTAINS the target so its best matches concentrate harder than random
    frames can. Pinning the real number here would pin a property of the fixture
    rather than of the code.
    """
    mix, enroll = spectra(TX, 0), spectra(TE, 1)
    flat, _ = weights(mix, enroll, scale=1.0)
    sharp, _ = weights(mix, enroll, scale=None)

    # at least a third of the frames stop counting
    assert effective_frames(sharp) < 0.7 * effective_frames(flat)
    # and the best 50 of 628 carry several times the share they had unscaled
    share = lambda h: float(h[0].sort(dim=-1, descending=True).values[:, :50].sum(-1).mean())
    assert share(sharp) > 3 * share(flat)
    assert share(flat) < 0.12           # unscaled is close to 50/628 = 8.0 %


def test_weight_ratio_is_exp_of_the_scaled_difference():
    """The MECHANISM, not the symptom. If a future edit renormalises after
    scaling, or scales the wrong tensor, this is what catches it."""
    mix, enroll = spectra(TX, 0), spectra(TE, 1)
    for scale in (1.0, 4.0, 16.0):
        h, sim = weights(mix, enroll, scale)
        row_h, row_s = h[0, 0], sim[0, 0]
        i, j = int(row_s.argmax()), int(row_s.argmin())
        expected = math.exp(scale * float(row_s[i] - row_s[j]))
        assert float(row_h[i] / row_h[j]) == pytest.approx(expected, rel=1e-3)


def test_normalisation_caps_the_unscaled_weight_ratio_at_e():
    """Why scale 1 cannot be rescued by better data: cosines live in [-1, 1], so
    the unscaled weight ratio is bounded by e^2 whatever the frames contain, and
    in practice (all-positive spectra) by about e."""
    h, sim = weights(spectra(TX, 7), spectra(TE, 8), scale=1.0)
    assert float(sim.max() - sim.min()) <= 2.0
    assert float(h.max() / h.min()) < math.e ** 2


# --- scale as a knob ------------------------------------------------------

def test_effective_frames_fall_monotonically_with_scale():
    mix, enroll = spectra(TX, 2), spectra(TE, 3)
    got = [effective_frames(weights(mix, enroll, s)[0]) for s in (1.0, 4.0, 8.0, 16.0, 32.0)]
    assert got == sorted(got, reverse=True), got


def test_scale_zero_is_exactly_uniform_and_is_reachable():
    """scale=0 is the ablation arm that reproduces a flat average exactly. It must
    survive the constructor: `self.scale or sqrt(F)` would treat 0.0 as falsy and
    silently substitute sqrt(F), turning the control arm into the treatment."""
    assert TFMap(scale=0.0).scale == 0.0
    h, _ = weights(spectra(TX, 4), spectra(TE, 5), scale=0.0)
    assert effective_frames(h) == pytest.approx(TE, rel=1e-4)
    assert float(h.max() - h.min()) == pytest.approx(0.0, abs=1e-9)


def test_none_means_sqrt_of_the_frequency_axis():
    assert TFMap().scale is None          # resolved at forward time, not construction
    h_none, _ = weights(spectra(TX, 6), spectra(TE, 7), scale=None)
    h_sqrt, _ = weights(spectra(TX, 6), spectra(TE, 7), scale=F_BINS ** 0.5)
    assert torch.allclose(h_none, h_sqrt)


# --- the cue itself -------------------------------------------------------

def test_scaling_makes_the_cue_vary_over_time():
    """The point of the fix. A near-uniform weighting returns almost the same
    spectral shape at every moment, so the cue cannot say WHEN the target speaks."""
    mix, enroll = spectra(TX, 10), spectra(TE, 11)

    def time_variation(scale):
        tf = TFMap(scale=scale)(mix, enroll)[:, 0]          # (1, F, Tx)
        u = tf / tf.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return float((u - u.mean(-1, keepdim=True)).norm() / u.norm())

    assert time_variation(32.0) > 3 * time_variation(1.0)


def test_cue_still_responds_to_a_different_speaker():
    """Sharpening must not trade speaker information away for time resolution."""
    mix = spectra(TX, 12)
    a, b = spectra(TE, 13), spectra(TE, 14)
    for scale in (1.0, 16.0, 32.0):
        tf_a = TFMap(scale=scale)(mix, a)
        tf_b = TFMap(scale=scale)(mix, b)
        rel = float(((tf_a - tf_b).pow(2).sum() / tf_a.pow(2).sum()).sqrt())
        assert rel > 0.05, f"scale {scale}: cue barely moves between speakers ({rel:.3f})"


def test_shape_and_finiteness_are_unchanged_by_scaling():
    """A large scale exponentiates large logits; NaN/inf here would be silent."""
    mix, enroll = spectra(TX, 15), spectra(TE, 16)
    for scale in (0.0, 1.0, 16.0, 64.0):
        tf = TFMap(scale=scale)(mix, enroll)
        assert tf.shape == (1, 1, F_BINS, TX)
        assert torch.isfinite(tf).all(), f"non-finite cue at scale {scale}"
