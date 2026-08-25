"""Unit tests for src/models/losses.py.

A loss function's failure mode is silence: the wrong objective still returns a
finite number, still decreases, and still produces a model. So each test pins
one decision from decisions-m1.md (2026-08-20) to a property that would break if
the code drifted.

Two of these are not hygiene, they are the reason the file exists:

  test_present_is_invariant_to_output_gain
      This is the test that caught Deviation 1. It FAILS on CARTSE eq (1) as
      published, which floors on tau*||s||^2 and therefore pays unbounded reward
      for amplifying the output.

  test_mrstft_is_more_sensitive_to_a_band_hole_than_si_sdr
      L_MR exists to price what L_pres under-weights. If a change to p, the
      window set or the reduction destroys that, every other test still passes.

Nothing here reads the corpora. Signals are synthetic and seeded, so the
expected answer is known exactly rather than approximately.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.models.losses import LossBSRNN  # noqa: E402

SR = 16000
T = 64000           # 4 s at 16 kHz, the training chunk length
N_FFT = 512
# tau is no longer one number: tau_pres floors L_pres, tau_abs floors L_abs
# (split 2026-08-25). The absent tests below read loss.tau_abs off the fixture
# rather than a module constant, so changing the default cannot silently make
# them assert the wrong floor -- which is exactly what happened at the split.
TAU_PRES = 0.001


@pytest.fixture
def loss():
    # wm/w are required constructor args; the values only matter to the
    # __call__ tests, which state their own arithmetic.
    return LossBSRNN(wm=6.6, w=0.458)


@pytest.fixture
def window():
    return torch.hann_window(N_FFT)


def speechlike(seed, batch=1):
    """White noise shaped by a steep spectral rolloff.

    Speech energy falls ~6 dB/octave, so a flat spectrum would make the band
    tests meaningless: deleting the top of a white signal removes half its
    energy, where in speech it removes ~2 %.
    """
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(batch, T, generator=g)
    win = torch.hann_window(N_FFT)
    S = torch.stft(w, N_FFT, N_FFT // 4, N_FFT, win, center=True, return_complex=True)
    f = torch.arange(S.shape[1]).float() * SR / N_FFT
    S = S * (1.0 / (1.0 + f / 250.0) ** 1.6)[None, :, None]
    return torch.istft(S, N_FFT, N_FFT // 4, N_FFT, win, center=True, length=T)


def lowpass(w, cutoff_hz):
    """Delete every bin above cutoff_hz, keeping the low band and its phase exact."""
    win = torch.hann_window(N_FFT)
    S = torch.stft(w, N_FFT, N_FFT // 4, N_FFT, win, center=True, return_complex=True)
    S[:, int(round(cutoff_hz / (SR / N_FFT))):, :] = 0
    return torch.istft(S, N_FFT, N_FFT // 4, N_FFT, win, center=True, length=w.shape[-1])


def plain_si_sdr(s_target, s_output):
    """Unfloored SI-SDR, written independently to cross-check L_pres."""
    a = (s_output * s_target).sum(-1, keepdim=True) / s_target.pow(2).sum(-1, keepdim=True)
    proj = a * s_target
    return -10 * torch.log10(proj.pow(2).sum(-1) / (s_output - proj).pow(2).sum(-1))


# --- L_pres: floored SI-SDR, CARTSE eq (1) --------------------------------

def test_present_is_exactly_minus_30_on_a_perfect_output(loss):
    """The ceiling is 1/tau = 30 dB. One line that validates the whole term:
    it only lands on -30 if every quantity in the ratio is an ENERGY (squared)
    and the floor is applied to the right one."""
    s = speechlike(0)
    assert float(loss._loss_target_present(s, s)[0]) == pytest.approx(-30.0, abs=1e-4)


@pytest.mark.parametrize("gain", [0.05, 0.2, 1.0, 5.0, 100.0])
def test_present_is_invariant_to_output_gain(loss, gain):
    """Deviation 1. FAILS on CARTSE eq (1) as published, which floors on
    tau*||s||^2: the numerator scales with the output gain g but the floor does
    not, so a perfect-shape output scaled by g scores -20log10(g) - 30 --
    unbounded reward for amplifying (measured: g=5 -> -43.98, g=100 -> -70).
    Flooring on ||s_proj||^2 makes them cancel."""
    s = speechlike(1)
    assert float(loss._loss_target_present(s, gain * s)[0]) == pytest.approx(-30.0, abs=1e-3)


def test_present_is_gain_invariant_on_an_imperfect_output(loss):
    """Gain invariance must hold away from the ceiling too, not only at -30."""
    s, x = speechlike(2), speechlike(2) + 0.5 * speechlike(3)
    ref = float(loss._loss_target_present(s, x)[0])
    for gain in (0.01, 1.0, 100.0):
        assert float(loss._loss_target_present(s, gain * x)[0]) == pytest.approx(ref, abs=1e-3)


def test_present_matches_plain_si_sdr_away_from_the_floor(loss):
    """Where the error is far above tau*||s_proj||^2 the floor is inert, so the
    term must agree with a textbook SI-SDR. Guards against the floor quietly
    distorting the normal operating range."""
    s, x = speechlike(4), speechlike(4) + 0.5 * speechlike(5)
    assert float(loss._loss_target_present(s, x)[0]) == pytest.approx(
        float(plain_si_sdr(s, x)[0]), abs=0.05)


def test_present_is_finite_when_the_model_collapses_to_silence(loss):
    """Total collapse makes numerator and denominator both exactly 0, and 0/0 is
    NaN, which no clamp removes -- hence eps on BOTH sides. 0.0 dB is worse than
    the ~-6 dB of passing the mixture through, so collapse is not an attractor."""
    s = speechlike(6)
    v = float(loss._loss_target_present(s, torch.zeros_like(s))[0])
    assert math.isfinite(v) and v == pytest.approx(0.0, abs=1e-4)


def test_present_is_nan_on_a_silent_target_by_design(loss):
    """NOT a bug. alpha is 0/0 when the target is all zero, which is why the
    caller must select present rows first. Pinning it as a test documents the
    masking requirement in a way a comment cannot."""
    s = speechlike(7)
    assert math.isnan(float(loss._loss_target_present(torch.zeros_like(s), s)[0]))


# --- L_abs: push-to-silence, CARTSE eq (2), normalised --------------------

def test_absent_floor_is_10log10_tau_abs_on_perfect_silence(loss):
    """The floor is 10log10(tau_abs), i.e. -20 dB at the default tau_abs=0.01,
    not the -30 dB it was when a single tau served both halves."""
    x = speechlike(8)
    assert float(loss._loss_target_absent(x, torch.zeros_like(x))[0]) == pytest.approx(
        10 * math.log10(loss.tau_abs), abs=1e-4)


def test_absent_do_nothing_anchor_is_not_zero(loss):
    """Deviation 2 gives 0 dB = "emitted the mixture unchanged". The exact value
    is 10log10(1 + tau_abs), NOT 0.0, because the tau floor sits in the
    numerator. A test asserting exactly 0.0 fails for a reason that looks like a
    bug and is not."""
    x = speechlike(9)
    assert float(loss._loss_target_absent(x, x)[0]) == pytest.approx(
        10 * math.log10(1 + loss.tau_abs), abs=1e-5)


def test_absent_is_positive_when_amplifying(loss):
    """>0 dB means louder than the mixture. A bug indicator, not a bad score --
    4x the energy is +6.02 dB."""
    x = speechlike(10)
    assert float(loss._loss_target_absent(x, 2 * x)[0]) == pytest.approx(
        10 * math.log10(4 + loss.tau_abs), abs=1e-3)


@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_absent_is_scale_invariant(loss, scale):
    """Deviation 2. CARTSE eq (2) shifts by 20log10(g) under a common rescale,
    so two loudness-matched silent outputs get different gradients. Dividing by
    ||x||^2 removes that."""
    x = speechlike(11)
    ref = float(loss._loss_target_absent(x, 0.1 * x)[0])
    assert float(loss._loss_target_absent(scale * x, scale * 0.1 * x)[0]) == pytest.approx(ref, abs=1e-3)


# --- L_MR: multi-resolution STFT, Yu et al. eq (3) ------------------------

def _mr(loss, a, b):
    return loss._loss_multi_res_stft(a, b, loss.windows, loss.p)


def test_mrstft_is_exactly_zero_on_a_perfect_output(loss):
    """The eps inside the magnitude must be applied identically to both signals,
    or a perfect output would not score 0."""
    s = speechlike(12)
    assert float(_mr(loss, s, s)[0]) == pytest.approx(0.0, abs=1e-7)


def test_mrstft_returns_one_value_per_example(loss):
    """torch.norm reduces over EVERY dim including batch, giving a scalar. The
    masked means in __call__ need per-example values, so reduction must be over
    (F, N) only."""
    s = speechlike(13, batch=5)
    assert tuple(_mr(loss, s, s + 0.1 * speechlike(14, batch=5)).shape) == (5,)


def test_mrstft_windows_are_milliseconds_not_samples(loss):
    """windows=(8, 16, 32, 64) are MS and convert to n_fft (128, 256, 512, 1024)
    at 16 kHz. Passed straight through, n_fft=8 gives a 5-bin FFT that still
    trains and is meaningless."""
    assert loss.windows == (8, 16, 32, 64)
    assert [int(round(w * loss.sample_rate / 1000)) for w in loss.windows] == [128, 256, 512, 1024]


@pytest.mark.parametrize("gain", [0.5, 2.0])
def test_mrstft_is_gain_sensitive(loss, gain):
    """The complement of test_present_is_invariant_to_output_gain, and it is
    wanted: L_MR compares magnitudes directly, so it is the only term that pins
    the output gain. In the wm = 0 ablation arm nothing does."""
    s = speechlike(15)
    assert float(_mr(loss, s, gain * s)[0]) > 0.1


def test_mrstft_is_more_sensitive_to_a_band_hole_than_si_sdr(loss):
    """The reason this term exists. Deleting everything above 4 kHz removes well
    under 1 % of a speech-shaped signal's energy, so L_pres barely moves, while
    L_MR registers most of the damage.

    Measured on this fixture: L_pres travels 0.17 of the way from perfect to
    do-nothing, L_MR travels 0.56 -- a factor of ~3.3. On real audio the factor
    is smaller (~1.4-3.4 depending on spectral tilt and cutoff), so the claim to
    make in writing is "L_MR is more sensitive", never "SI-SDR is blind".
    """
    s = speechlike(16)
    x = s + 0.5 * speechlike(17)                 # do-nothing reference
    holed = lowpass(s, 4000)

    energy_removed = 1 - float((holed ** 2).sum()) / float((s ** 2).sum())
    assert energy_removed < 0.01, "fixture is not speech-shaped enough for this test"

    p_perfect, p_nothing = float(loss._loss_target_present(s, s)[0]), float(loss._loss_target_present(s, x)[0])
    m_perfect, m_nothing = float(_mr(loss, s, s)[0]), float(_mr(loss, s, x)[0])
    frac_pres = (float(loss._loss_target_present(s, holed)[0]) - p_perfect) / (p_nothing - p_perfect)
    frac_mr = (float(_mr(loss, s, holed)[0]) - m_perfect) / (m_nothing - m_perfect)

    assert frac_mr > 2 * frac_pres, f"L_MR {frac_mr:.3f} vs L_pres {frac_pres:.3f}"


# --- __call__: the masked means and the weighting -------------------------

def batch(n=12, absent_idx=(2, 5, 9)):
    s, x = speechlike(20, batch=n), speechlike(20, batch=n) + 0.5 * speechlike(21, batch=n)
    crop_absent = torch.zeros(n, dtype=torch.bool)
    crop_absent[list(absent_idx)] = True
    s[crop_absent] = 0.0            # an absent crop's target stem is exactly zero
    return s, x, crop_absent


def test_call_weighting_matches_the_written_objective(loss):
    """L = (1-w) * mean_present[L_pres + wm*L_MR] + w * mean_absent[L_abs].
    Recomputed by hand from the parts dict, so a swapped w/wm is caught -- both
    are weights, and swapping them trains the wrong thing without erroring."""
    s, x, absent = batch()
    total, parts = loss(s, x, x, absent)
    expected = ((1 - loss.w) * (parts["L_pres"] + loss.wm * parts["L_MR"])
                + loss.w * parts["L_abs"])
    assert float(total) == pytest.approx(expected, rel=1e-5)


def test_call_survives_a_batch_with_no_absent_crops(loss):
    """~1.5 % of batches at batch 12 and the measured 0.297 absent rate
    (0.703^12). An unguarded mean over an empty selection is NaN."""
    s, x, absent = batch(absent_idx=())
    total, parts = loss(s, x, x, absent)
    assert math.isfinite(float(total))
    assert parts["n_absent"] == 0 and math.isnan(parts["L_abs"])


def test_call_survives_a_batch_with_no_present_crops(loss):
    s, x, absent = batch(absent_idx=tuple(range(12)))
    total, parts = loss(s, x, x, absent)
    assert math.isfinite(float(total))
    assert parts["n_present"] == 0 and math.isnan(parts["L_pres"])


def test_call_rejects_a_present_crop_whose_target_is_silent(loss):
    """The most likely caller error: branching on the manifest condition label
    instead of the loader's crop_absent. 5.8 % of both/target_only crops land in
    target silence (decisions-m1.md 2026-08-18), so the label sends ~1 crop in
    17 down the L_pres path with an all-zero target."""
    s, x, _ = batch()
    with pytest.raises(AssertionError, match="crop_absent disagrees"):
        loss(s, x, x, torch.zeros(12, dtype=torch.bool))


def test_call_parts_dict_has_a_stable_key_set(loss):
    """The logger's schema must not change between batches, or a missing half
    becomes a missing column instead of a gap in the curve."""
    keys = {"L_pres", "L_MR", "L_abs", "n_present", "n_absent", "total"}
    for idx in ((), (2, 5, 9), tuple(range(12))):
        s, x, absent = batch(absent_idx=idx)
        assert set(loss(s, x, x, absent)[1]) == keys


@pytest.mark.parametrize("absent_idx", [(), (2, 5, 9), tuple(range(12))])
def test_call_gradients_are_finite(loss, absent_idx):
    """A NaN anywhere in the loss destroys every weight in the model on the step
    it appears, so this is checked on all three batch compositions."""
    s, x, absent = batch(absent_idx=absent_idx)
    out = x.clone().requires_grad_(True)
    loss(s, out, x, absent)[0].backward()
    assert bool(torch.isfinite(out.grad).all())
