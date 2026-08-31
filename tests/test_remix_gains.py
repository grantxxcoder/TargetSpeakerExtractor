"""Unit tests for the per-epoch SIR/SNR remix (dataset_loader._remix).

The remix takes a rendered mixture apart and puts it back together at a
different loudness balance. Its failure modes are all silent -- a wrong sign, a
corrupted reference, a drifted clip ceiling -- so each test pins one property the
2026-08-30 decision entry depends on:

  * the reference stem is not damaged, because it is the training target
  * re-drawing a trial's OWN numbers reproduces the rendered mixture exactly,
    which is what makes remix on/off a clean ablation rather than a confound
  * the sign is right: a lower SIR means a LOUDER interferer
  * target-absent trials pass through, because their recorded numbers are
    relative to a different loudness anchor
  * validation never remixes

Signals are synthetic and the "noise" is a distinct constant, so every component
is recoverable from the output by arithmetic rather than by approximation.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import render  # noqa: E402
from src.data.dataset_loader import CLIP_CEILING, TrialDataset  # noqa: E402

SR = 16000
CHUNK_S = 0.5
CLIP = int(CHUNK_S * SR) * 4          # clip is 4 crops long, so offsets vary

# One row per condition, so the eligibility rules are all exercised.
TRIALS = [
    #  id        condition          absent  sir    snr   regime
    ("t-000", "both",            0,  3.0,  12.0, "base"),
    ("t-001", "both",            0, -6.0,  18.0, "base"),
    ("t-002", "both",            0,  8.0,   9.0, "hard"),
    ("t-003", "target_only",     0,  0.0,  15.0, "base"),
    ("t-004", "interferer_only", 1,  2.0,  11.0, "base"),
    ("t-005", "noise_only",      1,  1.0,  10.0, "hard"),
]


def _write(path, value):
    sf.write(path, np.full(CLIP, value, dtype=np.float32), SR, subtype="FLOAT")


@pytest.fixture
def data_root(tmp_path):
    """A miniature split whose stems are distinct constants.

    target 0.20, interferer 0.05, noise 0.01 -> mixture 0.26, so subtracting the
    two stems out of the mixture must return exactly 0.01. FLOAT subtype, not
    PCM_16: the test is about arithmetic, and 16-bit quantisation would put a
    ~3e-5 floor under every comparison for no reason.
    """
    split = "unit"
    rendered = tmp_path / "rendered" / split
    rows = []
    for tid, cond, absent, sir, snr in [(t[0], t[1], t[2], t[3], t[4]) for t in TRIALS]:
        d = rendered / tid
        d.mkdir(parents=True)
        tgt = 0.0 if cond in ("interferer_only", "noise_only") else 0.20
        itf = 0.0 if cond in ("target_only", "noise_only") else 0.05
        _write(d / "target.wav", tgt)
        _write(d / "interferer.wav", itf)
        _write(d / "mixture.wav", tgt + itf + 0.01)
        for e in ("enrollment", "interferer_enrollment"):
            _write(d / f"{e}.wav", 0.3)
        rows.append({"trial_id": tid, "condition": cond, "target_absent": absent,
                     "sir_db": sir, "snr_db": snr, "overlap_achieved": 0.5,
                     "regime": [t[5] for t in TRIALS if t[0] == tid][0],
                     "same_gender": 0.0, "interferer_enrollment_phantom": 0})
    manifest = tmp_path / "unit.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path, manifest, split


def make(data_root, **kw):
    root, manifest, split = data_root
    opts = dict(manifest_csv=manifest, data_root=root, split=split,
                chunk_s=CHUNK_S, sample_rate=SR, seed=42, both_directions=True)
    opts.update(kw)
    return TrialDataset(**opts)


# --- the constant must not drift from the renderer's ----------------------

def test_clip_ceiling_matches_the_renderer():
    """The remix re-runs A6. A ceiling that drifted from render.py would
    silently rescale every crop the remix touches."""
    assert CLIP_CEILING == render.CLIP_CEILING


# --- off means off ---------------------------------------------------------

def test_remix_off_is_byte_identical(data_root):
    """The control arm. Restructuring __getitem__ to read the mixture once must
    not have changed a single sample of what it returns."""
    off = make(data_root, remix_gains=False)
    for i in range(len(TRIALS)):
        ex = off[i]
        assert torch.equal(ex[0]["mixture"], ex[1]["mixture"])
        assert float(ex[0]["mixture"][0]) == pytest.approx(
            float(ex[0]["target"][0]) + float(ex[1]["target"][0]) + 0.01, abs=1e-6)


def test_redrawing_a_trials_own_numbers_reproduces_the_render(data_root, monkeypatch):
    """Take the mixture apart and put it back with the SAME SIR and SNR, and the
    rendered audio must come back exactly. This is what proves the arithmetic is
    a decomposition and not an approximation."""
    ds = make(data_root, remix_gains=True)
    monkeypatch.setattr(ds, "_draw_gains",
                        lambda idx, row: (float(row["sir_db"]), float(row["snr_db"])))
    plain = make(data_root, remix_gains=False)
    for i in range(len(TRIALS)):
        got, want = ds[i][0], plain[i][0]
        assert torch.allclose(got["mixture"], want["mixture"], atol=1e-6)
        assert torch.allclose(got["target"], want["target"], atol=1e-6)


# --- the sign, and the reference ------------------------------------------

def test_lower_sir_makes_the_interferer_louder(data_root, monkeypatch):
    """SIR is target level MINUS interferer level. Getting this backwards would
    make every 'hard' trial easy and still train without error."""
    ds = make(data_root, remix_gains=True)
    # trial 0 was rendered at SIR 3 dB; ask for -3 dB, i.e. 6 dB more interferer
    monkeypatch.setattr(ds, "_draw_gains", lambda idx, row: (-3.0, None))
    itf = float(ds[0][1]["target"][0])          # direction 1's reference IS the interferer
    assert itf == pytest.approx(0.05 * 10 ** (6.0 / 20.0), rel=1e-4)


def test_the_reference_target_is_untouched(data_root, monkeypatch):
    """`target.wav` is what the loss scores against. The remix is allowed to
    change what the model HEARS and nothing else -- unless the clip guard fires,
    which is tested separately."""
    ds = make(data_root, remix_gains=True)
    monkeypatch.setattr(ds, "_draw_gains", lambda idx, row: (-9.0, 4.0))
    assert float(ds[0][0]["target"][0]) == pytest.approx(0.20, abs=1e-6)


def test_snr_scales_only_the_noise(data_root, monkeypatch):
    """6 dB less SNR doubles the noise and leaves both speakers alone."""
    ds = make(data_root, remix_gains=True)
    monkeypatch.setattr(ds, "_draw_gains", lambda idx, row: (None, 12.0 - 6.02))
    ex = ds[0]
    noise = float(ex[0]["mixture"][0]) - float(ex[0]["target"][0]) - float(ex[1]["target"][0])
    assert noise == pytest.approx(0.02, rel=1e-3)


# --- the clip guard --------------------------------------------------------

def test_clip_guard_scales_mixture_and_reference_together(data_root, monkeypatch):
    """A6 applies ONE factor across every stem so the balance survives the fix.
    Scaling the mixture alone would silently corrupt the reference level that
    L_gain measures."""
    ds = make(data_root, remix_gains=True)
    monkeypatch.setattr(ds, "_draw_gains", lambda idx, row: (-30.0, None))
    ex = ds[0]
    assert float(ex[0]["mixture"].abs().max()) <= CLIP_CEILING + 1e-6
    # target and interferer kept their ratio to the mixture
    ratio = float(ex[0]["target"][0]) / float(ex[0]["mixture"][0])
    assert ratio == pytest.approx(0.20 / (0.20 + 0.05 * 10 ** (33.0 / 20.0) + 0.01),
                                  rel=1e-3)


# --- eligibility -----------------------------------------------------------

def test_target_absent_trials_pass_through(data_root):
    """On these the loudness anchor is the interferer or the noise, not the
    target, so the recorded SIR/SNR are not target-relative and re-applying them
    would be meaningless arithmetic."""
    ds = make(data_root, remix_gains=True)
    plain = make(data_root, remix_gains=False)
    for i in (4, 5):                                   # interferer_only, noise_only
        for epoch in range(6):
            ds.set_epoch(epoch); plain.set_epoch(epoch)
            assert torch.equal(ds[i][0]["mixture"], plain[i][0]["mixture"])


def test_donors_come_from_the_same_regime(data_root):
    """The difficulty dial is per regime. Drawing a `base` SIR onto a `hard`
    trial would quietly flatten it."""
    ds = make(data_root, remix_gains=True)
    base_sir, base_snr = ds._gain_pools["base"]
    hard_sir, hard_snr = ds._gain_pools["hard"]
    assert sorted(base_sir) == [-6.0, 3.0]      # both-condition base trials only
    assert sorted(hard_sir) == [8.0]
    assert sorted(base_snr) == [12.0, 15.0, 18.0]   # every target-present base trial
    assert sorted(hard_snr) == [9.0]


def test_the_mixture_changes_across_epochs(data_root):
    """The whole point: the same trial must be a different difficulty on
    different epochs, or nothing has been augmented."""
    ds = make(data_root, remix_gains=True)
    seen = set()
    for epoch in range(24):
        ds.set_epoch(epoch)
        seen.add(round(float(ds[0][0]["mixture"][0]), 6))
    assert len(seen) > 1, "the mixture never changed -- the remix is inert"


# --- validation ------------------------------------------------------------

def test_validation_never_remixes(data_root):
    """random_crop=False marks a fixed evaluation set. A val set whose difficulty
    moved per epoch would make its own loss curve unreadable, which is the curve
    this whole change exists to move."""
    val = make(data_root, remix_gains=True, random_crop=False)
    plain = make(data_root, remix_gains=False, random_crop=False)
    assert val.remix_gains is False
    for epoch in (0, 4, 11):
        val.set_epoch(epoch); plain.set_epoch(epoch)
        for i in range(len(TRIALS)):
            assert torch.equal(val[i][0]["mixture"], plain[i][0]["mixture"])


# --- metadata honesty ------------------------------------------------------

def test_meta_reports_the_realised_levels(data_root, monkeypatch):
    """Stratified diagnostics read these. After a remix they must describe the
    audio the model heard, not the audio still sitting on disk."""
    ds = make(data_root, remix_gains=True)
    monkeypatch.setattr(ds, "_draw_gains", lambda idx, row: (-7.5, 14.0))
    meta = ds[0][0]["meta"]
    assert meta["sir_db"] == pytest.approx(-7.5)
    assert meta["snr_db"] == pytest.approx(14.0)
    assert make(data_root, remix_gains=False)[0][0]["meta"]["sir_db"] == pytest.approx(3.0)
