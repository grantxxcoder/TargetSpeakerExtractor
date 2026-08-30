"""Unit tests for the per-epoch enrollment bank (dataset_loader + config wiring).

The failure mode being guarded against is silence, in the literal sense that a
bank that is misconfigured still trains, still produces a loss curve, and still
looks exactly like a run that had the augmentation on. So each test pins one
property that the 2026-08-30 decision entry relies on:

  * K=1 is byte-identical to the pre-bank behaviour, or the ablation arm is a
    confound rather than a control
  * the cue actually rotates across epochs, or the augmentation does nothing
  * validation never rotates, or every val number in the project's history
    becomes incomparable
  * a config asking for a bank that is not on disk fails LOUDLY at construction

Nothing here reads the corpora or renders audio: each "recording" is a distinct
constant, so which file was opened is recoverable exactly from the samples.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402

SR = 16000
CHUNK_S = 0.5
N_TRIALS = 6
ENROLS = ("enrollment", "interferer_enrollment")


def _write(path, value, seconds=1.0):
    """A constant-valued clip. The constant IS the file's identity."""
    sf.write(path, np.full(int(seconds * SR), value, dtype=np.float32), SR,
             subtype="PCM_16")


@pytest.fixture
def data_root(tmp_path):
    """A miniature rendered split plus a manifest, with a 4-deep bank.

    Variant k of trial i gets the constant (i+1)/100 + k/1000, so a returned
    enrollment tensor identifies the exact file that was read.
    """
    split = "unit"
    rendered = tmp_path / "rendered" / split
    rows = []
    for i in range(N_TRIALS):
        tid = f"unit-{i:03d}"
        d = rendered / tid
        d.mkdir(parents=True)
        for stem in ("mixture", "target", "interferer"):
            _write(d / f"{stem}.wav", 0.5, seconds=2.0)
        for e in ENROLS:
            _write(d / f"{e}.wav", (i + 1) / 100.0)
            for k in range(4):
                _write(d / f"{e}_v{k:02d}.wav", (i + 1) / 100.0 + k / 1000.0)
        rows.append({"trial_id": tid, "condition": "both", "target_absent": 0,
                     "sir_db": 0.0, "snr_db": 10.0, "overlap_achieved": 0.5,
                     "regime": "base", "same_gender": 0.0,
                     "interferer_enrollment_phantom": 0})
    manifest = tmp_path / "unit.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path, manifest, split


def make(data_root, **kw):
    root, manifest, split = data_root
    opts = dict(manifest_csv=manifest, data_root=root, split=split,
                chunk_s=CHUNK_S, sample_rate=SR, seed=42, both_directions=True)
    opts.update(kw)
    return TrialDataset(**opts)


def cue(ds, idx, direction=0):
    """The constant carried by the enrollment this example was conditioned on."""
    return round(float(ds[idx][direction]["enrollment"][0]), 4)


# --- K=1 must be the old behaviour exactly --------------------------------

def test_one_variant_reads_the_original_enrollment(data_root):
    """The control arm. If K=1 quietly read v00 instead, the ablation would be
    comparing two *different* renders and nothing could be attributed to the
    bank."""
    ds = make(data_root, enrollment_variants=1)
    for i in range(N_TRIALS):
        assert cue(ds, i) == pytest.approx((i + 1) / 100.0, abs=1e-4)


def test_one_variant_ignores_the_epoch(data_root):
    ds = make(data_root, enrollment_variants=1)
    seen = set()
    for epoch in range(8):
        ds.set_epoch(epoch)
        seen.add(cue(ds, 0))
    assert len(seen) == 1


# --- the cue actually rotates ---------------------------------------------

def test_the_enrollment_changes_across_epochs(data_root):
    """The whole point. A fixed cue across 20+ epochs is what made the identity
    signal a memorisable lookup in the 2026-08-29 run."""
    ds = make(data_root, enrollment_variants=4)
    seen = set()
    for epoch in range(24):
        ds.set_epoch(epoch)
        seen.add(cue(ds, 0))
    assert len(seen) > 1, "the enrollment never changed -- the bank is inert"


def test_every_variant_is_reachable(data_root):
    """A biased draw would leave part of the bank unused, so the storage would
    be paid for and not spent."""
    ds = make(data_root, enrollment_variants=4)
    seen = set()
    for epoch in range(40):
        ds.set_epoch(epoch)
        for i in range(N_TRIALS):
            seen.add(round(cue(ds, i) - (i + 1) / 100.0, 3))
    assert len(seen) == 4, f"expected 4 distinct variants, saw {sorted(seen)}"


def test_the_mixture_crop_is_unaffected(data_root):
    """The bank must not perturb the crop stream. If adding it shifted the crop
    RNG, every run before and after would be incomparable for a second,
    undocumented reason."""
    a = make(data_root, enrollment_variants=1)
    b = make(data_root, enrollment_variants=4)
    for epoch in (0, 3, 7):
        a.set_epoch(epoch); b.set_epoch(epoch)
        for i in range(N_TRIALS):
            assert a._crop_offset_start(i, 2 * SR) == b._crop_offset_start(i, 2 * SR)


# --- the two directions must vary independently ---------------------------

def test_directions_draw_their_variants_independently(data_root):
    """The two enrollments are different speakers, so locking their variant
    choice together would halve the number of distinct cue pairs seen."""
    ds = make(data_root, enrollment_variants=4)
    differed = False
    for epoch in range(24):
        ds.set_epoch(epoch)
        for i in range(N_TRIALS):
            ex = ds[i]
            kt = round(float(ex[0]["enrollment"][0]) - (i + 1) / 100.0, 4)
            ki = round(float(ex[1]["enrollment"][0]) - (i + 1) / 100.0, 4)
            differed |= kt != ki
    assert differed, "target and interferer always picked the same variant"


# --- validation must never rotate -----------------------------------------

def test_validation_is_pinned_to_variant_zero(data_root):
    """random_crop=False is how the val set is held fixed. The cue has to be
    held with it, and to variant 0 specifically, because v00 IS enrollment.wav
    -- that is what keeps val comparable with every run before the bank."""
    val = make(data_root, enrollment_variants=4, random_crop=False)
    baseline = make(data_root, enrollment_variants=1)
    for epoch in (0, 5, 19):
        val.set_epoch(epoch)
        for i in range(N_TRIALS):
            assert cue(val, i) == pytest.approx(cue(baseline, i), abs=1e-4)


# --- a missing bank is a configuration error, not a fallback --------------

def test_missing_bank_raises_at_construction(data_root, tmp_path):
    """Loudly, and before the first epoch. A silent fallback would produce a
    history.csv that disagrees with the config that produced it -- the same
    failure the `amp` flag is config-driven to avoid."""
    root, manifest, split = data_root
    for e in ENROLS:
        (root / "rendered" / split / "unit-000" / f"{e}_v00.wav").unlink()
    with pytest.raises(FileNotFoundError, match="render_enrollment_bank"):
        make(data_root, enrollment_variants=4)


def test_short_bank_falls_back_to_variant_zero(data_root):
    """A speaker with few long utterances legitimately supports fewer variants.
    That is a weakening of the augmentation for that speaker, not a broken run,
    so it degrades to variant 0 instead of failing."""
    root, manifest, split = data_root
    (root / "rendered" / split / "unit-001" / "enrollment_v03.wav").unlink()
    ds = make(data_root, enrollment_variants=4)
    for epoch in range(24):
        ds.set_epoch(epoch)
        got = cue(ds, 1)
        assert got != pytest.approx(0.02 + 3 / 1000.0, abs=1e-5)
        assert got in [pytest.approx(0.02 + k / 1000.0, abs=1e-4) for k in range(3)]
