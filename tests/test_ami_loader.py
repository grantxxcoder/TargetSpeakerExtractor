"""The two loader columns M8 adds (decisions-m8.md 2026-10-12).

  * `audio_dir`: one manifest draws trials from several rendered splits, so the
    50/50 AMI + sir0 blend needs no copied or symlinked audio;
  * `remixable`: AMI rows are not target + interferer + noise (different
    microphones), so they must reach the model exactly as recorded and never
    donate SIR/SNR to a synthetic row.

And the gated target: a crop wholly outside the target's own speech is exactly
zero, so `crop_absent` fires on real audio as it does on synthetic.
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
CLIP = int(CHUNK_S * SR) * 4


def _write(path, value):
    sf.write(path, np.full(CLIP, value, dtype=np.float32), SR, subtype="FLOAT")


@pytest.fixture
def blend(tmp_path):
    """Two synthetic rows under rendered/synth, two AMI-like rows under
    rendered/real. The AMI rows' mixture is NOT the sum of their stems."""
    rows = []
    for split, tid, target, remixable, regime, sir in (
            ("synth", "s-000", 0.20, 1, "base", 3.0),
            ("synth", "s-001", 0.20, 1, "base", -6.0),
            ("real", "r-000", 0.20, 0, "ami", 40.0),
            ("real", "r-001", 0.00, 0, "ami", 40.0)):    # target gated silent throughout
        d = tmp_path / "rendered" / split / tid
        d.mkdir(parents=True)
        _write(d / "target.wav", target)
        _write(d / "interferer.wav", 0.05)
        _write(d / "mixture.wav", 0.123)                  # unrelated to the stems for AMI
        for e in ("enrollment", "interferer_enrollment"):
            _write(d / f"{e}.wav", 0.3)
            _write(d / f"{e}_v00.wav", 0.3)
        rows.append({"trial_id": tid, "audio_dir": split, "condition": "both",
                     "target_absent": 0, "sir_db": sir, "snr_db": 12.0,
                     "overlap_achieved": 0.1, "regime": regime, "same_gender": 1,
                     "interferer_enrollment_phantom": 0, "remixable": remixable})
    manifest = tmp_path / "blend.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return tmp_path, manifest


def make(blend, **kw):
    root, manifest = blend
    opts = dict(manifest_csv=manifest, data_root=root, split="unused", chunk_s=CHUNK_S,
                sample_rate=SR, seed=42, both_directions=True, remix_gains=True,
                enrollment_variants=3)
    opts.update(kw)
    return TrialDataset(**opts)


def test_rows_are_read_from_their_own_audio_dir(blend):
    ds = make(blend)
    assert [ex["trial_id"] for i in range(4) for ex in ds[i]][::2] == [
        "s-000", "s-001", "r-000", "r-001"]


def test_ami_rows_reach_the_model_exactly_as_recorded_under_remix(blend):
    ds = make(blend)
    for epoch in range(3):
        ds.set_epoch(epoch)
        for i in (2, 3):
            assert np.allclose(ds[i][0]["mixture"].numpy(), 0.123)
            assert ds[i][0]["meta"]["sir_db"] == 40.0


def test_ami_rows_never_donate_gains_to_synthetic_rows(blend):
    ds = make(blend)
    assert set(ds._gain_pools) == {"base"}
    assert 40.0 not in ds._gain_pools["base"][0]


def test_a_gated_silent_target_crop_is_absent(blend):
    ds = make(blend)
    target_dir, interferer_dir = ds[3]
    assert target_dir["crop_absent"] and not interferer_dir["crop_absent"]
    assert not ds[2][0]["crop_absent"]


def test_a_missing_bank_in_either_audio_dir_fails_at_construction(blend):
    root, _ = blend
    (root / "rendered/real/r-000/enrollment_v00.wav").unlink()
    with pytest.raises(FileNotFoundError, match="r-000"):
        make(blend)


def test_manifests_without_the_new_columns_behave_as_before(blend):
    root, manifest = blend
    df = pd.read_csv(manifest)
    df = df[df.audio_dir == "synth"].drop(columns=["audio_dir", "remixable"])
    plain = root / "plain.csv"
    df.to_csv(plain, index=False)
    ds = make(blend, manifest_csv=plain, split="synth")
    assert ds._audio_dir(df.iloc[0]) == "synth" and ds._remixable(df.iloc[0])
    assert set(ds._gain_pools) == {"base"} and len(ds._gain_pools["base"][0]) == 2
