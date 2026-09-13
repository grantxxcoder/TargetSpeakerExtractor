"""The loader must report WHERE its crop starts, and the offset is not zero.

This pins the contract that `scripts/derive_w_state.py` silently violated until
2026-09-12. That script rebuilt synthetic partial-suppression anchors as
`target + beta * interferer + noise`, reading the interferer stem with
`audio[:n_samples]` on the stated grounds that "random_crop=False, so the crop
is deterministic and starts at 0". It is deterministic and it is not zero:
`_crop_offset_start` draws from (seed, 0, idx) in both modes.

The consequence was invisible at the only place anyone looked. `noise` is
recovered as `mixture - target - interferer`, so a misaligned interferer leaves
the real one inside `noise` and subtracts a shifted copy. At beta = 1 the
mixture still reconstructs exactly -- so the passthrough anchor was right -- and
below that the anchor gains a second voice at amplitude (1 - beta), which is why
the measured readings went UP as the interferer was attenuated.

Stems here are RAMPS, not constants: a constant signal is offset-invariant, so
the existing fixture in test_remix_gains.py could not have caught this and a new
one is required rather than an extra assertion on the old.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset  # noqa: E402

SR = 16000
CHUNK_S = 0.5
CHUNK = int(CHUNK_S * SR)
CLIP = CHUNK * 6                 # six crops of slack, so offsets vary widely
N_TRIALS = 8


def ramp(scale, offset):
    """A signal whose value identifies its own sample index."""
    return (offset + scale * np.arange(CLIP, dtype=np.float32) / CLIP).astype(np.float32)


@pytest.fixture
def data_root(tmp_path):
    split = "unit"
    rendered = tmp_path / "rendered" / split
    rows = []
    for i in range(N_TRIALS):
        tid = f"t-{i:03d}"
        d = rendered / tid
        d.mkdir(parents=True)
        target, interferer, noise = ramp(0.2, 0.1), ramp(0.05, 0.02), ramp(0.01, 0.005)
        sf.write(d / "target.wav", target, SR, subtype="FLOAT")
        sf.write(d / "interferer.wav", interferer, SR, subtype="FLOAT")
        sf.write(d / "mixture.wav", target + interferer + noise, SR, subtype="FLOAT")
        for e in ("enrollment", "interferer_enrollment"):
            sf.write(d / f"{e}.wav", ramp(0.3, 0.0), SR, subtype="FLOAT")
        rows.append({"trial_id": tid, "condition": "both", "target_absent": 0,
                     "sir_db": 3.0, "snr_db": 12.0, "overlap_achieved": 0.5,
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


def test_crop_start_is_reported(data_root):
    dataset = make(data_root, random_crop=False)
    start = dataset[0][0]["meta"]["crop_start"]
    assert isinstance(start, int)
    assert 0 <= start <= CLIP - CHUNK


def test_a_fixed_set_does_not_crop_at_zero(data_root):
    """THE FALSE ASSUMPTION, pinned. `random_crop=False` means a deterministic
    offset, not offset 0, and a script that reads a stem from the start of the
    file is reading different audio from the one the loader handed it."""
    dataset = make(data_root, random_crop=False)
    starts = [dataset[i][0]["meta"]["crop_start"] for i in range(N_TRIALS)]
    assert any(s > 0 for s in starts), (
        "every offset came back 0, so this fixture cannot detect the bug it "
        "exists for -- lengthen CLIP relative to CHUNK")


def test_a_fixed_set_is_reproducible(data_root):
    """Deterministic it certainly is: two constructions agree exactly."""
    a = [make(data_root, random_crop=False)[i][0]["meta"]["crop_start"]
         for i in range(N_TRIALS)]
    b = [make(data_root, random_crop=False)[i][0]["meta"]["crop_start"]
         for i in range(N_TRIALS)]
    assert a == b


def test_both_directions_share_one_crop(data_root):
    """They are the same mixture asked two ways. Different offsets would make
    the pairing meaningless."""
    dataset = make(data_root, random_crop=False)
    for i in range(N_TRIALS):
        target_side, interferer_side = dataset[i]
        assert target_side["meta"]["crop_start"] == interferer_side["meta"]["crop_start"]


def test_reading_a_stem_at_crop_start_reproduces_the_batch(data_root):
    """THE CONTRACT derive_w_state.py depends on: a stem read at the reported
    offset is sample-for-sample the audio the loader returned."""
    root, _, split = data_root
    dataset = make(data_root, random_crop=False)
    for i in range(N_TRIALS):
        example = dataset[i][0]
        start = example["meta"]["crop_start"]
        path = root / "rendered" / split / example["trial_id"] / "target.wav"
        audio, _ = sf.read(str(path), dtype="float32", start=start, frames=CHUNK)
        assert torch.allclose(example["target"], torch.from_numpy(audio), atol=0)


def test_the_noise_recovery_is_exact_at_the_right_offset(data_root):
    """The arithmetic the synthetic anchors are built on. Recover the noise as
    mixture - target - interferer, put it back at beta = 0.1, and the result
    must contain exactly one tenth of the interferer -- no shifted second copy."""
    root, _, split = data_root
    dataset = make(data_root, random_crop=False)
    example = dataset[0][0]
    start = example["meta"]["crop_start"]
    path = root / "rendered" / split / example["trial_id"] / "interferer.wav"
    interferer, _ = sf.read(str(path), dtype="float32", start=start, frames=CHUNK)
    interferer = torch.from_numpy(interferer)

    noise = example["mixture"] - example["target"] - interferer
    for beta in (1.0, 0.5, 0.1, 0.0):
        rebuilt = example["target"] + beta * interferer + noise
        expected = example["mixture"] - (1.0 - beta) * interferer
        assert torch.allclose(rebuilt, expected, atol=1e-6)


def test_reading_from_zero_is_wrong_when_the_offset_is_not_zero(data_root):
    """The bug itself, as a test. Without this the fix could be reverted and
    every downstream number would look plausible."""
    root, _, split = data_root
    dataset = make(data_root, random_crop=False)
    for i in range(N_TRIALS):
        example = dataset[i][0]
        if example["meta"]["crop_start"] == 0:
            continue
        path = root / "rendered" / split / example["trial_id"] / "target.wav"
        audio, _ = sf.read(str(path), dtype="float32", start=0, frames=CHUNK)
        assert not torch.allclose(example["target"], torch.from_numpy(audio), atol=1e-6)
        return
    pytest.fail("no nonzero offset in the fixture; see the test above")
