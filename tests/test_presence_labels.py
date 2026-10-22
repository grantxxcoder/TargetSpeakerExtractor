"""Per-frame presence labels v_t from the loader (src/data/dataset_loader.py).

What the presence gate's supervised term rests on: one label per model frame,
on the crop's own timeline, for the direction's own target. A one-frame shift
here would train the gate against the wrong frames and nothing downstream
would notice. docs/extra/variational-presence-gate.md, eq. (12).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.dataset_loader import TrialDataset, collate_pairs  # noqa: E402
from src.models.stft import STFT  # noqa: E402

SR, N_FFT, HOP = 16000, 512, 128
CHUNK_S = 0.5
CHUNK = int(CHUNK_S * SR)
CLIP = CHUNK * 6
N_TRIALS = 8
SPANS = {"target": [(0.1234, 0.9876), (1.5, 2.7777)], "interferer": [(0.3333, 2.2222)]}


@pytest.fixture
def data_root(tmp_path):
    split = "unit"
    rows, index = [], []
    for i in range(N_TRIALS):
        tid = f"t-{i:03d}"
        d = tmp_path / "rendered" / split / tid
        d.mkdir(parents=True)
        for name in ("target", "interferer", "mixture", "enrollment",
                     "interferer_enrollment"):
            sf.write(d / f"{name}.wav", np.full(CLIP, 0.1, np.float32), SR, subtype="FLOAT")
        rows.append({"trial_id": tid, "condition": "both", "target_absent": 0,
                     "sir_db": 3.0, "snr_db": 12.0, "overlap_achieved": 0.5,
                     "regime": "base", "same_gender": 0.0,
                     "interferer_enrollment_phantom": 0})
        if i < N_TRIALS - 1:      # the last trial has no labels, like an AMI row
            fmt = lambda spans: "|".join(f"{a:.4f}:{b:.4f}" for a, b in spans)
            index.append({"trial_id": tid, "n_samples": CLIP,
                          "target_spans": fmt(SPANS["target"]),
                          # trailing separator, as in the real index
                          "interferer_spans": fmt(SPANS["interferer"]) + "|"})
    manifest = tmp_path / "unit.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    state_index = tmp_path / "state_unit.csv"
    pd.DataFrame(index).to_csv(state_index, index=False)
    return tmp_path, manifest, split, state_index


def make(data_root, labels=True, **kw):
    root, manifest, split, state_index = data_root
    opts = dict(manifest_csv=manifest, data_root=root, split=split, chunk_s=CHUNK_S,
                sample_rate=SR, seed=42, both_directions=True, random_crop=False)
    if labels:
        opts.update(state_index=state_index, label_n_fft=N_FFT, label_hop=HOP)
    opts.update(kw)
    return TrialDataset(**opts)


def expected(spans, crop_start):
    """Frame t is active if its hop step [t*hop, (t+1)*hop) of the crop overlaps
    a span, after cutting the span at the crop's edges."""
    n = STFT(N_FFT, HOP, SR)(torch.zeros(1, CHUNK)).shape[-1]
    start, hop_s = crop_start / SR, HOP / SR
    v = np.zeros(n, np.float32)
    for a, b in spans:
        a, b = max(a - start, 0.0), min(b - start, CHUNK_S)
        for t in range(n):
            if a < (t + 1) * hop_s and b > t * hop_s:
                v[t] = 1.0
    return v


def test_no_index_means_no_labels(data_root):
    assert "v" not in make(data_root, labels=False)[0][0]


def test_one_label_per_model_frame(data_root):
    frames = STFT(N_FFT, HOP, SR)(torch.zeros(1, CHUNK)).shape[-1]
    assert make(data_root)[0][0]["v"].shape == (frames,)


def test_labels_follow_the_crop_and_the_direction(data_root):
    dataset = make(data_root)
    for i in range(N_TRIALS - 1):
        for example in dataset[i]:
            start = example["meta"]["crop_start"]
            assert example["v_valid"]
            assert np.array_equal(example["v"].numpy(),
                                  expected(SPANS[example["direction"]], start))


def test_frames_past_the_crop_are_zero(data_root, tmp_path):
    root, manifest, split, _ = data_root
    whole = tmp_path / "whole.csv"
    pd.DataFrame([{"trial_id": "t-000", "n_samples": CLIP, "target_spans": f"0.0:{CLIP / SR}",
                   "interferer_spans": ""}]).to_csv(whole, index=False)
    v = make(data_root, state_index=whole)[0][0]["v"]
    inside = int(np.ceil(CHUNK / HOP))           # frames whose hop step starts inside
    assert v[:inside].eq(1).all() and v[inside:].eq(0).all()


def test_trial_without_labels_is_flagged(data_root):
    example = make(data_root)[N_TRIALS - 1][0]
    assert not example["v_valid"] and example["v"].sum() == 0


def test_labels_collate_into_a_batch(data_root):
    dataset = make(data_root)
    batch = collate_pairs([dataset[0], dataset[N_TRIALS - 1]])
    assert batch["v"].shape[0] == 4 and batch["v_valid"].tolist() == [True, True, False, False]
