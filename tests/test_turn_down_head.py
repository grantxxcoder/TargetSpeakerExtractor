"""Tests for the turn-down head (src/models/turn_down_head.py), on synthetic tensors. Nothing here updates a weight.

Claude Code wrote these tests on 2026-10-29.
"""

import torch

from src.models.bands import band_plan
from src.models.turn_down_head import apply_band_gains, build_turn_down_head, oracle_gain_db, turn_down_logits, turn_down_loss

BAND_WIDTHS = band_plan(16000, 512)
NUM_BINS = 257
NUM_FRAMES = 50


# this makes a random complex spectrogram shaped (batch, bins, frames)
def random_spectrogram(batch_size):
    return torch.complex(torch.randn(batch_size, NUM_BINS, NUM_FRAMES), torch.randn(batch_size, NUM_BINS, NUM_FRAMES))


# this runs a fresh head on random inputs and gives its logits
def fresh_logits():
    torch.manual_seed(0)
    head = build_turn_down_head(len(BAND_WIDTHS), 128, 64)
    features = torch.randn(2, len(BAND_WIDTHS), 128, NUM_FRAMES)
    return head, turn_down_logits(head, features, random_spectrogram(2), random_spectrogram(2), BAND_WIDTHS)


def test_fresh_head_leaves_the_estimate_almost_alone():
    head, logits = fresh_logits()
    assert logits.shape == (2, len(BAND_WIDTHS), NUM_FRAMES)
    assert torch.sigmoid(logits).min() > 0.98


def test_unit_gains_change_nothing():
    spectrogram = random_spectrogram(2)
    gains = torch.ones(2, len(BAND_WIDTHS), NUM_FRAMES)
    assert torch.equal(apply_band_gains(spectrogram, gains, BAND_WIDTHS), spectrogram)


def test_oracle_only_turns_down_and_leaves_absent_crops_alone():
    estimate = random_spectrogram(2) * 3.0
    target = random_spectrogram(2)
    crop_absent = torch.tensor([False, True])
    gain_db = oracle_gain_db(estimate, target, crop_absent, BAND_WIDTHS, -40.0, -30.0)
    assert gain_db.max() <= 0.0
    assert gain_db.min() >= -30.0
    # The second crop is marked absent, so the oracle leaves every band of it at 0 dB.
    assert torch.equal(gain_db[1], torch.zeros_like(gain_db[1]))
    # The first crop's estimate is louder than its target, so the oracle turns it down.
    assert gain_db[0].mean() < -5.0


def test_loss_trains_only_the_head():
    head, logits = fresh_logits()
    target_gain_db = torch.full_like(logits, -6.0)
    loss = turn_down_loss(logits, target_gain_db, random_spectrogram(2), BAND_WIDTHS, -60.0)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in head.parameters():
        assert parameter.grad is not None
