"""The turn-down head: a small network on top of a frozen e21 that turns each band of e21's output down by a learned amount, never up.

It learns the oracle of the D1 test (decisions-pending.md 2026-10-28 and 2026-10-29): the gain that brings each band of e21's estimate down to the clean target's energy.
The head reads, per band and per frame, e21's separator state and the log band energies of e21's estimate and of the mixture, and outputs one gain in (0, 1).
It looks at one frame at a time, so it is causal and streams like the rest of the model.
Its per-band trunk copies the shape of the mask estimator (Yu et al., Interspeech 2023, section 2): a channel norm, a 1x1 convolution, a tanh and a 1x1 convolution.
Claude Code wrote this file on 2026-10-29 from the option the author chose.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import build_channel_norm, channel_norm_forward

# Added to every band energy before a log, so a silent band has a finite log energy.
ENERGY_FLOOR = 1e-8
# The log energies are scaled down before they join the 128 separator channels, so the channel norm is not dominated by them.
LOG_ENERGY_SCALE = 0.1
# The output bias at the start: sigmoid(5) is 0.993, so the head starts close to leaving e21's output alone.
START_BIAS = 5.0
DB_PER_NEPER = 20.0 / math.log(10.0)


# this builds one small trunk per band, reading the 128 separator channels plus two log energies
def build_turn_down_head(num_bands, feature_dim, hidden_dim):
    head = nn.Module()
    head.trunks = nn.ModuleList()
    for band_index in range(num_bands):
        output_layer = nn.Conv1d(hidden_dim, 1, 1)
        nn.init.constant_(output_layer.bias, START_BIAS)
        head.trunks.append(nn.ModuleList([build_channel_norm(feature_dim + 2), nn.Conv1d(feature_dim + 2, hidden_dim, 1), nn.Tanh(), output_layer]))
    return head


# this sums a complex spectrogram's energy over the bins of each band, giving (batch, bands, frames)
def band_energy(spectrogram, band_widths):
    power = spectrogram.real ** 2 + spectrogram.imag ** 2
    energies = []
    start = 0
    for width in band_widths:
        energies.append(power[:, start:start + width].sum(dim=1))
        start += width
    return torch.stack(energies, dim=1)


# this gives the head's logits, shaped (batch, bands, frames). The gain is their sigmoid
def turn_down_logits(head, features, estimate_spectrogram, mixture_spectrogram, band_widths):
    """features is the separator output shaped (batch, bands, 128, frames), from bsrnn.extract_spectrogram."""
    estimate_log_energy = torch.log(band_energy(estimate_spectrogram, band_widths) + ENERGY_FLOOR) * LOG_ENERGY_SCALE
    mixture_log_energy = torch.log(band_energy(mixture_spectrogram, band_widths) + ENERGY_FLOOR) * LOG_ENERGY_SCALE
    logits = []
    for band_index, trunk in enumerate(head.trunks):
        band_input = torch.cat([features[:, band_index], estimate_log_energy[:, band_index:band_index + 1], mixture_log_energy[:, band_index:band_index + 1]], dim=1)
        hidden = channel_norm_forward(trunk[0], band_input)
        for layer in trunk[1:]:
            hidden = layer(hidden)
        logits.append(hidden[:, 0])
    return torch.stack(logits, dim=1)


# this multiplies every bin of each band by that band's gain
def apply_band_gains(spectrogram, gains, band_widths):
    widths = torch.tensor(band_widths, device=gains.device)
    per_bin = torch.repeat_interleave(gains, widths, dim=1)
    return spectrogram * per_bin


# this gives the oracle gain in dB, shaped (batch, bands, frames): each band of the estimate brought down to the clean target's energy, only in frames where the target is active
def oracle_gain_db(estimate_spectrogram, target_spectrogram, crop_absent, band_widths, activity_floor_db, gain_floor_db):
    """A frame is active when the target's whole-frame energy is within activity_floor_db of its loudest frame in the crop. A crop the loader marks absent has no active frames.

    Outside active frames the oracle is 0 dB, so the head never learns to blank the target's pauses.
    """
    estimate_energy = band_energy(estimate_spectrogram, band_widths)
    target_energy = band_energy(target_spectrogram, band_widths)
    ratio_db = 10.0 * torch.log10((target_energy + ENERGY_FLOOR) / (estimate_energy + ENERGY_FLOOR))
    gain_db = ratio_db.clamp(min=gain_floor_db, max=0.0)
    frame_energy = target_energy.sum(dim=1)
    loudest = frame_energy.max(dim=1, keepdim=True).values
    active = frame_energy > loudest * 10.0 ** (activity_floor_db / 10.0)
    active = active & ~crop_absent.bool()[:, None]
    return torch.where(active[:, None, :], gain_db, torch.zeros_like(gain_db))


# this is the training loss: the squared error between the head's gain and the oracle gain, in dB, over bands the estimate is not silent in
def turn_down_loss(logits, target_gain_db, estimate_spectrogram, band_widths, audible_floor_db):
    """The head's gain in dB is 20 log10(sigmoid(logit)), computed through logsigmoid so it stays finite. A band counts when its energy is within audible_floor_db of the crop's loudest band and frame."""
    predicted_db = DB_PER_NEPER * F.logsigmoid(logits)
    estimate_energy = band_energy(estimate_spectrogram, band_widths)
    loudest = estimate_energy.flatten(1).max(dim=1).values
    audible = estimate_energy > loudest[:, None, None] * 10.0 ** (audible_floor_db / 10.0)
    squared_error = (predicted_db - target_gain_db) ** 2
    return squared_error[audible].mean()
