"""The short-time Fourier transform the extractor works in, and its inverse.

Every function takes the holder made by build_stft, which carries the frame size, the hop and the Hann window.

This file was debugged by Claude Code and errors fixed.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# this builds the holder for the STFT settings and the window
def build_stft(n_fft=512, hop_length=128, sample_rate=16000):
    """Hold the frame size, hop and Hann window that every STFT call here uses.

    A plain nn.Module is used only as a container, so the window moves with model.to(device). The window is not saved in checkpoints.
    """
    stft = nn.Module()
    stft.n_fft = n_fft
    stft.hop_length = hop_length
    stft.sample_rate = sample_rate
    # The signal is padded on the left by one window minus one hop, so the first frame finishes after just one hop of real audio.
    stft.padding = n_fft - hop_length
    stft.register_buffer("window", torch.hann_window(n_fft), persistent=False)
    return stft


# this gives the delay the framing adds, in milliseconds
def stft_latency_ms(stft):
    """The delay the framing adds: one full window to fill plus one hop to emit.

    Say which convention you use when quoting it. CARTSE counts the window minus a hop, so their 24 ms and our 40 ms describe the same framing.
    """
    latency_samples = stft.n_fft + stft.hop_length
    return 1000 * latency_samples / stft.sample_rate


# this counts how many frames a signal of num_samples samples produces
def stft_frame_count(stft, num_samples):
    # Both ends get the same padding. Without it on the right, the last samples would sit where the overlap-add has faded to almost nothing.
    return math.ceil((num_samples + 2 * stft.padding - stft.n_fft) / stft.hop_length) + 1


# this turns a batch of waveforms, shaped (batch, samples), into complex spectrograms shaped (batch, frequency bins, frames)
def stft_forward(stft, waveform):
    num_samples = waveform.shape[-1]
    num_frames = stft_frame_count(stft, num_samples)
    # Pad the right just enough for the last frame to be complete.
    right_padding = (num_frames - 1) * stft.hop_length + stft.n_fft - (num_samples + stft.padding)
    padded = F.pad(waveform, (stft.padding, right_padding))
    return torch.stft(padded, stft.n_fft, stft.hop_length, stft.n_fft, stft.window, center=False, return_complex=True)


# this turns complex spectrograms back into waveforms of the given length
def stft_inverse(stft, spectrogram, length):
    """Inverse STFT by weighted overlap-add, divided by the summed squared window (Griffin & Lim, 1984).

    The left padding is cut off and the result is trimmed to length samples.
    """
    frames = torch.fft.irfft(spectrogram, n=stft.n_fft, dim=1) * stft.window[None, :, None]
    batch_size, frame_length, num_frames = frames.shape
    total_length = (num_frames - 1) * stft.hop_length + stft.n_fft
    # F.fold lays each frame down one hop after the previous one and adds them up where they overlap.
    summed_frames = F.fold(frames, (1, total_length), (1, stft.n_fft), stride=(1, stft.hop_length)).view(batch_size, total_length)
    squared_window = stft.window.pow(2)[None, :, None].expand(1, -1, num_frames)
    window_envelope = F.fold(squared_window, (1, total_length), (1, stft.n_fft), stride=(1, stft.hop_length)).view(total_length)
    # The tiny floor stops a division by zero at the very edges, where the envelope is zero.
    waveform = summed_frames / window_envelope.clamp_min(1e-11)
    return waveform[:, stft.padding:stft.padding + length]
