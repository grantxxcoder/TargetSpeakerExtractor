"""The whole extractor: the baseline, and the extension built on top of it.

Both are a causal band-split RNN (Luo & Yu, "Music Source Separation with Band-Split RNN", TASLP 2023; Yu et al., "High Fidelity Speech Enhancement with Band-split RNN", Interspeech 2023),
told who to extract by the TF-Map cue (Zhang et al., "Multi-Level Speaker Representation for Target Speaker Extraction", ICASSP 2025).
The extension hands the cue over as its three parts and multiplies in a frozen speaker embedding (decisions-m2.md 2026-09-22 and 2026-09-23).
The architecture decisions are in decisions-m1.md 2026-08-18 and 2026-08-19.
"""

import torch
import torch.nn as nn

from src.models.bands import band_plan
from src.models.conditioning import build_context_fusion, context_fusion_forward, tfmap
from src.models.modules import band_sequence_model_forward, build_band_sequence_model, build_estimator, build_subband_norm, estimator_forward, split_bands, subband_norm_forward
from src.models.stft import build_stft, stft_forward, stft_inverse


# this builds the baseline, or the extension when embedding_dim is given
def build_bsrnn_tfmap(sample_rate, n_fft, hop, feature_dim, hidden_dim, num_repeat, mlp_hidden, tfmap_scale, tfmap_parts, tfmap_part_scales, embedding_dim):
    """Return a plain nn.Module holding the model's weights. Run it with extract().

    The baseline passes tfmap_parts=False, no part scales and embedding_dim=None. The extension passes tfmap_parts=True, its three part scales and embedding_dim=192.
    """
    model = nn.Module()
    model.band_widths = band_plan(sample_rate, n_fft)
    # The network sees the mixture's real and imaginary parts plus the cue: one cue channel for the baseline, three for the extension.
    if tfmap_parts:
        in_channels = 2 + 3
    else:
        in_channels = 2 + 1
    model.tfmap_scale = tfmap_scale
    model.tfmap_parts = tfmap_parts
    model.tfmap_part_scales = tfmap_part_scales

    # Built in this order so a fixed seed draws the same starting weights the trained models started from.
    model.stft = build_stft(n_fft, hop, sample_rate)
    model.subband_norm = build_subband_norm(model.band_widths, in_channels, feature_dim)
    model.separator = build_band_sequence_model(feature_dim, hidden_dim, num_repeat)
    model.estimator = build_estimator(model.band_widths, feature_dim, mlp_hidden)
    # The extension's fusion is built last, so every other weight starts exactly as it does in the baseline.
    if embedding_dim is not None:
        model.context = build_context_fusion(embedding_dim, feature_dim)
    return model


# this is where the extension multiplies in the speaker embedding, while the baseline passes the features straight through
def fuse_context(model, features, enrol_embedding):
    """A mismatch either way is an error, because an extension run without its embedding would still produce audio and still score.

    Used by both extract() and the streaming code, so the two can never drift apart.
    """
    if not hasattr(model, "context"):
        if enrol_embedding is not None:
            raise ValueError("The baseline has no speaker-embedding fusion, but it was given an embedding.")
        return features
    if enrol_embedding is None:
        raise ValueError("The extension needs the enrolled speaker's embedding, but none was given.")
    return context_fusion_forward(model.context, features, enrol_embedding)


# this runs the model on whole clips: mixtures shaped (batch, samples) and enrolments shaped (batch, enrolment samples), giving estimates shaped like the mixtures
def extract(model, mixture, enrollment, enrol_embedding):
    """enrol_embedding is None for the baseline. For the extension it is the (batch, 192) embedding from context_encoder.embed_enrolment, computed once per utterance."""
    estimate_spectrogram, features, mixture_spectrogram = extract_spectrogram(model, mixture, enrollment, enrol_embedding)
    return stft_inverse(model.stft, estimate_spectrogram, mixture.shape[-1])


# this is extract() without the inverse STFT. It also gives the separator's features, shaped (batch, bands, features, frames), and the mixture's spectrogram, which the turn-down head reads (decisions-pending.md 2026-10-29)
def extract_spectrogram(model, mixture, enrollment, enrol_embedding):
    """Split out of extract() by Claude Code on 2026-10-29. extract() gives the same output as before."""
    mixture_spectrogram = stft_forward(model.stft, mixture)
    enrollment_spectrogram = stft_forward(model.stft, enrollment)
    cue = tfmap(mixture_spectrogram.abs(), enrollment_spectrogram.abs(), model.tfmap_scale, model.tfmap_parts, model.tfmap_part_scales)

    # The network's input is the mixture's real and imaginary parts, with the cue stacked on as extra channels.
    real_and_imaginary = torch.stack([mixture_spectrogram.real, mixture_spectrogram.imag], dim=1)
    network_input = torch.cat([real_and_imaginary, cue], dim=1)

    # The mask is applied to the complex mixture itself, so the cue only ever enters the network's input.
    mixture_bands = split_bands(mixture_spectrogram, model.band_widths)
    input_bands = split_bands(network_input, model.band_widths)

    features = subband_norm_forward(model.subband_norm, input_bands)
    features = fuse_context(model, features, enrol_embedding)
    features = band_sequence_model_forward(model.separator, features)
    estimate_spectrogram = estimator_forward(model.estimator, features, mixture_bands)
    return estimate_spectrogram, features, mixture_spectrogram
