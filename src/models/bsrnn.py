"""End-to-end band-split RNN target speaker extractor.

Assembly only -- every component lives in its own module and is wired together
here. Two model classes:

  BSRNN        mixture path only. A causal speech *enhancer*: it will clean up a
               two-voice mixture but has no notion of which voice to keep.
  BSRNN_TFMAP  adds TF-Map conditioning, which is what makes it an *extractor*.

Provenance, four papers:
  band splitting and the dual-path alternation
      Luo & Yu, "Music Source Separation with Band-Split RNN", IEEE/ACM TASLP 2023
  speech adaptation, 512/128 framing, N=128, six layers, mask + residual
      Yu et al., "High Fidelity Speech Enhancement with Band-split RNN",
      Interspeech 2023
  TF-Map conditioning (Spectral Similarity variant, eq. 2)
      Zhang et al., "Multi-Level Speaker Representation for Target Speaker
      Extraction", ICASSP 2025
  the target-absent split loss (not built here)
      CARTSE submission to the REAL-TSE Challenge Track 1

Decisions implemented, all in docs/decisions/decisions-m1.md:

  2026-08-18  STFT 512/128, center=False with symmetric padding, justified
              against our 200-300 ms budget rather than the challenge's 100 ms.
  2026-08-18  band plan is inherited and unjustified in the literature; kept as
              the baseline with five alternatives available for ablation.
  2026-08-18  channel-wise LayerNorm, not the paper's BatchNorm and not
              GroupNorm (which pools over time and leaks the future).
  2026-08-18  sized down deliberately: hidden_dim 192 (not wesep's 256) and
              n_hidden 1 (the paper does not specify depth). 7.16 M parameters
              against the REAL-TSE causal baselines' 25-27 M.
  2026-08-18  lookahead_frames is a 0-16 config knob. It is applied as a shift
              of the feature sequence before the mask head, NOT as a target
              shift: a multiplicative mask cannot move energy in time.
  2026-08-19  TF-Map uses Spectral Similarity (eq. 2). The Embedding Similarity
              variant needs frame-level embeddings of the *live mixture*, which
              is not causal at any acceptable latency.

Deviations from Yu et al. recorded deliberately: their BSRNN-S split (bidirectional
band modelling below 8 kHz, unidirectional above) is omitted as inapplicable at
16 kHz, where Nyquist *is* 8 kHz; their MetricGAN and multi-resolution spectrogram
discriminators are omitted because they optimise PESQ directly.
"""

import torch
import torch.nn as nn

from src.models.bands import band_plan
from src.models.conditioning import TFMap
from src.models.modules import (
    BandSequenceModel,
    BandSplit,
    Estimator,
    SubbandNorm,
    lookahead_shift,
)
from src.models.stft import STFT


class BSRNN(nn.Module):
    """Causal band-split RNN, mixture path only (no speaker conditioning)."""

    def __init__(self, sample_rate=16000, n_fft=512, hop=128, band_segments=None,
                 feature_dim=128, hidden_dim=192, num_repeat=6, mlp_hidden=384,
                 n_hidden=1, lookahead_frames=0, causal=True,
                 residual_branch=True, in_channels=2):
        super().__init__()
        self.lookahead_frames = lookahead_frames
        self.band_widths = band_plan(sample_rate, n_fft, band_segments)

        self.stft         = STFT(n_fft, hop, sample_rate)
        self.split        = BandSplit(self.band_widths)
        self.subband_norm = SubbandNorm(self.band_widths, in_channels, feature_dim, causal)
        self.separator    = BandSequenceModel(feature_dim, hidden_dim, num_repeat, causal)
        self.estimator    = Estimator(self.band_widths, feature_dim, mlp_hidden,
                                      n_hidden, causal, residual_branch)

    def forward(self, mixture):
        """(B, T_samples) -> (B, T_samples)"""
        n = mixture.shape[-1]        # STFT pads, so inverse() cannot infer this

        X   = self.stft(mixture)                        # (B, F, T) complex
        Xri = torch.stack([X.real, X.imag], dim=1)      # (B, 2, F, T) real

        mix_bands  = self.split(X)      # complex -- what the mask multiplies into
        feat_bands = self.split(Xri)    # real    -- what the network consumes

        z = self.subband_norm(feat_bands)                # (B, K, N, T)
        z = self.separator(z)                            # (B, K, N, T)
        z = lookahead_shift(z, self.lookahead_frames)    # no-op at k=0

        return self.stft.inverse(self.estimator(z, mix_bands), n)


class BSRNN_TFMAP(nn.Module):
    """Causal band-split RNN with TF-Map speaker conditioning.

    Identical to BSRNN except that the enrollment's magnitude spectrogram is
    turned into a TF-Map feature and concatenated as a third input channel, so
    in_channels defaults to 3. The mask is still applied to the *complex*
    mixture: TF-Map only enters the network's input, never the thing being masked.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop=128, band_segments=None,
                 feature_dim=128, hidden_dim=192, num_repeat=6, mlp_hidden=384,
                 n_hidden=1, lookahead_frames=0, causal=True,
                 residual_branch=True, in_channels=3, tfmap_scale=16.0):
        super().__init__()
        self.lookahead_frames = lookahead_frames
        self.band_widths = band_plan(sample_rate, n_fft, band_segments)

        self.stft         = STFT(n_fft, hop, sample_rate)
        self.tfmap        = TFMap(scale=tfmap_scale)
        self.split        = BandSplit(self.band_widths)
        self.subband_norm = SubbandNorm(self.band_widths, in_channels, feature_dim, causal)
        self.separator    = BandSequenceModel(feature_dim, hidden_dim, num_repeat, causal)
        self.estimator    = Estimator(self.band_widths, feature_dim, mlp_hidden,
                                      n_hidden, causal, residual_branch)

    def forward(self, mixture, enrollment):
        """(B, T_samples), (B, T_enroll) -> (B, T_samples)"""
        n = mixture.shape[-1]

        X  = self.stft(mixture)                          # (B, F, Tx) complex
        Xe = self.stft(enrollment)                       # (B, F, Te) complex
        tf = self.tfmap(X.abs(), Xe.abs())               # (B, 1, F, Tx)

        Xri      = torch.stack([X.real, X.imag], dim=1)  # (B, 2, F, Tx)
        feats_in = torch.cat([Xri, tf], dim=1)           # (B, 3, F, Tx)

        mix_bands  = self.split(X)          # complex, unconditioned -- for the mask
        feat_bands = self.split(feats_in)   # 3 channels -- for the network

        z = self.subband_norm(feat_bands)
        z = self.separator(z)
        z = lookahead_shift(z, self.lookahead_frames)

        return self.stft.inverse(self.estimator(z, mix_bands), n)
