"""End-to-end band-split RNN target speaker extractor.

Assembly only -- every component lives in its own module and is wired together
here. Two model classes:

  BSRNN_TFMAP          the BASELINE: a causal band-split RNN with TF-Map speaker
                       conditioning, which is what makes it an *extractor*.
  BSRNN_TFMAP_CONTEXT  the EXTENSION: the same, with the cue handed over as its
                       parts (item 1a, `tfmap_parts`) and a frozen speaker
                       embedding modulating the features (item 1c).

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

Decisions, all in docs/decisions/decisions-m1.md:

  2026-08-18  STFT 512/128, center=False, sized to our 200-300 ms budget not the
              challenge's 100 ms; band plan inherited, five ablation candidates;
              channel-wise LayerNorm, not BatchNorm and not GroupNorm (which
              pools over time and leaks the future); sized down deliberately to
              7.19 M against the REAL-TSE causal baselines' 25-27 M (hidden 192,
              n_hidden 1); lookahead_frames 0-16 shifts the FEATURE sequence, not
              the target -- a multiplicative mask cannot move energy in time.
  2026-08-19  TF-Map uses Spectral Similarity (eq. 2); Embedding Similarity
              (eq. 3) needs frame-level embeddings of the live mixture, not
              causal at any acceptable latency.
  2026-10-22  every experiment arm the final baseline and extension do not use
              was removed (cue re-injection D4a, state head D14, presence gate,
              inference-time mask switches, look-ahead > 0, the non-causal
              mode, the no-residual-branch path). None of them is in either
              final checkpoint, so both still load with strict=True. Git
              history before this date has them.

Deliberate omissions from Yu et al.: BSRNN-S's bidirectional sub-8 kHz band
modelling (inapplicable at 16 kHz, where Nyquist IS 8 kHz), and their MetricGAN
and spectrogram discriminators (they optimise PESQ directly).
"""

import torch
import torch.nn as nn

from src.models.bands import band_plan
from src.models.conditioning import ContextFusion, TFMap
from src.models.modules import (
    BandSequenceModel,
    BandSplit,
    Estimator,
    SubbandNorm,
)
from src.models.stft import STFT


class BSRNN_TFMAP(nn.Module):
    """Causal band-split RNN with TF-Map speaker conditioning.

    The enrollment's magnitude spectrogram is turned into a TF-Map feature and
    concatenated with the mixture's real and imaginary parts as extra input
    channels. The mask is still applied to the *complex* mixture: TF-Map only
    enters the network's input, never the thing being masked.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop=128, band_segments=None,
                 feature_dim=128, hidden_dim=192, num_repeat=6, mlp_hidden=384,
                 n_hidden=1, in_channels=3, tfmap_scale=16.0,
                 tfmap_parts=False, tfmap_part_scales=None):
        super().__init__()
        self.band_widths = band_plan(sample_rate, n_fft, band_segments)

        # ITEM 1a. The cue contributes 3 channels instead of 1 (direction,
        # normalised similarity, unexplained residual), so the network input is
        # 2 + 3 = 5 rather than 2 + 1 = 3. DERIVED, never configured: the two
        # numbers cannot disagree without the first 1x1 conv silently reading
        # the wrong channels. `in_channels` is still accepted so pre-2026-09-22
        # configs load unchanged, and is checked against the derivation.
        cue_channels = 3 if tfmap_parts else 1
        derived = 2 + cue_channels
        if not tfmap_parts and in_channels != derived:
            raise ValueError(
                f"in_channels={in_channels} but the TF-Map supplies "
                f"{cue_channels} channel(s) beside real+imag, so it must be "
                f"{derived}. Set tfmap_parts to change the cue's width.")
        self.tfmap_parts = tfmap_parts
        in_channels = derived

        self.stft         = STFT(n_fft, hop, sample_rate)
        self.tfmap        = TFMap(scale=tfmap_scale, return_parts=tfmap_parts,
                                  part_scales=tfmap_part_scales)
        self.split        = BandSplit(self.band_widths)
        self.subband_norm = SubbandNorm(self.band_widths, in_channels, feature_dim)
        self.separator    = BandSequenceModel(feature_dim, hidden_dim, num_repeat)
        self.estimator    = Estimator(self.band_widths, feature_dim, mlp_hidden, n_hidden)

    def _fuse_context(self, z, context):
        """No-op in the base class. See BSRNN_TFMAP_CONTEXT."""
        return z

    def forward(self, mixture, enrollment, *, context=None):
        """(B, T_samples), (B, T_enroll) -> (B, T_samples)"""
        n = mixture.shape[-1]

        X  = self.stft(mixture)                          # (B, F, Tx) complex
        Xe = self.stft(enrollment)                       # (B, F, Te) complex
        tf = self.tfmap(X.abs(), Xe.abs())               # (B, C, F, Tx), C = 1 or 3

        Xri      = torch.stack([X.real, X.imag], dim=1)  # (B, 2, F, Tx)
        feats_in = torch.cat([Xri, tf], dim=1)           # (B, 2 + C, F, Tx)

        mix_bands  = self.split(X)          # complex, unconditioned -- for the mask
        feat_bands = self.split(feats_in)   # 2 + C channels -- for the network

        z = self.subband_norm(feat_bands)
        # Item 1c's insertion point. The base class ignores `context` entirely;
        # BSRNN_TFMAP_CONTEXT overrides `_fuse_context`. A hook rather than a
        # duplicated forward: copying thirty lines to insert one is how two
        # paths drift apart.
        z = self._fuse_context(z, context)
        z = self.separator(z)

        return self.stft.inverse(self.estimator(z, mix_bands), n)   # M * X + R


class BSRNN_TFMAP_CONTEXT(BSRNN_TFMAP):
    """BSRNN_TFMAP plus a frozen speaker embedding modulating the features.

    Item 1c (decisions-m2.md 2026-09-23). A SUBCLASS IN ITS OWN CLASS so that
    the parent is untouched: the baseline reproduces BY CONSTRUCTION rather
    than by a flag being false, and `build_model()` returns the parent unless
    the config asks for `context_embedding`.

    `enrol_embedding` IS REQUIRED AND KEYWORD-ONLY, deliberately. The encoder
    lives outside the model (see src/models/context_encoder.py for why), which
    would otherwise create the failure this project guards against hardest: a
    call site that forgets the embedding running an UNCONDITIONED model that
    still produces audio and still scores. Python raises TypeError instead.
    Nothing has to be remembered, and no runtime check has to be trusted.
    decisions-pending.md E8.

    The encoder itself is NOT a submodule, so it is absent from state_dict(),
    from parameters() and from the optimiser. The only weights this class adds
    are the fusion's projection.
    """

    def __init__(self, *args, embedding_dim=192, **kwargs):
        super().__init__(*args, **kwargs)
        # Constructed LAST so that turning this arm on leaves every other weight
        # in the model, and the dataloader's shuffle, bit-identical to the 1a
        # control.
        self.context = ContextFusion(embedding_dim, self.subband_norm.feature_dim)

    def _fuse_context(self, z, context):
        if context is None:
            raise ValueError(
                "BSRNN_TFMAP_CONTEXT reached the separator with no embedding. "
                "forward() requires enrol_embedding; this can only happen if "
                "forward was bypassed.")
        return self.context(z, context)

    def forward(self, mixture, enrollment, *, enrol_embedding):
        """(B, T), (B, T_enroll), (B, D) -> (B, T).

        `enrol_embedding` comes from ContextEncoder.embed(), computed ONCE per
        utterance. It is not recomputed per chunk, which is what keeps the
        per-frame cost of this arm at exactly zero.
        """
        return super().forward(mixture, enrollment, context=enrol_embedding)
