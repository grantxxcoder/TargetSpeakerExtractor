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
  2026-09-11  optional per-block TF-Map re-injection (decisions-pending.md D4a),
              OFF by default. +37,698 params (+0.52 %) when on, zero-initialised
              gates so the arm starts as exactly the baseline function. NOT
              parameter-matched to its control -- unlike the state head, this one
              does carry a (small) capacity confound.
  2026-09-11  optional auxiliary speaker-state head (decisions-pending.md D14
              piece A), 516 params, OFF by default, training-only and deleted at
              inference -- so with state_head=False this file is arithmetically
              the frozen 2026-08-28 architecture and every earlier checkpoint
              still loads with strict=True.

Deliberate omissions from Yu et al.: BSRNN-S's bidirectional sub-8 kHz band
modelling (inapplicable at 16 kHz, where Nyquist IS 8 kHz), and their MetricGAN
and spectrogram discriminators (they optimise PESQ directly).
"""

import torch
import torch.nn as nn

from src.models.bands import band_plan
from src.models.conditioning import TFMap, TFMapInjector
from src.models.modules import (
    BandSequenceModel,
    BandSplit,
    Estimator,
    SubbandNorm,
    lookahead_shift,
)
from src.models.state_head import AuxStateHead
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

    With `tfmap_inject=True` (D4a) the same TF-Map is additionally re-projected
    and handed back to every separator block. That changes where the cue is
    READ, never what is masked -- the mask still multiplies the unconditioned
    complex mixture, so the arm cannot smuggle the enrollment into the output
    except through the mask the separator predicts.
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop=128, band_segments=None,
                 feature_dim=128, hidden_dim=192, num_repeat=6, mlp_hidden=384,
                 n_hidden=1, lookahead_frames=0, causal=True,
                 residual_branch=True, in_channels=3, tfmap_scale=16.0,
                 state_head=False, state_head_detach=False,
                 tfmap_inject=False, tfmap_gate_init=0.0, mask_floor=0.0,
                 tfmap_parts=False):
        super().__init__()
        self.lookahead_frames = lookahead_frames
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

        # D4a re-projects the cue assuming ONE channel (TFMapInjector asserts
        # C == 1). Parts make it three, so the two arms cannot run together
        # until the injector is generalised -- refused loudly rather than
        # tripping an assert six frames deep in a Kaggle log.
        if tfmap_parts and tfmap_inject:
            raise NotImplementedError(
                "tfmap_parts and tfmap_inject cannot both be on: the injector "
                "projects a 1-channel cue. decisions-pending.md D4 says run "
                "D4a before D4b; item 1b is ranked after this arm for the same "
                "reason.")

        self.stft         = STFT(n_fft, hop, sample_rate)
        self.tfmap        = TFMap(scale=tfmap_scale, return_parts=tfmap_parts)
        self.split        = BandSplit(self.band_widths)
        self.subband_norm = SubbandNorm(self.band_widths, in_channels, feature_dim, causal)
        self.separator    = BandSequenceModel(feature_dim, hidden_dim, num_repeat, causal)
        self.estimator    = Estimator(self.band_widths, feature_dim, mlp_hidden,
                                      n_hidden, causal, residual_branch,
                                      mask_floor=mask_floor)
        # D4a, decisions-pending.md D4. OFF by default: with tfmap_inject=False
        # the cue enters once as an input channel, which is the architecture
        # frozen on 2026-08-28 as the baseline of record.
        self.tfmap_inject = (
            TFMapInjector(self.band_widths, feature_dim, num_repeat, causal,
                          gate_init=tfmap_gate_init)
            if tfmap_inject else None)

        # Head A, decisions-pending.md D14. OFF by default, and the attribute is
        # None rather than absent so a caller can test for it without try/except.
        # Constructed LAST and under a forked RNG (see AuxStateHead), so turning
        # it on leaves every other weight in this model, and the dataloader's
        # shuffle, bit-identical to the baseline.
        self.state_head = (AuxStateHead(feature_dim, detach_features=state_head_detach)
                           if state_head else None)

    def forward(self, mixture, enrollment, return_state=False,
                return_mask=False):
        """(B, T_samples), (B, T_enroll) -> (B, T_samples)

        With `return_state=True`, returns `(waveform, state_logits)` where the
        logits are `(B, 4, T_frames)`. The default return type is unchanged, so
        every existing caller -- eval, latency, the streaming runner -- is
        untouched and the head is invisible to them.
        """
        n = mixture.shape[-1]

        X  = self.stft(mixture)                          # (B, F, Tx) complex
        Xe = self.stft(enrollment)                       # (B, F, Te) complex
        tf = self.tfmap(X.abs(), Xe.abs())               # (B, C, F, Tx), C = 1 or 3

        Xri      = torch.stack([X.real, X.imag], dim=1)  # (B, 2, F, Tx)
        feats_in = torch.cat([Xri, tf], dim=1)           # (B, 2 + C, F, Tx)

        mix_bands  = self.split(X)          # complex, unconditioned -- for the mask
        feat_bands = self.split(feats_in)   # 2 + C channels -- for the network

        # D4a: the same TF-Map, band-split again and projected on its own, handed
        # back to every block inside the stack. None when the arm is off, and
        # BandSequenceModel then skips the addition entirely.
        cue = gates = None
        if self.tfmap_inject is not None:
            cue = self.tfmap_inject(self.split(tf))
            gates = self.tfmap_inject.gates

        z = self.subband_norm(feat_bands)
        z = self.separator(z, cue=cue, gates=gates)

        # BEFORE lookahead_shift, deliberately. The shift moves frame t's
        # features to position t-k so the MASK for t is built from a state that
        # has seen t+k; the state LABEL for t is still about t. Reading the head
        # off the shifted tensor would train it on frame t's label against frame
        # t+k's features -- invisible at the current lookahead_frames: 0, and a
        # silent k-frame misalignment the moment that config key is raised.
        state_logits = None if self.state_head is None else self.state_head(z)

        z = lookahead_shift(z, self.lookahead_frames)
        # D17: ask the estimator to keep the mask in the graph BEFORE the call,
        # and clear the flag after, so a caller that does not want the mask
        # never pays for a retained tensor.
        self.estimator.keep_mask_grad = bool(return_mask)
        waveform = self.stft.inverse(self.estimator(z, mix_bands), n)
        mask = self.estimator.last_mask_grad if return_mask else None
        self.estimator.keep_mask_grad = False
        self.estimator.last_mask_grad = None

        if return_mask and not return_state:
            return waveform, mask
        if return_mask and return_state:
            # Three-tuple ONLY when both are asked for, so neither existing
            # caller's return shape changes. Same guard as the state-only path.
            if state_logits is None:
                raise RuntimeError(
                    "return_state=True but this model was built without head A. "
                    "Construct BSRNN_TFMAP(state_head=True), or set "
                    "model.state_head: true in the config.")
            return waveform, state_logits, mask

        if not return_state:
            return waveform
        if state_logits is None:
            # Loud, because the alternative is a training run that quietly adds
            # nothing and is then reported as head A having had no effect.
            raise RuntimeError(
                "return_state=True but this model was built without head A. "
                "Construct BSRNN_TFMAP(state_head=True), or set "
                "model.state_head: true in the config.")
        return waveform, state_logits
