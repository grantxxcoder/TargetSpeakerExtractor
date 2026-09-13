import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.modules import ChannelWiseLayerNorm


class TFMap(nn.Module):
    """Spectral-level speaker cue.
    
    Zhang et al., "Multi-Level Speaker Representation for Target Speaker
    Extraction", ICASSP 2025, sec II-A, eq (1)-(2):
        F_tfmap = B_e H ,   H = Softmax(B_e^T B_x)

    B_e is the enrollment magnitude spectrogram used directly as NMF-style basis
    vectors -- every enrollment frame is a basis vector rather than a learned
    dictionary. H weights them per mixture frame by cosine similarity.

    Spectral Similarity variant (eq 2). The Embedding Similarity variant (eq 3)
    needs a speaker encoder; this needs none, so it adds no parameters.

    Causal: each mixture frame attends only over the enrollment, which is fixed
    and fully available before the stream starts.
    """
    def __init__(self, eps=1e-8, scale=None):
        super().__init__()
        self.eps = eps
        # None -> sqrt(F) at forward time. `is None`, never `or`: scale=0.0 is a
        # legitimate ablation arm (uniform weights, i.e. the pre-2026-08-25
        # behaviour) and `or` would silently swap it for sqrt(F).
        self.scale = scale
     
    def forward(self, mix_mag, enroll_mag):
        """mix_mag (B,F,Tx), enroll_mag (B,F,Te) -> (B,1,F,Tx)"""
        # cosine similarity: normalise each frame over frequency (shape, not loudness)
        bx = F.normalize(mix_mag,    p=2, dim=1, eps=self.eps)
        be = F.normalize(enroll_mag, p=2, dim=1, eps=self.eps)
        
        sim = torch.matmul(bx.transpose(1, 2), be)                # (B, Tx, Te)

        # SCALE THE LOGITS. Without it the softmax averages instead of
        # selecting and the cue goes static. Softmax compares logits by
        # DIFFERENCE, and F.normalize (needed: shape, not loudness) bounds every
        # cosine to [-1, 1], so the best enrollment frame can outweigh the worst
        # by at most e^1 = 2.7x -- nothing across Te ~ 628 frames. MEASURED
        # 2026-08-25: 619.6 of 628 frames effectively used, top frame 0.22 % vs
        # 0.16 % for a flat average; the cue became the long-term mean spectrum,
        # varying 4.7 %, and the model ignored it. Zhang et al. eq (2) is written
        # on UN-normalised products (measured 0..932), which select sharply on
        # their own; normalising removed that range and this restores it.
        # sqrt(F) ~ 16 at F=257: top 50 frames then carry ~59 %, variation ~39 %.
        scale = mix_mag.shape[1] ** 0.5 if self.scale is None else self.scale
        sim = sim * scale
        h   = torch.softmax(sim, dim=-1)                          # over enrollment
        tf  = torch.matmul(h, be.transpose(1, 2)).transpose(1, 2) # B_e H -> (B,F,Tx)
        
        # energy recovery: project the mixture magnitude onto the unit TF-Map frame
        tf = tf / tf.norm(dim=1, keepdim=True).clamp_min(self.eps)
        tf = (mix_mag * tf).sum(dim=1, keepdim=True) * tf
        return tf.unsqueeze(1)


class TFMapInjector(nn.Module):
    """Re-present the TF-Map to every separator block. decisions-pending.md D4a.

    THE PROBLEM IT ADDRESSES, IN PLAIN WORDS. Today the speaker cue enters the
    model once, as a third input channel, and is projected through a single 1x1
    conv. From there it has to survive six BSNet blocks -- twelve LSTMs -- of
    residual mixing before it can influence the mask. Nothing reminds the
    network who it was asked for. This module hands the same cue back at the
    input of every block, so the cue's credit path to the loss is one block
    deep instead of six.

        TF-Map (B, 1, F, T)
            -> band-split, exactly as the mixture is split
            -> per-band LayerNorm + 1x1 conv to feature_dim   ONE projection
            -> c, (B, K, N, T), the cue in the separator's own feature space
            -> block i sees  z + gate[i, band] * c

    WHY ONE SHARED PROJECTION AND NOT SIX. D4a's wording ("project it per band
    and add it into each of the six blocks") admits both. Shared is the honest
    test of the stated hypothesis -- "the cue is fine, the network is losing
    it" -- because it re-presents the IDENTICAL cue at every depth and adds no
    expressiveness that could explain a gain on its own. Six separate
    projections would cost 222 k parameters (+3.1 %) and reintroduce exactly the
    capacity confound that makes an architecture arm hard to read. This costs
    37,698 (+0.52 %), which is small but NOT zero, and must be stated: unlike
    head A, this arm is not parameter-matched to its control.

    WHY A PER-BLOCK, PER-BAND GATE. Two reasons, one of them the real one.
      * It lets each block choose its own dose of cue, which is the only thing
        the shared projection gives up, at 6 x 32 = 192 parameters.
      * IT IS A DIAGNOSTIC. The gates are a direct, readable answer to the
        question this whole arm exists to ask: if they stay near zero, the
        network does not want more cue, and D3a's 2026-08-30 conclusion stands
        with a second independent measurement. Log them per epoch.
    Per band rather than one scalar per block because the band plan is the one
    axis M5 chose deliberately -- speech energy sits low, artefacts sit high,
    and there is no reason the cue should be equally useful in both.

    ZERO-INIT GATES: the arm starts as EXACTLY the baseline function. At
    gate = 0 the injected term vanishes and the model computes what
    `bsrnn_baseline.yaml` computes, so any divergence is learned rather than an
    initialisation artefact, and the two runs are comparable at step 0. Cite
    ReZero, Bachlechner et al., UAI 2021, for zero-initialised residual gates
    (there for depth, here for interpretability). Consequence to know: the
    projection weights get no gradient on the first step, because the gradient
    reaching `c` is scaled by the gate. The gates move first -- they sit one
    multiply from the loss -- and the projection follows. `gate_init: 1.0` is
    the "inject from step 0" arm if that ordering turns out to matter.

    Causal and latency-free. The TF-Map is already causal per frame (each
    mixture frame attends only over the fixed enrollment) and everything here is
    a 1x1 convolution over time, so no frame ever sees a later frame. The
    streaming budget is unchanged.

    Provenance. Per-layer conditioning of a separator on a speaker cue:
    FiLM, Perez et al., AAAI 2018; in TSE, the multiplicative adaptation of
    Delcroix et al., "Improving speaker discrimination of target speech
    extraction with time-domain SpeakerBeam", ICASSP 2020. BORROWED WITH A
    DIFFERENCE, and the difference is the point: FiLM and SpeakerBeam condition
    on a LEARNED fixed-length speaker embedding, which can memorise the training
    voices. The TF-Map is parameter-free and is recomputed from the enrollment
    at inference, so nothing here can memorise a speaker -- only the projection
    into feature space is learned. That is why D4a was ranked before D4b.
    """

    def __init__(self, band_widths, feature_dim, num_blocks, causal=True,
                 gate_init=0.0, init_seed=20260911, isolate_rng=True):
        super().__init__()
        self.band_widths = list(band_widths)
        self.num_blocks = num_blocks

        # CONSTRUCTED UNDER A FORKED RNG for the reason measured on 2026-09-11
        # (decisions-pending.md D14): any module added before the first batch is
        # drawn advances the global RNG, which changes the dataloader's shuffle
        # and therefore which trial `drop_last` discards -- so an arm and its
        # control silently stop seeing the same data in the same order. Forked,
        # every other weight in the model and the whole data order stay
        # bit-identical to the baseline, and the ONLY difference is this module.
        # The narrow fix; the general one was declined.
        if isolate_rng:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                self._build(feature_dim, causal, gate_init)
        else:
            self._build(feature_dim, causal, gate_init)

    def _build(self, feature_dim, causal, gate_init):
        # Mirrors SubbandNorm's per-band block exactly -- same norm choice, same
        # 1x1 projection. Deliberate: the TF-Map's magnitudes are energy-
        # recovered (TFMap.forward rescales by the mixture's projection onto the
        # cue), so they are NOT on the scale of the normalised residual stream.
        # Adding a raw projection of them into it would be a scale mismatch
        # dressed up as conditioning.
        self.blocks = nn.ModuleList()
        for bw in self.band_widths:
            norm = ChannelWiseLayerNorm(bw) if causal else nn.GroupNorm(1, bw)
            self.blocks.append(nn.Sequential(norm, nn.Conv1d(bw, feature_dim, 1)))
        self.gates = nn.Parameter(
            torch.full((self.num_blocks, len(self.band_widths)), float(gate_init)))

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def forward(self, tf_bands):
        """list of K x (B, 1, BW, T) -> (B, K, N, T), the cue in feature space.

        Computed ONCE per forward and re-used by every block. The per-block
        difference is the gate, not the projection, so this costs one pass over
        32 small convolutions rather than six.
        """
        out = []
        for block, band in zip(self.blocks, tf_bands):
            B, C, BW, T = band.shape
            assert C == 1, f"the TF-Map is one channel, got {C}"
            # reshape, NOT view: BandSplit uses narrow(), so `band` is a
            # non-contiguous slice of the full TF-Map.
            out.append(block(band.reshape(B, BW, T)))
        return torch.stack(out, dim=1)
