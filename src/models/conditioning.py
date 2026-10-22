import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, eps=1e-8, scale=None, return_parts=False,
                 part_scales=None):
        super().__init__()
        self.eps = eps
        # PER-CHANNEL SCALES for the parts, DERIVED not chosen:
        # scripts/derive_cue_scales.py matches each channel's RMS to the RMS of
        # the real and imaginary channels it is concatenated with.
        #
        # WHY IT IS NEEDED, measured 2026-09-22. The decomposition spans 30x --
        # `direction` is unit-norm over 257 bins so its per-bin RMS is exactly
        # 1/sqrt(F) = 0.0624, while `match_fraction` is a cosine in [0,1] --
        # and SubbandNorm standardises all five channels JOINTLY within a band,
        # preserving their relative sizes rather than equalising them. Result at
        # init: `direction` drove 0.5 % of the separator's input against the old
        # single cue's 50.9 %.
        #
        # AND THE NETWORK DOES NOT FIX IT FAST ENOUGH. On the 2-epoch probe the
        # per-channel LayerNorm gain for `direction` moved 1.0000 -> 1.1687:
        # the right direction, ~17 % of the way against the ~280 % needed, while
        # `match_fraction` grew 135.3 % -> 185.1 %. An optimisation problem, so
        # the fix is conditioning the input. decisions-m2.md 2026-09-22.
        #
        # None => all ones => the un-scaled behaviour of the first 1a probe, kept
        # so that run reproduces.
        self.register_buffer(
            "part_scales",
            torch.ones(3) if part_scales is None
            else torch.tensor([float(v) for v in part_scales]),
            persistent=False)
        # RETURN THE PARTS, NOT THE PRODUCT (ranked-next-steps.md item 1a).
        # False reproduces every run up to 2026-09-22 bit-identically; the whole
        # change is confined to the last four lines of forward(). See the
        # "WHAT return_parts CHANGES" note there for the algebra.
        self.return_parts = return_parts
        # None -> sqrt(F) at forward time. `is None`, never `or`: scale=0.0 is a
        # legitimate ablation arm (uniform weights, i.e. the pre-2026-08-25
        # behaviour) and `or` would silently swap it for sqrt(F).
        self.scale = scale
     
    def forward(self, mix_mag, enroll_mag):
        """mix_mag (B,F,Tx), enroll_mag (B,F,Te) -> (B,C,F,Tx)

        C = 1 (the product matched_level * direction) by default, or 3 with
        `return_parts`: the direction, the match fraction, and what is left over.

        B = batch, F = 257 frequency bins, Tx = mixture frames,
        Te ~ 628 enrollment frames. EVERY per-frame quantity below is a
        reduction over dim=1, the frequency axis.
        """
        # STEP 1. Divide each frame by its own length, so what survives is the
        # SHAPE of the spectrum and not how loud it was. This is the "scale
        # invariant" in scale-invariant cosine similarity.
        mix_shape   = F.normalize(mix_mag,    p=2, dim=1, eps=self.eps)  # (B,F,Tx)
        enrol_shape = F.normalize(enroll_mag, p=2, dim=1, eps=self.eps)  # (B,F,Te)

        # STEP 2. Every mixture frame against every enrollment frame. The matmul
        # contracts over frequency, and both sides are unit vectors, so each
        # entry is a cosine: one library of Te scores per mixture frame.
        similarity = torch.matmul(mix_shape.transpose(1, 2), enrol_shape)  # (B,Tx,Te)

        # STEP 3. SCALE THE LOGITS. Without it the softmax averages instead of
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
        sharpness = mix_mag.shape[1] ** 0.5 if self.scale is None else self.scale
        enrol_weights = torch.softmax(similarity * sharpness, dim=-1)      # (B,Tx,Te)

        # STEP 4. Blend the enrollment frames by those weights. ONE TEMPLATE PER
        # MIXTURE FRAME -- the blend is recomputed every frame, it is not a
        # single fixed template for the clip.
        template = torch.matmul(enrol_weights, enrol_shape.transpose(1, 2)).transpose(1, 2)  # (B,F,Tx)

        # STEP 5. A weighted average of unit vectors is SHORTER than 1, so
        # renormalise. Now it is a pure direction: which way the target's
        # spectrum points, with no size of its own.
        direction = template / template.norm(dim=1, keepdim=True).clamp_min(self.eps)

        # STEP 6. THE PROJECTION. Elementwise multiply and sum over frequency is
        # the dot product <x_t, direction_t>: how far the mixture frame reaches
        # along that direction. One number per frame -- the length of the frame's
        # shadow on the template's line.
        #
        # NOTE IT USES mix_mag, NOT mix_shape. That is the whole of item 1a:
        #
        #     matched_level = <x, dir> = ||x|| * cos(theta)
        #                                 ----   ----------
        #                                 loud    match
        #
        # Had it used mix_shape, this would already BE the match fraction.
        matched_level = (mix_mag * direction).sum(dim=1, keepdim=True)      # (B,1,Tx)

        if not self.return_parts:
            # The shadow reattached as a vector. Loudness and match fused into
            # one channel, inseparable downstream.
            return (matched_level * direction).unsqueeze(1)                 # (B,1,F,Tx)

        # WHAT return_parts CHANGES, AND WHY. ranked-next-steps.md item 1a.
        #
        # MEASURED 2026-09-21: corr(matched_level, frame loudness) = 0.990.
        # That is ARITHMETIC, not a training failure -- no change to the data
        # distribution moves it. And the network cannot undo it: it is never
        # given ||x|| as a quantity, and SubbandNorm normalises the channels
        # jointly within each band, destroying the per-frame scale a division
        # would need.
        #
        # THE DEEPER REASON, and the one to put in the report. The mask the model
        # must predict is a RATIO -- keep this fraction of this bin. matched_level
        # is an ABSOLUTE quantity. "5 units of target here" cannot answer "keep
        # how much of it?", because it does not say whether that 5 is the whole
        # frame or 70 % of it. An absolute measurement was feeding an inherently
        # relative decision.
        #
        # Hand over the exact orthogonal decomposition instead:
        #
        #     x = matched_level * direction + unmatched,   <direction, unmatched> = 0
        #     ||x||^2 = matched_level^2 + ||unmatched||^2
        #
        # Nothing is lost: the old cue is match_fraction * ||x|| * direction, so
        # the network can rebuild the product if that is what it wants.
        #
        # THE LEFTOVER IS THE POINT. Every enrollment frame is non-negative and
        # every softmax weight is non-negative, so `direction` is a non-negative
        # combination of the TARGET'S OWN spectra. The cue can say "this looks
        # like them" and has no way to say "this energy belongs to the other
        # person". `unmatched` is the first negative evidence anywhere in this
        # path, and it says WHERE IN FREQUENCY the other speaker sits -- which is
        # what decides which bins to cut.
        frame_loudness = mix_mag.norm(dim=1, keepdim=True).clamp_min(self.eps)  # ||x||

        # match_fraction is ONE number per frame; the other two are
        # F-dimensional. Broadcast across frequency so it rides as a channel --
        # a view, not a copy. A DESIGN CHOICE, not a necessity: a 1-bin channel
        # would need a different band-split.
        match_fraction = (matched_level / frame_loudness).expand(-1, mix_mag.shape[1], -1)
        unmatched = mix_mag - matched_level * direction                     # (B,F,Tx)

        parts = torch.stack([direction, match_fraction, unmatched], dim=1)
        # Applied HERE, at the last line, so everything above -- and every
        # closed-form property the tests assert about the decomposition -- is
        # stated in the cue's own units. A constant per channel, so the
        # decomposition stays exact up to that constant.
        return parts * self.part_scales.to(parts.dtype).view(1, -1, 1, 1)


class ContextFusion(nn.Module):
    """Item 1c: a speaker embedding modulates the separator's features.

    Multiplicative adaptation, following Delcroix, Zmolikova, Ochiai, Kinoshita
    & Nakatani, "Improving speaker discrimination of target speech extraction
    with time-domain SpeakerBeam", ICASSP 2020; the same form as the gamma term
    of FiLM (Perez, Strub, de Vries, Dumoulin & Courville, AAAI 2018). The
    borrowed WeSep checkpoint reaches for the same thing with `fusion: multiply`
    on a 512-d context embedding -- BORROWED METHOD, and our data, metric and
    protocol differ, so nothing here is comparable to a published number.

        gamma = W e_hat + b                 (D -> N)
        z'    = z * (1 + gamma)             broadcast over bands and frames

    WHY MULTIPLY, NOT CONCATENATE OR ADD
    ------------------------------------
    Concatenating makes the embedding one more input the network MAY use, and it
    then has to survive six blocks of residual mixing to reach the mask. Adding
    contributes a constant offset per feature, which the next normalisation is
    free to remove -- it shifts, it does not modulate. Multiplying changes HOW
    STRONGLY each feature dimension is expressed for this particular speaker,
    and cannot be normalised away as an offset.

    WHAT IT VARIES OVER, WHICH IS THE WHOLE POINT
    ---------------------------------------------
    `gamma` depends on the speaker and on NOTHING ELSE -- not time, not
    frequency, not the mixture. So it carries no per-frame loudness and cannot
    be confounded by which speaker happens to be louder, which is the failure
    measured on 2026-09-22 (the cue tracks the louder voice 94.7 % of the time
    at SIR >= +5 dB). It is a per-utterance prior on WHO, multiplying per-frame
    evidence about WHEN and WHERE.

    ZERO INIT. W and b start at zero, so gamma is zero and z' == z EXACTLY: at
    initialisation this model is bit-identical to its item-1a control, and the
    arm cannot be credited with a different starting point. It still learns --
    dL/dW = (dL/dz') * z * e_hat^T is non-zero wherever z is.

    (1 + gamma) IS UNCONSTRAINED, so a learned gamma < -1 flips a feature's
    sign. That is a legitimate solution rather than a pathology, and it matches
    the residual-gate form in the literature. Bounded alternatives -- exp(gamma),
    1 + tanh(gamma) -- are one line each if `gamma_norm` ever shows it running
    away.
    """

    def __init__(self, embedding_dim, feature_dim):
        super().__init__()
        self.project = nn.Linear(embedding_dim, feature_dim)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    @property
    def n_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def gamma(self, enrol_embedding):
        """(B, D) -> (B, N). Exposed so a diagnostic can report ||gamma||
        without re-running the model: gamma staying at zero is this arm's
        headline failure mode and it must be cheap to check."""
        return self.project(enrol_embedding)

    def forward(self, z, enrol_embedding):
        """z (B, K, N, T), embedding (B, D) -> (B, K, N, T), same shape."""
        if enrol_embedding.dim() != 2:
            raise ValueError(
                f"enrol_embedding must be (B, D), got {tuple(enrol_embedding.shape)}")
        if enrol_embedding.shape[0] != z.shape[0]:
            raise ValueError(
                f"batch mismatch: z is {z.shape[0]}, embedding is "
                f"{enrol_embedding.shape[0]}.")
        g = self.gamma(enrol_embedding)                 # (B, N)
        return z * (1.0 + g[:, None, :, None])          # over bands and frames
