import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class BandSplit(nn.Module):
    """ 
    Slice and dice a spectrogram along frequency into per band tensors. 
    
    I will be following the code from  Luo & Yu, "Music Source Separation with Band-Split RNN", TASLP 2023.
    """

    def __init__(self, band_widths):
        super().__init__()
        self.band_widths = list(band_widths) # remember that we have many bands, and each band has a different width in terms of the number of frequency bins.
        self.offsets = np.cumsum([0] + self.band_widths).tolist() # offsets for each band, to slice the spectrogram into bands

    
    def forward(self, spec):
        """
        (B, F, T) complex    -> list of (B, BW, T) complex
        (B, C, F, T) real    -> list of (B, C, BW, T) real
        """
        assert spec.dim() in (3, 4), f"expected 3D or 4D, got {spec.dim()}D"
        assert spec.shape[-2] == self.offsets[-1],  f"{spec.shape[-2]} bins but band plan covers {self.offsets[-1]}"
        return [spec.narrow(-2, lo, w) for lo, w in zip(self.offsets[:-1], self.band_widths)]

    

class ChannelWiseLayerNorm(nn.Module):
    """LayerNorm over the channel axis at each time step.
    
    Causal and STATELESS: frame t is normalised using only its own channels, so nothing carries between frames and a 4 s training chunk normalises identically to a 60 s deployed stream.

    NAMING: wesep calls this "cLN", but Conv-TasNet's cLN (Luo & Mesgarani, 2019) means *cumulative* layer norm. Different thing. Cite as channel-wise.
    """
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)
        
    def forward(self, x):                        # (B, C, T)
        return self.norm(x.transpose(1, -1)).transpose(1, -1)

    

class SubbandNorm(nn.Module):
    """Per-band normalisation + projection to a common feature dim.

    Yu et al., Interspeech 2023: band-specific fully-connected layers convert each variable-width band into a fixed N-dimensional subband feature, N=128 for their 16 kHz model. Band splitting from Luo & Yu, TASLP 2023.

    This is where the architecture earns its keep. Speech energy falls ~6 dB per octave, so normalising all 257 bins together would leave the high bands numerically invisible. Each band gets its OWN normalisation and its own projection, so a quiet 6 kHz band reaches the RNN with the same representational budget as a loud 200 Hz one.
    """
    def __init__(self, band_widths, in_channels, feature_dim, causal=True):
        super().__init__()
        self.band_widths, self.feature_dim = list(band_widths), feature_dim
        self.blocks = nn.ModuleList()
        for bw in self.band_widths:
            d = bw * in_channels
            norm = ChannelWiseLayerNorm(d) if causal else nn.GroupNorm(1, d)
            self.blocks.append(nn.Sequential(norm, nn.Conv1d(d, feature_dim, 1)))
        
    def forward(self, bands):
        """list of K x (B, C, BW, T) -> (B, K, N, T)"""
        out = []
        for blk, band in zip(self.blocks, bands):
            B, C, BW, T = band.shape
            # reshape, NOT view: BandSplit uses narrow(), which is non-contiguous
            out.append(blk(band.reshape(B, C * BW, T)))
        return torch.stack(out, dim=1)


class ResRNN(nn.Module):
    """Residual LSTM block: norm -> LSTM -> project -> add the input back.

    Luo & Yu, "Music Source Separation with Band-Split RNN", TASLP 2023.
    Yu et al., Interspeech 2023 §4.2 use a 192-dim LSTM in a six-layer stack.
    Residual connection follows He et al. (2016).

    AXIS-AGNOSTIC. The caller decides whether the sequence is time or band by how it reshapes before calling. `bidirectional` MUST be False on the time axis (a backward pass reads the future) and should be True on the band axis (frequency is not a causal dimension).
    """ 
    def __init__(self, feature_dim, hidden_dim, bidirectional=False, causal_norm=True):
        super().__init__()
        self.norm = (ChannelWiseLayerNorm(feature_dim) if causal_norm else nn.GroupNorm(1, feature_dim))
        self.rnn  = nn.LSTM(
            feature_dim, 
            hidden_dim, 
            num_layers=1, 
            batch_first=True,
             bidirectional=bidirectional) # bidirec if bands not time
             
        # bidirectional doubles the LSTM's output width, so the projection back
        # to feature_dim must account for it or the residual add is illegal.
        self.proj = nn.Linear(hidden_dim * (2 if bidirectional else 1), feature_dim)

    def forward(self, x):
        """(B, N, L) -> (B, N, L).  N = feature_dim, L = sequence length."""
        y = self.norm(x)                  # pre-norm: residual path stays clean
        y = y.transpose(1, 2)             # (B, L, N) — nn.LSTM wants batch_first
        y, _ = self.rnn(y)                # (B, L, H * dirs)
        y = self.proj(y)                  # (B, L, N)
        return x + y.transpose(1, 2)      # (B, N, L), identity + branch



class BSNet(nn.Module):
    """One band-and-sequence block: model along time, then across bands.

    Luo & Yu (TASLP 2023) alternate sequence-level and band-level modelling.
    Yu et al. (Interspeech 2023) §4.2 stack six such blocks with 192-dim LSTMs.

    (wesep names these band_rnn / band_comm; time_rnn / band_rnn is clearer.)
    """
    def __init__(self, feature_dim, hidden_dim, causal=True):
        super().__init__()
        # TIME axis: unidirectional when causal -- a backward pass reads the future.
        self.time_rnn = ResRNN(feature_dim, hidden_dim,  bidirectional=not causal, causal_norm=causal)
        # BAND axis: ALWAYS bidirectional, even in the causal model. Frequency is
        # not a causal dimension and all bands of a frame arrive together.
        # This is correct, not an oversight.
        self.band_rnn = ResRNN(feature_dim, hidden_dim,  bidirectional=True, causal_norm=causal)


    def forward(self, x):
          """(B, K, N, T) -> (B, K, N, T)"""
          B, K, N, T = x.shape
  
          # along TIME: each (clip, band) pair is an independent sequence.
          # B and K are already adjacent, so a plain reshape is correct.
          y = self.time_rnn(x.reshape(B * K, N, T)).reshape(B, K, N, T)
 
          # across BANDS: each (clip, frame) pair is an independent sequence.
          # T must be moved next to B first -- reshaping without this permute
          # silently succeeds and scrambles bands with frames.
          z = y.permute(0, 3, 2, 1).reshape(B * T, N, K)
          z = self.band_rnn(z)
          return z.reshape(B, T, N, K).permute(0, 3, 2, 1)


      
class BandSequenceModel(nn.Module):
    """The six-layer stack (Yu et al., Interspeech 2023 §4.2)."""
    def __init__(self, feature_dim, hidden_dim, num_repeat=6, causal=True):
        super().__init__()
        self.blocks = nn.ModuleList(
            [BSNet(feature_dim, hidden_dim, causal) for _ in range(num_repeat)]
        )

    def forward(self, x, cue=None, gates=None):
        """(B, K, N, T) -> (B, K, N, T).

        `cue` and `gates` are D4a's per-block speaker-cue re-injection
        (decisions-pending.md D4; built by conditioning.TFMapInjector). Both
        default to None, which is the architecture frozen on 2026-08-28 --
        arithmetically, not approximately, since the injection is an addition
        that is skipped rather than an addition of zero.

            cue    (B, K, N, T)   the TF-Map projected into feature space, one
                                  tensor shared by every block
            gates  (num_repeat, K)  how much of it block i takes in band k

        THE CUE IS ADDED BEFORE EVERY BLOCK, THE FIRST ONE INCLUDED. The first
        injection is not redundant with the input channel: SubbandNorm projects
        the TF-Map jointly with the mixture's real and imaginary parts through
        one shared conv, so the cue there is entangled with the mixture and not
        separately addressable. This is a separate projection of the cue alone.
        """
        if (cue is None) != (gates is None):
            raise ValueError("pass cue and gates together or neither")
        for i, blk in enumerate(self.blocks):
            if cue is not None:
                # gates[i] is per BAND, so it broadcasts over (B, ., N, T).
                x = x + gates[i].view(1, -1, 1, 1) * cue
            x = blk(x)
        return x


def apply_hysteresis(mr, mi, hi_rel, lo_rel, down):
    """Region-grow the mask across frequency, then hold each frame's level.

    mr, mi: (B, F, T) real and imaginary parts of the complex mask.
    Returns the same, with the magnitude redistributed across frequency.

    WHY. MEASURED 2026-09-12 on 12 sir0_val trials: 84.0 % of the mask's
    variance is one number per frame, and it varies 6.5x less across frequency
    than the ideal mask does. The model has learned a broadband volume knob.
    Two voices overlapping in time occupy the same frequencies, so a broadband
    gain cannot separate them at all -- it can only be loud when someone speaks.
    This adds the frequency structure the mask is missing, WITHOUT retraining,
    so the hypothesis can be tested before a session is spent on it.

    HOW. Hysteresis, as in Canny's edge linking (Canny, IEEE PAMI 1986): a bin
    survives if it is strongly target-like, or weakly target-like AND adjacent in
    frequency to a surviving bin. Grouping weak evidence onto strong by
    continuity is also the core rule of computational auditory scene analysis
    (Bregman 1990; Wang & Brown 2006). BORROWED WITH A DIFFERENCE: there the
    grown regions ARE the system and are built from harmonicity and onset cues;
    here they sharpen a learned mask, and the acceptance test is downstream
    content fidelity rather than ideal-binary-mask overlap.

    CAUSAL. Growth runs across FREQUENCY within a frame, never across time. All
    frequencies of a frame arrive together, which is the same argument that lets
    the band-wise LSTM be bidirectional inside a causal model. No latency is
    spent.

    THE LEVEL IS HELD FIXED, deliberately. Each frame is rescaled to the mean
    magnitude it had before, so this changes only the SHAPE across frequency.
    Without it the experiment would confound sharpening with turning the output
    down, and turning the output down has already been measured to move these
    metrics on its own (2026-09-12, the mask floor).

    Vectorised, no Python loop over frames: the frequency scan is a cumulative
    maximum run in each direction, which propagates a seed through an unbroken
    stretch of candidates exactly as a flood fill would.
    """
    magnitude = (mr.pow(2) + mi.pow(2) + 1e-12).sqrt()       # (B, F, T)
    frame_mean = magnitude.mean(dim=1, keepdim=True).clamp_min(1e-8)
    candidate = magnitude >= lo_rel * frame_mean
    seed = (magnitude >= hi_rel * frame_mean) & candidate

    # Propagate a seed along frequency through unbroken candidate runs. Running
    # a cummax of seed*candidate up and then down the frequency axis reaches
    # every bin connected to a seed, and a non-candidate bin resets the run
    # because it multiplies the carry by zero.
    grown = seed.clone()
    for flip in (False, True):
        carry = seed.flip(1) if flip else seed
        cand = candidate.flip(1) if flip else candidate
        out = torch.zeros_like(carry)
        running = torch.zeros_like(carry[:, :1])
        chunks = []
        for f in range(carry.shape[1]):
            running = (running | carry[:, f:f+1]) & cand[:, f:f+1]
            chunks.append(running)
        out = torch.cat(chunks, dim=1)
        grown = grown | (out.flip(1) if flip else out)

    scale = torch.where(grown, torch.ones_like(magnitude),
                        torch.full_like(magnitude, down))
    sharpened = magnitude * scale

    # RESTORE THE FRAME'S ENERGY, NOT ITS MEAN MAGNITUDE. This was mean-based
    # until 2026-09-12 and it blew the output up: concentrating the same MEAN
    # magnitude into fewer bins multiplies the RMS, and the audio level follows
    # the RMS. On the real (nearly flat) mask that produced an output +7.38 dB
    # ABOVE the mixture -- 12 dB louder than the baseline -- and the evaluation
    # of it measured distortion rather than the idea: 65 % of clips returned no
    # transcript at all. The rough output-domain ratio hid the bug, because
    # killing its small values barely moves its mean.
    rms = lambda a: a.pow(2).mean(dim=1, keepdim=True).clamp_min(1e-12).sqrt()  # noqa: E731
    keep_level = rms(magnitude) / rms(sharpened)

    # A frame where NOTHING survived has no energy to redistribute, and scaling
    # it back up would divide by ~0. Leave such frames exactly as they were:
    # the sharpening has nothing to say about them.
    survived = grown.any(dim=1, keepdim=True)
    gain = torch.where(survived,
                       (sharpened * keep_level) / magnitude.clamp_min(1e-8),
                       torch.ones_like(magnitude))
    return mr * gain, mi * gain


class Estimator(nn.Module):
    """Per-band complex mask + residual spectrogram.
    
    Yu et al., Interspeech 2023 §2: band-specific MLPs predict a complex T-F mask M; "regarding the artifacts brought by the complex mask, an MLP is additionally used to directly predict the residual spectrogram R", giving eq. 2:  S = M (x) X + R.  Width 384 is the paper's; depth 2 follows the wesep reference, which the paper does not specify. GLU: Dauphin et al. 2017.

    The wesep reference omits the residual branch -- including it is deliberate.
    """
    def __init__(self, band_widths, feature_dim, mlp_hidden=384, n_hidden=2, causal=True, residual_branch=True,
                 mask_floor=0.0):
        super().__init__()
        self.band_widths = list(band_widths)
        self.residual_branch = residual_branch
        # MASK FLOOR -- the smallest magnitude the mask is allowed to take.
        # 0.0 is off, which is every model trained before 2026-09-12.
        #
        # WHY IT EXISTS. MEASURED 2026-09-12 on 12 sir0_val trials: the baseline
        # drives 32.8 % of time-frequency bins below 0.1 (a cut deeper than
        # 20 dB) and the state-teacher model drives 44.0 % below it. A bin at
        # zero is a HOLE -- no content at all -- and the target's own energy in
        # that bin goes with it. A floored bin instead carries a quiet copy of
        # the real mixture, which invents nothing because it IS the recorded
        # audio, only attenuated.
        #
        # Borrowed: the gain floor in Wiener-filter speech enhancement, which
        # exists for exactly this reason (Berouti, Schwartz & Makhoul, ICASSP
        # 1979, spectral subtraction with a spectral floor). BORROWED WITH A
        # DIFFERENCE: there the floor is tuned to suppress musical noise for a
        # human listener; here the acceptance test is content fidelity for a
        # downstream model, and we have measured that our mask is SMOOTHER than
        # the ideal one rather than rougher, so musical noise is not the
        # mechanism being treated -- over-removal is.
        #
        # SETTABLE AT INFERENCE on an already-trained model, which is the whole
        # point: it makes the hypothesis testable in minutes instead of a
        # 10-hour training run.
        self.mask_floor = float(mask_floor)
        # MASK HYSTERESIS, inference-time, decisions-pending.md D16. None = off.
        # A (hi, lo, down) triple; see apply_hysteresis.
        self.mask_hysteresis = None

        # RESIDUAL SCALE, inference-time. 1.0 = the trained model unchanged,
        # 0.0 = the residual branch deleted (S = M (x) X alone).
        #
        # WHY IT EXISTS. The mask is MULTIPLIED by the mixture, so it can only
        # scale and rotate energy the recording already contains -- in a bin
        # where |X| = 0 no mask value produces output. R is ADDED, comes from a
        # raw Conv1d with no GLU and no bound, and is therefore free to place
        # energy in bins the microphone never recorded. That operation is what
        # "invented word" means physically, and invented words are 41.5 % of our
        # wrong content words (decisions-pending.md, 2026-09-11) with no
        # mechanism yet assigned to them. D6 flagged R as unbounded and
        # unconditioned; nothing has ever measured what it carries.
        #
        # SETTABLE AT INFERENCE, like mask_floor, so the hypothesis costs an
        # afternoon of CPU rather than a training run.
        #
        # WHAT IT CANNOT SETTLE. This ablates R from a model TRAINED WITH R, so
        # the mask has learned to rely on it. A loss here is not evidence that R
        # is harmful -- only a trained-without-R arm answers that. Read this
        # measurement as "is R implicated in fabrication", never as "R is bad".
        self.residual_scale = 1.0

        # Stash the two output paths separately for analysis. Off by default:
        # it retains tensors and is for diagnostics, never for training.
        self.capture_parts = False
        self.last_parts = None
        self.trunks, self.mask_heads = nn.ModuleList(), nn.ModuleList()
        self.res_heads = nn.ModuleList() if residual_branch else None # we need the additional prediction head to predict the residual spectrogram 
        
        for bw in self.band_widths:
            layers = [ChannelWiseLayerNorm(feature_dim) if causal
                    else nn.GroupNorm(1, feature_dim)]
            d = feature_dim
            for _ in range(n_hidden):
                layers += [nn.Conv1d(d, mlp_hidden, 1), nn.Tanh()]
                d = mlp_hidden
            self.trunks.append(nn.Sequential(*layers))
            # 4*bw = (value, gate) x (real, imag) x bw bins
            self.mask_heads.append(nn.Conv1d(mlp_hidden, 4 * bw, 1))
            if residual_branch:
                self.res_heads.append(nn.Conv1d(mlp_hidden, 2 * bw, 1))

    # use this method to actually move the features through the model and get the output complex spectrogram
    def forward(self, feats, mix_bands):
        """feats: (B, K, N, T);  mix_bands: list of K x (B, BW, T) complex-> (B, F, T) complex

        THE BANDS ARE GATHERED BEFORE THE MASK IS APPLIED, changed 2026-09-12.
        It used to multiply and accumulate band by band. Arithmetically the same
        -- verified bit-identical -- but any post-processing of the mask that
        cares about NEIGHBOURING FREQUENCIES cannot be written inside a per-band
        loop, because the band edges break the adjacency it needs. The gather is
        what lets `mask_hysteresis` exist at all.
        """
        masks, residuals = [], []
        for i, bw in enumerate(self.band_widths):
            h = self.trunks[i](feats[:, i])                  # (B, H, T)
            B, _, T = h.shape

            # need to do the GLU activation here to BOUND an unstable mask
            raw  = self.mask_heads[i](h)                        # (B, 4*bw, T)
            masks.append(F.glu(raw, dim=1).reshape(B, 2, bw, T))
            if self.residual_branch:
                residuals.append(self.res_heads[i](h).reshape(B, 2, bw, T))

        mask = torch.cat(masks, dim=2)                          # (B, 2, F, T)
        mix  = torch.cat(mix_bands, dim=-2)                     # (B, F, T) complex
        mr, mi = mask[:, 0], mask[:, 1]

        if self.mask_hysteresis is not None:
            mr, mi = apply_hysteresis(mr, mi, *self.mask_hysteresis)

        if self.mask_floor > 0.0:
            magnitude = (mr.pow(2) + mi.pow(2) + 1e-12).sqrt()
            mr = mr + (self.mask_floor - magnitude).clamp_min(0.0)

        xr, xi = mix.real, mix.imag
        er = xr * mr - xi * mi
        ei = xr * mi + xi * mr
        masked_r, masked_i = er, ei          # S = M (x) X, before R is added
        rr = ri = None
        if self.residual_branch:
            r = torch.cat(residuals, dim=2)
            # residual_scale is 1.0 for every model trained before 2026-09-13,
            # so this is arithmetically the old path at the default.
            rr = r[:, 0] * self.residual_scale
            ri = r[:, 1] * self.residual_scale
            er, ei = er + rr, ei + ri
        if self.capture_parts:
            self.last_parts = {
                "masked": torch.complex(masked_r, masked_i).detach(),
                "residual": (torch.complex(rr, ri).detach()
                             if rr is not None else None),
                "mixture": mix.detach(),
                "mask_mag": (mr.pow(2) + mi.pow(2) + 1e-12).sqrt().detach(),
            }
        return torch.complex(er, ei)

    def _legacy_forward(self, feats, mix_bands):
        """The pre-2026-09-12 band-by-band form, kept so the equivalence of the
        gathered version can be asserted rather than assumed."""
        outs = []
        for i, bw in enumerate(self.band_widths):
            h = self.trunks[i](feats[:, i])                  # (B, H, T)
            B, _, T = h.shape

            raw  = self.mask_heads[i](h)                        # (B, 4*bw, T)
            mask = F.glu(raw, dim=1).reshape(B, 2, bw, T)       # (B, 2, bw, T)
            mr, mi = mask[:, 0], mask[:, 1]

            if self.mask_floor > 0.0:
                # Top the mask up towards PASS-THROUGH, not up along its own
                # direction. In a bin the model wants to kill, the complex mask
                # is near the origin and its phase is whatever noise happened to
                # land there -- scaling that up would inject an arbitrary phase
                # rotation. Adding a real, positive component instead leaves the
                # mixture's own phase in place, which is what "a quiet copy of
                # the recorded audio" has to mean.
                magnitude = (mr.pow(2) + mi.pow(2) + 1e-12).sqrt()
                mr = mr + (self.mask_floor - magnitude).clamp_min(0.0)

            
            xr, xi = mix_bands[i].real, mix_bands[i].imag
            er = xr * mr - xi * mi                           # complex multiply
            ei = xr * mi + xi * mr
            
            if self.residual_branch:
                r = self.res_heads[i](h).reshape(B, 2, bw, T)
                er, ei = er + r[:, 0], ei + r[:, 1]
        
            outs.append(torch.complex(er, ei))
        return torch.cat(outs, dim=-2)



def lookahead_shift(h, k):
    """Give the mask head k frames of future context.

    h: (B, K, N, T) features out of the causal sequence stack.
    Returns h' where h'[..., t] == h[..., t + k], the last k frames edge-padded.

    The mask for frame t is then built from a hidden state that has consumed
    frames up to t + k, i.e. k * hop of lookahead -- while the mask itself stays
    aligned to mixture frame t, which is required because a multiplicative mask
    cannot shift energy in time. See decisions-m1.md 2026-08-18.

    Module-level function, not a method: it is stateless and is called from the
    model's forward between the separator and the estimator.
    """
    if k == 0:
        return h
    return F.pad(h[..., k:], (0, k), mode="replicate")
