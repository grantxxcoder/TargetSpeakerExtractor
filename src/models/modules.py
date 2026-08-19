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
        
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x


class Estimator(nn.Module):
    """Per-band complex mask + residual spectrogram.
    
    Yu et al., Interspeech 2023 §2: band-specific MLPs predict a complex T-F mask M; "regarding the artifacts brought by the complex mask, an MLP is additionally used to directly predict the residual spectrogram R", giving eq. 2:  S = M (x) X + R.  Width 384 is the paper's; depth 2 follows the wesep reference, which the paper does not specify. GLU: Dauphin et al. 2017.

    The wesep reference omits the residual branch -- including it is deliberate.
    """
    def __init__(self, band_widths, feature_dim, mlp_hidden=384, n_hidden=2, causal=True, residual_branch=True):
        super().__init__()
        self.band_widths = list(band_widths)
        self.residual_branch = residual_branch
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
        """feats: (B, K, N, T);  mix_bands: list of K x (B, BW, T) complex-> (B, F, T) complex"""
        outs = []
        for i, bw in enumerate(self.band_widths):
            h = self.trunks[i](feats[:, i])                  # (B, H, T)
            B, _, T = h.shape
            
            # need to do the GLU activation here to BOUND an unstable mask
            raw  = self.mask_heads[i](h)                        # (B, 4*bw, T)
            mask = F.glu(raw, dim=1).reshape(B, 2, bw, T)       # (B, 2, bw, T)
            mr, mi = mask[:, 0], mask[:, 1]

            
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
