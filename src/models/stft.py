
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class STFT(nn.Module):
    """ 
    Short-time Fourier Transform (STFT) and its inverse (ISTFT).
    This class is intended to take a waveform and return a complex spectrogram, and vice versa. It uses a Hann window for the STFT and applies overlap-add for the inverse transform. 
    """
    def __init__(self, n_fft=512, hop_length=128, sample_rate=16000):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        self.padding = n_fft - hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
    
          
    def latency_ms(self, lookahead_frames=0):
        """Algorithmic latency. Convention: window fill + emit hop + lookahead.
        State the convention when you quote it -- CARTSE counts window - hop
        instead, so their 24 ms and our 40 ms describe the same framing."""
        samples = self.n_fft + self.hop_length + lookahead_frames * self.hop_length
        return 1000 * samples / self.sample_rate 

    def _n_frames(self, T):
      # +2*padding: the tail needs the same ramp-out room the head gets, or the
      # final samples land where the OLA envelope has decayed to ~0.
      return math.ceil((T + 2 * self.padding - self.n_fft) / self.hop_length) + 1
 
    # (B, T) -> (B, F, N) complex
    def forward(self, x):                        
        T = x.shape[-1]
        n_frames = self._n_frames(T)
        right = (n_frames - 1) * self.hop_length + self.n_fft - (T + self.padding)
        xp = F.pad(x, (self.padding, right))
        return torch.stft(xp, self.n_fft, self.hop_length, self.n_fft,
                            self.window, center=False, return_complex=True)

    # (B, F, N) complex -> (B, length)                  
    def inverse(self, X, length):              
        frames = torch.fft.irfft(X, n=self.n_fft, dim=1) * self.window[None, :, None]
        B, _, N = frames.shape
        total = (N - 1) * self.hop_length + self.n_fft
        fold = lambda t: F.fold(t, (1, total), (1, self.n_fft), stride=(1, self.hop_length))
        
        out = fold(frames).view(B, total)
        env = fold(self.window.pow(2)[None, :, None].expand(1, -1, N)).view(total)
        return (out / env.clamp_min(1e-11))[:, self.padding:self.padding + length]



