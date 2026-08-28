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
