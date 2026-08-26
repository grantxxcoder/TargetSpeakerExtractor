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

        # SCALE THE LOGITS. Not cosmetic -- without it this softmax averages
        # instead of selecting, and the whole speaker cue goes static.
        #
        # softmax compares logits by DIFFERENCE, not ratio:
        #     weight_i / weight_j = exp(s_i - s_j)
        # The F.normalize above (needed: we want spectral shape, not loudness)
        # bounds every cosine to [-1, 1], so the largest achievable difference
        # is ~1 and the best-matching enrollment frame can never outweigh the
        # worst by more than e^1 = 2.7x. Spread over Te ~ 628 frames that is
        # nothing: MEASURED 2026-08-25, 619.6 of 628 frames effectively used,
        # top frame holding 0.22 % of the weight against 0.16 % for a flat
        # average. The cue became the enrollment's long-term mean spectrum,
        # varying only 4.7 % over time, and the model ignored it.
        #
        # Zhang et al. eq (2) is written on UN-normalised products, measured
        # here at 0..932 -- a range that selects sharply on its own. Normalising
        # removed that range; this restores it. Same reasoning as attention's
        # 1/sqrt(d), applied in the opposite direction because normalising
        # already removed the magnitude growth that scaling usually tames.
        #
        # sqrt(F) by default. At F=257 that is ~16: the top 50 of 628 frames
        # then carry ~59 % of the weight, and time variation rises to ~39 %.
        scale = mix_mag.shape[1] ** 0.5 if self.scale is None else self.scale
        sim = sim * scale
        h   = torch.softmax(sim, dim=-1)                          # over enrollment
        tf  = torch.matmul(h, be.transpose(1, 2)).transpose(1, 2) # B_e H -> (B,F,Tx)
        
        # energy recovery: project the mixture magnitude onto the unit TF-Map frame
        tf = tf / tf.norm(dim=1, keepdim=True).clamp_min(self.eps)
        tf = (mix_mag * tf).sum(dim=1, keepdim=True) * tf
        return tf.unsqueeze(1)
