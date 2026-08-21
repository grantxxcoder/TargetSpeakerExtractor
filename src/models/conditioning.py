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
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
     
    def forward(self, mix_mag, enroll_mag):
        """mix_mag (B,F,Tx), enroll_mag (B,F,Te) -> (B,1,F,Tx)"""
        # cosine similarity: normalise each frame over frequency (shape, not loudness)
        bx = F.normalize(mix_mag,    p=2, dim=1, eps=self.eps)
        be = F.normalize(enroll_mag, p=2, dim=1, eps=self.eps)
        
        sim = torch.matmul(bx.transpose(1, 2), be)                # (B, Tx, Te)
        h   = torch.softmax(sim, dim=-1)                          # over enrollment
        tf  = torch.matmul(h, be.transpose(1, 2)).transpose(1, 2) # B_e H -> (B,F,Tx)
        
        # energy recovery: project the mixture magnitude onto the unit TF-Map frame
        tf = tf / tf.norm(dim=1, keepdim=True).clamp_min(self.eps)
        tf = (mix_mag * tf).sum(dim=1, keepdim=True) * tf
        return tf.unsqueeze(1)
