"""The frozen speaker encoder the extension uses: one enrolment waveform gives one identity vector.

The backbone is ECAPA-TDNN (Desplanques, Thienpondt & Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN Based Speaker Verification", Interspeech 2020),
loaded from the snapshot beside the repo, never from the network.
The TF-Map cue is a spectral average, so it confuses speakers with similar average spectra. ECAPA was trained to tell thousands of speakers apart, which is the one property it adds (decisions-m2.md 2026-09-22).
It lives outside the model, so checkpoints stay small and the stream never runs it per frame: the embedding is computed once per utterance.
"""

from pathlib import Path

import torch
import torch.nn as nn
from speechbrain.inference.speaker import EncoderClassifier

# The ECAPA snapshot sits beside the repo.
ECAPA_DIR = Path(__file__).resolve().parents[3] / "ecapa_pretrained"


# this loads the frozen ECAPA encoder from the snapshot
def load_context_encoder(device):
    """Return a plain nn.Module holding ECAPA's three stages. It is never trained and never saved in the extractor's checkpoints."""
    if not (ECAPA_DIR / "embedding_model.ckpt").exists():
        raise FileNotFoundError(f"No ECAPA snapshot at {ECAPA_DIR}. Put the snapshot beside the repo.")
    # speechbrain wants "cuda:0" rather than "cuda", or it warns before falling back to device 0.
    if str(device) == "cuda":
        device = "cuda:0"
    classifier = EncoderClassifier.from_hparams(source=str(ECAPA_DIR), savedir=str(ECAPA_DIR), run_opts={"device": str(device)})
    classifier.eval()
    encoder = nn.Module()
    encoder.compute_features = classifier.mods.compute_features
    encoder.mean_var_norm = classifier.mods.mean_var_norm
    encoder.embedding_model = classifier.mods.embedding_model
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    encoder.eval()
    return encoder


# this turns enrolment waveforms shaped (batch, samples) into unit-length speaker embeddings shaped (batch, 192)
@torch.no_grad()
def embed_enrolment(encoder, enrolment):
    """eval() runs on every call, because ECAPA has BatchNorm and training mode would change its running statistics.

    The embedding is made unit length, because only its direction carries identity, not how long or loud the enrolment was.
    """
    encoder.eval()
    features = encoder.compute_features(enrolment)
    features = encoder.mean_var_norm(features, torch.ones(enrolment.shape[0], device=enrolment.device))
    embedding = encoder.embedding_model(features).squeeze(1)
    return torch.nn.functional.normalize(embedding, p=2, dim=-1)
