"""Frozen speaker encoder: enrolment waveform -> one identity vector per trial.

Item 1c, ranked-next-steps.md. Backbone is ECAPA-TDNN (Desplanques, Thienpondt &
Demuynck, "ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation
in TDNN Based Speaker Verification", Interspeech 2020), loaded from our pinned
snapshot and never from the network.

WHY THIS EXISTS, IN ONE PARAGRAPH
---------------------------------
The TF-Map cue builds its template as a convex combination of the ENROLMENT'S
OWN SPECTRA, so it is a spectral average: two speakers with similar average
spectra -- similar pitch, similar vocal tract length -- give similar templates.
MEASURED 2026-09-22: the cue lands on the right voice 71.8 % of the time
cross-gender and 52.6 % same-gender, which is a coin flip (decisions-m2.md).
ECAPA is trained with a speaker-discriminative objective over thousands of
speakers, so its geometry separates exactly the speakers a spectral average
cannot. That is the single property being imported.

WHY IT IS OUTSIDE THE MODEL
---------------------------
Weighed 2026-09-22 and recorded in decisions-m2.md. Four reasons, in order:

  * `nn.DataParallel` re-replicates a module on EVERY forward, so 20.77 M
    frozen weights would be scattered to each card every step for no benefit.
    (MEASURED 2026-09-22 from the loaded snapshot, not taken from the WeSep
    config -- its 14.597 M `spk_ft` is a DIFFERENT frontend and conflating the
    two would understate this by 30 %.)
  * The checkpoint stays small: ~83 MB per saved epoch, times the three kept,
    for weights that never change and are already hash-pinned on disk.
  * The fusion is testable with a synthetic 192-vector -- no speechbrain import,
    no snapshot, milliseconds -- which matters with a 9-minute suite.
  * It keeps the streaming claim literally true: the deployed model contains NO
    speaker encoder in its per-frame path. The embedding is computed once,
    before the stream begins, and the per-frame cost is exactly zero.

The risk this trades against -- a call site forgetting to pass the embedding and
silently running an UNCONDITIONED model that still scores -- is closed by
`BSRNN_TFMAP_CONTEXT.forward` making it a REQUIRED keyword argument. Python
raises TypeError; nothing has to be remembered. See decisions-pending.md E8 for
why silent-wrong-arm is the failure mode this project guards against hardest.

L2 NORMALISATION IS DELIBERATE
------------------------------
ECAPA's embedding magnitude varies with how long and how loud the enrolment
clip was; only its DIRECTION carries identity, which is why speaker
verification scores with cosine similarity rather than Euclidean distance.
Normalising removes a nuisance variable that has nothing to do with who is
speaking. Recorded rather than assumed: the un-normalised form is a one-line
ablation if it is ever wanted.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

# The files that define the backbone's behaviour. label_encoder.ckpt and
# classifier.ckpt belong to VoxCeleb's closed-set head, which we never call, so
# they are deliberately NOT hashed -- pinning them would fail a run for a change
# that cannot reach our numbers.
BACKBONE_FILES = ("embedding_model.ckpt", "mean_var_norm_emb.ckpt",
                  "hyperparams.yaml")


def resolve_ecapa_dir(ecapa_dir):
    """Find the snapshot, or fail naming every place that was tried.

    THE BUG THIS EXISTS FOR, found 2026-09-22 before it cost a session. The
    Kaggle bundle stages the snapshot INSIDE the code tree
    (kaggle_bundle/code/ecapa_pretrained -> /kaggle/working/repo/ecapa_pretrained)
    while every config says `../ecapa_pretrained`, which from the repo root --
    train.py's working directory there -- is /kaggle/working/ecapa_pretrained.
    Different directory. Locally the sibling path is right; on Kaggle it is not,
    and nothing in the notebook rewrites it.

    Resolving in code rather than per-config because a path that is correct in
    one environment and silently absent in the other is exactly the shape of
    failure that gets discovered after the upload.
    """
    tried = []
    for candidate in (Path(ecapa_dir),
                      Path(__file__).resolve().parents[2] / "ecapa_pretrained",
                      Path(__file__).resolve().parents[3] / "ecapa_pretrained"):
        tried.append(str(candidate))
        if (candidate / "embedding_model.ckpt").exists():
            return candidate
    raise FileNotFoundError(
        "no ECAPA snapshot with embedding_model.ckpt. Tried:\n  "
        + "\n  ".join(tried)
        + "\nStage it with scripts/make_kaggle_bundle.py, or set "
          "model.ecapa_dir.")


def file_hashes(ecapa_dir):
    """sha256 of every file that can move a number here. Recorded in meta.yaml
    so a silent weights change is detectable after the fact, and assertable
    before the fact via `expected_hashes`."""
    ecapa_dir = Path(ecapa_dir)
    out = {}
    for name in BACKBONE_FILES:
        path = ecapa_dir / name
        if path.exists():
            out[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


class ContextEncoder(nn.Module):
    """(B, T) enrolment waveform -> (B, D) unit-norm identity vector.

    NOT a parameter of the extractor. It is constructed beside the model, its
    weights are frozen, and it never appears in the model's state_dict or in
    the optimiser.
    """

    def __init__(self, ecapa_dir="../ecapa_pretrained", device="cpu",
                 expected_hashes=None, normalise=True):
        super().__init__()
        self.ecapa_dir = str(resolve_ecapa_dir(ecapa_dir))
        self.normalise = bool(normalise)
        self.hashes = file_hashes(self.ecapa_dir)
        if expected_hashes:
            for name, want in expected_hashes.items():
                got = self.hashes.get(name)
                if got != want:
                    raise RuntimeError(
                        f"{name} is {got}, the run was pinned to {want}. A "
                        f"silent weights change would move every number this "
                        f"encoder produces with nothing appearing in a diff.")

        # Imported HERE, not at module scope: speechbrain is a heavy dependency
        # and only this arm needs it. src/models/bsrnn.py stays free of it, so
        # the baseline and item-1a paths import nothing new.
        from speechbrain.inference.speaker import EncoderClassifier

        # "cuda:0", not "cuda": speechbrain parses the string itself and warns
        # "Could not parse CUDA device string" before falling back to device 0.
        if str(device) == "cuda":
            device = "cuda:0"
        classifier = EncoderClassifier.from_hparams(
            source=self.ecapa_dir, savedir=self.ecapa_dir,
            run_opts={"device": str(device)})
        classifier.eval()
        # Taken directly rather than via encode_batch(), matching
        # state_teacher.py: that method may wrap its forward in no_grad, and
        # depending on it would make the behaviour of this class a property of
        # somebody else's convenience wrapper.
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

        # FROZEN, and asserted rather than assumed. A frozen module whose
        # parameters quietly require grad is a training-time bug that shows up
        # only as a slightly wrong optimiser state.
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @property
    def embedding_dim(self):
        """Read from the loaded weights rather than hardcoded: a different
        ECAPA snapshot would change it, and 192 asserted in three places is
        three places to forget."""
        return int(self.embedding_model.fc.conv.weight.shape[0])

    @torch.no_grad()
    def embed(self, enrolment):
        """(B, T) waveform -> (B, D).

        no_grad is structural, not an optimisation: the enrolment is an INPUT,
        nothing upstream of it is being optimised, and this module has no
        trainable parameters at all.
        """
        features = self.compute_features(enrolment)
        features = self.mean_var_norm(
            features, torch.ones(enrolment.shape[0], device=enrolment.device))
        embedding = self.embedding_model(features).squeeze(1)      # (B, D)
        if self.normalise:
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
        return embedding

    def train(self, mode=True):
        """Stays in eval mode whatever the training loop does to it. ECAPA has
        BatchNorm; letting model.train() reach it would update running
        statistics from our mixtures and silently change a frozen encoder."""
        return super().train(False)

    def describe(self):
        """One-line provenance for meta.yaml."""
        return json.dumps(dict(backbone="ECAPA-TDNN", dir=self.ecapa_dir,
                               embedding_dim=self.embedding_dim,
                               normalise=self.normalise, hashes=self.hashes),
                          sort_keys=True)
