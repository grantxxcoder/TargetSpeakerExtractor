"""Frozen speaker-state teacher, used as a differentiable training signal.

Two questions per window of the extractor's output, both conditioned on the
enrolment:

    output 0   is the TARGET audible here?
    output 1   is a NON-TARGET audible here?

Nothing in this file trains. The backbone is a stock speaker-verification model
and the head was fitted once, offline, on the rendered stems; both are frozen
and hash-pinned. See decisions-pending.md D14.

WHY A FROZEN TEACHER RATHER THAN A JOINTLY TRAINED ONE
------------------------------------------------------
A detector trained alongside the extractor can satisfy a shared objective by
agreeing with the extractor while the audio does not change -- the two collude
and the loss falls with nothing improved. Frozen, the only available move is to
change the audio. This is the reward-model separation from RLHF, and it is the
same discipline CLAUDE.md already requires of the differentiable proxies.

The failure mode that survives freezing is overoptimisation: a frozen learned
scorer gets gamed off-distribution eventually. Its signature is this term
falling while an independent measure stalls, which has happened twice in this
project already (2026-08-25, total loss falling into a mute; 2026-09-04,
conditioning diagnostics peaking on an epoch below pass-through). Mitigations
are the caller's job: never select on this term, keep its weight modest, and
log the teacher's accuracy on the extractor's own outputs each epoch -- we own
the ground-truth stems for every output, so calibration drift is directly
observable.

Provenance:
  backbone   Desplanques, Thienpont & Demuynck, "ECAPA-TDNN: Emphasized
             Channel Attention, Propagation and Aggregation in TDNN Based
             Speaker Verification", Interspeech 2020. SpeechBrain's VoxCeleb
             checkpoint (Ravanelli et al., 2021), snapshotted with hashes.
  framing    frame-level speaker activity from a speaker profile: personal VAD
             (Ding et al., ICASSP 2020) and TS-VAD (Medennikov et al.,
             Interspeech 2020). BORROWED WITH A DIFFERENCE: there it is the
             system output; here it is a frozen training signal, and the
             backbone is a stock verification model rather than one trained
             jointly with a separator.
  precedent  a frozen network as a training-only scorer: perceptual / deep
             feature losses, Johnson et al., ECCV 2016; in speech enhancement,
             Germain, Chen & Koltun, Interspeech 2019. Those match features of
             WHAT was said; this matches WHO is audible when.

MEASURED, and these are the numbers that justify the design (D14, 2026-09-10):
  identity, target vs interferer      0.940 balanced  (WavLM managed 0.570)
  is the target audible               AUC 0.944
  is a non-target audible             AUC 0.795
  the same question from plain cosine       0.405  -- BELOW chance, which is
      why the learned head exists at all rather than a similarity threshold

RESOLUTION. The teacher works on ~1 s windows because a verification embedding
needs about a second of audio. It can say "the target is speaking around here";
it CANNOT mark a word boundary, and it must never be described as per-frame.

NOT the judge, not the evaluation ASR, not WeSep. WeSep's own checkpoint carries
an ECAPA speaker encoder, but it was trained jointly with WeSep's separator and
WeSep is this project's comparison baseline -- a teacher derived from it would
couple the system under test to the system it is measured against.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import torch.nn as nn

# Output columns, in order. Index 0 is the target question.
QUESTIONS = ("target_audible", "non_target_audible")
TARGET_AUDIBLE, NON_TARGET_AUDIBLE = 0, 1


def head_shape_from_state_dict(state_dict):
    """(hidden, n_layers) read off the saved weights.

    The notebook's checkpoints do not record the trunk's shape -- verified
    2026-09-11, the saved keys are the metrics and the data settings only -- so
    it is inferred rather than configured. That is the more robust choice
    anyway: a recorded shape can disagree with the weights, an inferred one
    cannot.

    `project.0.weight` is (hidden, 3*embedding_dim + 1). A bidirectional LSTM
    writes `weight_ih_l{k}` and `weight_ih_l{k}_reverse` per layer, so counting
    the forward ones gives the depth.
    """
    hidden = state_dict["project.0.weight"].shape[0]
    n_layers = sum(1 for k in state_dict
                   if k.startswith("across_windows.weight_ih_l")
                   and not k.endswith("_reverse"))
    return int(hidden), int(n_layers)


class StateHead(nn.Module):
    """The trainable part of the teacher, as fitted in notebooks/state_detector.

    Kept here rather than imported from the notebook so a checkpoint loads
    without the notebook existing. Attribute names match both the notebook's
    `StateHead` and its `SearchableHead`, so a hand-picked and a tuned
    checkpoint load into the same class.

    WHY A RECURRENT TRUNK RATHER THAN A PER-WINDOW MLP
    --------------------------------------------------
    MEASURED 2026-09-10, same features, same split, same objective:

        per-window MLP  149 k   mean AUC 0.8836   non-target 0.8199
        BiLSTM          339 k   mean AUC 0.9216   non-target 0.8710
        self-attention  473 k   mean AUC 0.9170   non-target 0.8672

    +5.1 points on the question this loss uses, against +1.7 for doubling the
    data, +0.8 for rotating the enrolment and +0.7 for a 42-trial
    hyperparameter search. Ten times the effect of the next best intervention.

    A single window at cosine +0.23 to the enrolment is ambiguous between "the
    target alone, quieter" and "the target plus someone else" -- `both` sits at
    +0.233 between `target` at +0.321 and `interferer` at +0.044, because a
    verification embedding describes whichever voice dominates. The neighbouring
    windows resolve much of that.

    Bidirectional, which costs nothing because the teacher never streams. It
    also means this head must NEVER be reused inside the causal extractor.
    """

    def __init__(self, embedding_dim=192, hidden=128, n_layers=1,
                 dropout=0.0, n_outputs=len(QUESTIONS)):
        super().__init__()
        self.norm_window = nn.LayerNorm(embedding_dim)
        self.norm_enrolment = nn.LayerNorm(embedding_dim)
        self.project = nn.Sequential(
            nn.Linear(3 * embedding_dim + 1, hidden), nn.GELU())
        self.across_windows = nn.LSTM(
            hidden, hidden, num_layers=n_layers, batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0.0)
        self.output = nn.Sequential(nn.Dropout(dropout),
                                    nn.Linear(2 * hidden, n_outputs))

    def per_window_features(self, window_embeddings, enrolment_embedding):
        """(B, W, 192) and (B, 192) -> (B, W, 577).

        The cosine is computed on the RAW embeddings, before LayerNorm, and
        passed as its own input: it is the single most informative feature
        available -- MEASURED AUC 0.951 on the identity question with no
        training at all -- and a linear layer would otherwise have to learn to
        sum 192 product terms to recover it.

        The product is there because agreement is a PRODUCT: no linear map of
        [window, enrolment] can compute an inner product between its halves.
        The two LayerNorms put the three blocks on comparable scales so the
        product is not the quietest input.
        """
        window_unit = nn.functional.normalize(window_embeddings, dim=-1)
        enrolment_unit = nn.functional.normalize(enrolment_embedding, dim=-1)
        cosine = (window_unit * enrolment_unit.unsqueeze(1)).sum(-1,
                                                                 keepdim=True)
        window = self.norm_window(window_embeddings)
        enrolment = self.norm_enrolment(enrolment_embedding)
        enrolment = enrolment.unsqueeze(1).expand_as(window)
        return torch.cat([window, enrolment, window * enrolment, cosine],
                         dim=-1)

    def forward(self, window_embeddings, enrolment_embedding):
        """(B, W, 192) and (B, 192) -> logits (B, W, 2)."""
        hidden = self.project(
            self.per_window_features(window_embeddings, enrolment_embedding))
        hidden, _ = self.across_windows(hidden)
        return self.output(hidden)


class StateTeacher(nn.Module):
    """ECAPA + head, frozen, scoring windows of a waveform.

    Gradients flow THROUGH it into whatever produced the waveform, which is the
    entire point. Its own parameters never receive any.
    """

    def __init__(self, checkpoint_path, ecapa_dir, device="cpu",
                 verify_hashes=True):
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu",
                                weights_only=False)

        self.sample_rate = 16000
        self.window_seconds = float(checkpoint["window_seconds"])
        self.window_hop_seconds = float(checkpoint["window_hop_seconds"])
        self.window_samples = int(round(self.window_seconds * self.sample_rate))
        self.window_hop_samples = int(round(self.window_hop_seconds
                                            * self.sample_rate))
        # Recorded for provenance, and because the same weights mean something
        # different under a different audibility threshold: it decides how much
        # residual interferer still counts as a second voice being present.
        self.audible_db = float(checkpoint["audible_db"])
        self.checkpoint_path = str(checkpoint_path)

        self._load_backbone(Path(ecapa_dir), checkpoint, verify_hashes, device)

        hidden, n_layers = head_shape_from_state_dict(checkpoint["state_dict"])
        self.head = StateHead(
            embedding_dim=int(checkpoint.get("embedding_dim", 192)),
            hidden=hidden, n_layers=n_layers,
            # dropout 0.0 regardless of what it was trained at: the teacher must
            # be DETERMINISTIC. Dropout here would make the same audio score
            # differently on consecutive steps, so the extractor would be
            # chasing a moving target for no reason.
            dropout=0.0,
        )
        # strict=True: a shape or name mismatch means the checkpoint and this
        # class disagree about the architecture, and a partially-loaded teacher
        # would produce plausible numbers from partly-random weights.
        self.head.load_state_dict(checkpoint["state_dict"], strict=True)
        self.head_shape = dict(hidden=hidden, n_layers=n_layers)

        self.to(device)
        self.eval()
        # THE FREEZE. requires_grad=False, never no_grad() on the forward: under
        # no_grad the term still computes and still logs a plausible number
        # while having EXACTLY ZERO effect on the model. That is the same dead-
        # gradient failure recorded against the original "add WER to the loss"
        # idea, and it would be invisible for a whole training run.
        for parameter in self.parameters():
            parameter.requires_grad = False

    def _load_backbone(self, ecapa_dir, checkpoint, verify_hashes, device):
        """The frozen ECAPA, loaded from our snapshot and never from the network.

        Kaggle runs with networking off, and an unpinned fetch would make every
        number irreproducible.
        """
        from speechbrain.inference.speaker import EncoderClassifier

        if verify_hashes and checkpoint.get("ecapa_hashes"):
            for name, expected in checkpoint["ecapa_hashes"].items():
                path = ecapa_dir / name
                if not path.exists():
                    raise FileNotFoundError(
                        f"{path} is missing. The teacher's backbone is "
                        f"hash-pinned; re-snapshot it rather than substituting "
                        f"another checkpoint.")
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(
                        f"{name} is {actual[:16]}..., the head was fitted "
                        f"against {expected[:16]}.... A silent weights change "
                        f"would move every number this term produces with "
                        f"nothing appearing in a diff.")

        classifier = EncoderClassifier.from_hparams(
            source=str(ecapa_dir), savedir=str(ecapa_dir),
            run_opts={"device": device})
        classifier.eval()
        # The three modules, taken directly rather than via encode_batch():
        # that method may wrap its forward in no_grad, which would silently sever
        # the gradient path this whole file exists to provide.
        self.compute_features = classifier.mods.compute_features
        self.mean_var_norm = classifier.mods.mean_var_norm
        self.embedding_model = classifier.mods.embedding_model

    def window_starts(self, n_samples):
        """Offsets of every whole window that fits inside n_samples."""
        return list(range(0, n_samples - self.window_samples + 1,
                          self.window_hop_samples))

    def n_windows(self, n_samples):
        return len(self.window_starts(n_samples))

    def embed(self, audio):
        """(B, T) waveform -> (B, 192) embedding. Differentiable in `audio`."""
        features = self.compute_features(audio)
        features = self.mean_var_norm(
            features, torch.ones(audio.shape[0], device=audio.device))
        return self.embedding_model(features).squeeze(1)

    def forward(self, estimate, enrolment_embedding, window_starts=None):
        """Score windows of the extractor's output.

        estimate            (B, T) waveform, the thing being trained
        enrolment_embedding (B, 192), precomputed -- see `embed_enrolment`
        window_starts       which windows to score; None scores all of them

        -> logits (B, W, 2)

        `enrolment_embedding` is precomputed on purpose. The enrolment is 5 s
        against a 1 s scored window, so embedding it here would be the LARGER
        half of the cost, repeated every step, for a signal that never changes.
        """
        if window_starts is None:
            window_starts = self.window_starts(estimate.shape[-1])
        if not window_starts:
            raise ValueError(
                f"{estimate.shape[-1]} samples is shorter than one "
                f"{self.window_samples}-sample window")

        batch_size = estimate.shape[0]
        # All windows of all examples in ONE backbone call: the cost is the
        # 20.8 M-parameter forward, and batching it is the only lever there is.
        windows = torch.stack(
            [estimate[:, start:start + self.window_samples]
             for start in window_starts], dim=1)          # (B, W, window)
        n_windows = windows.shape[1]
        flat = windows.reshape(batch_size * n_windows, self.window_samples)

        embeddings = self.embed(flat).reshape(batch_size, n_windows, -1)
        return self.head(embeddings, enrolment_embedding)

    def embed_enrolment(self, enrolment):
        """(B, T) enrolment waveform -> (B, 192).

        Call once per batch and reuse. Under no_grad deliberately: the
        enrolment is an input, not something being optimised, so nothing
        upstream of it needs a gradient.
        """
        with torch.no_grad():
            return self.embed(enrolment)

    def describe(self):
        """One-line provenance, for the run log."""
        return json.dumps(dict(
            checkpoint=Path(self.checkpoint_path).name,
            window_seconds=self.window_seconds,
            window_hop_seconds=self.window_hop_seconds,
            audible_db=self.audible_db,
            head_hidden=self.head_shape["hidden"],
            head_layers=self.head_shape["n_layers"],
            frozen_parameters=sum(p.numel() for p in self.parameters()),
        ), sort_keys=True)


def required_answers(target_audible_in_input):
    """What a CORRECT output looks like, in the teacher's two questions.

    target_audible_in_input  (B, W) bool -- was the target speaking there?
    -> (B, W, 2) float, the answers a perfect extractor would produce

    The extraction task written in the teacher's own terms:

        the target must still be audible wherever it was speaking
        a non-target must NOT be audible anywhere

    The second column is therefore a CONSTANT ZERO. That is the whole
    interferer-suppression pressure and it needs no labels at all -- "remove
    the second voice" is unconditional. Only the first column needs to know
    what the input contained.
    """
    target_column = target_audible_in_input.float()
    return torch.stack([target_column, torch.zeros_like(target_column)], dim=-1)
