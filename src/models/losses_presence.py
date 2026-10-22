"""The M2 objective plus the presence gate's two terms. Eq. (12) of
docs/extra/variational-presence-gate.md.

    J = base + beta * KL(q || p) + lam * BCE

`base` is LossBSRNN unchanged, computed on the GATED output: in training that
output was built with one soft draw of the gate (eq. 20), so `base` is the
one-sample estimate of E_q[loss]. The two new terms read the gate's logits.

A SUBCLASS in its own file, for the same reason as losses_state_head.py: the base
class is untouched, and build_loss_fn() only returns this class when the config
sets model.presence_gate, so no old config can reach this code path.

Borrowed, with differences:
  * the ELBO and amortised posterior: Kingma & Welling (2014). Ours replaces the
    log-likelihood with e21's loss (a pseudo-likelihood, eq. 11).
  * beta on the KL: Higgins et al. (2017), beta-VAE. Here beta stands in for the
    unknown kappa of eq. (11), not a disentanglement knob.
  * the supervised term: Kingma, Rezende, Mohamed & Welling (2014), "Semi-
    Supervised Learning with Deep Generative Models". They have one label per
    example; we have one per frame, and only on simulated clips.
The Markov prior and its closed-form KL (eq. 16) are Claude-originated (the
note's AI-use line).
"""

from __future__ import annotations

import torch

from src.models.losses import LossBSRNN
from src.models.presence_gate import (markov_kl, markov_prior, objective,
                                      supervised_bce)


class LossBSRNNPresence(LossBSRNN):
    """LossBSRNN plus `beta * KL + lam * BCE`.

    KL and BCE are SUMS over frames per clip, as in (12), then averaged over
    clips. A 4 s crop has ~500 frames, so both are large next to the base loss;
    beta and lam absorb that scale and must be derived, not guessed.
    """

    # The train loop asks before passing the gate's inputs (train.gate_kwargs).
    # s_other (ACCEPTS_OTHER, inherited True) is passed straight through.
    ACCEPTS_GATE = True

    def __init__(self, *args, beta, lam, d_on, d_off, **kwargs):
        super().__init__(*args, **kwargs)
        assert beta >= 0.0 and lam >= 0.0, f"beta and lam must be >= 0, got {beta}, {lam}"
        self.beta = beta
        self.lam = lam
        # Eqs. (8)-(9). Fixed for the run: d_on and d_off are measured once on
        # the training labels (experiments/results/2026-10-14-activity-runs.json).
        self.a11, self.a01, self.rho = markov_prior(d_on, d_off)

    def __call__(self, s_target, s_output, x_input, crop_absent,
                 gate_logits=None, v=None, v_valid=None, s_output_ungated=None, **base_kwargs):
        """(total, parts). `gate_logits` (B, T) from model(..., return_gate=True);
        `v` (B, T) and `v_valid` (B,) from the loader.

        `s_output_ungated` (B, T_samples) is given in the model's "postfilter" mode
        (decisions-m2.md 2026-10-18) and changes what trains what:

            total = base(ungated) + base(gated) + beta * KL + lam * BCE

        base(ungated) is e21's objective on the extractor's own output: the ONLY
        term whose gradient reaches the separator. base(gated) is the same loss on
        the gated output, built in the model from a detached spectrum, so its
        gradient reaches the gate's 129 weights alone: it is the first term of
        eq. (12) with the Monte Carlo draw (20) replaced by the mean output (21).
        Plain part names (`L_pres`, `L_abs`, ...) describe the GATED output, the
        system as deployed; `*_ungated` describe the extractor alone, comparable
        with e21's own curves.
        """
        # Loud, because a silently skipped term would train e21 with a gate that
        # only the output loss teaches, and be reported as the method failing.
        if gate_logits is None or v is None or v_valid is None:
            raise RuntimeError(
                "LossBSRNNPresence needs gate_logits, v and v_valid. Call the model "
                "with return_gate=True and pass train.gate_kwargs(...) to the loss.")
        assert gate_logits.shape == v.shape, (
            f"gate logits {tuple(gate_logits.shape)} do not line up with labels "
            f"{tuple(v.shape)}: the loader and the model disagree on framing")

        base, parts = super().__call__(s_target, s_output, x_input, crop_absent,
                                       **base_kwargs)
        if s_output_ungated is not None:
            # The extractor's term. Logged under *_ungated; the plain names stay with the system output above.
            base_ungated, parts_ungated = super().__call__(s_target, s_output_ungated, x_input, crop_absent,
                                                           **base_kwargs)
            for k in ("L_pres", "L_MR", "L_abs"):
                parts[f"{k}_ungated"] = parts_ungated[k]
            base = base + base_ungated

        # fp32 regardless of AMP, as every other loss term here.
        logits = gate_logits.float()
        pi = torch.sigmoid(logits)

        # Eq. (16) on EVERY crop: the prior says how speech behaves in time,
        # whether or not this crop has a label.
        kl = markov_kl(pi, self.a11, self.a01, self.rho).mean()

        # Supervised term on LABELLED crops only (v_valid). Absent crops are
        # included: their label is all zeros, which is exactly "gate off".
        valid = v_valid.bool()
        if bool(valid.any()):
            bce = supervised_bce(logits[valid], v[valid]).mean()
        else:
            bce = logits.new_zeros(())

        # Eq. (12).
        total = objective(base, kl, bce, self.beta, self.lam)

        # Logged UNWEIGHTED, so beta and lam can be derived from them the way
        # derive_w_g.py derives wg. NaN BCE when no crop in the batch had a label.
        parts["L_kl"] = float(kl.detach())
        parts["L_bce"] = float(bce.detach()) if bool(valid.any()) else float("nan")
        parts["pi_mean"] = float(pi.mean().detach())
        parts["total"] = float(total.detach())
        return total, parts
