"""Variational presence gate: the prior and the closed-form KL.

Equation numbers refer to docs/extra/variational-presence-gate.md.
Amortised variational posterior and the ELBO: Kingma & Welling (2014), "Auto-Encoding Variational Bayes". The closed-form KL to a Markov prior (16) follows the completely factorised variational inference of Ghahramani & Jordan (1997), "Factorial Hidden Markov Models", §3.4, applied to a two-state chain (literature/novelty-review-presence-gate.md).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PresenceGate(nn.Module):
    """pi_t = sigmoid(w^T LN(h_t) + b), h_t = mean over bands of u_{k,t}.

    Returns logits; pi = sigmoid(logits).

    Eq. (2): the amortised posterior q(z | x, e) (Kingma & Welling, 2014). The band-average + linear design is
    Claude-originated; the band mean follows the D14 state head (src/models/state_head.py).

    LN is layer normalisation over the N features of frame t (Ba, Kiros & Hinton, 2016, "Layer Normalization"), for
    the same reason BSRNN's Estimator normalises this separator output before its mask MLP (src/models/modules.py):
    the separator is pre-norm residual, so the scale of its output is never controlled. Measured 2026-10-14: |h_t| is
    ~2 at random init and 494-760 in e21, and without LN the gate's logits on e21 were +10 to +56 (pi pinned at 1).
    No learned scale/shift: w and b already provide them. Statistics use frame t only, so the gate stays causal.
    """

    # The isolate rng is the seeder for the linear layer, so that the gate is not dependent on the random seed of the rest of the model. 
    # The conv layer is needed for the sliding window calculation, and since the kernel size is one, it is equivalent to a linear layer, but this linear layer has the additional benefit of being able to be used in a sliding window fashion, which is needed for the presence gate.
    def __init__(self, feature_dim=128, init_seed=43, isolate_rng=True, eps=1e-5):
        super().__init__()
        self.eps = eps          # added to the variance in LN, as nn.LayerNorm's default
        if isolate_rng:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(init_seed)
                self.linear = nn.Conv1d(feature_dim, 1, 1)
        else:
            self.linear = nn.Conv1d(feature_dim, 1, 1)

    # essentially I use this layer to take the mean over all the frequency bands, and then I need the logits to be used for the BCE loss.
    def forward(self, u):
        """(B, K, N, T) separator features -> (B, T) logits."""
        assert u.dim() == 4, f"expected (B, K, N, T), got {tuple(u.shape)}"
        h = u.mean(dim=1).float()                          # (B, N, T): h_t, average over bands; fp32 for the statistics
        mu = h.mean(dim=1, keepdim=True)                   # (B, 1, T): mu_t, mean over the N features of frame t
        var = h.var(dim=1, unbiased=False, keepdim=True)   # (B, 1, T): s_t^2, variance over the N features of frame t
        h_hat = (h - mu) / torch.sqrt(var + self.eps)      # (B, N, T): hat h_t, unit scale whatever |h_t| is
        return self.linear(h_hat).squeeze(1)


def gain_from_logits(logits, floor=0.0):
    """pi = sigmoid(logits) -> the per-frame gain floor + (1 - floor) * pi, eq. (21) with a floor.

    floor = 0 is eq. (21) exactly. A floor > 0 maps z = 0 to a gain of `floor` instead of a mute, so a confidently wrong
    "off" frame is attenuated, not deleted. Borrowed: Zhang, Wu & Li (2026, arXiv:2609.30774) attenuate to 0.05 below
    their threshold; here the floor is applied to the soft gain, and in training and inference alike. The KL and BCE
    still read pi, not the gain.
    """
    return floor + (1.0 - floor) * torch.sigmoid(logits)


def apply_gate(spectrogram, gate):
    """Apply the presence gate to the spectrogram.

    S_t(z_t) = z_t (M_t X_t + R_t).

    Training passes the relaxed sample and inference passes pi.
    Eq. (1). Claude-originated design, no paper behind it.
    """
    return spectrogram * gate.unsqueeze(1)


def supervised_bce(logits, v):
    """-sum_t [v_t log pi_t + (1-v_t) log(1-pi_t)], per clip. The supervised term of Kingma, Rezende, Mohamed & Welling (2014), "Semi-Supervised Learning with Deep Generative Models"; they use one label per example, we use one per frame.

    Takes logits, not pi, so log(0) cannot occur. fp32 regardless of AMP, as state_cross_entropy.
    """
    return F.binary_cross_entropy_with_logits(logits.float(), v.float(), reduction="none").sum(-1)


def objective(expected_loss, kl, bce, beta, lam):
    """J = E_q[loss] + beta * KL + lam * BCE, averaged over the batch.

    Each term is (B,) or a scalar. beta = 0 gives (23).
    Eq. (12): the negative ELBO (Kingma & Welling, 2014), with beta on the KL (Higgins et al., 2017) and the supervised
    term of Kingma, Rezende, Mohamed & Welling (2014).
    """
    return (expected_loss + beta * kl + lam * bce).mean()


def markov_prior(d_on, d_off):
    """Mean speech-run and silence-run lengths (frames) -> (A11, A01, rho).

    Eqs. (8)-(9). A Markov state lasts a geometric number of frames (Rabiner, 1989), so A11 = 1 - 1/d_on is the
    maximum-likelihood fit to the mean run. rho is the chain's stationary P(z = 1).
    """
    if d_on <= 1 or d_off <= 1:
        raise ValueError(f"mean run lengths must exceed 1 frame, got d_on={d_on}, d_off={d_off}")
    a11 = 1.0 - 1.0 / d_on
    a01 = 1.0 / d_off
    rho = a01 / (a01 + (1.0 - a11))
    return a11, a01, rho


def bernoulli_kl(a, b, eps=1e-6):
    """KL(Bern(a) || Bern(b)) = a log(a/b) + (1-a) log((1-a)/(1-b)).

    This is the KL divergence for a coin flip, whre we use it for aa single frame.
    Eq. (17): relative entropy (Kullback & Leibler, 1951) for one binary variable.
    """
    a = a.clamp(eps, 1.0 - eps)
    b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
    return a * torch.log(a / b) + (1 - a) * torch.log((1 - a) / (1 - b))


def markov_kl(pi, a11, a01, rho, eps=1e-6):
    """KL(q || p) per clip, q = prod_t Bern(pi_t), p the Markov chain. This is the KL divergence for a whole clip.

    Frame 1 has no previous frame, so it is compared with Bern(rho).
    Eq. (16), derived in the note (Claude-originated). Exact, with no sampling, because q factorises over frames (mean-field).
    """
    first = bernoulli_kl(pi[..., 0], rho, eps)
    prev, cur = pi[..., :-1], pi[..., 1:]
    rest = (prev * bernoulli_kl(cur, a11, eps) + (1 - prev) * bernoulli_kl(cur, a01, eps))
    return first + rest.sum(-1)

def soft_on_off_draw(logits, tau):
    """Draw a relaxed sample from the presence gate, using the reparameterisation trick.

    This is a soft version of the on/off draw, where we use the Gumbel-Softmax trick to sample from a Bernoulli distribution. The temperature tau controls the smoothness of the sample, with lower values giving more discrete samples.
    Binary Concrete: Maddison et al. (2017); Gumbel-Softmax: Jang et al. (2017).

    Takes the gate's logits, log(pi / (1 - pi)), so no log of pi is needed.
    """
    u = torch.rand_like(logits)
    noise = torch.log(u + 1e-20) - torch.log(1 - u + 1e-20)   # logistic noise L_t
    return torch.sigmoid((logits + noise) / tau)