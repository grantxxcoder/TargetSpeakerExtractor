# Variational presence gate: derivation

**Motivation.** On `ami_eval`, e21 is silent on 110/300 clips and misses 45.3 % of target words. Fine-tuning on AMI traded this for leakage (42.0 % against 25.2 %). Goal: a calibrated, per-frame probability that the target is speaking, used as a gate.

**AI-use.** Claude-originated: (1), the design of (2), (11), (22), (25) and the predictions. (16) is Claude's application of Ghahramani & Jordan (1997) §3.4 [4], not a new result. Everything else is cited or standard. Prior art for the gate itself: `literature/novelty-review-presence-gate.md` (closest: USEF-TP, 2025).

## Notation

| Symbol | Meaning |
|---|---|
| $x$, $X_t$ | mixture waveform; its STFT frame $t$ (8 ms hop) |
| $e$, $s$ | enrolment; clean target |
| $M_t$, $R_t$ | e21's mask and residual at frame $t$ |
| $u_{k,t} \in \mathbb{R}^{128}$ | e21 separator output, band $k \in \{1..32\}$ |
| $z_t \in \{0,1\}$ | latent: target active in frame $t$ |
| $v_t \in \{0,1\}$ | true activity label (simulated data) |
| $D$, $d$ | length in frames of one speech run (random variable under the prior); a value it takes |
| $\bar d_{\text{on}}$, $\bar d_{\text{off}}$ | sample mean length in frames of the speech runs (consecutive 1s) and silence runs (consecutive 0s) in the training labels $v_t$; measured once before training |
| $\ell(\hat s, s)$ | e21's existing training loss |

## Model

$$\hat S_t(z_t) = z_t\,(M_t \odot X_t + R_t) \tag{1}$$
*From:* design choice (this work).

$$q_\phi(z_{1:T} \mid x, e) = \prod_{t=1}^{T} \mathrm{Bern}(z_t;\pi_t), \qquad \pi_t = \sigma(w^\top \hat h_t + b), \qquad \hat h_{t,n} = \frac{h_{t,n} - \mu_t}{\sqrt{s_t^2 + \epsilon}}, \qquad h_t = \frac{1}{32}\sum_{k=1}^{32} u_{k,t} \tag{2}$$
with $\mu_t = \frac{1}{128}\sum_{n=1}^{128} h_{t,n}$, $s_t^2 = \frac{1}{128}\sum_{n=1}^{128}(h_{t,n} - \mu_t)^2$ and $\epsilon = 10^{-5}$.
*From:* amortised variational posterior [1]. The factorisation over frames is a mean-field assumption: the posterior cannot represent correlation between neighbouring frames, which a structured posterior keeping the chain would [4] (as in Sadeghi & Alameda-Pineda, 2021). Band-mean pooling is as in the D14 state head (`src/models/state_head.py`). $\hat h_t$ is layer normalisation over the 128 features of frame $t$ [14], for the reason BSRNN's estimator normalises the same separator output before its mask MLP (`src/models/modules.py`): the separator is pre-norm residual, so its output scale is uncontrolled ($\lVert h_t \rVert \approx 2$ at init, 494–760 in e21; without it the gate's logits on e21 were +10 to +56 and $\pi_t$ was pinned at 1, measured 2026-10-14). No learned scale or shift, since $w^\top(\gamma \odot \hat h_t + c) + b = (w \odot \gamma)^\top \hat h_t + (w^\top c + b)$; the gate stays at 129 parameters. Causal, because $u_{k,t}$ depends on $x_{\le t}$ only and $\mu_t$, $s_t$ use frame $t$ only.

## Prior

$$p(z_{1:T}) = p(z_1)\prod_{t=2}^{T} p(z_t \mid z_{1:t-1}) \tag{3}$$
*From:* the chain rule of probability.

$$p(z_t \mid z_{1:t-1}) = p(z_t \mid z_{t-1}) \triangleq A_{z_{t-1} z_t} \tag{4}$$
*From:* the first-order Markov assumption. $A_{ij}$ = probability of state $j$ given previous state $i$; rows sum to 1.

$$p(z_{1:T}) = p(z_1)\prod_{t=2}^{T} A_{z_{t-1} z_t} \tag{5}$$
*From:* (4) substituted into (3).

$$p(d) = P(D = d) = A_{11}^{\,d-1}(1 - A_{11}), \qquad d = 1, 2, \dots \tag{6}$$
*From:* (5). A run of length $d$ stays in state 1 for $d-1$ steps (each $A_{11}$), then leaves ($A_{10} = 1 - A_{11}$). This is the geometric distribution, as in Rabiner (1989) eq. (5) [12].

$$\mathbb{E}[D] = \sum_{d=1}^{\infty} d\,p(d) = \sum_{d=1}^{\infty} d\,A_{11}^{\,d-1}(1-A_{11}) = \frac{1}{1-A_{11}} \tag{7}$$
*From:* the definition of expectation, $\mathbb{E}[X] = \sum_x x\,p(x)$, with (6) substituted.
1. $(1 - A_{11})$ does not depend on $d$: $\mathbb{E}[D] = (1-A_{11})\sum_{d=1}^{\infty} d\,A_{11}^{\,d-1} = (1-A_{11})\big(1 + 2A_{11} + 3A_{11}^2 + \dots\big)$.
2. Geometric series, $0 \le A_{11} < 1$: $\sum_{d=0}^{\infty} A_{11}^{\,d} = 1 + A_{11} + A_{11}^2 + \dots$ Multiplying by $A_{11}$ gives $A_{11} + A_{11}^2 + \dots$, and subtracting gives $(1 - A_{11})\sum_{d=0}^{\infty} A_{11}^{\,d} = 1$, so $\sum_{d=0}^{\infty} A_{11}^{\,d} = \dfrac{1}{1-A_{11}}$.
3. Differentiate both sides of step 2 with respect to $A_{11}$. Left: $\sum_{d=1}^{\infty} d\,A_{11}^{\,d-1} = 1 + 2A_{11} + 3A_{11}^2 + \dots$ Right, by the chain rule: $\dfrac{d}{dA_{11}}(1-A_{11})^{-1} = (1-A_{11})^{-2}$. So $\sum_{d=1}^{\infty} d\,A_{11}^{\,d-1} = \dfrac{1}{(1-A_{11})^2}$.
4. Substitute step 3 into step 1: $\mathbb{E}[D] = (1-A_{11})\cdot\dfrac{1}{(1-A_{11})^2} = \dfrac{1}{1-A_{11}}$.


$$A_{11} = 1 - \frac{1}{\bar d_{\text{on}}}, \quad A_{10} = \frac{1}{\bar d_{\text{on}}}, \quad A_{00} = 1 - \frac{1}{\bar d_{\text{off}}}, \quad A_{01} = \frac{1}{\bar d_{\text{off}}} \tag{8}$$
*From:* (7), with the expectation replaced by the sample mean (method of moments, route 1), which is also the maximum-likelihood estimate (route 2). Both routes give the same $A_{11}$. $A$ is fixed, not learned.

**Route 1: match the mean (method of moments).**
1. The mean run length under the chain, (7): $\mathbb{E}[D] = \dfrac{1}{1-A_{11}}$. *From:* (6) and the definition of expectation; Rabiner (1989) eqs. (6a)–(6b) [12].
2. The sample mean of the $n$ speech runs with lengths $d_1, \dots, d_n$ (runs of consecutive 1s in the training labels $v_t$): $\bar d_{\text{on}} = \frac{1}{n}\sum_{r=1}^{n} d_r$. *From:* the definition of the sample mean.
3. Set the model's mean equal to the measured mean: $\dfrac{1}{1-A_{11}} = \bar d_{\text{on}}$. *From:* the method of moments: choose the parameter so the model reproduces the measured statistic.
4. Take the reciprocal of both sides: $1 - A_{11} = \dfrac{1}{\bar d_{\text{on}}}$.
5. Rearrange: $A_{11} = 1 - \dfrac{1}{\bar d_{\text{on}}}$.

**Route 2: maximum likelihood.**
1. The probability of the $n$ observed runs, each geometric (6) and treated as independent: $\mathcal{L}(A_{11}) = \prod_{r=1}^{n} A_{11}^{\,d_r - 1}(1 - A_{11})$. *From:* (6); the probability of independent events is the product of their probabilities.
2. Take logs: $\log \mathcal{L}(A_{11}) = \Big(\sum_{r=1}^{n} d_r - n\Big)\log A_{11} + n\log(1 - A_{11})$. *From:* $\log(ab) = \log a + \log b$ and $\log a^k = k\log a$.
3. Set the derivative to zero: $\dfrac{d}{dA_{11}}\log \mathcal{L}(A_{11}) = \dfrac{\sum_{r} d_r - n}{A_{11}} - \dfrac{n}{1 - A_{11}} = 0$. *From:* $\frac{d}{dx}\log x = \frac{1}{x}$, and the chain rule for $\log(1 - x)$.
4. Multiply through by $A_{11}(1 - A_{11})$: $(1 - A_{11})\Big(\sum_{r} d_r - n\Big) = n A_{11}$.
5. Expand: $\sum_{r} d_r - n - A_{11}\sum_{r} d_r + n A_{11} = n A_{11}$.
6. The $nA_{11}$ terms cancel: $\sum_{r} d_r - n = A_{11}\sum_{r} d_r$.
7. Solve, then use $\dfrac{n}{\sum_r d_r} = \dfrac{1}{\bar d_{\text{on}}}$ from route 1 step 2: $A_{11} = 1 - \dfrac{n}{\sum_{r} d_r} = 1 - \dfrac{1}{\bar d_{\text{on}}}$.

**The other three entries.**
1. Row sum: $A_{10} = 1 - A_{11} = 1/\bar d_{\text{on}}$.
2. Silence runs (consecutive 0s) give $A_{00} = 1 - 1/\bar d_{\text{off}}$ and $A_{01} = 1/\bar d_{\text{off}}$ the same way.

**Values.** $\bar d_{\text{on}} = 596.34$ frames (4.77 s) and $\bar d_{\text{off}} = 489.53$ frames (3.92 s): `sir0_train`, 9,955 trials, each speaker in turn as the target (`both`, as in training), all examples, D14 label with a $1 \times T_{60}$ tail (`experiments/results/2026-10-14-activity-runs.json`, `sim_tail1`). So $A_{11} = 0.99832$: in a frame where the target speaks, the prior gives a 0.17 % chance that the turn ends before the next frame.

**Caveat.** Route 2 assumes every run is seen whole. A run cut by a trial's start or end is shorter than the real turn, so $\bar d_{\text{on}}$ slightly understates it. Dropping the edge runs moves $\bar d_{\text{on}}$ from 596.34 to 595.21 frames, so the effect on $A_{11}$ is negligible. It is not negligible for silences: interior pauses average 54.0 frames (0.43 s) against 489.53 overall, because most silent time is leading or trailing silence and target-absent trials.

$$\rho = \rho A_{11} + (1-\rho)A_{01} \;\Longrightarrow\; \rho = \frac{A_{01}}{A_{01} + A_{10}}, \qquad p(z_1 = 1) = \rho \tag{9}$$
*From:* the stationary distribution of the chain. Frame 1 has no previous frame; $\rho = P(z_t{=}1)$ is chosen to be the same at every $t$.
1. Law of total probability: $P(z_t{=}1) = P(z_{t-1}{=}1)\,A_{11} + P(z_{t-1}{=}0)\,A_{01}$.
2. Stationarity, $P(z_t{=}1) = P(z_{t-1}{=}1) = \rho$: $\rho = \rho A_{11} + (1-\rho)A_{01}$.
3. Rearrange: $\rho(1 - A_{11}) = (1-\rho)A_{01}$, and with $1 - A_{11} = A_{10}$: $\rho A_{10} = A_{01} - \rho A_{01}$.
4. Solve: $\rho = \dfrac{A_{01}}{A_{01} + A_{10}}$.

Check with (8): $\rho = \dfrac{1/\bar d_{\text{off}}}{1/\bar d_{\text{off}} + 1/\bar d_{\text{on}}} = \dfrac{\bar d_{\text{on}}}{\bar d_{\text{on}} + \bar d_{\text{off}}}$, the share of time spent speaking.

## Objective

### Quantities

| Symbol | Meaning | Known during training? |
|---|---|---|
| $x$ | the mixture waveform: target + other speaker + noise | yes (input) |
| $e$ | the enrolment: a separate clean recording of the target's voice | yes (input) |
| $s$ | the clean target waveform: what the output should match | yes (simulated data) |
| $z = (z_1, \dots, z_T)$ | $z_t = 1$ if the target speaks in frame $t$, else $0$; written $z$ for $z_{1:T}$ | **no**: latent, never given to the model |
| $\hat s(z)$ | the model's output waveform: inverse STFT of $\hat S_t(z_t)$ in (1) | computed |
| $\theta$, $\phi$ | e21's weights; the gate's weights $(w, b)$ | learned |
| $p(z)$ | the prior over $z$, (5) | fixed |
| $q_\phi(z \mid x, e)$ | the approximate posterior over $z$, (2) | computed |

### Step 1: what training should maximise

$$\max_{\theta,\phi}\; \log p_\theta(s \mid x, e)$$
*From:* maximum likelihood. This quantity is called the **evidence**.

### Step 2: bring in $z$

$$p_\theta(s \mid x, e) = \sum_{z} p_\theta(s \mid x, e, z)\, p(z)$$
*From:* the law of total probability over all $2^T$ values of $z$. The prior $p(z)$ does not depend on $x$ or $e$.

**Problem:** $2^T$ terms ($2^{1000}$ for an 8 s clip), so this sum cannot be computed.

### Step 3: rewrite the sum as an expectation under $q_\phi(z \mid x, e)$

$$p_\theta(s \mid x, e) = \sum_{z} p_\theta(s \mid x, e, z)\, p(z) = \sum_{z} q_\phi(z \mid x, e)\,\frac{p_\theta(s \mid x, e, z)\, p(z)}{q_\phi(z \mid x, e)} = \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\!\left[\frac{p_\theta(s \mid x, e, z)\, p(z)}{q_\phi(z \mid x, e)}\right]$$
*From:* the second equality multiplies each term by $q_\phi(z \mid x,e)/q_\phi(z \mid x,e) = 1$, valid where $q_\phi(z \mid x,e) > 0$. The third is the definition of expectation: a sum over $z$ of $q_\phi(z \mid x,e)$ times a quantity that depends on $z$ equals the expectation of that quantity for $z \sim q_\phi(z \mid x,e)$. Here the quantity is $p_\theta(s \mid x,e,z)\,p(z)/q_\phi(z \mid x,e)$.

### Step 4: Jensen's inequality

$$\log p_\theta(s \mid x, e) = \log \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\!\left[\frac{p_\theta(s \mid x, e, z)\, p(z)}{q_\phi(z \mid x, e)}\right] \;\ge\; \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\!\left[\log \frac{p_\theta(s \mid x, e, z)\, p(z)}{q_\phi(z \mid x, e)}\right]$$
*From:* Jensen. For a concave function $f$, $f(\mathbb{E}[Y]) \ge \mathbb{E}[f(Y)]$; here $f = \log$, which is concave.

### Step 5: expand the logarithm

$$\mathbb{E}_{z \sim q_\phi(z \mid x, e)}\!\left[\log \frac{p_\theta(s \mid x, e, z)\, p(z)}{q_\phi(z \mid x, e)}\right] = \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\log p_\theta(s \mid x, e, z)\big] - \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\log q_\phi(z \mid x, e) - \log p(z)\big]$$
*From:* $\log(ab/c) = \log a + \log b - \log c$, then linearity of expectation.

$$\mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\log q_\phi(z \mid x, e) - \log p(z)\big] = \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
*From:* the definition of KL divergence (13).

### Result: the evidence lower bound (ELBO)

$$\log p_\theta(s \mid x, e) \;\ge\; \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\log p_\theta(s \mid x, e, z)\big] \;-\; \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big) \tag{10}$$
*From:* steps 1–5 [1].
- The first term measures how well the output matches $s$, averaged over $z \sim q_\phi(z \mid x,e)$.
- The second measures how far the posterior is from the prior.
- The gap is $\log p_\theta(s \mid x,e) - \text{RHS} = \mathrm{KL}\big(q_\phi(z \mid x,e)\,\|\,p_\theta(z \mid s,x,e)\big) \ge 0$, which is zero only if $q_\phi(z \mid x,e) = p_\theta(z \mid s,x,e)$.

### Step 6: make the first term computable

(10) needs the **probability** $p_\theta(s \mid x,e,z)$, but training only gives the **loss** $\ell(\hat s(z), s)$. Build the probability from the loss, then take its expected log.

1. Start from the Gaussian density for one sample, mean $\hat s$, variance $\sigma^2$:
$$\mathcal{N}(s;\hat s,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(s-\hat s)^2}{2\sigma^2}\right)$$
2. Rename: $\ell(\hat s, s) = (s-\hat s)^2$ (squared error), $\kappa = 2\sigma^2$, $Z = \sqrt{2\pi\sigma^2}$:
$$\mathcal{N}(s;\hat s,\sigma^2) = \frac{1}{Z}\exp\!\left(-\frac{\ell(\hat s, s)}{\kappa}\right)$$
3. $Z$ is the area under the unnormalised curve (the Gaussian integral). Substituting $u = s' - \hat s$ shows it does not depend on $\hat s$:
$$Z = \int \exp\!\left(-\frac{(s'-\hat s)^2}{2\sigma^2}\right) ds' = \int \exp\!\left(-\frac{u^2}{2\sigma^2}\right) du = \sqrt{2\pi\sigma^2}$$
4. **Assumption** (this work): keep this form, but put e21's loss in place of squared error and the model's output $\hat s(z)$ in place of $\hat s$. It stays positive ($\exp > 0$), lower loss stays more probable, and it still integrates to 1 (division by $Z$):
$$p_\theta(s \mid x, e, z) = \frac{1}{Z}\exp\!\left(-\frac{\ell(\hat s(z), s)}{\kappa}\right), \qquad Z = \int \exp\!\left(-\frac{\ell(\hat s(z), s')}{\kappa}\right) ds' \tag{11}$$
5. Take logs. The log of a product is the sum of the logs:
$$\log p_\theta(s \mid x, e, z) = \log\frac{1}{Z} + \log\exp\!\left(-\frac{\ell(\hat s(z), s)}{\kappa}\right)$$
6. $\log(1/Z) = -\log Z$ and $\log(\exp y) = y$:
$$\log p_\theta(s \mid x, e, z) = -\frac{\ell(\hat s(z), s)}{\kappa} - \log Z$$
7. Average over $z \sim q_\phi(z \mid x,e)$, by the definition of expectation, $\mathbb{E}[f(z)] = \sum_z q(z)\,f(z)$:
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p_\theta(s \mid x, e, z)\big] = \sum_z q_\phi(z \mid x,e)\left[-\frac{\ell(\hat s(z), s)}{\kappa} - \log Z\right]$$
8. Multiply out. $1/\kappa$ is a constant. **Assume $Z$ does not depend on $\hat s(z)$** (true for squared error, item 3), so $Z$ is the same for every $z$ and $\log Z$ also leaves the sum:
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p_\theta(s \mid x, e, z)\big] = -\frac{1}{\kappa}\sum_z q_\phi(z \mid x,e)\,\ell(\hat s(z), s) \;-\; \log Z \sum_z q_\phi(z \mid x,e)$$
9. Probabilities sum to 1, $\sum_z q_\phi(z \mid x,e) = 1$, and the remaining sum is an expectation (definition, as in item 7). **Result, used in Step 7:**
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p_\theta(s \mid x, e, z)\big] = -\frac{1}{\kappa}\,\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] - \log Z$$

- **Exactness.** Exact for squared error, where $Z = \sqrt{2\pi\sigma^2}$ per sample, independent of $\hat s(z)$. Not exact here: $L_{\text{pres}}$ is a floored SI-SDR (`src/models/losses.py`), which gives $\ell(\hat s, c\,s) = \ell(\hat s, s)$ for any $c > 0$, so the integral $Z$ diverges. (11) is a pseudo-likelihood, and (10) is then not a strict bound on a true log-probability.
- **Theory.** $\frac{1}{Z}\exp(-\ell/\kappa)$ is the Gibbs form [9], the loss-based (Gibbs) posterior of Bissiri, Holmes & Walker (2016) [13]: the maximum-entropy distribution with a fixed expected loss [10]. Maximise $-\sum p\log p$ subject to $\sum p = 1$ and $\sum p\,\ell = U$. The Lagrangian $-\sum p\log p - \alpha(\sum p - 1) - \lambda(\sum p\,\ell - U)$ has stationary point $-\log p - 1 - \alpha - \lambda\ell = 0$, so $p = e^{-1-\alpha}e^{-\lambda\ell} = \frac{1}{Z}e^{-\lambda\ell}$, with $\lambda = 1/\kappa$ [11].

### Step 7: the objective to minimise

Training minimises a loss by gradient descent, so turn the ELBO (10) into one.

1. Call the right-hand side of (10) $\mathcal{L}(\theta, \phi)$:
$$\mathcal{L}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p_\theta(s \mid x, e, z)\big] - \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
2. Substitute the Step 6 result (item 9) for the first term:
$$\mathcal{L}(\theta, \phi) = -\frac{1}{\kappa}\,\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] - \log Z - \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
3. Maximising $\mathcal{L}$ is the same as minimising $-\mathcal{L}$. Negate every term:
$$-\mathcal{L}(\theta, \phi) = \frac{1}{\kappa}\,\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] + \log Z + \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
4. Multiply by $\kappa > 0$. Scaling by a positive constant does not move the minimum ($\arg\min_\theta c\,f(\theta) = \arg\min_\theta f(\theta)$ for $c > 0$):
$$-\kappa\,\mathcal{L}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] + \kappa\log Z + \kappa\,\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
5. Drop $\kappa\log Z$. It does not depend on $\theta$ or $\phi$ (Step 6 item 8), so its gradient is zero and the minimum does not move:
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] + \kappa\,\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
6. $\kappa$ is unknown (the SI-SDR loss has no $\sigma^2$), so rename it $\beta$ and tune it [2]. $\beta$ is the single temperature: a separate β-VAE weight on top of $\kappa$ would not be identifiable, since only their product scales the KL. From here the objective is $-\mathcal{L}$ only when $\beta = \kappa$:
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] + \beta\,\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big)$$
7. Add the optional supervised term on labelled frames [3]: the binary cross-entropy between $\pi_t$ and the label $v_t$, weighted by $\lambda$ ($\lambda = 0$ turns it off). This term is not part of the ELBO. **Result:**
$$\mathcal{J}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\ell(\hat s(z), s)\big] + \beta\,\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big) - \lambda\sum_{t=1}^{T}\big[v_t\log\pi_t + (1-v_t)\log(1-\pi_t)\big] \tag{12}$$

| Term | What it is | How it is computed |
|---|---|---|
| $\mathbb{E}_{z \sim q_\phi(z \mid x,e)}[\ell(\hat s(z), s)]$ | e21's existing loss on the gated output | one relaxed sample $\tilde z$ per example, (20) |
| $\beta\,\mathrm{KL}(q_\phi(z \mid x,e) \Vert p(z))$ | penalty for $\pi_t$ disagreeing with the prior | exactly, from (16); no sampling |
| $-\lambda\sum_t [v_t\log\pi_t + (1-v_t)\log(1-\pi_t)]$ | direct supervision of $\pi_t$ | labels $v_t$ (simulated data only) |

## Closed-form KL

The KL term in (12) is a sum over all $2^T$ on/off patterns. Reduce it to a sum over frames that uses only $\pi_t$, $A$ and $\rho$.

1. Definition of KL divergence, written as two expectations:
$$\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big) = \mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log q_\phi(z \mid x, e)\big] - \mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p(z)\big] \tag{13}$$
2. Take the log of (2). The log of a product is the sum of the logs, and $z_t \in \{0,1\}$ picks out one term of each Bernoulli:
$$\log q_\phi(z \mid x, e) = \sum_{t=1}^{T}\big[z_t\log\pi_t + (1-z_t)\log(1-\pi_t)\big]$$
3. Take the expectation. The expectation of a sum is the sum of the expectations, and $\mathbb{E}[z_t] = \pi_t$ (the mean of a Bernoulli is its probability of 1):
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log q_\phi(z \mid x, e)\big] = \sum_{t=1}^{T}\big[\pi_t\log\pi_t + (1-\pi_t)\log(1-\pi_t)\big] \tag{14}$$
4. Now term 2 Take the log of the prior (5):
$$\log p(z) = \log p(z_1) + \sum_{t=2}^{T}\log A_{z_{t-1} z_t}$$
5. Write each piece so it holds for any binary values. **Why:** term 2 needs $\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log A_{z_{t-1} z_t}\big]$, but the random variables sit in the *subscript* of $A$, and an expectation cannot be taken through a subscript. Rewrite it as an ordinary expression in $z_{t-1}$ and $z_t$: the same trick as item 2 for term 1. $z_1$ picks $\rho$ or $1-\rho$; exactly one of the four products below equals 1 and picks the matching transition:
$$\log p(z_1) = z_1\log\rho + (1-z_1)\log(1-\rho)$$
$$\log A_{z_{t-1}z_t} = z_{t-1}z_t\log A_{11} + z_{t-1}(1-z_t)\log A_{10} + (1-z_{t-1})z_t\log A_{01} + (1-z_{t-1})(1-z_t)\log A_{00}$$
*From:* $z_t \in \{0,1\}$. Check, all four cases:

   | $z_{t-1}$ | $z_t$ | products (11, 10, 01, 00) | what survives |
   |---|---|---|---|
   | 1 | 1 | 1, 0, 0, 0 | $\log A_{11}$ |
   | 1 | 0 | 0, 1, 0, 0 | $\log A_{10}$ |
   | 0 | 1 | 0, 0, 1, 0 | $\log A_{01}$ |
   | 0 | 0 | 0, 0, 0, 1 | $\log A_{00}$ |

6. Take the expectation. Everything is now linear in the products. **Mean-field step:** under (2), $z_{t-1}$ and $z_t$ are independent, so $\mathbb{E}[z_{t-1}z_t] = \mathbb{E}[z_{t-1}]\,\mathbb{E}[z_t] = \pi_{t-1}\pi_t$ (*From:* for independent random variables, the expectation of a product is the product of the expectations), and the same for the other three products, e.g. $\mathbb{E}[z_{t-1}(1-z_t)] = \pi_{t-1}(1-\pi_t)$. Frame 1 uses $\mathbb{E}[z_1] = \pi_1$. Without mean-field, $\mathbb{E}[z_{t-1}z_t]$ would need the joint probability of neighbouring frames under $q$, and (15) would no longer follow from the $\pi_t$ alone:
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\log p(z)\big] = \pi_1\log\rho + (1-\pi_1)\log(1-\rho) + \sum_{t=2}^{T}\Big[\pi_{t-1}\pi_t\log A_{11} + \pi_{t-1}(1-\pi_t)\log A_{10} + (1-\pi_{t-1})\pi_t\log A_{01} + (1-\pi_{t-1})(1-\pi_t)\log A_{00}\Big] \tag{15}$$
7. Subtract (15) from (14), as (13) says. Frame 1, using $\log a - \log b = \log(a/b)$:
$$\pi_1\log\frac{\pi_1}{\rho} + (1-\pi_1)\log\frac{1-\pi_1}{1-\rho}$$
8. Frames $t \ge 2$. The (14) term has no $\pi_{t-1}$; the (15) term does. `CLAUDE STEP` Multiply the (14) term by $1 = \pi_{t-1} + (1-\pi_{t-1})$, subtract, and group by weight:
$$\pi_{t-1}\Big[\pi_t\log\frac{\pi_t}{A_{11}} + (1-\pi_t)\log\frac{1-\pi_t}{A_{10}}\Big] + (1-\pi_{t-1})\Big[\pi_t\log\frac{\pi_t}{A_{01}} + (1-\pi_t)\log\frac{1-\pi_t}{A_{00}}\Big]$$
9. Each bracket has the shape of the KL between two coin flips. Apply (13) to one binary variable with $P(z{=}1) = a$ under $q$ and $b$ under $p$; the expectation is over $z = 1$ (probability $a$) and $z = 0$ (probability $1-a$):
$$\mathrm{KL}\big(\mathrm{Bern}(a)\,\|\,\mathrm{Bern}(b)\big) = a\log\frac{a}{b} + (1-a)\log\frac{1-a}{1-b} \tag{17}$$
10. Rows of $A$ sum to 1, so $A_{10} = 1 - A_{11}$ and $A_{00} = 1 - A_{01}$. Item 7 is then (17) with $(\pi_1, \rho)$, and the brackets in item 8 are (17) with $(\pi_t, A_{11})$ and $(\pi_t, A_{01})$. **Result:**
$$\mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big) = \mathrm{KL}\big(\mathrm{Bern}(\pi_1)\,\|\,\mathrm{Bern}(\rho)\big) + \sum_{t=2}^{T}\Big[\pi_{t-1}\,\mathrm{KL}\big(\mathrm{Bern}(\pi_t)\,\|\,\mathrm{Bern}(A_{11})\big) + (1-\pi_{t-1})\,\mathrm{KL}\big(\mathrm{Bern}(\pi_t)\,\|\,\mathrm{Bern}(A_{01})\big)\Big] \tag{16}$$
*From:* the completely factorised variational inference of Ghahramani & Jordan (1997) §3.4 [4], applied to a two-state chain; the expected log-transition cross-term is theirs. Exact, with no sampling. Checked against brute-force enumeration of all $2^T$ patterns in `tests/test_presence_gate.py`.

## Training and inference

### Training: a random draw the gradient can pass through

The first term of (12), $\mathbb{E}_{z \sim q_\phi(z \mid x,e)}[\ell(\hat s(z), s)]$, is an average over $2^T$ on/off patterns. Estimate it with one random pattern per clip, in a way that still sends e21's loss back to $\pi_t$. (The KL term needs none of this: it uses (16) exactly.)

1. Estimate the average with one random pattern (Monte Carlo):
$$\mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big] \approx \ell(\hat s(z), s), \qquad z \sim q_\phi(z \mid x,e)$$
2. Draw each $z_t$ from uniform noise $U_t \sim \mathcal{U}(0,1)$; it is 1 with probability $\pi_t$:
$$z_t = \mathbf{1}[U_t > 1 - \pi_t]$$
**Problem:** $z_t$ is a step in $\pi_t$, so $\partial z_t / \partial \pi_t = 0$ almost everywhere, and e21's loss sends no gradient to the gate.
3. Turn the uniform noise into logistic noise, $L_t = \log U_t - \log(1-U_t)$. Its CDF is the sigmoid, since $L_t \le l \iff U_t \le \frac{e^l}{1+e^l}$:
$$P(L_t \le l) = \frac{e^l}{1+e^l} = \sigma(l)$$
4. Write the same draw with $L_t$, where $H(y) = 1$ if $y > 0$, else $0$. $L_t$ increases with $U_t$, so $U_t > 1 - \pi_t \iff L_t > \log\frac{1-\pi_t}{\pi_t} = -\log\frac{\pi_t}{1-\pi_t}$:
$$z_t = H\!\left(\log\frac{\pi_t}{1-\pi_t} + L_t\right)$$
5. Replace the step $H(y)$ by a steep sigmoid $\sigma(y/\tau)$, with temperature $\tau > 0$; as $\tau \to 0$, $\sigma(y/\tau) \to H(y)$. Large $\tau$ gives smooth, low-variance gradients but draws far from 0 and 1 (biased); small $\tau$ gives near 0/1 draws but noisy gradients. $\tau$ is annealed from 1 to 0.5. **Result:**
$$\tilde z_t = \sigma\!\left(\frac{\log\frac{\pi_t}{1-\pi_t} + \log U_t - \log(1-U_t)}{\tau}\right), \qquad U_t \sim \mathcal{U}(0,1) \tag{20}$$
*From:* the binary Concrete relaxation [5, 6]. $U_t$ does not depend on $\phi$, so $\tilde z_t$ is a smooth function of $\pi_t$ and the gradient passes through (the reparameterisation trick [1]). $\tilde z$ replaces $z$ in item 1, one draw per clip. **The training objective is therefore a relaxed ELBO-style objective, not an ELBO:** a relaxed sample in the extraction term with the analytic discrete KL (16) is not guaranteed to be a lower bound (Maddison et al., 2017, App. C.2 [5]).

#### Why one random draw is enough

1. The Monte Carlo estimator with $L$ draws $z^{(1)}, \dots, z^{(L)}$ from $q$, its mean and its variance [12]:
$$\hat f = \frac{1}{L}\sum_{l=1}^{L} f(z^{(l)}), \qquad \mathbb{E}[\hat f] = \mathbb{E}[f], \qquad \mathrm{var}[\hat f] = \frac{1}{L}\,\mathbb{E}\big[(f - \mathbb{E}[f])^2\big]$$
2. Take $L = 1$ and $f(z) = \ell(\hat s(z), s)$. By the definition of expectation, the one-draw estimate in Training item 1 equals the true average on average (**unbiased**):
$$\mathbb{E}_{z^{(1)} \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z^{(1)}), s)\big] = \sum_z q_\phi(z \mid x,e)\,\ell(\hat s(z), s) = \mathbb{E}_{z \sim q_\phi(z \mid x,e)}\big[\ell(\hat s(z), s)\big]$$
3. One draw is noisy: the variance in item 1 is not divided down. Each training step averages over the clips in the batch, and training runs for thousands of steps, so the noise averages out. Kingma & Welling also use one draw per example (with a minibatch of 100) [1].
4. The gradient. In (20) the noise $U = (U_1, \dots, U_T)$ does not depend on $\phi$, so the gradient moves inside the average, and the gradient of one draw is unbiased too. This swap is not allowed for $z \sim \mathrm{Bern}(\pi_t)$ drawn directly, whose distribution depends on $\phi$:
$$\nabla_\phi\,\mathbb{E}_{U}\big[\ell(\hat s(\tilde z), s)\big] = \mathbb{E}_{U}\big[\nabla_\phi\,\ell(\hat s(\tilde z), s)\big]$$
5. Caveat: with $\tau > 0$ this is unbiased for the smoothed objective, not for the exact 0/1 one. That gap is the bias $\tau$ controls.

### Inference: one fixed output

1. Under (1), the output for frame $t$ is random:
$$Y_t = z_t\,(M_t \odot X_t + R_t), \qquad z_t \sim \mathrm{Bern}(\pi_t)$$
2. A live system needs one fixed output $a$. Choose the $a$ closest to $Y_t$ on average, in squared error:
$$\min_a\; \mathbb{E}\big[(Y_t - a)^2\big]$$
3. Expand (for each real and imaginary part), $\mathbb{E}[(Y_t - a)^2] = \mathbb{E}[Y_t^2] - 2a\,\mathbb{E}[Y_t] + a^2$, and set the derivative $-2\,\mathbb{E}[Y_t] + 2a$ to zero:
$$a = \mathbb{E}[Y_t]$$
4. $M_t \odot X_t + R_t$ does not depend on $z_t$, so it leaves the expectation, and $\mathbb{E}[z_t] = \pi_t$. **Result:**
$$\hat S_t = \mathbb{E}_{z_t \sim \mathrm{Bern}(\pi_t)}\big[z_t\,(M_t \odot X_t + R_t)\big] = \pi_t\,(M_t \odot X_t + R_t) \tag{21}$$

## Special cases (ablation arms)

$$A_{11} = A_{01} = \rho = \tfrac12 \;\Longrightarrow\; \mathrm{KL}\big(q_\phi(z \mid x, e)\,\|\,p(z)\big) = \sum_{t=1}^{T}\Big[\pi_t\log(2\pi_t) + (1-\pi_t)\log\big(2(1-\pi_t)\big)\Big] = \sum_{t=1}^{T}\Big[\log 2 + \pi_t\log\pi_t + (1-\pi_t)\log(1-\pi_t)\Big] \tag{22}$$
*From:* (16) and (17) with every Bernoulli parameter $= \tfrac12$. The weights $\pi_{t-1}$ and $1-\pi_{t-1}$ then multiply identical terms and sum to 1. This is $\log 2$ minus the binary entropy of $\pi_t$: no temporal structure, which isolates the Markov chain. Derived here.

$$\beta = 0 \;\Longrightarrow\; \mathcal{J}(\theta, \phi) = \mathbb{E}_{z \sim q_\phi(z \mid x, e)}\big[\ell(\hat s(z), s)\big] - \lambda\sum_{t=1}^{T}\big[v_t\log\pi_t + (1-v_t)\log(1-\pi_t)\big] \tag{23}$$
*From:* (12). A gate with no prior, which isolates the variational term.

## Evaluation

$$\mathrm{BS} = \frac{1}{N}\sum_{t}(\pi_t - v_t)^2 \tag{24}$$
*From:* the Brier score [7]: mean squared error between predicted probability and the 0/1 outcome. Range [0, 1]; lower is better.
1. Strictly proper. If $P(v_t{=}1) = p$, then $\mathbb{E}[\mathrm{BS}] = p(\pi-1)^2 + (1-p)\pi^2$, and $\frac{d}{d\pi} = 2(\pi - p) = 0$ gives $\pi = p$.
2. Constant predictor at the base rate $\rho$: $\mathrm{BS} = \rho(1-\rho)^2 + (1-\rho)\rho^2 = \rho(1-\rho)$. A useful gate must beat this.
3. That predictor has ECE ≈ 0, so ECE alone cannot reject it; BS can.

$$\mathrm{ECE} = \sum_{m=1}^{15}\frac{|B_m|}{N}\left|\frac{1}{|B_m|}\sum_{t\in B_m} v_t - \frac{1}{|B_m|}\sum_{t\in B_m}\pi_t\right|, \qquad S(\pi) = \frac{1}{T\Delta}\sum_{t=2}^{T}\Big|\mathbf{1}[\pi_t \ge \tfrac12] - \mathbf{1}[\pi_{t-1} \ge \tfrac12]\Big| \tag{25}$$
*From:* ECE uses 15 equal-width bins $B_m$ [8]. The switch rate $S$ (with $\Delta$ = 8 ms) is defined here.

## Registered predictions

Arms: A = (12) in full; B = (23); C = (22), optional. Each is a 6-epoch fine-tune from e21, selected on `sir0_val`.

| | Pass criterion | Data |
|---|---|---|
| P1 | $\mathrm{ECE}(A) < \mathrm{ECE}(B)$; speaker-clustered paired 95 % CI below 0 | `sir0_val` |
| P2 | $\lvert S(\pi_A) - S(v)\rvert < \lvert S(\pi_B) - S(v)\rvert$ | `sir0_val` |
| P3 | silent outputs < 110/300, target words missed < 45.25 %, mean leak < 40.70 %; scored once | `ami_eval` |
| P4 | LCF-WER(A) − LCF-WER(e21) < +8 points | `sir0_val` |

P3 and P4 use the Gemini judge: `gemini-3.7-flash`, audio-in / text-out, prompt `d118b7d3bf30`.

## References

1. D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," *ICLR*, 2014.
2. I. Higgins et al., "β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework," *ICLR*, 2017.
3. D. P. Kingma, D. J. Rezende, S. Mohamed and M. Welling, "Semi-Supervised Learning with Deep Generative Models," *NeurIPS*, 2014.
4. Z. Ghahramani and M. I. Jordan, "Factorial Hidden Markov Models," *Machine Learning*, 29, 245–273, 1997, doi:10.1023/A:1007425814087. §3.4, completely factorised variational inference (full text checked 2026-10-14).
5. C. J. Maddison, A. Mnih and Y. W. Teh, "The Concrete Distribution," *ICLR*, 2017.
6. E. Jang, S. Gu and B. Poole, "Categorical Reparameterization with Gumbel-Softmax," *ICLR*, 2017.
7. G. W. Brier, "Verification of Forecasts Expressed in Terms of Probability," *Monthly Weather Review*, 78(1), 1950.
8. C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, "On Calibration of Modern Neural Networks," *ICML*, 2017.
9. J. W. Gibbs, *Elementary Principles in Statistical Mechanics*, Scribner's, 1902.
10. E. T. Jaynes, "Information Theory and Statistical Mechanics," *Physical Review*, 106(4), 620–630, 1957.
11. Y. LeCun, S. Chopra, R. Hadsell, M. Ranzato and F. J. Huang, "A Tutorial on Energy-Based Learning," in *Predicting Structured Data*, MIT Press, 2006.
12. L. R. Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proceedings of the IEEE*, 77(2), 257–286, 1989.
13. P. G. Bissiri, C. C. Holmes and S. G. Walker, "A General Framework for Updating Belief Distributions," *JRSS-B*, 78(5), 1103–1130, 2016, doi:10.1111/rssb.12158.
14. J. L. Ba, J. R. Kiros and G. E. Hinton, "Layer Normalization," arXiv:1607.06450, 2016.
12. C. M. Bishop, *Pattern Recognition and Machine Learning*, Springer, 2006. §11.1, eqs. (11.2)–(11.3).
