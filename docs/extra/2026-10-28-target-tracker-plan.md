# Target tracker: plan (2026-10-28)

Start a new session with: **"Read docs/extra/2026-10-28-target-tracker-plan.md and do step 0 only."**
Out-of-curiosity work after the report. It is not a thesis result unless the author decides so (submission 2026-11-05).

## Idea in plain words
While the target talks alone, the extractor knows how loud the target is in each frequency band. When the other speaker comes in, predict the target's loudness from that recent history, compare it with what the network outputs, and turn the output down where it is louder than the target should be.
A Kalman filter per band does exactly this: predict from the past, correct with the current frame, each weighted by how much it is trusted. The network learns how far to trust its own output in each frame, which should be low during overlap.
Difference from the old gate: the gate asked "is the target speaking?" and stayed open during overlap. This asks "how much of this band is the target?", so it acts during overlap.

## Model
Symbols:
- $t$ frame (8 ms hop); $k = 1, \dots, 32$ band; $B_k$ the bins of band $k$ (`src/models/bands.py`).
- $\hat{S}_{f,t}$ the network's estimate, $M \otimes X + R$ (`estimator_forward` in `src/models/modules.py`).
- $o_{k,t} = \log\big(\sum_{f \in B_k} |\hat{S}_{f,t}|^2 + \epsilon\big)$, the log band energy of the estimate; $\epsilon$ a fixed floor, set in the config.
- $s_{k,t}$ the log band energy of the clean target, defined the same way (latent; known only in training).
- $y_{k,t} \in \mathbb{R}^{128}$ the separator state of band $k$ (the old gate's input).

Prior, a random walk: $s_{k,t} = s_{k,t-1} + w_{k,t}$, with $w_{k,t} \sim \mathcal{N}(0, q_k)$.
$q_k$ is fixed from the training data: the variance of $s_{k,t} - s_{k,t-1}$ over frames where the target is active at both $t-1$ and $t$. This mirrors how the gate's transition matrix was fitted from labels.

Observation: $o_{k,t} = s_{k,t} + v_{k,t}$, with $v_{k,t} \sim \mathcal{N}(0, r_{k,t})$ and $r_{k,t} = \exp(u_k^\top y_{k,t} + b_k)$.
The learned trust head has $u_k \in \mathbb{R}^{128}$ and $b_k \in \mathbb{R}$ per band, so $32 \times 129 = 4{,}128$ parameters.

Filter, per band and per frame (Kalman, 1960). Predict, then correct:

$$m^-_t = m_{t-1}, \qquad P^-_t = P_{t-1} + q_k$$

$$K_t = \frac{P^-_t}{P^-_t + r_{k,t}}$$

$$m_t = m^-_t + K_t\,(o_{k,t} - m^-_t), \qquad P_t = (1 - K_t)\,P^-_t$$

Start values: $m_1 = o_{k,1}$ and $P_1 = r_{k,1}$.

Output: $g_{k,t} = \min\big(1, \exp\big((m_t - o_{k,t})/2\big)\big)$, and $\hat{S}'_{f,t} = g_{k,t}\,\hat{S}_{f,t}$ for $f \in B_k$.
The phase is untouched. Because $g \le 1$, the tracker can only remove energy, which targets leaked words, never add it back.

Starts as e21: with $u_k = 0$ and $b_k = -10$, $r \approx 4.5 \times 10^{-5}$, so $K \approx 1$, $m_t \approx o_t$ and $g \approx 1$. The fine-tune starts from e21's exact output.

Training loss: the existing `bsrnn_loss` on $\hat{S}'$, plus $\lambda L_r$, where

$$L_r = \frac{1}{N} \sum_{\text{target-active } (k,t)} \tfrac{1}{2}\Big[\log r_{k,t} + \frac{(s_{k,t} - \mathrm{sg}(o_{k,t}))^2}{r_{k,t}}\Big]$$

This is the Gaussian negative log-likelihood (Nix & Weigend, 1994; Kendall & Gal, 2017), and $N$ is the number of target-active (band, frame) pairs. The stop-gradient $\mathrm{sg}$ means $L_r$ trains only the trust head, which learns $r$ = how wrong the network's band energy is in that frame. Set $\lambda = 0.1$ and register it before the run; it is not tuned.

Streaming: two floats ($m$, $P$) per band per stream, so 64 floats of state and $O(32)$ work per frame.

## Steps (one per turn: maths, then one small function, explain, stop)
0. Branch `m6-target-tracker` from main. Add the hypothesis below to `docs/decisions/decisions-pending.md` before any result exists.
1. **Fit $q_k$:** `scripts/fit_tracker_prior.py`. It reads clean targets and activity labels of the training split, forward-only (laptop is fine), and writes `experiments/results/<date>-tracker-prior/` (q per band, frame count, seed, commit). Add a run_times row.
2. **`src/models/tracker.py`:** `build_tracker`, `tracker_forward` (whole clip) and `tracker_step` (streaming). Tests in `tests/test_tracker.py`:
   - (a) at init, the output equals the input within 1e-3 relative;
   - (b) stepping frame by frame equals the whole-clip result;
   - (c) with $r \to \infty$, $m$ follows the prediction;
   - (d) $g \le 1$ always.
3. **Wire it in:** behind config key `model.tracker: true` in `src/models/bsrnn.py`, built last (like `context`) so the seed draws the same weights. Call it from `extract()` and `src/models/streaming.py`. `tests/test_streaming.py` must still pass.
4. **Forward-only diagnostics** on sir0_val `both` (103), on e21's output, with no weight updates:
   - **D1, oracle energy:** $g^* = \min(1, \sqrt{E_\text{target}/E_\text{est}})$ per band. This is the upper bound for any band-level gain.
   - **D2, oracle trust:** the Kalman filter with step 1's $q_k$. Set $r$ small ($10^{-4}$) on frames where the true labels say not both speakers are active. In overlap frames, use $r_\text{overlap} \in \{\bar{q}, 10\bar{q}, 100\bar{q}\}$, where $\bar{q}$ is the median $q_k$. Register these three values first and report all three. This tests the continuity idea itself, without training.
   - **Judging:** Gemini (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio in), with a paired bootstrap against e21 (10,000 draws, seed 42), as in `experiments/results/2026-10-18-paired-oracle-gate-judge/`.
   - **Stop rule:** if no D2 setting lowers LCF-WER with a 95 % interval below 0, do not train. Write it up as a negative result.
5. **Training** (only if D2 passes; needs a GPU, never the laptop):
   - **Tooling:** restore warm-start (`training.init_weights`, `scripts/train.py` before commit 04859ef) and the Kaggle bundle tooling (`scripts/make_kaggle_bundle.py`, `make_kaggle_notebook.py`, `preflight_kaggle.py` before db74c05).
   - **Config:** `experiments/configs/bsrnn_tracker_ft.yaml` = `git show 04859ef^:experiments/configs/bsrnn_interf_ft_control.yaml`, plus `model.tracker: true` and `loss.lambda_r: 0.1`. That means: from e21, lr 1.25e-4, 6 epochs, seed 42, no `w_schedule`. Leave `persistent_workers` as it is, so the data matches the control.
   - **Time:** e21 measured 2,886 s/epoch on a Kaggle T4 (run_times.md 2026-09-24), so 6 epochs would take about 4.8 h at that rate. That is a projection: measure the tracker's first epoch and log it.
6. **Evaluate** epoch 5 against the existing control, `models/model_sir0_interf-ft-control-e5-full.pt`, on sir0_val `both`:
   - Gemini, paired bootstrap, on LCF-WER, mean leak, invented per trial and target words missed.
   - Mechanism check: mean $K$ in overlap frames vs target-alone frames.
   - Timing with the existing real-time-factor script.
7. **Log:** `experiments/results/<date>-...` (config, commit, metrics, date), a `decisions-m2.md` entry, run_times rows and a row in `docs/ai-declaration-todo.md`.

## Pre-registered hypothesis
Tracking the target's band energy lowers leaked words in overlapped frames without raising invented words, so LCF-WER falls against the control fine-tune (paired 95 % interval below 0).
The mechanism must hold too: $K$ is lower in overlap frames than in target-alone frames. If LCF-WER falls but $K$ does not differ, the gain came from something else, and the write-up says so.

## Limits to state
- 32 band gains are coarse. Overlap inside a band is still the mask's job.
- $g \le 1$ can only remove energy, so watch target words missed (deletions).
- A random walk lags fast changes in the target. The trust head decides how far to lean on it.
- D2 holds the target's last known level through an overlap, which is crude for long overlaps.
- Results are optimised for Gemini, on two-speaker mixtures only. Validation chose $r_\text{overlap}$, so its score is optimistic.

## References (verify each before citing in the thesis)
- Kalman, R. E. (1960). A new approach to linear filtering and prediction problems. *J. Basic Engineering* 82(1).
- Paliwal, K. & Basu, A. (1987). A speech enhancement method based on Kalman filtering. *ICASSP*.
- Nix, D. A. & Weigend, A. S. (1994). Estimating the mean and variance of the target probability distribution. *ICNN*.
- Kendall, A. & Gal, Y. (2017). What uncertainties do we need in Bayesian deep learning for computer vision? *NeurIPS*.
- Related: Revach, G. et al. (2022). KalmanNet: neural network aided Kalman filtering. *IEEE TSP*. That paper learns the gain with an RNN; this plan learns only the observation noise.
