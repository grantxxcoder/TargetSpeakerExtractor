# Decision Log — M2 (baseline trained)

Covers the training objective and every training run: the loss terms and their
weights, the data splits made in response to training failures, the
augmentations, and the performance work that made the runs affordable. See
`milestones.md` M2.

Created 2026-08-31 when the log was split by milestone. Entries dated 2026-08-20
onward moved here from `decisions-m1.md`, which now holds only the architecture
and infrastructure decisions. References elsewhere in the repo were updated in
the same commit.

**Reading order.** The 08-20 objective entry is the design; everything from 08-25
onward is the objective failing, being diagnosed and being repaired. Read them in
date order or the repairs will not make sense.

**2026-09-15 — what may supervise training changed.** Gemini is no longer held
out; it may act as teacher, reward, data filter or selection criterion. It still
cannot be a loss term (not differentiable), and `sir0_privval`/`eval_private`
remain untouchable. Decision and its cost: `decisions-m4.md` 2026-09-15.

---

## 2026-08-20 — M2 training objective: three terms, six deviations from CARTSE

**Decision: the M2 loss is**

```
L = (1 - w) * mean_present[ L_pres + w_m * L_MR ]  +  w * mean_absent[ L_abs ]
```

```
                            ||s_proj||^2
L_pres  =  -10 log10  ---------------------------------
                      ||s_hat - s_proj||^2 + tau*||s_proj||^2

                     ||s_hat||^2 + tau*||x||^2
L_abs   =   10 log10 -------------------------
                              ||x||^2

                <s_hat, s>
s_proj  =  ---------------- * s
                 ||s||^2

            1
L_MR    =  --- SUM_i [ || |S_i|^p - |S_hat_i|^p ||_1  +  || S_i - S_hat_i ||_1 ]
            I

tau = 1e-3        p = 0.3    I = 4, windows [128, 256, 512, 1024]
w = 0.458         w_m = set by measurement (below)
```

The present/absent switch is `crop_absent` from `dataset_loader.py`, **never the
manifest condition label** — 5.8 % of `both`/`target_only` crops land in target
silence (2026-08-18 entry), and branching on the label sends ~1 crop in 17 down
the `L_pres` path with an all-zero target, which is a `NaN`.

### Provenance

| term | source |
| --- | --- |
| `L_pres` floored SI-SDR | CARTSE Track 1 (Li & Seki, 2026) eq (1). SI-SDR itself: Le Roux et al., ICASSP 2019; as a separation objective: Luo & Mesgarani, TASLP 2019 |
| `L_abs` push-to-silence | CARTSE eq (2). Target-absent/false-alarm framing: Delcroix et al., Interspeech 2022 |
| `L_MR` multi-resolution | Yu et al., Interspeech 2023 eq (3) (`p = 0.3`, windows 10-40 ms). Multi-resolution STFT loss: Yamamoto et al., ICASSP 2020. Compression exponent: Braun & Tashev, 2021 (verify venue string before citing) |

CARTSE applied eqs (1)-(2) to real pseudo-labelled conversational audio; we apply
them to constructed LibriSpeech mixtures. **Same formulae, different data — no
number produced under this loss is comparable to a published REAL-TSE result.**

### Deviation 1 — `L_pres` floors on `||s_proj||^2`, not `||s||^2`

**Found by running the sanity test, not by reading the paper.** CARTSE eq (1)
floors the denominator on `tau*||s||^2`, tied to the *target's* energy. The
numerator `||s_proj||^2` scales with the output gain `g`; the floor does not. So
eq (1) as written is **not scale-invariant and has no lower bound**: a
perfect-shape output scaled by `g` scores `-20 log10(g) - 30`.

Measured, perfect-shape output, sweeping `g`:

| `g` | floor `tau*||s||^2` (eq 1) | floor `tau*||s_proj||^2` (ours) |
| --- | --- | --- |
| 0.05 | -3.98 | **-30.00** |
| 0.2 | -16.02 | **-30.00** |
| 1.0 | -30.00 | **-30.00** |
| 5.0 | **-43.98** | **-30.00** |
| 100 | **-70.00** | **-30.00** |

So eq (1) pays **unlimited reward for amplifying the output**, and its 30 dB
ceiling exists only at unity gain. On an imperfect output the drift is smaller
but present (-11.85 at `g=0.2` rising to -13.95 as `g -> inf`, converging on
plain unfloored SI-SDR).

Flooring on `||s_proj||^2` makes numerator and floor scale together, so they
cancel: **flat -30 dB at every gain, and the range really is `[-30, inf)`.**

Why fix it at the source rather than leaving it to `L_MR`: `L_MR` compares
magnitudes directly and so does pin the output gain — but the `w_m = 0` ablation
arm is required, and in that arm nothing else bounds it.

Cost of the fix: on total collapse (`s_hat = 0`) numerator and denominator are
both exactly 0, so `0/0` is `NaN` where eq (1) would have given `+inf`. `NaN`
survives any clamp, so **eps is added to both sides**, making collapse read
0.0 dB — finite, and worse than the ~-6 dB of passing the mixture through, so
not an attractor.

Not a criticism of CARTSE: their objective carries a mel-filterbank L1 and a
DNSMOS term that both pin the output gain, so the defect is masked in their
system. It is exposed in ours only because a `w_m = 0` arm exists.

### Deviation 2 — `L_abs` is normalised by `||x||^2`

CARTSE eq (2) is `eta * 10 log10(||s_hat||^2 + tau*||x||^2)`, which is **not
scale-invariant**: scale `x` and `s_hat` by `g` and the value shifts by
`20 log10 g`. Two loudness-matched absent trials that are both perfectly silent
therefore receive different losses and different gradients. `L_pres` is already
scale-invariant, so the pair was mismatched.

Dividing by `||x||^2` fixes it and buys a free anchor:

| `L_abs` | meaning |
| --- | --- |
| `0` | emitted the mixture unchanged — **did nothing** |
| `-10` | suppressed 10 dB |
| `-30` | floor: at or below 30 dB down, i.e. silent |
| `> 0` | **amplifying. A bug, not a bad score** — flag it in the run log |

`0` = do-nothing on every trial regardless of loudness, so the term reads without
knowing the trial's level. CARTSE's form has no such anchor.

### Deviations 3-5 — how the halves combine

**Deviation 3: masked means per half, not one batch-wide mean.** Averaging each
term over its own subset. Under a single batch-wide mean the absent half's share
of the gradient is whatever the batch happened to draw — 15 % at 1 absent crop in
12, 67 % at 6 — so the present/absent balance fluctuates step to step on sampling
luck.

**Deviation 4: `eta` is removed and folded into `w`.** Under the masked-mean form
`w` and `eta` appear only as the product `w * eta`, so `(0.297, 2.0)` and
`(0.594, 1.0)` give identical gradients. Two dials, one degree of freedom.
CARTSE needed `eta` because they used a batch-wide mean and it was their only
weight; we do not.

**Deviation 5: `w = 0.458`, not `0.297`.** The 2026-08-18 entry requires the
weighting use the measured crop-level absent rate 0.297. It does — inside the
calculation, not as the weight itself:

```
present coefficient = 1 - p       = 0.703
absent  coefficient = p * eta     = 0.297 * 2.0 = 0.594
                                     total mass = 1.297
w = 0.594 / 1.297 = 0.458
```

`w = 0.458` reproduces CARTSE eqs (1)+(2) with `eta = 2.0` at *our* measured
0.297. `eta = 2.0` was their deliberate choice to weight silence **above** its
data frequency; `w = 0.297` would silently discard that and is logged as the
data-frequency-neutral ablation arm instead.

Note the coefficients sum to 1, scaling the whole loss by `1/1.297` relative to a
batch-wide mean. Interacts with learning rate only — but CARTSE's `1e-4` is not
directly transferable because of it.

### Deviation 6 — `L_MR` window set straddles the model's own framing

**Windows `[128, 256, 512, 1024]` samples (8/16/32/64 ms), hop = window/4.**

CARTSE used `[512, 1024, 2048]`, all at or above our `n_fft`. Yu et al. used
10-40 ms. Ours brackets 512 deliberately: an STFT with a 32 ms window averages
everything inside 32 ms into one number per band, and the model builds its output
by masking 32 ms frames and overlap-adding, so its characteristic artefacts —
frame-boundary discontinuities, per-frame gain jumps, warble at the frame rate —
have exactly the structure a 32 ms analysis integrates away. The 8 and 16 ms
windows resolve them; 64 ms catches harmonic structure the short ones blur.

Powers of two throughout, unlike Yu et al.'s `[160, 320, 480, 640]`, so no
zero-padded windows.

**`L_MR` is applied to present crops only.** With an all-zero reference both L1
terms reduce to "minimise output energy", which duplicates `L_abs`'s job in
non-dB, unnormalised units and makes the effective silence weight unknowable.

**Reduction is `mean`, pinned in config and in the code comment.** `||.||_1` in
Yu et al. eq (3) literally means a sum over ~257 x 500 = ~128,000 coefficients,
while `auraloss` and the ParallelWaveGAN reference use a mean — a factor of
~1e5. Under sum-reduction with `w_m = 1.0`, `L_MR ~ 1e4` against
`L_pres ~ 1e1` and **Term 1 becomes numerically invisible with no error
message.** Published `w_m` values do not transfer unless the reduction matches.

`p = 0.3` is an empirical convention, not derived. It compresses ~60 dB of
in-frame dynamic range to ~8:1 so quiet high-frequency bins (fricatives,
sibilants, stop bursts) can compete for gradient.

**Measured, on a real chunk** (`train-42-010130`, 4 s at the highest-energy
offset; top of the spectrum ablated, low band and phase kept exact):

| output | energy kept | `L_pres` | `L_MR` |
| --- | --- | --- | --- |
| perfect (`s_hat = s`) | 100.00 % | **-30.00** | **0.0000** |
| >6 kHz deleted | 99.50 % | -22.28 | 0.0560 |
| >4 kHz deleted | 98.07 % | -16.86 | 0.1263 |
| >2 kHz deleted | 97.67 % | -16.06 | 0.1896 |
| >1 kHz deleted | 96.69 % | **-14.61** | **0.2314** |
| unprocessed mixture | - | -5.60 | 0.2535 |

**This is the justification for the term, and it is reportable as a result.**
A signal with everything above 1 kHz destroyed - a muffled mumble, every
consonant gone - keeps 96.7 % of the energy, so `L_pres` scores it **-14.61**,
i.e. 9 dB *better* than doing nothing. `L_MR` scores the same signal at 0.2314
against the do-nothing mixture's 0.2535, i.e. 91 % of the way to "you achieved
nothing" - the correct judgement, and one `L_pres` cannot reach at any weight.

Supersedes the earlier order-of-magnitude estimate in this entry (~1 % of energy
above 4 kHz, "SI-SDR still reads ~20 dB"). Measured: 1.93 % and -16.86 dB. The
argument holds; the numbers are now measured rather than projected.

Consequence: the `w_m = 0` ablation arm is not a formality. It is the arm in
which this blindness is live.

### `w_m` is set by measurement, not by sweep

**Measure both terms at `s_hat = x` (the mixture passed through) before
training.** Model-free, seed-independent, and roughly what the model does after a
few hundred steps. An untrained model is *not* a valid anchor — its output depends
on the random init.

Target `w_m * L_MR ~ 0.3 * |L_pres|` at that anchor. First measurement, one
real chunk (`train-42-010130`): `L_pres = -5.605`, `L_MR = 0.2535`, so
`w_m ~ 6.6` -- **not** CARTSE's 1.0, which would have put `L_MR` at ~4.5 % of the
present branch. Provisional until run over a few hundred crops and medianed;
`L_MR` varies with a trial's spectral content far more than `L_pres` does.

**The ratio drifts monotonically during training.** `|L_pres|` grows as the model
improves (-5.6 -> -30) while `L_MR` shrinks (0.25 -> 0), so `L_MR`'s share of the
loss value falls throughout. An early-strong / late-weak spectral term is
defensible, but it must be a stated choice rather than an accident, and it is a
further reason to log both terms every step. `L_pres` defines the task;
`L_MR` prices what it cannot see. Ablate `w_m` at `{0, 0.3x, 1.0x}` — the `0` arm
is required to show the term earns its place, and is direct thesis material if
`L_MR` moves LCF without moving SI-SDR.

Caveat to state when reporting: matching loss *values* is a proxy for matching
*gradients*. Record gradient norms for each term once at the anchor
(`torch.autograd.grad`). No paper in `review_synthesis.md` reports this.

### Not in the M2 loss

| term | source | status |
| --- | --- | --- |
| scenario-aware frame-level split | CARTSE eq (3) | **deferred to M4.** Needs frame-level `y`; see below |
| speaker consistency | CARTSE eq (4) | deferred. Using it forfeits SpkSim as a held-out number |
| mel-filterbank L1 | CARTSE eq (5) | deferred to M4 |
| frozen-encoder feature matching | CARTSE eq (7), PS4 (Ning et al., 2026) | deferred to M4 — the primary proxy |
| **DNSMOS maximisation** | CARTSE eq (6) | **rejected, permanently** |

**DNSMOS is rejected on two independent grounds.** The organisers found
DNSMOS-OVRL over-optimised to the point of ~zero human-MOS correlation on Track 1
(LCC +0.003) and swapped the official metric post hoc; `metric-definitions.md` §4
names this as the cautionary tale this project designs against, and CARTSE
explicitly trains on it. Second, it optimises perceptual quality, which is
explicitly not the objective (`CLAUDE.md`). Recorded as a rejection with the
citation, not an omission.

**Why M2 is conventional at all**, given the thesis argues conventional
objectives are the wrong target: the divergence between conventional metrics and
LCF *is* the finding, and it needs a competently-trained conventional arm to
diverge from; `research-plan.md` §5 requires the proxy models share a base
checkpoint with the baseline or the ablation is unattributable; SI-SDR carries
calibration a proxy loss does not, so a bad number means a bad model rather than
an ambiguity; and every proxy paper in the set (CARTSE, PS4, Ma et al.)
fine-tunes from a signal-loss checkpoint — none trains from random init.

### Consequences to carry

1. **`L_MR` reduction, and the complex-term convention, must be pinned in the
   config.** L1 on a complex tensor is ambiguous: `L1(real) + L1(imag)` or
   `sum |S - S_hat|` (modulus). They are different numbers. Yu et al. eq (3)
   leaves the complex term **uncompressed** — some of the literature compresses
   both. Follow the paper; state the choice in the code comment.
2. **Frame-level `y` for M4 is not free.** `target.wav` is exactly zero only
   where no utterance was *placed*. Within-utterance pauses (LibriSpeech carries
   ~0.331 s leading silence; 86.0 % of a file is speech, 2026-08-15) are room
   noise and reverb tail after RIR convolution, not zeros. Frame-accurate `y`
   needs `vad_segments.csv` mapped through `target_onsets_s`, and a rule for the
   reverb tail: up to `t60` (<= 0.6 s) of the target's own energy follows the
   last word, is present in the reference, and is rewarded by `L_pres` — a strict
   VAD label would mark those frames silent and have `L_TS` punish correct
   behaviour. Either extend active regions by `t60` or threshold on stem energy.
3. **Supervision is against the *reverberant* target.** `render.py` returns "the
   target through its own room, alone", so this loss asks the model to preserve
   the room, not dereverberate. Our SI-SDR is therefore not comparable to
   dry-target-supervised numbers.
4. Every weight (`tau`, `w`, `w_m`, `p`, the window list, the reduction) lives in
   `experiments/configs/`. None in the loss module.
5. Compute the loss in float32 even under AMP — the squared-norm sums and the
   logs are unreliable in fp16.

### Tests to keep

Implemented as `tests/test_losses.py` (30 tests, synthetic seeded signals
only, no corpora read). The gain-invariance test is verified to FAIL on
CARTSE eq (1) as published, which is what makes it worth keeping.

| assertion | expected |
| --- | --- |
| `L_pres(s, s)` | exactly `-30.0` at `tau = 1e-3`. Validates the whole implementation in one line |
| `L_pres` gain invariance | `L_pres(s, g*s_hat) == L_pres(s, s_hat)` for `g` in [0.05, 100]. **This is the test that caught Deviation 1** — it fails on CARTSE eq (1) as written |
| `L_pres(s, 0)` | `0.0`, finite. Total collapse must be neither `NaN` nor `inf` |
| `L_pres(s, x)` | the mixture's own SI-SDR, ~ `sir_db`. The floor anchor |
| `L_MR(s, s)` | `0.0` |
| `L_abs(0, x)` | `-30.0`, on any `x` |
| `L_abs(x, x)` | `10 log10(1 + tau)` = **`0.00434`**, not `0.0`. On any `x` |
| masked means | do not divide by zero when a batch holds 0 present or 0 absent crops |
| silent target | `L_pres(0, x)` is `NaN` **by design**. Assert it, so the masking requirement is pinned by a test rather than a comment |

**Log every term separately from step 1, and keep absent trials in `val`**
(0.35, `decisions-m0.md` 2026-08-11). A total loss that falls while `L_abs` sits
flat near `0` is a model passing the interferer straight through whenever the
target is silent — invisible in the total, invisible in SI-SDR, and visible at
eval only as a blown-up ICR.

---

## 2026-08-25 — The first full run collapsed to a mute; `tau` split, `mid` split added

**Decision: split `tau` into `tau_pres` (0.001) and `tau_abs` (0.01), and add a
`mid` split — 2,000 trials subset from the already-rendered `train` — as the next
training target. `w_m` is NOT changed.**

The 100-epoch smoke run (`experiments/results/2026-08-24-train-smoke-resume`)
drove total loss to -15.44 while `L_MR` got steadily *worse*, 0.279 -> 0.318.
Diagnosed on the saved checkpoint over the 20 val crops.

### What the model actually learned

| measurement | value |
| --- | --- |
| output energy vs mixture, present crops | -34.11 dB |
| output energy vs mixture, absent crops | -35.46 dB |
| present-minus-absent discrimination | **+1.34 dB** |
| gain that minimises `L_MR` | **30x** (-29.5 dB too quiet) |
| `L_MR` at that gain | 0.224, vs 0.319 as trained |
| output change when the enrolment is swapped | **-17.15 dB** |
| `L_pres` cost of a swapped enrolment | **+0.62 dB** |

A uniform mute that nearly ignores the enrolment. `L_pres = -7.18` looks healthy
only because SI-SDR is scale-invariant *and* satisfiable by generic
speech-shaped output on a two-speaker mixture — it is flattering the model twice.

### Why `L_MR` was the term that paid

`dL/d(log g)` at the operating point: `L_pres` **0.00000**, `L_MR` -0.031,
`L_abs` +1.999. `L_MR` is the *only* term that can see output gain, so as the
model learned silence `L_MR` recorded the bill. Over epochs 3-99,
`corr(val_L_MR, val_L_abs) = -0.967` — one variable seen twice.

Loss units delivered over the run: `L_abs` **-10.55**, `L_pres` -1.64,
`L_MR` **+0.20**. A *perfect* `L_MR` is worth 1.66 units, ~3 % of the range.

### Neither hypothesis on the table was right

- **Not `w_m` miscalibration.** 9.62 is correct for what it was calibrated to:
  a `L_MR`-vs-`L_pres` ratio at the do-nothing anchor. Two things make it miss —
  `grad_norms.csv` never measured the absent branch, and the anchor sits at
  `s_hat = x` where the gain is already right, so the attenuated region where the
  trade-off bites was never sampled. But no `w_m` fixes it: 243 would be needed
  at g=10, and at g=30 the required value goes *negative*.
- **Not capacity.** `train_L_MR` 0.278 vs `val_L_MR` 0.319 — a 0.04 gap on 48
  crops against 7.19 M parameters. A capacity-bound model memorises; this one
  does not even try. And the failure is one global scalar.

### Root cause

No enrolment conditioning -> cannot tell present from absent -> one shared gain
serves both branches -> correct level costs **+24 dB** on absent crops
(x `w` = ~+11 loss units) -> the mute is genuinely optimal.

**The objective is not broken.** With present/absent gains free it already
prefers the right answer: -16.47 at (g_pres=30, g_abs=0) vs -15.44 achieved. The
model cannot use it.

### `tau_abs` is a knob, not the fix — logged so it is not retried

Raising `tau_abs` was the first proposal and it is **measured powerless**: the
argmin along the shared-gain diagonal is g\*=0.3 for every `tau_abs` from 0.001
to 0.1. `L_abs` at correct gain is -5.90 dB at *all* of them. `tau` floors the
quiet end; what pins the model quiet is the penalty at the loud end.

Giving `L_pres` gain authority (plain SNR instead of SI-SDR) does work as
intended — spread over the gain range goes from 0.0000 dB to 6.92 dB — but the
diagonal optimum still lands on the mute. Not adopted; it is a real deviation
from CARTSE eq (1) and it does not solve this. Revisit only if conditioning
works and the gain is still wrong.

The split is kept anyway because a single `tau` for two differently-scaled halves
was conflating two things, and `-20 dB` is already inaudible suppression.
Consequence: the plot's total floor is no longer `10log10(tau)` but
`(1-w)*10log10(tau_pres) + w*10log10(tau_abs)` = **-25.42**, now computed by
`total_loss_floor()` in `scripts/train.py`. The hardcoded do-nothing anchor line
moves -2.240 -> -2.222; left as -2.24, noted in the code.

### The `mid` split

Speaker diversity is what conditioning needs, and smoke has **20** speakers.
`mid_train` is 2,000 trials subset from `train`'s 19,938 — **940 target
speakers**, condition mix held to within 0.03 % by proportional stratification
(the absent rate is what `w` was calibrated against, so it must not drift).
`mid_val` is `val` unchanged, 200 trials over 40 unseen speakers.

No new audio: `TrialDataset`'s `split` is only the directory under
`data/rendered/`, so `SPLIT_MANIFESTS` now carries `(manifest, audio_dir)`
separately and `mid` reads the already-rendered `train`/`val` trials. Widening
smoke's 20 speakers instead would have cost a re-render and still not tested the
hypothesis.

**Not run yet.** ~2.7 h/epoch on the laptop (projection from the measured
4.92 s/trial-epoch, batch 3, CPU) — i.e. ~3.4 days for 30 epochs. Intended for
Kaggle at batch 12.

### What to watch, instead of `L_MR`

`L_pres` is scale-invariant so it **cannot** show a mute, and `L_MR` shows it
only as a lagging side-effect. The leading indicators are enrolment sensitivity
(dB change when the enrolment is swapped) and the present-minus-absent output
energy gap. Neither is in `history.csv` yet.

### Caveats

15 present + 5 absent val crops, one checkpoint, one seed. The gain sweep is a
1-D slice holding the learned mask shape fixed.

---

## 2026-08-25 — Turning off the silence reward did not help. The model still ignores the voice sample

**Decision: stop changing the loss. The next thing to investigate is how the
voice sample is fed into the model (`src/models/conditioning.py`), not the
scoring.** Ran 10 epochs on `mid` with a warm-up schedule; the result was a
clean negative and it rules out the loss as the cause.

Result: `experiments/results/2026-08-25-train-mid-warmup/`. 5.4 h on a Kaggle
T4, batch 6, 1,950 s/epoch.

### The job, and what the model is actually doing

Every clip has two people talking over each other plus background noise. We also
hand the model a short sample of the voice we want. It should output only that
person.

It is ignoring the sample. The test: run the same clip twice, once with the
right person's sample and once with a stranger's. If the model were listening,
the two outputs would sound like different people. The output changes by
**2.6 %**. Same answer either way — so it is doing something generic to the
audio rather than picking out a person.

### Why it can score well without doing the job

The score rewards two separate things: sound like the target while she is
talking, and stay silent while she is not. That leaves two shortcuts, and
neither one needs the voice sample.

  1. **Say nothing at all.** Lose points on the first half, max out the second.
     This is what the 2026-08-24 run did (entry above).
  2. **Hand the recording back nearly unchanged.** The target is usually the
     louder of the two voices, so the original mixture already resembles her.
     Decent score for doing almost nothing.

### What was tried

Switch off the silence reward for the first 4 epochs (`w = 0`), then ramp it in
over 3. With shortcut 1 unavailable the model should be forced to actually learn
to pick the person out. Config: `loss.w_schedule`, implemented in
`w_at_epoch()`.

Thresholds were fixed **before** the run so the result could not be argued into
whatever we hoped for. Measured on how much the output moves when the sample is
swapped: better than -6 dB = worked; -6 to -10 = partial; worse than -10 = the
schedule is not the answer.

### What happened

**At the end of the warm-up (epoch 3): -15.86 dB.** It started at -15.98. Flat
across all four epochs. It never began listening to the sample.

It took shortcut 2 instead. Compared with just handing the recording back
unchanged (measured 2026-08-20: `L_pres` -5.909, `L_MR` 0.1842):

| | end of warm-up (epoch 3) | vs handing it back unchanged |
| --- | --- | --- |
| how close to the target | -6.657 | only **0.75 dB better** |
| second quality measure | 0.1900 | **worse** (+0.0058) |

Four epochs bought three quarters of a dB over doing nothing, and the second
measure never beat doing nothing at all.

Then the silence reward came back and so did shortcut 1: output on
target-silent clips fell to -18.3 dB below the mixture, and the second quality
measure went back up, 0.190 -> 0.236.

### The trap: the headline number improved the whole way

Total loss fell from -3.40 to **-10.74** across the run, and the best score was
the very last epoch. That reads as a successful run. It is not — the score
improved *because the model got quieter*. The thing making the number look good
is the thing making the model useless.

This is the second time that has happened, and it is why the two extra columns
now exist (`val_enrol_sens_db`, `val_pres_abs_gap_db`). They are the only
numbers in the log that told the truth. **Never report `val_total` from this
objective without checking them.**

### One thing that did improve

Against the 20-speaker run, this one had 940 and ends slightly less deaf to the
voice sample: **5.5 % vs 2.5 %**, and the loud/quiet gap reached 3.09 dB against
1.34. Real, small, pointing the right way. More speakers helps a little; it is
not the main problem.

### What this rules out

Two runs, two different schedules for the silence reward, same blindness to the
voice sample. Combined with the 2026-08-24 measurements — no value of `w_m`
works (it would need ~243 and flips sign at the correct volume), and `tau_abs`
does nothing (the score at correct volume is -5.90 dB at 0.001, 0.01 and 0.1
alike) — **the loss is not what is stopping the model from using the sample.**

The remaining suspect is the path the sample takes into the model. If swapping
it for a stranger's changes the output by a few percent, that connection may be
too weak to influence what comes out, and no amount of rebalancing the score
will fix that. See the 2026-08-19 entry on Spectral Similarity conditioning.

### Kept, even though the warm-up did not work

`loss.w_schedule` stays in the config, defaulting to a schedule but returning
the constant `w` when the block is deleted. It is cheap, it is tested
(`tests/test_w_schedule.py`), and it is the arm this entry reports — removing it
would make the result unreproducible.

One implementation note worth keeping: every `total` is computed at the **final**
`w`, never the epoch's own `w`. Otherwise the number means something different
each epoch, and two things that read it break — the learning-rate scheduler sees
the ramp as improvement and never steps down, and best-checkpoint selection
picks whichever epoch had the largest `w` rather than the best model.

### Caveats

200 validation clips, one seed, one run. `meta.yaml` and the final checkpoint
were not downloaded before the Kaggle session ended, so this result carries no
config hash or commit — see the `NOTE.md` beside it. The checkpoint on disk is
epoch 3, the end of the warm-up, which happens to be the state the decision
turned on.

---

## 2026-08-25 — The model ignores the voice sample because the data lets it. New `sir0` split

**Decision: stop changing the model and the loss. Build `sir0` -- `mid` with the
target/interferer loudness ratio centred on zero instead of 90 % target-louder --
and retrain. `mid` is kept as the control arm.**

A day of measurement, mostly ruling things out. Everything below is measured on
`mid_train` / `mid_val`, on CPU, with no training.

### The job, and what the model actually does

Each clip has two people talking over each other plus background noise. We hand
the model a 5 s sample of the voice we want. It should output only that person.

It ignores the sample. Run the same clip twice, once with the right person's
sample and once with a stranger's: the output changes by **2.6 %**. Same answer
either way, so it is doing something generic to the audio rather than picking out
a person.

### Why: the data answers the question without the sample

**90 % of two-speaker trials have the target LOUDER than the interferer**, median
+6 dB, because `regimes.base` narrows `sir_db` to [0, 12]. So "keep the loud
voice" is right ~90 % of the time -- and the model sees the mixture, so it gets
loudness for free.

Measured as a hit rate on "is the target the dominant voice at this moment?":

| who is louder | n | speaker cue | loudness (free) |
| --- | --- | --- | --- |
| interferer louder (sir < 0) | 66 | 59.2 % | 54.8 % |
| target louder 0-6 dB | 302 | 61.4 % | 66.9 % |
| target louder 6+ dB | 379 | 58.5 % | **81.5 %** |

The cue is flat across all three -- it tracks *who*, not *how loud*, which is the
right behaviour. Loudness climbs to 81.5 % where the target dominates, and 379 of
747 trials live in that bin. Faced with an 81 %-accurate free strategy and a
58 %-accurate one that must be learned, the model picks the free one. It is
behaving correctly; the task barely requires the enrollment.

`difficulty-dial.md` (2026-08-13) ranked `sir_db` #1 of 14 dials and stated the
mechanism exactly: "At -5 dB the interferer is louder than the target, so nothing
but the enrollment can identify which voice to keep." It framed the narrowing as
**difficulty** relief. It is also **relevance** relief. Those are different: a
task can be easy and still require the enrollment. That distinction is the thing
this entry adds.

### One real bug in the cue, found and fixed

`TFMap` compares every mixture frame against every enrollment frame and softmaxes
the scores. Softmax compares logits by DIFFERENCE, not ratio:
`w_i / w_j = exp(s_i - s_j)`. `F.normalize` (needed -- we want spectral shape,
not loudness) bounds every cosine to [-1, 1], so the largest achievable
difference was ~1 and the best-matching frame could never outweigh the worst by
more than `e^1 = 2.7x`. Spread over 628 enrollment frames that is nothing.

Measured: **619.6 of 628 frames effectively used**; the top frame held 0.22 % of
the weight against 0.16 % for a flat average. The softmax was averaging, not
selecting, so the cue was the enrollment's long-term mean spectrum -- varying
only 4.7 % over time.

Zhang et al. eq (2) is written on UN-normalised products, measured here at
0..932, a range that selects sharply on its own. Normalising removed the range;
`model.tfmap_scale` restores it. Default `sqrt(F)` ~ 16 at F=257. Worth **+3.5
to +5.3 points** on the hit-rate table above. Pinned by
`tests/test_tfmap_scale.py`, which tests the mechanism (`w_i/w_j == exp(scale *
(s_i - s_j))`) and not just the symptom.

### Ruled out, with numbers

- **The loss, and the silence reward.** Two runs, two schedules. The `w = 0`
  warm-up moved enrollment sensitivity by 0.1 dB over four epochs; the model
  switched from the mute to passthrough. See the entry above.
- **`w_m` and `tau_abs`.** `w_m` would need ~243 and flips sign at the correct
  volume; `tau_abs` is identical at 0.001, 0.01 and 0.1.
- **Enrollment length.** 5 s -> 10 s buys **0.8** points. 5 s -> 20 s buys the
  same 0.8. Four times the audio, no further gain. **Do not re-render for a
  longer enrollment.**
- **Enrollment EQ.** 0.3 points (1 % of the gap), and 49.2 % of trials carry it.
- **Reverb mismatch.** Giving the sample the trial's room at a mirrored source
  position: **0.0** points.
- **The cue's ability to recognise voices.** Given two clean clips it separates
  "same person, different words" from "different person" in **95.3 %** of 319
  pairs (97.4 % at scale 8). Spectral Similarity is a strong speaker
  discriminator; the 2026-08-19 choice of eq (2) over eq (3) is NOT the problem.

### A4 is doing real work -- keep it

The only enrollment arm that helped was giving the sample the target's **exact**
source position: +4.9 points, 14 % of the gap. That is positional fingerprinting,
which is precisely what A4 (2026-08-12, "the enrollment carries NO room") exists
to prevent. Worth recording that A4's stated reason -- matching on room instead
of voice -- is subtly wrong for a two-speaker trial, since both talkers share the
room and room-matching cannot separate them. The real cheat is POSITION, and A4
blocks it. Right decision, slightly wrong justification.

### The finding that keeps the problem open

The cue scores **95 % on clean clips and 56-61 % on mixtures**. An overlapped
frame is the SUM of two people, so it resembles neither alone; judging whether
the target dominates a frame almost requires having separated it first. So even
with the loudness shortcut removed the per-frame signal is weak. It may still be
enough -- the network has 7.19 M parameters and six LSTM layers, and the cue only
has to say WHICH voice to favour while the network separates -- but that is
reasoning, not evidence, and `sir0` is what tests it.

### `sir0`

`sir_db: [-10.0, 10.0]`, symmetric, so the shortcut becomes a coin flip. Width
kept wide rather than [-6, 6] deliberately: it leaves easy trials (target +10 dB)
to bootstrap on instead of making every trial equally hard. Realism cost is real
and acknowledged -- difficulty-dial.md calls -10 dB "plausible but uncommon".
The protocol gates only `overlap_ratio` behind supervisor agreement, not
`sir_db`, so this is a logged decision rather than an escalation.

Both regimes resolve to [-10, 10]: the split-level value is what `hard`
inherits, and the split's own `regimes` block omits `base.sir_db` so `base`
cannot re-narrow it. Every other parameter matches `train`, so the split differs
in exactly one axis.

`speakers_from: train` (new, `scripts/build_manifest.py`) borrows train's 1,172
speakers so `splits.yaml` -- generated and pinned before any data existed --
needs no hand edit. Speaker-disjointness is inherited from the borrowed split.

**`sir0_val` carries the same symmetric range as `sir0_train`.** Training on one
loudness distribution and scoring on another would measure neither. This departs
from the B4 note in `generator.yaml` that eval composition matches train, which
is why it is logged here.

### Caveats

One seed, one run per arm. The 66-trial interferer-louder cell is the population
the whole argument rests on and its cue-vs-loudness margin (+4.4) came back at
`mean/se = +1.1`, i.e. **not distinguishable from noise** -- so "the cue beats
loudness where loudness fails" is NOT established. What is established is the
shortcut's size (81.5 % vs 58 %) and its prevalence (90 %).

An attempt to confirm the model literally follows loudness was **inconclusive**:
the only checkpoint on disk is epoch 1 of the warm-up run (`w = 0`, 333 steps),
where passthrough behaviour is what the schedule was designed to produce. It
tracked the mixture at 0.937 against the target at 0.761, but at that epoch that
shows nothing. The epoch-9 checkpoint was not downloaded.

---

## 2026-08-26 — `both_directions`: every mixture is trained twice, once per speaker

**Decision: `both_directions: true`. Each rendered trial becomes two training
examples — the same mixture audio, asked once for the target and once for the
other speaker.**

**RECONSTRUCTED 2026-08-31.** This decision was implemented on 2026-08-26 and is
cited from five places in the code, but **no log entry was ever written** — the
citations pointed at a date that did not exist in any decision file. The entry
below is assembled *only* from what those code comments and
`experiments/results/RESULTS.md` already state. No reasoning has been added that
was not already recorded somewhere; where the original rationale is not
recoverable, this entry says so rather than inventing it.

### What it does

One trial, two examples. The mixture is unchanged; only the enrolment and the
reference target swap. `interferer_only` trials are the sharpest case: identical
audio, and the correct answer is **silence** one way and **a voice** the other.

### Why — the measurement that forced it

**This is the only thing in the data that makes reading the enrolment
compulsory.** With one direction per trial, an enrolment-ignoring model fits
every example, and measured, it did: **output moved only 8 % on an enrolment
swap** (`scripts/build_manifest.py`, `experiments/configs/bsrnn_baseline.yaml`).
Neither the loss schedule nor removing the loudness shortcut changed that — both
were tried first and both failed, which is what left a data-side fix as the only
option. In plain terms: if the same mixture always has the same right answer, the
model can learn the answer from the mixture alone and never look at the voice
sample. Asking for both speakers makes that impossible, because an
enrolment-blind model would have to give one answer to two different questions.

### Consequences that must be carried

**`batch_size` now counts TRIALS, not examples.** A step sees 2x the configured
batch. Any per-step memory or throughput figure quoted across the 08-26 boundary
is comparing different amounts of work.

**Epoch time doubled: 1,869 s -> 3,875 s** between the 08-26 and 08-27 runs
(RESULTS.md). Attributed there to `s0-v2` rendering the interferer stems, so
`both_directions` doubles the examples per epoch. This is a real cost of the
decision, not a regression.

**It requires re-rendered data.** `interferer.wav` and
`interferer_enrollment.wav` must exist, so **a split rendered before 2026-08-26
cannot use it.** That is why the 08-26 run is marked "pre-`both_directions`
rendering" in RESULTS.md and is superseded rather than compared.

**Config-driven, and absent-key-safe.** A missing `both_directions` key means the
old single-direction behaviour, so older configs and checkpoints are unaffected
and the arm is recorded with the run (`scripts/train.py`).

**`__getitem__` returns a LIST of directions even when `both_directions` is
False**, which is why call sites index `[0]`
(`scripts/pass_a_test_case_through.py`).

**Phantom enrolments.** On `target_only` / `noise_only` nobody interferes, so the
reverse direction enrols a speaker who is genuinely absent and the answer is
silence — "the purest target-absent example there is". Skipping those trials
instead would have dropped the absent rate from ~28 % to ~16 % and invalidated
the `w` the loss was calibrated on (`scripts/build_manifest.py`).

### What is NOT recoverable

Whether alternatives to doubling the data were weighed on 08-26, and against what
criteria, is not recorded anywhere. **Do not present this entry as the
contemporaneous reasoning** — it is a reconstruction, and the 8 % measurement is
the only evidence in it that is independently logged.
---

## 2026-08-27 — sir0 run: conditioning works, the mute does not go away. Adding `L_gain`

`sir0`, 1,989 train / 200 val, 8 epochs, Tesla T4, batch 3, 8.6 h, 3,875 s/epoch.
Seed 42, config md5 `2da2d7a9...`, bundle commit `67de944...-dirty`. `kaggle_out/`.

**The model learned who to listen for and still will not speak up.** It attenuates
rather than separates.

**Good, and real.** Enrolment sensitivity -14.82 -> -8.25 dB: a stranger's
enrolment now moves the output **39 %, up from 18 %**. That is what `a6baf77`'s
unbounded conditioning scaling was for. Only interpretable *because* of the
2026-08-25 `sir0` change -- on `mid`, 90 % of trials had the target louder, so a
model could look conditioned while tracking the loud voice.

**Bad.** 95.0 % of the total's improvement is the absent half (smoke was 93 %),
decomposed at the reporting w of 0.458:

| branch | ep 0 | ep 7 | change | share |
| --- | --- | --- | --- | --- |
| present `(1-w)(L_pres + w_m·L_MR)` | +0.437 | +0.069 | -0.369 | 5.0 % |
| absent `w·L_abs` | -1.582 | -8.592 | -7.010 | **95.0 %** |

`L_MR` ended **5.5 % worse than it started** (0.2407 -> 0.2540). It improved to
0.2231 across epochs 0-3 -- exactly the warm-up epochs where `w = 0` -- then
reversed the epoch the ramp began. Confound named: 0-3 is also early training.
But it is a reversal, not a slowdown, and it lands on the ramp.

Derived, not logged: `L_abs` -18.76 puts absent crops 24.8 dB below the mixture;
with the logged 2.45 dB gap, present crops sit at **-22.4 dB** (smoke: -24.9).

`L_abs` is at -18.76 against its -20 floor -- **94 % spent**. The cheap direction
is nearly exhausted.

**Correction to the first reading:** `L_pres` is scale-invariant SI-SDR, so -2.32
is **+2.32 dB** and the term improved 0.81 dB. Smoke went *backwards*. "Not
separating at all" was too strong.

### Decision: add `L_gain`, deadzone level match, default OFF

    L = (1-w) · mean_present[L_pres + w_m·L_MR + w_g·L_gain] + w · mean_absent[L_abs]
    L_gain = max(0, |20·log10(RMS_out / RMS_target)| - delta_db),  delta_db = 3.0

**Objective, not architecture or data.** The architecture just learned
conditioning; the data got harder in the right way and it learned anyway. The
rest is arithmetic: `L_pres` is scale-invariant by design (Deviation 1) and
cannot see a mute, `L_abs`'s optimum is zero output at weight 0.458, and the only
push-back is `L_MR` at effective weight 5.21. The model optimised the objective
correctly; the objective was wrong.

**Does not undo Deviation 1.** That bug was *unbounded one-directional* reward --
a correct output scaled by g scored -20log10(g) - 30, so amplifying paid forever.
`L_gain` is symmetric in log-level and minimised AT the correct level. Scale
variance was never the hazard; unbounded monotone reward was. `L_pres` stays
scale-invariant and measures shape alone -- two terms, two jobs.

**Deadzone ±3 dB** (amplitude 0.71x-1.41x): no gradient on sub-dB errors, and it
puts `.abs()`'s kink inside the zeroed region. **dB, not percent** -- 10 %
amplitude is 0.83 dB, *stricter* than a listener can resolve.

**Dataset-mean anchor rejected.** Trial levels vary by construction (BS.1770,
`sir_db` in [-10, 10], varying SNR and `target_activity_ratio`). A global mean
rewards making quiet targets louder and loud ones quieter -- automatic gain
control, a new degenerate solution -- and contradicts A1, whose reference is what
the mic heard, level included.

**Deviations to carry:** Deviation 7 is ours, not CARTSE's. RMS not BS.1770 (not
differentiable; both signals measured identically so the comparison stays
symmetric). Present crops only. Floor unchanged at -25.42: `L_gain` is 0 at
perfect reconstruction.

**Shipped OFF (`w_g = 0.0`)** but still computed and logged -- that is what the
anchor run reads. `test_wg_defaults_to_zero_and_reproduces_the_three_term_total`
pins that the default reproduces the 2026-08-20 objective, so the ablation's
control arm is a real control.

**Watch `pres_abs_gap_db`, not the total.** If it works the gap *widens*. If it
stays flat while both ends rise, the model traded a mute for a pass-through.

**Incidental:** during warm-up `w = 0`, so `L_gain` runs at full strength while
the silence pressure is off.

Code: `losses.py::_loss_gain_match`; `train.py` (`build_loss_fn`,
`HISTORY_FIELDS`, `add_parts`, `epoch_report`); `bsrnn_baseline.yaml`. Seven
tests. **`history.csv` schema changed** -- `train_L_gain`/`val_L_gain` are new,
so older histories cannot be concatenated without filling them.

---

## 2026-08-28 — `w_g` = 1.69 derived. And `L_MR` does not do its documented job

`scripts/derive_w_g.py`, 200 `sir0_val` crops, 6 min CPU, seed 42.
`experiments/results/2026-08-28-wg-anchor-sir0/`.

### The finding that matters more than the weight

**`L_MR` rewards the mute.** The 2026-08-20 entry and `losses.py` both called it
the term that "pins the output gain". Holding the audio fixed and changing only
volume:

| anchor | `L_pres` | `L_MR` | `L_gain` | `L_abs` |
| --- | --- | --- | --- | --- |
| oracle (= clean target) | -30.000 | 0.0000 | 0.0000 | -20.000 |
| pass-through (mixture) | -1.593 | 0.2735 | 4.901 | 0.043 |
| mixture muted to the checkpoint's level | -1.593 | **0.2438** | 17.960 | -18.818 |
| epoch-7 checkpoint | -3.326 | 0.2417 | 17.960 | -18.818 |

Muting ~21 dB **improves** `L_MR`, 0.2735 -> 0.2438. Nothing opposed the mute:
the absent branch paid 8.639, `L_MR` a further 0.155, and the only other
present-branch term is scale-invariant. **There was no fight to lose.** This
supersedes the 2026-08-27 wording that `L_MR` was "losing".

Why the earlier reading was wrong: `w_m = 9.62` was derived on `train` data where
the anchor read `L_MR` 0.1842 / `L_pres` -5.909; on `sir0` it reads 0.2735 /
-1.593. The magnitude was calibrated on a distribution that no longer applies and
the *sign* against attenuation was never tested.

**`passthrough_muted`** is the anchor that made this visible -- the mixture scaled
per crop to the checkpoint's level. Model-vs-pass-through moves gain *and*
separation and would price both. Two wiring checks passed: oracle `L_gain` =
0.000000, and `L_pres` drifted 1.19e-06 pass-through -> muted, which is
Deviation 1's scale-invariance confirmed by measurement rather than by reading code.

### Anchors the 2026-08-27 entry said were missing

Pass-through SI-SDR **+1.59 dB**; the checkpoint **+3.33 dB**; so the model beats
doing nothing by **1.73 dB** -- modest, real, and anchored rather than inferred.
Not comparable to `history.csv`'s -2.317 (different crop protocol); the
comparison *within* this run is valid, both rows using identical crops.

`L_gain` 17.960 puts the output **20.96 dB** off the target's level, consistent
with the -22.4 dB re-mixture figure inferred on 2026-08-27.

### The derivation

    buys on absent   :  w * (0.043 - (-18.818))            = +8.639
    costs via L_MR   : (1-w) * w_m * (0.2438 - 0.2735)     = -0.155  (a BENEFIT)
    L_gain headroom  :  17.960 - 4.901                     = +13.060
    break-even w_g   = (8.639 + 0.155) / ((1-w) * 13.060)  =  1.242

**Chosen: 1.69**, geometric midpoint of [1.242, 2.297], clear of both edges.

**The 2.297 ceiling is a heuristic, not a bound** -- where `w_g` x headroom
reaches `L_pres`'s full 30 dB range. Stated as heuristic because the *achievable*
range is nearer 10-13 dB, which would put it lower. 1.69 clears break-even by 36 %.

**Cross-check.** Differentiating w.r.t. a global attenuation g while both terms
are linear gives `d(total)/dg = -w + (1-w)·w_g`, zero at `w/(1-w) = 0.845`. That
marginal bound and the measured integrated break-even agree in magnitude; they
differ because `L_abs` saturates at -20 and `L_MR` supplies a bonus. Two routes,
same order.

### Consequences

- **Re-derive, never inherit.** Assumes the mute is global (justified at a 2.45 dB
  gap). A checkpoint with a real gate invalidates it.
- **`ablate_w_m` gains a reason.** Its 0 arm now tests a term measured to be mildly
  counterproductive.
- **Write-up:** "L_MR pins the output gain" must be corrected at source in the
  2026-08-20 entry, not quietly dropped. The correction is *why* a fourth term
  was needed.
- `--n-crops 300` yields 200: `sir0_val` has 200 trials and this script builds the
  dataset single-direction.

---

## 2026-08-28 — `L_gain` works. Current architecture FROZEN as the baseline

**Decision: `BSRNN_TFMAP` as it stands at `38bf48f` — TF-Map concatenated once as
a third input channel, no per-layer injection — is the baseline architecture for
M2.** Every architecture change from here is measured against it. Recorded
because the next change (D4a, per-block TF-Map re-injection) would otherwise
leave no fixed point to compare to.

Config `experiments/configs/bsrnn_baseline.yaml`, seed 42, split `sir0`,
7,189,644 parameters, 32 bands. Run in progress on Kaggle T4; `w_g` = 1.69,
`gain_delta_db` = 3.0, warmup 4 + ramp 3.

### Why now: the mute is fixed, and it is measured at a matched epoch

`w` is on the same schedule in both runs, so epoch 5 (`w` = 0.30533) compares
like for like against `2026-08-27-train-sir0`, whose only difference is
`w_g` = 0.

| epoch 5, `w` = 0.305 | control (`w_g`=0) | this run (`w_g`=1.69) |
|---|---|---|
| output level, present crops | **-19.4 dB** re mixture | **-4.6 dB** |
| `val_pres_abs_gap_db` | 2.06 | **2.63** |
| `val_enrol_sens_db` | -10.41 (9.1 %) | **-9.81 (10.5 %)** |
| `val_L_pres` | -1.942 | **-2.212** |
| `val_L_MR` | 0.2490 | **0.2080** |

**In plain terms: under the same silence pressure the old model went quiet and
this one did not.** The target sits ~3.9 dB below the mixture in `sir0`
(SIR ~ U[-10, 10]); this run's output sits at -4.6 dB, i.e. about right, while
the control sat 15 dB below where the target actually is and called it
separation. It also wins on reconstruction and detail, which `L_gain` was not
designed to touch, and its selectivity gap already exceeds the control's own
epoch-7 best of 2.45.

**Caveats, both load-bearing.** The two level figures are *reconstructed* from
`val_L_abs` and `val_pres_abs_gap_db`, not measured — they mix mean-of-dB with
dB-of-mean and are worth about +-1 dB. Quote them as indicative; measure
directly before they go in the thesis. And epoch 6 is the first at full
`w` = 0.458, so this is 2/3 pressure, not full. The control had already
collapsed by epoch 4-5, so the comparison holds, but confirm at epoch 7.

### What this baseline is, and is not

- **It is a strong extraction baseline.** No mute, correct output level,
  improving reconstruction.
- **It is a weak conditioning baseline.** 10.5 % enrolment sensitivity: roughly
  seven eighths of the output is still decided without reference to who was
  asked for. This is deliberate as a starting point — it makes D4a's effect
  measurable — but it must never be described as "conditioning works".
- **It is not yet the baseline of record.** Every number above is a training
  diagnostic. The project's primary metric is live-model content fidelity
  (`docs/data/metric-definitions.md`) and this checkpoint has never been scored
  on it. **A baseline that exists only as a loss curve is not a baseline.**
  Scoring it through the eval harness is the blocking next step.

### Consequences

- D4a (per-block TF-Map re-injection, parameter-free) is the first change
  measured against this. See `decisions-pending.md` D4.
- The `ablate_w_g` 0 arm is now also the architecture-baseline arm.
- `2026-08-27-train-sir0` remains the control for the `L_gain` claim
  specifically, not a general baseline: it is a muted model.

---

## 2026-08-28 — Training made 7x faster. Three changes, one of them the whole story

**Measured on the T4, batch 3: 4.741 -> 0.674 s/step.** Nothing about the model
changed. Evidence and the full sweep in `decisions-pending.md` E3b-E3f.

| # | change | where | gain |
|---|---|---|---|
| 1 | `chunk_s` 4.0 -> **4.008** | `bsrnn_baseline.yaml` | **4.44x** |
| 2 | mixed precision (`amp: true`) | `bsrnn_baseline.yaml`, `train.py` | 1.58x, 1.86x less memory |
| 3 | `pin_memory=True` | `get_data_loaders()` | small |

### 1. The 8 ms that bought 4.44x

**fp16 tensor cores require the batch dimension to be a multiple of 8.**
`BSNet.forward` reshapes to `(B*T, N, K)` for `band_rnn`, so its batch is
`examples x T`. At `chunk_s` 4.0 the STFT yields **T = 503, which is prime**, so
only batch sizes divisible by 4 aligned — and cuDNN silently fell back to a
non-tensor-core kernel for everything else. **4.008 s gives T = 504 = 8 x 63, so
every batch size aligns.**

Found by accident: batch 4 profiled 4x faster than 3, 5 and 6, reproduced six
times at 0.3 % spread, and was the only size whose `examples x T` divided by 8.
The three slow sizes agreed with each other to 0.4 % — one shared fallback kernel.

**The crop is 0.2 % LONGER, not shorter, `chunk_s` is a crop applied by
`TrialDataset` so nothing was re-rendered, and batch stays 3 so the optimisation
is untouched.** Moving to batch 4 would have worked equally well but changes
gradient noise and drops optimiser steps per epoch 663 -> 497; that is a
modelling change and this is not.

**T was mis-stated as 497 for three days.** `profile_step.py` used
`(n - n_fft)//hop + 1`, but `src/models/stft.py` pads by `n_fft - hop` on *both*
sides for overlap-add ramp room. It now measures T from the real STFT and prints
an ALIGNED / NOT ALIGNED verdict.

### 2. Mixed precision

fp16 for the model forward, fp32 for master weights and **the entire loss** --
`L_pres`/`L_abs`/`L_gain` carry 1e-12 epsilons inside `log10` and fp16's smallest
normal is ~6e-5, so an fp16 loss returns NaN. `amp_ctx()` is the single
definition used by train, val and the diagnostic so they cannot drift apart.

Two correctness details that are easy to get wrong: **`scaler.unscale_()` must
precede `clip_grad_norm_`** or the clip compares a ~65536x inflated norm against
`grad_clip` and crushes every gradient to zero — training looks stable and learns
nothing; and the diagnostic's swapped forward is cast back with `.float()` before
its sums of squares, which would otherwise overflow fp16's 65504 ceiling.

Val runs in the same precision as training deliberately: a metric measured in a
precision the model was not trained in describes a model that does not exist.

### Consequences

- **Epoch time is 3.9-7.0x better, not 7x.** The compute step is 7.03x, but of
  the measured 3875 s/epoch only ~3143 s was training; ~516 s is unaccounted
  overhead that may not scale. Projected 550-1000 s/epoch, so 10 epochs in
  **1.5-2.8 h against 10.8 h**. First real run settles it — do not quote 7x for
  an epoch.
- **`amp` is config-driven, not hardcoded**, because it changes training numerics:
  a `history.csv` is only readable next to the flag that produced it. Runs before
  and after this entry are not numerically comparable.
- **Gradient checkpointing withdrawn** and the band-loop refactor dropped: E3e
  measured throughput per example flat across batch sizes, and E3b measured the
  32-band loops at 4 % of forward.
- **Audio compression is settled as worthless** -- loader measured at 0.044
  s/batch on Kaggle, 4.5 % of a step.
- 189 tests pass. The 4.008 s crop and `amp: true` are both live in
  `bsrnn_baseline.yaml`, so the next run uses them.

---

## 2026-08-28 — The 10-epoch `L_gain` run: conditioning finally works

`experiments/results/2026-08-28-train-sir0-e10/`. `sir0`, seed 42, `w_g`=1.69,
warmup 4+3, batch 3, fp32, 10.5 h on a T4 (pre-speed-fix).

### In plain words

**Blocking the model's escape route taught it to listen to the voice sample.**
Swapping in a stranger's enrolment now changes the output by **37.6 %**, against
2.9 % at the start of this run and 14.9 % for the control that had no `L_gain`.
The model also finally knows when to speak: it is **7.1 dB louder on crops where
the target is talking** than on crops where it is not, against 2.45 dB for the
control.

### Head-to-head at epoch 7, both at full `w` = 0.458

| | control (`w_g`=0) | this run (`w_g`=1.69) |
|---|---|---|
| enrolment sensitivity | 14.95 % | **16.88 %** |
| present/absent gap | 2.45 dB | **2.79 dB** |
| **output level, present crops** | **-22.4 dB** re mixture | **-4.2 dB** |
| `val_L_MR` | 0.254 | **0.212** |

The target sits at ~-3.9 dB in `sir0`. **The control was 18 dB below where the
target actually is — it had muted and called it separation.** This run sits at
-4.2 dB, i.e. right.

### The trajectory, and why it matters

| epoch | `w` | enrol % | gap dB | e_pres dB |
|---|---|---|---|---|
| 3 | 0.000 | 5.3 | 1.63 | -3.77 |
| 5 | 0.305 | 10.5 | 2.63 | -4.58 |
| 7 | 0.458 | 16.9 | 2.79 | -4.22 |
| 9 | 0.458 | **37.6** | **7.10** | -4.72 |

**Conditioning improved as a CONSEQUENCE of removing the mute, not because
anything in the conditioning path changed.** With silence available it was free
— `L_pres` is scale-invariant, `L_MR` was measured to reward muting, `L_abs`
rewards silence — so going quiet paid and using the enrolment did not. `L_gain`
priced the escape; the model then had to actually discriminate. **The
architecture was never the problem here; the objective was.** That is the
defensible claim, and it is the strongest result the project has.

### Three things this does NOT say

1. **It is not converged.** `best_val` is epoch **9, the last one**, and
   `early_stopped=False`. Enrolment sensitivity went 28.0 -> 37.6 % on the final
   epoch alone; the curve is still climbing steeply. **Training longer is now the
   cheapest available experiment** — 8.4 min/epoch after the speed fix.
2. **`L_gain` did not achieve its literal objective.** It fell only 3.902 ->
   3.782 (-3.1 %), so per-crop level error is still ~6.8 dB. The output level is
   right *on average* (-4.72 vs -3.9 ideal) while individual crops remain badly
   off — the model is not tracking level per utterance, it has just stopped
   muting. The term worked as a *constraint*, not as a *regression target*.
3. **Still no number on the project metric.** Every figure here is a training
   diagnostic.

### Consequences

- **D4a drops in priority.** It was proposed when enrolment sensitivity read
  10.5 % at epoch 5 of this same run and the diagnosis was "the network is
  throwing the cue away". At 37.6 % that diagnosis no longer holds. 62 % of the
  output is still enrolment-independent so there is headroom, but this is no
  longer the emergency it looked like. Order now: **train longer, then score on
  the metric, then reconsider D4a.**
- The `w` warmup schedule is vindicated: every jump in enrolment sensitivity
  tracks a `w` increase (5.7 -> 10.5 -> 21.0 % across epochs 4-6).
- Epoch 7 is a visible regression on every diagnostic (21.0 -> 16.9 %, gap
  3.69 -> 2.79) before recovering. Single-epoch noise at this scale; do not read
  a trend into any one epoch.

---

## 2026-08-29 — AMP validated over a full 10 epochs. Adopt it

`experiments/results/2026-08-29-train-sir0-e10-amp/`. Same seed (42), split and
`w_g` as the fp32 run; `amp: true` and `chunk_s` 4.008 are the two differences.

**523.0 s/epoch against 3772.6 = 7.21x. Ten epochs in 1.45 h, was 10.5 h.**

### The thing that needed testing, and passed

The 2-epoch speed check only reached `w` = 0. **Epochs 4-9 — the `w` ramp and
full silence pressure, where `L_abs` starts driving and the earlier model
collapsed — had never run in fp16.** They now have: all 10 epochs completed,
`mixed precision: ON (fp16 forward, fp32 loss)` confirmed in the log, and **zero
NaN, inf or error mentions across the whole session**. The `1e-12` epsilons are
safe behind the fp32 loss cast.

### AMP vs fp32, enrolment sensitivity %, epochs 5-9

|  | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| AMP | 8.5 | 24.1 | 20.4 | 28.2 | 31.9 |
| fp32 | 10.5 | 21.0 | 16.9 | 28.0 | 37.6 |

fp32 finishes ahead (37.6 vs 31.9 %, `val_total` -1.673 vs -1.413), **but the
difference is inside the noise**:

- mean |between-run gap| **2.9** points, against mean |within-run epoch-to-epoch
  swing| **8.3**;
- **the sign flips** — AMP is ahead at epochs 6, 7 and 8, behind at 5 and 9.

A systematic precision penalty would not change sign three times. **Do not claim
either run produced the better model.**

**Caveat on that claim.** Epoch-to-epoch variance within one run is a *proxy*
for run-to-run variance, not the same quantity, and n=2 runs supports no real
statistics. Two variables also differ, not one (`amp` and `chunk_s`), though both
are tiny perturbations. Settling it properly needs several seeds — which now
costs 1.45 h each rather than 10.5.

### Consequences

- **Adopt AMP.** It is already the config default; this is the evidence for it.
- Both runs have `best_val` at epoch **9, the last**, and neither early-stopped.
  **Still not converged, in either precision.** Training longer remains the
  cheapest experiment on the board.
- `model_sir0.pt` stays the fp32 checkpoint on best `val_total`; the AMP one is
  `model_sir0_amp-e10.pt`. See `models/README.md` — the choice is arbitrary
  within noise.

---

## 2026-08-29 — Trained longer. It memorises the training set

`experiments/results/2026-08-29-train-sir0-e50-resume/`. Resumed the AMP epoch-9
checkpoint, `sir0`, seed 42, `w_g`=1.69, batch 3, `amp: true`, target epoch 50.
Ran epochs 10-24 and **early-stopped on patience 10**; best `val_total` at epoch
**14**, −2.178. 15 epochs in 2.20 h at 527.8 s/epoch.

### In plain words

Both `L_gain` runs stopped with their best score on the last epoch and neither
early-stopped, so **"just train it longer" was the cheapest experiment on the
board** and this run was it. It answered the question, and the answer is that
there is nothing left to gain from more epochs on this data: the model spends
them learning the 1,989 training trials by heart. It gets better every epoch on
audio it has already seen and steadily worse on audio it has not.

### The two curves that diverge

Separation, as SI-SDR in dB (`L_pres` negated, higher better):

| epoch | train | held-out | gap |
|---|---|---|---|
| 10 | 2.97 | 1.52 | 1.45 |
| 12 | 3.24 | 2.18 | 1.06 |
| **14** | **3.38** | **2.14** | **1.24** |
| 17 | 3.70 | 1.46 | 2.24 |
| 20 | 4.53 | 0.98 | 3.55 |
| 24 | **5.51** | **−0.17** | **5.68** |

Train improves monotonically across all 15 epochs. Held-out improves for four,
then falls the rest of the way — and **by epoch 24 it is below the pass-through
anchor**, i.e. worse than not running the model at all. `L_gain` splits the same
way: 2.92 → 1.51 on train, 3.77 → 5.79 held-out. The generalisation gap widens
by a factor of four.

### The headline that moved for a bad reason

`val_enrol_sens_db` went −3.66 → −0.98 dB, i.e. **43 % → 80 %** output movement
on an enrolment swap, climbing all the way through the collapse. Taken alone that
reads as conditioning tripling. It is not. **An output that has stopped
resembling the target moves a lot when you perturb its input, and the diagnostic
cannot tell that apart from discrimination.**

**Consequence for the write-up: `val_enrol_sens_db` is only interpretable
alongside held-out `L_pres`.** Quote 37.6 % (epoch 9) and 41.7 % (epoch 14).
Never quote the 80 %.

### What this rules out, and what it rules in

**Rules out an under-capacity architecture.** A 7.19 M-parameter model that can
drive training separation to 5.51 dB on 1,989 trials is not too small for the
task; it is too large for the data. This inverts the standing instinct that the
backbone needs replacing — **a bigger or richer model overfits sooner, not
later.** It does not rule out a *better-shaped* model; it rules out a bigger one
as the response to this particular result.

**Rules in data volume as the binding constraint.** Random cropping already
varies the window per epoch (`_crop_offset_start` keys on `(seed, epoch, idx)`)
and `both_directions` already doubles the examples, so the effective set is about
24,000 four-second crops and it still memorises. 19,938 trials are rendered but
only 1,989 are in `sir0`, which is the split where the loudness shortcut is
closed — so the larger set has to be re-rendered symmetric, not just pointed at.

### Consequences

- **`models/model_sir0.pt` should now be the epoch-14 checkpoint**, not epoch 9
  of the e10 run. Everything after epoch 14 is a worse model on held-out data.
- **Next training run is more data, not more epochs and not more parameters.**
  ~5,000 `sir0` trials is ~1,315 s/epoch, so 20 epochs in 7.3 h — inside Kaggle's
  cap. Retrain from scratch: resuming would carry the memorisation forward.
- **Two regularisers are free and currently off.** `weight_decay` is 0.0 by
  explicit choice (`bsrnn_baseline.yaml`) and there is no dropout anywhere. Both
  are config-level. Neither substitutes for data, but both are one line.
- **Early stopping fired, contradicting the M2 checklist note** that it would not
  fire as configured. `patience: 10` on `val_total` stopped the run at 24 from a
  best of 14. That checklist item can be closed.
- **The held-out margin over doing nothing is thin.** Best held-out separation is
  2.14 dB. The pass-through anchor of +1.59 dB was measured by a different
  protocol (`derive_w_g.py`, 200 fixed crops) than the validation loop, so the
  two are not directly subtractable — **re-run the anchor script on the epoch-14
  checkpoint before quoting any "beats doing nothing by X dB" figure.**
- Not logged as an ablation arm: the run is a resume of the AMP run, so it shares
  its seed and its history. It is one trajectory, not an independent sample.

---

## 2026-08-30 — Enrollment bank: rotate the identity cue per epoch (D8a)

Response to the 2026-08-29 overfitting entry. Implemented, tested, unrun.

### The specific hole it closes

`enrollment.wav` is rendered once and read **in full** on every epoch
(`dataset_loader._example`, no crop on the enrollment path). So across the 24
epochs of the 2026-08-29 run, every trial presented the model with the *same
5 s waveform* as its identity cue. Random cropping does not touch this: it
rotates the mixture window, i.e. it resamples the same acoustic scene, while
the thing the model is supposed to generalise over — who is speaking — stayed
bit-identical.

**That makes "this exact waveform -> this exact voice" a lookup table over 1,989
entries, and fitting it is sufficient to explain the observed failure**: training
separation 2.97 -> 5.51 dB while held-out fell 1.52 -> -0.17 dB. Val speakers are
disjoint by construction (`splits.yaml`), so the table transfers nothing.

### What was built

`scripts/render_enrollment_bank.py` renders K enrollment recordings per trial per
direction, `enrollment_v00.wav` .. `v{K-1}.wav`; `dataset_loader` picks one per
`(seed, epoch, idx, direction)`; `data.enrollment_variants` in the config selects
K. **Additive only** — mixtures, targets and interferers are untouched, so no
existing audio, manifest or checkpoint is invalidated.

Variants are distinct **utterances** by the same speaker, not distinct windows of
one utterance: different sentences, usually a different chapter or book, and a
different recording session. A window of the same utterance would leave the
channel and the session identical and is the weak version of this fix.

### Five properties that are deliberate, not incidental

1. **`v00` reproduces `enrollment.wav` byte for byte** — same utterance, same
   offset, same EQ seed. Verified over 24 banks. Without this, `K=1` vs `K=4`
   would compare two different renders and nothing could be attributed to the
   augmentation.
2. **Every variant is levelled to the trial's own `target_loudness_lufs`**, so
   which variant is in play cannot be read off loudness. Verified: spread
   under 3 dB across a bank. Level is a cue closed everywhere else and this does
   not reopen it.
3. **The B8/B10 guard tiers are applied per variant**, through the *same*
   `pick_enrollment` the manifest builder uses, re-evaluated at each draw with
   the already-taken utterances removed. Measured on 12 trials: 32 book-tier,
   40 chapter-tier, 24 utterance-tier draws. The content-leak guarantee holds
   variant by variant rather than only for the first.
4. **Each variant gets its own EQ curve** (the CARTSE channel-gap augmentation,
   Li & Seki 2026, already in the renderer), seeded from `trial_id#v{k}`. So the
   bank varies channel as well as content.
5. **Validation never rotates.** `random_crop=False` forces
   `enrollment_variants` to 1, so val reads `enrollment.wav` itself. A val set
   that moved would make every val number in the project's history incomparable,
   and it is the held-out curve that this whole change is trying to move.

### Cost, and why K defaults to 4

`(K-1) x 2 x ~160 kB` per trial. K=4 on `sir0_train` is **+1.9 GB, ~45 % on top
of the split**, and **~38 min to render** (measured: 100 trials in 2 min, 8
workers). K=8 would double the addition. 4 gives each variant roughly 4-5
appearances over a 20-epoch run, which is enough to break a fixed lookup; the
binding constraint on raising it is the Kaggle dataset upload, not the renderer.

Every `sir0_train` speaker has at least 10 utterances of >= 5 s (median 109), so
no speaker is short of candidates at K=4. Speakers who cannot fill a bank get a
shorter one and the loader falls back to `v00`; the script reports the count
rather than padding with repeats, because a silent repeat would weaken the
augmentation exactly where the speaker is rarest.

### What this is NOT

**Not a substitute for more scenes.** It varies the identity cue and nothing
else: the same 1,989 speaker pairs, rooms, noise beds and SIR/SNR draws remain.
It attacks one memorisation route, the one that is cheapest to close and most
specific to the conditioning failure this project has been chasing since
2026-08-25. Expect it to narrow the train/held-out gap, not to eliminate it.

**Not yet evidence of anything.** 9 unit tests pass and the renderer is verified
on real data; no training run has used it.

### How it will be judged

Run `K=1` and `K=4` from scratch on the same seed and split. The claim is
supported if the **train/held-out separation gap at a matched epoch narrows**.
Held-out `L_pres` improving is the outcome that matters; `val_enrol_sens_db`
alone is not admissible evidence here, for the reason recorded on 2026-08-29 —
it rose through the last collapse.

---

## 2026-08-30 — Per-epoch SIR/SNR remix (D8b). No re-render needed

Second response to the 2026-08-29 overfitting entry, and it composes with the
enrollment bank above. Implemented, tested, unrun.

### The idea in one line

Every trial's loudness balance was a random draw made once, on 2026-08-26, and
then frozen into `mixture.wav`. **The ingredients are still on disk, so the draw
can be made again at load time** — same voices, same words, same room, different
difficulty, every epoch.

### Why it needs no new audio

`render_trial` sums three signals, and two of them are written out, so the third
is recoverable:

    noise = mixture - target - interferer

**Exactly, on every trial**, including the 69 of 1,989 (3.5 %) where A6's clip
guard fired — because A6 applies its common gain to the mixture *and* both stems
(`render_trial` step 4), so the equality survives it. An earlier draft of this
entry claimed those trials needed dividing by `common_gain` first; they do not,
and no `meta.json` read is required.

The rebuild is then two scalar multiplies:

    interferer *= 10 ** ((sir_rendered - sir_new) / 20)
    noise      *= 10 ** ((snr_rendered - snr_new) / 20)
    mixture     = target + interferer + noise

**Only the mixture changes.** `target.wav` is the training reference and is
returned untouched, unless the new sum would clip — in which case it takes the
same common gain the mixture does, which is A6's own rule and keeps the
output/reference level relationship `L_gain` measures intact. Applied to the
crop rather than the clip, because the crop is all the loader has; recorded as a
deviation from the renderer, which guards per clip.

### The new levels are RESAMPLED FROM THE MANIFEST, not from generator.yaml

Each trial borrows another trial's `(sir_db, snr_db)` from **the same difficulty
regime**. Three reasons, and they are the argument for the design:

1. Every value is one the generator actually produced, so no epoch can train on
   an out-of-distribution mixture. Sampling from declared ranges could.
2. The regime mix and any within-regime correlation between SIR and SNR survive
   for free.
3. It needs no second copy of the sampling config to drift from the one the
   manifest was built with. `sir0_train` overrides `sir_db` at the split level to
   `[-10, 10]` while the `hard` regime is derived rather than declared, so a
   re-implementation of that resolution is exactly the kind of duplicate that
   goes stale.

**This does not reopen the loudness shortcut.** `sir0` is a *symmetric* range,
not a pinned value; the shortcut came from the `base` regime's asymmetric
`[0, 12]`, where the target was louder 90 % of the time. Drawing from the same
symmetric pool preserves the property the split exists to enforce.

### Which trials are eligible, and why the rest are not

| condition | n | SIR redrawn | SNR redrawn |
|---|---|---|---|
| `both` | 984 | yes | yes |
| `target_only` | 513 | no interferer to rebalance | yes |
| `interferer_only` | 376 | no | no |
| `noise_only` | 116 | no | no |

**1,497 of 1,989 trials (75.3 %) get the augmentation.** The target-absent
quarter passes through as rendered: on those the loudness anchor is the
interferer or the noise itself (2026-08-11), so the recorded numbers are not
target-relative and re-applying them would be meaningless arithmetic.

### The I/O cost is negative, not positive

The remix needs both speaker stems, so it adds one windowed read per trial. But
`__getitem__` was restructured to read the shared mixture crop **once** instead
of once per direction — the two directions share the crop by construction, so
the second read was always a duplicate.

| | windowed reads per trial |
|---|---|
| before | 6 (mixture x2, target, interferer, enrollment x2) |
| after, remix on | **5** |

Unmeasured for this change. The recorded loader cost is 0.044 s/batch, 4.5 % of
a step (2026-08-28), and `scripts/profile_step.py` is what would settle it.

### Consequences

- `data.remix_gains`, default `false` = the pre-2026-08-30 behaviour. Pinned off
  on any fixed set: `random_crop=False` forces it off, so validation never
  changes difficulty and its curve stays readable.
- **`meta["sir_db"]` and `meta["snr_db"]` now report the REALISED levels**, which
  differ from the manifest whenever the remix fired. Any stratified diagnostic
  therefore describes the audio the model heard, not the audio on disk.
- `CLIP_CEILING` is duplicated in `dataset_loader.py` rather than imported, to
  keep scipy and pyloudnorm out of every DataLoader worker. A test pins the two
  values together.
- Restructuring `__getitem__` is behaviour-preserving when the remix is off, and
  a test asserts the returned tensors are unchanged.

### How it will be judged

Same rule as the bank: `remix_gains` on vs off, same seed and split, and the
claim is supported if the **train/held-out separation gap at a matched epoch
narrows**. The two are separable arms and should be run as such before being
combined, or a joint improvement cannot be attributed.

**What it does not do.** It varies difficulty, not diversity: still 1,172
speakers, still 1,989 rooms, still 1,989 pairs of sentences. It is not a
substitute for rendering more trials, and the learning curve over dataset size
is still the measurement that decides whether more are needed.

---

## 2026-08-30 — D3a: the cue carries identity and the network amplifies it. Conditioning is NOT the bottleneck

`experiments/results/2026-08-30-cue-diag-sir0/`, `scripts/diagnose_cue.py`.
200 fixed crops of `sir0_val`, `model_sir0_e50es.pt` (epoch 14), CPU, 14 min.

### The measurement

Roll the enrollment within the batch and measure the SAME statistic at two
points one layer apart — `||a_true - a_swap||^2 / ||a_true||^2` — so the two
stages are comparable rather than merely similar.

| stratum | n | cue moves | output moves |
|---|---|---|---|
| all | 200 | **28.6 %** | **48.2 %** |
| same-gender trials | 131 | 26.8 % | 44.4 % |
| cross-gender trials | 69 | 31.0 % | 56.1 % |

### The partition, and it is the third branch

The conditioning path has two halves and no previous run distinguished them.
**Neither half is failing.** Swapping a stranger's enrollment moves the cue by
28.6 %, so the cue plainly carries identity; the output then moves by 48.2 %,
i.e. **the network amplifies the cue rather than discarding it.**

The standing diagnosis since 2026-08-25 — "the network is throwing the cue
away" — is wrong, measured. The ~2.3 dB held-out ceiling is in the **separator
or the objective**, not in getting speaker identity into the network.

### Consequences for the D-list

- **D4a (re-inject the TF-Map at every block) — DROP.** It exists to fix
  dilution of the cue across six BSNet blocks. There is no dilution to fix: the
  cue survives and is amplified. Its ranking as "cheapest large change" was
  conditional on D3a, and D3a says no.
- **D5 (speaker encoder + auxiliary speaker-ID loss) — DEMOTE.** The argument
  was that no parameter is devoted to identity. True, and it turns out not to be
  the binding constraint. Not refuted, but no longer indicated.
- **D2 (attention temperature) — ANSWERED WITHOUT RUNNING IT.** The softmax now
  blends **138 of 628** enrollment frames, max weight 0.067 against a uniform
  0.00159. The 2026-08-25 measurement that motivated D2 had it blending ~620 of
  628. `tfmap_scale` = 16 already did what the temperature sweep was going to
  test. Close it.
- **D1 (learned dictionary) — DROP the premise.** D1 and D2 shared one
  hypothesis: more selective matching against enrollment content improves
  extraction. The matching is already selective and extraction is still capped.

### D3c, and a correction

Sensitivity is **higher on cross-gender trials (56.1 %) than same-gender
(44.4 %)**. That is the direction a pitch shortcut produces: where the two
speakers differ in gender the enrollment's identity matters more to the output.
It is a weak signal, not proof — but it is not nothing, and `sir0_train`'s 50/50
gender balance caps rather than removes the shortcut (a model using pitch alone
scores ~75 % correct with no enrollment at all).

**Correction:** a 24-crop smoke run of the same script read the opposite
ordering (same-gender 44.9 % vs cross-gender 37.1 %) and was briefly described
as showing no pitch reliance. At n=200 the ordering reverses. The 24-crop figure
was noise and must not be quoted.

### Caveats

- One checkpoint, one split. The cue half needs no trained model (TFMap is
  parameter-free) so it generalises across checkpoints; the output half does not.
- `same_gender` describes whether a trial's TARGET and INTERFERER share a
  gender, not the gender of the rolled-in enrollment. It is the right
  stratification for detecting a pitch shortcut and is not a statement about the
  swap itself.
- This says where the ceiling is NOT. It does not say where it is.

---

## 2026-08-31 — Response to memorisation: 5,000 trials, weight decay on, bank K=3

**Three config changes, one hypothesis.** The 2026-08-29 run memorised its
training set (see that entry): train separation improved every epoch, 2.97 →
5.51 dB, while held-out peaked at 2.14 dB and fell to −0.17 dB, below
pass-through. The diagnosis was **data volume, not architecture** — 1,989 trials
is not enough for 7.19 M parameters. These changes act on that.

| setting | was | now | file |
|---|---|---|---|
| `splits.sir0_train.n_trials` | 2,000 | **5,000** | `generator.yaml` |
| `training.weight_decay` | 0.0 | **1.0e-4** | `bsrnn_baseline.yaml` |
| `data.enrollment_variants` | 4 | **3** | `bsrnn_baseline.yaml` |

### Why 5,000 and not more

**It is the diagnostic size, not the maximum.** 2.5x data is enough to answer
"does the train/held-out gap narrow". If it does not narrow at 2.5x, the
data-volume diagnosis is wrong and a bigger render would have been wasted
effort and upload. Scale to 10,000 only after 5,000 confirms it.

Projected from the measured 523–568 s/epoch at 1,989 trials: **~1,360 s/epoch,
so ~29 epochs fit one ~12 h Kaggle session.** The previous best epoch (14) sat at
~18,600 optimiser steps; at 5,000 trials that is around **epoch 6**, so 25–29
epochs gives roughly 4–5x headroom past where the model previously turned.

### Why weight decay, and the confound it creates

`weight_decay` was 0.0. AdamW's 0.01 default would decay every LSTM weight,
which is why it was refused before; **1e-4 is two orders below that,
deliberately timid.**

**CONFOUND, stated rather than discovered later: this run changes data volume AND
regularisation together, so an improvement cannot be attributed between them.**
Accepted, because both attack the same hypothesis (too much capacity for the
data available) and a workable checkpoint matters more than a clean ablation with
six weeks to the freeze.

**Correction to `project-state.md`, which called weight decay and dropout "the
two free regularisers... both config-level".** Only one is. **There is no dropout
anywhere in `src/models/`, `bsrnn_baseline.yaml` or `scripts/train.py`** —
adding it means editing the architecture frozen on 2026-08-28, which is its own
arm, not a free config flip.

### Why the bank drops to K=3

Purely to buy upload budget. The bank costs (K−1) × 2 × ~160 kB per trial, so
K=4 → K=3 saves 320 kB/trial, **~1.6 GB at 5,000 trials.**

**This is a change to a measured setting.** The arm that established the bank
used K=4 (2026-08-30), so this run cannot be compared to that arm on the bank
axis alone. Untested whether 3 rotations regularise as well as 4.

### Disk: the projection in `run_times.md` was wrong by 2.6x

`run_times.md` projects **1.26 MB/trial**. Measured on `sir0_train`, it is
**3.25 MB/trial** — 6.3 GB for 1,989 trials — because sir0 renders the interferer
stem *and* two enrolment banks:

| component | share |
|---|---|
| mixture + target + interferer stems | 56 % |
| target enrolment bank | 24 % |
| interferer enrolment bank | 24 % |

So 5,000 trials is **~15 GB, not ~6 GB**, and upload is the binding constraint
rather than render time or disk.

### Two things that will bite silently

**`sir0_val` must NOT be re-rendered.** The whole comparison is against the
08-29 baseline and is only valid if the validation set is byte-identical. Only
the *train* manifest is rebuilt. The old 1,989-trial manifest is archived to
`data/manifests/archive/` because `build_manifest.py` overwrites in place, and
without it the 08-29/08-30 runs lose the manifest they trained against.

**All 5,000 must be rendered fresh; the existing 1,989 cannot be reused.**
Editing `generator.yaml` changes its `config_md5`, and `render_trials.py`
records that hash and refuses to extend a directory built under a different one.
That refusal is correct: the manifest fixes every level, position and onset, so
every row now describes different audio.

**The `w` warmup is defined in EPOCHS, not steps.** `warmup_epochs: 4` +
`ramp_epochs: 3` means full `w` at epoch 7 — ~9,300 steps at 1,989 trials, but
~23,300 at 5,000. That is a **2.5x longer warmup in real terms**, silently. The
warmup exists to stop the early mute so longer is likely safer rather than
harmful, but it is a genuine difference from the baseline, and the config already
flags `warmup_epochs` as "A GUESS".

### What to watch

**The train/held-out gap, not the total.** Both totals fell the whole way through
the run that overfitted. The gap on `L_pres` was 1.24 dB at the old best epoch
and 5.68 dB by epoch 24. **Confirmation looks like: the gap stays near 1 dB while
held-out separation climbs past 2.14 dB.** The per-epoch breakdown printed to
stderr now shows that column directly (see the logging change below).

### Logging changed in the same commit

`tqdm` removed from `scripts/train.py`. At ~1,666 batches per epoch it emitted
more lines than the whole rest of the run and buried the per-epoch numbers.
Replaced by `format_epoch_breakdown()`, which prints the train/val/gap table once
per epoch **to stderr** — stdout still carries one CSV row per epoch and must stay
a valid `history.csv` so a killed Kaggle session can be recovered by pasting it
into a file.

---

## 2026-09-01 — 2.5x the data: the diagnosis is CONFIRMED, and the model is still data-limited

`experiments/results/2026-09-01-train-sir0-5000/`. 4,976 trials, 18 epochs of a
requested 25, early-stopped on patience, 6.2 h at 1,244 s/epoch. Best epoch 7
(zero-indexed, as `meta.yaml` records it). Checkpoint kept as
`models/model_sir0_5000-e7.pt`.

**The prediction was that the train/held-out gap would narrow if the 2026-08-29
memorisation was a data-volume problem. It did, and held-out separation improved
with it.**

| | 1,989 trials (08-29) | 4,976 trials (09-01) |
|---|---|---|
| best held-out separation | 2.14 dB | **2.58 dB** |
| margin over pass-through (1.59 dB) | 0.55 dB | **0.99 dB** |
| gap at the best epoch | 1.24 dB | **1.08 dB** |
| gap at the last epoch | 5.68 dB (ep 24) | 4.16 dB (ep 18) |
| epochs to reach 2.14 dB | 14 | **2** |

**The margin over doing nothing nearly doubled**, 0.55 -> 0.99 dB. That is the
honest framing: 2.58 dB sounds close to 2.14 dB, but the reference point is
pass-through at 1.59 dB, and what the extractor adds is what matters.

### It is NOT at capacity. It is still data-limited

The two are distinguishable by signature, and this run shows the second:

- **capacity-limited** looks like: train loss plateaus, val plateaus with it, the
  gap stays small. The model cannot fit even what it has seen.
- **data-limited** looks like: train keeps improving, val peaks then declines,
  the gap grows.

Train `L_pres` fell monotonically from 2.30 to 5.64 dB across all 18 epochs and
never plateaued, while held-out peaked at epoch 7 and fell to 1.48 dB by epoch
17. The gap grew 0.58 -> 4.16 dB. **That is unambiguously the data-limited
signature**, so more data would still help.

**But the return is diminishing.** 2.5x the data bought +0.44 dB of peak
held-out separation. A further 2x should be expected to buy appreciably less,
and costs ~2.6 h of rendering, ~2 h of enrolment-bank rendering, a ~33 GB
upload, and caps a Kaggle session at ~14 epochs.

**Decision: do NOT render more data now.** The extractor is not this project's
contribution and there is no live-model measurement taken yet. See
`decisions-pending.md` for the two cheaper levers if the model is revisited.

### Two selection findings worth keeping

**`select_on: present_branch` earned its place.** `val_total` was lowest at epoch
15 (-2.7455), not at the selected epoch 7 (-2.4261). Epoch 15 wins on `total`
only because its `L_abs` is better (-12.84 against -11.43) -- it is *quieter*,
not better at separating; its held-out separation is 2.40 dB against epoch 7's
2.58 dB. `total` contains `L_abs`, and `L_abs` rewards silence, which is exactly
why it cannot rank finished models (`decisions-m2.md` 2026-08-30). **A plot of
`val_total` will therefore disagree with the selected epoch, by design.**

**The `select_abs_max` silence bar also fired.** Epoch 4 had the best raw
separation of the whole run at 2.94 dB but `L_abs` -8.38, above the -10.0 bar,
so it was ineligible. It separated well while remaining audible on crops where
the target never speaks. The constraint rejected it correctly.

### Conditioning: unchanged, and read it with care

Enrolment sensitivity at the selected epoch is **-3.79 dB**, against -3.80 dB for
the 08-29 baseline. **No change.** More data improved separation without
improving conditioning.

It reaches -1.53 dB by epoch 17, and **that figure must not be quoted**: held-out
separation was collapsing over the same epochs (2.58 -> 1.48 dB). This is the
same "headline moving for a bad reason" pattern recorded on 2026-08-29 -- an
output that has stopped resembling the target moves a great deal when its input
changes, for no useful reason. `pres_abs_gap` behaves the same way, 8.86 dB at
the selected epoch against 15.12 dB at the end.

### Housekeeping

`kaggle_out/` deleted after extraction. Kept: `history.csv`, `history_live.csv`,
`meta.yaml`, `loss_plot.png`, the config that ran, `bundle_commit.txt`, and a
`source_that_ran/` snapshot. Checkpoints kept as `models/model_sir0_5000-e7.pt`
(selected) and `models/model_sir0_5000-last.pt`. The `keep_top_k` checkpoints
e004/e005/e007 were discarded: e007 is the selected epoch and is already kept,
and the other two were epochs the selection rule rejected.

---

## 2026-09-03 — The `w` warmup is now indexed in GRADIENT STEPS, not epochs

**Change `loss.w_schedule` from `warmup_epochs`/`ramp_epochs` to
`warmup_steps`/`ramp_steps`, and apply it per batch instead of once per epoch.**
An epoch-indexed schedule changes length in the unit the optimiser actually
moves in whenever the training set changes size, which silently confounds the
data-scaling curve.

### The problem, in numbers

`warmup_epochs: 4` + `ramp_epochs: 3` is a fixed number of *epochs*, so its
length in gradient steps is a function of the manifest:

| trials | steps/epoch | warmup | full `w` at |
|---|---|---|---|
| 1,989 | 663 | 2,652 steps | step 4,641 |
| 4,976 | 1,658 | 6,632 steps | step 11,606 |
| 9,955 | 3,318 | 13,272 steps | step 23,212 |

Steps/epoch is `floor(n_trials / batch_size)` — the loader counts TRIALS, and
`both_directions` widens each batch rather than lengthening the epoch.

So the three points of the data-scaling curve were each running a **different
schedule**, at 1x, 2.5x and 5x the warmup length. The curve exists to measure
data volume alone; this was a second variable moving with it, unlogged and
unnoticed. It was flagged as a risk on 2026-08-31 ("the `w` warmup is defined in
EPOCHS, not steps") and accepted for the 4,976 run; at 9,955 it doubles again,
so it is now fixed before the run rather than after.

**Why it matters specifically.** `w` is the one knob that stops the early mute
(2026-08-25: the model muted to −18.5 dB by epoch 1 and learned silence before
conditioning). A longer warmup is probably *safer*, not harmful — but "probably
safer" is not a controlled comparison, and the 9,955 run's best epoch is
projected around 4–5, so under the old schedule the selected checkpoint would
have been taken **mid-ramp**, before the objective was ever the one being
reported, and compared against a 4,976 checkpoint taken at full `w`.

### What was chosen

`warmup_steps: 6632`, `ramp_steps: 4974` — exactly 4 and 3 epochs at the
**4,976-trial baseline's** 1,658 steps/epoch. The 2026-09-01 run is the arm the
new run must be comparable to, so its schedule is the one pinned. Full `w` from
step 11,606: epoch 7.00 at 4,976 trials (identical to the baseline), epoch 3.50
at 9,955.

`warmup_epochs`/`ramp_epochs` are still honoured by `schedule_in_steps()`, so
every run up to 2026-09-01 reproduces from its logged config. Setting both forms
at once is refused with an assertion — silently honouring one unit and ignoring
the other is the failure mode worth failing loudly on.

### The one real difference from the baseline, stated

The epoch-indexed schedule was a three-tread **staircase**: `w` held flat for a
whole epoch, then jumped. The step-indexed one is a **continuous ramp** over the
same span. They agree exactly at each epoch's *last* step and share both
endpoints (where warmup ends, where full `w` arrives), but not in between —
measured, the mean `w` across the ramp is **50.0 % of the final `w`, against the
staircase's 66.7 %**. So slightly less absent-branch pressure is applied during
the ramp. This is not a rescaling of the curve and it is not free of
consequence; it is recorded here so a difference between the 4,976 and 9,955
runs is not attributed to data when part of it may be this.
Pinned in `tests/test_w_schedule.py::test_step_schedule_matches_epoch_schedule_at_epoch_ends`.

### Implementation notes

- `_w_ramp()` holds the shape in whatever unit is passed, so the step-indexed
  and legacy epoch-indexed schedules are provably the same curve — only the unit
  of `t` differs. `schedule_in_steps()` is the single place the config is
  converted to steps, so the loop, the startup print and the tests cannot
  disagree about where the ramp ends.
- `global_step` counts **batches**, not successful optimiser updates. AMP's
  `GradScaler` skips a step whose gradients hold inf/NaN, and a schedule that
  moved with those skips would not be reproducible from the config and seed
  alone.
- **`global_step` is persisted in both resumable checkpoints and restored on
  `--resume`.** Without it a resumed run restarts the warmup and re-runs the
  early-mute risk — the 2026-08-29 run was a resume. Checkpoints written before
  today have no such key; the fallback reconstructs it as
  `start_epoch * steps_per_epoch`, which is exact provided the training set has
  not changed size, and a resume across a config change is already refused.
- The `w` column in `history.csv` is now the **mean over the epoch's steps**.
  Identical to the old value wherever `w` is constant — every epoch after the
  ramp, and every epoch of a legacy run — and inside the ramp it is the weight
  that actually trained the epoch rather than one endpoint of it. Validation
  totals are unaffected: `epoch_report()` recombines the raw per-term means at
  `w_report`, never at the instantaneous `w`.
- `w_at_epoch()` is kept as the legacy entry point (this file cites it by name
  under 2026-08-25) and asserts if handed a step-indexed config.

### Not done

**The absolute length of the warmup is still a guess.** 6,632 steps is "what the
baseline happened to do", not a measured requirement — no experiment says how
many steps conditioning needs. What this change buys is that the guess is now
the *same* guess at every dataset size. The end-of-warmup
`val_enrol_sens_db` remains the thing that tells you whether it was enough.

---

## 2026-09-03 — Getting 9,955 trials onto Kaggle: `--new-only`, and a measured upload wall

**Add `--new-only N` to `make_kaggle_bundle.py` and a two-dataset merge cell to
the notebook,** so the half of the trials already on Kaggle need not be sent
again. Recorded here rather than in m1 because it gates the 9,955-trial run, not
the architecture.

### The situation

The sir0 manifest grew 4,976 -> 9,955 rows. The bundle is `ZIP_STORED`, so the
zip is 30.3 GB (130,619 files), built in 2.1 h. Uploading it in full is a large
cost paid for nothing: `build_manifest.py` **appended** rows rather than
regenerating, so the first 4,976 still describe the same audio.

**Proven, not assumed.** Header + first 4,976 rows of the current manifest is
1,898,599 bytes; the `sir0_train.csv` already stored in `tse-audio-s0-v2` is
1,898,599 bytes. Identical. `check_additive()` re-runs that comparison at bundle
time against `--prefix-manifest` and aborts on mismatch, because the failure it
guards is silent: a dataset whose first half describes different audio than its
manifest claims trains without error and is simply wrong.

### Design notes

- Train rows are trimmed; **val is always staged whole**. It is ~200 trials
  against ~5,000, so the saving is not worth a second class of partial-bundle
  bug, and it leaves the new dataset able to verify itself.
- The staged **manifest stays complete**. The older dataset carries the short
  4,976-row manifest it was built with, and a run that read that one would train
  on half the data and report success. The notebook therefore takes manifests
  from the NEW dataset.
- `verify()` gained a `man_dir` override. Under `--new-only` the bundle holds
  the full manifest but only the tail of the audio, so verifying against it
  would die on the first un-staged trial -- a false alarm indistinguishable from
  a real missing-file bug. It now verifies against a temporary manifest filtered
  to what was actually staged, so the check still does its job.
- The notebook merges the two mounts by symlinking at the **trial-directory**
  level: ~10k links, not ~130k. It links the manifests in too, so the merged
  tree is a complete data root -- the batch-size probe derives its manifest
  directory as `<data-root>/manifests`, and a rendered-only merge would fail
  there in a way that looks like an OOM rather than a path bug. It then counts
  manifest rows against trials on disk, which is what catches a half-merged tree.

### The upload is throughput-bound, and `--new-only` does not fix that

Measured 2026-09-03, uploading the full zip with the Kaggle CLI:

| path | rate |
|---|---|
| Kaggle dataset upload | **~95 kB/s** |
| 50 MB to an unrelated public endpoint, concurrently | **785 kB/s** |
| the line, per the user's speed test | 192 Mbps (~24 MB/s) |
| sequential read off the zip (HDD) | 81 MB/s |

Neither the disk nor the line explains it; the ingest is ~8x slower than an
unrelated target on the same connection at the same moment. **At 95 kB/s the
30.3 GB zip is ~86 h, and the `--new-only` half is still ~43 h.** Halving the
bytes does not rescue a rate that low, so `--new-only` is worth having but is
not the answer to this: the fix is a faster path to Kaggle (campus network),
not a smaller payload.

Compression was considered and **rejected for now**: DEFLATE measured 37 % off
on 25 real trials (68.5 MB -> 43.1 MB), confirmed by Kaggle storing the previous
15.35 GB upload as 9.55 GB. Real, but it re-zips 30 GB on a seek-bound HDD to
attack the wrong bottleneck. The bundler's comment claiming DEFLATE "buys almost
nothing" is measurably wrong and should be corrected when the flag is revisited.

---

## 2026-09-04 — 2x the data again: the scaling curve HAS a slope, and it is log-linear

**9,955 trials. Best held-out separation 2.900 dB against the 4,976-trial run's
2.584, and the margin over pass-through 0.991 → 1.307 dB, +32 %.** Selected
epoch 6, `L_abs` −11.30 so eligible, 16 epochs, no early stop, 10.5 h at
2,364 s/epoch. `experiments/results/2026-09-04-train-sir0-10000/`,
`models/model_sir0_10000-e6.pt`.

### The curve, which is the point of the run

| trials | best eligible | margin over pass-through | increment |
|---|---|---|---|
| 1,989 | 2.135 | 0.542 | — |
| 4,976 | 2.584 | 0.991 | +0.449 (x2.50 data) |
| **9,955** | **2.900** | **1.307** | **+0.316 (x2.00 data)** |

**Roughly linear in log(data), and still paying.** Fitting the first two points
and extrapolating predicted +0.333 for a 2x increase; the measured value is
+0.316. So each doubling buys ~0.32 dB of margin and the next one costs ~20,000
trials — about 30 GB rendered and a further day of upload for the same increment.
**That is the answer `project-state.md`'s "Do NOT render more data" was
asserting without a third point.** The advice was directionally right and is now
measured rather than projected; the entry should be updated to say so.

**The 2.5x -> 2.0x confound is now retired.** The 09-01 run changed data volume
AND turned on `weight_decay` 1e-4 together, so its +0.449 could not be attributed.
This run changes data volume alone against that arm, and produces an increment
that lands on the log-linear extrapolation of a gain that was *supposed* to be
partly weight decay. Weakly, that suggests the 09-01 gain was mostly data — not
proof, since one point cannot separate two variables retrospectively, but it is
the only evidence available and it points that way.

### Model selection did real work, twice

**The selected epoch is 6 at 2.900 dB, not epoch 7 at 2.941.** Epoch 7 separated
0.041 dB better and reconstructed at a worse level (`L_gain` 3.528 against
3.404), losing on the present-branch score 4.768 against 4.599. The rule
correctly refused a checkpoint that separates marginally better while getting
the output loudness more wrong. **Never quote 2.941 — it is not the deliverable.**

**The silence bar rejected epochs 1 and 2**, at 2.820 and 2.896 dB with `L_abs`
−7.55 and −9.63. Both sat inside the `w` warmup, where nothing yet pushes the
model quiet, so both were loud on crops where the target never speaks. This is
the third time the bar has fired on a high-separation epoch (09-01's epoch 4 at
2.943 dB, `L_abs` −8.38) and it remains correct each time.

**Consequence worth stating in the write-up:** this run reaches the 09-01 run's
*all-time raw peak* (2.943 dB) while obeying the silence constraint that
disqualified it. That is a cleaner claim than the +0.316 dB, because it is about
satisfying a requirement the previous model could not.

### Overfitting: later, from a higher peak, and it still arrives

Gap at the selected epoch **0.905 against 1.079**, 16 % narrower, and flat over
epochs 4–7 (0.999, 0.922, 1.018, 0.905) where the 09-01 run's was already
widening. Past epoch 7 it goes 1.340 → 4.353 and held-out separation collapses to
**1.282 dB, below pass-through**. So more data delays memorisation and raises the
peak; it does not remove the ceiling.

**Two diagnostics improve throughout that collapse** — `enrol_sens` to −0.99 dB
and `pres_abs_gap` to 15.30 dB, both their best values of the whole run, at the
epoch where the model is worse than doing nothing. Identical to 2026-08-29.
**Neither may be quoted as a conditioning or selectivity result.** `enrol_sens`
is additionally unreadable past about −3 dB (2026-08-30: an ideal extractor
scores 0.00 and an arbitrary one +3.01, so the good and bad cases are not
separable there).

### The step-indexed `w` schedule, validated

First run under it (`decisions-m2.md` 2026-09-03). Predicted mean `w` of 0.000 /
0.420 / 0.458 at epochs 1 / 4 / 5; measured 0.000 / 0.4201548 / 0.458. Full
weight at step 11,606 at 9,955 trials exactly as at 4,976 — **the schedule is now
invariant to dataset size, which is what made this run comparable to its
baseline at all.** `global_step` 53,088 is persisted in `_last.pt`, so the
resume path carries the schedule position as designed.

Epoch time 2,364 s against 1,244 at 4,976 trials: **1.90x for 2.00x the data**,
i.e. slightly sublinear, and 16 epochs fits one Kaggle session with 1.5 h spare.

### Not answered

**Whether any of this reaches the downstream metric.** The 09-01 checkpoint's
2.584 dB converted to only 10.3 % of the available LCF-WER headroom, and the
conversion has been poor at every point on this curve. A 12 % gain in separation
should not be assumed to move LCF-WER, ICR or FR at all. Estimates and
`evaluate.py` on this checkpoint are the next step, and the result is a finding
either way.

---

## 2026-09-12 — the state-teacher arm, read. It moved its own term by 1.4 % and cost 30 % of the selection score

**Run** `experiments/results/2026-09-11-train-sir0-state/`, `bsrnn_state.yaml`,
seed 42, `sir0` (9,955 trials), 10 epochs, 10.25 h on a T4, `early_stopped:
false`. 7,189,644 parameters — identical to the control, so no capacity
confound. **Control** `models/model_sir0_10000-e6.pt`. Checkpoints installed as
`models/model_sir0_state-e{4,5,6,9}.pt`.

**In plain words.** The model was given one extra instruction: a frozen teacher
listens to the output and says whether a second voice is still audible, and the
model is pushed towards "no". It did that, slightly. It also got worse at
reconstructing the target and no better at separating it.

### At epoch 6, which both runs selected

| | control | arm |
|---|---|---|
| selection score (present branch) | **4.599** | **5.977** (+30 %) |
| `L_pres` | −2.900 | −2.847 |
| `L_MR` | 0.1815 | 0.3072 (+69 %) |
| `L_abs` | −11.30 | −11.95 |
| selectivity gap | 9.005 | 9.832 |
| enrolment sensitivity | −3.660 | −3.874 |

**88 % of the 1.379 gap is `L_MR` alone** (9.62 × 0.1257 = 1.209).

`L_state` on validation went 1.986 → 1.796 (e6) → 1.679 (e9). Against the
2026-09-11 anchors — clean target 0.121, mixture 2.106, control 1.820 — that is
**1.4 % of the control's remaining leakage reading removed at the selected
epoch**, 8.3 % by epoch 9.

### Why this is NOT yet a verdict

`L_MR` and the present-branch score are signal-fidelity measures and this
project does not optimise signal fidelity (CLAUDE.md). A model that suppresses
harder, reconstructs less prettily and leaks less can still win on live-model
content fidelity. **Nothing in this run measures the project's metric.** The
eval suite decides it; `scripts/run_eval_suite.py` now runs the whole battery in
one command.

### "It was still going down" — checked against the control, and no

The arm stopped at the 10-epoch ceiling, not at convergence, and `train_total`
was still falling (−5.072). So was the control's, to −8.608 at epoch 15. **The
control's SELECTION score got worse every epoch after 6**: 4.599 (e6) → 6.314
(e10) → 8.019 (e15), `L_pres` −2.900 → −1.282, while `L_abs` fell to −13.8, the
gap grew to 13.5 and enrolment sensitivity improved to −0.99. Quieter, more
selective, progressively worse at reconstruction, training loss falling
throughout — the 2026-08-25 pattern, slower. The arm is already on it: `L_pres`
−2.847 (e6) → −2.417 (e9), gap 9.8 → 12.7.

**Open, and cheap.** The control's late epochs have never been scored on the
project metric. `models/model_sir0_10000-last.pt` is epoch 15. If
over-suppression helps the judge, "the back half is worse" is a claim about the
wrong metric, and the selection criterion itself is the thing to revisit.

### The teacher is sound — the anchor table was not

`scripts/diagnose_state_teacher.py`, 8 `sir0_val` trials, whole clips, interferer
attenuated with the noise bed held fixed:

| interferer | 0 dB | −6 | −12 | −20 | −30 | none | target only | noise only |
|---|---|---|---|---|---|---|---|---|
| `L_state` | 3.092 | 2.640 | 2.144 | 1.117 | 0.296 | 0.079 | 0.052 | 0.029 |

Monotonic throughout; 0.029 on noise alone, so it is not firing on the noise bed.

**The non-monotonic table in `2026-09-11-wstate-anchor-sir0/meta.yaml` (2.106 →
2.589 → 2.714 → 2.744 as the interferer was attenuated 0 → 20 dB) was a bug in
`derive_w_state.py`.** `read_interferer` took `audio[:n_samples]` on the stated
grounds that `random_crop=False` crops at 0. It does not: `_crop_offset_start`
draws from `(seed, 0, idx)` in both modes, so the offset is reproducible and
almost never zero. A misaligned stem leaves the real interferer inside the
recovered `noise` and subtracts a shifted copy, so `target + beta*interferer +
noise` carries a SECOND voice at amplitude (1 − beta) — exactly zero at beta = 1,
which is why the passthrough anchor looked right and nothing was caught.

Fixed: the loader now reports `meta["crop_start"]` and the script reads each stem
there. `tests/test_crop_alignment.py` pins it with ramp-valued stems — the
existing fixture uses constants, which are offset-invariant and could never have
caught this.

**`w_state = 0.002692` STANDS and does not need re-deriving.** It comes from the
measured gradient share at the model anchor (`share / measured_share`, verified
by direct re-measurement at 15.000 %), and the partial anchors never entered it.
Only the dynamic-range display was wrong.

### MEASURED 2026-09-12 — the arm's content verdict. The teacher WORKED, and it still lost

`experiments/results/2026-09-12-eval-state-e6-asr/`, `sir0_val` `both`, n=103,
same trials as the control. **Listener is faster-whisper `small.en`, a STAND-IN
for the judge — not a live-model result.** Estimates in
`2026-09-12-est-state-e6` (13 min), evaluation 6 min.

**In plain words: it removed some of the other speaker's words, added some
invented ones, and the listener transcribed more junk overall.**

| | control e6 | state arm e6 | |
|---|---|---|---|
| **LCF-WER** | **59.52** | **61.23** | **+1.72 worse** |
| substitutions | 28.20 | 28.35 | +0.15 |
| deletions | 11.79 | 11.15 | −0.64 better |
| **insertions** | **19.53** | **21.74** | **+2.21 worse** |
| ICR@2 (leakage) | 50.49 | 46.60 | **−3.88 better** |
| mean leaked | 34.58 | 33.19 | **−1.38 better** |
| FR@2 (trials with >=2 invented) | 68.00 | 64.65 | −3.35 better |
| invented per trial | 3.08 | 3.18 | +0.10 worse |
| no response | 2.91 | 3.88 | +0.97 worse |

### The error attribution, which is what makes this interpretable

`scripts/analyse_leakage_share.py`, the arm added to `SYSTEMS`:

| | control | arm | |
|---|---|---|---|
| wrong content words reported | 742 | 732 | −10 |
| of those, the interferer's | **434** | **417** | **−17** |
| of those, nobody's (invented) | **308** | **315** | **+7** |
| leakage share of the error | 58.49 % | 56.97 % | −1.52 |

**It traded 17 leaked words for 7 invented ones.** On content words that is a
net gain of 10. On the defined metric it is a 1.72-point loss, because the
insertions it added are largely NOT content words — the content-word count strips
stopwords and LCF-WER does not. A rougher output makes the listener emit more
filler.

### The mechanism signature, and it is unambiguous

| | control | arm |
|---|---|---|
| WER of the LEAST-leaky quartile | 24.95 | **26.26 (worse)** |
| WER of the MOST-leaky quartile | 101.53 | **99.57 (better)** |

**The arm helps where leakage is the problem and hurts where it is not.** That is
exactly what a term trading artefact for suppression should look like, and it is
the strongest evidence yet that the two error families are separate and that
attacking one alone cannot close the gap.

### What this authorises

- **The teacher is vindicated as a mechanism.** `L_state` moved only 1.4 % and
  leakage still fell 3.9 points on an independent measure. The term does what it
  claims; the claim is just not sufficient.
- **D14's own caveat is now measured, not argued.** "Removing leakage removes the
  error" was explicitly NOT authorised by step 0, and this is the demonstration:
  leakage down, total error up.
- **D15 (mask roughness) gains real support.** It predicted that suppression
  bought with a rougher mask would raise invention and insertions. Measured:
  leakage −17 words, invention +7, insertions +2.21 points, `L_MR` +69 %.
  D15 step 0 is free and is now the best-motivated open measurement.

### What it does NOT authorise

**NO SIGNIFICANCE CLAIM.** n=103, one run, one seed. No same-config replicate
exists anywhere in `experiments/results/`, so run-to-run scatter has never been
measured on this codebase (decisions-pending.md, 2026-09-11). **+1.72 LCF-WER is
not separable from noise** and must not be written up as a defeat, only as "did
not improve". The same applies to the −3.88 ICR@2 in the arm's favour.

**And this is the ASR stand-in, not the judge.** Whisper prices artefacts as
insertions. A live conversational model may weigh them differently — in either
direction. The arm passed the leakage gate, so the judge run is now the
interesting question rather than a formality.

---

## 2026-09-15 — D17's structure term CANNOT SEE the difference it was built to make. D18 answered from data already on disk

**In plain words: the model was given an extra instruction — "make your mask
vary across frequency the way the perfect mask does" — and the instruction
turned out to be unable to tell a detailed mask apart from a featureless one.
On half the audio it actually preferred the featureless one. The model then did
what it was told and made its mask more featureless.**

**No new run.** The measurement was already in
`experiments/results/2026-09-13-wstruct-anchor-sir0`, written by
`scripts/derive_w_struct.py`, which scores `L_struct` on three reference masks.
This entry re-reads it; `per_crop.csv` had never been opened.

### The three anchors, and the margin between them

| `L_struct` scored on | value | |
|---|---|---|
| a **flat** mask — zero frequency deviation everywhere | 0.16537 | |
| **the model's own** mask | **0.16472** | better by **0.39 %** |
| the **oracle** mask | 0.0 | the term's true optimum |

**0.39 % is the entire range the term had to work with** at the point where the
model actually sits. It was then weighted, by `scripts/derive_w_struct.py`, to
take **15 % of the parameter gradient** (`w_struct` = 46.2981).

### The per-crop table, which is where the mean lies

25 batches / 50 crops, `sir0`, `model_sir0_10000-e6`:

| | |
|---|---|
| crops where a **flat mask scores BETTER** than the model's | **13 of 25** |
| median gap (flat − model; positive = model better) | **−0.0023 — NEGATIVE** |
| mean gap | +0.00065 |
| range | −0.064 to +0.070 |

**On the typical crop the flat mask wins.** The mean is positive only because a
handful of crops swing hard in the model's favour. **A term whose sign flips on
half its data is not a training signal, it is a coin flip**, and it was bought
at 15 % of the gradient.

### The mechanism, and it predicts the direction the mask actually moved

`_loss_mask_shape` is an **L1** between the model's mean-removed frequency shape
and the oracle's. An L1 gradient has **constant magnitude regardless of how wrong
the prediction is** — only its sign carries information. So when the model cannot
predict the sign of the oracle's deviation in a cell, it receives a fixed-size
push in an effectively random direction, and the L1-optimal response under that
uncertainty is the **conditional median of the oracle deviation, which is ~0**.
Not deviating at all is the cheapest way to stop being punished for deviating
wrongly.

**Measured, and this is the confirmation:** the arm's mask variation across
frequency **halved** (`d_freq` 0.0253 → 0.0121) and freq/time fell 0.616 → 0.306
against an ideal of 1.014 (`2026-09-15-mask-flatness-struct`). The term was
built to raise that number and it lowered it.

### A CORRECTION to how 2026-09-13's number was read

`decisions-m3.md` 2026-09-13 records `fraction_of_the_way_to_flat` = 0.9961 as
**motivation** — the mask is 99.6 % of the way to a volume knob, so there is
headroom. **The same number, read the other way, says the term has 0.4 % of
usable range at the operating point.** Both readings are arithmetically correct
and the second one is the one that predicts the outcome. The derivation was
sound; what was missing was checking the anchor **spread** before spending a
training run on the mean.

### What this authorises

- **D17 is explained, not merely null.** The judge could not separate the arm
  from the baseline (55.36 against 55.59, inside the ~3-point paired floor), and
  this says why: the term could not separate them either.
- **The structure HYPOTHESIS is untested, not refuted.** It was never given a
  loss capable of testing it. Any successor must be **scale-free across
  frequency** — correlation or cosine, where a flat prediction scores *worst*
  rather than indifferently — or must match the deviation's variance. An L1 or
  L2 to an unpredictable target has a flat mask at its own conditional optimum
  and will keep producing this result.
- **A cheap pre-flight is now mandatory for any future auxiliary term:** score it
  on a degenerate output and on the model's, PER CROP, and refuse to run the arm
  if the sign flips on a material fraction. `derive_w_struct.py` already computes
  this; nothing new is needed but reading it.

### What it does NOT authorise

**This says nothing about whether frequency structure helps content fidelity.**
It says one particular loss could not deliver it. Flatness remains a property of
the mask, not of what a listener transcribes, and only LCF-WER scores an
intervention (`decisions-pending.md` 2026-09-13).

**And `train_L_struct` is absent from `2026-09-14-train-sir0-struct/history.csv`**
— the logging fix landed alongside the run — so whether the term descended during
training is still unmeasured. The anchors are measured at the checkpoint, not
along the trajectory.

---

## 2026-09-21 — Step up the sizing ladder to the wesep reference, and use the second T4

Two changes, one config: `experiments/configs/bsrnn_wesep_ref.yaml`.
`bsrnn_baseline.yaml` is untouched, so every existing run reproduces and every
checkpoint on disk still resumes.

### The sizing: 7.19 M -> 14.73 M, and it is not an arbitrary number

`decisions-m1.md` 2026-08-19 recorded exactly two deviations from the wesep
reference, both deliberately toward smaller, and closed with "If it underfits,
step up this ladder and record which rung and why."

**It underfits.** WeSep captures 51.6 % of the offline-ASR word-error headroom
against our 10.4 % (`project-state.md`). This entry is that record.

| | baseline | this config | source of the value |
|---|---|---|---|
| `lstm_hidden` | 192 | **256** | wesep reference, `feature_dim * 2` |
| `n_hidden` | 1 | **2** | wesep reference; the paper gives width, not depth |
| separator | 4.90 M | 7.71 M | |
| estimator | 2.19 M | 6.92 M | |
| **total** | **7.19 M** | **14.73 M** | 2.05x |

Nothing else moves. `mlp_hidden` stays at 384 because that IS the paper's stated
width (Yu et al., Interspeech 2023 §4.2) and is not a deviation.

### Why the capacity is split, and not all put in the estimator

Memory on the T4 is essentially all activations -- 7.19 M params is ~29 MB and
AdamW state ~57 MB against ~13,000 MB measured (`decisions-pending.md` E8). So
activation cost tracks LSTM width x depth, not parameter count, and the routes to
~2x parameters cost very different amounts of memory:

| route | params | activations | batch that fits |
|---|---|---|---|
| `mlp_hidden` 512 / `n_hidden` 2 | 16.3 M | ~1.00x | 3, unchanged |
| **`lstm_hidden` 256 / `n_hidden` 2** | **14.7 M** | **~1.33x** | **3 per card** |
| `feature_dim` 192 / `lstm_hidden` 320 | 18.0 M | ~2.50x | 1-2 |

The all-estimator route is cheapest in memory and `decisions-m1.md` 2026-08-19
says why it is also the least useful: "Capacity added there buys per-band readout
richness, not temporal or cross-band modelling", and at 384x2 the estimator is
already 58 % of the model, larger than the six-layer separator. The chosen rung
spends on both paths and stays inside the ceiling.

**The activation multipliers are PROJECTIONS, from the measured linear law
(0.12 GB fixed + 2.15 GB per trial, E3b-E3f) scaled by LSTM width.** They are not
measured. `scripts/profile_step.py --amp-only` measures them and E1's analytic
model was already found 2.4-3x low, so profile before committing a session. Part
of the memory -- `L_MR`'s eight retained STFTs -- does not scale with width at
all, so the LSTM-width row is probably pessimistic.

**`n_hidden` was unreachable from the yaml until today.** `build_model()` left it
at the ctor default of 1, which `decisions-m1.md` 2026-08-19 flagged ("Both belong
in the yaml") and nothing acted on. It is now passed, defaulting to 1 when the key
is absent, so no existing config or checkpoint changes meaning.

### The second T4 (E7), which has been idle on every run to date

`nn.DataParallel`, gated on `training.data_parallel` and `device_count() > 1`,
default off. Kaggle's "GPU T4 x2" gives two cards; `torch.device("cuda")` is
`cuda:0` and nothing asked for the other.

**It splits the batch, not the model.** Both cards hold a full replica, so this
buys batch headroom and throughput, NOT room for a wider model -- the measured
per-card ceiling still applies. E7's "two cards is also 2x the memory" is true of
batch and must not be read as licence to widen.

**The loss stays outside the model, and that is required.** `LossBSRNN` means over
subsets (`n_present`, `n_absent`); a per-device reduction would silently reweight
them. `DataParallel` gathers to `cuda:0` before the loss, so this is preserved.
Direction pairing survives for the same reason: `collate_pairs` keeps both
directions in one batch, splitting separates some pairs in the forward, but the
model is per-example independent and the contrast lives in the gathered loss.

**The trap, handled:** `DataParallel` prefixes every `state_dict()` key with
`module.`. All three `torch.save` sites, the resume `load_state_dict`,
`model.stft` in `oracle_mask_and_mag` and `model.band_widths` in `log_results`
now go through `unwrap()`. Checkpoints are written unwrapped, so files stay
interchangeable between one-card and two-card runs.

### `batch_size` 3 -> 6, and what it is NOT for

Six, split 3 per card, so per-card memory is exactly what every run to date used.

**It does not make one card faster.** Per-trial throughput is flat across batch
3/5/6 (0.990 / 0.994 / 1.005 s/trial, E3b-E3f) -- the T4 is saturated at batch 3,
which is why gradient checkpointing was withdrawn. The 7.66x came from
tensor-core alignment, not batch size. The gain here is the second card.

**`bsrnn_baseline.yaml`'s comment promising batch 12 is wrong by 4x** and is not
corrected in place, because that file must keep reproducing past runs. 6 is the
fp16 ceiling on one T4 (13.02 GB of 14.56); 7 needs 15.17 GB; fp32 caps at 3.

### The `w` schedule had to move with the batch, or the run is silently wrong

The absent-branch warmup is indexed in optimiser STEPS, which makes it invariant
to dataset size (2026-09-03) but **not** to batch size: at batch 6 each step
consumes twice the audio. The warmup exists to stop the early mute and its length
in EXAMPLES is what matters, so the invariant held is `warmup_steps x batch`:

    warmup  6632 x 3 = 19,896  ->  3316 x 6 = 19,896
    ramp    4974 x 3 = 14,922  ->  2487 x 6 = 14,922

Left alone, the warmup would have covered twice the audio it was calibrated for.

### The Kaggle probe had to change too, and one of its bugs was NOT the known one

Two problems, both of which would have quietly wasted the second card or
corrupted the schedule. `scripts/make_kaggle_notebook.py`.

**1. The probe was not DataParallel-aware.** `_probe_batch.py` runs one process
on `cuda:0`, so what it measures is what ONE card must hold -- but the config's
`batch_size` is the GLOBAL batch, which DataParallel splits. Probing the global
batch on one card caps the run at the single-card ceiling and leaves the second
T4 half idle, which is the exact waste E7 exists to remove. The probe is now
handed `B // n_gpu` and candidates are filtered to multiples of `n_gpu`.

**2. A probe-chosen batch did not drag the `w` schedule with it, and nothing
said so.** The probe steps the batch down until one fits and writes the winner
into the config that trains. The absent-branch warmup is indexed in optimiser
STEPS, so a batch the probe lowered from the configured value silently made the
warmup cover MORE examples than it was calibrated for -- the same class of
confound the step-indexing of 2026-09-03 was introduced to remove, arriving
through a different door. The notebook now rescales `warmup_steps` and
`ramp_steps` to hold `steps x batch` constant whenever `chosen != CFG_BATCH`,
and prints the rescale. Rounded UP: a warmup one step short is harmless, one
step long is not.

**This second one is not in E8** and was found while wiring E7. E8 says only
"do not let the probe pick a new batch size during the 9,955-trial run" and
prescribes pinning the knob by hand -- a workaround that depends on remembering
it. The rescale makes the coupling automatic.

### How this must be read when it lands

**Speed claim only for the DataParallel half.** Numerics are not bit-identical
(different kernel split, different fp16 reduction order), so a 2-epoch A/B shows
val terms agreeing within known between-run noise, never exactly, and a difference
must never be reported as a quality change.

**Two variables move at once** -- capacity and the device count. They are
separable by their signatures (throughput vs. held-out separation) but the run is
not a clean single-variable arm and the write-up says so.

**Read it on `sir0_privval` word error, not on dB.** 2026-09-04 measured 2x the
data buying +0.316 dB of separation while LCF-WER moved the WRONG way
(59.05 -> 59.52 %) and headroom captured fell 10.4 -> 9.6 %. A capacity gain that
shows up only in dB is not yet a result.

**Expect overfitting to arrive earlier, not later.** The diagnosis at every data
scale is data-limited (train falls monotonically, held-out peaks then collapses).
Adding capacity to a data-limited model moves the peak earlier and deepens the
collapse. If the peak lands before epoch 6 that is the predicted signature, not a
bug. `select_on: present_branch` and the `select_abs_max` silence bar stay on.

### Not done

- **Not profiled.** `profile_step.py --amp-only` on this config, and its ALIGNED
  verdict, before any session. `examples x T` must stay divisible by 8 or the
  4.09x fallback kernel fires and the bigger model looks slow for the wrong reason.
- **E8's "the notebook probe measures fp32" is STALE and is corrected here.**
  It was fixed on 2026-09-04: `_probe_batch.py` wraps its forward in
  `amp_ctx(use_amp)` and the notebook refuses a probe whose reported precision
  disagrees with `training.amp`. The E8 bullet should be marked closed.
- **`DistributedDataParallel` deferred**, as E7 records: better tool, needs
  `spawn` inside a Kaggle notebook, not worth it for two cards on one host.

---

## 2026-09-21 — The capacity arm sizes on Kaggle: batch 10 across two T4s, and the memory law predicts both ends

**MEASURED**, notebook probe, `bsrnn_wesep_ref.yaml` (14,731,404 params), T4 x2, fp16:

    DataParallel: 2 cards, probing the PER-CARD share (global batch = per-card x 2)
      batch 12: OOM  (6/card)
      batch 10: FITS   peak allocated 12.79 GiB, reserved 14.31 GiB, probed in amp  (5/card x 2)

### Three things this confirms at once

**E7 works.** Batch 10 cannot fit on one T4 under any config here -- the 7.19 M
baseline's single-card fp16 ceiling was 6 at 13.02 GB, and 10 would need
~21.6 GB. So the probe accepting 10 is direct evidence both cards carry load.
Half the allocated hardware is no longer idle.

**The arm that ran is the 14.73 M one, checkable from the number alone.** At
5 trials/card the 7.19 M model would allocate 0.12 + 5 x 2.15 = 10.87 GB. The
probe read 12.79. So this is not the baseline wearing a new config name.

**The linear memory law now predicts a fit AND an OOM.** Per-trial is
(12.79 - 0.12) / 5 = **2.534 GB**, so 6/card needs 0.12 + 6 x 2.534 = 15.32 GB
against ~14.56 usable -- batch 12 cannot fit, and did not. Two independent
points from one line.

### A CORRECTION to this file's own projection, in the safe direction

The 2026-09-21 entry above projected ~1.33x the baseline's activation cost from
scaling by LSTM width. Measured, it is **1.18x** (2.534 against 2.15). The
projection was 13 % pessimistic **for exactly the reason that entry gave**:
`L_MR`'s eight retained STFTs are a large share of memory and do not scale with
LSTM width at all. The caveat was right; the number was not. Quote 1.18x.

### What the probe does NOT measure, and it matters at this margin

`_probe_batch.py` is a single process with **no DataParallel wrapper**. It
measures one card's share of the forward and the loss over *that share only*.
The real run gathers to `cuda:0` and computes `LossBSRNN` over the **full**
batch there -- which is required, not incidental, since the loss means over
subsets and a per-device reduction would reweight them. So `cuda:0`'s true peak
is higher than 12.79 GiB by the gather plus the full-batch loss.

Estimated at a few hundred MB against 1.77 GB of headroom, so it should hold --
but `reserved` already reads 14.31 GiB, which is 98 % of the card. **This is the
tightest configuration this project has run.** The 2-epoch run is the test, and
it is a real test: the probe already exercises fwd+bwd+step, so training is the
binding case, not validation (which runs under `no_grad` and has no backward).

**The cost of being wrong is bounded**: `_last.pt` is written every epoch with
`global_step`, so an OOM loses at most one epoch and `--resume` recovers the
schedule position. Dropping to batch 8 (4/card, 10.26 GB) costs almost nothing
in throughput -- per-trial time is flat across this range (0.990/0.994/1.005
s/trial at batch 3/5/6, E3b-E3f) -- so it is the cheap fallback, not a
concession.

### The `w` schedule rescale fired, and landed where it always has

    w_schedule.warmup_steps 3316 -> 1990 (holding steps x batch = 19,896 examples)
    w_schedule.ramp_steps   2487 -> 1493 (holding steps x batch = 14,922 examples)

Both invariants match the original `6632 x 3` and `4974 x 3` exactly. At 9,955
trials and batch 10 that is 995 steps/epoch, so warmup is **2.0 epochs** and the
ramp **1.5**, reaching full `w` at epoch 3.5 -- the same position it occupied at
batch 3 and at batch 6. The absent branch is fully engaged well before the
expected peak. Without the rescale the warmup would have covered 3.3x the audio
it was calibrated for, silently.

### The notebook's stock NOTE is fine here, and should be read down

It warns that batch 10 is "below the requested 12" so `L_abs` is a noisier
estimate. True, but the comparison that matters is against what has actually
run: every previous run was batch 3 or 6. **Batch 10 is the closest this project
has ever been to the batch 12 `w = 0.458` was calibrated against**, so the
absent-branch estimate is better than any run to date, not worse.

### Learning rate deliberately NOT changed

`lr: 0.0005` was set at batch 3 and the effective batch is now 3.3x that.
Textbook scaling says raise it; this run does not, because (a) it would move two
variables at once and `decisions-m1.md` 2026-08-18 records effective batch as a
training-dynamics parameter whose silent change makes curves incomparable,
(b) `ReduceLROnPlateau` (factor 0.5, patience 3) adapts downward anyway, and
(c) the batch-6 structure run held the same lr and peaked at -4.201 held-out
against the batch-3 baseline's -2.941. If this run underfits, raising lr is the
first follow-up **as its own arm**.
