# Open decisions

**Written 2026-08-10.** Groups A–C were the pre-generation data decisions; all are
closed — C2 last, on 2026-08-30. Full reasoning for each lives in `decisions-m0.md` under its
date. Group D holds open *modelling* questions, Group E the training-speed
work, and Group J the *judge and metric* questions; decisions
actually taken go to the decision log of the milestone they belong to —
`decisions-m1.md` (architecture), `decisions-m2.md` (training), `decisions-m3.md`
(conventional evaluation), `decisions-m4.md` (the metric and the judge).

---

## Still open

- **J2 — which judge, and the open-weight anchor.** The cost half is answered
  (~$5-25); what remains is reproducibility and the candidate gate. **No longer
  blocked — J1 closed 2026-08-31.** See Group J.
- **J3 — CLOSED 2026-09-03.** Threshold `count>=2`, signed off into
  `metric-definitions.md` §3.2. The sweep showed the choice of k changes no
  conclusion (floor 57.0 -> 42.9 across k=1/2/3/5, ceiling 0.0 throughout), so the
  sensitivity requirement is discharged by that rather than by defending 2 over 3.
  decisions-m4.md 2026-09-03.
- **D11 — inference-time mix-back.** RUN 2026-09-01 as the screening test.
  Globally alpha=1 wins, but the per-difficulty optimum spans 0 to 1, so
  adaptation is motivated and a global gentleness shift is not. See Group D and
  `decisions-m3.md` 2026-09-01.
- **D12 — mixture of experts over masking behaviours.** Viable at +11 % params
  with a shared trunk; `K > 2` deferred until the judge work is done. See Group D.
- **D13 — DECIDED 2026-09-01: build the per-band gate**, as the `n_experts = 2`
  case of D12 with the identity as expert 0. See Group D.
- **D14 — PROPOSAL: per-frame speaker-state supervision.** Free four-state labels
  from the rendered stems; an auxiliary head, a frozen state detector used as a
  training-only loss (zero inference cost, no architecture change), and the state
  posterior as the supervised input to D13's gate controller. **Gated on D10's
  free ICR measurement**, and it corrects a recorded ceiling: M6's 2.2-point
  oracle was measured per trial, not per frame. See Group D.
- **J4 — PROPOSAL: a metric *system* (normalised requirement axes, composed and
  plotted) rather than metrics in isolation.** Diagnosis accepted; ranking by
  polygon area rejected as order-dependent, and the baseline normalisation must
  be floor-to-ceiling rather than percentage change. See Group J.
- **E7 — PROPOSAL: use the second T4.** Kaggle's "GPU T4 x2" gives two cards and
  the code uses one; the other has been idle for every run to date. **~2x** for
  about half a day's work, the trap being `DataParallel`'s `module.` state-dict
  prefix. **Conditional on E8's `batch_size` 3 -> 6** — at batch 3 the split
  halves an already-saturated card and disappoints. Sequenced *after* the
  9,955-trial run so the data-scaling curve does not move two variables at once.
  See Group E.
- **E8 — PROPOSAL: raising `batch_size`, and a BUG capping it at 3.** The
  notebook's batch probe runs fp32 while training runs fp16, so it finds the
  fp32 ceiling (batch 3) and writes it into an AMP run that fits 6. A repeat of
  the fp32-warmup bug already fixed in `profile_step.py`. Fixing the probe is
  the cheap half of E8 and is what unblocks E7. Memory is essentially all
  activations, which rules out 8-bit optimisers (~43 MB of ~13 GB) and audio
  compression (cannot touch GPU memory at all). Beyond 6 needs `hop` 128->256.
  See Group E.
- **D18 — PROPOSAL: why `L_struct` FLATTENED the mask.** The shape term is an L1
  to a target the model cannot predict, whose optimum is a flat mask. Free test,
  no training. See the 2026-09-15 menu.
- **D19 — PROPOSAL: run the flatness instrument on WeSep.** Is the volume knob
  ours or the task's? No training run. The single cheapest redirect available.
  See the 2026-09-15 menu.
- **J5 — PROPOSAL: run-to-run training variance.** Ten training runs, all
  `seed: 42`, no replicate anywhere. The third noise source is unmeasured and M6
  needs it. See the 2026-09-15 menu.
- **G1 — PROPOSAL: the Gemini tuner.** A local stand-in for the judge, trained on
  Gemini labels, used as the differentiable signal the API cannot give. Measured
  starting point: a free local listener already explains 68 % of the judge's
  per-trial error (r^2 0.680) but is 21.7 points out on the level. See the
  2026-09-15 G1 section at the end of this file.
- **O4 — OBLIGATION, raised 2026-09-21: we ARE the REAL-TSE Challenge baseline
  (`BSRNN_TFMAP_CAUSAL`), and our three negative M5 results independently
  reproduce that challenge's published consensus.** Must be reported as a
  reproduction, not as project-specific failure. Includes the organisers'
  mid-challenge metric-gaming incident, which is first-hand evidence for this
  project's metric contribution. Four confounds in the WeSep comparison are
  listed there and must travel with it. See the 2026-09-21 O4 section.
- **D5 — REOPENED 2026-09-21: the discriminative speaker encoder.** Demoted
  2026-08-30 on `rel_movement`, a magnitude-without-direction diagnostic that
  cannot distinguish "moved toward the interferer" from "changed level". The
  directional test was never run. D5 is the largest conditioning difference
  between us and the reference model. See `ranked-next-steps.md` items 1-2.
- **O1/O2/O3 — obligations, not options.** Score the struct control on
  `sir0_privval` (1.2 h, its registered acceptance test is unrun), write D17 up,
  and produce M6's stratified tables. See the 2026-09-15 menu.

- **A1 needs sign-off only, not a decision.** Reference is the full reverberant
  target: separate and denoise, do not dereverberate. Removing a 0.6 s tail inside
  a 300 ms causal window is not possible, and trying trades residue for artefacts,
  which hurt recognition more. Dereverberation kept as an ablation if time allows.

## Closed (A, B) — see `decisions-m0.md` for the reasoning

| id | decision | date |
|---|---|---|
| A1 | full reverberant reference, "what the mic heard" | 08-13 |
| A2 | noise bed wraps around | 08-11 |
| A3 | BS.1770 integrated loudness | 08-12 |
| A4 | no room on the enrollment | 08-12 |
| A5 | pad the tail by `t60_s` | 08-13 |
| A6 | common-gain rescale at 0.95 | 08-13 |
| B1 | `overlap_ratio` is a difficulty-dial setting, not a standalone decision. Narrow it **last**: its 0.7 ceiling is matched to REAL-TSE | 08-13 |
| B2 | measure overlap from detected speech (Silero VAD 6.2.1, pinned) | 08-13 |
| B3 | enrollment fixed 5 s, kept configurable | 08-12 |
| B4 | eval carries the same absent fraction as train, scored on its own row | 08-13 |
| B5 | Whisper `EnglishTextNormalizer` | 08-13 |
| B6 | 500 eval trials generated, 200 the minimum scored | 08-13 |
| B7 | per-epoch resampling off for the main run, kept as a switch | 08-13 |
| B8 | enrollment from a different book | 08-11 |
| B9 | 50 % both / 25 % absent / 25 % target-only; variable `target_activity_ratio` | 08-13 |
| C2 | task difficulty accepted as measured: floor **57.4 %** on `eval_public` `both` (n=230), **65.2 %** on `sir0_val` (n=103); ceiling ~6 %. Straddles the 60–80 % target band, which was an aim, not a constraint. **Open consequence: which set defines the benchmark** — the two differ by 7.8 points because `eval_public` keeps the target-louder distribution and `sir0` is symmetric | 08-30 |
| B10 | three enrollment tiers recorded per trial; eval pools redrawn. Executes B8's own documented contingency (60.2 % of speakers dropped out), not a reversal | 08-13 |
| B11 | report a latency decay curve, never cap T60. Largely defused by A1 | 08-13 |
| B12 | two regimes, sampler layer, no relational constraints. PR1/PR2 landed 08-14 | 08-13 |
| B13 | stratified reporting per condition, no combinations, 100 trials per bucket | 08-13 |

Two B12 items remain implementation, not decisions: `overlap_ratio` narrowing in
`base` (needs supervisor agreement, one config line) and `length_mode`.

---

## D. Modelling — open, not blocking M1

*Added 2026-08-19. This file was written for M0 data decisions; group D extends it
to open modelling questions, which have nowhere else to live. Decisions actually
taken go to the milestone log they belong to, usually `decisions-m2.md`.*

### D1. Phoneme-template speaker cue as an alternative to TF-Map

**Status: idea, unscheduled. M5-scale (M5 is already marked CUTTABLE). Do not
start before the cheap precursor in D2 has been run.**

**The problem it addresses.** TF-Map compares magnitude spectra, and magnitude
spectra are dominated by *what is being said* rather than *who is saying it*. An
interferer saying "ah" resembles the enrollment's "ah". The cue is therefore
partly phonetic rather than speaker-discriminative. See `decisions-m1.md`
2026-08-19.

**The proposal.** Replace the raw enrollment frames (TF-Map's basis vectors) with
a phonetically organised, speaker-adapted dictionary:

1. define a library of the language's phonetic units;
2. from the enrollment, build a spectral template for each unit this speaker
   actually produced;
3. for units absent from the enrollment, *predict* the speaker's realisation from
   the ones present, using similarity between sounds;
4. match mixture frames against this dictionary instead of against raw frames.

**Refinement that makes it work (agreed 2026-08-19).** The proposal as stated does
*not* remove the confound — the interferer's "ah" still matches the target's "ah"
template. It needs a contrastive term:

- `S_target` — similarity to *this speaker's* template for a unit
- `S_background` — similarity to a *speaker-independent average* template for the
  same unit, built across many speakers
- feed the network `S_target - S_background`

Phonetic content cancels; what survives is "how much more target-like than
average-speaker-like is this frame". **This is the GMM-UBM likelihood-ratio
structure from speaker verification**, applied per-frame per-band as a streaming
extractor feature. That combination appears to be novel and is the part worth
claiming.

**Related work: searched 2026-08-19, written up in full at
`literature/novelty-review-contrastive-phonetic-cue.md`.** Verdict: novel as a
combination; every ingredient has prior art. Closest existing work is
arXiv:2502.16611 (NeurIPS 2025), which contrasts positive against negative
enrollments -- but its negative reference is actual interfering speakers from the
same recording rather than a speaker-independent background, it compares at the
embedding level rather than densely, and its TF-GridNet/BiLSTM backbone is
non-causal. Step 3 has direct prior art in Weiss & Ellis (2010) eigenvoice
speaker adaptation for separation, and steps 1-2 in phone-dependent NMF; both must
be cited prominently. The SLT 2026 REAL-TSE overview surveys all 24 submissions
from 12 teams and lists no contrastive and no phonetic conditioning.

**Known obstacles.**

- *Phonetic labels.* Step 2 needs forced alignment of the enrollment. We are
  unusually well placed — `meta.json` carries `target_text`, exact ground-truth
  transcripts — so it is feasible offline on our data. A deployed system would not
  have the enrollment transcript, which weakens any generality claim and must be
  stated.
- *Enrollment sparsity.* English has ~44 phonemes; 5 s of speech contains maybe
  15-20 tokens and perhaps a dozen distinct units. **Most of the dictionary would
  be predicted rather than observed, so step 3 is doing the work, not the
  enrollment.** Enrollment length is a knob we control
  (`enrollment_length_s`, currently fixed at 5 s per `decisions-m0.md`
  2026-08-12) and lengthening it is the obvious first mitigation — it is a config
  change and a regeneration, not new code.
- *Step 3 is a speaker encoder in disguise.* Predicting unseen units properly
  requires a model of speaker space learned across many speakers. Note however
  that **the latency objection does not apply here**: all of this is enrollment
  side and therefore offline. Matching against ~44 templates is *cheaper* at
  runtime than against 628 enrollment frames. So this is a legitimate route to
  encoder-quality conditioning without the streaming cost that ruled out
  Zhang et al.'s eq. 3.

**Open concern (raised 2026-08-19): model size.** Whether this is affordable
with or without an encoder is unresolved, and is the main risk to the idea. It
needs a parameter budget before any implementation. Current model is 7.19 M
against challenge baselines at 25-27 M, so there is headroom, but a speaker-space
model for step 3 could consume all of it.

**Evaluation.** Compare against TF-Map on extraction quality *and* on efficiency
(parameters, RTF, added latency), not quality alone — the whole argument for
TF-Map over the embedding variant was efficiency, so a replacement must be judged
on the same axis.

### D2. Attention temperature in TF-Map — the cheap precursor to D1

**Status: unrun, one line of code. Run this before scheduling D1.**

Measured 2026-08-19: TF-Map's softmax weights are almost uniform — it blends ~621
of 628 enrollment frames, max weight 0.00239 against a uniform 0.00159. This is
**forced by the arithmetic, not a property of the audio**: magnitude spectra are
non-negative, so cosine similarities lie in [0, 1], and a softmax over a range of
1 can produce weight ratios of at most e ~ 2.7, which is nearly uniform across 628
items. The wesep reference has no temperature either, so this is the published
behaviour.

Consequence: TF-Map's time-variation comes almost entirely from the energy
recovery step, not from the attention. What it actually supplies is "the target's
average spectral shape, scaled per frame by how much energy the mixture has in
that direction" — a useful signal, but not the frame-selective mechanism the NMF
framing describes.

**The experiment.** `h = softmax(sim / tau)` with `tau` in {1.0 (current), 0.2,
0.05}. D1 and the temperature share one hypothesis: *more selective matching
against enrollment content improves extraction.* The temperature tests that
hypothesis in an afternoon. **If sharpening the existing mechanism does not help,
a richer dictionary is unlikely to, and D1 should not be scheduled.**

Also worth confirming the near-uniformity across several trials before it is
written up — measured on one so far, though the argument above says it is
structural.

### D3. RUN 2026-08-30. ANSWERED: conditioning is not the bottleneck

**Measured over 200 `sir0_val` crops: the cue moves 28.6 % on an enrollment swap
and the output moves 48.2 %.** The network amplifies the cue, it does not discard
it — so the conditioning path is not what fails. Look at the separator or the
objective instead. **D4a and D1 dropped, D5 demoted, D2 closed** (the softmax now
blends 138 of 628 frames against ~620 before `tfmap_scale`). D3c ran alongside:
cross-gender sensitivity 56.1 % vs same-gender 44.4 %, a weak signal of gender
reliance. `scripts/diagnose_cue.py`, `decisions-m2.md` 2026-08-30.

`sir0_train` is balanced 680/680 same/cross-gender, which **caps but does not
remove** the gender shortcut: a model using pitch alone gets the cross-gender half
right and coin-flips the rest, i.e. ~75 % correct with no enrollment at all. That
is invisible in pooled numbers, so gender-stratified reporting stays on.

**D3b is still available and unrun** — replace the TF-Map channel with the clean
target's magnitude spectrogram and train briefly, to get conditioning's ceiling
as an upper-bound experiment.

### D4. Inject the speaker cue at every block, not once at the input

**Status: proposal, not scheduled. Cheapest large change available. Gated on D3a.**

**The structural problem.** `BSRNN_TFMAP.forward` concatenates the TF-Map as a
third input channel, `SubbandNorm` projects it once through a 1x1 conv, and from
there it must survive **six `BSNet` blocks — twelve LSTMs — of residual mixing**
to reach the mask head. Nothing re-injects it. The extraction loss has to
propagate identity backwards through that entire stack before the cue earns its
place, which is a long credit path for a signal worth one third of one
projection.

**The proposal.** Derive a fixed-length embedding from the enrollment and apply
FiLM at each block: `z <- gamma(e) * z + beta(e)`. Cite Perez et al., AAAI 2018
for FiLM; in TSE the multiplicative-adaptation precedent is Delcroix et al.,
"Improving speaker discrimination of target speech extraction with
time-domain SpeakerBeam", ICASSP 2020. Roughly 400 k parameters from a 256-d
embedding across six blocks — affordable against the 7.19 M / 25-27 M headroom.

**Why this is ranked first.** It is the single largest divergence from TSE
systems that demonstrably condition, and it does not require the cue itself to
change — so it composes with whatever D3a says.

**Two variants, and the cheap one needs no encoder at all. Run D4a before D4b.**

- **D4a — re-inject the TF-Map itself at every block.** The TF-Map is already
  `(B, 1, F, Tx)`: frequency-shaped and time-aligned with the feature map, so it
  band-splits exactly like the mixture does. Project it per band to
  `feature_dim` and add it into each of the six `BSNet` blocks. **The cue stays
  parameter-free, so it cannot memorise anything** — only the projections are
  learned. This isolates the dilution hypothesis with no new concepts and no new
  failure modes, and it is the honest test of "the cue is fine, the network is
  losing it".
- **D4b — FiLM from a learned fixed-length embedding.** Strictly stronger and
  strictly riskier; requires D5's encoder. Only worth it if D4a moves
  `val_enrol_sens_db` and then stalls.

**On the memorisation worry (raised 2026-08-28).** Adding speaker parameters
risks the encoder learning the 1172 training voices rather than learning to
extract. Three things already defuse it, and they should be stated in the
write-up rather than discovered later:

1. **The splits are speaker-disjoint by construction** (`speakers_from:` in
   `generator.yaml`), so val speakers are never trained on. Memorisation shows
   up directly as train extraction improving while val does not — *the
   experiment already detects the failure mode.*
2. **It is the standard setup in speaker verification.** x-vector and ECAPA-TDNN
   are trained with exactly this closed-set classification loss and transfer to
   unseen speakers; that transfer *is* speaker verification. The classification
   head is discarded at test time.
3. **Our data design already breaks the likelier confound.** The real risk is
   the encoder latching onto channel rather than voice — LibriSpeech speakers
   each have their own sessions. But the enrollment is dry (A4, no room) and
   from a different book (B8) while the mixture is reverberant, so channel
   matching is actively unavailable and voice is what is left.

**The asymmetry that settles the ordering.** A model leaning too hard on the
speaker cue can be regularised; a model that ignores the cue cannot be
regularised into using it. The current failure is the second kind, so the risk
is worth taking — behind D4a, which carries none of it.

### BUILT 2026-09-11 — D4a exists as code and as a runnable config. It REVERSES a recorded drop

`conditioning.TFMapInjector`, `BandSequenceModel.forward(x, cue, gates)`, the
`tfmap_inject` flag on `BSRNN_TFMAP` and in `build_model`,
`experiments/configs/bsrnn_tfmap_inject.yaml`, `tests/test_tfmap_inject.py`
(20 tests, all passing; suite 467).

**Stated plainly: D4a was DROPPED on 2026-08-30 and this un-drops it.** The drop
was correct on its own terms — `diagnose_cue.py` measured the cue surviving the
stack (swap a stranger's enrolment, the cue moves 28.6 % and the output moves
48.2 %), so the *dilution* premise is not supported. What survives that
measurement is the headroom: 62 % of the output is still enrolment-independent.
**This arm must therefore be written up as testing the remaining headroom, never
as testing dilution**, and the honest prior is that it does little.

**One shared per-band projection, not six.** 37,698 parameters, +0.524 % on
7,189,644 — verified by building both configs, not by arithmetic in a document.
Six separate projections would be 222 k (+3.1 %) and would add expressiveness
that could explain a gain by itself. Shared re-presents the identical cue at
every depth, which is the honest form of the claim.

**NOT parameter-matched to its control**, unlike D14's head A. +0.52 % is small
but not zero, and a gain of that order is not attributable.

**Gates start at zero, so the arm begins as exactly the baseline function** — a
test asserts the two models agree to 1e-6 when built at the same seed. Cite
ReZero (Bachlechner et al., UAI 2021). The ordering this implies is real and
tested: at gate 0 the projection receives no gradient and the gates, sitting one
multiply from the loss, move first.

**THE GATES ARE THE MEASUREMENT, and this is the reason to run it even expecting
nothing.** 6 x 32 scalars saying how much cue each block wants in each band. If
they stay near zero, D3a's conclusion is confirmed by a second, completely
different method, and D4b/D5 can be closed rather than left hanging. That is a
publishable negative result for the cost of one run. Log them every epoch.

**Cost: +2.7 % forward on CPU** (1.210 -> 1.243 s per 4.008 s chunk, 4 threads,
forward only, 3 reps). Not a training-step measurement and must not be quoted as
one. Latency is unchanged in kind: the TF-Map is already causal per frame and
every added op is a 1x1 convolution over time, asserted by a causality test.

**RNG forked at construction**, as with head A, so the arm and its control draw
the same trials in the same order. Narrow fix, not the general one that was
declined.

### D5. A speaker encoder with an auxiliary speaker-ID loss

**Status: proposal, not scheduled. Larger than D4 and subsumes part of D1.**

**The problem.** **No parameter in the model is devoted to speaker identity and
no loss term rewards it.** TF-Map is deliberately parameter-free, and the M2
objective is four terms about signal level and content (`L_pres`, `L_MR`,
`L_gain`, `L_abs`). Identity is something we hope extraction discovers, never
something we train for. SpEx+ (Ge et al., Interspeech 2020) is the standing
evidence that the auxiliary speaker-classification loss is what makes
conditioning stick rather than an optional extra.

**The proposal.** A small encoder over the enrollment producing a 256-d
embedding, trained jointly with a cross-entropy over the ~1172 training
speakers, feeding D4's FiLM. **The enrollment is fully available before the
stream starts, so this encoder may be non-causal and costs no streaming
latency** — the objection that ruled out Zhang et al.'s eq. (3) does not apply
to the enrollment side.

**It completes the paper we already cite.** Zhang et al., ICASSP 2025 is
*Multi-Level* Speaker Representation: we implemented the spectral level (eq. 2)
and skipped the embedding level (eq. 3) for want of an encoder. Adding one makes
eq. (3) available and the write-up becomes "we implemented the spectral level,
measured it insufficient, and added the embedding level the paper specifies".

**Constraints that are not negotiable.** The encoder must be a different model
family from the ASR proxy, and the judge must never appear in it in any form
(CLAUDE.md). If a pretrained speaker model is used rather than training from
scratch, the family check must be recorded, not assumed.

### D6. Two levers that are already built and currently switched off

**Status: both are config changes, hours not days. Run alongside D3.**

- **`lookahead_frames` is 0.** `lookahead_shift()` is implemented and tested; the
  spec allows 200-300 ms and 16 frames is 128 ms. Ablate {0, 8, 16}. Free
  performance we are declining to take.
- **The residual branch `R` is unbounded and unconditioned.** In `Estimator`,
  GLU bounds the mask but `res_heads` is a raw `Conv1d` added straight to the
  masked spectrogram, so it can synthesise output **ignoring both the mixture
  and the enrollment**. `residual_branch` is already a constructor flag; give it
  an ablation arm like `ablate_w_m` / `ablate_w_g`.

### D7. Status correction — D2 was run without being logged as an arm

`tfmap_scale: 16.0` in `bsrnn_baseline.yaml` is D2's temperature (scale = 1/tau,
so tau ~ 0.0625, sharper than D2's sharpest proposed arm). It worked mechanically:
619.6 of 628 frames effectively used before, top 50 frames carrying ~59 % after.

**But the gain cannot be attributed to it.** Enrollment sensitivity went 2.6 % →
15 % while `tfmap_scale`, `both_directions` and the `sir0` split all changed —
three variables, one number. An ablation arm on `tfmap_scale` alone is what would
close D2 honestly.

**Consequence for D1.** D2's stopping rule was that if sharpening the existing
mechanism does not help, a richer dictionary is unlikely to. Sharpening helped but
left 85 % of the output enrollment-blind, so D1 is not cleanly ruled out — but it
stays M5-scale against D4 and D6, which are hours to days.

## E. Performance and memory — open

*Added 2026-08-28. Group D is modelling; this is engineering. Both are open
questions with nowhere else to live. Anything actually decided goes to
`decisions-m2.md`.*

### E1. `batch_size: 3` is a memory ceiling, not a preference

The config comment says batch "should be 12" on GPU. **It cannot be, in fp32.**
Activation memory saved for the backward pass, analytic (4 bytes, ~4x hidden for
cuDNN gate buffers), at `T` = 497 frames, `K` = 32, `N` = 128, `H` = 192, 6 blocks:

| batch (trials) | examples | LSTM activations | fits 15 GB T4 |
|---|---|---|---|
| 3 | 6 | ~4.9 GB | yes |
| 6 | 12 | ~9.8 GB | tight |
| 12 | 24 | ~19.7 GB | **no** |

**These are computed, not measured** — `scripts/profile_step.py` measures them.
But they explain the observed `batch_size: 3` exactly, and the comment promising
12 should be corrected rather than left as an aspiration.

### E2. The band RNN is the dominant cost, and that is counter-intuitive

Per forward at batch 3, LSTM work splits:

    time_rnn  batch = B*K =  192, seq = T = 497           281 GFLOP
    band_rnn  batch = B*T = 2982, seq = K =  32, bidir    563 GFLOP   <- 67 %

**The "across frequency" RNN costs twice what the "over time" RNN does.** It
looks cheap — 32 steps against 497 — but `BSNet.forward` reshapes to
`(B*T, N, K)`, so it runs a separate 32-step bidirectional sequence *for every
frame of every example*: an effective batch of 2982 at `batch_size` 3. It is
also the larger half of the memory in E1, for the same reason.

**Consequence: cost is linear in `T` through BOTH RNNs** — directly for
`time_rnn`'s sequence, and through `band_rnn`'s *batch*. Halving `T` roughly
halves compute and memory together.

### E3. The 3.7x that is not accounted for

Measured 5.84 s/step (3875.2 s / 663 steps, `docs/run_times.md` 2026-08-27).
Analytic floor from E2's FLOPs, assuming LSTMs realise ~20 % of the T4's 8.1
TFLOP/s fp32: **~1.6 s**. So **roughly 3.7x of the step is unexplained** and is
one of: data loading, the 32-iteration Python band loops in `SubbandNorm` and
`Estimator`, or low GPU occupancy at batch 3.

**These have different fixes and we do not know which it is.** Run
`scripts/profile_step.py` before optimising anything. This is the whole reason
that script exists.

### E3b-E3f. MEASURED 2026-08-28 — CLOSED. Evidence tables for `decisions-m2.md` 2026-08-28

Conclusions, mechanism and consequences are written up in `decisions-m2.md`
2026-08-28 ("Training made 7x faster"). This section keeps only the measurements
that entry cites.

**CPU profile** (`profile_step.py`, batch 1, CPU x4, fp32). 11.381 s/step, peak
RSS 5100 MB, forward 34 % of wall.

    stft 0.0 %   tfmap 0.1 %   subband_norm 0.5 %   separator 95.8 %   estimator 3.5 %
      time_rnn  seq=497 batch=64    1.207 s   31.6 % of forward
      band_rnn  seq=32  batch=994   2.276 s   59.6 % of forward

E2's analytic prediction confirmed: band RNN is 65 % of the two RNNs (59.6/91.2).
Loader worst case `--loader-only` 0.382 s/batch for 6 examples at
`num_workers=0`, 6.5 % of a 5.84 s step; 0.044 s/batch on Kaggle at
`num_workers=4`. **This is the only valid attribution** — the GPU attribution is
void (below).

**fp32 vs AMP, batch 3** (T4 14.56 GiB, torch 2.10.0+cu128, 8 steps).

| batch 3 | s/step | peak GPU |
|---|---|---|
| fp32 | 4.741 | 12.20 GB |
| AMP (fp16) | 2.968 | 6.57 GB |

1.60x faster, 1.86x less memory. **Batch 3 is the fp32 ceiling** — 4, 5, 6 and 12
all OOM inside `band_rnn`'s `_VF.lstm`. `bsrnn_baseline.yaml`'s comment promising
batch 12 is wrong by 4x.

**fp16 sweep** (`--amp-only`, 8 steps).

| batch | s/step | s/trial | peak GB | GB/trial |
|---|---|---|---|---|
| 3 | 2.971 | 0.990 | 6.57 | 2.19 |
| 4 | 0.926 | *0.232* | 8.72 | 2.18 |
| 5 | 4.969 | 0.994 | 10.87 | 2.17 |
| 6 | 6.028 | 1.005 | 13.02 | 2.17 |
| 12 | OOM | — | — | — |

Memory is exactly linear: 0.12 GB fixed + 2.15 GB per trial, so batch 7 needs
15.17 GB against 14.56 available — **batch 6 is the fp16 ceiling**. Throughput
per trial is flat across 3, 5 and 6 (0.990 / 0.994 / 1.005, a 1.4 % spread over a
2x batch range): **the T4 is already saturated at batch 3**, which is why
gradient checkpointing was withdrawn.

**Tensor-core alignment, batch 4 reproduced six times across two sessions.**

| batch | ex | runs | mean s/step | spread | s/example | `band_rnn` batch = ex*T | %8 |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 4 | 2.993 | 0.9 % | 0.499 | 3018 | 2 |
| **4** | 8 | 6 | **0.977** | **0.3 %** | **0.122** | **4024** | **0** |
| 5 | 10 | 2 | 5.005 | 0.0 % | 0.501 | 5030 | 6 |
| 6 | 12 | 1 | 6.007 | — | 0.501 | 6036 | 4 |

Aligned vs unaligned is **4.09x**; the three unaligned sizes agree to 0.4 %, the
signature of a shared fallback kernel. The hypothesis predicts the data with
nothing fitted. **T = 503, not 497**, as stated in earlier drafts — `stft.py` pads
by `n_fft - hop` on both sides. Timings and memory figures are unaffected.

Fix confirmed: `chunk_s` 4.008 gives 0.674 s/step at batch 3 (reproduced twice,
0.1 % spread), and the real 2-epoch `sir0` run measured **505.7 s/epoch against
3875.2 = 7.66x**. Losses unchanged — epoch 1 agrees on all twelve logged terms to
within 3 %.

**E1's analytic memory model is 2.4-3x low** (5032 MB predicted at batch 3 vs
12.20 GB measured; 1677 MB vs 5100 MB on CPU). The gap is the autograd graph,
gradients, optimiser state and `L_MR`'s eight retained STFTs. Multiply E1's table
by ~2.5 before predicting a ceiling.

**Two sets of VOID results, both fixed 2026-08-28.** The GPU attribution reported
`estimator` 97.4 % / `separator` 0.8 %, inverting the CPU result: CUDA kernels are
asynchronous, so unsynchronised `perf_counter` hooks measure queuing and whichever
module blocks last absorbs the queue; and the AMP loop accumulated into the fp32
counters. Separately, the first `--amp-only` attempt ran its warmup step
unconditionally in fp32 and so OOM'd at batch 4 before AMP was exercised —
**discard any `--amp-only` result from a bundle built before 2026-08-28 21:00.**

**Operational: profiling locally can kill the terminal.** `systemd-oomd` kills the
whole cgroup scope, not the offending process, and fires on sustained PSI pressure
rather than absolute exhaustion. Batch 3 needs ~15 GB on a 15 GB laptop. Run
local profiling under a scope:

    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -- \
        ../tse_venv/bin/python scripts/profile_step.py --batch 1

`profile_step.py` now refuses to start when its estimate exceeds half of available
RAM, keeps `torch.profiler` behind `--deep`, measures T from the real STFT, takes
`--chunk-s`, and prints an ALIGNED / NOT ALIGNED verdict.

Still open from this group:
- [ ] **Re-run the GPU attribution** with the fixed script. E3b's CPU split is
      the only valid one, and kernel-launch overhead may raise the band-loop
      share on the T4.
- [ ] **fp16 through the `w` ramp is untested** — both validation epochs ran at
      `w` = 0.0, so the absent branch and its mute pressure never engaged.

### E4. Ranked levers, cheapest first

1. **Mixed precision (fp16 + `GradScaler`).** T4 is Turing: fp16 tensor cores
   yes, bf16 **no**. Expect ~1.5-2x on LSTM-heavy work and ~2x less activation
   memory, which alone may unlock batch 6-8. **The loss must stay fp32**:
   `L_pres`/`L_abs`/`L_gain` carry 1e-12 epsilons inside `log10`, and fp16's
   smallest normal is ~6e-5, so they underflow to zero and return NaN. Wrap only
   the model forward in `autocast`; cast to `.float()` before `LossBSRNN`.
2. **`pin_memory=True` on both loaders.** Currently unset in
   `build_loaders()`; `persistent_workers` and `prefetch_factor` already are.
   One argument.
3. **Gradient checkpointing over the six `BSNet` blocks.** Trades ~33 % extra
   compute for roughly 5x less activation memory. Usually a net *win* here: at
   batch 3 `time_rnn` has only 192 parallel sequences, which is poor occupancy,
   and an LSTM's batch dimension is its only parallelism. Bigger batch may pay
   for the recompute outright.
4. **`hop` 128 -> 256.** Halves `T` (497 -> 249) and therefore ~halves both
   compute and memory (E2). Latency granularity 8 ms -> 16 ms, comfortably
   inside the 200-300 ms budget; 50 % overlap still reconstructs. **Costs mask
   time-resolution, so it is an architecture change and needs its own ablation
   arm and decision entry** — not a free win, but the largest single one.
   Note `chunk_s` 4.0 -> 2.0 halves `T` too and is *not* equivalent: it cuts the
   context the model has to learn conditioning in, which is exactly our weak
   point. Prefer the hop change.
5. **Collapse the 32-band Python loops into grouped convs.** `Estimator`'s
   trunks are already uniform (`LayerNorm(128)` + `Conv1d(128->384)` + `Tanh`
   for every band), so all 32 become one `Conv1d(32*128 -> 32*384, groups=32)`
   — identical arithmetic, ~32x fewer launches. The mask/res heads and
   `SubbandNorm` have band-dependent widths and need padding to `max(bw)` plus a
   slice. **Do this last and re-measure first**: it is fixed overhead, so raising
   the batch size shrinks its share. It may not be worth the refactor risk once
   1-3 have landed.
6. **`torch.compile`.** Fuses pointwise work and cuts dispatch overhead; cuDNN
   LSTMs are untouched. Shapes are static (4 s chunks) so recompilation is not a
   risk. Cheap to try, modest gain.

### E5. Audio compression: what it can and cannot do

**It cannot reduce GPU memory.** Peak memory is LSTM activations —
`batch x time x hidden` (E1) — and how a waveform was stored on disk is
irrelevant once it is a tensor. Compression addresses *disk* and *data-loading*
only, so it is worth doing **only if E3's profile shows loading is the
bottleneck**.

- Audio is already 16-bit PCM (`docs/run_times.md`). The easy win is taken.
- FLAC would halve disk but adds CPU decode on a box with ~4 vCPU already
  feeding 4 workers. Could make things **worse**. Measure first.
- **Caching STFTs is a net loss.** A `complex64` spectrogram is 257 x 497 x 8 B
  = 1.02 MB against 128 KB for the int16 waveform — **8x more storage** — to
  skip an operation that is well under 1 % of compute.

### E6. Why this is worth doing at all

At the measured 3875 s/epoch, on the 10-epoch schedule:

| | per epoch | 10 epochs | 12 ablation runs | one 10k-trial run |
|---|---|---|---|---|
| today | 3875 s | 10.8 h | 129 h | 54 h |
| 2x faster | 1938 s | 5.4 h | 65 h | 27 h |
| 3x faster | 1279 s | 3.6 h | 43 h | 18 h |

The planned arms alone (`ablate_w_m` 3, `ablate_w_g` 3, lookahead 3, plus D4a
and a `tfmap_scale` arm to close D2) are ~12 runs. **Speed converts directly
into how many questions the thesis can answer**, and Kaggle's ~12 h session cap
makes the 10k-trial run multi-session at today's rate and single-session at 3x.

### E7. PROPOSAL — `DataParallel` across the second T4, which is currently idle

**Raised 2026-09-04.** Kaggle's "GPU T4 x2" accelerator gives **two** T4s. The
code uses one: `torch.device("cuda")` is `cuda:0`, and there is no
`DataParallel`, `DistributedDataParallel` or `device_count` anywhere in
`scripts/train.py` or `src/models/`. **Half the allocated hardware has been idle
for every run so far.** The P100 option is one card, so this lever exists only
on the T4 x2 selection.

**Why it is cheaper than it looks.** The loss is computed *outside* the model:

```python
s_output = model(mixture, enrollment)                     # parallelisable
loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent)
```

`DataParallel` scatters the batch, runs the forward on both cards, and gathers
outputs back to `cuda:0` — so `LossBSRNN` still sees the whole batch on one
device. That matters because the loss means over *subsets* (`n_present`,
`n_absent`) and a per-device reduction would silently reweight them.

**It does NOT break the direction pairing, contrary to the first reading.**
`collate_pairs` guarantees both directions of a trial land in the same batch,
and splitting the batch across devices does separate some pairs in the forward
pass — but the model is per-example independent and the contrast lives in the
*loss*, which sees the gathered batch. Recorded because the worry is the obvious
one and it is wrong.

**The work — six call sites, one real trap.**

1. Wrap after `build_model()`/`.to(device)`, gated on
   `torch.cuda.device_count() > 1` **and** a config flag defaulting to off, so
   every existing run reproduces.
2. Keep `core = model.module if wrapped else model` and use `core` for:
   - `state_dict()` at the three `torch.save` sites (**the trap**:
     `DataParallel` prefixes every key with `module.`, which would break
     `--resume`, `make_estimates.py`, and every checkpoint already on disk);
   - `load_state_dict()` on resume;
   - `model.band_widths` in `log_results()` — `DataParallel` does not forward
     arbitrary attributes, so this raises `AttributeError`, not a wrong number.
3. `diagnostic_accumulate()` calls `model(...)` for the enrolment swap; it can
   take the wrapped model, but must be the *same* one the forward used.
4. Log `device_count` in `meta.yaml` — a run's s/epoch is unreadable without it.
5. The notebook's batch-size probe measures one GPU; with two the effective
   batch doubles, so the probe's answer changes meaning.

**Expected gain: ~2x, but ONLY at `batch_size: 6`. REVISED 2026-09-04** — the
first estimate here was 1.4-1.7x, reasoning that splitting batch 3 would leave
each card with poor LSTM occupancy. E3b-E3f's fp16 sweep already answers that
and the first estimate was too pessimistic:

| batch | s/trial | peak GB |
|---|---|---|
| 3 | 0.990 | 6.57 |
| 5 | 0.994 | 10.87 |
| 6 | 1.005 | 13.02 |

Per-trial throughput is **flat over a 2x batch range (1.4 % spread)**, i.e. a
single T4 is already saturated at batch 3. So batch 6 split 3/3 gives each card
a *saturated* batch 3, and the ceiling is ~2x rather than 1.4-1.7x. Splitting
batch 3 into 1.5/1.5 is the case that would disappoint.

**CAVEAT, and it is not small: the saturation evidence is from the UNALIGNED
regime.** Noticed 2026-09-04. Batches 3, 5 and 6 in that sweep are exactly the
three sizes E3b-E3f identifies as sharing a fallback kernel ("agree to 0.4 %,
the signature of a shared fallback kernel"); batch 4, the aligned one, ran 4.09x
faster per example. Flatness across three points all running the same slow
kernel shows that kernel is linear in work — it does **not** show the GPU is
saturated on the fast path. Since `chunk_s: 4.008` everything runs aligned, and
in that regime there are only two comparable points: batch 3 at ~0.112
s/example (0.674 s/step / 6) and batch 4 at 0.122. Roughly flat, but two points
from different measurement sets.

The 2x claim requires per-example throughput to stay flat from **6 to 12
examples while aligned**, which has never been measured. If an aligned card is
not saturated at 6 examples, DataParallel returns less than 2x.

**Settle this before writing any DataParallel code.** An aligned batch sweep
(3/4/5/6 at `chunk_s: 4.008`, `profile_step.py --amp-only`) is ~20 min of GPU
and decides whether E7 is worth half a day. It is also the cheapest possible
test: if throughput per example *rises* with batch on the fast path, the honest
conclusion is that a bigger batch on ONE card is the win and the second card
adds little.

**So E7 depends on E8: `batch_size` must go 3 -> 6 first**, and that needs no new
technique — 6 is the measured fp16 ceiling (13.02 GB of 14.56) and fits today.
Two cards is also 2x the memory that forced batch 3 in the first place (E1).

Replication overhead is *not* a concern: 7.19 M params is ~29 MB fp32, broadcast
per step over PCIe at ~6 GB/s is ~5 ms against a ~750 ms step (1244 s / 1658
steps at 4,976 trials), i.e. **under 1 %**. What remains is scatter/gather
latency.

**Alignment must be re-checked, not assumed.** The 4.09x tensor-core cliff
(E3b-E3f) depends on `examples x T` staying divisible by 8. `chunk_s: 4.008` was
chosen so `T` is divisible by 4, which makes every batch size aligned today —
but a DataParallel split changes the per-device batch, so verify with
`profile_step.py`'s ALIGNED verdict before reading any timing. A 4x regression
that looks like "the second GPU didn't help" is the easiest wrong conclusion
available here.

**Cost: roughly half a day**, most of it verification rather than code. Requires
a 2-epoch A/B at fixed seed. Numerics will not be bit-identical (different
kernel split and fp16 reduction order), so the A/B must show val terms agreeing
within the known between-run noise, not exactly — and per the 2026-08-29 AMP
row, that noise is large enough to swallow real differences, so this is a
**speed** claim only and must never be reported as a quality change.

**`DistributedDataParallel` is the better tool and is deferred.** It avoids the
per-step replication and the gather, but needs `spawn` multiprocessing inside a
Kaggle notebook, which fights the single-process training loop and the tee'd
`history.csv` recovery path. Not worth it for 2 cards on one host.

**Sequence.** After the 9,955-trial run lands, not before — that run is the
third point on the data-scaling curve and must not change two variables at once.

### E8. PROPOSAL — raising `batch_size`: what works, and what cannot

**Raised 2026-09-04**, from asking how compression could buy a bigger batch.
E7 needs batch 6; this is how far batch size can go and at what cost.

**First, the premise is wrong for speed.** Per E3b-E3f, per-trial throughput is
flat from batch 3 to 6 (0.990 / 0.994 / 1.005 s/trial). **A bigger batch does
not make training faster on one T4** — the card is saturated at batch 3, which
is exactly why gradient checkpointing was withdrawn. The 7.66x came from
tensor-core *alignment*, not batch size: aligned vs unaligned is 4.09x, batch
3->6 is 1.0x. So a bigger batch is worth wanting for only three reasons:

1. **To feed DataParallel (E7)** — the real one. Two saturated cards, ~2x.
2. Different gradient statistics — a change to training dynamics, needing its
   own arm, not a speed optimisation.
3. Headroom for a longer `chunk_s` or `lookahead_frames` later.

**Batch 6 needs no new technique.** Measured ceiling is 13.02 GB of 14.56
available; memory is exactly linear at **0.12 GB fixed + 2.15 GB per trial**.
Batch 7 needs 15.17 GB and does not fit. So E7's prerequisite is a config edit,
not an engineering project.

**BUG, found 2026-09-04: the notebook's batch probe measures fp32 and the run
trains in fp16, so it has been capping every Kaggle run at batch 3.**
`_probe_batch.py` in `scripts/make_kaggle_notebook.py` does:

    loss, _ = L(s, m(x, e), x, a)
    loss.backward(); opt.step()          # no autocast, no GradScaler

while `train.py` wraps its forward in `amp_ctx(use_amp)` with `amp: true`. So the
probe finds the **fp32** ceiling — measured in E3b-E3f as exactly batch 3, with
4/5/6/12 all OOM in `band_rnn`'s `_VF.lstm` — and writes it into the config that
then trains at 1.86x less memory (12.20 GB fp32 vs 6.57 GB AMP, both batch 3).
This explains the standing belief that one T4 fits only batch 3: true in fp32,
and the runs are not fp32.

**This is a repeat of a bug already fixed elsewhere.** E3b-E3f records "the first
`--amp-only` attempt ran its warmup step unconditionally in fp32 and so OOM'd at
batch 4 before AMP was exercised". That fix was applied to `profile_step.py` and
never propagated to the notebook probe.

Fixing the probe is the whole of E8's cheap half: wrap its forward in the same
`amp_ctx` the trainer uses and let it find 6. Until then, no Kaggle run can
reach the batch size E7 depends on, and any "batch 6 does not fit" report from
the notebook is measuring the wrong precision.

**Memory is essentially ALL activations**, and that kills a whole category of
proposals before anyone spends a day on them. The model is 7.19 M params
(~29 MB); AdamW state is ~57 MB. Against ~13,000 MB:

- **8-bit optimisers (bitsandbytes) would save ~43 MB.** Not worth the
  dependency. Recorded so it is not proposed again.
- **Audio compression cannot reduce GPU memory at all** — E5 already says this;
  how a waveform was stored on disk is irrelevant once it is a tensor. Caching
  STFTs is 8x *more* storage (E5).

**The levers that would actually work, beyond batch 6:**

1. **`hop` 128 -> 256 — the only one that reaches batch 12.** Halves `T`, and
   per E2 cost is linear in `T` through BOTH RNNs, so activations and compute
   halve together: ~1.08 GB/trial, so batch 12 fits. Costs mask time resolution
   (8 ms -> 16 ms granularity, still well inside the 200-300 ms budget). Already
   E4's item 4 and ranked the largest single win; an architecture change needing
   its own ablation arm.
2. **Gradient checkpointing over the six `BSNet` blocks — reconsider for THIS
   purpose.** ~5x less activation memory for ~33 % more compute. Correctly
   withdrawn as a *speed* lever on a saturated card, but buying batch headroom
   to feed a second GPU is a different trade and the arithmetic changes.
3. **`L_MR`'s eight retained STFTs.** Named in E3b-E3f as part of the 2.4-3x gap
   between E1's analytic model and measured memory. `windows_ms: [8,16,32,64]`
   is four resolutions held for the backward pass. Recomputing them, or dropping
   to two windows, attacks a *measured* contributor without touching the
   architecture — but it changes the objective, so it needs an ablation.
4. **Gradient accumulation**, if what is wanted is large-batch *statistics*
   rather than throughput. Zero extra memory, and mathematically exact here
   because the model uses LayerNorm/SubbandNorm and not BatchNorm, so there is
   no cross-example statistic to corrupt. Makes nothing faster.

**Rejected: shorter `chunk_s`.** Halves memory the same way as `hop`, but cuts
the context the model has to learn conditioning in — the documented weak point
(enrolment sensitivity flat at -3.79 dB across the 2.5x data increase,
`decisions-m2.md` 2026-09-01). Same objection E4 already records.

**Alignment governs all of it.** `examples x T` must stay divisible by 8 or the
4.09x fallback kernel fires. `chunk_s: 4.008` makes `T` divisible by 4, so every
batch size is aligned today — but levers 1 and 3 change `T` or the STFT set and
can silently break that. Re-check with `profile_step.py`'s ALIGNED verdict
before believing any timing.

**CHANGING `batch_size` SILENTLY CHANGES THE `w` SCHEDULE.** The absent-branch
warmup is indexed in optimiser *steps* (`decisions-m2.md` 2026-09-03), which
makes it invariant to **dataset size** — the confound it was built to remove —
but **not** to batch size. At batch 6 each step consumes twice the examples, so
the same 11,606-step warmup covers 2x the audio it covered at batch 3. The
warmup exists to stop the early mute, and its length in examples is the thing
that matters for that, so this is a real change to the objective's schedule and
not a bookkeeping detail. Either hold `warmup_steps`/`ramp_steps` x batch
constant when batch changes, or treat a batch change as its own arm. **Do not
let the probe pick a new batch size during the 9,955-trial run** — set the
notebook's `BATCH_SIZE` knob to 3 to pin it, since candidates are filtered to
`<= BATCH_SIZE`.

**Sequence.** `batch_size: 3 -> 6` is the cheap prerequisite for E7 and can ride
with it. Levers 1-3 are only needed to go beyond 6 and should wait until E7 has
measured what two saturated cards actually deliver.

### D9. ASR cross-entropy as a differentiable content proxy

**Status: proposal, deferred. This is M5, which is CUTTABLE and the first thing
to cut. Raised 2026-08-30.**

**The origin.** The idea was first put as "transcribe the output each epoch,
compute WER, add it to the loss and hold it constant through the next epoch".
That specific mechanism does nothing: **a term that does not vary with the
weights has zero gradient**, so backprop would produce identical updates and only
the printed loss would change. Recomputing per batch does not rescue it either —
WER passes through `argmax`/beam search and edit distance, neither of which is
differentiable. Recorded because the reasoning is the useful part, not the
conclusion.

**The working version.** Push the estimate through a FROZEN ASR and take the
cross-entropy of its token distribution against the known transcript,
teacher-forced. No decoding, so no `argmax`; the gradient flows through
continuous logits into the mask. CLAUDE.md names this directly: "training uses
differentiable proxies (frozen-ASR/SSL feature matching, optionally ASR
cross-entropy)". Feature matching against the clean target's ASR-encoder
features is the cheaper sibling and needs no text at all.

**The rigorous version of the original idea exists**: minimum-WER training
(Prabhavalkar et al., ICASSP 2018) optimises expected WER by sampling hypotheses
and using a score-function estimator rather than a true gradient. Expensive and
finicky; cross-entropy is the usual choice for a reason.

**Hard constraint.** The proxy ASR must NOT be `small.en` — that is the
evaluation scorer, and training against your own evaluator makes the offline WER
meaningless (CLAUDE.md rule 2). A different model family, recorded not assumed,
and never the judge in any form.

**Where WER DOES belong right now, and it is free:** as a model-SELECTION
criterion. It cannot make gradients but it ranks finished models perfectly well,
and `training.select_on` (2026-08-30) now has a defined place for it. Scoring
four or five candidate epochs through estimates -> ASR -> WER and checking
whether the present-branch proxy ranks them the same way is one afternoon and a
thesis table either way.

### D10. Penalise resembling the INTERFERER, not just missing the target

**Status: proposal, unscheduled. The best remaining model-side idea, and the
only one aimed at extraction-vs-enhancement. Raised 2026-08-30.**

**The problem.** `L_pres` maximises SI-SDR to the target, which buckets every
error together — interferer leakage, residual noise and artefacts are one
undifferentiated residual. For THIS project they are not equally bad: **the
interferer's words are the worst possible error**, because a live judge
transcribes them as the target's speech. Demonstrated on
`eval_public-42-000132`, where the ASR reads the target correctly for 17 words
and then transcribes the interferer's sentence verbatim.

**The proposal.** An explicit repulsion term against `interferer.wav`, which is
already rendered per trial and already loaded (`both_directions`). Weight
interferer leakage above other residual error, so the objective prefers a noisy
extraction over a clean confusion.

**Why it fits this project specifically.** It is the training-side mirror of
**ICR** (`metric-definitions.md` 3.2), which the metric already defines as "the
score that makes the metric two-sided, and the one an offline WER-based metric
structurally cannot see". Objective and metric would then measure the same
failure, which is a clean thing to write up.

**What must be checked before building it.** `L_pres` already counts interferer
energy as error, so the term is partly redundant; what it adds is a WEIGHTING,
and the ablation has to show the weighting earns its place (0 arm required, as
for `w_m` and `w_g`). It also risks a new degenerate solution — output silence
resembles neither speaker — which `L_gain` now blocks but which must be
re-verified, not assumed.

**MEASURE FIRST, and it costs nothing extra.** `interferer_text` is in every
`meta.json`, so transcribing the estimates yields WER against the target AND
content overlap against the interferer in the same pass. That says whether
leakage is actually the dominant error before any term is written to fix it.
Ordering: measure ICR, then decide.

**Evidence that the model is NOT purely enhancing** (2026-08-30): an enrolment
swap moves the output 48.2 % (D3a) where a pure enhancer would move 0 %;
`both_directions` requires two different answers from one mixture; `sir0` removes
the loudness shortcut. The worry is a matter of degree — 52 % of the output is
still enrolment-independent — not a yes/no.

---

## Group J — the judge and the metric

Raised 2026-08-30. M4 is 1 item done of 9 and the judge gates the rest, so these
are the decisions with the longest lead time in the project.

### J1. Must the judge be full-duplex speech-to-speech, or is audio-in enough?

**Status: CLOSED 2026-08-31 — audio-in / text-out. The recommendation below was
taken.** Reasoning, the three gains, the cost and the ~50-trial full-duplex
confirmation run are in `decisions-m4.md` 2026-08-31. The analysis below is kept
as the argument that produced the decision.

**The tension.** CLAUDE.md and spec note 10 both say the objective is what a
"**live speech-to-speech model** (Gemini Live and similar)" recovers. Read
strictly that requires full duplex. But `metric-definitions.md` 3.1's own
mechanism does not:

> A live speech-to-speech model consumes audio through a learned audio encoder
> over a much wider distribution, and appears to be more sensitive to the
> artefacts of the processing itself than to the interfering speech it removed.

**The property being measured is the AUDIO ENCODER, not the duplexing.** An
audio-in / text-out LLM has exactly that encoder and exactly that wide training
distribution. Nothing in LCF-WER, ICR or NRR reads the judge's turn-taking.
3.1 step 2 already permits a text response -- "*If* the model responds in audio,
transcribe the response" -- so audio output is optional in the protocol as
written.

**What relaxing it buys.**

1. **A much wider field.** Ultravox, Voxtral and Qwen3-Omni all qualify;
   full-duplex narrows it to roughly Moshi and the closed APIs.
2. **One fewer component in the measuring instrument.** A text response removes
   the response-ASR entirely. 3.1 step 2 calls that ASR "a component of the
   measuring instrument, and changing it invalidates comparisons" -- so deleting
   it removes a whole class of invalidation.
3. **Cheaper and faster** on both the API and the self-hosted side.
4. **Ultravox is the closest conceptual fit**: it skips the separate ASR stage,
   which is precisely the property 1 hypothesises about.

**What it costs.** It is a deviation from the stated objective and must be
argued, not assumed. The defensible sentence is: *we used an audio-in model
because the measurement depends on the audio encoder rather than on duplexing*.
If that argument is not made explicitly in the write-up, a reviewer is entitled
to say the thesis measured something other than what it set out to.

**Recommendation: relax it, and record the argument above.** But it needs
supervisor sign-off, because it edits the project's stated objective rather than
an implementation detail.

### J2. Which judge -- and the open-weight anchor is about reproducibility, not cost

**Status: OPEN. The cost half is ANSWERED and was smaller than assumed.**

**The cost model M4 asks for, measured 2026-08-30.**
`gemini-3.1-flash-live-preview` publishes $0.005/min audio in, $0.018/min audio
out, $4.50/1M text out, with a free tier. Trials are ~18 s, and M4's protocol is
200 trials x k=3 repeats x 4 systems = 2,400 calls:

| | |
|---|---|
| audio in, 720 min | $3.60 |
| text out, ~0.12M tokens | $0.54 |
| **audio condition total** | **~$4.14** |
| with audio responses instead | ~$15 |
| plus a prompt-sensitivity ablation | ~$25 all in |

**So M4's "closed API (money) or self-hosted open-weight (GPU-hours)" is not a
budget question.** ~$25 is noise, and the free tier covers the pilot. Take the
closed API for the headline.

**The open-weight anchor is still required, for a different reason.** Both
candidate Gemini IDs are marked `preview` and preview models get deprecated. If
the headline judge disappears before submission the primary result becomes
unreproducible -- which is why CLAUDE.md already demands the exact model ID and
run date on every judge result. The anchor exists so someone can reproduce the
headline in two years, not to save $25. **That is the argument to make in the
write-up; a cost argument would be weaker and also false.** Published prices
carry expiry dates too ("through December 31, 2026"), so record the price
alongside the ID.

**Compute for a self-hosted anchor is no longer contended.** M4 worried about
GPU-hours competing with training quota; training is being stopped (2026-08-30),
so the Kaggle T4 is free. A 3B model in fp16 fits it and 2,400 short generations
is one session. The laptop cannot do it -- no usable GPU, and CPU generation puts
the full protocol at about a day of wall clock -- but it can run a 20-trial
pilot.

**The gate that must be applied to EVERY judge candidate, open or closed.**
Score the **ceiling condition first**: feed the clean target audio and read
LCF-WER. The offline ASR ceiling is 6.1 %. If a judge cannot reliably report
clean speech, the judge is the bottleneck and every system comparison beneath it
is noise. Roughly 20 trials and an hour, and it disqualifies candidates before
any of them cost a benchmark run.

**Why this gate matters more for open-weight candidates.** On FullDuplexBench,
task adherence is 1.26/5 for Moshi and 3.82/5 for Qwen2.5-Omni. Our prompt is
trivial -- "report what you heard" -- but a judge that wanders off-prompt,
refuses, or chats instead of reporting lands in **NRR**, which was designed to
catch a degenerate EXTRACTOR. **A degenerate JUDGE is indistinguishable from it
in the numbers.** Choose the anchor for instruction-following on a
transcription-style prompt, not for conversational ability.


### J3. The ICR overlap threshold — declared, not signed off

**Status: CLOSED 2026-09-03.** Signed off into `metric-definitions.md` §3.2 —
threshold, sensitivity table, exclusion rule, the floor's by-construction caveat
and the prompt constraint are all stated there now. decisions-m4.md 2026-09-03.
The reasoning below is kept as the record of how the value was chosen.

`metric-definitions.md` 3.2 defines ICR as "content-word overlap between `r` and
`d`, excluding words that also appear in `t`, thresholded" and requires the
threshold to be **fixed in advance with its sensitivity reported**. It does not
say what the threshold is. Two candidate rules:

| rule | statement | problem |
|---|---|---|
| **`count>=2`** | ≥2 interferer-exclusive content words appear in `r` | insensitive to how much the interferer said |
| `frac>=θ` | that count as a fraction of the interferer-exclusive words available | scale-dependent on the interferer's utterance length, which varies per trial by construction |

**Declared: `count>=2`.** One shared content word between a response and the
interferer is coincidence at the rate English repeats nouns; two is signal. The
fraction rule varies with a property of the trial rather than of the system,
which makes it the worse primary and the better secondary. Both are computed and
reported, with a sweep over counts 1/2/3/5 and fractions 0.05–0.50, per 3.2's
sensitivity requirement.

**Two things to settle at sign-off.**

1. **Trials where the interferer said nothing the target did not also say** carry
   no evidence of contamination either way. They are **excluded** from ICR, not
   scored as clean — scoring them clean would dilute the rate towards zero with
   trials that could never have fired. The exclusion count is reported.
2. **The floor row's ICR is partly set by construction, not measured.** The judge
   never sees the enrolment, so on an unprocessed two-speaker mixture it cannot
   know which speaker is the target and will pick one. That makes the floor's ICR
   tend towards a coin flip. This is the correct behaviour and it *is* the
   finding — doing nothing gets you the wrong speaker half the time — but it must
   be stated when the floor row is quoted, not discovered in a results table. The
   fixed prompt must therefore **not** instruct the judge to choose a speaker
   ("the clearest voice", "the loudest speaker"): that hands the extractor's job
   to the judge and turns a measurement into an instruction.

### J4. PROPOSAL — a metric *system*: normalised requirement axes, composed, plotted

**Status: OPEN, proposed 2026-08-31 (Grant's idea). Not a decision yet. The
diagnosis is right, the normalisation and the composition rule both need
changing before it is defensible, and one part of it as pitched is unsound.**

**The problem it solves, and it is real.** LCF-WER, ICR, NRR, SI-SDR,
DNSMOS (P.835 since 2026-09-01), offline WER and latency is seven numbers, and B13 requires each of
them broken out per condition — so the honest results table is roughly 35 cells
per system. **Nobody can rank two models by reading 35 cells**, and a thesis that
asks the reader to is failing to make its own argument. There is currently no
defined way in this project to say "model A is holistically better than model B",
only "A is better on this row".

### The proposal

1. The user declares **n requirements** for their speech model (e.g. *Speaker
   learning*, *Sound separation*, *Content fidelity*), placed as n equally
   spaced axes on a circle.
2. Each requirement is fed by **several underlying metrics**. Example given for
   *Speaker learning*: (a) how many words of the estimate appear in the
   interferer's speech, (b) how long the model tracked the interferer during
   target silence, (c) the same during target speech.
3. Each metric is scored **relative to a declared baseline** (a real or
   hypothetical reference model), so the axis is an improvement, not a raw unit.
4. Radius = how good: **out toward the rim is better**, near the centre is worse.
5. Two models are overlaid on one chart, and the **shape** shows what each is
   good and bad at.
6. **Ranking by total area** enclosed, plus ranking by a single axis or by a
   group of axes.

**Declared axiom (keep it, it is the right instinct):** every metric admitted to
the system must be able to rank two models against each other and say which is
better.

### What is right about it

**It makes two-sidedness structural rather than a convention.**
`metric-definitions.md` 4 already requires LCF-WER, ICR and NRR to *always* be
reported together, because suppressing everything wins on ICR and passing
everything through wins on NRR. Today that is enforced by discipline. On an axis
plot you cannot show one without the others — they are spokes of the same figure.
That is a genuine strengthening of an existing commitment, not decoration.

**Baseline-relative axes are correct**, and the anchors already exist:
`metric-definitions.md` 3.4 makes floor (unprocessed mixture) and ceiling (clean
target) mandatory on every results table.

**Grouping metrics into requirement classes is worth it for the viva.** "Better
at holding onto the right speaker, worse at avoiding processing artefacts" is a
sentence a reader can carry; seven numbers is not.

### Three things that must change first

**1. RANKING BY AREA IS UNSOUND. Do not do it.** Radar-polygon area depends on
the *order the axes are drawn in*, which is arbitrary. For n equally spaced axes
with radii `r_i`:

```
Area = ½ · sin(2π/n) · Σ_i r_i · r_{i+1}
```

Only **adjacent** pairs multiply, so a model strong on two neighbouring axes
scores more area than one equally strong on two opposite axes. Concretely, n=4:

| model | scores in drawn order | Σ r_i·r_{i+1} | area |
|---|---|---|---|
| X | 1, 1, 0, 0 | 1 | > 0 |
| Y | 1, 0, 1, 0 | 0 | **exactly 0** |

**Identical multisets of scores, and Y encloses no area at all.** The ranking came
from where the labels were placed, not from the models. Worse, this is the exact
failure mode `metric-definitions.md` 4 was designed against: a score with a free
parameter (axis order) that can be tuned to change the winner is a gameable
score, and REAL-TSE had to swap its official metric after the fact for a
comparable reason.

**Fix: keep the picture, take the ranking from an explicit weighted mean of the
normalised axis scores.** Order-invariant, the weights are visible and arguable,
and the chart still does the job it is good at — showing shape.

**2. "Percentage increase over baseline" breaks on signed and dB quantities.**
The worked example in the proposal is do-nothing SI-SDR **−2.12 dB** and model
**5.15 dB**. Percentage change between them is `(5.15 − −2.12)/(−2.12) = −343 %`
— a negative number for an improvement, because the denominator is negative.
Undefined at baseline = 0, and meaningless for any quantity that crosses zero.

**Fix: normalise to the floor–ceiling interval**, which is dimensionless,
well-defined for signed and dB quantities, and reuses anchors the protocol
already mandates:

```
s = (x − floor) / (ceiling − floor)        clipped to [0, 1]
```

`s = 0` is "doing nothing", `s = 1` is "the best achievable on this judge". For a
lower-is-better metric the interval simply runs the other way — WER with
floor 57.4 % and ceiling 6.1 % gives `s = (57.4 − x) / (57.4 − 6.1)`. Same
formula, so every axis is on one comparable scale and "toward the rim is better"
is true by construction rather than by per-metric convention.

**3. The axiom needs strengthening.** "Can rank two models" is *ordinal*, and an
ordinal metric cannot be placed at a radius — knowing a model is 2nd of 3 does
not tell you how far out to draw it. The real requirement is that each metric be
**monotone in goodness and cardinally normalised**, which item 2 supplies.

### Two design questions to settle before building

**Which mean, and it matters more than it looks.** An arithmetic mean (and area,
and any sum) lets a model **compensate**: superb LCF-WER hides catastrophic NRR,
which is precisely the degenerate mute this project already caught once. A
**geometric mean** `(Π s_i)^(1/n)` collapses to zero if the model is at floor on
*any* axis, so it cannot be gamed by trading one requirement away. **That is the
mathematically principled version of the two-sidedness rule** and is the
recommended headline; report the arithmetic mean beside it, and state which is
the headline. Needs a decision on flooring `s_i` so one axis at exactly 0 does
not erase an otherwise-informative model.

**User-defined weights are a gaming surface.** Configurable requirements and
weights are good in a *tool* and fatal in a *benchmark*: if anyone can reweight,
anyone can make their model win. **The benchmark must publish one fixed,
pre-registered weighting, frozen before results are seen**, exactly as the prompt,
the normaliser and the ASR are frozen. The configurable version is a separate
exploration mode, labelled as not the benchmark number.

### Scope, so this does not balloon six weeks from freeze

**This is a presentation and composition layer, not a new metric.** It does not
change LCF-WER, ICR or NRR, and it cannot invalidate them — which is what makes
it cheap. Realistic size: one module that takes the existing per-condition scores
plus a frozen weighting file, and emits the figure and the composite. It is a
Chapter 4 figure and a ranking rule, not a rebuild.

**Honest framing for the write-up.** Radar charts are old and are criticised in
the visualisation literature, largely for the area problem above — so the
contribution is *not* the chart. The contribution is **a composition rule for TSE
evaluation**: floor/ceiling-normalised axes, grouped into declared requirement
classes, aggregated by a compensation-resistant mean, under a pre-registered
weighting. Claim that, not the picture.

**Consider a dot/parallel-coordinates companion plot.** Same data, no area
artefact, exact values readable. The radar answers "what shape is this model";
the dot plot answers "by how much". Cheap to emit both from the same numbers.

### Open sub-questions

- Does **latency** belong on a quality axis at all? It is a **constraint** with a
  200–300 ms budget, not a dimension to trade off — put it on the radar and a
  model can win on shape by being fast and mediocre. Probably a pass/fail gate
  plus B11's decay curve, kept off the composite.
- The proposed *Speaker learning* metrics (b) and (c) — time spent tracking the
  interferer during target silence and during target speech — **are not built and
  are not in `metric-definitions.md`.** They need a definition and a
  ground-truth source (the VAD index gives per-speaker activity, so this is
  feasible) before they can be axes.
- How do axes behave for a model **below the floor**? The 08-29 checkpoint at
  epoch 24 was *worse than pass-through*. Clipping at 0 hides that; allowing
  negative radii breaks the plot. Probably clip, and flag "at or below
  do-nothing" on the axis label.
- Does the composite get reported **per B13 condition** as well as pooled? It
  must, or the composite becomes the aggregate-that-appears-alone that B13
  forbids.

### D11. Inference-time mix-back: a measuring instrument, NOT a fix to the model

**Status: OPEN. Raised 2026-09-01. Grant's objection is recorded and accepted —
this is a cheap post-hoc patch rather than a strategy for fixing the process.
Kept because its value as an INSTRUMENT is separate from its value as a fix.**

**The proposal.** Blend the model's output back with its own input at inference,
`s_alpha = alpha * s_hat + (1 - alpha) * x`. `alpha = 1` is the current model,
`alpha = 0` is doing nothing. Costs one multiply-add per sample, adds **zero
algorithmic latency** (output sample n needs only input sample n), and requires
no retraining because `alpha` is not a model parameter. All values of `alpha`
come from a single forward pass.

**Why it was proposed.** The 2026-09-01 measurement found the model applies the
same transform regardless of difficulty — SIR improvement 3.80 to 4.32 dB and SAR
degradation -17.45 to -21.41 dB are essentially flat across easy-to-hard trials —
while the word-error outcome swings from -4.2 to +23.1 points. A single global
knob is therefore the right *shape* of intervention, because the model's signal
behaviour is constant.

### The objection, which stands

**This does not fix anything.** It trades away the model's benefit on hard trials
to stop it hurting easy ones, using a constant chosen offline. The model still
cannot tell the two cases apart, still produces the same artefacts, and still has
no mechanism to modulate itself. A single global `alpha` is a compromise, not a
capability.

**So it must not be presented as an improvement to the extractor.** In any
results table it is *the extractor with a mix-back gain*, one system and a
parameter, never five systems.

### Why it is still worth running: it is the divergence instrument

The sweep produces a family of systems from one checkpoint, walking the
artefact-versus-residue trade in a controlled way — letting the mixture back in
raises SAR and lowers SIR by construction. That family is what M6's divergence
result needs, and it needs no training and no second architecture. **Its value is
as a measurement, and it should be described that way.**

Cost is transcription, not compute: alpha = 0 and 1 are already transcribed, so 5
values over 200 trials is ~600 new transcriptions, about 30 min of CPU.

### The actual strategies, if the underlying problem is to be fixed

Recorded so the cheap version does not crowd them out. In rough order of
principle:

**1. A learned, input-conditioned gate.** Have the model predict its own
`alpha`, per frame, from the mixture — filter hard where there is interferer
energy to remove, barely at all where there is not. This is the principled form
of the same idea: it gives the model the capability the global knob fakes. Small
head, needs retraining, and it directly addresses "the model cannot tell the
cases apart".

**2. A differentiable artefact penalty in the objective.** The deeper cause is
that `L_pres` collects residual interference and invented artefact into one
denominator, so per unit of energy they cost the same. **This project has all
three clean sources, so the SIR/SAR split is computable at training time**, and
a term penalising the artefact residue specifically is therefore possible. That
attacks the cause rather than the symptom, and is the strongest M5 candidate on
the table.

**3. Reconsider whether aggressive masking is the right output parameterisation
at all.** The artefacts are a property of masking. This is the expensive option
and is almost certainly out of scope before the freeze.

**Recommendation: run the sweep as an instrument for M6, and log option 2 as the
modelling response.** Do not let the sweep be written up as the answer to the
easy-trial regression.

### D12. Mixture of experts over masking behaviours — the scaling idea

**Status: OPEN, and more viable than first assessed. Raised by Grant 2026-09-01.
Out of scope before the 14 October freeze, kept because the parameter accounting
turned out favourable and the experiment design is sound.**

**The idea.** `K` masking behaviours with a gate that infers, per frame and per
band, which applies. Test at fixed data first; scale after. Grant's framing:
*"test on the same amount of data and then scaling"* — which is the right
experiment order, because a win at fixed data is the informative result.

### The parameter accounting, measured 2026-09-01

The original objection was capacity: this model is data-limited, so adding
capacity makes the measured problem worse. **That objection was based on
replicating the whole estimator, which is not necessary.** Measured breakdown of
the 7,189,644 parameters:

| module | parameters | share |
|---|---|---|
| separator (LSTM stack) | 4,898,304 | 68.1 % |
| estimator | 2,187,014 | 30.4 % |
| — of which `trunks` | 1,593,344 | 72.9 % of the estimator |
| — of which `mask_heads` | 395,780 | 18.1 % |
| — of which `res_heads` | 197,890 | 9.0 % |
| subband_norm | 104,326 | 1.5 % |

**Share the trunks, replicate only the heads:**

| variant | added | total | increase |
|---|---|---|---|
| K=5, full estimator replicated | +8.75 M | 15.9 M | **+122 % — fatal** |
| K=3, shared trunk, both heads | +1.19 M | 8.38 M | +16 % |
| **K=3, shared trunk, mask heads only** | **+0.79 M** | **7.98 M** | **+11 %** |
| K=5, shared trunk, mask heads only | +1.58 M | 8.77 M | +22 % |

**+11 % at K=3 is defensible even on a data-limited model**, and the overfitting
risk is lower than the count implies: the separator (68 %) and the trunks (22 %)
stay shared, so the experts are `K` read-outs of one representation rather than
`K` models. **The earlier "high capacity risk" assessment was wrong** and is
corrected here.

### The maths, and why it does not need a Gumbel trick

Soft routing is differentiable as it stands:

```
g = softmax( f(X, e) )        in R^K
S_hat = sum_k  g_k ( m_k * X )
dL/dg_k = < dL/dS_hat , m_k * X >
```

The variational form Grant asked about treats difficulty as a discrete latent
`z` with prior `p(z)` and posterior `q(z|X,e)`, optimising the ELBO. **For small
`K` the expectation is computed exactly by enumeration** — no sampling, no
reparameterisation, no REINFORCE variance:

```
L = sum_k  q_k L_recon( S_hat_k , S )  -  lambda sum_k q_k log( q_k / p_k )
```

**This mixes the LOSSES, not the masks**, which is the better property: each
expert must be individually good on the cases assigned to it, whereas averaging
masks can produce a mask worse than any individual one. The KL term stops the
gate collapsing onto one expert, and `lambda` controls how decisively it
specialises.

The enumeration point is worth keeping as thesis material regardless of whether
this is built: most treatments of discrete latents reach for Gumbel-Softmax or
REINFORCE, and for `K` of 5-10 neither is needed.

### Why it is deferred, not rejected

**M5's per-band gate is this idea at `K = 2` with the identity as the second
expert:**

```
alpha (m * X) + (1 - alpha) X  ==  [ alpha*m + (1-alpha)*1 ] * X
```

So the base case is already the next planned experiment, at a few tens of
thousands of parameters. **Build that first.** It tests the routing hypothesis at
near-zero cost, and the gate's learned behaviour answers the prerequisite
question: does the model *want* different treatment for different inputs? The
sweep says the optimum varies from alpha = 0 to 1 across difficulty
(`decisions-m3.md` 2026-09-01), so the answer is probably yes — but a trained
gate that saturates at 1 everywhere would say otherwise, cheaply.

**Then scale K.** If the K=2 gate captures a good share of the oracle's 2.2
points, K=3 with shared trunks is the natural follow-up at +11 %.

**The remaining objection is scope, not viability.** Six weeks to freeze, no
live-model measurement yet, and the contribution of this project is the metric.
`K > 2` is a genuinely interesting architecture result and a different thesis
from a new metric — worth stating as further work with the mathematics intact.

Full comparison with the other five options, rendered:
`docs/extra/adaptive-masking-options.pdf`.

### D13. DECIDED 2026-09-01 — build the per-band gate, implemented as the K=2 case of D12

**Decision: the M5 architecture change is the per-band gate, implemented through
a general `n_experts` mechanism in which expert 0 is always the parameter-free
identity mask. `n_experts = 2` IS the gate. `K > 2` is gated behind the judge
work.**

### Why the gate rather than a general two-expert mixture

They are not the same model, and the difference is the point:

| | experts | added parameters |
|---|---|---|
| **the gate** | one learned mask **+ the identity** | tens of thousands (gate only) |
| general MoE at K=2 | **two learned masks** | ~396 k (second mask head) + gate |

The gate's second expert is the identity, which is free. **More importantly it is
the correct prior:** the mix-back sweep measured that easy trials want exactly
`alpha = 0`, i.e. *do nothing* (`decisions-m3.md` 2026-09-01). Hard-coding "do
nothing" as an available option encodes a measured fact. A general two-expert
mixture would have to *discover* that one expert should be near-identity,
spending capacity and training signal on something already known.

### Why the general form is still what gets written

Softmax over two logits **is** a sigmoid:

```
softmax([z0, z1])_1  =  exp(z1) / (exp(z0) + exp(z1))  =  sigmoid(z1 - z0)
```

So with expert 0 fixed as the identity, `n_experts = 2` is *exactly* the gate —
same model, same parameter count, one code path. The generalisation therefore
costs nothing today and `n_experts` becomes the ablation axis:

```
n_experts = 1   ->  the current model, bit for bit
n_experts = 2   ->  identity + learned mask   = the per-band gate
n_experts = 3   ->  identity + two learned masks   (+11 %, D12)
```

### Sequence, and the condition on going further

1. Implement the general form, expert 0 the identity, `n_experts` in config.
2. **Verify `n_experts = 1` reproduces the current model bit for bit on a fixed
   crop.** Without this every later comparison measures the implementation rather
   than the idea. Same requirement as the `BETA` arm.
3. Train `n_experts = 2`, one arm, ~6.2 h. Compare against the oracle's **56.9 %**
   to state how much of the available 2.2 points was captured.
4. **Inspect the learned gate against trial difficulty.** Does `alpha` actually
   fall on easy trials? That is the hypothesis. **A gate that saturates near 1
   everywhere is a cheap negative result** and would rule out `K > 2` before any
   money is spent on it.
5. **`K > 2` is not to be touched until the judge work is done**, regardless of
   how well `n_experts = 2` goes. Six weeks to freeze and no live-model
   measurement exists; writing the machinery makes running `K = 5` tempting and
   that temptation is the risk this clause exists to block.

### On the architecture freeze

This unfreezes the 2026-08-28 architecture, but additively: `n_experts = 1` is
the current model exactly, so the baseline stays recoverable and every previous
run stays comparable. That is the only sense of the freeze that matters.

Rendered comparison of all six options:
`docs/extra/adaptive-masking-options.pdf`.

### D14. Per-frame speaker-state supervision — labels, an auxiliary head, and a frozen state detector as a loss

**Status: PROPOSAL, raised 2026-09-08 (Grant). M5-scale. Only PART of it fits
before the 14 Oct freeze — see Sequence. The three pieces are separable and must
NOT be run as one arm.**

**The gap it addresses.** The model has no per-frame representation of who is
speaking. The only present/absent signal anywhere in the system is `crop_absent`
— one bit per 4 s crop. Every level failure in M2 traces back to this: the
2026-08-25 mute, and the 2026-08-27 diagnosis that one shared gain serves both
branches at a cost of +24 dB on absent crops. `L_gain` penalises the symptom;
nothing supplies the missing variable.

**The four states.** target only / interferer only / both / none. Exactly what
B9's mix produces, and only coherent under the two-speaker boundary
(`decisions-m0.md` 2026-08-14) — a third talker would break the state set, which
is one more reason not to add one.

**Labels are free.** `target.wav` and `interferer.wav` are rendered per trial.
Per STFT frame (hop 128 = 8 ms, ~497 per 4 s crop), active/inactive on each stem
gives the state. No re-render, no new data.

**Label from the REVERBERANT stems, not from VAD on the dry sources.** A1 makes
the reference the full reverberant target, so during a decay tail the correct
output *is* the tail and the frame must be labelled target-active. Labelling from
dry voice activity would train the model to gate off exactly the tail the
reference contains. Consequence: `none` is rarer than B9's 25 % trial-level
absent rate suggests, because tails fill the gaps.

### Three separable pieces

| | what changes | inference cost |
|---|---|---|
| **A. auxiliary state head** | **516 params** (corrected 2026-09-11; the "~10 k" first written here was never derived -- `Conv1d(128, 4, 1)` is 4x128 weights + 4 biases), training pressure only; deleted at inference | zero |
| **B. frozen state detector as a loss** | the objective only. **No architecture change** | **zero** |
| **C. state posterior into D13's gate controller** | one term in a line D13 already specifies | ~1 k params |

**A — auxiliary head.** Mean-pool the separator output over the 32 bands ->
`(B, 128, T)` -> 1x1 conv -> 4 logits per frame. Cross-entropy against the
labels. Teaches the separator's features to encode who is talking; changes no
audio, so it is deletable at inference unless C is built.

**B — the frozen detector, and why it is the cleanest arm in Group D.** Train a
small detector once, offline, on stems and mixtures with the same labels, then
FREEZE it. In the loss, score the model's output spectrogram against the *mapped*
state:

    input state      required output state
    target only  ->  target only
    both         ->  target only      <- the interferer must go
    interferer   ->  none
    none         ->  none

Two readouts of one variable: the labels stay honest (what IS happening), the
target is aspirational (what SHOULD be audible). When the interferer is still
audible the frozen detector puts mass on `both`, cross-entropy against
`target only` is large, and the gradient flows back through the estimator saying
"reduce whatever made this frame look like two voices". That is how a
classification becomes a pressure on the audio.

**The freeze is the mechanism, not a detail.** A jointly-trained detector and
extractor share one objective and can satisfy it by agreeing with each other
while the audio does not change. Frozen, the only available move is to change the
audio.

**Zero parameters added at inference.** The extractor is architecturally
identical — same 7.19 M, same band plan, same latency — and only its weights
differ. **This is the only proposal in Group D with no capacity confound**, which
makes its comparison against the baseline unusually clean. Build it non-causal:
it never streams, so future context is free accuracy, the same argument that
makes an enrollment-side encoder latency-free in D5. Build it convolutional, not
recurrent — recurrent activations over a 497-step sequence are what created E1's
batch ceiling and there is no reason to reintroduce that in a scorer. Score the
estimator's spectrogram directly, skipping an iSTFT/STFT round trip in the loss
path.

**A sibling of the planned proxy, not a new category.** CLAUDE.md already commits
to frozen-ASR/SSL feature matching. Cite: perceptual / deep-feature losses,
Johnson et al., ECCV 2016; in speech enhancement, Germain, Chen & Koltun,
"Speech Denoising with Deep Feature Losses", Interspeech 2019. **BORROWED, with a
difference to state:** those losses match features of *what was said*; this
matches *who is audible when*. Complementary axis, and the identity axis is where
the 2026-08-30 gender shortcut hides.

**Constraints.** The detector must not be `small.en` (the eval scorer), must be a
different family from the ASR proxy, and must never be the judge in any form.
Family check recorded, not assumed.

**Reward-model overoptimisation is the failure mode, and it is already familiar.**
A frozen learned scorer optimised against gets gamed off-distribution eventually.
Signature: the term falls while an independent measure stalls — seen twice
(2026-08-25, total loss falling into a mute; 2026-09-04, `enrol_sens` and
`pres_abs_gap` at their best on an epoch below pass-through). Mitigations, all
cheap: never select on the term, keep its weight modest, and **measure the
detector's accuracy on the extractor's own outputs against true labels every
epoch** — we own the stems for every output, so calibration drift is directly
observable. If it drifts, re-fit the detector on model outputs *with true labels*
(distribution repair, not syncing to the model's preference). Post-freeze.

**C — the state posterior as D13's gate controller.** D13 already specifies
`alpha[t,b] = sigmoid(w_b . h[t,b] + b_b)`: an implicit, unsupervised controller
learning from the extraction loss alone. Add the posterior as an extra input:

    alpha[t,b] = sigmoid( w_b . h[t,b] + u_b . p[t] + b_b )

**Augment, do not replace.** `p[t]` is one vector per frame shared across bands;
using it alone would discard the per-band expressiveness M5 chose deliberately
(artefacts in high bands, speech energy in low). ~1 k extra parameters.

**It attacks D13's own stated failure mode.** D13 step 4 fears "a gate that
saturates near 1 everywhere". An unsupervised scalar has one weak signal to learn
from and saturation is the safe answer; supervising the intermediate gives it a
strong gradient and a reason to vary. It also makes the gate diagnosable — state
accuracy is directly measurable, where an implicit gate that fails explains
nothing.

**Detach the posterior before the gate, as an ablation arm.** Otherwise head A
gets a second gradient from the extraction loss, which may prefer it to always
claim `both` so the mask always applies — the same collusion as B, internalised.
Detached, A is trained only by honest cross-entropy and the gate learns to use it.

### The recorded 2.2-point ceiling does not apply to a per-frame gate

M6 warns that D13's oracle ceiling is 2.2 LCF-WER points (59.1 -> 56.9) against a
judge SEM of ~0.5, so a realistic gate may not clear the noise floor. **That
ceiling was measured by bucketing on trial difficulty, one constant per trial**
(`decisions-m3.md` 2026-09-01). A per-frame gate is strictly more expressive than
a best-constant-per-trial gate, so the relevant bound sits above the cheating
per-trial oracle's 6.6 points, not below 2.2. **Nobody has measured a per-frame
oracle.** That is a gap in the log and closing it needs no training.

### Prior art — the state set is published; the composition may not be

- Frame-level {non-speech, target, non-target} conditioned on a speaker embedding
  is **personal VAD**: Ding et al., ICASSP 2020. Three of the four states.
- Per-frame per-speaker activity from speaker profiles is **TS-VAD**: Medennikov
  et al., Interspeech 2020, the standard diarization approach.
- HMM and factorial-HMM separation: Roweis, 2000; Mysore & Smaragdis, 2010-11.
  Hybrid HMM-DNN is the 2012-16 ASR era.
- **A temporal smoother over the posterior was considered and dropped.** A
  learned transition matrix with causal filtering only (never forward-backward,
  which needs the future) is cheap and optional — but a learned duration prior
  would encode our sampler's `overlap_ratio` distribution rather than real
  turn-taking, because LibriSpeech reads have none. It would look good on
  `sir0_val` and is a prime candidate to fail the AMI transfer check.
- **The GMM-UBM contrast** (target vs a speaker-independent average; Reynolds et
  al., 2000) resurfaced here as a candidate observation channel and is **not
  revived**: D3a closed D1, and a static corpus-average channel would repeat the
  measured 2026-08-25 failure where the cue degenerated into the long-term mean
  spectrum, varied 4.7 %, and was ignored. See
  `literature/novelty-review-contrastive-phonetic-cue.md`.

**Requires a novelty review before any of B is written**, as D1 got. The claim
cannot be the state set or the HMM framing; at most it is the composition, plus
tuning its operating point against a live-model content metric rather than SI-SDR
or diarization error.

### MEASURED 2026-09-10 — WavLM cannot do this job. Backbone switched to ECAPA-TDNN

**A frozen general-purpose SSL encoder does not carry frame-level speaker
identity at 0 dB interference. WavLM Base+ is therefore not usable as the
teacher's backbone, and this is a reportable negative result, not just a
setback.**

### What was built and what it scored

Frozen WavLM Base+ (torchaudio, sha256 `1697ecbb...`), layer 8, plus a 594 k
`StateHead` reading `[frames, enrolment, frames * enrolment]`. Cache: 500
`sir0_train` trials x 6 gain variants, 600 k labelled frames, ~1 h. Trained 12
epochs on CPU in ~3 min.

| | recall |
|---|---|
| `none` (silence) | **0.934**, stable every epoch |
| `target` | 0.26-0.74, unstable |
| `interferer` | 0.17-0.52, unstable |
| `both` | 0.35-0.61 |

**Activity detection works; identity does not.** `target` and `interferer`
recall move inversely with a near-constant sum (~0.8), which is the signature of
a model that detects "one voice" and then guesses which. Balanced recall was
flat across the run: 0.557 -> 0.578 -> 0.573.

**Balanced identity accuracy on single-speaker frames: 0.570 against 0.500
chance.** Do NOT quote the raw 0.584 -- the split is 2.4:1 target-to-interferer,
so always answering "target" scores 0.722, and the raw figure is worse than a
constant predictor.

### Three parameter-free tests, and they rule out every cheap fix

Run on cached features, no training. AUC over 14,406 single-speaker frames from
130 trials; 0.500 is a coin flip.

| scoring | AUC |
|---|---|
| pooled enrolment (what was trained) | 0.521 |
| attention over 249 enrolment frames, best temperature | 0.535 |
| contrastive: attention minus a 500-speaker background, best | **0.547** |

1. **The layer is not the problem.** All 12 layers score 0.700-0.800 on a
   per-trial variant of the test (n=40, SE +-0.068 -- the whole spread is noise).
   Layer 5 is nominally best by 3 trials. No re-cache is justified.
2. **Mean pooling destroys identity, but removing it does not save it.**
   Pooled enrolment vectors from 40 DIFFERENT speakers are 64-90 % similar at
   every layer. Attention (the K>1 generalisation of pooling, and D2's TF-Map
   mechanism applied to SSL features) buys 1.4 AUC points.
3. **The contrastive term works in the right direction and is far too weak.**
   +1.2 points at low temperature, and it goes BELOW chance (0.494) as the
   temperature sharpens -- sharper matching matches phonetic content harder,
   confirming D1's confound is real and that the contrast cannot cancel it here.

**Correction to a number quoted earlier in this entry's working:** a per-trial
test gave 0.725 and was misleading -- it averaged hundreds of frames per side.
Per frame, which is what the teacher must do, the ceiling is AUC 0.55.

### Why, and what it implies

WavLM's pretraining objective is masked prediction, which rewards phonetic
content. Layer 8 is mid-stack and ASR-oriented. Speaker identity is present but
not linearly separable per frame under 0 dB interference. The head reached
roughly what AUC 0.547 permits, so **the bottleneck was the representation, not
the classifier.**

**Reportable finding, independent of whether the teacher is ever built:** a
general-purpose speech representation does not supply frame-level speaker
identity in two-speaker mixtures at 0 dB, which is a measured argument for why
TSE needs purpose-built speaker models rather than SSL features. It also
retrospectively supports `decisions-m1.md` 2026-08-19's choice of the spectral
TF-Map over an SSL embedding path.

### Decision

**Switch the backbone to ECAPA-TDNN** (Desplanques et al., Interspeech 2020),
SpeechBrain's VoxCeleb-trained `spkrec-ecapa-voxceleb`, snapshotted to
`../ecapa_pretrained/` with hashes pinned. Verification training rewards exactly
the discrimination masked prediction does not.

**Rejected, both on independence grounds:** WeSep's checkpoint carries 512 ECAPA
tensors but they were jointly trained with its separator, and WeSep is this
project's comparison baseline -- a teacher derived from it would couple the
system under test to the system it is measured against. `wespeaker` (installed
in `../wesep_venv`) is the same toolkit family.

**Consequence to carry: resolution drops from 20 ms to ~0.5-1 s.** ECAPA emits
one embedding per window, not one vector per frame. Acceptable for gating,
and it must be stated wherever the teacher is described as "per-frame".

**HARD GATE before any rewrite.** Re-run the same AUC test on ECAPA embeddings
at 0.5 / 1 / 2 s windows. **Continue only above ~0.75.** If ECAPA also lands in
the 0.50s, frame-level identity is not recoverable from off-the-shelf models at
this difficulty, and the teacher is dropped in favour of `BETA` (M5's artefact
weight) or D10 -- one number each, neither depending on identity being
recoverable.

### Artefacts deleted 2026-09-10

`../wavlm_pretrained/` (361 MB), `../state_detector_cache/` (1.1 GB),
`scripts/upload_kaggle_wavlm.py`, `models/state_detector_notebook.pt`. All
regenerable; the numbers above are the deliverable. The label pipeline
(`src/data/state_labels.py`, `scripts/build_state_labels.py`,
`data/index/state_*.csv`) is backbone-independent and is KEPT.

### MEASURED 2026-09-10 — the teacher WORKS on ECAPA. Two runs, and the output shape moved identity but not overlap detection

**Frozen ECAPA-TDNN + a 150 k head tells the two speakers apart 94.0 % of the
time. WavLM managed 57.0 %. The teacher is viable.** What it does NOT do
reliably is detect a second voice: 76.2 %, flat across both output
parameterisations.

### The gate that authorised the switch

Before any head was built, plain cosine similarity between a window embedding
and the enrolment embedding, no training at all, `sir0_train`, 120 trials:

| window | windows | AUC | cos own speaker | cos other | margin |
|---|---|---|---|---|---|
| 0.5 s | 1,113 | 0.866 | +0.203 | +0.029 | +0.174 |
| **1.0 s** | 666 | **0.957** | +0.324 | +0.045 | +0.279 |
| 2.0 s | 241 | 0.992 | +0.443 | +0.057 | +0.387 |

**1.0 s adopted.** Longer is more accurate and coarser; this is the knee.
**Consequence to carry into every description: the teacher's resolution is ~1
SECOND, not per-frame.** It can say "the target is speaking around here"; it
cannot mark a word boundary.

**A1's dry-enrolment/reverberant-mixture gap is NOT a problem here.** Similarity
to the *other* speaker sits at +0.03 to +0.06 while own-speaker rises to +0.44.
A channel mismatch would depress both together. Retires a suspect.

### Run 1 — four-way softmax. 500 trials cached, 425 trained, epoch 3 of 20

| | all windows | pure only |
|---|---|---|
| identity (which speaker) | **0.911** | 0.920 |
| is the target audible | 0.857 | 0.882 |
| is a non-target audible | 0.767 | 0.785 |
| `both` as a 4-way label | 0.369 | 0.391 |

Identity is symmetric (target 0.904, interferer 0.918) and always answering
"target" would score 0.644, so the figure is real. **Transitions cost only ~2
points** (0.911 vs 0.920), which settles the window length: 1 s was right, and
shrinking to 0.5 s for purity would have cost more identity than it gained.

**Measured purity distribution** (150 trials, 11,700 windows): mean 0.937,
median 1.000, **74.0 % perfectly pure**, 85.7 % at or above the 0.8 training
threshold, ambiguous (tied majority) 0.02 %. Ties are real but negligible;
handled by exclusion rather than argued away.

### The finding that motivated run 2

**`both` scored 0.369 as a 4-way class while THE SAME PREDICTIONS gave 0.857 and
0.767 on the two questions `both` is the conjunction of.** The four states were
always a pair of bits (`sl.states_from_masks`: target = bit 0, interferer = bit
1), so a four-way softmax forced a commitment to one category and hid what the
head knew.

### Run 2 — two independent binary outputs. Same cache, epoch 2 of 20

| | 4-way | binary | change |
|---|---|---|---|
| identity | 0.911 | **0.940** | **+2.9** |
| target audible | 0.857 | 0.860 | flat |
| **non-target audible** | 0.767 | **0.762** | **flat** |
| `both` recall (reporting only) | 0.369 | 0.416 | +4.7 |

**The reframe bought identity, not overlap detection.** The prediction was
"modest gain, watch the non-target question"; that question did not move.
**So `both` was hiding information about IDENTITY, not about second-voice
detection**, and the non-target limit is data or information rather than output
shape.

**Identity 0.940 now exceeds the untrained cosine reference** (AUC 0.951,
roughly 0.88-0.90 balanced at its best threshold), so the head adds to identity
rather than passing it through. Bases differ — indicative, not an exact
comparison.

**Do NOT read the non-target recall gain as progress.** Recall rose 0.729 →
0.756 while specificity fell 0.806 → 0.767: the operating point sliding, with
balanced accuracy unchanged.

### What 0.762 costs the loss, stated plainly

Misses 24 % of genuine leakage; false-alarms on 23 % of already-clean windows.
**A noisy teacher, directionally right — never to be described as a detector.**
The gradient is useful because it averages over many windows and many steps.

### Two corrections made in the course of this work

1. **I claimed ECAPA had a structural ceiling on second-voice detection**, from
   plain cosine scoring 0.405 (below chance, because one scalar cannot separate
   "target" from "target plus someone else"). Wrong: the head reaches 0.762-0.767
   from the full 192-d embedding. The information is there; cosine discards it.
   **That question is where the head earns its keep** — everything else it
   roughly inherits from cosine.
2. **The diagnostics were reporting the LAST epoch, not the selected one.** Cell
   25 evaluated whatever was in memory after the loop. On run 1 that was epoch 20
   against a selected epoch 3, understating identity 0.854 vs 0.911 and `both`
   0.246 vs 0.369. Now reloads the checkpoint and asserts it reproduces the
   recorded score.

### Both runs are badly data-limited

Best epoch **3 of 20** then **2 of 20**, on 425 training trials — memorising
almost immediately, with 150 k parameters against ~33,000 windows. Train loss
fell 12x while held-out loss rose 65 %.

**Actions taken 2026-09-10:** cache extended 500 → 2,000 trials (70/30
rich-to-random held fixed so the class balance does not move with the data
volume; ~4.8 h at a measured 11.4 s/trial, ~65 MB). And a threshold sweep, since
every number above is at 0.5 for both questions and AUC will say whether that is
simply the wrong place to stand.

**The two questions must NOT share an operating point, and neither should be
tuned for balanced accuracy.** Target-audible drives "do not mute my speaker" —
a miss deletes speech the judge never hears, so favour recall. Non-target-audible
drives "remove the interferer" — a false alarm penalises output that was already
clean and fights the signal-domain terms directly, so favour specificity.
Thresholds are reporting and inference choices; the loss uses raw probabilities,
so tuning them needs no retraining.

### Costs, for the record

| | |
|---|---|
| trial selection | quota, not a sort: 5,025 of 9,955 trials have NO overlapping frames, so random sampling starves `both`; sorting by overlap overshoots to 69.6 % `both` |
| cache, 500 trials | 95 min, 16 MB (WavLM's was 881 MB for the same trials) |
| head training | ~3 min on CPU, 4 threads |
| variant recipe | 3 fixed + 3 partial; an earlier 4-fixed recipe drove `both` from 17.0 % of frames to 7.5 % |

`mixture == target + interferer + noise` verified exact 2026-09-10, so any
suppression level is synthesisable from the stems.

### 2026-09-10 — 1,000 trials, and the enrolment bank turned on for the teacher

**Doubling the data bought ~2 points on the hard question and nothing on the
easy one. The head still peaks at epoch 2 of 20, so the limit is memorisation,
not sample size — and the fix already exists in this repo.**

### The data-scaling point

| | 500 trials (428 fit) | 1,000 trials (850 fit) |
|---|---|---|
| target audible, AUC | 0.944 | 0.938 - 0.947 |
| **non-target audible, AUC** | **0.795** | **0.812 - 0.820** |

Two measurements of the 1,000-trial model on different splits, so the range is
the honest form. **Identity was already saturated**; second-voice detection
improved ~2 points, consistent with roughly +2 per doubling, which would predict
~0.835 at 2,000 trials for a further 3.3 h of caching. Not yet spent.

Class balance at 1,000 trials, 78,000 windows: target audible 54.8 %
(`pos_weight` 0.82), non-target audible 33.3 % (`pos_weight` 2.01). Window
purity mean 0.965 / 0.977, with 92 % / 95 % at or above the 0.8 training
threshold.

### The threshold question, and a correction

**Reported thresholds returned to 0.50 / 0.50.** They had been set to
0.50 / 0.65 on the argument that the non-target question should favour
specificity, because a false alarm penalises output that was already clean.

**That argument was wrong for this term.** The loss uses raw probabilities --
BCE against a constant 0 for the non-target column -- so it never thresholds.
The threshold affects REPORTING and any hard inference-time decision, and
nothing else. Reporting at 0.65 merely made the number look worse: 0.724
against 0.733.

**The sweep also shows 0.5 is essentially optimal anyway.** Best balanced
accuracy is 0.866 at threshold 0.60 for the target question (0.862 at 0.50) and
0.734 at 0.55 for the non-target (0.733 at 0.50). **Gains of +0.004 and +0.002:
the operating point is not what is limiting this.**

**Also corrected: an invalid diagnostic of my own.** A "headroom" column
computed as AUC minus balanced-accuracy-at-0.5 is meaningless -- the two are
different scales. Replaced with the best balanced accuracy found by sweeping,
minus what 0.5 gives, which is the only honest form. AUC remains the number to
quote as a question's ceiling because it is threshold-free.

### DECIDED: rotate the enrolment per epoch (bank K=3)

**The head peaks at epoch 2 of 20 at 425, 428 AND 850 training trials.**
Training loss falls 0.473 -> 0.057 while held-out rises 0.464 -> 0.769. Data
volume does not move the peak, so the head is memorising something that more
trials do not dilute.

**The enrolment is the route.** One fixed 192-d vector per trial, seen 78 times
per epoch (6 variants x 13 windows). At 850 trials that is 850 vectors to
memorise, and 149 k parameters against ~10,000 semi-independent windows is
about 15 parameters per effective sample.

**This project has already had and fixed this exact failure.** The extractor's
identity cue "stayed a fixed waveform across all 24 epochs and was memorisable"
(`decisions-m2.md` 2026-08-30), and `scripts/render_enrollment_bank.py` is the
fix built for it. K=3 is **already rendered on disk** for `sir0_train` as
`enrollment_v00/01/02.wav` -- different sentences, offsets and EQ per variant.

**Cost is ~2 extra ECAPA forwards per trial**, because `v00` is byte-identical
to `enrollment.wav` (verified 2026-09-10) so the cached embedding IS variant 0.
Roughly 15 min to patch 1,000 existing cache files, and no re-render.

**Rotation is TRAINING-only.** The holdout keeps one fixed enrolment so its
curve stays comparable epoch to epoch and run to run -- the same rule
`dataset_loader.py` enforces by forcing `enrollment_variants` to 1 when
`random_crop` is off. A short bank falls back DOWN to v00 rather than failing,
matching `_enrollment_path`.

`ROTATE_ENROLMENT` is an ablation flag: `False` reproduces every run before
2026-09-10 exactly. **The arm to report is 1,000 trials with one enrolment
versus three, everything else held.** If the best epoch moves from 2 to 6-8,
the memorisation diagnosis is confirmed and the head has real headroom.

### Sequencing: the hyperparameter search was STOPPED and deferred

Optuna (`optuna==5.0.0`, pinned) is set up over head width and depth, dropout,
learning rate, weight decay, batch size and `min_purity`, on a THREE-way split
(700 fit / 150 search / 150 report) with a threshold-free mean-AUC objective and
median pruning. `AUDIBLE_DB` is deliberately excluded from the search: it defines
what counts as a second voice, so tuning it would optimise the question rather
than the answer.

**It was killed after one trial.** Reason: if the bank removes the
memorisation, the settings the search is currently finding -- heavier dropout,
heavier weight decay, early stopping -- are tuned to compensate for a failure
about to be removed, and none would be the right values afterwards. The search
runs AFTER the bank arm, on the final data setup.

The threshold being wrong was NOT a reason to stop it: the objective is mean
AUC, which is threshold-free by construction.

### Not yet done

- **`sir0_val` is still uncached.** All splits above come from `sir0_train` and
  share speakers. The teacher is not validated until it is measured on a
  speaker-disjoint set.
- **GPU cost of the term is unmeasured**, and it gates the whole integration.
  13 windows x 6 examples = 78 ECAPA forwards AND backwards per step against a
  current 0.674 s/step. The 78 ms/window figure is CPU; training is a T4.
  `scripts/profile_step.py` is where that number comes from, and no more
  integration code should be written before it exists.
- Rows for `docs/run_times.md`: the 1,000-trial cache (~68 min for 350 new
  trials) and the rate difference that made it, 12.1 s/trial clean against
  44.9 s/trial while a browser and an IDE competed for the same 8 cores.

### 2026-09-11 — the teacher is FINISHED at AUC 0.87, and the integration is built

**A sequence model over the windows was worth ten times everything else tried.
The teacher is frozen at that point and no further work on it is planned.** The
loss term, its subclass, its tests, its weight-derivation script and its config
are written; the one thing that can still stop the arm is unmeasured.

### The architecture change, which is the result

Same features, same split, same threshold-free objective. Only the head differs:

| head | params | mean AUC | target | **non-target** |
|---|---|---|---|---|
| per-window MLP | 149 k | 0.8836 | 0.9473 | **0.8199** |
| **BiLSTM** | **339 k** | **0.9216** | **0.9721** | **0.8710** |
| self-attention | 473 k | 0.9170 | 0.9669 | 0.8672 |

Against every other intervention on the same question:

| | gain |
|---|---|
| 500 -> 1,000 trials | +1.7 |
| enrolment bank, K=3 | +0.8 |
| 42-trial hyperparameter search | +0.7 |
| **sequence model over the 13 windows** | **+5.1** |

**Why it works.** A single window at cosine +0.23 to the enrolment is genuinely
ambiguous between "the target alone, quieter" and "the target plus someone
else": `both` sits at +0.233 between `target` at +0.321 and `interferer` at
+0.044, because a verification embedding describes whichever voice dominates.
The neighbouring windows resolve much of that.

**BiLSTM over attention:** better (0.9216 vs 0.9170), 30 % fewer parameters,
converged at epoch 1 rather than 7.

**CAVEAT TO CARRY INTO THE WRITE-UP.** Part of the gain is comparison across
windows and part is temporal SMOOTHING, since speaker states come in runs of
hundreds of milliseconds. The prior is legitimate, but it means **an isolated
window of leakage may be smoothed away and go unpenalised** -- the teacher is
better at sustained leakage than at brief leakage.

### Final teacher, and the numbers the loss depends on

1,000 `sir0_train` trials (850 fit / 150 holdout), BiLSTM hidden 128 / 1 layer,
enrolment bank K=3 rotated per epoch, epoch 3, seed 42, audibility -20 dB.
sha256 `16f6d8a43d5ecacf24e1...`.

| | balanced | recall | specificity | precision |
|---|---|---|---|---|
| is the target audible | 0.906 | 0.908 | 0.903 | 0.914 |
| **is a non-target audible** | **0.802** | **0.795** | **0.808** | **0.679** |

identity on single-speaker windows **0.942** (WavLM managed 0.570).

**The arm uses the NON-TARGET column only**, which is the worse one. Not a
choice: its required answer is a constant zero, so it needs no labels. The
target column's required answer depends on whether the target was speaking in
that window, which needs per-window labels in the loader -- the second
increment.

### Three interventions that did NOT work, recorded so they are not retried

1. **Enrolment bank (K=3, rotated per epoch).** PREDICTED the best epoch would
   move from 2 to 6-8 if the enrolment was the memorisation route. **It stayed
   at exactly 2.** +0.008 AUC. The prediction was wrong and the diagnosis with
   it.
2. **Doubling the data, 500 -> 1,000 trials.** +0.017 on the hard question,
   nothing on the easy one. Best epoch still 2.
3. **42 Optuna trials over the MLP head.** 0.8836 -> 0.8907, and the whole run
   was wasted anyway: it tuned an architecture the BiLSTM had already replaced,
   0.7 points of search against 3.8 points of architecture. `SearchableHead`
   now carries a comment recording that so it is not repeated.

**The pattern underneath all three:** the head converges in ONE epoch and then
degrades. It is not overfitting so much as exhausting what the features contain,
which is why data, regularisation and hyperparameters all bought under a point
each and the architecture bought five.

**A false negative I produced along the way, worth recording as a method
lesson.** I tested "does temporal context help" with hand-made scalar summaries
of neighbours' cosine -- own-minus-local-max scored 0.509, i.e. nothing -- and
reported the question settled. A learned model over the full 192-d sequence then
gained 5.1 points. **A cheap proxy test can only rule something out when the
proxy is as expressive as the thing it stands in for.**

### The integration, and what it deliberately does not touch

| file | |
|---|---|
| `src/models/state_teacher.py` | new. Frozen ECAPA + BiLSTM readout, hash-checked, shape inferred from the state dict because the checkpoint does not record it |
| `src/models/losses_state.py` | new. `LossBSRNNState(LossBSRNN)` -- a SUBCLASS in its own file, so `losses.py` is untouched |
| `scripts/derive_w_state.py` | new. Anchor measurement and weight derivation |
| `tests/test_losses_state.py` | new. 8 tests |
| `experiments/configs/bsrnn_state.yaml` | new. Four keys differ from the baseline, verified |
| `scripts/train.py` | five changes |

**`build_loss_fn` BRANCHES on `w_state` rather than multiplying by zero.** A
config without the key gets the plain `LossBSRNN` object, so every run before
today reproduces by construction and no old config can reach the new code path
at all. Verified: the baseline config still builds `LossBSRNN`.

**`total` deliberately EXCLUDES `L_state`.** It is what `ReduceLROnPlateau` and
the loss curve read, and the four terms are what it has always meant; a fifth
would make every run before 2026-09-11 incomparable with every run after.
`L_state` gets its own column, NaN when the term is off -- a missing term is a
gap in the curve, not a zero.

**`selection_score` includes it in no mode, and that is the point.** A frozen
learned scorer can be improved by finding its blind spots rather than by
removing the interferer, and the teacher has never heard masked audio -- only
real mixtures and synthetic suppressions. Selecting on it would make that
invisible, because the quantity being gamed would also be the quantity choosing
the checkpoint. **The signature to watch is `L_state` falling while `L_pres`
stalls**, and both now print side by side in the epoch breakdown.

### The test that matters, and the one I mislabelled

`test_the_state_term_is_what_carries_the_gradient` is the test with teeth.
Proven by deliberately introducing the bug: with the teacher's forward wrapped
in `no_grad`, `|grad(w_state=1) - grad(w_state=0)|` goes from **182.17 to
exactly 0.000000**.

**`test_gradient_reaches_the_waveform`, which I had called "THE test", still
PASSES under that bug** -- the four base terms deliver gradient regardless, so
the waveform's gradient only moves 26874.5 -> 26836.6, a 0.14 % change. "Gradient
is non-zero" is not a test of this term.

**And `L_state` logs 0.1048 either way.** The number in the training curve is
identical whether the term works or not. That is the failure mode, confirmed
rather than argued -- the same dead-gradient shape D9 already records for the
original "add WER to the loss" idea.

### w_state must be DERIVED, and the first attempt found two bugs

**MEASURED: at `w_state = 1.0` the term contributed 0.68 % of the gradient on
synthetic audio and 5571 % on real audio.** Two orders of magnitude in one
direction and three in the other, so ANY guessed value would be wrong. L_state
is a cross-entropy in nats; the others are dB. There is no meaningful
conversion -- BCE is a log-probability, dB is a power ratio -- and `L_MR` is the
standing precedent that a term need not be in dB provided its weight reconciles
the units.

**Anchors, 6 batches of `sir0_val`:**

| anchor | L_state |
|---|---|
| oracle (clean target) | **0.1212** |
| model (`model_sir0_10000-e6.pt`) | 1.8200 |
| passthrough (the mixture) | **2.1064** |

**Headroom 1.99 nats**, and the wiring check passes: the clean target contains
no second voice and reads near zero. **The checkpoint has removed only ~15 % of
the teacher's leakage reading**, so there is room for the term to push into.

**Two bugs the first run exposed, both fixed:**

1. **`remix_gains: true` invalidated every synthetic anchor.** The loader
   rebuilds the mixture at a fresh SIR each epoch, so what it returns is NOT
   `target + interferer + noise` from disk -- the residual measured 0.2x to 1.7x
   the mixture's own level. The `partial_6/12/20db` anchors read as WORSE than
   the raw mixture on `L_pres`, `L_gain` and `L_abs`, which is what gave it
   away. The script now forces `remix_gains=False`.
2. **The suggested weight printed as `0.0`** because `%.1f` was applied to
   0.0027. It now prints `%.4g` and, more importantly, RE-MEASURES the share at
   the suggested value instead of trusting a linear extrapolation from a 5571 %
   measurement -- which is nowhere near the small-perturbation regime that
   extrapolation assumes.

### The one thing that can still stop this, and it is unmeasured

**GPU cost per step.** 13 windows x 6 examples is 78 ECAPA forwards AND
backwards every step, against a current 0.674 s/step. Every timing in this
entry is CPU; training is a T4. `scripts/profile_step.py` is where that number
comes from, and **no further integration code should be written before it
exists** -- if it says 5 s/step the term must score a random subset of windows
instead of all 13, which changes the shape of `losses_state.py`.

**Levers, and the first one is less available than it looks.**

**Subsampling windows is the SGD argument, and it is sound in principle.** The
term is a mean over 13 windows, so scoring a random subset is an unbiased
estimate of it -- the same reason a mini-batch estimate of the full-dataset
gradient works, and the reason even a single sample converges given enough
steps. Variance costs steps, not correctness.

**But the BiLSTM couples the windows, so random subsampling saves almost
nothing.** The head runs bidirectionally over the whole 13-window sequence, so:

- computing the loss on 4 windows still needs all 13 ECAPA embeddings to feed
  the recurrence, so the expensive forward is unchanged;
- and the backward flows through the recurrence to all 13 hidden states anyway,
  so the backward is unchanged too.

**What DOES save time is scoring a shorter CONTIGUOUS segment** -- 2.5 s of the
4.008 s chunk is 7 windows instead of 13, and the recurrence runs over 7. The
cost is that the head then sees a shorter context than the 13 windows it was
fitted on, which is a mild distribution shift and must be measured rather than
assumed.

So the levers, in order: **score a shorter contiguous segment**; put the teacher
on the second T4; precompute enrolment embeddings (worth ~5 s of the ~18 s of
audio per example, so a real saving but not the main one -- an earlier claim
that it was the larger half was true only for a single-window design).

**AND THE NUMBER 4 WAS INVENTED.** An earlier note in this conversation
suggested "score 4 of 13 random windows" as though it were derived. It was not.
The only non-arbitrary anchor available is that a 1 s window at 0.25 s hop means
about **4 NON-OVERLAPPING windows span the 4 s chunk**, so 13 windows carry
nowhere near 13 windows of independent information -- but that is an argument
about redundancy, not a derivation of a subsample size. The segment length
should come from the profile: the longest one that fits the session cap.

### Still not done

- **`sir0_val` is uncached**, so the teacher has never been measured on
  speaker-disjoint data. All 1,000 trials come from `sir0_train`.
- The target column, which needs per-window labels in `dataset_loader.py`.
- Head A, which is a separate idea and shares those labels.

### MEASURED 2026-09-11 on a T4 — the arm is affordable, and the lever is EPOCHS not the teacher

**The full-fidelity teacher costs +53 % per step. Shortening the scored segment
does NOT buy the same measurement more cheaply -- `L_state` moves 4.1x across
segment lengths -- so the epoch count gives way instead. 13 windows at 10 epochs
lands at 10.0 h against a 12 h cap.**

`experiments/results/2026-09-11-state-teacher-cost/`, Tesla T4, batch 3 trials =
6 examples, AMP on, synthetic audio.

| windows | s/step | overhead | 16-ep hours | peak GB | L_state |
|---|---|---|---|---|---|
| 0 (baseline) | 0.669 | — | 10.5 | 6.59 | — |
| 1 | 0.743 | +11 % | 11.7 | 6.67 | 0.2542 |
| 3 | 0.780 | +17 % | 12.2 | 6.67 | 0.1385 |
| 5 | 0.822 | +23 % | 12.9 | 6.73 | 0.1038 |
| 7 | 0.870 | +30 % | 13.7 | 7.14 | 0.0892 |
| 9 | 0.926 | +38 % | 14.5 | 7.55 | 0.0742 |
| **13 (full)** | **1.022** | **+53 %** | **16.0** | **8.37** | **0.0623** |

**The profile is trustworthy because the baseline reproduces.** 0.669 s/step
against the recorded 0.674 (`decisions-m2.md` 2026-08-28).

### The finding: subsampling is not a saving

**`L_state` runs 0.2542 at one window to 0.0623 at thirteen -- a factor of
4.1.** A shorter segment gives the BiLSTM readout less context than the 13
windows it was fitted on, so it is a DIFFERENT measurement, not a cheaper one.
`w_state` would need re-deriving at every length, and the numbers would not be
comparable across arms.

That retires the lever I had been planning on. **The mini-batch argument for
subsampling was sound in principle and inapplicable in practice**: the term is a
mean over windows, so a random subset would be unbiased -- but the head is a
BiLSTM over the sequence, so a loss on 4 windows still needs all 13 embeddings
to feed the recurrence and the backward flows through it to all 13 regardless.
Only a shorter CONTIGUOUS segment shortens both, and that is what drifts.

**Caveat on the 4.1x.** The sweep ran on synthetic noise, which contains no
interferer, so `L_state` there measures how confidently the teacher says "no
second voice" -- and more context makes it more confident, which is the right
direction. The EXISTENCE of drift is the finding; the magnitude is specific to
this input and should not be quoted as a general property.

### DECIDED: 13 windows, 10 epochs

**Full-fidelity teacher, no drift, no re-derivation.** 10.0 h with two hours
spare.

**10 epochs costs nothing that was being used.** The 2026-09-04 run selected
epoch 6 of 16; the run before it selected epoch 2 of 20. The back half has never
produced a checkpoint. Early stopping still applies, so it is a ceiling and not
a target.

**Memory does not bind: 8.37 GB of 14.6.** So the teacher goes on the same card
and `state_device: cuda`. The second T4 buys nothing here and is better spent on
E7's data parallelism if that lands.

### w_state = 0.002692, derived and verified

`scripts/derive_w_state.py`, `experiments/results/2026-09-11-wstate-anchor-sir0`.
**Verified, not extrapolated:** at that value the term contributes **15.00 %** of
the gradient reaching the waveform, measured with two backward passes at the
suggested weight rather than scaled from a measurement elsewhere.

**At w_state = 1.0 it contributes 5571 % on real audio -- 370x too strong.** Any
guessed value would have been wrong by orders of magnitude, and a run at 1.0
would have looked like the idea failing when it was the arithmetic.

**15 % is a judgement, not a derivation**, and is recorded as one: four existing
terms share the update, so an equal share is ~25 %, and a new unvalidated proxy
should get less than an equal say. The script tabulates 5 % to 30 %. If the arm
shows nothing at 15 %, the next question is 30 %, not whether the idea failed.

### Three bugs the profiling found, all fixed

1. **The profiler ran fp32 while training runs AMP.** Baseline came back at
   4.765 s/step and 12.23 GB against the recorded 0.674 -- 7x slow. Every
   overhead percentage would have been divided by a baseline seven times too
   large, making the teacher look seven times cheaper than it is. It now mirrors
   `train.py`: `autocast` on the model forward only, `GradScaler`, loss in fp32,
   and it PRINTS the recorded 0.674 beside the measured baseline so a
   misconfigured profile is visible before the sweep rather than after.
2. **`cudnn RNN backward can only be called in training mode`.** The teacher's
   BiLSTM was in `eval()`, and cuDNN refuses RNN backward there -- but gradients
   must flow through the recurrence to reach the audio. The head now runs in
   `train()` mode, which is numerically free ONLY because dropout is forced to
   0.0 at construction: `Dropout(0.0)` is the identity in both modes and
   LayerNorm is mode-independent. An assertion now fails loudly if a head with
   nonzero dropout is ever loaded, since a stochastic teacher would make the
   extractor chase a moving target.
3. **Two Kaggle runs produced the same traceback from an already-fixed file.**
   The notebook's copy cell skipped files that already existed, and
   `/kaggle/working` persists across runs in a session -- so it kept the
   previous run's code after the dataset was updated. The line numbers in the
   traceback were the only evidence. It now overwrites, and asserts the presence
   of two fixed lines before running anything.

### And one that was not a bug in the code

**`../ecapa_pretrained/` is 24 KB of SYMLINKS into the HuggingFace cache**, not
files -- SpeechBrain's `from_hparams` links rather than copies. Zipped as links,
the upload succeeds, the dataset lists five files, and the teacher fails to load
on Kaggle with an error about a missing path. `make_kaggle_bundle.py` now
dereferences (85 MB resolved) and re-hashes against the hashes stored inside the
teacher checkpoint, so what is uploaded is provably what the head was fitted
against -- the same discipline as `--prefix-manifest`.

### Figures, report-ready

`scripts/plot_state_teacher_cost.py` renders three single-panel PDFs from
`results.json`, half-textwidth each so any two sit side by side:

| | |
|---|---|
| `state_teacher_cost.pdf` | run length against windows, 16 vs 10 epochs, with the cap |
| `state_teacher_drift.pdf` | **the finding** -- L_state against windows |
| `state_teacher_memory.pdf` | peak memory, which does not bind |

Separate rather than multi-panel because they argue different things: the cost
figure supports a scheduling decision, the drift figure is a result about the
teacher.

### The arm is now fully specified

`experiments/configs/bsrnn_state.yaml` differs from the baseline in exactly five
keys: `w_state` 0.002692, `state_teacher`, `ecapa_dir`, `state_device` cuda, and
`epochs` 10. Nothing else -- same architecture, same parameter count, same
latency, same data, same seed, same schedule. **No capacity confound: the
teacher is training-only and deleted at inference.**

Control is `models/model_sir0_10000-e6.pt`, not a fresh baseline run.

### Two controls, without which head A means nothing

1. **Ablate the enrollment and re-measure state accuracy.** If accuracy holds up
   without the cue, the head is a voice-activity detector and the "who" half is
   unearned.
2. **Split accuracy same- vs cross-gender.** The extractor already leans on
   gender (56.1 % vs 44.4 %, 2026-08-30) and a frame classifier is an easier
   place to hide it; pooled accuracy would conceal it.

`both_directions` helps: roles swap per mixture, so "target = louder" is not
learnable from the distribution.

### Weighting: derive it, do not pick it

Head A's neutral default is inverse class frequency from the measured label
histogram — `both` will dominate and an unweighted head will simply predict it.
Head B's asymmetry (a missed target frame becomes a deletion the judge never
hears; a leaked interferer frame becomes an insertion it attributes to the
target) costs differently in the metric, and the metric can measure which. Same
discipline as `w_g` = 1.69 and `w` from the measured 0.297 absent rate: derived,
not tuned.

### Sequence, and what fits before 14 October

0. **MEASURE ICR FIRST — D10's free measurement, which gates this whole family.**
   **ANSWERED 2026-09-11, and it passes. See the measured section at the end of
   D14.** Leakage is 58.5 % of our content-word error mass, and the per-trial
   rank correlation between how much leaked and how bad the WER was is +0.622
   over 103 trials, surviving inside difficulty strata. Head B and D10 keep their
   motivation. **No transcription pass was needed** — the ICR aggregate had been
   computed on 2026-09-04 and never read back, and every transcript was already
   in `experiments/results/transcripts.csv`.
1. **Label script.** Hours. Shared dependency of everything below, and of the
   sweep.
2. **State-conditioned oracle sweep. No training, no GPU.** Extend `sweep_alpha`
   with a per-state blend set from the true labels. Parameterisation limit to fix
   first: D11's `alpha` blends toward the *mixture*, which on target-absent frames
   passes the interferer straight through, so a second axis is required — a gain
   toward silence on target-absent frames. A 3x3 grid over `sir0_val` `both`
   (n=103) is ~900 transcriptions, under an hour of CPU. **A deliverable either
   way:** it closes the per-frame-oracle gap and either retires or confirms M6's
   measurement risk for M5's gate.
3. **C, folded into M5's already-scheduled gate arm** — one term in one line,
   ~1 k parameters, no separate run. Only if step 2 says the headroom clears the
   judge's SEM.
4. **A alone, then B alone. Post-freeze, or if something is cut.** One variable
   per run: the 2026-08-25 lesson was three variables and one number,
   attributable to none of them.

### Stopping rules

- **Step 0 says leakage is not dominant** -> drop B; A and C survive on the
  level/gating argument alone.
- **Step 2 lands near 2.2 points** -> drop C, and M6's recorded caution stands.
  **Says little about B**, which is not restricted to blending two existing
  signals and can change the mask itself.
- **Head A fails either control** -> C is not built. A gate driven by a bad state
  estimate is worse than no gate.

### Honest cost ranking against the rest of Group D

B is the most interesting piece and the most expensive: a label script, a
detector architecture, a detector training run, then the arm. `BETA` is one number
with `BETA = 1` recovering `L_pres` exactly, and D10 is one term on a stem already
loaded. **Neither is beaten by this on cost-to-evidence.** What D14 contributes
inside the freeze is steps 1-3, not B.


### MEASURED 2026-09-11 — step 0 passes. Leakage is the dominant single error, but it is not the only one

`../tse_venv/bin/python scripts/analyse_leakage_share.py`, 1.2 s measured, no GPU, no ASR
pass. Results in `experiments/results/2026-09-11-leakage-share/`. `sir0_val`
`both`, n=103, the same trials and the same cached transcripts that produced the
published numbers, so this is not a different measurement of a different set.

**The aggregate already existed and had never been read.** `results.json` in
`experiments/results/2026-09-04-train-sir0-10000/` carries `icr_at_2` and
`mean_leak` for floor, our baseline and ceiling, written 2026-09-04. D14 was
written as though the number did not exist. It did. Cost of the actual
measurement: reading a file.

Also found: **`scripts/eval_asr_wer.py` is a zero-byte file**, committed empty in
`38bf48f` on 31 August. D14's step 0 named it as the tool to run. It has never
contained anything.

#### The published aggregate, read back

| system | WER | sub | del | ins | ICR@2 | mean leaked |
|---|---|---|---|---|---|---|
| floor, raw mixture | 65.2 | 32.9 | 9.3 | 23.1 | 67.0 % | 51.3 % |
| ours, baseline | 59.5 | 28.2 | 11.8 | 19.5 | 50.5 % | 34.6 % |
| WeSep | 34.6 | 18.5 | 8.9 | 7.2 | 15.5 % | 9.0 % |
| ceiling, clean target | 5.8 | 3.4 | 0.7 | 1.7 | 0.0 % | 0.0 % |

#### Why the aggregate alone does not answer the question

ICR@2 = 50.5 % says the interferer leaks *often*. It does not say leakage causes
most of the *damage*. Three measures were added underneath it.

**1. Error-mass attribution.** Of the content words the listener reported that
the target did not say, how many did the interferer actually say? Same
normaliser and stopword list as `icr.py`, so this is consistent with the
published ICR rather than a parallel definition.

| system | wrong content words | the interferer's | neither speaker's | leakage share |
|---|---|---|---|---|
| floor | 904 | 635 | 269 | **70.2 %** |
| ours | 742 | 434 | 308 | **58.5 %** |
| WeSep | 406 | 94 | 312 | **23.2 %** |
| ceiling | 91 | 0 | 91 | 0 % |

**2. Per-trial rank correlation, leaked fraction against WER, within one
system.** The cross-system comparison is n=2 and confounded — WeSep is better at
everything at once. Within our baseline, across 103 trials, rho = **+0.622**.
WeSep +0.468, floor +0.628.

**3. Leakage quartiles, our baseline.** Least-leaky quarter of trials: WER
24.9 %. Most-leaky quarter: **101.5 %**. A 76.6-point spread inside a single
frozen model.

#### The confound control, which is what makes this defensible

A positive correlation could be nothing but difficulty: a heavily overlapped
trial leaks more *and* is harder for every other reason. Repeating measure 2
inside strata holds difficulty roughly fixed. Our baseline:

| stratum | n | rho | mean WER | mean leaked |
|---|---|---|---|---|
| overlap low | 65 | +0.602 | 64.4 % | 37.4 % |
| overlap mid | 33 | +0.740 | 66.6 % | 31.2 % |
| target louder | 42 | +0.687 | 42.1 % | 14.5 % |
| interferer louder | 55 | +0.300 | 89.0 % | 53.8 % |

It survives everywhere it can be measured. The one weak cell, interferer louder
at +0.300, is a ceiling effect and not a counterexample: mean WER there is
89.0 %, so there is almost no room left for more leakage to make things worse.
Overlap `none` (n=0) and `high` (n=5) are below the n>=8 floor and are not ranked.

#### What this authorises, and what it does not

**Authorises head B.** The thing head B applies pressure against is the single
largest identified component of our error, it is over half of it, and it tracks
WER trial by trial rather than only in a two-point average.

**Does NOT authorise the assumption that removing leakage removes the error.**
Two limits, both measured here:

1. **41.5 % of our wrong content words were said by nobody.** WeSep, having
   largely solved leakage (23.2 % share), has *more* invented words in absolute
   terms than we do — 312 against our 308. Driving leakage to zero leaves that
   residue untouched, and `fabrication.py` already recorded the same effect
   independently: both extractors raise fabrication ~48 % above doing nothing.
   The realistic prize is roughly the 434 leaked words, not the 742.

2. **Our model already deletes more target content than doing nothing.**
   Deletions: mixture 9.3, ours 11.8, ceiling 0.7. Extraction is *adding* 2.5
   points of deletion while removing 3.5 points of insertion and 4.7 of
   substitution — net WER gain of only 5.7 points for a 16.7-point cut in
   leakage. **Head B's pressure is one-sided: it rewards making no non-target
   voice audible and says nothing about keeping the target audible.** Nothing in
   `L_state` opposes deleting the target; only `L_pres` and `L_gain` do. Deletion
   rate and enrolment sensitivity are therefore the two numbers to watch on the
   M5 arm, not `L_state` alone.

Recorded so it is not re-derived: the M5 arm's epoch 1 read enrolment
sensitivity -13.54 dB against the baseline's -12.08 dB at the same epoch — less
responsive to the enrolment, which is the direction this failure mode predicts.
One epoch, one seed, and the baseline climbed to -9.17 by epoch 2, so it is a
thing to check at epoch 3, not a finding.

### MEASURED 2026-09-11 — the arm and its control do NOT see the same training order. Building the teacher reshuffles the dataset

`bsrnn_state.yaml`'s header claims "same data, same seed, same schedule -- one
term added". **The first two words are wrong and this is why.**

The arm's epoch 1 logged 12,624 present / 7,284 absent crops against the
baseline's 12,625 / 7,283. One crop in 19,908, which looked like rounding.
It is not.

**The chain, each link verified:**

1. The train loader is `shuffle=True, drop_last=True, batch_size=3`
   (`train.py` ~899). `sir0_train` holds **9,955** trials. 9,955 / 3 leaves one
   over, so `drop_last` discards **whichever trial lands last in the shuffle**.
   9,954 x 2 directions = 19,908, matching both runs exactly.
2. Counted locally through the real `TrialDataset` at seed 42, epoch 0, without
   `drop_last`: **12,626 / 7,284, total 19,910**. Both logged runs are that
   minus one trial's two crops — confirming the data on disk is the same and
   the difference is entirely *which* trial was dropped.
3. `build_loss_fn` (`train.py` 555) runs before the first batch is drawn. For
   the arm it constructs `StateHead` — two `LayerNorm`, two `Linear`, one
   bidirectional `LSTM` — and **every one of those draws from the global RNG at
   construction time**.
4. Direct test, seed 42, the same stand-in for `build_model`'s consumption:

   | | permutation starts | trial dropped |
   |---|---|---|
   | no teacher built | 6707, 1302, 4478, 4939 | 8201 |
   | teacher built | 7502, 7622, 5487, 2436 | 1658 |

**So the arm and the control see the same crops in a completely different
order.** The crops themselves are stable — `_crop_offset_start` keys on
`(seed, epoch, idx)`, so a given trial always yields the same window — but the
SGD trajectory is not.

**Consequence for reading the arm.** The epoch-1 result (val `L_pres` -2.4230
against the baseline's -1.6081 at the same epoch) carries an ordering
perturbation of unknown size on top of the teacher's effect. There is no
same-config replicate anywhere in `experiments/results/`, so ordering noise has
never been measured on this codebase and cannot be subtracted. The gain is
~1.9x the epoch-0 spread across all eight prior runs, which is the best
available evidence it is real, but those runs differ by more than ordering.
**Epoch 3-4 remains the check**: ordering noise washes out, a teacher effect
does not.

**Two things this was NOT, both claimed in conversation and both wrong:**

- `num_workers`. The arm's source config says 0 and the baseline's *recorded*
  config says 4, but that is a post-patch artefact: both Kaggle notebooks set
  `NUM_WORKERS = 4` and write it back before `train.py` reads it, and both
  *source* configs say 0. The loader is worker-count independent by design
  anyway — every draw keys on `(seed, epoch, idx)` explicitly to sidestep
  per-worker RNG copies.
- "No LR scheduler." There is one: `ReduceLROnPlateau` (`train.py` ~1132). It
  is plateau-driven rather than horizon-driven, so `epochs: 100 -> 10` still
  does not change the training path, but the reason given was wrong.

**DECLINED 2026-09-11.** A fix exists -- give the train loader its own
`torch.Generator`, reseeded per epoch from `(seed, epoch)` to stay correct on
resume -- and it was not taken. The reasoning, which stands:

Ordering is seed noise. Every single-run A/B comparison in this project has
carried it, including the 2026-09-04 baseline that everything is measured
against, which is itself one run. Pinning one future arm's ordering does not
give the project a noise floor and does not change how any existing number
reads. It removes one source of run-to-run variation while leaving the rest --
cuDNN's nondeterministic RNN backward, and the fact that two arms have
different gradients from step 1 by construction.

**What the real gap is, and it is not this.** No same-config replicate exists
anywhere in `experiments/results/`, so ordering noise -- and run-to-run noise
generally -- has never been measured on this codebase. Until it is, no
single-run delta is separable from scatter, with or without the generator fix.
That is the same problem `report-todo.md` #9 records one layer up for the
judge: "a system difference smaller than it cannot honestly be claimed."

Re-open only if a Group D arm lands a delta small enough that ordering could
plausibly account for it, and only alongside an actual noise measurement.

**The general lesson, worth more than this instance.** Any arm that adds a
module — D13's gate, D12's experts, D5's speaker encoder — will consume RNG at
construction and silently reshuffle its own training set relative to its
control. Every future "one term added" comparison in Group D has this bug
unless the generator fix lands first.

### BUILT 2026-09-11 — head A as code. Runnable except for one thing: the loader owes it labels

`src/models/state_head.py` (the head, the class weights, the cross-entropy, the
per-class recall, the checkpoint stripper), `src/models/losses_state_head.py`
(`LossBSRNNStateHead`), the `state_head` flag on `BSRNN_TFMAP` and in
`build_model`, and `tests/test_state_head.py` (25 tests, all passing).

**516 parameters, verified by a test rather than by arithmetic in a document.**
`AuxStateHead(feature_dim=128).n_parameters == 516`.

**Where it taps: the separator's output, BEFORE `lookahead_shift`.** The shift
moves frame `t`'s features to position `t-k` so the MASK for `t` is built from a
state that has seen `t+k`; the state LABEL for `t` is still about `t`. Reading
the head off the shifted tensor would pair frame `t`'s label with frame `t+k`'s
features — invisible at today's `lookahead_frames: 0` and a silent k-frame
misalignment the moment that key is raised.

**It is trained on the HONEST label, not `REQUIRED_OUTPUT_STATE`.** That mapping
(`both -> target only`) belongs to head B, which scores output audio. Head A
reads internal features, and a model that has correctly noticed "both speakers
are here, and I am about to suppress one" should be rewarded for noticing.
Training it on the required-output mapping would ask the features to forget the
interferer they need in order to remove it.

**Class weights are inverse frequency normalised to mean 1.** Normalising is
what makes the term's magnitude independent of which split's histogram was used,
so a weight derived against it keeps meaning what it meant — and it fixes chance
at ln(4) = 1.386 nats, so the term is readable without a baseline run.

**The head forks the RNG at construction, which settles the 2026-09-11 shuffle
problem for this arm without the global generator fix that was DECLINED.**
Adding any module advances the global RNG, changes the dataloader's shuffle and
changes which trial `drop_last` discards. `torch.random.fork_rng` makes head A's
init deterministic and invisible to everything built after it; a test asserts
that `torch.randn(3)` after building the model is bit-identical with the head on
and off. **This is the narrow fix, not the general one** — D13's gate and D12's
experts still have the bug, because their parameters must sit in the main RNG
stream to stay comparable with anything else.

**Off by default; `forward`'s return type is unchanged.** `return_state=True`
opts in, and asking for state from a headless model RAISES rather than returning
`None` — a silently skipped term would train a baseline and be reported as head
A having had no effect. `drop_state_head()` strips the weights so a head-A
checkpoint loads `strict=True` into the baseline architecture at eval, instead
of eval being loosened to `strict=False` and swallowing a genuinely missing
separator weight too.

**A diagnostic arm came free: `state_head_detach`.** The head still learns to
read the features but no gradient reaches the separator, so it measures how
decodable state already is, online, with no pressure applied. That is the
control for "did the auxiliary loss change the features, or were they always
like this?" — the training-time counterpart of `scripts/probe_state_features.py`.

**NOT DONE, and the arm cannot run until it is:** `dataset_loader.py` does not
yet emit per-frame state labels for a crop, so `loss_fn.state_labels` has
nothing to be set from. One loader change, shared with head B's target column.
No config exists yet either, deliberately — an untested YAML for an arm that
cannot start is a liability, and the weight has to be derived against a real
`L_head` reading the way `w_g` and `w_state` were.

**A and B do not compose.** `LossBSRNNStateHead` and `LossBSRNNState` are
siblings, both subclassing `LossBSRNN`. That is the intended constraint, not an
oversight: D14 says A alone, then B alone. Running both gives one number
attributable to neither, which is the 2026-08-25 mistake.

### MEASURED 2026-09-11 — probing head A's features BEFORE building the arm. Capacity settled, and a gender shortcut found

`scripts/probe_state_features.py --limit 100 --max-seconds 8.0`, 13 min on CPU.
`experiments/results/2026-09-11-state-probe/`. 100 `sir0_val` trials, both
directions, `model_sir0_10000-e6.pt` frozen, `z` captured off `model.separator`
with a forward hook, probes fit on a TRIAL-DISJOINT 70/30 split (frames within a
trial are far too correlated for a frame-level split to mean anything).

**Why before the arm.** Head A is a ~10 h training run whose premise is that the
separator's features can be pushed to encode speaker state. Whether they ALREADY
do is free to check, and it decides the head's design.

#### Capacity: 516 parameters is right, and my band-resolved argument was wrong

| probe | params | balanced accuracy |
|---|---|---|
| linear, mean-pooled — **D14's specified head** | 516 | **61.0 %** |
| MLP, mean-pooled, hidden 128 | 17,028 | 62.0 % |
| linear, band-resolved (no mean-pool) | 16,388 | **59.0 %** |
| chance | | 25.0 % |

33x the parameters buys 1.0 point. **The band-resolved probe is WORSE**, which
kills the argument made in conversation that mean-pooling discards which
frequencies the second voice occupies and that capacity should go on the band
axis. It discards nothing the probe can use. Head A stays at 516.

#### The features do not already encode state usably

| state | recall |
|---|---|
| none | 93.1 % |
| target only | 72.0 % |
| interferer only | 39.6 % |
| **both** | **39.4 %** |

Nearly all of the 61 % is detecting silence. **The two states a gate needs --
`both` and `interferer only` -- sit at ~39 %.** Two consequences, opposite in
sign: head A has real work to do, so its auxiliary loss is not redundant; and
piece C driven by today's features would fail D14's own stopping rule ("a gate
driven by a bad state estimate is worse than no gate").

#### The gender shortcut, which D14 predicted

| | balanced accuracy | frames |
|---|---|---|
| same gender | **53.4 %** | 18,054 |
| different gender | **66.5 %** | 42,126 |

A 13-point gap. D14's control 2 called it: "a frame classifier is an easier place
to hide it; pooled accuracy would conceal it." On the case that matters -- two
speakers of the same gender -- the features are close to useless beyond silence
detection, and a substantial part of the headline is gender. **Head A trained on
this would be trained to lean on the shortcut harder, and any gate built on it
would inherit it.** Report the split, never the pooled number.

#### Control 1 was built wrong and is inconclusive

The "wrong enrolment" arm substituted the OTHER SPEAKER IN THE SAME TRIAL, which
is not an ablation -- it is a different valid instruction. It scored 61.3 %
against 61.0 %, which naively reads as "the cue is ignored". The per-class
recalls say otherwise:

    real enrolment     target 72.0   interferer 39.6
    wrong enrolment    target 39.9   interferer 72.2

**They swap cleanly**, which is the signature of features that track WHICH
speaker was requested, with the roles trading when the cue trades. Encouraging,
but it is not the control D14 asked for. The correct ablation is a stranger from
a DIFFERENT trial, where there is no role to swap into. Rerun before acting on
control 1 either way.

### D15. Penalise mask ROUGHNESS over time — attack fabrication, not leakage

**Status: PROPOSAL, raised 2026-09-12 (Grant). Step 0 is free and answers most
of it. Nothing here is built.**

**The gap it addresses, and it is the half nobody is working on.** Measured
2026-09-11: **41.5 % of our wrong content words were said by nobody** — 308
invented words against 434 leaked ones. Every open proposal in Group D attacks
leakage (D10, D14 B, D13's gate). Nothing attacks invention, and `fabrication.py`
recorded independently that both extractors raise fabrication ~48 % above doing
nothing — **extraction is CAUSING part of this, not failing to remove it.**

**The mechanism, and it is textbook.** A time-frequency mask that changes
abruptly between neighbouring frames produces isolated, flickering spectral
peaks — **musical noise**, the classic artefact of spectral-subtraction and
masking systems. It is perceived as chirping, and for our purposes the important
property is not how it sounds: a spurious spectral transient looks to an ASR
front end like an onset, and onsets are what word hypotheses are built from.
**Invented words are exactly the error that a flickering mask would produce.**

**Nothing in the current model or objective opposes it.** `Estimator` predicts a
complex mask per band per frame through a 1x1 conv on the feature stream, and GLU
bounds its MAGNITUDE. No term and no architectural constraint says anything about
how much it may change from frame t to frame t+1. `L_MR` prices detail at four
resolutions but is minimised by matching the reference, not by being smooth, and
was measured to reward muting (decisions-m2.md 2026-08-28).

**Borrowed, and the difference matters.** Temporal smoothing of a spectral gain
is standard in speech enhancement: the decision-directed a-priori SNR estimator
(Ephraim & Malah, IEEE TASSP 1984) is essentially a recursive smoother and is
the canonical musical-noise fix; cepstral-domain smoothing (Breithaupt, Gerkmann
& Martin, ICASSP 2008) is its modern form. In images the same idea is total
variation regularisation (Rudin, Osher & Fatemi, Physica D 1992). **BORROWED
WITH A DIFFERENCE:** those smooth a gain to improve PERCEIVED quality, and are
tuned on PESQ-style measures. Here the justification is content fidelity for a
downstream listener, and the arm would be judged on invented-word count and
LCF-WER, never on DNSMOS. That is a different claim with a different acceptance
test, and our metric can actually distinguish them.

### The trap, stated before the arm rather than after

**Speech has real transients.** Plosive releases, stop bursts and word onsets
are genuine fast changes, and a mask that cannot move quickly smears them. We
are ALREADY deleting more than doing nothing does — deletions 11.8 against the
raw mixture's 9.3 — so blunt smoothing attacks the error we have too much of by
making worse the other error we have too much of. **"Smoother is better" is
false and must not be the form of the hypothesis.**

**The fix for that is to derive the target from the ORACLE mask, which is free.**
We own `target.wav` and `mixture.wav` for every trial, so the ideal mask is
computable exactly, and with it the frame-to-frame variation a CORRECT mask
exhibits. The hypothesis then becomes falsifiable and self-limiting: penalise
roughness **beyond what the oracle mask itself shows**, per band, the same
deadzone shape as `L_gain`'s +-3 dB. If our mask is already no rougher than the
oracle's, there is nothing here and the proposal dies at step 0 for the cost of
an afternoon.

### Sequence

0. **MEASURE THE ROUGHNESS GAP. No training, no GPU, hours.** For `sir0_val`,
   compute per band the mean absolute first difference along time of (a) the
   oracle mask |S_target| / |X_mixture| and (b) our checkpoint's predicted mask.
   Three outcomes and all are useful:
   - ours is much rougher -> the premise holds, go to 1
   - ours is comparable -> **the proposal is dead**, recorded, no run spent
   - ours is SMOOTHER -> we are over-smoothing already, which would explain the
     deletions and points the opposite way
   Correlate the per-trial gap against that trial's invented-word count from
   `transcripts.csv`, the same way D14 step 0 correlated leakage against WER.
   A gap that does not track invention is not the mechanism.
1. **A loss term.** L1 of the mask's first difference along time, per band,
   deadzoned at the oracle's own roughness. One term, one weight, derived
   against a measured anchor exactly as `w_g` = 1.69 and `w_state` = 0.002692
   were — never picked.
2. **Or an architectural constraint instead**, if the term is hard to weight: a
   causal one-pole smoother on the mask with a per-band learned coefficient.
   Zero added latency (causal IIR, one multiply-add per bin), ~32 parameters,
   and a hard constraint rather than a soft penalty. Init at no smoothing so the
   arm starts as the baseline, the same discipline as D4a's zero-init gates.

**Cost-to-evidence: the best in Group D right now.** Step 0 needs no GPU, no
training and no API budget; it either kills the idea or hands the arm a derived
weight. Compare D14 B, which cost a label script, a detector architecture, a
detector training run and a 10.25 h arm to move its own term 1.4 %
(decisions-m2.md 2026-09-12).

**It also composes with everything.** It constrains the mask's behaviour over
time and says nothing about who the target is, so it is orthogonal to D4a, D13
and D14, and can be added to whichever of those survives.

### MEASURED 2026-09-12 — D15's premise is WRONG, and what replaced it is worse news

D15 proposed penalising mask ROUGHNESS on the theory that a flickering mask was
producing musical noise and inventing words. **Step 0 was run and it refutes
that.** `scripts/plot_mask_grid.py` and a 12-trial measurement on
`model_sir0_10000-e6.pt`, against the ideal mask `|target| / |mixture|`:

| | varies along TIME | varies along FREQUENCY | freq / time |
|---|---|---|---|
| our mask | 0.0445 | 0.0238 | **0.53** |
| ideal mask | 0.1540 | 0.1551 | **1.01** |
| shortfall | **3.5x** | **6.5x** | |

**Our mask is not too rough. It is 3.5x too SMOOTH along time and 6.5x too
smooth along frequency.** The third outcome D15 listed for step 0 — "ours is
SMOOTHER, which points the opposite way" — is the one that happened. Smoothing
is retired as an intervention. The step 0 measurement cost an afternoon and
saved a training run, which is exactly what it was for.

### The replacement finding, and it is structural

**84.2 % of our mask's variance is explained by a single number per frame.**

**The model has not learned a time-frequency mask. It has learned a broadband
volume knob.** It raises the output when the target speaks and lowers it when
they do not, applying nearly the same gain to every frequency in a frame. The
ideal mask varies equally in both axes; ours varies half as much across
frequency as across time.

This is not a tuning problem and no loss weight fixes it. Two overlapping voices
occupy the same frequencies at the same instant, and the only way to separate
them is to decide per time-frequency cell which voice owns it. **A broadband gain
cannot do that even in principle.** It can only be loud when the target talks,
which is voice activity detection wearing an extractor's architecture.

It explains, at one stroke:
- why leakage survives — a volume knob passes both voices when both speak;
- why 62 % of the output is enrolment-independent (2026-08-30) — a volume knob
  needs to know WHEN someone speaks, not WHO;
- why the state probe reads "somebody is speaking" at 93 % and "which of the
  two" at 39 % (2026-09-11);
- why the frozen state teacher moved leakage a little and cost fidelity — the
  only lever the model has is to turn the knob down harder.

### CONFIRMED 2026-09-12 — the holes really do delete target speech

`experiments/results/2026-09-12-eval-floor0.05`, an inference-time mask floor of
0.05 on the baseline checkpoint, n=103, ASR stand-in:

| | floor 0.00 | floor 0.05 |
|---|---|---|
| deletions | 11.79 | **8.64** (−3.14, −27 %) |
| ICR@2 | 50.49 | 58.25 (+7.77) |
| no response | 2.91 | 0.97 |
| LCF-WER | 59.52 | 63.33 |

**The deletion drop is the largest single metric movement this project has
produced.** It confirms that the zeroed bins were carrying target speech.

**And it shows the floor is the wrong cure.** Filling a hole with the raw mixture
fills it with BOTH speakers, so leakage rises more than deletions fall. What is
wanted is a fill that is target-selective — which is the argument for
redistributing the existing gain across frequency rather than adding the mixture
back, i.e. `scripts/postprocess_mask.py`.

**A floor is still worth keeping as a knob**, because deletions and leakage now
have a measured exchange rate and nothing else in the project trades between
them explicitly.

### MEASURED 2026-09-12 — frequency structure is the lever. A free post-hoc version beats a 10-hour training arm on leakage

All paired bootstraps, 10,000 draws, n=103, `sir0_val` `both`, ASR stand-in.
The metric's irrelevance floor is 1.57 points (decisions-m3.md), so anything
under that is not a claim.

| intervention | cost | leakage change | LCF-WER | verdict on leakage |
|---|---|---|---|---|
| frozen state teacher (D14 B) | **10.25 GPU-h** | -1.93 | +1.72 | inside noise |
| post-hoc frequency sharpening | **free** | **-6.99** | +10.01 | **REAL**, [-10.41, -3.85], 0.0 % opposite sign |
| mask floor 0.05 | free | +5.39 (worse) | +3.81 | REAL, wrong way |

**Sharpening the mask across frequency -- crudely, after the fact, with no
learning -- removed 3.6x more leakage than the entire teacher arm did, and unlike
the teacher the effect is unambiguous.** It took leakage from 34.58 to 27.42
against WeSep's 9.0, closing roughly a quarter of that gap with a post-processor.

It cost +10.01 LCF-WER, which is also real. That is the expected price of
imposing structure on a model never trained to produce it: the sharpening is
applied to the output ratio, which carries the additive residual and an analysis
mismatch, and nothing optimises the result.

### The two knobs point opposite ways and both lose, which is itself the finding

| | deletions | mean leaked | LCF-WER |
|---|---|---|---|
| baseline | 11.79 | 34.58 | **59.52** |
| fill the holes (floor 0.05) | **8.64** | 40.87 | 63.33 |
| sharpen across frequency | 15.13 | **27.42** | 69.53 |

**The baseline already sits near a local optimum on the deletion-versus-leakage
trade-off.** Moving it after the fact costs more than it gains in either
direction. So the remaining gain is not in re-weighting that trade-off -- it is in
giving the model the ability to make fine time-frequency decisions in the first
place, which is what the 84 %-volume-knob measurement says it cannot currently do.

**This is now the best-evidenced direction in Group D**, and it was established
for the cost of an afternoon of CPU rather than a training session.

### CORRECTION and the headline, 2026-09-12 — HARD sharpening: real leakage removal at no measurable WER cost

The `mid` setting (1.3/0.7, unsupported bins scaled to 0.3) cost +10.01 LCF-WER
and +7.19 insertions, and was written up above as "sharpening adds artefacts".
**That was the setting, not the idea.** The `hard` setting (1.5/0.5, unsupported
bins removed outright) behaves completely differently:

| | measured | 95 % interval | verdict |
|---|---|---|---|
| mean leaked % | **-6.38** | [-10.56, -2.45] | **REAL improvement** |
| LCF-WER | -0.90 | [-5.18, +3.12] | inside noise AND below the 1.57 floor |
| insertions | -3.81 | [-9.03, +0.63] | inside noise |
| deletions | +3.29 | [-1.78, +8.54] | inside noise |

**A free post-processor removed leakage for real and cost nothing measurable on
the headline metric.** The 10.25-hour teacher arm achieved neither.

**PARTIAL SUPPRESSION IS WORSE THAN COMPLETE SUPPRESSION.** down=0.3 leaves a
scaled copy of every unsupported bin and insertions rose 7.19; down=0.0 removes
them and insertions fell 3.81. That is the classic musical-noise result -- a
half-removed component is an artefact, a removed one is silence -- and it should
govern any future mask post-processing or gating rule this project writes.

**Two caveats that travel with it.**
1. **-0.90 LCF-WER is NOT an improvement.** It is inside the interval and below
   the irrelevance floor. The claim is "unchanged", never "better".
2. **It is not a uniform win**: better on 29 trials, worse on 46, tied on 28. The
   flat corpus number comes from helping a lot on a few trials and hurting
   slightly on many. Any write-up must say so.

### MEASURED 2026-09-12 — the corrected internal-mask run. Region growing needs something structured to grow FROM

Re-run on the fixed `apply_hysteresis` (RMS level restoration). Output level
-4.11 dB against the baseline's -5.65 dB, so the level explosion is gone and this
run measures the intervention.

| variant | leakage | LCF-WER |
|---|---|---|
| frozen state teacher, 10.25 GPU-h | -1.93 (noise) | +1.72 (at the 1.57 floor) |
| sharpen output ratio, partial removal | **-6.99 REAL** | +10.01 REAL worse |
| **sharpen output ratio, full removal** | **-6.38 REAL** | -0.90 unchanged |
| sharpen internal mask, full removal | **-10.37 REAL** | **+16.91 REAL worse** |

Internal-mask bootstrap: leakage -10.37 [-15.29, -5.54], 0.0 % opposite sign;
LCF-WER +16.91 [+5.28, +30.98], 0.1 % opposite sign. Both real.

**EVERY sharpening variant removes leakage substantially and unambiguously.** Four
independent settings, all outside the interval, against a training arm that could
not manage it once. **Frequency structure is the lever. That is settled.**

### The internal-mask failure is the informative result

It removed the MOST leakage (-10.37, best of the day, 34.58 -> 25.01 against
WeSep's 9.0) and did the MOST damage (+16.91 LCF-WER, insertions +11.32).

**Why: the mask it grows from is flat.** At 84 % of variance explained by one
number per frame, the bins that clear a relative threshold are chosen by tiny
fluctuations -- effectively noise. So the structure imposed is arbitrary. It
deletes the interferer, and it deletes everything else with equal indifference.
The output ratio works better precisely because it is NOT flat: it carries real
spectral content, so thresholding it selects meaningful bins.

**REGION GROWING NEEDS A MEANINGFUL CONFIDENCE MAP TO GROW FROM, AND THIS MODEL
DOES NOT PRODUCE ONE.** No post-processor can manufacture that. The model has to
learn to emit a mask that HAS structure worth growing -- which is the
architectural form of the idea, and it now rests on measurement rather than
intuition.

**Consequence for the plan.** The cheap post-hoc route is exhausted: its best
outcome is -6.38 leakage at no WER cost, already achieved, and the ceiling above
it is blocked by the mask's flatness rather than by the growing rule. The next
move is to make the mask structured during training, not to keep tuning
thresholds on a flat one.

### MEASURED 2026-09-13 — the volume knob, DECOMPOSED. What frequency shape exists is a FIXED EQ curve, and it carries no speaker information

`scripts/diagnose_mask_structure.py`, 12 `sir0_val` `both` trials, whole clips,
`model_sir0_10000-e6.pt`, seed 42, 16 min CPU.
`experiments/results/2026-09-13-mask-structure/`.

**Why, when 84.2 % was already measured.** "One number per frame explains the
mask" has two readings and the mask-grid picture cannot tell them apart, because
both draw as vertical stripes: the model applies a genuinely FLAT gain, or it
applies a FIXED spectral shape — the persistent dark band below 500 Hz is
visible in `mask_grid_sir0_val-42-000004_floor0.png` — scaled up and down by one
number per frame. Neither can separate two overlapping voices, but only the
second means the model learned anything about frequency at all.

**The decomposition.** Two-way additive, ours, standard ANOVA form on the mask
magnitude: `M(f,t) = mu + eq(f) + gain(t) + interaction(f,t)`. The three terms
are orthogonal by construction, so the variance shares are exact rather than
fitted. `interaction` is THE ONLY TERM THAT CAN SEPARATE TWO VOICES: it is the
only one that says "this frequency, at this instant, belongs to the target", a
statement whose answer must change from frame to frame.

| frames | mask | gain(t) | eq(f) | interaction |
|---|---|---|---|---|
| all | **ours** | **83.5 %** | 7.1 % | **9.4 %** |
| all | ideal | 53.0 % | 2.5 % | 44.5 % |
| speech (93 %) | ours | 82.6 % | 7.7 % | 9.7 % |
| speech | ideal | 52.0 % | 2.6 % | 45.4 % |
| **overlap (26 %)** | **ours** | **46.4 %** | **35.2 %** | **18.3 %** |
| **overlap** | ideal | 16.1 % | 8.0 % | **75.9 %** |

**Replicates 84.2 % by a different method** (83.5 % over all frames). The
earlier figure came from `postprocess_mask.py`'s frame-mean ratio; this is a
variance decomposition. Two methods, one answer.

**Overlap frames are the honest test and are reported separately.** Ducking
silence is free and is not a skill; pooling it inflates `gain(t)`. Overlap =
frames where the target AND the interferer are both active, the only regime where
separation is a question.

### The two findings, and the second is the one that was not already known

**1. During overlap, 81.7 % of our mask is "one fixed shape x one number".** A
quantity that cannot separate two voices under any setting of that number. The
genuine per-cell decision is 18.3 % against the correct answer's 75.9 %.

**2. The frequency shape our mask has is STATIC, so it carries no speaker
information.** 35.2 % during overlap is not a small number — it is the second
largest term — but it is the same curve held across the clip. A contour that does
not change when the speakers change cannot encode which of them owns a bin. What
looked like partial frequency selectivity is a baked-in EQ.

**And the ideal mask says the model is applying its one tool to the wrong
problem.** For the correct answer, `gain(t)` is worth only 16.1 % during overlap
— obviously, since you cannot turn one voice down without the other. Ours spends
46.4 % of its behaviour there.

### Three caveats that travel with these numbers

1. **Shares are relative to each mask's OWN variance.** Ours varies far less in
   absolute terms (0.0238 across frequency against the ideal's 0.1551,
   2026-09-12), so the absolute shortfall is LARGER than the share gap suggests.
   "18.3 % against 75.9 %" must never be read as "we do a quarter of the job".
2. **"Fixed EQ" is measured WITHIN a clip.** Whether it is the same curve across
   clips — i.e. baked into the weights rather than adapted per mixture — is NOT
   measured. The script stores shares, not the curves.
3. **n=12 trials, one checkpoint, no interval.** A structural share this large is
   not a candidate for sampling noise, but no significance is claimed.

### What it authorises

- **The next training arm prices the INTERACTION term, not the mask as a whole.**
  A plain mask-MSE against the ideal ratio mask would be largely satisfied by
  `gain(t)`, which the model already produces. The term has to target what is
  left after `gain(t)` and `eq(f)` are removed, or it buys nothing.
- **The interaction share is a per-epoch readout that does not need the ASR.**
  It reads out in one epoch, against a metric whose noise floor is +-8 LCF-WER
  points on `sir0_val`. That makes a 1-2 epoch smoke run a real gate before
  committing a 10 h session.
- **It does NOT authorise building anything yet.** The free step 0 is the oracle
  volume knob: take the IDEAL mask, flatten it to `gain(t) x eq(f)`, synthesise,
  and score it. If a PERFECT volume knob still transcribes badly, flatness is
  proven to be the cost and the arm is justified. If it scores near the 5.85
  ceiling, flatness is a red herring and our fault is that the GAIN is wrong — a
  far cheaper fix. One ASR pass, no GPU, decisive either way. **Run this first.**

### MEASURED 2026-09-13 — D6's residual-branch ablation. R is INERT, and the fabrication hypothesis it was built to test is DEAD

D6 flagged `Estimator`'s additive residual `R` as "unbounded and unconditioned"
and asked for an ablation arm. This runs it, at inference, for CPU hours.

**The hypothesis, stated before the run so it cannot be rewritten after.** The
output is `S = M (x) X + R`. The mask is MULTIPLIED, so in a bin where |X| = 0 it
contributes exactly 0 — verified, `masked_energy_in_silent_bins` measured
0.0000. `R` is ADDED from a raw Conv1d with no GLU and no bound, so it is the
ONLY path that can place energy in a cell the microphone never recorded.
Emitting sound nobody made is physically what an invented word is, and invented
words are 41.5 % of our wrong content words with no mechanism assigned. **The
proposal was that R is that mechanism.**

### Step 0, the correlation. It refused the hypothesis before the ablation ran

`scripts/diagnose_residual.py`, n=103, `sir0_val` `both`,
`model_sir0_10000-e6.pt`. Per-trial R contribution against per-trial invented
words, the same shape as D14 step 0 and D15 step 0:

| measure | mean | r vs invented words | significant at n=103? |
|---|---|---|---|
| R's share of output energy | 8.6 % | **-0.161** | no (crit 0.194) |
| R's share after cancellation | 6.9 % | **-0.178** | no |
| output energy in silent bins | **0.72 %** | **-0.149** | no |

**All three negative, none significant.** Trials where R contributes more do not
invent more.

**The sharpest number against the mechanism: R is not aimed at the silence.**
Silent bins are 10 % of all bins and hold **8.0 %** of R's energy (p10 7.1 %,
p90 9.0 %) — slightly LESS than proportional, and almost constant across trials.
A fabrication mechanism would concentrate there. R is spread uniformly.

### The ablation. Deleting R changes nothing measurable

`--residual-scale 0.0`, same checkpoint, same 103 trials, ASR stand-in.
Paired bootstrap, 10,000 draws.

| | baseline | R deleted | difference | 95 % interval | verdict |
|---|---|---|---|---|---|
| LCF-WER | 59.52 | 59.72 | **+0.20** | [-3.02, +3.30] | INSIDE NOISE, and below the 1.57 floor |
| mean leaked % | 34.58 | 37.21 | +2.13 | [-1.39, +5.88] | INSIDE NOISE |
| insertions | 19.53 | 18.63 | -0.90 | | below the floor |
| deletions | 11.79 | 10.77 | -1.02 | | below the floor |
| substitutions | 28.20 | 30.33 | +2.13 | | |
| **invented content words** | **308** | **306** | **-2** | | **0.6 % of 308** |

Per trial: better on 24, worse on 33, tied on 46.

**Deleting 8.6 % of the output energy moved the headline metric 0.20 points and
the invented-word count by two words.** None of the three outcomes registered in
advance occurred. R is not fabricating, and it is not earning its keep either.

### What this closes, and what it does not

- **The fabrication mechanism is NOT R.** Recorded as refuted. The 41.5 % of
  error mass that is invented words remains unexplained, and the next candidate
  has to come from somewhere else. This was my hypothesis and the data killed
  it; the cost was one afternoon of CPU, which is what step 0 is for.
- **D6's residual ablation is ANSWERED at inference.** `residual_branch: true`
  contributes nothing measurable to content fidelity on this checkpoint.
- **It does NOT authorise removing R.** This ablates R from a model TRAINED WITH
  R; "no measurable difference at +-3 points" is not "useless". A
  trained-without-R arm is the only thing that settles it, and at 197,890
  parameters (2.75 % of the model) the prize is small. **Not scheduled.**
- **The 84 % volume-knob finding is NOT qualified by this.** The worry was that
  it described only the multiplicative path while R did the real work. R carries
  8.6 % of the energy and removing it changes nothing, so the mask really is
  the model. **The structural finding stands, and is now stronger.**

### The reusable part

`Estimator.residual_scale` and `Estimator.capture_parts`, `--residual-scale` on
`make_estimates.py`, `scripts/diagnose_residual.py`,
`tests/test_residual_ablation.py` (6 tests). One of those tests feeds the model a
SILENT mixture and asserts the output is non-zero: the capability to fabricate is
real and is now pinned by a test, even though the measurement says it is not
being used.

### MEASURED 2026-09-13 — objective or data? It is the OBJECTIVE. More data makes the mask WORSE, not better

`scripts/measure_mask_flatness.py`, 50 `sir0_val` `both` trials, the SAME trials
for every checkpoint, 1.6 h CPU. Five checkpoints that already existed.

| checkpoint | trials | epoch | share(loud) | f/t ours |
|---|---|---|---|---|
| `model_sir0.pt` | ~1,989 | 9 | 0.6415 | 0.9561 |
| `model_sir0_5000-e7.pt` | ~4,976 | 7 | 0.7766 | 0.7646 |
| `model_sir0_10000-e6.pt` | ~9,955 | 6 | 0.7958 | 0.6158 |
| `model_sir0_10000-e7.pt` | ~9,955 | 7 | 0.7738 | 0.6026 |
| `model_sir0_10000-last.pt` | ~9,955 | 15 | 0.7739 | **0.4826** |
| **ideal mask** | | | | **1.0136** |

`share` = fraction of mask variance explained by one number per frame; 1.0 is a
pure volume knob. `f/t` = variation across frequency relative to across time;
**ours must RISE toward 1.0136, and falling is worse.**

### Paired bootstrap, 50 trials, 10,000 draws

| comparison | share(loud) | f/t (loud) |
|---|---|---|
| **DATA 5k -> 10k, epoch MATCHED at 7** | -0.0029 [-0.027, +0.022] **inside noise** | **-0.1620 [-0.205, -0.118] REAL** |
| DATA 2k -> 10k | +0.1322 [+0.105, +0.160] REAL | -0.3535 [-0.392, -0.315] REAL |
| **EPOCH 6 -> 15, data FIXED** | -0.0219 [-0.047, +0.001] inside noise | **-0.1332 [-0.187, -0.074] REAL** |

**The epoch-matched data comparison is the one that answers D15's successor.**
Doubling the training set from ~5k to ~10k trials, at the same epoch, left the
volume-knob share statistically UNCHANGED and made the frequency/time ratio
significantly WORSE.

**More training does the same thing.** At fixed data, epoch 6 -> 15 moved f/t
-0.133, also real, also the wrong way.

**Both axes push the model TOWARD the volume knob.** That is exactly what is
expected if the objective's optimum IS a volume knob — better optimisation, by
either route, converges on it harder.

### The trap in this table, stated because it inverts the obvious reading

**The ~2k model has the best-looking numbers and is the worst model.** Lowest
share (0.64) and an f/t of 0.956, nearly the ideal's 1.014. It is not more
structured: **an unstructured, noisy mask ALSO has f/t ~ 1, because noise varies
equally in both axes.** That checkpoint memorised its training set
(decisions-m2.md 2026-08-29) and is beaten by pass-through. Training then sheds
that noise and converges onto the knob.

**Nobody may read this table as "less data is better".** It says the opposite:
what training buys, on this objective, is a cleaner volume knob.

### Verdict, against the rule registered BEFORE the run

Three outcomes were registered: flatness constant (objective), flatness falling
with data (data), epoch swamping data (inconclusive). **The result is the first
on `share` and something stronger on `f/t` — not merely "data does not help" but
"data actively makes it worse".** Recording the mismatch rather than pretending
the rule anticipated it.

**Epoch does NOT swamp data.** On `share`, data moves 0.132 (2k->10k) against
epoch's 0.022. On `f/t` the two are comparable (-0.162 vs -0.133) but point the
SAME way, so the confound cannot explain the result away — it reinforces it.

### What this authorises, and what it does not

- **MORE DATA WILL NOT FIX THE FLAT MASK. The data hypothesis is REFUTED**, not
  merely unsupported. Scaling the training set is no longer a candidate answer to
  the 84 % finding.
- **The structure loss is justified by evidence rather than by elimination.**
  Supervise the mean-removed across-frequency deviation against the free oracle
  mask, with the residual branch handled — R is inert (2026-09-13) so it will not
  absorb the term, which this measurement also settles.
- **It does NOT predict the intervention works.** Flatness is a property of the
  mask, not of what a listener transcribes. This selects the arm; only LCF-WER on
  `sir0_privval`, per SIR band, scores it.
- **Single seed, n=50, one architecture.** Run-to-run training variance is still
  unmeasured and sits on top of every interval here.

---

## 2026-09-15 — THE MENU: what is open, what each costs, and what it buys

**Written as a decision menu, not a decision.** Nothing below is taken. Every
cost is from `run_times.md` or a completed comparable run; none is estimated.
Ordered by information per hour, obligations first.

### Obligations — these are not optional and two of them are cheap

| # | what | cost | why it is not optional |
|---|---|---|---|
| **O1** | **Score the struct control on `sir0_privval`.** `2026-09-14-est-privval-control` is 1,421 rendered trials with a `meta.yaml`, and it has never been through `evaluate.py`. | **1.2 h**, measured | D17's config names LCF-WER on `sir0_privval` **per SIR band** as its acceptance test. The arm has that number (47.37); the control does not. **The arm is currently unjudged against its own registered test.** |
| **O2** | **Write D17 up.** The arm ran, was evaluated four ways, and appears in no decision log. | ~1 h, writing | Project rule: every experiment gets a logged result. It is the only unlogged run in the repo. |
| **O3** | **M6's stratified tables (B13) and latency curve (B11).** Scoring only, no training. | unmeasured; `evaluate.py` already emits the columns | M6 is the thesis's central finding and its checklist is 2 of 8. A headline aggregate must never appear alone. |

### D17's result, stated here because the options below branch on it

**In plain words: the arm was told to give the mask frequency detail, and it
produced a mask with half the frequency detail it started with.** The live judge
could not tell the two systems apart.

| on `sir0_val` `both`, n=103, same trials | baseline e6 | struct e12 |
|---|---|---|
| LCF-WER (judge) | 55.59 | **55.36** (−0.23, inside the ~3-point paired floor) |
| leakage ICR@2 | 61.17 | 65.05 (worse) |
| fabrication FR@2 | 43.69 | 48.04 (worse) |
| mask variation across frequency, `d_freq` | 0.0253 | **0.0121 — halved** |
| freq/time ratio (ideal 1.014, must RISE) | 0.616 | **0.306 — fell** |

**Flag, per the rule about headline numbers moving for bad reasons:** LCF-WER is
flat and the *direction* is wrong on every diagnostic. Read this as a null with a
mechanism, not as a tie.

**And `train_L_struct` is absent from `2026-09-14-train-sir0-struct/history.csv`.**
The logging fix landed alongside the run. **We cannot say whether the term
descended.**

---

### D18. Why L_struct FLATTENED the mask — the degenerate-optimum hypothesis

**Status: proposal. The cheap half needs no training.**

**Hypothesis.** `_loss_mask_shape` is an energy-weighted **L1** between the
model's mean-removed frequency shape and the oracle's. When the target is not
predictable from the input, the L1-optimal constant prediction is the
**conditional median of the oracle deviation, which is ~0** — i.e. a flat mask.
So a shape loss the model cannot fit does not teach shape; **it actively pays the
model to remove the shape it had.** That is exactly the measured signature.

**Test 1, free, no training.** On cached crops, compute the term for (a) the
model's mask, (b) a constant-zero shape, (c) the oracle. **If (b) beats (a), the
term's own optimum is the flat mask** and every "supervise the shape" variant
inherits the fault.

**Test 2, one training run.** If Test 1 confirms it, the fix is a loss whose
optimum is not zero — correlation or cosine across frequency (scale-free, so a
flat prediction scores worst, not best), or a variance-matching penalty. Same
weight-derivation procedure.

**What it is worth.** D15's successor and D17 are the same idea; if the loss form
is the fault, the idea has never actually been tested. If Test 1 refutes it, the
shape idea is dead on evidence and M5 should spend its remaining weeks elsewhere.

### D19. Is the volume knob OURS or the TASK'S? Run the flatness instrument on WeSep

**Status: proposal. Cheapest high-information item on this page.**

WeSep `tfmap_context_causal_100` scores **26.40 LCF-WER against our 55.59** on the
same 103 trials — floor 63.27, ceiling 1.05. It is causal, out of domain by its
own config, and it beats us by ~29 points.

**The measurement.** Point `scripts/measure_mask_flatness.py` at WeSep's mask.

- **WeSep's `f/t` near the ideal 1.01** ⇒ the volume knob is a property of **our
  objective**, and that is the most defensible finding in the project: a
  published, better-scoring causal model on the same trials does not do it.
- **WeSep's `f/t` also ~0.5** ⇒ the knob is what causal band-split masking does,
  the flatness diagnosis does not explain the 29-point gap, **and D18/D15/D17 are
  the wrong tree entirely.**

Either answer redirects M5. Neither needs a training run. The one obstacle is
reaching WeSep's internal mask rather than its waveform; if that is not
exposed, the free fallback is the mask implied by output/input magnitude.

**Carry the caveat:** different data, different objective, different training
budget. This diagnoses *our* model; it is not a comparison claim.

### J5. Run-to-run training variance — the third noise source, still unmeasured

**Status: proposal. This one is uncomfortable and it is load-bearing.**

**Ten training runs exist and every one is `seed: 42`.** No same-config replicate
has ever been trained. Two noise sources are measured — trial sampling (~3 points
paired, `decisions-m3.md` 2026-09-12) and judge SEM (~0.5) — and **the third,
training itself, is a blank.**

Consequence: the state arm's −1.72 and D17's −0.23 are both compared against an
error bar nobody has drawn. Both were correctly written up as "did not improve"
rather than "lost", but **M6 cannot claim two systems differ without this.**

**The measurement.** Re-train `bsrnn_baseline.yaml` at two further seeds, score
all three on the same 103 trials. Cost: 2 training runs plus 2 judge passes
(14 min each, measured). **Do E8 first** — it roughly halves the training half.

**Register before running:** if the seed spread exceeds ~3 points, no
single-seed intervention in this project is separable from noise, and the M6
write-up becomes a spread-versus-spread comparison rather than a point one.

### Already-open items, re-costed against the above

| id | one line | cost | verdict today |
|---|---|---|---|
| **E8 + E7** | fp32 batch probe writes batch 3 into an fp16 run that fits 6; then the second T4 is idle | ~half a day, ~2x throughput | **Do this first if anything in J5 or D13 runs.** It is the multiplier on every remaining training hour. |
| **D13** | the per-band mix-back gate — **M5's actual scoped deliverable, and it is not built** | one build + one run | Oracle ceiling 2.2 points against a ~3-point paired floor. **Declare the effect size and k before running**, or the null is uninterpretable. M6 already says this. |
| **D4a** | `bsrnn_tfmap_inject.yaml` — built 2026-09-11, **never run** | one training run | A finished arm sitting unused. Cheapest untested modelling change on the page. |
| **D2** | attention temperature, one line of code | an afternoon | D7 records it as run but never logged as an arm. Either log it or re-run it as one. |
| **D14** | state teacher | done | Measured and written up. No further arm without a reason. |
| **D1** | phoneme-template cue | M5-scale | **Do not start.** Gated on D2, and M5 is cuttable with ~4 weeks left. |

### The sequencing this implies

1. **O1** (1.2 h) — finishes D17 against its own test.
2. **D19** + **D18 Test 1** — both free of training, both redirect M5.
3. **O2**, then **E8/E7**.
4. Then **one** of D13 or J5, chosen by what 2 and 3 say. **Not both — there is
   not room before 14 October**, and M6 must still run.

**The honest reading of the last three arms.** The state teacher moved its own
term and lost the metric; D17 moved its own diagnostic the wrong way and drew;
the data hypothesis was refuted outright. Three interventions, no gain. **D19 is
on this list because it is the only item that asks whether the diagnosis itself
is right**, and it costs no training.

---

## 2026-09-15 — G1: THE GEMINI TUNER. A local stand-in for the judge

**Status: PROPOSAL, raised 2026-09-15 (supervisor). Nothing below is taken.**
Unblocked by the same day's withdrawal of family separation (decisions-m4.md
2026-09-15): Gemini may now supply targets, rewards and filters offline.

**The ask, as given.** Build a local model — expected to be large — that learns
what Gemini Live would predict: the word error a mixture would produce given its
target and interferer. Regress the extractor against it.

### G1a vs G1b vs G1c — three different systems, and only two can train anything

| | input | output | trains the extractor? | what it is actually for |
|---|---|---|---|---|
| **G1a** difficulty model | mixture + enrolment | predicted LCF-WER | **NO** | curriculum, data curation, analysis |
| **G1b** scorer / reward | **extractor output** + target script | predicted LCF-WER | yes — 1 scalar per clip | cheap offline metric |
| **G1c** distilled judge | **extractor output** | **Gemini's transcript** | yes — ~100 tokens per clip | the training signal itself |

**Raise this at the next meeting before anything is built: the ask as worded is
G1a, and G1a cannot tune the extractor.** The extractor does not change the
mixture, so a function of the mixture has zero gradient with respect to the
extractor's weights. Predicting difficulty from a mixture is a genuinely useful
instrument — it is how you pick which trials to train on — but it is not a loss.
To be a loss the predictor must be fed **what the extractor produced**.

**Recommended shape: G1c, which subsumes the other two.** Run a distilled judge
on a mixture and you have G1a; score its transcript with the existing
`lcf_wer.py` / `icr.py` / `fabrication.py` and you have G1b, free.

### Why a transcript target and not a WER target

**A scalar is a starvation-level gradient.** One number per ~19.6 s clip, against
~100 reference words. Fitting it needs many labels, it hands the extractor a
single direction, and it is the classic reward-hacking target: the extractor
finds audio the scorer likes and the judge does not, and the scorer cannot say so
because it is off its own training distribution.

**A transcript is per-token supervision of the thing we actually want.** With a
distilled judge `G`, the extractor loss is the cross-entropy of the **true target
script** under `G` conditioned on the extractor's output — "make audio a
Gemini-like listener transcribes correctly". That is ASR cross-entropy, which
CLAUDE.md already lists as an allowed proxy; the novelty is only that the ASR has
been aligned to the judge's behaviour instead of being a generic Whisper. It is
an increment on a sanctioned path, not a new invention to defend from scratch.

### The measured facts that constrain the design

**All from disk. None estimated.**

1. **A free local listener already explains 68 % of the judge's per-trial word
   error.** `experiments/results/2026-09-15-judge-predictability/`, n=412
   (4 systems x 103 trials): pooled Pearson r 0.825, r² 0.680, Spearman 0.781.
   **The surrogate does not start at zero — it starts at r² 0.68 and that is the
   bar.** What the free proxy gets wrong is the LEVEL: mean absolute error
   **21.7 points**.
2. **Agreement is WORST on the best system.** WeSep — cleanest audio, lowest word
   error — falls to r² 0.453 while the three ~62 % systems sit at 0.70–0.72.
   **The free proxy degrades exactly in the regime an improving extractor moves
   into**, which is the argument for training one, and also the warning that a
   surrogate trained on today's mediocre outputs will be weakest on tomorrow's
   better ones.
3. **The offset is not a constant.** The ASR reads 3.9–4.9 points higher than the
   judge on both baselines and 7.4 higher on WeSep, but **1.1 points LOWER** on
   struct-e12. One calibration constant cannot fix it.
4. **The judge's own test-retest noise is UNMEASURED and it bounds everything.**
   Three clips in the cache carry repeats (5 calls each); one moved **16.0
   points**. If the judge's self-disagreement is ~15 points MAE, the whole
   surrogate project is competing for ~6 points of headroom against the free
   proxy's 21.7. **Nobody knows this number and it costs ~$0.40 to get.**
5. **Labels are cheap in money and expensive in wall-clock.** ~$0.0013 per clip
   (~53 cents over ~412 clips, project-state.md) at **10 requests/minute**
   (103 clips in 11–14 min, measured, `run_times.md`). So 5,000 labels ≈ **$6.50
   and 8.3 h**; 20,000 ≈ **$26 and 33 h**. **Check whether the prepay tier lifts
   10 rpm — that single fact sets the whole schedule.**
6. **2,636 estimate clips are already rendered on `sir0_val` across 19 system
   variants**, of which 412 are already judged. The remaining **2,224 cost ~$2.90
   and ~3.7 h** to label and need no inference: the mask post-processing sweeps
   (hysteresis, floor, follow), state-e6, noresidual, struct-e12, WeSep and both
   baselines are 19 distinct points in artefact space on the same 103 trials.
7. **There is no local GPU** (`run_times.md`). Training happens on a Kaggle T4
   under a 12 h session cap. **"Quite large, trained locally" is not available.**
   The realistic route is to inherit size from a pretrained checkpoint —
   fine-tune `whisper small.en` (already a project dependency) or a WavLM/wav2vec2
   encoder on the Gemini labels — which turns "learn what Gemini would say" from a
   from-scratch problem into domain adaptation with a few thousand labels.
8. **The judge scores ~19.6 s clips; training uses 4.008 s crops.** A surrogate
   trained on clips and applied to crops is out of distribution on every step.
   **Buy the training labels on 4 s crops** (same price per call) and keep clip
   labels only for validating that the surrogate predicts the real metric.

### Staged plan, each stage a decision point and each one a result on its own

| stage | what | cost | kills the project if |
|---|---|---|---|
| **0** | predictability floor — DONE, free | 0 | — (r² 0.680, MAE 21.7) |
| **1** | **judge test-retest noise floor**: k=5 repeats on ~100 clips spanning easy→hard | **~$0.65, ~1 h** | self-MAE ≈ 20 pts: there is nothing left to learn and G1 stops here |
| **2** | label the 2,224 rendered clips + a synthetic corruption ladder | **~$3, ~4 h** | — |
| **3** | fine-tune `small.en` on (audio → Gemini transcript); score r²/MAE **on a held-out SYSTEM** | 1 T4 session | held-out-system r² ≤ 0.68: it learned the systems, not the judge |
| **4** | CE of the true script under the distilled judge as an extractor loss term | 1 training run | — |
| **5** | trust region: re-judge 103 clips every N epochs, plot predicted vs actual | $0.13, 14 min each | divergence ⇒ buy labels on the new outputs, retrain, continue |

**Stage 1 before Stage 2.** It is the cheapest thing on this page and it is the
only one that can say the project is not worth doing.

### Where the training distribution comes from, and why it is the hard part

**The failure that kills reward models is distribution shift**, not accuracy: the
surrogate is trained on outputs of today's extractor and then asked to score
outputs of an extractor that has been optimising against it. Three sources, in
increasing order of what they buy:

1. **The 19 rendered variants** (fact 6) — real artefacts, already on disk, but
   only 103 distinct trials.
2. **Synthetic corruption ladders** — take the clean target and apply mix-back at
   a range of α, band drops, spectral holes, musical noise, level scaling,
   clipping. **Free to generate, unlimited trials, and it covers the artefact
   space deliberately rather than by accident.** This is the cheapest way to make
   the surrogate robust where it matters.
3. **Online re-labelling** (Stage 5) — the only thing that actually tracks the
   extractor as it moves.

**Trial diversity must come from `sir0_train`, never from `sir0_privval` or
`eval_private`.** Those two are the entire remaining holdout (decisions-m4.md
2026-09-15) and touching them for labels destroys the last defensible number.

### Registered before running

- **Declare the effect size.** M6 already requires this and the last three arms
  ignored it. State what LCF-WER gain counts as success BEFORE Stage 4, against
  the ~3-point paired floor and the still-unmeasured seed spread (J5).
- **Held-out-system evaluation is the honest test of the surrogate**, not
  held-out-trial. In use it must score a checkpoint it has never seen.
- **Every Gemini call in Stages 1–5 is a training-time call** and records model
  ID, prompt and date exactly as a judge call does. `judge.py` already does this,
  which is why no new logging is needed.
- **Every claim says *optimised for Gemini*, never *generalises to live models*.**

### What it costs against what is already on the page

**This does not fit alongside the 2026-09-15 menu.** Stages 0–3 are roughly a
week; Stages 4–5 are a training run plus iteration, against **4 weeks to the
2026-10-14 freeze** with O1/O2/O3 outstanding and M6 unrun. Taking G1 means
dropping D13 and J5, i.e. M5's scoped deliverable and the seed-variance
measurement. **That is a supervisor decision, not one to take here.**

**The argument for taking it anyway:** the last three arms (state teacher, D17,
the data hypothesis) all returned nulls, and G1 is a result either way — "how
well can a local model predict a live model's listening, and does regressing on
it help?" is publishable as a negative. **The argument against:** it puts the
thesis's stated primary contribution (a gaming-resistant metric) and its
replacement (the surrogate) on the same four weeks.

### G1c concretised — exact inputs, outputs, and where the term goes

**Raised 2026-09-15 (Grant): "backbone + a head fine-tuned by Gemini — but what
are the input and output, and how does it reach the extractor?"**

#### The surrogate has two lives, and the input/output differ between them

**Life 1 — TRAINING THE SURROGATE. Fixed audio; the label is what Gemini said.**

| | |
|---|---|
| input | any 4.008 s audio crop, `(B, 64128)` at 16 kHz |
| label | **Gemini's transcript of that same crop** — its errors, its leakage, its inventions |
| trained | the head only; backbone frozen |
| learns | *how this listener mishears* |

**Life 2 — TRAINING THE EXTRACTOR. The surrogate is frozen; the audio moves.**

| | |
|---|---|
| input | `s_output`, the extractor's waveform, `(B, 64128)` — WITH gradient |
| target | the clean stem `s_target`, already in the loss call |
| trained | the extractor; surrogate weights frozen |
| asks | *now do not mishear* — and pushes that back into the audio |

**This is the whole idea in one line: learn to mishear like Gemini, then ask the
extractor to make audio that defeats the mishearing.** The labels are Gemini
transcripts; the extractor's target is never a Gemini transcript.

#### The backbone: `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`

**No new dependency — torchaudio 2.11.0 is already installed and ships it.**
16 kHz native, matching `sample_rate`, and it already carries a trained CTC head
over a **29-symbol character vocabulary** (`'-', '|', A-Z, '`). So the starting
point is a working ASR whose head is fine-tuned onto Gemini behaviour, which is
exactly the proposed shape and not a from-scratch build.

**Do NOT use Whisper for the in-loop term.** Its encoder takes a fixed 30 s
window, so every 4.008 s crop would cost a 30 s forward pass — ~7.5x wasted
compute on every training step. `faster-whisper` is worse than unsuitable: it is
CTranslate2, an inference engine with no autograd at all. Whisper stays where it
is, as the offline stand-in listener.

Shapes, for the 4.008 s crop (conv stack strides 5,2,2,2,2,2,2 = 320, i.e. 50 Hz):

    waveform      (B, 64128)        B = 6 at batch_size 3 x both_directions
    features      (B, 200, 768)
    CTC logits    (B, 200, 29)

#### Two heads on the one frozen backbone

| head | output | trained against | used for |
|---|---|---|---|
| **T** transcription | `(B, 200, 29)` CTC logits | CTC vs **Gemini's transcript** | **the gradient path** |
| **S** score | one scalar | Huber vs measured per-trial LCF-WER | reading only, NEVER in the gradient |

**Head S is the instrument, not the trainer.** It answers "is the surrogate
actually predicting the judge" (against the r^2 0.680 / MAE 21.7 floor already
measured) and it gives free checkpoint selection. It must never be
backpropagated: a scalar is one number per clip against ~100 reference words, and
it is the textbook reward-hacking target.

#### The extractor's loss term, and why it needs no text alignment

**Option 1, SHIP THIS FIRST — "transcribe like the clean stem does".**

    L_gem = D( HeadT(G(s_output)) , HeadT(G(s_target)) )

`D` = KL over the frame-wise character posteriors, or L1 on the logits.
**No ground-truth text is needed anywhere**, because the reference is the clean
stem, which `losses.py` already receives. That matters: the loader yields 4 s
crops with no text, so anything text-based needs per-crop alignment work first.

**Option 2, LATER — "get the words right".**

    L_gem = CTC( HeadT(G(s_output)) , words spoken inside this crop )

More tolerant — any audio that transcribes correctly scores well, however it
sounds — but it needs the crop's text. That is buildable: the loader already
returns `crop_start` and `trial_id`, and the manifest carries `target_utts` and
`target_onsets_s`. A middle route avoids the alignment entirely: greedily decode
`HeadT(G(s_target))` and use THAT as the crop's pseudo-label.

#### Where it goes: one term beside `L_MR`, taking the same two tensors

`scripts/train.py:723-731` already computes exactly what is needed:

    s_output, mask = model(mixture, enrollment, return_mask=True)
    loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent, ...)

So `L_gem` enters `LossBSRNN.__call__` in the **present branch**, next to `L_MR`,
reading the same `s_target` / `s_output` pair and gated the same way — on an
absent crop there is no clean stem to listen to, and `L_abs` already owns that
case.

**The defensible framing, and it is the reason to build it this way.** `L_MR`
already compares output to target in a feature space; it just happens to be four
STFT magnitudes. **`L_gem` is `L_MR` with a learned, Gemini-aligned feature space
instead of a hand-chosen one.** One term, one weight, `w_gem = 0` reproduces
today's objective exactly, and the weight is DERIVED by the same procedure as
`w_g` and `w_struct` (`derive_w_g.py`), never chosen by hand.

#### The extractor does not change, and that protects the latency result

**The surrogate is training-time only.** At inference the shipped model is the
same 7.19 M parameters at RTF 0.528 and 162 ms. Nothing in the streaming or
latency story moves. This is the strongest practical argument for the whole
approach over changing the architecture.

#### Registered before building — the things that will actually bite

1. **Step-time and memory cost are UNMEASURED.** A base backbone runs forward AND
   backward on every step (frozen weights still pass gradient to the input),
   against a current 0.674 s/step at batch 3. **Measure with
   `scripts/profile_step.py` before committing** — and do not quote a projection
   as a measurement. Gradient checkpointing on the backbone is the mitigation if
   the T4's memory will not take it; E5 already records activations as the
   binding constraint.
2. **The front end must be torch ops end to end**, or the gradient stops. This is
   satisfied by the torchaudio bundle and is exactly what `faster-whisper` fails.
3. **Per-clip waveform normalisation inside the backbone destroys level
   information.** Wanted, not a bug — it means `L_gem` cannot be gamed by getting
   louder — but `L_gain` must stay in the objective to cover level.
4. **Train the surrogate on 4 s crops**, the same length it will see in the loop.
   A surrogate trained on 19.6 s clips is out of distribution on every step.
5. **Cache backbone features when training the head.** The audio is fixed in
   Life 1, so features are computed once and head training becomes minutes. In
   Life 2 the audio changes every step and nothing can be cached.
6. **Validate Head S on a held-out SYSTEM, not a held-out trial.** In use it
   scores checkpoints it has never seen.

### G1 REFRAMED — 2026-09-16. A surrogate answers scarcity, and scarcity may not bind

**Raised by Grant: "how do I tune to a transcription tool that has its own biases
and flaws, and how do I do it with Gemini if tokens are not scarce?"**

**The correction to G1 as proposed.** A differentiable surrogate exists for ONE
reason: you cannot backpropagate through an API. That is a constraint on
*gradients*, not on *tuning*. **There are four routes by which a black-box
listener's signal can reach a system, and only the last needs a surrogate.**

| level | what moves | needs the judge differentiable? | hacking risk |
|---|---|---|---|
| **1 selection** | which artefact you keep (epoch, config, trials) | no | none — no pressure on the audio |
| **2 parameter search** | non-learned knobs, scored directly by the judge | no | low — few parameters |
| **3 imitation of winners** | **the weights** — best-of-N, then retrain toward the winners | **no** | moderate |
| **4 differentiable surrogate** | the weights, through backprop | yes | high |

**Level 3 is the one that was missed.** Generate N candidates per trial, let
Gemini rank them, fine-tune the extractor against the WINNING audio using the
loss that already exists. The judge's preferences reach the weights and Gemini is
never differentiated. This is expert iteration / rejection-sampling fine-tuning.
**G1's surrogate drops from "necessary" to "last resort".**

#### Tuning to a biased listener — the two things that make it defensible

1. **Keep an unoptimised second listener as the control.** `small.en` is still
   untouched by training. If tuning to Gemini also moves Whisper, intelligibility
   improved; if it does not, we fitted Gemini's quirks. **Both are reportable, and
   this partially recovers what the 2026-09-15 withdrawal cost** — the holdout
   moved from the model to the data, and it can also move to the LISTENER.
   Measured baseline for that check already exists: r^2 0.680, MAE 21.7
   (`2026-09-15-judge-predictability`).
2. **Separate systematic bias from stochastic noise.** Bias is tunable; noise
   never is. One clip moved **16.0 points across five identical calls**, so a
   label built from k=1 is part noise. **With a free budget the right purchase is
   REPEATS, not more trials** — aggregates already have SEM ~0.5 over 103, but
   every method above consumes PER-TRIAL labels, and those are the noisy ones.

#### What "no shortage of tokens" does and does not unlock

- **Does:** mass labelling, k>=3 on everything, best-of-N at large N.
- **Does NOT:** Gemini inside the gradient loop. ~8 s per call against a 0.674
  s training step. That is latency, not cost, and no budget removes it.
- **UNVERIFIED AND IT SETS THE WHOLE PLAN.** `judge_gate.yaml` records
  `tier: "free"`, 10 rpm / 1500 rpd, both annotated *"VERIFY in AI Studio --
  third-party figure"*, while `project-state.md` says AI Studio **prepay**. These
  disagree. At 10 rpm, 100k labels is 167 h; at 1000 rpm it is 100 minutes.
  **Check the console, and check whether batch mode is available for this model
  — it is built for exactly this and usually runs far above the interactive
  limit.** Minutes of work, and it decides everything below.

#### The experiment to run first: the alpha-oracle

**It attacks the project's biggest measured failure using Gemini as the teacher,
needs no retraining for the diagnostic, and no surrogate.**

The #1 finding against our model is that **it applies one transform to
everything** — SIR/SAR flat from easy to hard while the outcome swings −4.2 to
+23.1 points (decisions-m3.md 2026-09-01). D11's mix-back is the knob:
`s_alpha = alpha * s_hat + (1 - alpha) * x`, **zero added latency, no retraining,
every alpha from one forward pass.** Not yet built — `postprocess_mask.py` does
hysteresis sharpening, not mix-back — but it is a multiply-add.

1. Sweep `alpha` over ~5 values on N trials (one forward pass, trivial DSP).
2. Score every `(trial, alpha)` through the judge at **k>=3**.
3. Take `alpha*(trial)`, the per-trial best.

**Read it:**

- **`alpha*` roughly constant** ⇒ nothing trial-dependent to learn. Ship the
  global knob; "tuned to Gemini" is one honest number, cheaply obtained.
- **`alpha*` varies with the trial** ⇒ **Gemini has just handed us a supervised
  target for a per-frame gate.** That is D11 strategy 1 (learned,
  input-conditioned alpha), it directly attacks "the model cannot tell the cases
  apart", and the labels came from the judge.

**This is NOT the patch D11's objection rejected.** That objection stands against
a global constant presented as a fix — it trades hard-trial gains for easy-trial
harm and gives the model no new capability. Here the sweep is (a) a probe for what
the judge actually wants and (b) a way to MANUFACTURE TRAINING TARGETS for the
principled per-frame version. Report it that way or the objection applies again.

#### Revised order

1. **Verify the rate limit and batch availability.** Minutes. Gates everything.
2. **Judge test-retest noise, k=5 on ~100 clips.** Still the gate: if the judge
   disagrees with itself by ~20 points there is nothing to tune to.
3. **The alpha-oracle sweep.** Either outcome is a result.
4. **Learned per-frame alpha head**, if step 3 says alpha is trial-dependent.
5. **Expert iteration** (best-of-N over alpha x hysteresis x checkpoints, retrain
   on winners), if time remains.
6. **G1c's surrogate LAST**, and only if a signal is needed inside the gradient.

**Carry to every claim either way: *optimised for Gemini*, never *generalises to
live models*. The Whisper control is what says which of the two we achieved.**

### J6 — 2026-09-16. Tuning the LISTENER: a second instrument, never a parameter

**Raised by Grant: "can the Gemini side be tuned to best account for my system?"**

**The trap, stated first.** LCF-WER is *what the live model reports*, so the live
model IS the transcriber. There is no separate frozen scorer that could grade a
tuned receiver. **Tuning the listener until it is kinder to our audio, then
reporting one number, games the benchmark more completely than anything the
2026-09-15 withdrawal allowed** — that withdrawal kept the DATA held out; this
would remove the last fixed reference point in the project.

**The resolution, and it costs nothing.** A listener configuration is an
**instrument, not a parameter**. Changing it does not tune anything: it creates a
SECOND instrument, on which **every system is re-scored, floor and ceiling
included**. Two tables, and the comparison between them is itself a finding.
`metric-definitions.md` already treats the prompt, the normaliser and the gate
this way; decoding configuration simply joins that list.

#### DEFECT, and it must be fixed before any of this is attempted

**The cache key does not include the generation config.**

    f"{model_id}@{backend}|{prompt_sha}|{trial}|{file}|{audio_sha}|r{repeat}"

Model ID and prompt hash are in it, so changing the prompt correctly forces
anchors to be re-bought. **Change the temperature and the cache silently serves
answers produced at the old setting.** Fix it BACKWARD-COMPATIBLY — fold the
config into the key only when it is non-default — or all 653 paid-for responses
are invalidated at a stroke.

#### The instrument is currently running on server defaults

`Judge._call_once` sets **no temperature, no seed, no top_p, no thinking budget**.
Everything is whatever the server picks.

**This is the likeliest source of the 16.0-point spread** measured across five
identical calls on one ambiguous clip (2.9 on another, 0.0 on the ceiling), and
that spread is exactly what blocks per-trial claims — project-state.md "Cannot":
*per-trial needs k>=3 and averaging*.

| knob | now | expected effect | verdict |
|---|---|---|---|
| **temperature / seed** | **unset, server default** | likely large — the noise source | **a REPAIR, not tuning** |
| **k-sample consensus** (vote over k transcripts) | k=1 on estimates | attacks per-trial noise directly | legitimate as a second instrument |
| **thinking budget** | unset | unknown; plausibly large on degraded audio | legitimate as a second instrument |
| prompt wording | frozen `d118b7d3bf30` | **MEASURED LOW** — cross-prompt range 18.0 against same-prompt noise 16.0 | answered; do not spend time here |
| enrolment audio as context | no | probably large | **FORBIDDEN by J3** — hands the extractor's job to the judge |
| fine-tuning Gemini on our outputs | no | large | **destroys the premise** — a tuned Gemini is no longer the off-the-shelf live model the project claims to serve |

**Pinning temperature is a repair rather than tuning** because it is applied
identically to every system and reduces variance rather than shifting the mean in
our favour. **That must be VERIFIED, not assumed:** re-score floor and ceiling
first and report the mean shift alongside the variance drop. If the mean moves
materially, it is a new instrument and every number moves with it.

#### The version that is a contribution rather than a problem

The project is a **pipeline**: extractor -> live model. In deployment both ends
get tuned, and **"where is the headroom, the front end or the receiver's
configuration?" is a real question nobody in TSE asks.**

Run it as systems x listener-configurations, every cell re-scoring its own floor
and ceiling. The ceiling is 1.05 % on clean audio, so listener tuning can buy
almost nothing there — **the whole question is whether a better-configured
listener is more ROBUST TO EXTRACTOR ARTEFACTS**, which is
`metric-definitions.md` 1's hypothesis approached from the other side.

**If a listener configuration closes more of the floor-to-ceiling gap than our
extractor does, that is a finding about where effort belongs in live-model
pipelines.** Uncomfortable, cheap (scoring, no training), and honest.

#### Order

1. **Fix the cache key, backward-compatibly.** Nothing else is safe until then.
2. **Pin temperature and seed; re-score anchors; report mean shift AND variance.**
3. **k-sample consensus as a declared second instrument.**
4. Do NOT tune the prompt, do NOT send the enrolment, do NOT fine-tune Gemini.

### 2026-09-16 — THE RECOMMENDED SEQUENCE. A recommendation, not a decision

**Four weeks to the 2026-10-14 freeze.** Ties together G1, G1c, the G1 reframe
and J6 above. **Dropping D13 is a real decision and is NOT taken here.**

**The convergence worth noticing: `project-state.md` "Next" item 1 already names
the mix-back sweep as the cheap route to M6's missing near-tie.** Free tokens do
not introduce a new idea — they promote that sweep from a *measuring instrument*
to a *teacher*, because every `(trial, alpha)` cell can now be scored densely
enough to read `alpha*` per trial.

#### Everything in this conversation bottlenecks on ONE number

Per-trial judge noise blocks all of it: per-trial claims (project-state
"Cannot"), a surrogate (noise cannot be fitted), `alpha*(trial)` (a per-trial
quantity), best-of-N (per-trial ranking), and every delta's error bar. **It is
unmeasured at n>=3 clips, and it may be a config line** — J6 found the instrument
running on server-default temperature. **Quieten the instrument first and
everything downstream gets cheaper at once.**

#### Tier 1 — this week. Cheap, unblocking, none of it optional

| | what | cost |
|---|---|---|
| 1 | verify the real rate limit + batch availability (J6, G1 reframe) | minutes |
| 2 | fix the judge cache key backward-compatibly (J6 defect) | small |
| 3 | pin temperature/seed; re-score anchors; report **mean shift AND variance** | ~600 calls |
| 4 | judge test-retest, k=5 on ~100 clips — the noise floor | ~$0.65 |
| 5 | **O1** (struct control on `sir0_privval`) and **O2** (write D17 up) | 1.2 h + 1 h |

#### Tier 2 — the one new experiment: the alpha-oracle sweep

**It does three jobs at once**, which is why it beats every other candidate:

1. attacks the top measured failure — one transform applied to everything;
2. **produces M6's missing near-tie family**, from one checkpoint, no retraining;
3. if `alpha*` is trial-dependent, Gemini has handed us a supervised target for
   the per-frame gate (D11 strategy 1).

**The methodological prize, and it is large: every member of the alpha family
shares one set of weights.** So within-family comparisons carry NO training
noise, and **J5's unmeasured seed variance cannot confound them** — the blank
underneath every delta reported so far simply does not apply here. No retraining
arm can say that.

#### Tier 3 — conditional. Learned per-frame alpha head, one training run

Only if Tier 2 says `alpha*` varies by trial. This becomes M5's deliverable in
place of D13, and is better motivated than D13 because the target came from the
judge rather than from an oracle mask.

#### Tier 4 — in parallel, and it is the part that cannot be cut

M6's stratified tables (**O3**) and the five findings that already exist and are
unwritten. **The thesis risk is not that the extractor is weak — that is itself
an honest, evidenced result. The risk is that M6, the stated central finding,
never runs.**

#### Recommended DROP, each with its reason

| item | why |
|---|---|
| **G1c surrogate** | answers scarcity; scarcity may not bind. Last resort, not first |
| **D13 per-band gate** | Tier 2 delivers M6's near-tie family more cheaply and with no training noise |
| **D18 / D19** | diagnostics of the mask-shape line, which Tier 2 routes around |
| **J5 seed variance** | deferrable *only because* Tier 2 is retraining-free. Required again the moment Tier 3 runs |

**Carry regardless: *optimised for Gemini*, never *generalises to live models*,
and `small.en` stays untouched as the control listener that tells the two apart.**

### AMENDMENT 2026-09-16 — the four routes are not four ways to do the SAME thing

**Correcting an implication in the G1 reframe above.** That section said the
surrogate "drops from necessary to last resort". That is right about SEQUENCE and
misleading about SUFFICIENCY, and the difference matters.

**The four routes differ in HOW MUCH OF THE SYSTEM the judge shapes.**

| route | what Gemini decides | parameters Gemini shapes | are the extractor's weights tuned? |
|---|---|---|---|
| selection | which artefact is kept | 0 — a choice among N | **no** |
| alpha sweep, global | one scalar | **1** | **no** |
| alpha sweep -> learned per-frame head | the head's training target | ~thousands | partially |
| **expert iteration** | which of N candidate outputs is best | **all 7.19 M** | **yes — within the span of the candidate set** |
| **differentiable surrogate** | the gradient itself, every step | **all 7.19 M** | **yes — including directions no candidate set contains** |

**So the alpha sweep is barely "tuning to Gemini" at all** — it is one knob whose
value Gemini picks. It was recommended because it is cheap, attacks the top
measured failure, and hands M6 its near-tie family. **Those are reasons of
project risk, not of fidelity to the stated goal.** Say so when presenting it.

**Expert iteration's ceiling, stated plainly.** Candidates built from mix-back and
mask post-processing are all variants of ONE separation, so retraining on the
winners can only teach the model to internalise post-processing it could already
have applied. **It cannot discover a separation that was never in the candidate
set.** Widening that set — stochastic masks, several checkpoints, several
architectures — raises the ceiling; it does not remove it.

**Which is exactly why the surrogate exists, and why the supervisor proposed it.**
A gradient explores continuously, so it can push toward outputs no candidate set
contains. **If the goal is genuinely "tune the extractor to Gemini", the surrogate
is the only route that fully does it. There is no cheap substitute for that
property.**

#### The middle path not previously named: surrogate as RANKER, never as gradient

Train G1c's **Head S only** (scalar LCF-WER, the cheap half), then use it to rank
thousands of candidates locally at zero API cost, keep the winners, retrain on
them, and **verify the winners with real judge calls**.

- Gemini labels train the ranker; the ranker supplies candidate throughput the
  rate limit could never buy.
- **A wrong ranker is far safer here than in a gradient loop**: it only re-orders
  a fixed candidate set, and every winner is checkable against the real judge.
- It needs no differentiable front end, no backward pass through a backbone, and
  no step-time cost in training.

**This is the best fidelity-per-risk point on the page** and it was missing from
the sequence above.

---

## 2026-09-21 — O4, OBLIGATION: we ARE the challenge baseline, and we reproduced its consensus independently

**Report this. Our architecture is the SLT 2026 REAL-TSE Challenge baseline
`BSRNN_TFMAP_CAUSAL`, and our three "negative" M5 results are that challenge's
published consensus, arrived at without knowing it.** Writing them up as
project-specific failures understates them; they are an independent
reproduction, which is a stronger claim and costs nothing to make.

### What we found on our own, and what the challenge found

| our result | date | the consensus it reproduces |
|---|---|---|
| 14.73 M bought **exactly zero** separation gain (-2.9020 vs -2.9000) | 09-21 | top teams used *nearly the same extractor* and won on data, recipe and post-processing, not on size |
| data scaling +0.32 dB/doubling with **no downstream conversion** | 09-04 | careful data simulation and real-data adaptation are the reported levers |
| SI-SDR moves the wrong way against content fidelity | 09-04, 09-12 | non-intrusive metrics were **gamed** mid-challenge and the organisers now recommend banning them during development |

**The third row is the one that matters for the thesis.** The organisers had to
change their official perceptual metric mid-challenge after a team inflated
SpkSim and DNSMOS with adversarial waveform perturbations without improving
extraction. That is dated, attributable, first-hand evidence that a
gaming-resistant content metric is a real contribution -- exactly what this
project exists to build. Cite it in the metric chapter, not as a footnote.

### VERIFIED LOCALLY on 2026-09-21 -- safe to report

`../wesep_pretrained/tfmap_context_causal_100/`, read directly:

- **Its speaker cue is `tfmap` + `context` (embed_dim 512, `fusion: multiply`),
  fed by a pretrained `ECAPA_TDNN_GLOB_c512`.** `spkemb` and `usef` are both
  disabled. **`src/models/conditioning.py` has only `TFMap` and
  `TFMapInjector` -- no encoder, no contextual embedding, no identity path.**
- **Parameters: `sep_model` 18.868 M + `spk_ft` 14.597 M = 33.465 M**, against
  our 7.19 M. The separator *hyperparameters* match ours (`causal: true`,
  `feature_dim: 128`, `num_repeat: 6`, `win: 512`, `stride: 128`) but the
  realised separator is 2.7x ours, so **this is NOT a controlled ablation.**
- **It trained clean:** `data/clean/train-100`, `noise_prob: 0`,
  `noise_enroll_prob: 0`. No noise, no reverb, no low-SIR curriculum.

### The comparison has FOUR confounds. State all of them or state none

Ours 88.2 WER vs WeSep 49.4 at SIR < -5 differs in capacity (4.7x total),
causality (it is **not streaming** -- global normalisation, ~6 orders of
magnitude of future leakage, `decisions-m3.md` 2026-09-03), training data
(clean vs our noise+reverb) and conditioning. **That comparison alone cannot
establish that the cue is the bottleneck, and must never be written as if it
does.**

What survives as evidence for the cue hypothesis is narrower and stands on its
own: (a) WeSep's *own* published cue ablation, which holds backbone, data and
schedule fixed and varies only the cue; (b) our direct measurement of our cue
(78 % rank-1, `corr(alpha_t, mixture loudness) = 0.990`, no negative evidence
anywhere in the path); (c) the injector's per-band `LayerNorm(bw)` deleting
`alpha_t`, which mechanically explains D4a's near-zero gates.

### NOT YET VERIFIED -- do not put these in the report until checked

Reported by a literature search on 2026-09-21, not read first-hand: the
challenge overview (arXiv 2607.15198) and its verbatim conclusion; the baseline
TER 0.808 on EVAL-2; the WeSep cue-ablation table (6.87 / 12.98 / 14.15 dB
causal, accuracy 79.9 -> 95.9 %); the gaming incident's details. **Read the two
papers before citing any of it.**

### The standing rule still applies

Different data, different metric, different protocol. Nothing of ours is
comparable to a published REAL-TSE number. The claim licensed here is
"we independently reproduced their conclusion", never "we match their results".
