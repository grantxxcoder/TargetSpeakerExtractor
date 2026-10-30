# Open decisions

**Written 2026-08-10.** Groups A–C (pre-generation data decisions) are closed, C2 last on 2026-08-30, with reasoning in `decisions-m0.md` by date. D holds open *modelling* questions, E training speed, J the *judge and metric*. Decisions taken go to their milestone log: `decisions-m1.md` (architecture), `decisions-m2.md` (training), `decisions-m3.md` (conventional evaluation), `decisions-m4.md` (metric and judge).

---

## Still open

- **J2: which judge, and the open-weight anchor.** Cost answered (~$5-25). Reproducibility and the candidate gate remain. **Unblocked since J1 closed 2026-08-31.** See Group J.
- **J3: closed 2026-09-03.** Threshold `count>=2`, signed off into `metric-definitions.md` §3.2. k changes no conclusion (floor 57.0 to 42.9 across k=1/2/3/5, ceiling 0.0 throughout), which discharges the sensitivity requirement. decisions-m4.md 2026-09-03.
- **D11: inference-time mix-back.** Run 2026-09-01 as the screening test. alpha=1 wins globally, but the per-difficulty optimum spans 0 to 1, which motivates adaptation, not a global gentleness shift. See Group D and `decisions-m3.md` 2026-09-01.
- **D12: mixture of experts over masking behaviours.** Viable at +11 % params with a shared trunk. `K > 2` deferred until the judge work is done. See Group D.
- **D13 (decided 2026-09-01): build the per-band gate** as the `n_experts = 2` case of D12, with the identity as expert 0. See Group D.
- **D14 (proposal): per-frame speaker-state supervision.** Free four-state labels from the rendered stems, an auxiliary head, a frozen state detector as a training-only loss (zero inference cost, no architecture change), and the state posterior as supervised input to D13's gate controller. **Gated on D10's free ICR measurement.** Corrects a recorded ceiling: M6's 2.2-point oracle was per trial, not per frame. See Group D.
- **J4 (proposal): a metric *system* (normalised requirement axes, composed and plotted) rather than metrics in isolation.** Diagnosis accepted. Polygon-area ranking rejected as order-dependent. Normalise floor-to-ceiling, not by percentage change. See Group J.
- **E7 (proposal): use the second T4.** Kaggle's "GPU T4 x2" has two cards; the code uses one, so the other has idled on every run. **~2x** for about half a day's work. Main risk: `DataParallel`'s `module.` state-dict prefix. **Conditional on E8's `batch_size` 3 to 6**, since at batch 3 the split halves an already-saturated card. Run *after* the 9,955-trial run so the data-scaling curve does not move two variables at once. See Group E.
- **E8 (proposal): raising `batch_size`, and a bug capping it at 3.** The notebook's batch probe runs fp32 but training runs fp16, so it writes the fp32 ceiling (batch 3) into an AMP run that fits 6. `profile_step.py` had the same fp32-warmup bug, already fixed. Fixing the probe is E8's cheap half and unblocks E7. Memory is almost all activations, ruling out 8-bit optimisers (~43 MB of ~13 GB) and audio compression (cannot touch GPU memory). Beyond 6 needs `hop` 128 to 256. See Group E.
- **D18 (proposal): why `L_struct` flattened the mask.** The shape term is an L1 to a target the model cannot predict, whose optimum is a flat mask. Free test, no training. See the 2026-09-15 menu.
- **D19 (proposal): run the flatness instrument on WeSep.** Is the volume knob ours or the task's? No training; the cheapest redirect available. See the 2026-09-15 menu.
- **J5 (proposal): run-to-run training variance.** Ten training runs, all `seed: 42`, no replicate. This third noise source is unmeasured and M6 needs it. See the 2026-09-15 menu.
- **G1 (proposal): the Gemini tuner.** A local judge stand-in trained on Gemini labels, giving the differentiable signal the API cannot. Starting point: a free local listener already explains 68 % of the judge's per-trial error (r^2 0.680) but is 21.7 points out on level. See the 2026-09-15 G1 section at the end of this file.
- **O4 (obligation, raised 2026-09-21): we are the REAL-TSE Challenge baseline (`BSRNN_TFMAP_CAUSAL`), and our three negative M5 results independently reproduce that challenge's published consensus.** Report it as a reproduction, not a project-specific failure, including the organisers' mid-challenge metric-gaming incident (first-hand evidence for our metric contribution). The four WeSep-comparison confounds listed there travel with it. See the 2026-09-21 O4 section.
- **D5 (reopened 2026-09-21, directional test run 2026-09-22).** Demoted 2026-08-30 on `rel_movement`, which has magnitude but no direction, so cannot tell "moved toward the interferer" from "changed level". The directional test sharpens D5: the cue lands on the right voice 71.8 % of the time cross-gender and **52.6 % same-gender, a coin flip** (n = 38 crops, p = 0.73; gender gap chi2 p = 0.022, wider inside the balanced-SIR slice). A frozen ECAPA is trained to discriminate speakers within gender, which is this failure. **The registered acceptance test for 1c is the same-gender rate, not an aggregate word error.** Item 1a is built and passed its acceptance test. Full entry: `decisions-m2.md` 2026-09-22. The cue is a pitch detector with a loudness default, not a loudness meter. See `ranked-next-steps.md` items 1-2.
- **O1/O2/O3: obligations, not options.** Score the struct control on `sir0_privval` (**done 2026-09-15**, 48.90; `sir0_privval` then dropped from evaluation 2026-09-24, `decisions-m3.md`), write up D17, and produce M6's stratified tables. See the 2026-09-15 menu.

- **A1 needs sign-off only.** The reference is the full reverberant target: separate and denoise, do not dereverberate. A 0.6 s tail cannot be removed inside a 300 ms causal window, and trying trades residue for artefacts, which hurt recognition more. Dereverberation stays an ablation if time allows.

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
| C2 | task difficulty accepted as measured: floor **57.4 %** on `eval_public` `both` (n=230), **65.2 %** on `sir0_val` (n=103); ceiling ~6 %. Straddles the 60–80 % target band, which was an aim, not a constraint. **Open consequence: which set defines the benchmark.** The two differ by 7.8 points because `eval_public` keeps the target-louder distribution and `sir0` is symmetric | 08-30 |
| B10 | three enrollment tiers recorded per trial; eval pools redrawn. Executes B8's own documented contingency (60.2 % of speakers dropped out), not a reversal | 08-13 |
| B11 | report a latency decay curve, never cap T60. Largely defused by A1 | 08-13 |
| B12 | two regimes, sampler layer, no relational constraints. PR1/PR2 landed 08-14 | 08-13 |
| B13 | stratified reporting per condition, no combinations, 100 trials per bucket | 08-13 |

Two B12 items remain implementation, not decisions: `overlap_ratio` narrowing in `base` (needs supervisor agreement, one config line) and `length_mode`.

---

## D. Modelling — open, not blocking M1

*Added 2026-08-19. Group D extends this M0 data-decision file to open modelling questions, which have nowhere else to live. Decisions taken go to their milestone log, usually `decisions-m2.md`.*

### D1. Phoneme-template speaker cue as an alternative to TF-Map

**Status: idea, unscheduled. M5-scale (M5 is already marked CUTTABLE). Run the cheap precursor D2 first.**

**Problem.** TF-Map compares magnitude spectra, which are dominated by *what is being said*, not *who is saying it*. An interferer's "ah" resembles the enrollment's "ah", so the cue is partly phonetic. See `decisions-m1.md` 2026-08-19.

**Proposal.** Replace the raw enrollment frames (TF-Map's basis vectors) with a phonetically organised, speaker-adapted dictionary:

1. define a library of the language's phonetic units;
2. from the enrollment, build a spectral template for each unit this speaker produced;
3. for units absent from the enrollment, *predict* the speaker's realisation from those present, using similarity between sounds;
4. match mixture frames against this dictionary instead of raw frames.

**Refinement that makes it work (agreed 2026-08-19).** As stated, the interferer's "ah" still matches the target's "ah" template. It needs a contrastive term:

- `S_target`: similarity to *this speaker's* template for a unit
- `S_background`: similarity to a *speaker-independent average* template for the same unit, built across many speakers
- feed the network `S_target - S_background`

Phonetic content cancels, leaving "how much more target-like than average-speaker-like is this frame". **This is the GMM-UBM likelihood-ratio structure from speaker verification**, applied per frame and per band as a streaming extractor feature. That combination appears novel and is the part worth claiming.

**Related work: searched 2026-08-19, written up in full at `literature/novelty-review-contrastive-phonetic-cue.md`.** Novel as a combination; every ingredient has prior art. Closest is arXiv:2502.16611 (NeurIPS 2025), which contrasts positive against negative enrollments. However, its negatives are actual interfering speakers from the same recording, not a speaker-independent background, it compares embeddings rather than densely, and its TF-GridNet/BiLSTM backbone is non-causal. Step 3 has direct prior art in Weiss & Ellis (2010) eigenvoice speaker adaptation for separation, and steps 1-2 in phone-dependent NMF. Cite both prominently. The SLT 2026 REAL-TSE overview of all 24 submissions from 12 teams lists no contrastive or phonetic conditioning.

**Known obstacles.**

- *Phonetic labels.* Step 2 needs forced alignment of the enrollment, feasible offline here because `meta.json` carries `target_text` (exact ground-truth transcripts). A deployed system would lack the enrollment transcript, which weakens any generality claim and must be stated.
- *Enrollment sparsity.* English has ~44 phonemes. 5 s of speech holds maybe 15-20 tokens and perhaps a dozen distinct units. **Most of the dictionary would be predicted, not observed, so step 3 does the work.** First mitigation: a longer enrollment (`enrollment_length_s`, fixed at 5 s per `decisions-m0.md` 2026-08-12), which is a config change and a regeneration, not new code.
- *Step 3 is a speaker encoder in disguise.* Predicting unseen units properly needs a model of speaker space learned across many speakers. However, **the latency objection does not apply**: it is all enrollment-side and offline, and matching ~44 templates is *cheaper* at runtime than 628 enrollment frames. It gives encoder-quality conditioning without the streaming cost that ruled out Zhang et al.'s eq. 3.

**Open concern (raised 2026-08-19): model size.** Affordability, with or without an encoder, is unresolved and the main risk. Set a parameter budget before implementing. The model is 7.19 M against challenge baselines at 25-27 M, but a speaker-space model for step 3 could use all that headroom.

**Evaluation.** Compare against TF-Map on extraction quality *and* efficiency (parameters, RTF, added latency), the axis on which TF-Map beat the embedding variant.

### D2. Attention temperature in TF-Map — the cheap precursor to D1

**Status: unrun, one line of code. Run this before scheduling D1.**

Measured 2026-08-19: TF-Map's softmax weights are almost uniform, blending ~621 of 628 enrollment frames (max weight 0.00239, uniform 0.00159). This is **forced by the arithmetic, not the audio**. Magnitude spectra are non-negative, so cosine similarities lie in [0, 1], and a softmax over a range of 1 gives weight ratios of at most e ~ 2.7. The wesep reference has no temperature either, so this is the published behaviour.

Consequence: TF-Map's time-variation comes almost entirely from the energy recovery step, not the attention. It supplies "the target's average spectral shape, scaled per frame by how much energy the mixture has in that direction", not the frame-selective mechanism the NMF framing describes.

**Experiment.** `h = softmax(sim / tau)` with `tau` in {1.0 (current), 0.2, 0.05}. It tests D1's hypothesis, *more selective matching against enrollment content improves extraction*, in an afternoon. **If sharpening the existing mechanism does not help, a richer dictionary is unlikely to, and D1 should not be scheduled.**

Confirm the near-uniformity on several trials before writing it up. Only one is measured, though the argument above says it is structural.

### D3. RUN 2026-08-30. ANSWERED: conditioning is not the bottleneck

**Measured over 200 `sir0_val` crops: the cue moves 28.6 % on an enrollment swap and the output 48.2 %.** The network amplifies the cue, so conditioning is not what fails. Look at the separator or the objective. **D4a and D1 dropped, D5 demoted, D2 closed** (the softmax now blends 138 of 628 frames against ~620 before `tfmap_scale`). D3c ran alongside: cross-gender sensitivity 56.1 % vs same-gender 44.4 %, a weak sign of gender reliance. `scripts/diagnose_cue.py`, `decisions-m2.md` 2026-08-30.

`sir0_train` is balanced 680/680 same/cross-gender, which **caps but does not remove** the gender shortcut: pitch alone gets the cross-gender half right and coin-flips the rest, ~75 % correct with no enrollment. Pooled numbers hide this, so gender-stratified reporting stays on.

**D3b is still unrun**: replace the TF-Map channel with the clean target's magnitude spectrogram and train briefly, as an upper bound on conditioning.

### D4. Inject the speaker cue at every block, not once at the input

**Status: proposal, not scheduled. Cheapest large change available. Gated on D3a.**

**Structural problem.** `BSRNN_TFMAP.forward` concatenates the TF-Map as a third input channel, and `SubbandNorm` projects it once through a 1x1 conv. It must then survive **six `BSNet` blocks (twelve LSTMs) of residual mixing** to reach the mask head, never re-injected. That is a long credit path for a signal worth one third of one projection.

**Proposal.** Derive a fixed-length embedding from the enrollment and apply FiLM at each block: `z <- gamma(e) * z + beta(e)`. Cite Perez et al., AAAI 2018 for FiLM, and for the TSE multiplicative-adaptation precedent Delcroix et al., "Improving speaker discrimination of target speech extraction with time-domain SpeakerBeam", ICASSP 2020. About 400 k parameters from a 256-d embedding across six blocks, affordable within the 7.19 M / 25-27 M headroom.

**Why it is ranked first.** It is the largest divergence from TSE systems that demonstrably condition, and it leaves the cue unchanged, so it composes with whatever D3a says.

**Two variants. Run D4a, which needs no encoder, before D4b.**

- **D4a: re-inject the TF-Map itself at every block.** The TF-Map is already `(B, 1, F, Tx)`, frequency-shaped and time-aligned with the feature map, so it band-splits like the mixture. Project it per band to `feature_dim` and add it into each of the six `BSNet` blocks. **The cue stays parameter-free, so it cannot memorise anything**; only the projections learn. This tests dilution ("the cue is fine, the network is losing it") with no new concepts or failure modes.
- **D4b: FiLM from a learned fixed-length embedding.** Strictly stronger and riskier; needs D5's encoder. Only worth it if D4a moves `val_enrol_sens_db` and then stalls.

**Memorisation worry (raised 2026-08-28).** Speaker parameters risk the encoder learning the 1172 training voices instead of learning to extract. Three things defuse it; state them in the write-up:

1. **The splits are speaker-disjoint by construction** (`speakers_from:` in `generator.yaml`). Memorisation would show as train extraction improving while val does not, so *the experiment already detects it.*
2. **It is the standard setup in speaker verification.** x-vector and ECAPA-TDNN train with this closed-set classification loss and transfer to unseen speakers. The classification head is discarded at test time.
3. **Our data design breaks the likelier confound**, the encoder latching onto channel rather than voice (each LibriSpeech speaker has their own sessions). The enrollment is dry (A4, no room) and from a different book (B8), while the mixture is reverberant, so only voice can match.

**Asymmetry that settles the ordering.** A model over-using the cue can be regularised. One that ignores it cannot be regularised into using it. Ours is the second kind, so the risk is worth taking, behind D4a, which carries none of it.

### BUILT 2026-09-11 — D4a exists as code and as a runnable config. It REVERSES a recorded drop

`conditioning.TFMapInjector`, `BandSequenceModel.forward(x, cue, gates)`, the `tfmap_inject` flag on `BSRNN_TFMAP` and in `build_model`, `experiments/configs/bsrnn_tfmap_inject.yaml`, `tests/test_tfmap_inject.py` (20 tests, all passing; suite 467).

**This un-drops D4a (dropped 2026-08-30).** The drop was right on its own terms: `diagnose_cue.py` measured the cue surviving the stack (swap a stranger's enrolment, the cue moves 28.6 % and the output 48.2 %), so the *dilution* premise is not supported. What remains is headroom: 62 % of the output is still enrolment-independent. **Write this arm up as testing that headroom, never dilution.** The realistic prior is that it does little.

**One shared per-band projection, not six.** 37,698 parameters, +0.524 % on 7,189,644, verified by building both configs. Six separate projections would be 222 k (+3.1 %), adding expressiveness that could explain a gain by itself. Sharing presents the identical cue at every depth.

**Not parameter-matched to its control**, unlike D14's head A. +0.52 % is small but not zero, so a gain of that order is not attributable.

**Gates start at zero, so the arm begins as the baseline function.** A test asserts the two models agree to 1e-6 at the same seed. Cite ReZero (Bachlechner et al., UAI 2021). At gate 0 the projection gets no gradient, so the gates (one multiply from the loss) move first; this is tested.

**The gates are the measurement, so run it even expecting nothing.** 6 x 32 scalars show how much cue each block wants in each band. If they stay near zero, a second, different method confirms D3a, and D4b/D5 can be closed: a publishable negative result for one run. Log them every epoch.

**Cost: +2.7 % forward on CPU** (1.210 to 1.243 s per 4.008 s chunk, 4 threads, forward only, 3 reps). Not a training-step measurement; never quote it as one. Latency is unchanged in kind: the TF-Map is already causal per frame and every added op is a 1x1 convolution over time, asserted by a causality test.

**RNG forked at construction**, as with head A, so the arm and its control draw the same trials in the same order. A narrow fix, not the declined general one.

### D5. A speaker encoder with an auxiliary speaker-ID loss

**Status: proposal, not scheduled. Larger than D4 and subsumes part of D1.**

**Problem.** **No parameter in the model is devoted to speaker identity and no loss term rewards it.** TF-Map is deliberately parameter-free, and the M2 objective's four terms (`L_pres`, `L_MR`, `L_gain`, `L_abs`) cover signal level and content. SpEx+ (Ge et al., Interspeech 2020) is the standing evidence that the auxiliary speaker-classification loss is what makes conditioning stick.

**Proposal.** A small enrollment encoder producing a 256-d embedding, trained jointly with a cross-entropy over the ~1172 training speakers, feeding D4's FiLM. **The enrollment is complete before the stream starts, so this encoder may be non-causal at no streaming latency**; the objection that ruled out Zhang et al.'s eq. (3) does not apply.

**It completes the paper we already cite.** Zhang et al., ICASSP 2025 is *Multi-Level* Speaker Representation. We implemented the spectral level (eq. 2) and skipped the embedding level (eq. 3) for want of an encoder. With one, the write-up becomes "we implemented the spectral level, measured it insufficient, and added the embedding level the paper specifies".

**Non-negotiable constraints.** The encoder must be a different model family from the ASR proxy, and the judge must never appear in it in any form (CLAUDE.md). For a pretrained speaker model, record the family check rather than assume it.

### D6. Two levers that are already built and currently switched off

**Status: both are config changes, hours not days. Run alongside D3.**

- **`lookahead_frames` is 0.** `lookahead_shift()` is implemented and tested. The spec allows 200-300 ms; 16 frames is 128 ms. Ablate {0, 8, 16}.
- **The residual branch `R` is unbounded and unconditioned.** In `Estimator`, GLU bounds the mask but `res_heads` is a raw `Conv1d` added straight to the masked spectrogram, so it can synthesise output **ignoring both the mixture and the enrollment**. `residual_branch` is already a constructor flag. Give it an ablation arm like `ablate_w_m` / `ablate_w_g`.

### D7. Status correction — D2 was run without being logged as an arm

`tfmap_scale: 16.0` in `bsrnn_baseline.yaml` is D2's temperature (scale = 1/tau, so tau ~ 0.0625, sharper than D2's sharpest arm). Mechanically it worked: 619.6 of 628 frames effectively used before, the top 50 frames carrying ~59 % after.

**However, the gain cannot be attributed to it.** Enrollment sensitivity went from 2.6 % to 15 % while `tfmap_scale`, `both_directions` and the `sir0` split all changed. A `tfmap_scale`-only ablation arm would close D2.

**Consequence for D1.** D2's stopping rule was: if sharpening does not help, a richer dictionary is unlikely to. Sharpening helped but left 85 % of the output enrollment-blind, so D1 is not cleanly ruled out. It stays M5-scale against D4 and D6, which are hours to days.

## E. Performance and memory — open

*Added 2026-08-28. Group D is modelling; this is engineering. Anything decided goes to `decisions-m2.md`.*

### E1. `batch_size: 3` is a memory ceiling, not a preference

The config comment says batch "should be 12" on GPU. **In fp32 it cannot be.** Analytic activation memory saved for the backward pass (4 bytes, ~4x hidden for cuDNN gate buffers), at `T` = 497 frames, `K` = 32, `N` = 128, `H` = 192, 6 blocks:

| batch (trials) | examples | LSTM activations | fits 15 GB T4 |
|---|---|---|---|
| 3 | 6 | ~4.9 GB | yes |
| 6 | 12 | ~9.8 GB | tight |
| 12 | 24 | ~19.7 GB | **no** |

**These are computed, not measured** (`scripts/profile_step.py` measures them), but they explain the observed `batch_size: 3`. Correct the comment promising 12.

### E2. The band RNN is the dominant cost, and that is counter-intuitive

Per forward at batch 3, LSTM work splits:

    time_rnn  batch = B*K =  192, seq = T = 497           281 GFLOP
    band_rnn  batch = B*T = 2982, seq = K =  32, bidir    563 GFLOP   <- 67 %

**The "across frequency" RNN costs twice the "over time" one.** It looks cheap (32 steps against 497), but `BSNet.forward` reshapes to `(B*T, N, K)`, so it runs a separate 32-step bidirectional sequence *for every frame of every example*: an effective batch of 2982 at `batch_size` 3. That also makes it the larger half of E1's memory.

**Consequence: cost is linear in `T` through both RNNs**, via `time_rnn`'s sequence and `band_rnn`'s *batch*. Halving `T` roughly halves compute and memory.

### E3. The 3.7x that is not accounted for

Measured 5.84 s/step (3875.2 s / 663 steps, `docs/run_times.md` 2026-08-27). The analytic floor from E2's FLOPs, assuming LSTMs reach ~20 % of the T4's 8.1 TFLOP/s fp32, is **~1.6 s**. So **roughly 3.7x of the step is unexplained**: data loading, the 32-iteration Python band loops in `SubbandNorm` and `Estimator`, or low GPU occupancy at batch 3.

**Each has a different fix, and we do not know which it is.** Run `scripts/profile_step.py` (written for this) before optimising anything.

### E3b-E3f. MEASURED 2026-08-28 — CLOSED. Evidence tables for `decisions-m2.md` 2026-08-28

Conclusions, mechanism and consequences are in `decisions-m2.md` 2026-08-28 ("Training made 7x faster"). This section keeps only the measurements it cites.

**CPU profile** (`profile_step.py`, batch 1, CPU x4, fp32). 11.381 s/step, peak RSS 5100 MB, forward 34 % of wall.

    stft 0.0 %   tfmap 0.1 %   subband_norm 0.5 %   separator 95.8 %   estimator 3.5 %
      time_rnn  seq=497 batch=64    1.207 s   31.6 % of forward
      band_rnn  seq=32  batch=994   2.276 s   59.6 % of forward

E2's analytic prediction holds: the band RNN is 65 % of the two RNNs (59.6/91.2). Loader worst case (`--loader-only`): 0.382 s/batch for 6 examples at `num_workers=0`, 6.5 % of a 5.84 s step, and 0.044 s/batch on Kaggle at `num_workers=4`. **This is the only valid attribution**; the GPU one is void (below).

**fp32 vs AMP, batch 3** (T4 14.56 GiB, torch 2.10.0+cu128, 8 steps).

| batch 3 | s/step | peak GPU |
|---|---|---|
| fp32 | 4.741 | 12.20 GB |
| AMP (fp16) | 2.968 | 6.57 GB |

1.60x faster, 1.86x less memory. **Batch 3 is the fp32 ceiling**: 4, 5, 6 and 12 all OOM inside `band_rnn`'s `_VF.lstm`. `bsrnn_baseline.yaml`'s comment promising batch 12 is wrong by 4x.

**fp16 sweep** (`--amp-only`, 8 steps).

| batch | s/step | s/trial | peak GB | GB/trial |
|---|---|---|---|---|
| 3 | 2.971 | 0.990 | 6.57 | 2.19 |
| 4 | 0.926 | *0.232* | 8.72 | 2.18 |
| 5 | 4.969 | 0.994 | 10.87 | 2.17 |
| 6 | 6.028 | 1.005 | 13.02 | 2.17 |
| 12 | OOM | — | — | — |

Memory is exactly linear, 0.12 GB fixed + 2.15 GB per trial, so batch 7 needs 15.17 GB of 14.56 available. **Batch 6 is the fp16 ceiling.** Per-trial throughput is flat across 3, 5 and 6 (0.990 / 0.994 / 1.005, a 1.4 % spread over a 2x batch range). **The T4 is already saturated at batch 3**, which is why gradient checkpointing was withdrawn.

**Tensor-core alignment, batch 4 reproduced six times across two sessions.**

| batch | ex | runs | mean s/step | spread | s/example | `band_rnn` batch = ex*T | %8 |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 4 | 2.993 | 0.9 % | 0.499 | 3018 | 2 |
| **4** | 8 | 6 | **0.977** | **0.3 %** | **0.122** | **4024** | **0** |
| 5 | 10 | 2 | 5.005 | 0.0 % | 0.501 | 5030 | 6 |
| 6 | 12 | 1 | 6.007 | — | 0.501 | 6036 | 4 |

Aligned vs unaligned is **4.09x**. The three unaligned sizes agree to 0.4 %, the signature of a shared fallback kernel. The hypothesis predicts the data with nothing fitted. **T = 503, not 497** as earlier drafts said: `stft.py` pads by `n_fft - hop` on both sides. Timings and memory figures are unaffected.

Fix confirmed: `chunk_s` 4.008 gives 0.674 s/step at batch 3 (reproduced twice, 0.1 % spread), and the real 2-epoch `sir0` run measured **505.7 s/epoch against 3875.2 = 7.66x**. Losses unchanged: epoch 1 agrees on all twelve logged terms within 3 %.

**E1's analytic memory model is 2.4-3x low** (5032 MB predicted at batch 3 vs 12.20 GB measured; 1677 MB vs 5100 MB on CPU). The gap is the autograd graph, gradients, optimiser state and `L_MR`'s eight retained STFTs. Multiply E1's table by ~2.5 before predicting a ceiling.

**Two sets of void results, both fixed 2026-08-28.** The GPU attribution reported `estimator` 97.4 % / `separator` 0.8 %, inverting the CPU result. CUDA kernels are asynchronous, so unsynchronised `perf_counter` hooks measure queuing and the module that blocks last absorbs the queue. The AMP loop also accumulated into the fp32 counters. Separately, the first `--amp-only` attempt ran its warmup step unconditionally in fp32 and so OOM'd at batch 4 before AMP was exercised. **Discard any `--amp-only` result from a bundle built before 2026-08-28 21:00.**

**Operational: profiling locally can kill the terminal.** `systemd-oomd` kills the whole cgroup scope, not the offending process, and fires on sustained PSI pressure, not exhaustion. Batch 3 needs ~15 GB on a 15 GB laptop. Profile locally under a scope:

    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -- \
        ../tse_venv/bin/python scripts/profile_step.py --batch 1

`profile_step.py` now refuses to start when its estimate exceeds half of available RAM, keeps `torch.profiler` behind `--deep`, measures T from the real STFT, takes `--chunk-s`, and prints an ALIGNED / NOT ALIGNED verdict.

Still open from this group:
- [ ] **Re-run the GPU attribution** with the fixed script. E3b's CPU split is the only valid one, and kernel-launch overhead may raise the band-loop share on the T4.
- [ ] **fp16 through the `w` ramp is untested.** Both validation epochs ran at `w` = 0.0, so the absent branch and its mute pressure never engaged.

### E4. Ranked levers, cheapest first

1. **Mixed precision (fp16 + `GradScaler`).** T4 is Turing: fp16 tensor cores, **no** bf16. Expect ~1.5-2x on LSTM-heavy work and ~2x less activation memory, which alone may unlock batch 6-8. **The loss must stay fp32.** `L_pres`/`L_abs`/`L_gain` carry 1e-12 epsilons inside `log10`, below fp16's smallest normal (~6e-5), so they underflow to zero and return NaN. Wrap only the model forward in `autocast` and cast to `.float()` before `LossBSRNN`.
2. **`pin_memory=True` on both loaders.** Unset in `build_loaders()`; `persistent_workers` and `prefetch_factor` already are. One argument.
3. **Gradient checkpointing over the six `BSNet` blocks.** ~33 % extra compute for roughly 5x less activation memory. Usually a net *win* here: at batch 3 `time_rnn` has only 192 parallel sequences (poor occupancy), and batch is an LSTM's only parallelism, so a bigger batch may pay for the recompute outright.
4. **`hop` 128 -> 256.** Halves `T` (497 to 249), roughly halving compute and memory (E2). Latency granularity goes from 8 ms to 16 ms, well inside the 200-300 ms budget, and 50 % overlap still reconstructs. **It costs mask time-resolution, so it is an architecture change needing its own ablation arm and decision entry.** Not free, but the largest single win. `chunk_s` 4.0 to 2.0 also halves `T` but is *not* equivalent: it cuts the context the model learns conditioning in, our weak point. Prefer the hop change.
5. **Collapse the 32-band Python loops into grouped convs.** `Estimator`'s trunks are already uniform (`LayerNorm(128)` + `Conv1d(128->384)` + `Tanh` for every band), so all 32 become one `Conv1d(32*128 -> 32*384, groups=32)`: identical arithmetic, ~32x fewer launches. The mask/res heads and `SubbandNorm` have band-dependent widths and need padding to `max(bw)` plus a slice. **Do this last and re-measure first.** It is fixed overhead, so a bigger batch shrinks its share, and after 1-3 it may not be worth the refactor risk.
6. **`torch.compile`.** Fuses pointwise work and cuts dispatch overhead; cuDNN LSTMs are untouched. Shapes are static (4 s chunks), so no recompilation risk. Cheap to try, modest gain.

### E5. Audio compression: what it can and cannot do

**It cannot reduce GPU memory.** Peak memory is LSTM activations (`batch x time x hidden`, E1), and disk format is irrelevant once a waveform is a tensor. Compression only helps *disk* and *data-loading*, so do it **only if E3's profile shows loading is the bottleneck**.

- Audio is already 16-bit PCM (`docs/run_times.md`).
- FLAC would halve disk but add CPU decode on a ~4 vCPU box already feeding 4 workers, possibly making things **worse**. Measure first.
- **Caching STFTs is a net loss.** A `complex64` spectrogram is 257 x 497 x 8 B = 1.02 MB against 128 KB for the int16 waveform, **8x more storage**, to skip well under 1 % of compute.

### E6. Why this is worth doing at all

At the measured 3875 s/epoch, on the 10-epoch schedule:

| | per epoch | 10 epochs | 12 ablation runs | one 10k-trial run |
|---|---|---|---|---|
| today | 3875 s | 10.8 h | 129 h | 54 h |
| 2x faster | 1938 s | 5.4 h | 65 h | 27 h |
| 3x faster | 1279 s | 3.6 h | 43 h | 18 h |

The planned arms alone (`ablate_w_m` 3, `ablate_w_g` 3, lookahead 3, plus D4a and a `tfmap_scale` arm to close D2) are ~12 runs. **Speed converts directly into how many questions the thesis can answer.** Under Kaggle's ~12 h session cap the 10k-trial run is multi-session today and single-session at 3x.

### E7. PROPOSAL — `DataParallel` across the second T4, which is currently idle

**Raised 2026-09-04.** Kaggle's "GPU T4 x2" accelerator gives **two** T4s. The code uses one: `torch.device("cuda")` is `cuda:0`, and `scripts/train.py` and `src/models/` contain no `DataParallel`, `DistributedDataParallel` or `device_count`, so half the allocated hardware has idled on every run. The P100 option is one card, so this only applies to T4 x2.

**Why it is cheap.** The loss is computed *outside* the model:

```python
s_output = model(mixture, enrollment)                     # parallelisable
loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent)
```

`DataParallel` scatters the batch, runs the forward on both cards, and gathers outputs back to `cuda:0`, so `LossBSRNN` still sees the whole batch on one device. This matters because the loss means over *subsets* (`n_present`, `n_absent`), which a per-device reduction would silently reweight.

**It does not break the direction pairing, despite the first reading.** `collate_pairs` keeps both directions of a trial in one batch, and the device split separates some pairs in the forward pass. However, the model is per-example independent and the contrast lives in the *loss*, which sees the gathered batch.

**The work: six call sites, one real risk.**

1. Wrap after `build_model()`/`.to(device)`, gated on `torch.cuda.device_count() > 1` **and** a config flag defaulting to off, so every existing run reproduces.
2. Keep `core = model.module if wrapped else model` and use `core` for:
   - `state_dict()` at the three `torch.save` sites (**the risk**: `DataParallel` prefixes every key with `module.`, which would break `--resume`, `make_estimates.py`, and every checkpoint already on disk);
   - `load_state_dict()` on resume;
   - `model.band_widths` in `log_results()`. `DataParallel` does not forward arbitrary attributes, so this raises `AttributeError`, not a wrong number.
3. `diagnostic_accumulate()` calls `model(...)` for the enrolment swap. It can take the wrapped model, but must be the *same* one the forward used.
4. Log `device_count` in `meta.yaml`; s/epoch is unreadable without it.
5. The notebook's batch-size probe measures one GPU. With two, the effective batch doubles and the probe's answer changes meaning.

**Expected gain: ~2x, but only at `batch_size: 6`.** Revised 2026-09-04 up from 1.4-1.7x, which assumed splitting batch 3 would leave each card with poor LSTM occupancy. E3b-E3f's fp16 sweep shows that was too pessimistic:

| batch | s/trial | peak GB |
|---|---|---|
| 3 | 0.990 | 6.57 |
| 5 | 0.994 | 10.87 |
| 6 | 1.005 | 13.02 |

Per-trial throughput is **flat over a 2x batch range (1.4 % spread)**, so one T4 is saturated at batch 3. Batch 6 split 3/3 gives each card a *saturated* batch 3, for ~2x; batch 3 split 1.5/1.5 would not.

**Caveat (noticed 2026-09-04): the saturation evidence is from the unaligned regime.** Batches 3, 5 and 6 are the sizes E3b-E3f found sharing a fallback kernel ("agree to 0.4 %, the signature of a shared fallback kernel"), while aligned batch 4 ran 4.09x faster per example. Flatness on one slow kernel shows that kernel is linear in work, **not** that the GPU is saturated on the fast path. Since `chunk_s: 4.008` everything runs aligned, with two comparable points: batch 3 at ~0.112 s/example (0.674 s/step / 6) and batch 4 at 0.122. Roughly flat, but from different measurement sets.

The 2x claim needs per-example throughput to stay flat from **6 to 12 examples while aligned**, which is unmeasured.

**Settle this before writing any DataParallel code.** An aligned batch sweep (3/4/5/6 at `chunk_s: 4.008`, `profile_step.py --amp-only`), ~20 min of GPU, decides whether E7 is worth half a day. If per-example throughput *rises* with batch on the fast path, a bigger batch on one card is the win and the second card adds little.

**So E7 depends on E8: `batch_size` goes from 3 to 6 first**, which fits today (13.02 GB of 14.56). Two cards also double the memory that forced batch 3 (E1).

Replication overhead is negligible: 7.19 M params (~29 MB fp32) broadcast over PCIe at ~6 GB/s takes ~5 ms of a ~750 ms step (1244 s / 1658 steps at 4,976 trials), **under 1 %**. Scatter/gather latency remains.

**Re-check alignment.** The 4.09x tensor-core cliff (E3b-E3f) needs `examples x T` divisible by 8, and a DataParallel split changes the per-device batch. Check `profile_step.py`'s ALIGNED verdict before reading any timing, or a 4x regression will look like "the second GPU didn't help".

**Cost: roughly half a day**, mostly verification, including a 2-epoch A/B at fixed seed. Numerics will differ slightly (kernel split, fp16 reduction order), so val terms need only agree within between-run noise. Per the 2026-08-29 AMP row that noise can swallow real differences, so this is a **speed** claim only, never a quality change.

**`DistributedDataParallel` is the better tool, but deferred.** It avoids the per-step replication and the gather, but needs `spawn` multiprocessing inside a Kaggle notebook, which fights the single-process training loop and the tee'd `history.csv` recovery path. Not worth it for 2 cards on one host.

**Sequence.** After the 9,955-trial run, the third point on the data-scaling curve, which must not change two variables at once.

### E8. PROPOSAL — raising `batch_size`: what works, and what cannot

**Raised 2026-09-04**, from asking how compression could buy the bigger batch E7 needs.

**The premise is wrong for speed.** Per E3b-E3f, per-trial throughput is flat from batch 3 to 6 (0.990 / 0.994 / 1.005 s/trial), so **a bigger batch does not make training faster on one T4**, which is saturated at batch 3 (hence gradient checkpointing was withdrawn). The 7.66x came from tensor-core *alignment* (aligned vs unaligned is 4.09x), not batch size (batch 3 to 6 is 1.0x). A bigger batch is only worth having for three reasons:

1. **To feed DataParallel (E7)**, the real one. Two saturated cards, ~2x.
2. Different gradient statistics: a change to training dynamics needing its own arm, not a speed optimisation.
3. Headroom for a longer `chunk_s` or `lookahead_frames` later.

**Batch 6 needs only a config edit.** Memory is exactly linear at **0.12 GB fixed + 2.15 GB per trial**: batch 6 peaks at 13.02 GB of 14.56 available, and batch 7 (15.17 GB) does not fit.

**Bug, found 2026-09-04: the notebook's batch probe measures fp32 and the run trains in fp16, so it has capped every Kaggle run at batch 3.** `_probe_batch.py` in `scripts/make_kaggle_notebook.py` does:

    loss, _ = L(s, m(x, e), x, a)
    loss.backward(); opt.step()          # no autocast, no GradScaler

while `train.py` wraps its forward in `amp_ctx(use_amp)` with `amp: true`. The probe finds the **fp32** ceiling (batch 3 in E3b-E3f, with 4/5/6/12 all OOM in `band_rnn`'s `_VF.lstm`) and writes it into a config that trains at 1.86x less memory (12.20 GB fp32 vs 6.57 GB AMP, both batch 3). Hence the belief that one T4 fits only batch 3: true in fp32, but the runs are not fp32.

**The same bug was fixed before.** E3b-E3f records "the first `--amp-only` attempt ran its warmup step unconditionally in fp32 and so OOM'd at batch 4 before AMP was exercised". That fix went into `profile_step.py` but never reached the notebook probe.

Fixing the probe is E8's cheap half: wrap its forward in the trainer's `amp_ctx` and let it find 6. Until then, any "batch 6 does not fit" from the notebook measured the wrong precision.

**Memory is almost all activations.** The model is 7.19 M params (~29 MB) and AdamW state ~57 MB, against ~13,000 MB:

- **8-bit optimisers (bitsandbytes) would save ~43 MB.** Not worth the dependency.
- **Audio compression cannot reduce GPU memory at all** (E5). Caching STFTs is 8x *more* storage (E5).

**Levers that would work beyond batch 6:**

1. **`hop` 128 -> 256, the only one that reaches batch 12.** Halves `T`, and cost is linear in `T` through both RNNs (E2), so activations and compute halve: ~1.08 GB/trial, and batch 12 fits. Costs mask time resolution (8 ms to 16 ms granularity, well inside the 200-300 ms budget). E4's item 4, ranked the largest single win, and an architecture change needing its own ablation arm.
2. **Gradient checkpointing over the six `BSNet` blocks, reconsidered for this purpose.** ~5x less activation memory for ~33 % more compute. Rightly withdrawn as a *speed* lever on a saturated card, but buying batch headroom to feed a second GPU is a different trade.
3. **`L_MR`'s eight retained STFTs.** Part of the 2.4-3x gap between E1's analytic model and measured memory (E3b-E3f): `windows_ms: [8,16,32,64]` holds four resolutions for the backward pass. Recomputing them, or dropping to two windows, attacks a *measured* contributor without touching the architecture, but changes the objective, so it needs an ablation.
4. **Gradient accumulation**, if the goal is large-batch *statistics*, not throughput. Zero extra memory, and exact here because the model uses LayerNorm/SubbandNorm, not BatchNorm, so no cross-example statistic is corrupted. Makes nothing faster.

**Rejected: shorter `chunk_s`.** Halves memory like `hop`, but cuts the context the model learns conditioning in, the documented weak point (enrolment sensitivity flat at -3.79 dB across the 2.5x data increase, `decisions-m2.md` 2026-09-01). E4 records the same objection.

**Alignment governs all of it.** `examples x T` must stay divisible by 8 or the 4.09x fallback kernel fires. `chunk_s: 4.008` makes `T` divisible by 4, so every batch size is aligned today, but levers 1 and 3 change `T` or the STFT set and can silently break that. Check `profile_step.py`'s ALIGNED verdict first.

**Changing `batch_size` silently changes the `w` schedule.** The absent-branch warmup counts optimiser *steps* (`decisions-m2.md` 2026-09-03). This makes it invariant to **dataset size**, the confound it was built to remove, but **not** to batch size. At batch 6 the same 11,606-step warmup covers 2x the audio it did at batch 3. The warmup exists to stop the early mute, for which its length in examples is what matters, so this changes the objective's schedule. Either hold `warmup_steps`/`ramp_steps` x batch constant when batch changes, or treat a batch change as its own arm. **Do not let the probe pick a new batch size during the 9,955-trial run.** Set the notebook's `BATCH_SIZE` knob to 3 to pin it, since candidates are filtered to `<= BATCH_SIZE`.

**Sequence.** `batch_size: 3 -> 6` is E7's cheap prerequisite and can ride with it. Levers 1-3 only matter beyond 6, and wait until E7 has measured what two saturated cards deliver.

### D9. ASR cross-entropy as a differentiable content proxy

**Status: proposal, deferred. M5, which is CUTTABLE and the first thing to cut. Raised 2026-08-30.**

**Origin.** The idea was first put as "transcribe the output each epoch, compute WER, add it to the loss and hold it constant through the next epoch". That does nothing: **a term that does not vary with the weights has zero gradient**, so only the printed loss changes. Recomputing per batch does not help, since WER passes through `argmax`/beam search and edit distance, neither differentiable.

**Working version.** Push the estimate through a frozen ASR and take the teacher-forced cross-entropy of its token distribution against the known transcript. With no decoding there is no `argmax`, so the gradient flows through continuous logits into the mask. CLAUDE.md names this directly: "training uses differentiable proxies (frozen-ASR/SSL feature matching, optionally ASR cross-entropy)". Feature matching against the clean target's ASR-encoder features is the cheaper sibling and needs no text.

**The rigorous version of the original idea** is minimum-WER training (Prabhavalkar et al., ICASSP 2018), which optimises expected WER by sampling hypotheses with a score-function estimator instead of a true gradient. It is expensive and finicky, hence cross-entropy is the usual choice.

**Hard constraint.** The proxy ASR must not be `small.en`, the evaluation scorer: training against your own evaluator makes the offline WER meaningless (CLAUDE.md rule 2). It must be a different model family, recorded not assumed, and never the judge in any form.

**WER belongs in model selection now, for free.** It cannot make gradients, but it ranks finished models, and `training.select_on` (2026-08-30) has a place for it. Transcribing four or five candidate epochs and checking whether the present-branch proxy ranks them the same way is one afternoon and a thesis table either way.

### D10. Penalise resembling the INTERFERER, not just missing the target

**Status: proposal, unscheduled. The best remaining model-side idea, and the only one aimed at extraction-vs-enhancement. Raised 2026-08-30.**

**Problem.** `L_pres` maximises SI-SDR to the target, lumping interferer leakage, residual noise and artefacts into one residual. For this project they are not equally bad: **the interferer's words are the worst error**, because a live judge transcribes them as the target's speech. On `eval_public-42-000132` the ASR reads the target correctly for 17 words, then transcribes the interferer's sentence verbatim.

**Proposal.** An explicit repulsion term against `interferer.wav`, already rendered per trial and already loaded (`both_directions`). Weight interferer leakage above other residual error, so the objective prefers a noisy extraction to a clean confusion.

**Why it fits.** It is the training-side mirror of **ICR** (`metric-definitions.md` 3.2), which the metric defines as "the score that makes the metric two-sided, and the one an offline WER-based metric structurally cannot see". Objective and metric would then measure the same failure.

**Check before building.** `L_pres` already counts interferer energy as error, so the term only adds a weighting, which the ablation must justify (0 arm required, as for `w_m` and `w_g`). It also risks a new degenerate solution, since silence resembles neither speaker. `L_gain` now blocks this, but re-verify it.

**Measure first, at no extra cost.** `interferer_text` is in every `meta.json`, so one transcription pass gives WER against the target and content overlap against the interferer, showing whether leakage dominates before any term is written. Measure ICR, then decide.

**Evidence that the model is not purely enhancing** (2026-08-30): an enrolment swap moves the output 48.2 % (D3a), where a pure enhancer would move 0 %. `both_directions` requires two different answers from one mixture, and `sir0` removes the loudness shortcut. It is a matter of degree, since 52 % of the output is still enrolment-independent.

---

## Group J — the judge and the metric

Raised 2026-08-30. M4 has 1 of 9 items done and the judge gates the rest, so these decisions have the longest lead time in the project.

### J1. Must the judge be full-duplex speech-to-speech, or is audio-in enough?

**Status: closed 2026-08-31, audio-in / text-out, as recommended below.** Reasoning, the three gains, the cost and the ~50-trial full-duplex confirmation run are in `decisions-m4.md` 2026-08-31.

**Tension.** CLAUDE.md and spec note 10 define the objective as what a "**live speech-to-speech model** (Gemini Live and similar)" recovers, which read strictly requires full duplex. `metric-definitions.md` 3.1's own mechanism does not:

> A live speech-to-speech model consumes audio through a learned audio encoder over a much wider distribution, and appears to be more sensitive to the artefacts of the processing itself than to the interfering speech it removed.

**The property measured is the audio encoder, not the duplexing.** An audio-in / text-out LLM has that encoder and that wide training distribution, and nothing in LCF-WER, ICR or NRR reads the judge's turn-taking. 3.1 step 2 already permits a text response ("*If* the model responds in audio, transcribe the response"), so audio output is optional in the protocol as written.

**Gains from relaxing it.**

1. **A much wider field.** Ultravox, Voxtral and Qwen3-Omni all qualify. Full-duplex narrows it to roughly Moshi and the closed APIs.
2. **One fewer component in the measuring instrument.** A text response removes the response-ASR, which 3.1 step 2 calls "a component of the measuring instrument, and changing it invalidates comparisons", and with it a whole class of invalidation.
3. **Cheaper and faster** on both the API and the self-hosted side.
4. **Ultravox is the closest conceptual fit**: it skips the separate ASR stage, the property 1 hypothesises about.

**Cost.** It deviates from the stated objective, so the write-up must argue it: *we used an audio-in model because the measurement depends on the audio encoder rather than on duplexing*. Otherwise a reviewer can say the thesis measured something other than what it set out to.

**Recommendation: relax it and record the argument above.** It needs supervisor sign-off, since it edits the stated objective.

### J2. Which judge -- and the open-weight anchor is about reproducibility, not cost

**Status: open. The cost half is answered, and smaller than assumed.**

**The cost model M4 asks for, measured 2026-08-30.** `gemini-3.1-flash-live-preview` publishes $0.005/min audio in, $0.018/min audio out, $4.50/1M text out, with a free tier. Trials are ~18 s, and M4's protocol is 200 trials x k=3 repeats x 4 systems = 2,400 calls:

| | |
|---|---|
| audio in, 720 min | $3.60 |
| text out, ~0.12M tokens | $0.54 |
| **audio condition total** | **~$4.14** |
| with audio responses instead | ~$15 |
| plus a prompt-sensitivity ablation | ~$25 all in |

**So M4's "closed API (money) or self-hosted open-weight (GPU-hours)" is not a budget question.** ~$25 is negligible and the free tier covers the pilot. Use the closed API for the headline.

**The open-weight anchor is still required, for reproducibility.** Both candidate Gemini IDs are marked `preview`, and preview models get deprecated. If the headline judge disappears before submission, the primary result becomes unreproducible (hence CLAUDE.md's exact model ID and run date on every judge result). The anchor lets someone reproduce the headline in two years. **Argue reproducibility in the write-up, not cost, which would be weaker and false.** Published prices expire too ("through December 31, 2026"), so record the price with the ID.

**Compute for a self-hosted anchor is no longer contended.** M4 worried about GPU-hours competing with training quota, but training is being stopped (2026-08-30), so the Kaggle T4 is free. A 3B model in fp16 fits it, and 2,400 short generations is one session. The laptop has no usable GPU (CPU generation would take about a day of wall clock for the full protocol), but can run a 20-trial pilot.

**Gate for every judge candidate, open or closed.** Score the **ceiling condition first**: feed the clean target audio and read LCF-WER (offline ASR ceiling: 6.1 %). If a judge cannot reliably report clean speech, it is the bottleneck and every comparison under it is noise. About 20 trials and an hour, and it disqualifies candidates before they cost a benchmark run.

**This gate matters more for open-weight candidates.** On FullDuplexBench, task adherence is 1.26/5 for Moshi and 3.82/5 for Qwen2.5-Omni. Our prompt is trivial ("report what you heard"), but a judge that wanders off-prompt, refuses or chats lands in **NRR**, which was designed to catch a degenerate extractor. **A degenerate judge looks the same in the numbers.** Choose the anchor for instruction-following on a transcription-style prompt, not conversation.

### J3. The ICR overlap threshold — declared, not signed off

**Status: closed 2026-09-03.** Signed off into `metric-definitions.md` §3.2, which now states the threshold, sensitivity table, exclusion rule, the floor's by-construction caveat and the prompt constraint. decisions-m4.md 2026-09-03. Below is how the value was chosen.

`metric-definitions.md` 3.2 defines ICR as "content-word overlap between `r` and `d`, excluding words that also appear in `t`, thresholded", and requires a threshold **fixed in advance with its sensitivity reported**, but gives no value. Two candidate rules:

| rule | statement | problem |
|---|---|---|
| **`count>=2`** | ≥2 interferer-exclusive content words appear in `r` | insensitive to how much the interferer said |
| `frac>=θ` | that count as a fraction of the interferer-exclusive words available | scale-dependent on the interferer's utterance length, which varies per trial by construction |

**Declared: `count>=2`.** One content word shared with the interferer is coincidence at the rate English repeats nouns; two is signal. The fraction rule varies with the trial, not the system, so it is the worse primary and the better secondary. Both are reported, with a sweep over counts 1/2/3/5 and fractions 0.05–0.50, per 3.2's sensitivity requirement.

**Two things to settle at sign-off.**

1. **Trials where the interferer said nothing the target did not also say** carry no evidence of contamination either way. They are **excluded** from ICR, not scored as clean (which would dilute the rate towards zero), and the exclusion count is reported.
2. **The floor row's ICR is partly set by construction, not measured.** The judge never sees the enrolment, so on an unprocessed two-speaker mixture it cannot know the target and picks a speaker, and the floor's ICR tends towards a coin flip. This is correct and is the finding (doing nothing gets the wrong speaker half the time), but state it whenever the floor row is quoted. The fixed prompt must **not** tell the judge to choose a speaker ("the clearest voice", "the loudest speaker"), as that hands the extractor's job to the judge and turns a measurement into an instruction.

### J4. PROPOSAL — a metric *system*: normalised requirement axes, composed, plotted

**Status: open, proposed 2026-08-31 (Grant's idea). Not a decision yet. The diagnosis is right, but the normalisation and composition rule need changing first, and one part as pitched is unsound.**

**Problem.** LCF-WER, ICR, NRR, SI-SDR, DNSMOS (P.835 since 2026-09-01), offline WER and latency are seven numbers, and B13 requires each broken out per condition: roughly 35 cells per system. **Nobody can rank two models by reading 35 cells.** The project has no defined way to say "model A is holistically better than model B", only "A is better on this row".

### The proposal

1. The user declares **n requirements** for their speech model (e.g. *Speaker learning*, *Sound separation*, *Content fidelity*), placed as n equally spaced axes on a circle.
2. Each requirement is fed by **several underlying metrics**. Example for *Speaker learning*: (a) how many words of the estimate appear in the interferer's speech, (b) how long the model tracked the interferer during target silence, (c) the same during target speech.
3. Each metric is scored **relative to a declared baseline** (a real or hypothetical reference model), so the axis is an improvement, not a raw unit.
4. Radius = how good: **out toward the rim is better**, near the centre is worse.
5. Two models are overlaid on one chart, and the **shape** shows what each is good and bad at.
6. **Ranking by total area** enclosed, plus ranking by a single axis or a group of axes.

**Declared axiom (keep it):** every metric admitted to the system must be able to rank two models and say which is better.

### What is right about it

**It makes two-sidedness structural, not a convention.** `metric-definitions.md` 4 already requires LCF-WER, ICR and NRR to *always* be reported together, because suppressing everything wins on ICR and passing everything through wins on NRR. Today that relies on discipline. On an axis plot they are spokes of one figure and cannot be shown apart.

**Baseline-relative axes are correct**, and the anchors already exist: `metric-definitions.md` 3.4 makes floor (unprocessed mixture) and ceiling (clean target) mandatory on every results table.

**Grouping metrics into requirement classes helps the viva.** "Better at holding onto the right speaker, worse at avoiding processing artefacts" is a sentence a reader can carry. Seven numbers are not.

### Three things that must change first

**1. Ranking by area is unsound. Do not do it.** Radar-polygon area depends on the *order the axes are drawn in*, which is arbitrary. For n equally spaced axes with radii `r_i`:

```
Area = ½ · sin(2π/n) · Σ_i r_i · r_{i+1}
```

Only **adjacent** pairs multiply, so strength on two neighbouring axes scores more area than equal strength on two opposite axes. For n=4:

| model | scores in drawn order | Σ r_i·r_{i+1} | area |
|---|---|---|---|
| X | 1, 1, 0, 0 | 1 | > 0 |
| Y | 1, 0, 1, 0 | 0 | **exactly 0** |

**Identical multisets of scores, and Y encloses no area.** The ranking comes from label placement, not the models. This is the failure mode `metric-definitions.md` 4 was designed against: a free parameter (axis order) that can be tuned to change the winner makes a score gameable. REAL-TSE had to swap its official metric after the fact for a comparable reason.

**Fix: keep the picture, but rank by an explicit weighted mean of the normalised axis scores.** Order-invariant, with visible and arguable weights, and the chart still shows shape.

**2. "Percentage increase over baseline" breaks on signed and dB quantities.** The proposal's worked example is do-nothing SI-SDR **−2.12 dB** and model **5.15 dB**: `(5.15 − −2.12)/(−2.12) = −343 %`, negative for an improvement because the denominator is negative. It is undefined at baseline = 0 and meaningless for any quantity that crosses zero.

**Fix: normalise to the floor–ceiling interval**, which is dimensionless, well-defined for signed and dB quantities, and reuses anchors the protocol already mandates:

```
s = (x − floor) / (ceiling − floor)        clipped to [0, 1]
```

`s = 0` is "doing nothing", `s = 1` is "the best achievable on this judge". For a lower-is-better metric the interval runs the other way: WER with floor 57.4 % and ceiling 6.1 % gives `s = (57.4 − x) / (57.4 − 6.1)`. Every axis is then on one scale, and "toward the rim is better" holds by construction.

**3. The axiom needs strengthening.** "Can rank two models" is *ordinal*, and knowing a model is 2nd of 3 does not say what radius to draw it at. Each metric must be **monotone in goodness and cardinally normalised**, which item 2 supplies.

### Two design questions to settle before building

**Which mean.** An arithmetic mean (like area, or any sum) lets a model **compensate**: superb LCF-WER hides catastrophic NRR, the degenerate mute this project already caught once. A **geometric mean** `(Π s_i)^(1/n)` collapses to zero if the model is at floor on *any* axis, so it cannot be gamed by trading one requirement away. **It is the principled form of the two-sidedness rule** and the recommended headline, with the arithmetic mean reported beside it and the headline stated. Decide how to floor `s_i`, so one axis at exactly 0 does not erase an otherwise-informative model.

**User-defined weights are a gaming surface.** Configurable weights suit a *tool* but are fatal in a *benchmark*: anyone could reweight until their model wins. **The benchmark must publish one fixed, pre-registered weighting, frozen before results are seen**, as the prompt, the normaliser and the ASR are frozen. The configurable version is a separate exploration mode, labelled as not the benchmark number.

### Scope, so this does not balloon six weeks from freeze

**This is a presentation and composition layer, not a new metric.** It cannot change or invalidate LCF-WER, ICR or NRR, which keeps it cheap: one module that takes the per-condition scores and a frozen weighting file and emits the figure and the composite. A Chapter 4 figure and a ranking rule, not a rebuild.

**Framing for the write-up.** Radar charts are old and criticised in the visualisation literature, largely for the area problem above, so the contribution is *not* the chart. It is **a composition rule for TSE evaluation**: floor/ceiling-normalised axes, grouped into declared requirement classes, aggregated by a compensation-resistant mean, under a pre-registered weighting.

**Consider a dot/parallel-coordinates companion plot.** Same data, no area artefact, exact values readable, and cheap to emit alongside. The radar answers "what shape is this model", the dot plot "by how much".

### Open sub-questions

- Does **latency** belong on a quality axis at all? It is a **constraint** (200–300 ms budget), not a trade-off: on the radar a fast, mediocre model could win on shape. Probably a pass/fail gate plus B11's decay curve, kept off the composite.
- The proposed *Speaker learning* metrics (b) and (c), time spent tracking the interferer during target silence and during target speech, **are not built and are not in `metric-definitions.md`.** They need a definition and a ground-truth source first (feasible, since the VAD index gives per-speaker activity).
- How do axes behave for a model **below the floor**? The 08-29 checkpoint at epoch 24 was *worse than pass-through*. Clipping at 0 hides that, and negative radii break the plot. Probably clip, and flag "at or below do-nothing" on the axis label.
- Is the composite reported **per B13 condition** as well as pooled? It must be, or it becomes the aggregate-that-appears-alone that B13 forbids.

### D11. Inference-time mix-back: a measuring instrument, NOT a fix to the model

**Status: open. Raised 2026-09-01. Grant's objection is accepted: this is a cheap post-hoc patch, not a strategy for fixing the process. Kept for its separate value as an instrument.**

**Proposal.** Blend the output back with the input at inference: `s_alpha = alpha * s_hat + (1 - alpha) * x`. `alpha = 1` is the current model, `alpha = 0` is doing nothing. It costs one multiply-add per sample, adds **zero algorithmic latency** (output sample n needs only input sample n), and needs no retraining, since `alpha` is not a model parameter. One forward pass serves every `alpha`.

**Why it was proposed.** The 2026-09-01 measurement found the model applies the same transform regardless of difficulty (SIR improvement 3.80 to 4.32 dB, SAR degradation -17.45 to -21.41 dB, essentially flat from easy to hard trials), while the word-error outcome swings from -4.2 to +23.1 points. Since the signal behaviour is constant, one global knob is the right *shape* of intervention.

### The objection, which stands

**This does not fix anything.** It gives up the benefit on hard trials to stop the harm on easy ones, using a constant chosen offline. The model still cannot tell the cases apart, still produces the same artefacts, and cannot modulate itself.

**Never present it as an improvement to the extractor.** In any results table it is *the extractor with a mix-back gain*: one system and a parameter, never five systems.

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

**The head peaks at epoch 2 of 20 at 425, 428 and 850 training trials.** Training loss falls from 0.473 to 0.057 while held-out loss rises from 0.464 to 0.769. More data does not move the peak, so the head is memorising something that data does not dilute.

**The enrolment is the route.** One fixed 192-d vector per trial is seen 78 times per epoch (6 variants x 13 windows). 850 trials means 850 vectors to memorise, and 149 k parameters against ~10,000 semi-independent windows is about 15 parameters per effective sample.

**This failure has been fixed before.** The extractor's identity cue "stayed a fixed waveform across all 24 epochs and was memorisable" (`decisions-m2.md` 2026-08-30), and `scripts/render_enrollment_bank.py` was the fix. K=3 is already rendered for `sir0_train` as `enrollment_v00/01/02.wav`, with different sentences, offsets and EQ per variant.

**Cost: ~2 extra ECAPA forwards per trial.** `v00` is byte-identical to `enrollment.wav` (verified 2026-09-10), so the cached embedding is variant 0. Patching 1,000 existing cache files takes ~15 min, with no re-render.

**Rotation is training-only.** The holdout keeps one fixed enrolment so its curve stays comparable across epochs and runs, the same rule `dataset_loader.py` enforces by forcing `enrollment_variants` to 1 when `random_crop` is off. A short bank falls back to v00, matching `_enrollment_path`.

`ROTATE_ENROLMENT` is an ablation flag; `False` reproduces every run before 2026-09-10 exactly. **The arm to report is 1,000 trials with one enrolment versus three, all else held.** If the best epoch moves from 2 to 6-8, the memorisation diagnosis is confirmed and the head has real headroom.

### Sequencing: the hyperparameter search was STOPPED and deferred

Optuna (`optuna==5.0.0`, pinned) covers head width and depth, dropout, learning rate, weight decay, batch size and `min_purity`, on a three-way split (700 fit / 150 search / 150 report) with a threshold-free mean-AUC objective and median pruning. `AUDIBLE_DB` is excluded: it defines what counts as a second voice, so tuning it would optimise the question.

**It was stopped after one trial.** If the bank removes the memorisation, the settings the search favours (heavier dropout, weight decay, early stopping) would be compensating for a failure that is gone. The search runs after the bank arm, on the final data. The wrong threshold was not the reason, as mean AUC is threshold-free.

### Not yet done

- **`sir0_val` is still uncached.** All splits come from `sir0_train` and share speakers, so the teacher is unvalidated until measured on a speaker-disjoint set.
- **GPU cost of the term is unmeasured** and gates the integration: 13 windows x 6 examples = 78 ECAPA forwards and backwards per step, against 0.674 s/step now. The 78 ms/window figure is CPU; training is on a T4. Measure with `scripts/profile_step.py` before writing more integration code.
- Rows for `docs/run_times.md`: the 1,000-trial cache (~68 min for 350 new trials), and the rate difference, 12.1 s/trial clean against 44.9 s/trial with a browser and an IDE competing for the same 8 cores.

### 2026-09-11 — the teacher is FINISHED at AUC 0.87, and the integration is built

**A sequence model over the windows was worth ten times everything else tried. The teacher is frozen there, with no further work planned.** The loss term, subclass, tests, weight-derivation script and config are written.

### The architecture change, which is the result

Same features, split and objective; only the head differs:

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

**Why it works.** A single window at cosine +0.23 to the enrolment could be "the target alone, quieter" or "the target plus someone else". `both` sits at +0.233, between `target` at +0.321 and `interferer` at +0.044, because a verification embedding describes the dominant voice. Neighbouring windows resolve much of that.

**BiLSTM over attention:** better (0.9216 vs 0.9170), 30 % fewer parameters, converged at epoch 1 rather than 7.

**Caveat for the write-up.** Part of the gain is cross-window comparison and part is temporal smoothing, since speaker states come in runs of hundreds of milliseconds. The prior is legitimate, but **an isolated window of leakage may be smoothed away and go unpenalised**. The teacher catches sustained leakage better than brief leakage.

### Final teacher, and the numbers the loss depends on

1,000 `sir0_train` trials (850 fit / 150 holdout), BiLSTM hidden 128 / 1 layer, enrolment bank K=3 rotated per epoch, epoch 3, seed 42, audibility -20 dB. sha256 `16f6d8a43d5ecacf24e1...`.

| | balanced | recall | specificity | precision |
|---|---|---|---|---|
| is the target audible | 0.906 | 0.908 | 0.903 | 0.914 |
| **is a non-target audible** | **0.802** | **0.795** | **0.808** | **0.679** |

Identity on single-speaker windows: **0.942** (WavLM: 0.570).

**The arm uses only the non-target column**, the worse one, because its required answer is always zero and needs no labels. The target column depends on whether the target spoke in each window, so it needs per-window labels in the loader (the second increment).

### Three interventions that did NOT work, recorded so they are not retried

1. **Enrolment bank (K=3, rotated per epoch).** Predicted to move the best epoch from 2 to 6-8. **It stayed at 2** (+0.008 AUC), so the prediction and the diagnosis were wrong.
2. **Doubling the data from 500 to 1,000 trials.** +0.017 on the hard question, nothing on the easy one. Best epoch still 2.
3. **42 Optuna trials over the MLP head.** 0.8836 to 0.8907, and wasted: it tuned an architecture the BiLSTM had already replaced (0.7 points of search against 3.8 of architecture). `SearchableHead` carries a comment recording this.

**The pattern:** the head converges in one epoch, then degrades. It is exhausting what the features contain, so data, regularisation and hyperparameters each bought under a point and the architecture bought five.

**A false negative I produced (method lesson).** I tested "does temporal context help" with hand-made scalar summaries of the neighbours' cosine (own-minus-local-max scored 0.509, i.e. nothing) and called it settled. A learned model over the full 192-d sequence then gained 5.1 points. **A cheap proxy test only rules something out if the proxy is as expressive as the thing it stands in for.**

### The integration, and what it deliberately does not touch

| file | |
|---|---|
| `src/models/state_teacher.py` | new. Frozen ECAPA + BiLSTM readout, hash-checked, shape inferred from the state dict because the checkpoint does not record it |
| `src/models/losses_state.py` | new. `LossBSRNNState(LossBSRNN)`, a subclass in its own file, so `losses.py` is untouched |
| `scripts/derive_w_state.py` | new. Anchor measurement and weight derivation |
| `tests/test_losses_state.py` | new. 8 tests |
| `experiments/configs/bsrnn_state.yaml` | new. Four keys differ from the baseline, verified |
| `scripts/train.py` | five changes |

**`build_loss_fn` branches on `w_state` rather than multiplying by zero.** A config without the key gets plain `LossBSRNN`, so earlier runs reproduce by construction and old configs cannot reach the new code path. Verified: the baseline config still builds `LossBSRNN`.

**`total` excludes `L_state`.** `ReduceLROnPlateau` and the loss curve read `total`, and a fifth term would make runs before and after 2026-09-11 incomparable. `L_state` gets its own column, NaN when the term is off (a gap, not a zero).

**`selection_score` never includes it.** A frozen learned scorer can be beaten by finding its blind spots instead of removing the interferer, and the teacher has never heard masked audio (only real mixtures and synthetic suppressions). Selecting on it would let the gamed quantity choose the checkpoint. **Watch for `L_state` falling while `L_pres` stalls**; both print side by side in the epoch breakdown.

### The test that matters, and the one I mislabelled

`test_the_state_term_is_what_carries_the_gradient` is the real test. With the bug introduced on purpose (teacher forward wrapped in `no_grad`), `|grad(w_state=1) - grad(w_state=0)|` drops from **182.17 to exactly 0.000000**.

**`test_gradient_reaches_the_waveform`, which I had called "THE test", still passes under that bug.** The four base terms deliver gradient anyway, so the waveform's gradient only moves from 26874.5 to 26836.6 (0.14 %). "Gradient is non-zero" does not test this term.

**`L_state` also logs 0.1048 either way**, so the training curve cannot show whether the term works. This is the dead-gradient shape D9 records for the original "add WER to the loss" idea.

### w_state must be DERIVED, and the first attempt found two bugs

**Measured at `w_state = 1.0`: the term gave 0.68 % of the gradient on synthetic audio and 5571 % on real audio**, two orders of magnitude off one way and three the other, so no guess could be right. L_state is a cross-entropy in nats and the others are dB, with no meaningful conversion (BCE is a log-probability, dB a power ratio). `L_MR` is the precedent that a term need not be in dB if its weight reconciles the units.

**Anchors, 6 batches of `sir0_val`:**

| anchor | L_state |
|---|---|
| oracle (clean target) | **0.1212** |
| model (`model_sir0_10000-e6.pt`) | 1.8200 |
| passthrough (the mixture) | **2.1064** |

**Headroom is 1.99 nats.** The wiring check passes: the clean target has no second voice and reads near zero. **The checkpoint removes only ~15 % of the teacher's leakage reading**, so the term has room to push.

**Two bugs the first run exposed, both fixed:**

1. **`remix_gains: true` invalidated every synthetic anchor.** The loader rebuilds the mixture at a fresh SIR each epoch, so it is not `target + interferer + noise` from disk; the residual measured 0.2x to 1.7x the mixture's level. The giveaway was the `partial_6/12/20db` anchors reading worse than the raw mixture on `L_pres`, `L_gain` and `L_abs`. The script now forces `remix_gains=False`.
2. **The suggested weight printed as `0.0`** (`%.1f` on 0.0027). It now prints `%.4g` and re-measures the share at the suggested value instead of linearly extrapolating from 5571 %, which is far outside the small-perturbation regime.

### The one thing that can still stop this, and it is unmeasured

**GPU cost per step:** 78 ECAPA forwards and backwards (13 windows x 6 examples) against 0.674 s/step now. Every timing here is CPU; training is on a T4. **Write no further integration code before `scripts/profile_step.py` measures it.** At 5 s/step the term would have to score a random subset of the 13 windows, which changes the shape of `losses_state.py`.

**Levers. Subsampling windows is sound in principle (the SGD argument).** The term is a mean over 13 windows, so a random subset is an unbiased estimate, as with a mini-batch gradient. Variance costs steps, not correctness.

**But the BiLSTM couples the windows, so random subsampling saves almost nothing.** The head runs bidirectionally over all 13 windows:

- a loss on 4 windows still needs all 13 ECAPA embeddings to feed the recurrence, so the forward is unchanged;
- the backward flows to all 13 hidden states anyway.

**A shorter contiguous segment does save time.** 2.5 s of the 4.008 s chunk is 7 windows instead of 13, and the recurrence runs over 7. The head then sees less context than the 13 windows it was fitted on, a mild distribution shift to be measured.

Levers in order: **score a shorter contiguous segment**; put the teacher on the second T4; precompute enrolment embeddings (~5 s of the ~18 s of audio per example; an earlier claim that this was the larger half held only for a single-window design).

**The number 4 was invented.** "Score 4 of 13 random windows" was suggested as if derived; it was not. The only anchor is that a 1 s window at 0.25 s hop means about **4 non-overlapping windows span the 4 s chunk**, so 13 windows carry far less than 13 windows of independent information. That argues redundancy, not a subsample size. Take the segment length from the profile: the longest that fits the session cap.

### Still not done

- **`sir0_val` is uncached**: the teacher has never been measured on speaker-disjoint data. All 1,000 trials are from `sir0_train`.
- The target column, which needs per-window labels in `dataset_loader.py`.
- Head A, a separate idea sharing those labels.

### MEASURED 2026-09-11 on a T4 — the arm is affordable, and the lever is EPOCHS not the teacher

**The full-fidelity teacher costs +53 % per step. A shorter scored segment is not a cheaper version of the same measurement (`L_state` moves 4.1x across lengths), so the epoch count gives way: 13 windows at 10 epochs is 10.0 h against a 12 h cap.**

`experiments/results/2026-09-11-state-teacher-cost/`, Tesla T4, batch 3 trials = 6 examples, AMP on, synthetic audio.

| windows | s/step | overhead | 16-ep hours | peak GB | L_state |
|---|---|---|---|---|---|
| 0 (baseline) | 0.669 | — | 10.5 | 6.59 | — |
| 1 | 0.743 | +11 % | 11.7 | 6.67 | 0.2542 |
| 3 | 0.780 | +17 % | 12.2 | 6.67 | 0.1385 |
| 5 | 0.822 | +23 % | 12.9 | 6.73 | 0.1038 |
| 7 | 0.870 | +30 % | 13.7 | 7.14 | 0.0892 |
| 9 | 0.926 | +38 % | 14.5 | 7.55 | 0.0742 |
| **13 (full)** | **1.022** | **+53 %** | **16.0** | **8.37** | **0.0623** |

**The baseline reproduces:** 0.669 s/step against the recorded 0.674 (`decisions-m2.md` 2026-08-28).

### The finding: subsampling is not a saving

**`L_state` runs from 0.2542 at one window to 0.0623 at thirteen, a factor of 4.1.** Fewer windows give the BiLSTM less context than it was fitted on, so it is a different measurement. `w_state` would need re-deriving at every length, and arms would not be comparable.

This retires the lever I had planned on. Only a contiguous segment shortens both forward and backward, and that is what drifts.

**Caveat on the 4.1x.** The sweep ran on synthetic noise with no interferer, so `L_state` measures how confidently the teacher says "no second voice", and more context raises confidence. The drift is the finding; its size is specific to this input and should not be quoted as general.

### DECIDED: 13 windows, 10 epochs

**Full-fidelity teacher, no drift, no re-derivation.** 10.0 h, two hours spare.

**10 epochs loses nothing used so far.** The 2026-09-04 run selected epoch 6 of 16 and the one before it epoch 2 of 20; the back half has never produced a checkpoint. Early stopping still applies, so 10 is a ceiling.

**Memory does not bind (8.37 GB of 14.6)**, so the teacher shares the card: `state_device: cuda`. The second T4 is better spent on E7's data parallelism if that lands.

### w_state = 0.002692, derived and verified

`scripts/derive_w_state.py`, `experiments/results/2026-09-11-wstate-anchor-sir0`. **Verified, not extrapolated:** at that value the term gives **15.00 %** of the gradient reaching the waveform, measured with two backward passes at that weight.

**At w_state = 1.0 it gives 5571 % on real audio, 370x too strong.** A run at 1.0 would have looked like the idea failing when the arithmetic was at fault.

**15 % is a judgement, not a derivation.** Four existing terms share the update, so an equal share is ~25 %, and a new unvalidated proxy should get less. The script tabulates 5 % to 30 %. If the arm shows nothing at 15 %, try 30 % before calling the idea failed.

### Three bugs the profiling found, all fixed

1. **The profiler ran fp32; training runs AMP.** The baseline came back at 4.765 s/step and 12.23 GB against the recorded 0.674, 7x slow, which would have made the teacher look seven times cheaper. It now mirrors `train.py` (`autocast` on the model forward only, `GradScaler`, loss in fp32) and prints the recorded 0.674 beside the measured baseline.
2. **`cudnn RNN backward can only be called in training mode`.** The teacher's BiLSTM was in `eval()`, where cuDNN refuses RNN backward, but gradients must flow through it to reach the audio. The head now runs in `train()` mode. This is numerically free only because dropout is forced to 0.0 at construction (`Dropout(0.0)` is the identity in both modes; LayerNorm is mode-independent). An assertion fails if a head with nonzero dropout is loaded, since a stochastic teacher is a moving target.
3. **Two Kaggle runs gave the same traceback from an already-fixed file.** The notebook's copy cell skipped existing files, and `/kaggle/working` persists across runs in a session, so the old code stayed after the dataset was updated. Only the traceback's line numbers showed it. It now overwrites, and asserts two fixed lines are present before running.

### And one that was not a bug in the code

**`../ecapa_pretrained/` is 24 KB of symlinks into the HuggingFace cache**, because SpeechBrain's `from_hparams` links rather than copies. Zipped as links, the upload succeeds and lists five files, but the teacher fails to load on Kaggle with a missing-path error. `make_kaggle_bundle.py` now dereferences (85 MB resolved) and re-hashes against the hashes stored in the teacher checkpoint, the same discipline as `--prefix-manifest`.

### Figures, report-ready

`scripts/plot_state_teacher_cost.py` renders three single-panel, half-textwidth PDFs from `results.json`, so any two sit side by side:

| | |
|---|---|
| `state_teacher_cost.pdf` | run length against windows, 16 vs 10 epochs, with the cap |
| `state_teacher_drift.pdf` | **the finding**: L_state against windows |
| `state_teacher_memory.pdf` | peak memory, which does not bind |

Separate rather than multi-panel, because the cost figure supports a scheduling decision and the drift figure is a result about the teacher.

### The arm is now fully specified

`experiments/configs/bsrnn_state.yaml` differs from the baseline in exactly five keys: `w_state` 0.002692, `state_teacher`, `ecapa_dir`, `state_device` cuda, and `epochs` 10. Architecture, parameter count, latency, data, seed and schedule are unchanged. **No capacity confound: the teacher is training-only and deleted at inference.**

The control is `models/model_sir0_10000-e6.pt`, not a fresh baseline run.

### Two controls, without which head A means nothing

1. **Ablate the enrollment and re-measure state accuracy.** If accuracy holds without the cue, the head is a voice-activity detector and the "who" half is unearned.
2. **Split accuracy same- vs cross-gender.** The extractor already leans on gender (56.1 % vs 44.4 %, 2026-08-30), a frame classifier can hide that more easily, and pooled accuracy would conceal it.

`both_directions` helps: roles swap per mixture, so "target = louder" is not learnable.

### Weighting: derive it, do not pick it

Head A's neutral default is inverse class frequency from the measured label histogram; otherwise `both` dominates and an unweighted head just predicts it. Head B's errors cost differently in the metric (a missed target frame becomes a deletion the judge never hears; a leaked interferer frame becomes an insertion it attributes to the target), and the metric can measure which. Same discipline as `w_g` = 1.69 and `w` from the measured 0.297 absent rate: derived, not tuned.

### Sequence, and what fits before 14 October

0. **Measure ICR first (D10's free measurement, which gates this family). Answered 2026-09-11, and it passes** (see the measured section at the end of D14). Leakage is 58.5 % of our content-word error mass, and its per-trial rank correlation with WER is +0.622 over 103 trials, surviving inside difficulty strata. Head B and D10 keep their motivation. **No transcription pass was needed:** the ICR aggregate was computed on 2026-09-04 and never read back, and every transcript was already in `experiments/results/transcripts.csv`.
1. **Label script.** Hours. A shared dependency of everything below, and of the sweep.
2. **State-conditioned oracle sweep. No training, no GPU.** Extend `sweep_alpha` with a per-state blend from the true labels. First fix a limit: D11's `alpha` blends toward the *mixture*, which on target-absent frames passes the interferer through, so a second axis is needed (a gain toward silence on those frames). A 3x3 grid over `sir0_val` `both` (n=103) is ~900 transcriptions, under an hour of CPU. **A deliverable either way:** it closes the per-frame-oracle gap and either retires or confirms M6's measurement risk for M5's gate.
3. **C, folded into M5's already-scheduled gate arm.** One term in one line, ~1 k parameters, no separate run. Only if step 2 says the headroom clears the judge's SEM.
4. **A alone, then B alone. Post-freeze, or if something is cut.** One variable per run: the 2026-08-25 lesson was three variables and one number, attributable to none.

### Stopping rules

- **Step 0 says leakage is not dominant:** drop B. A and C survive on the level/gating argument.
- **Step 2 lands near 2.2 points:** drop C; M6's recorded caution stands. **This says little about B**, which can change the mask itself rather than blend two existing signals.
- **Head A fails either control:** do not build C. A gate driven by a bad state estimate is worse than no gate.

### Honest cost ranking against the rest of Group D

B is the most interesting piece and the most expensive: a label script, a detector architecture, a detector training run, then the arm. `BETA` is one number (`BETA = 1` recovers `L_pres` exactly), and D10 is one term on a stem already loaded. **Neither is beaten by this on cost-to-evidence.** Inside the freeze, D14 contributes steps 1-3, not B.


### MEASURED 2026-09-11 — step 0 passes. Leakage is the dominant single error, but it is not the only one

`../tse_venv/bin/python scripts/analyse_leakage_share.py`, 1.2 s measured, no GPU, no ASR pass. Results in `experiments/results/2026-09-11-leakage-share/`. `sir0_val` `both`, n=103, the same trials and cached transcripts as the published numbers.

**The aggregate already existed and had never been read.** `results.json` in `experiments/results/2026-09-04-train-sir0-10000/` has `icr_at_2` and `mean_leak` for floor, our baseline and ceiling, written 2026-09-04. D14 assumed the number did not exist; measuring it meant reading a file.

Also: **`scripts/eval_asr_wer.py` is a zero-byte file**, committed empty in `38bf48f` on 31 August. D14's step 0 named it as the tool to run.

#### The published aggregate, read back

| system | WER | sub | del | ins | ICR@2 | mean leaked |
|---|---|---|---|---|---|---|
| floor, raw mixture | 65.2 | 32.9 | 9.3 | 23.1 | 67.0 % | 51.3 % |
| ours, baseline | 59.5 | 28.2 | 11.8 | 19.5 | 50.5 % | 34.6 % |
| WeSep | 34.6 | 18.5 | 8.9 | 7.2 | 15.5 % | 9.0 % |
| ceiling, clean target | 5.8 | 3.4 | 0.7 | 1.7 | 0.0 % | 0.0 % |

#### Why the aggregate alone does not answer the question

ICR@2 = 50.5 % says the interferer leaks *often*, not that leakage causes most of the *damage*. Three measures were added.

**1. Error-mass attribution.** Of the content words the listener reported that the target did not say, how many did the interferer say? Same normaliser and stopword list as `icr.py`, so it is consistent with the published ICR.

| system | wrong content words | the interferer's | neither speaker's | leakage share |
|---|---|---|---|---|
| floor | 904 | 635 | 269 | **70.2 %** |
| ours | 742 | 434 | 308 | **58.5 %** |
| WeSep | 406 | 94 | 312 | **23.2 %** |
| ceiling | 91 | 0 | 91 | 0 % |

**2. Per-trial rank correlation of leaked fraction against WER, within one system.** The cross-system comparison is n=2 and confounded (WeSep is better at everything). Within our baseline, across 103 trials, rho = **+0.622**. WeSep +0.468, floor +0.628.

**3. Leakage quartiles, our baseline.** WER is 24.9 % in the least-leaky quarter of trials and **101.5 %** in the most-leaky, a 76.6-point spread inside one frozen model.

#### The confound control, which is what makes this defensible

A positive correlation could just be difficulty: a heavily overlapped trial leaks more *and* is harder anyway. Repeating measure 2 inside strata holds difficulty roughly fixed. Our baseline:

| stratum | n | rho | mean WER | mean leaked |
|---|---|---|---|---|
| overlap low | 65 | +0.602 | 64.4 % | 37.4 % |
| overlap mid | 33 | +0.740 | 66.6 % | 31.2 % |
| target louder | 42 | +0.687 | 42.1 % | 14.5 % |
| interferer louder | 55 | +0.300 | 89.0 % | 53.8 % |

It survives everywhere it can be measured. The weak cell, interferer louder at +0.300, is a ceiling effect: mean WER there is 89.0 %, leaving little room for leakage to make it worse. Overlap `none` (n=0) and `high` (n=5) are below the n>=8 floor and are not ranked.

#### What this authorises, and what it does not

**It authorises head B.** Leakage is the largest identified component of our error, over half of it, and it tracks WER trial by trial, not only in a two-point average.

**It does not authorise assuming that removing leakage removes the error.** Two measured limits:

1. **41.5 % of our wrong content words were said by nobody.** WeSep, having largely solved leakage (23.2 % share), has *more* invented words than we do: 312 against our 308. Zero leakage leaves that residue, and `fabrication.py` found the same independently (both extractors raise fabrication ~48 % above doing nothing). The realistic prize is roughly the 434 leaked words, not the 742.

2. **Our model already deletes more target content than doing nothing.** Deletions: mixture 9.3, ours 11.8, ceiling 0.7. Extraction *adds* 2.5 points of deletion while removing 3.5 of insertion and 4.7 of substitution, a net WER gain of only 5.7 points for a 16.7-point cut in leakage. **Head B's pressure is one-sided: it rewards no non-target voice being audible and says nothing about keeping the target audible.** Only `L_pres` and `L_gain` oppose deleting the target, not `L_state`. So watch deletion rate and enrolment sensitivity on the M5 arm, not `L_state` alone.

Recorded so it is not re-derived: at epoch 1 the M5 arm's enrolment sensitivity was -13.54 dB against the baseline's -12.08 dB, i.e. less responsive to the enrolment, as this failure mode predicts. One epoch, one seed, and the baseline reached -9.17 by epoch 2, so check at epoch 3; it is not a finding.

### MEASURED 2026-09-11 — the arm and its control do NOT see the same training order. Building the teacher reshuffles the dataset

The header of `bsrnn_state.yaml` claims "same data, same seed, same schedule -- one term added". **The first two words are wrong.**

The arm's epoch 1 logged 12,624 present / 7,284 absent crops against the baseline's 12,625 / 7,283. One crop in 19,908 looked like rounding. It is not.

**The chain, each link verified:**

1. The train loader is `shuffle=True, drop_last=True, batch_size=3` (`train.py` ~899). `sir0_train` holds **9,955** trials; 9,955 / 3 leaves one over, so `drop_last` discards **whichever trial lands last in the shuffle**. 9,954 x 2 directions = 19,908, matching both runs.
2. Counted locally through the real `TrialDataset` at seed 42, epoch 0, without `drop_last`: **12,626 / 7,284, total 19,910**. Both runs are that minus one trial's two crops, so the data is the same and only *which* trial was dropped differs.
3. `build_loss_fn` (`train.py` 555) runs before the first batch. For the arm it builds `StateHead` (two `LayerNorm`, two `Linear`, one bidirectional `LSTM`), and **each draws from the global RNG at construction**.
4. Direct test, seed 42, same stand-in for `build_model`'s consumption:

   | | permutation starts | trial dropped |
   |---|---|---|
   | no teacher built | 6707, 1302, 4478, 4939 | 8201 |
   | teacher built | 7502, 7622, 5487, 2436 | 1658 |

**So the arm and the control see the same crops in a completely different order.** The crops are stable (`_crop_offset_start` keys on `(seed, epoch, idx)`), but the SGD trajectory is not.

**Consequence for reading the arm.** The epoch-1 result (val `L_pres` -2.4230 against the baseline's -1.6081 at the same epoch) carries an ordering perturbation of unknown size. No same-config replicate exists in `experiments/results/`, so ordering noise has never been measured and cannot be subtracted. The gain is ~1.9x the epoch-0 spread across all eight prior runs, the best available evidence it is real, but those runs differ by more than ordering. **Epoch 3-4 remains the check**: ordering noise washes out, a teacher effect does not.

**Two explanations claimed in conversation, both wrong:**

- `num_workers`. The arm's source config says 0 and the baseline's *recorded* config says 4, but that is a post-patch artefact: both Kaggle notebooks set `NUM_WORKERS = 4` and write it back before `train.py` reads it, and both *source* configs say 0. The loader is worker-count independent anyway, since every draw keys on `(seed, epoch, idx)`.
- "No LR scheduler." There is one, `ReduceLROnPlateau` (`train.py` ~1132). It is plateau-driven, so `epochs: 100 -> 10` still does not change the training path, but the stated reason was wrong.

**DECLINED 2026-09-11.** A fix exists (give the train loader its own `torch.Generator`, reseeded per epoch from `(seed, epoch)` to stay correct on resume) and was not taken:

Ordering is seed noise, carried by every single-run A/B comparison here, including the 2026-09-04 baseline (itself one run). Pinning one future arm's ordering gives no noise floor, changes no existing number, and leaves the other sources: cuDNN's nondeterministic RNN backward, and two arms having different gradients from step 1 by construction.

**The real gap is elsewhere:** until run-to-run noise is measured, no single-run delta is separable from scatter, with or without the generator fix. `report-todo.md` #9 records the same problem for the judge: "a system difference smaller than it cannot honestly be claimed."

Re-open only if a Group D arm lands a delta small enough for ordering to explain, and only with an actual noise measurement.

**The general lesson.** Any arm that adds a module (D13's gate, D12's experts, D5's speaker encoder) consumes RNG at construction and silently reshuffles its training set relative to its control. Every future "one term added" comparison in Group D has this bug unless the generator fix lands first.

### BUILT 2026-09-11 — head A as code. Runnable except for one thing: the loader owes it labels

`src/models/state_head.py` (head, class weights, cross-entropy, per-class recall, checkpoint stripper), `src/models/losses_state_head.py` (`LossBSRNNStateHead`), the `state_head` flag on `BSRNN_TFMAP` and in `build_model`, and `tests/test_state_head.py` (25 tests, all passing).

**516 parameters, verified by a test:** `AuxStateHead(feature_dim=128).n_parameters == 516`.

**It taps the separator's output before `lookahead_shift`.** The shift moves frame `t`'s features to `t-k`, so the mask for `t` sees `t+k`, but the state label for `t` is still about `t`. Reading off the shifted tensor would misalign label and features by k frames: invisible at today's `lookahead_frames: 0`, silent the moment that key is raised.

**It is trained on the actual state label, not `REQUIRED_OUTPUT_STATE`.** That mapping (`both -> target only`) is for head B, which scores output audio. Head A reads internal features, and a model that notices "both speakers are here, and I am about to suppress one" should be rewarded. The required-output mapping would ask the features to forget the interferer they need to remove it.

**Class weights are inverse frequency, normalised to mean 1.** This makes the term's magnitude independent of which split's histogram was used, so a derived weight keeps its meaning. It also fixes chance at ln(4) = 1.386 nats, readable without a baseline run.

**The head forks the RNG at construction**, which settles the 2026-09-11 shuffle problem for this arm without the declined global fix. `torch.random.fork_rng` makes head A's init deterministic and invisible to everything built after it; a test asserts that `torch.randn(3)` after building the model is bit-identical with the head on and off. **This is the narrow fix.** D13's gate and D12's experts still have the bug, because their parameters must sit in the main RNG stream to stay comparable.

**Off by default; `forward`'s return type is unchanged.** `return_state=True` opts in. Asking a headless model for state raises an error instead of returning `None`, since a silently skipped term would train a baseline reported as head A. `drop_state_head()` strips the weights so a head-A checkpoint loads `strict=True` into the baseline architecture at eval, rather than loosening eval to `strict=False`, which could swallow a missing separator weight.

**A free diagnostic arm: `state_head_detach`.** The head learns to read the features but no gradient reaches the separator, so it measures how decodable state already is, online, with no pressure. It is the control for "did the auxiliary loss change the features, or were they always like this?", and the training-time counterpart of `scripts/probe_state_features.py`.

**Not done, and the arm cannot run without it:** `dataset_loader.py` does not yet emit per-frame state labels for a crop, so `loss_fn.state_labels` has nothing to be set from. One loader change, shared with head B's target column. No config exists yet, on purpose: an untested YAML for an arm that cannot start is a liability, and the weight must be derived against a real `L_head` reading, as `w_g` and `w_state` were.

**A and B do not compose.** `LossBSRNNStateHead` and `LossBSRNNState` are sibling subclasses of `LossBSRNN`, by design: D14 says A alone, then B alone. Both together give one number attributable to neither, the 2026-08-25 mistake.

### MEASURED 2026-09-11 — probing head A's features BEFORE building the arm. Capacity settled, and a gender shortcut found

`scripts/probe_state_features.py --limit 100 --max-seconds 8.0`, 13 min on CPU. `experiments/results/2026-09-11-state-probe/`. 100 `sir0_val` trials, both directions, `model_sir0_10000-e6.pt` frozen, `z` captured off `model.separator` with a forward hook, probes fit on a trial-disjoint 70/30 split (frames within a trial are too correlated for a frame-level split).

**Why before the arm:** head A is a ~10 h run premised on pushing the separator's features to encode speaker state. Whether they already do is free to check and decides the head's design.

#### Capacity: 516 parameters is right, and my band-resolved argument was wrong

| probe | params | balanced accuracy |
|---|---|---|
| linear, mean-pooled — **D14's specified head** | 516 | **61.0 %** |
| MLP, mean-pooled, hidden 128 | 17,028 | 62.0 % |
| linear, band-resolved (no mean-pool) | 16,388 | **59.0 %** |
| chance | | 25.0 % |

33x the parameters buys 1.0 point. **The band-resolved probe is worse**, which kills the argument (made in conversation) that mean-pooling discards which frequencies the second voice occupies and that capacity should go on the band axis. Head A stays at 516.

#### The features do not already encode state usably

| state | recall |
|---|---|
| none | 93.1 % |
| target only | 72.0 % |
| interferer only | 39.6 % |
| **both** | **39.4 %** |

Nearly all of the 61 % is silence detection. **The two states a gate needs, `both` and `interferer only`, sit at ~39 %.** So head A has real work to do and its auxiliary loss is not redundant. But piece C driven by today's features would fail D14's own stopping rule ("a gate driven by a bad state estimate is worse than no gate").

#### The gender shortcut, which D14 predicted

| | balanced accuracy | frames |
|---|---|---|
| same gender | **53.4 %** | 18,054 |
| different gender | **66.5 %** | 42,126 |

A 13-point gap, as D14's control 2 predicted: "a frame classifier is an easier place to hide it; pooled accuracy would conceal it." For two speakers of the same gender, the features are close to useless beyond silence detection, and much of the headline is gender. **Head A trained on this would lean on the shortcut harder, and any gate built on it would inherit it.** Report the split, never the pooled number.

#### Control 1 was built wrong and is inconclusive

The "wrong enrolment" arm used the other speaker in the same trial. That is a different valid instruction, not an ablation. It scored 61.3 % against 61.0 %, which naively reads as "the cue is ignored". The per-class recalls say otherwise:

    real enrolment     target 72.0   interferer 39.6
    wrong enrolment    target 39.9   interferer 72.2

**They swap cleanly**: the features track which speaker was requested, and the roles trade when the cue trades. Encouraging, but not the control D14 asked for. The correct ablation is a stranger from a different trial, with no role to swap into. Rerun before acting on control 1 either way.

### D15. Penalise mask ROUGHNESS over time — attack fabrication, not leakage

**Status: PROPOSAL, raised 2026-09-12 (Grant). Step 0 is free and answers most of it. Nothing here is built.**

**The gap nobody is working on.** Measured 2026-09-11: **41.5 % of our wrong content words were said by nobody** (308 invented against 434 leaked). Every open Group D proposal attacks leakage (D10, D14 B, D13's gate); none attacks invention. `fabrication.py` found independently that both extractors raise fabrication ~48 % above doing nothing, so **extraction is causing part of this, not failing to remove it.**

**The mechanism is textbook.** A mask that changes abruptly between frames produces isolated, flickering spectral peaks: **musical noise**, the classic artefact of spectral-subtraction and masking systems. It sounds like chirping, but more importantly a spurious spectral transient looks like an onset to an ASR front end, and word hypotheses are built from onsets. **Invented words are the error a flickering mask would produce.**

**Nothing in the model or objective opposes it.** `Estimator` predicts a complex mask per band per frame through a 1x1 conv on the feature stream, and GLU bounds its magnitude, but nothing limits how much it changes from frame t to frame t+1. `L_MR` prices detail at four resolutions but is minimised by matching the reference, not by smoothness, and was measured to reward muting (decisions-m2.md 2026-08-28).

**Borrowed, with a difference.** Temporal smoothing of a spectral gain is standard in speech enhancement: the decision-directed a-priori SNR estimator (Ephraim & Malah, IEEE TASSP 1984) is essentially a recursive smoother and the canonical musical-noise fix, and cepstral-domain smoothing (Breithaupt, Gerkmann & Martin, ICASSP 2008) is its modern form. In images it is total variation regularisation (Rudin, Osher & Fatemi, Physica D 1992). Those smooth a gain for perceived quality and are tuned on PESQ-style measures. Here the aim is content fidelity for a downstream listener, judged on invented-word count and LCF-WER, never DNSMOS. That is a different claim with a different acceptance test, which our metric can distinguish.

### The trap, stated before the arm rather than after

**Speech has real transients.** Plosive releases, stop bursts and word onsets are fast changes, and a mask that cannot move quickly smears them. We already delete more than doing nothing (deletions 11.8 against the raw mixture's 9.3), so blunt smoothing would trade one excess error for more of the other. **"Smoother is better" must not be the hypothesis.**

**The fix: derive the target from the oracle mask, which is free.** We own `target.wav` and `mixture.wav` for every trial, so the ideal mask and its frame-to-frame variation are exact. Penalise roughness **beyond what the oracle mask shows**, per band, with the same deadzone shape as `L_gain`'s +-3 dB. This makes the hypothesis falsifiable and self-limiting. If our mask is no rougher than the oracle's, the proposal dies at step 0 for an afternoon's cost.

### Sequence

0. **Measure the roughness gap. No training, no GPU, hours.** On `sir0_val`, compute per band the mean absolute first difference along time of (a) the oracle mask |S_target| / |X_mixture| and (b) our predicted mask. Every outcome is useful:
   - ours is much rougher: the premise holds, go to 1
   - ours is comparable: **the proposal is dead**, recorded, no run spent
   - ours is SMOOTHER: we already over-smooth, which would explain the deletions and points the opposite way
   Correlate the per-trial gap with that trial's invented-word count from `transcripts.csv`, as D14 step 0 did for leakage against WER. A gap that does not track invention is not the mechanism.
1. **A loss term.** L1 of the mask's first difference along time, per band, deadzoned at the oracle's roughness. One term, one weight, derived against a measured anchor like `w_g` = 1.69 and `w_state` = 0.002692, never picked.
2. **Or an architectural constraint**, if the term is hard to weight: a causal one-pole smoother on the mask with a learned per-band coefficient. Zero added latency (causal IIR, one multiply-add per bin), ~32 parameters, a hard constraint instead of a soft penalty. Initialise at no smoothing so the arm starts as the baseline, like D4a's zero-init gates.

**Best cost-to-evidence in Group D right now.** Step 0 needs no GPU, training or API budget, and either kills the idea or hands the arm a derived weight. D14 B, by contrast, cost a label script, a detector architecture, a detector training run and a 10.25 h arm to move its own term 1.4 % (decisions-m2.md 2026-09-12).

**It composes with everything.** It constrains the mask over time, not who the target is, so it is orthogonal to D4a, D13 and D14 and can be added to whichever survives.

### MEASURED 2026-09-12 — D15's premise is WRONG, and what replaced it is worse news

D15 assumed a flickering mask was producing musical noise and inventing words. **Step 0 refutes that.** `scripts/plot_mask_grid.py` and a 12-trial measurement on `model_sir0_10000-e6.pt`, against the ideal mask `|target| / |mixture|`:

| | varies along TIME | varies along FREQUENCY | freq / time |
|---|---|---|---|
| our mask | 0.0445 | 0.0238 | **0.53** |
| ideal mask | 0.1540 | 0.1551 | **1.01** |
| shortfall | **3.5x** | **6.5x** | |

**Our mask is not too rough. It is 3.5x too smooth along time and 6.5x along frequency**, the third outcome D15 listed ("ours is SMOOTHER, which points the opposite way"). Smoothing is retired as an intervention. Step 0 cost an afternoon and saved a training run.

### The replacement finding, and it is structural

**84.2 % of our mask's variance is explained by a single number per frame.**

**The model has learned a broadband volume knob, not a time-frequency mask.** It raises the output when the target speaks and lowers it when they do not, with nearly the same gain at every frequency in a frame. The ideal mask varies equally along both axes; ours varies half as much across frequency as across time.

No loss weight fixes this. Two overlapping voices share frequencies at the same instant, and separating them means deciding per time-frequency cell which voice owns it. **A broadband gain cannot do that even in principle.** It can only be loud when the target talks, which is voice activity detection in an extractor's architecture.

It explains, at once:
- why leakage survives: a volume knob passes both voices when both speak;
- why 62 % of the output is enrolment-independent (2026-08-30): a volume knob needs to know when someone speaks, not who;
- why the state probe reads "somebody is speaking" at 93 % and "which of the two" at 39 % (2026-09-11);
- why the frozen state teacher moved leakage a little and cost fidelity: the model's only lever is turning the knob down harder.

### CONFIRMED 2026-09-12 — the holes really do delete target speech

`experiments/results/2026-09-12-eval-floor0.05`, an inference-time mask floor of 0.05 on the baseline checkpoint, n=103, ASR stand-in:

| | floor 0.00 | floor 0.05 |
|---|---|---|
| deletions | 11.79 | **8.64** (−3.14, −27 %) |
| ICR@2 | 50.49 | 58.25 (+7.77) |
| no response | 2.91 | 0.97 |
| LCF-WER | 59.52 | 63.33 |

**The deletion drop is the largest single metric movement this project has produced**, confirming that the zeroed bins carried target speech.

**It also shows the floor is the wrong cure.** The raw mixture fills a hole with both speakers, so leakage rises more than deletions fall. A target-selective fill means redistributing the existing gain across frequency instead, i.e. `scripts/postprocess_mask.py`.

**A floor is still worth keeping as a knob**: it gives deletions and leakage a measured exchange rate, and nothing else in the project trades between them explicitly.

### MEASURED 2026-09-12 — frequency structure is the lever. A free post-hoc version beats a 10-hour training arm on leakage

All paired bootstraps, 10,000 draws, n=103, `sir0_val` `both`, ASR stand-in. The metric's irrelevance floor is 1.57 points (decisions-m3.md); anything under that is not a claim.

| intervention | cost | leakage change | LCF-WER | verdict on leakage |
|---|---|---|---|---|
| frozen state teacher (D14 B) | **10.25 GPU-h** | -1.93 | +1.72 | inside noise |
| post-hoc frequency sharpening | **free** | **-6.99** | +10.01 | **REAL**, [-10.41, -3.85], 0.0 % opposite sign |
| mask floor 0.05 | free | +5.39 (worse) | +3.81 | REAL, wrong way |

**Crude post-hoc sharpening across frequency, with no learning, removed 3.6x more leakage than the whole teacher arm, and unambiguously.** It took leakage from 34.58 to 27.42 against WeSep's 9.0, closing roughly a quarter of that gap.

It also cost a real +10.01 LCF-WER, the expected price of imposing structure on a model never trained for it: the sharpening acts on the output ratio, which carries the additive residual and an analysis mismatch, and nothing optimises the result.

### The two knobs point opposite ways and both lose, which is itself the finding

| | deletions | mean leaked | LCF-WER |
|---|---|---|---|
| baseline | 11.79 | 34.58 | **59.52** |
| fill the holes (floor 0.05) | **8.64** | 40.87 | 63.33 |
| sharpen across frequency | 15.13 | **27.42** | 69.53 |

**The baseline already sits near a local optimum on the deletion-versus-leakage trade-off**; moving it after the fact costs more than it gains either way. The remaining gain is in letting the model make fine time-frequency decisions at all, which the 84 % volume-knob measurement says it currently cannot.

**This is now the best-evidenced direction in Group D**, for an afternoon of CPU rather than a training session.

### CORRECTION and the headline, 2026-09-12 — HARD sharpening: real leakage removal at no measurable WER cost

The `mid` setting (1.3/0.7, unsupported bins scaled to 0.3) cost +10.01 LCF-WER and +7.19 insertions, and was written up above as "sharpening adds artefacts". **That was the setting, not the idea.** The `hard` setting (1.5/0.5, unsupported bins removed outright) behaves completely differently:

| | measured | 95 % interval | verdict |
|---|---|---|---|
| mean leaked % | **-6.38** | [-10.56, -2.45] | **REAL improvement** |
| LCF-WER | -0.90 | [-5.18, +3.12] | inside noise AND below the 1.57 floor |
| insertions | -3.81 | [-9.03, +0.63] | inside noise |
| deletions | +3.29 | [-1.78, +8.54] | inside noise |

**A free post-processor removed leakage for real at no measurable cost on the headline metric.** The 10.25-hour teacher arm achieved neither.

**Partial suppression is worse than complete suppression.** down=0.3 leaves a scaled copy of every unsupported bin and insertions rose 7.19; down=0.0 removes them and insertions fell 3.81. This is the classic musical-noise result (a half-removed component is an artefact, a removed one is silence), and should govern any future mask post-processing or gating rule here.

**Two caveats that travel with it.**
1. **-0.90 LCF-WER is not an improvement.** It is inside the interval and below the irrelevance floor: claim "unchanged", never "better".
2. **It is not a uniform win**: better on 29 trials, worse on 46, tied on 28. The flat corpus number comes from helping a few trials a lot and hurting many slightly. Any write-up must say so.

### MEASURED 2026-09-12 — the corrected internal-mask run. Region growing needs something structured to grow FROM

Re-run on the fixed `apply_hysteresis` (RMS level restoration). Output level is -4.11 dB against the baseline's -5.65 dB, so the level explosion is gone.

| variant | leakage | LCF-WER |
|---|---|---|
| frozen state teacher, 10.25 GPU-h | -1.93 (noise) | +1.72 (at the 1.57 floor) |
| sharpen output ratio, partial removal | **-6.99 REAL** | +10.01 REAL worse |
| **sharpen output ratio, full removal** | **-6.38 REAL** | -0.90 unchanged |
| sharpen internal mask, full removal | **-10.37 REAL** | **+16.91 REAL worse** |

Internal-mask bootstrap: leakage -10.37 [-15.29, -5.54], 0.0 % opposite sign; LCF-WER +16.91 [+5.28, +30.98], 0.1 % opposite sign. Both real.

**Every sharpening variant removes leakage substantially and unambiguously**: four independent settings, all outside the interval, against a training arm that never managed it. **Frequency structure is the lever.**

### The internal-mask failure is the informative result

It removed the most leakage (-10.37, best of the day, 34.58 to 25.01 against WeSep's 9.0) and did the most damage (+16.91 LCF-WER, insertions +11.32).

**The mask it grows from is flat.** With 84 % of variance in one number per frame, the bins that clear a relative threshold are picked by tiny fluctuations, effectively noise, so the imposed structure is arbitrary. It deletes the interferer and everything else alike. The output ratio works better because it is not flat: its real spectral content makes thresholding select meaningful bins.

**Region growing needs a meaningful confidence map, and no post-processor can supply one this model does not produce.** The model has to learn a mask with structure worth growing: the architectural form of the idea, now backed by measurement rather than intuition.

**Consequence for the plan.** The cheap post-hoc route is exhausted. Its best outcome, -6.38 leakage at no WER cost, is achieved, and the mask's flatness, not the growing rule, blocks anything better. The next move is to make the mask structured during training.

### MEASURED 2026-09-13 — the volume knob, DECOMPOSED. What frequency shape exists is a FIXED EQ curve, and it carries no speaker information

`scripts/diagnose_mask_structure.py`, 12 `sir0_val` `both` trials, whole clips, `model_sir0_10000-e6.pt`, seed 42, 16 min CPU. `experiments/results/2026-09-13-mask-structure/`.

**Why, when 84.2 % was already measured.** "One number per frame explains the mask" has two readings that both draw as vertical stripes on the mask grid. Either the gain is flat, or a fixed spectral shape is scaled up and down per frame (the persistent dark band below 500 Hz is visible in `mask_grid_sir0_val-42-000004_floor0.png`). Neither separates two overlapping voices, but only the second means the model learned anything about frequency.

**The decomposition.** Our own two-way additive ANOVA form on the mask magnitude: `M(f,t) = mu + eq(f) + gain(t) + interaction(f,t)`. The terms are orthogonal by construction, so the variance shares are exact, not fitted. Only `interaction` can separate two voices: it alone says "this frequency, at this instant, belongs to the target", which must change from frame to frame.

| frames | mask | gain(t) | eq(f) | interaction |
|---|---|---|---|---|
| all | **ours** | **83.5 %** | 7.1 % | **9.4 %** |
| all | ideal | 53.0 % | 2.5 % | 44.5 % |
| speech (93 %) | ours | 82.6 % | 7.7 % | 9.7 % |
| speech | ideal | 52.0 % | 2.6 % | 45.4 % |
| **overlap (26 %)** | **ours** | **46.4 %** | **35.2 %** | **18.3 %** |
| **overlap** | ideal | 16.1 % | 8.0 % | **75.9 %** |

**This replicates 84.2 % by a different method** (83.5 % over all frames): the earlier figure was `postprocess_mask.py`'s frame-mean ratio, this is a variance decomposition.

**Overlap frames are the real test and are reported separately.** Ducking silence is free and not a skill, and pooling it inflates `gain(t)`. Overlap means frames where target and interferer are both active, the only regime where separation is a question.

### The two findings, and the second is the one that was not already known

**1. During overlap, 81.7 % of our mask is "one fixed shape x one number".** No setting of that number can separate two voices. The real per-cell decision is 18.3 % against the correct answer's 75.9 %.

**2. Our mask's frequency shape is static, so it carries no speaker information.** At 35.2 % during overlap it is the second largest term, but it is the same curve across the clip, so it cannot encode which speaker owns a bin. What looked like partial frequency selectivity is a baked-in EQ.

**The ideal mask shows the model applying its one tool to the wrong problem.** For the correct answer, `gain(t)` is only 16.1 % during overlap, since you cannot turn one voice down without the other. Ours spends 46.4 % there.

### Three caveats that travel with these numbers

1. **Shares are relative to each mask's own variance.** Ours varies far less in absolute terms (0.0238 across frequency against the ideal's 0.1551, 2026-09-12), so the absolute shortfall is larger than the share gap. Never read "18.3 % against 75.9 %" as "we do a quarter of the job".
2. **"Fixed EQ" is measured within a clip.** Whether the curve is the same across clips (baked into the weights, not adapted per mixture) is not measured; the script stores shares, not curves.
3. **n=12 trials, one checkpoint, no interval.** A structural share this large is unlikely to be sampling noise, but no significance is claimed.

### What it authorises

- **The next training arm prices the interaction term, not the whole mask.** A plain mask-MSE against the ideal ratio mask would be largely satisfied by `gain(t)`, which the model already produces. The term must target what is left after removing `gain(t)` and `eq(f)`.
- **The interaction share is a per-epoch readout that needs no ASR.** It reads out in one epoch, against a metric whose noise floor is +-8 LCF-WER points on `sir0_val`, so a 1-2 epoch smoke run is a real gate before a 10 h session.
- **It does not authorise building anything yet.** The free step 0 is the oracle volume knob: flatten the ideal mask to `gain(t) x eq(f)`, synthesise and score it. If a perfect volume knob still transcribes badly, flatness is the cost and the arm is justified. If it scores near the 5.85 ceiling, flatness is a red herring and our gain is wrong, a far cheaper fix. One ASR pass, no GPU, decisive either way. **Run this first.**

### MEASURED 2026-09-13 — D6's residual-branch ablation. R is INERT, and the fabrication hypothesis it was built to test is DEAD

D6 flagged `Estimator`'s additive residual `R` as "unbounded and unconditioned" and asked for an ablation arm. This runs it at inference, for CPU hours.

**The hypothesis, stated before the run.** The output is `S = M (x) X + R`. The mask multiplies, so where |X| = 0 it contributes exactly 0 (verified: `masked_energy_in_silent_bins` measured 0.0000). `R` is added from a raw Conv1d with no GLU and no bound, so it is the only path that can put energy in a cell the microphone never recorded. Emitting sound nobody made is physically what an invented word is, and invented words are 41.5 % of our wrong content words with no assigned mechanism. **The proposal was that R is that mechanism.**

### Step 0, the correlation. It refused the hypothesis before the ablation ran

`scripts/diagnose_residual.py`, n=103, `sir0_val` `both`, `model_sir0_10000-e6.pt`. Per-trial R contribution against per-trial invented words, the same shape as D14 step 0 and D15 step 0:

| measure | mean | r vs invented words | significant at n=103? |
|---|---|---|---|
| R's share of output energy | 8.6 % | **-0.161** | no (crit 0.194) |
| R's share after cancellation | 6.9 % | **-0.178** | no |
| output energy in silent bins | **0.72 %** | **-0.149** | no |

**All three negative, none significant.** Trials where R contributes more do not invent more.

**The strongest evidence against: R is not aimed at the silence.** Silent bins are 10 % of all bins and hold **8.0 %** of R's energy (p10 7.1 %, p90 9.0 %), slightly under proportional and almost constant across trials. A fabrication mechanism would concentrate there; R is spread uniformly.

### The ablation. Deleting R changes nothing measurable

`--residual-scale 0.0`, same checkpoint, same 103 trials, ASR stand-in. Paired bootstrap, 10,000 draws.

| | baseline | R deleted | difference | 95 % interval | verdict |
|---|---|---|---|---|---|
| LCF-WER | 59.52 | 59.72 | **+0.20** | [-3.02, +3.30] | INSIDE NOISE, and below the 1.57 floor |
| mean leaked % | 34.58 | 37.21 | +2.13 | [-1.39, +5.88] | INSIDE NOISE |
| insertions | 19.53 | 18.63 | -0.90 | | below the floor |
| deletions | 11.79 | 10.77 | -1.02 | | below the floor |
| substitutions | 28.20 | 30.33 | +2.13 | | |
| **invented content words** | **308** | **306** | **-2** | | **0.6 % of 308** |

Per trial: better on 24, worse on 33, tied on 46.

**Deleting 8.6 % of the output energy moved the headline metric 0.20 points and the invented-word count by two words.** None of the three outcomes registered in advance occurred. R neither fabricates nor earns its keep.

### What this closes, and what it does not

- **The fabrication mechanism is not R.** Recorded as refuted. The 41.5 % of error mass that is invented words remains unexplained. This was my hypothesis, and the data killed it for one afternoon of CPU.
- **D6's residual ablation is answered at inference.** `residual_branch: true` contributes nothing measurable to content fidelity on this checkpoint.
- **It does not authorise removing R.** This ablates R from a model trained with R, and "no measurable difference at +-3 points" is not "useless". Only a trained-without-R arm settles it, and at 197,890 parameters (2.75 % of the model) the prize is small. **Not scheduled.**
- **The 84 % volume-knob finding stands, and is now stronger.** The worry was that it described only the multiplicative path while R did the real work. R carries 8.6 % of the energy and removing it changes nothing, so the mask really is the model.

### The reusable part

`Estimator.residual_scale` and `Estimator.capture_parts`, `--residual-scale` on `make_estimates.py`, `scripts/diagnose_residual.py`, `tests/test_residual_ablation.py` (6 tests). One test feeds the model a silent mixture and asserts a non-zero output: the capability to fabricate is real and now pinned by a test, though the measurement says it is unused.

### MEASURED 2026-09-13 — objective or data? It is the OBJECTIVE. More data makes the mask WORSE, not better

`scripts/measure_mask_flatness.py`, 50 `sir0_val` `both` trials, the same for every checkpoint, 1.6 h CPU, five existing checkpoints.

| checkpoint | trials | epoch | share(loud) | f/t ours |
|---|---|---|---|---|
| `model_sir0.pt` | ~1,989 | 9 | 0.6415 | 0.9561 |
| `model_sir0_5000-e7.pt` | ~4,976 | 7 | 0.7766 | 0.7646 |
| `model_sir0_10000-e6.pt` | ~9,955 | 6 | 0.7958 | 0.6158 |
| `model_sir0_10000-e7.pt` | ~9,955 | 7 | 0.7738 | 0.6026 |
| `model_sir0_10000-last.pt` | ~9,955 | 15 | 0.7739 | **0.4826** |
| **ideal mask** | | | | **1.0136** |

`share` is the fraction of mask variance explained by one number per frame (1.0 is a pure volume knob). `f/t` is variation across frequency relative to across time. **Ours must rise toward 1.0136; falling is worse.**

### Paired bootstrap, 50 trials, 10,000 draws

| comparison | share(loud) | f/t (loud) |
|---|---|---|
| **DATA 5k -> 10k, epoch MATCHED at 7** | -0.0029 [-0.027, +0.022] **inside noise** | **-0.1620 [-0.205, -0.118] REAL** |
| DATA 2k -> 10k | +0.1322 [+0.105, +0.160] REAL | -0.3535 [-0.392, -0.315] REAL |
| **EPOCH 6 -> 15, data FIXED** | -0.0219 [-0.047, +0.001] inside noise | **-0.1332 [-0.187, -0.074] REAL** |

**The epoch-matched data comparison answers D15's successor.** Doubling the training set from ~5k to ~10k trials at the same epoch left the volume-knob share statistically unchanged and made the frequency/time ratio significantly worse.

**More training does the same.** At fixed data, epoch 6 to 15 moved f/t -0.133, also real and also the wrong way.

**Both axes push the model toward the volume knob**, as expected if the objective's optimum is one.

### The trap in this table, stated because it inverts the obvious reading

**The ~2k model has the best-looking numbers and is the worst model.** Its share is lowest (0.64) and its f/t of 0.956 is nearly the ideal's 1.014. It is not more structured: **a noisy, unstructured mask also has f/t ~ 1, because noise varies equally along both axes.** That checkpoint memorised its training set (decisions-m2.md 2026-08-29) and is beaten by pass-through. Training sheds the noise and converges onto the knob.

**Never read this table as "less data is better"**: on this objective, training buys a cleaner volume knob.

### Verdict, against the rule registered BEFORE the run

Three outcomes were registered: flatness constant (objective), flatness falling with data (data), epoch swamping data (inconclusive). **The result is the first on `share`, and stronger on `f/t`: not "data does not help" but "data actively makes it worse"**, which the rule did not anticipate.

**Epoch does not swamp data.** On `share`, data moves 0.132 (2k to 10k) against epoch's 0.022. On `f/t` they are comparable (-0.162 vs -0.133) but point the same way, so the confound reinforces the result.

### What this authorises, and what it does not

- **More data will not fix the flat mask (the 84 % finding). The data hypothesis is refuted**, not merely unsupported.
- **The structure loss is justified by evidence, not by elimination.** Supervise the mean-removed across-frequency deviation against the free oracle mask, with the residual branch handled. R is inert (2026-09-13), so it will not absorb the term.
- **It does not predict the intervention works.** Flatness is a property of the mask, not of what a listener transcribes. This selects the arm; only LCF-WER on `sir0_privval`, per SIR band, scores it.
- **Single seed, n=50, one architecture.** Run-to-run training variance is still unmeasured and sits on top of every interval here.

---

## 2026-09-15 — THE MENU: what is open, what each costs, and what it buys

**A decision menu, not a decision: nothing below is taken.** Every cost is from `run_times.md` or a completed comparable run; none is estimated. Ordered by information per hour, obligations first.

### Obligations — these are not optional and two of them are cheap

| # | what | cost | why it is not optional |
|---|---|---|---|
| **O1** | **Score the struct control on `sir0_privval`.** `2026-09-14-est-privval-control` (1,421 rendered trials, with a `meta.yaml`) has never been through `evaluate.py`. | **1.2 h**, measured | D17's config names LCF-WER on `sir0_privval` **per SIR band** as its acceptance test. The arm has that number (47.37); the control does not, **so the arm is unjudged against its own registered test.** |
| **O2** | **Write D17 up.** The arm ran and was evaluated four ways, but appears in no decision log. | ~1 h, writing | Project rule: every experiment gets a logged result. It is the only unlogged run in the repo. |
| **O3** | **M6's stratified tables (B13) and latency curve (B11).** Scoring only, no training. | unmeasured; `evaluate.py` already emits the columns | M6 is the thesis's central finding and its checklist is 2 of 8. A headline aggregate must never appear alone. |

### D17's result, stated here because the options below branch on it

**In plain words: the arm was told to give the mask frequency detail, and it halved the frequency detail it started with.** The live judge could not tell the two systems apart.

| on `sir0_val` `both`, n=103, same trials | baseline e6 | struct e12 |
|---|---|---|
| LCF-WER (judge) | 55.59 | **55.36** (−0.23, inside the ~3-point paired floor) |
| leakage ICR@2 | 61.17 | 65.05 (worse) |
| fabrication FR@2 | 43.69 | 48.04 (worse) |
| mask variation across frequency, `d_freq` | 0.0253 | **0.0121 — halved** |
| freq/time ratio (ideal 1.014, must RISE) | 0.616 | **0.306 — fell** |

**Flag (headline moving for a bad reason):** LCF-WER is flat and the *direction* is wrong on every diagnostic. Read this as a null with a mechanism, not a tie.

**`train_L_struct` is absent from `2026-09-14-train-sir0-struct/history.csv`**, as the logging fix landed alongside the run. **We cannot say whether the term descended.**

---

### D18. Why L_struct FLATTENED the mask — the degenerate-optimum hypothesis

**Status: proposal. The cheap half needs no training.**

**Hypothesis.** `_loss_mask_shape` is an energy-weighted **L1** between the model's mean-removed frequency shape and the oracle's. When the target is unpredictable from the input, the L1-optimal constant prediction is the **conditional median of the oracle deviation, ~0**, i.e. a flat mask. So a shape loss the model cannot fit **pays it to remove the shape it had**, which is the measured signature.

**Test 1, free, no training.** On cached crops, compute the term for (a) the model's mask, (b) a constant-zero shape, (c) the oracle. **If (b) beats (a), the term's own optimum is the flat mask**, and every "supervise the shape" variant inherits the fault.

**Test 2, one training run.** If Test 1 confirms it, use a loss whose optimum is not zero: correlation or cosine across frequency (scale-free, so a flat prediction scores worst), or a variance-matching penalty. Same weight-derivation procedure.

**Value.** D15's successor and D17 are the same idea, so if the loss form is at fault, the idea has never been tested. If Test 1 refutes it, the shape idea is dead on evidence and M5 should spend its remaining weeks elsewhere.

### D19. Is the volume knob OURS or the TASK'S? Run the flatness instrument on WeSep

**Status: proposal. Cheapest high-information item on this page.**

WeSep `tfmap_context_causal_100` scores **26.40 LCF-WER against our 55.59** on the same 103 trials (floor 63.27, ceiling 1.05). It is causal, out of domain by its own config, and beats us by ~29 points.

**The measurement.** Point `scripts/measure_mask_flatness.py` at WeSep's mask.

- **WeSep's `f/t` near the ideal 1.01:** the volume knob is a property of **our objective**. That would be the most defensible finding in the project, since a published, better-scoring causal model does not do it on the same trials.
- **WeSep's `f/t` also ~0.5:** the knob is what causal band-split masking does, the flatness diagnosis does not explain the 29-point gap, **and D18/D15/D17 are the wrong tree entirely.**

Either answer redirects M5 without a training run. The one obstacle is reaching WeSep's internal mask rather than its waveform; if it is not exposed, the free fallback is the mask implied by output/input magnitude.

**Caveat:** different data, objective and training budget. This diagnoses *our* model; it is not a comparison claim.

### D20. Derive the silence bar from the listener, not from a dB number

**Status: proposal, opened 2026-09-23 (`decisions-m2.md` 2026-09-23).**

**Why.** `select_abs_max: -10.0` was never derived, and dB suppression is not silence to the listener: the control at -11.0 dB on `interferer_only` still hands the ASR 81 % of the other speaker's words.

**The measurement.** For each absent-target estimate, suppression (dB) against the words the listener transcribes, using the speech gate's 0.10 s speech floor. This needs a small per-trial script, as `eval_by_case.py` writes only aggregates. No training run, and no API if the ASR is the listener.

- **A dB level where words reach ~0:** that is the bar, derived.
- **No such level short of the -20 dB floor:** the silence term should target content, not loudness, and `L_abs` is the wrong instrument for it.

### J5. Run-to-run training variance — the third noise source, still unmeasured

**Status: proposal. Uncomfortable, and other claims rest on it.**

**Ten training runs exist, all `seed: 42`.** No same-config replicate has been trained. Two noise sources are measured, trial sampling (~3 points paired, `decisions-m3.md` 2026-09-12) and judge SEM (~0.5), but **the third, training itself, is blank.**

So the state arm's −1.72 and D17's −0.23 were compared against an error bar nobody has drawn. Both were correctly written up as "did not improve" rather than "lost", but **M6 cannot claim two systems differ without this.**

**The measurement.** Re-train `bsrnn_baseline.yaml` at two more seeds and score all three on the same 103 trials. Cost: 2 training runs plus 2 judge passes (14 min each, measured). **Do E8 first**; it roughly halves the training part.

**Register before running:** if the seed spread exceeds ~3 points, no single-seed intervention here is separable from noise, and the M6 write-up becomes a spread-versus-spread comparison, not a point one.

### Already-open items, re-costed against the above

| id | one line | cost | verdict today |
|---|---|---|---|
| **E8 + E7** | fp32 batch probe writes batch 3 into an fp16 run that fits 6; then the second T4 is idle | ~half a day, ~2x throughput | **Do this first if anything in J5 or D13 runs.** It multiplies every remaining training hour. |
| **D13** | the per-band mix-back gate: **M5's actual scoped deliverable, and it is not built** | one build + one run | Oracle ceiling 2.2 points against a ~3-point paired floor. **Declare the effect size and k before running**, or the null is uninterpretable (M6 already says this). |
| **D4a** | `bsrnn_tfmap_inject.yaml`, built 2026-09-11, **never run** | one training run | A finished arm sitting unused. Cheapest untested modelling change on the page. |
| **D2** | attention temperature, one line of code | an afternoon | D7 records it as run but never logged as an arm. Log it or re-run it as one. |
| **D14** | state teacher | done | Measured and written up. No further arm without a reason. |
| **D1** | phoneme-template cue | M5-scale | **Do not start.** Gated on D2, and M5 is cuttable with ~4 weeks left. |

### The sequencing this implies

1. **O1** (1.2 h): finishes D17 against its own test.
2. **D19** + **D18 Test 1**: both training-free, both redirect M5.
3. **O2**, then **E8/E7**.
4. Then **one** of D13 or J5, chosen by what 2 and 3 say. **Not both: there is not room before 14 October**, and M6 must still run.

**The last three arms.** The state teacher moved its own term and lost the metric; D17 moved its own diagnostic the wrong way and drew; the data hypothesis was refuted outright. **D19 is listed because it alone asks whether the diagnosis itself is right**, at no training cost.

---

## 2026-09-15 — G1: THE GEMINI TUNER. A local stand-in for the judge

**Status: PROPOSAL, raised 2026-09-15 (supervisor). Nothing below is taken.** Unblocked by the same day's withdrawal of family separation (decisions-m4.md 2026-09-15): Gemini may now supply targets, rewards and filters offline.

**The ask, as given.** Build a local model (expected to be large) that learns what Gemini Live would predict: the word error a mixture would produce given its target and interferer. Regress the extractor against it.

### G1a vs G1b vs G1c — three different systems, and only two can train anything

| | input | output | trains the extractor? | what it is actually for |
|---|---|---|---|---|
| **G1a** difficulty model | mixture + enrolment | predicted LCF-WER | **NO** | curriculum, data curation, analysis |
| **G1b** scorer / reward | **extractor output** + target script | predicted LCF-WER | yes — 1 scalar per clip | cheap offline metric |
| **G1c** distilled judge | **extractor output** | **Gemini's transcript** | yes — ~100 tokens per clip | the training signal itself |

**Raise this at the next meeting before anything is built: the ask as worded is G1a, and G1a cannot tune the extractor.** The extractor does not change the mixture, so a function of the mixture has zero gradient with respect to its weights. Predicting difficulty is useful for picking training trials, but a loss must be fed **what the extractor produced**.

**Recommended shape: G1c, which subsumes the other two.** A distilled judge run on a mixture is G1a; scoring its transcript with the existing `lcf_wer.py` / `icr.py` / `fabrication.py` gives G1b for free.

### Why a transcript target and not a WER target

**A scalar is too weak a gradient:** one number per ~19.6 s clip against ~100 reference words. It needs many labels, gives the extractor a single direction, and is the classic reward-hacking target: the extractor finds audio the scorer likes and the judge does not, and the scorer, off its own training distribution, cannot tell.

**A transcript supervises per token, on what we actually want.** With a distilled judge `G`, the extractor loss is the cross-entropy of the **true target script** under `G` given the extractor's output ("make audio a Gemini-like listener transcribes correctly"). That is ASR cross-entropy, already an allowed proxy in CLAUDE.md; the only novelty is an ASR aligned to the judge instead of a generic Whisper, so it extends a sanctioned path rather than being a new invention.

### The measured facts that constrain the design

**All from disk. None estimated.**

1. **A free local listener already explains 68 % of the judge's per-trial word error.** `experiments/results/2026-09-15-judge-predictability/`, n=412 (4 systems x 103 trials): pooled Pearson r 0.825, r² 0.680, Spearman 0.781. **The surrogate starts at r² 0.68, not zero, and that is the bar.** The free proxy gets the level wrong: mean absolute error **21.7 points**.
2. **Agreement is worst on the best system.** WeSep (cleanest audio, lowest word error) falls to r² 0.453, while the three ~62 % systems sit at 0.70–0.72. **The free proxy degrades where an improving extractor is heading.** That argues for training one, but a surrogate trained on today's mediocre outputs will be weakest on tomorrow's better ones.
3. **The offset is not constant.** The ASR reads 3.9–4.9 points higher than the judge on both baselines and 7.4 higher on WeSep, but **1.1 points lower** on struct-e12. One calibration constant cannot fix it.
4. **The judge's own test-retest noise is unmeasured, and it bounds everything.** Three cached clips have repeats (5 calls each); one moved **16.0 points**. If the judge's self-disagreement is ~15 points MAE, the surrogate competes for ~6 points of headroom against the free proxy's 21.7. **Nobody knows this number, and it costs ~$0.40 to get.**
5. **Labels are cheap in money, expensive in wall-clock.** ~$0.0013 per clip (~53 cents over ~412 clips, project-state.md) at **10 requests/minute** (103 clips in 11–14 min, measured, `run_times.md`). So 5,000 labels ≈ **$6.50 and 8.3 h**; 20,000 ≈ **$26 and 33 h**. **Check whether the prepay tier lifts 10 rpm; that one fact sets the schedule.**
6. **2,636 estimate clips are already rendered on `sir0_val` across 19 system variants**, 412 of them judged. The other **2,224 cost ~$2.90 and ~3.7 h** to label, with no inference needed: the mask post-processing sweeps (hysteresis, floor, follow), state-e6, noresidual, struct-e12, WeSep and both baselines are 19 distinct points in artefact space on the same 103 trials.
7. **There is no local GPU** (`run_times.md`). Training is on a Kaggle T4 under a 12 h session cap. **"Quite large, trained locally" is not available.** The realistic route is to inherit size from a pretrained checkpoint, fine-tuning `whisper small.en` (already a dependency) or a WavLM/wav2vec2 encoder on the Gemini labels. "Learn what Gemini would say" then becomes domain adaptation with a few thousand labels.
8. **The judge scores ~19.6 s clips; training uses 4.008 s crops.** A surrogate trained on clips is out of distribution on every training step. **Buy training labels on 4 s crops** (same price per call), and keep clip labels only to validate that the surrogate predicts the real metric.

### Staged plan, each stage a decision point and each one a result on its own

| stage | what | cost | kills the project if |
|---|---|---|---|
| **0** | predictability floor — DONE, free | 0 | — (r² 0.680, MAE 21.7) |
| **1** | **judge test-retest noise floor**: k=5 repeats on ~100 clips spanning easy→hard | **~$0.65, ~1 h** | self-MAE ≈ 20 pts: there is nothing left to learn and G1 stops here |
| **2** | label the 2,224 rendered clips + a synthetic corruption ladder | **~$3, ~4 h** | — |
| **3** | fine-tune `small.en` on (audio → Gemini transcript); score r²/MAE **on a held-out SYSTEM** | 1 T4 session | held-out-system r² ≤ 0.68: it learned the systems, not the judge |
| **4** | CE of the true script under the distilled judge as an extractor loss term | 1 training run | — |
| **5** | trust region: re-judge 103 clips every N epochs, plot predicted vs actual | $0.13, 14 min each | divergence ⇒ buy labels on the new outputs, retrain, continue |

**Stage 1 before Stage 2.** It is the cheapest item on this page and the only one that can say the project is not worth doing.

### Where the training distribution comes from, and why it is the hard part

**Distribution shift, not accuracy, kills reward models**: the surrogate learns from today's extractor, then scores one that has been optimising against it. Three sources, in increasing order of value:

1. **The 19 rendered variants** (fact 6): real artefacts, on disk, but only 103 distinct trials.
2. **Synthetic corruption ladders**: apply to the clean target mix-back at a range of α, band drops, spectral holes, musical noise, level scaling and clipping. **Free, unlimited trials, covering artefact space deliberately rather than by accident.**
3. **Online re-labelling** (Stage 5): the only source that tracks the extractor as it moves.

**Trial diversity must come from `sir0_train`, never from `sir0_privval` or `eval_private`.** Those two are the entire remaining holdout (decisions-m4.md 2026-09-15), and labelling them destroys the last defensible number.

### Registered before running

- **Declare the effect size.** M6 requires it and the last three arms ignored it. State what LCF-WER gain counts as success before Stage 4, against the ~3-point paired floor and the still-unmeasured seed spread (J5).
- **Test the surrogate on a held-out system**, not a held-out trial: in use it must score a checkpoint it has never seen.
- **Every Gemini call in Stages 1–5 is a training-time call** and records model ID, prompt and date exactly as a judge call does. `judge.py` already does this, so no new logging is needed.
- **Every claim says *optimised for Gemini*, never *generalises to live models*.**

### What it costs against what is already on the page

**This does not fit alongside the 2026-09-15 menu.** Stages 0–3 take roughly a week and Stages 4–5 a training run plus iteration, against **4 weeks to the 2026-10-14 freeze** with O1/O2/O3 outstanding and M6 unrun. Taking G1 means dropping D13 and J5 (M5's scoped deliverable and the seed-variance measurement). **That is a supervisor decision, not one to take here.**

**For:** the last three arms (state teacher, D17, the data hypothesis) all returned nulls, and G1 is a result either way. "How well can a local model predict a live model's listening, and does regressing on it help?" is publishable as a negative. **Against:** it puts the thesis's stated primary contribution (a gaming-resistant metric) and its replacement (the surrogate) on the same four weeks.

### G1c concretised — exact inputs, outputs, and where the term goes

**Raised 2026-09-15 (Grant): "backbone + a head fine-tuned by Gemini — but what are the input and output, and how does it reach the extractor?"**

#### The surrogate has two lives, and the input/output differ between them

**Life 1: training the surrogate.** Fixed audio; the label is what Gemini said.

| | |
|---|---|
| input | any 4.008 s audio crop, `(B, 64128)` at 16 kHz |
| label | **Gemini's transcript of that same crop**: its errors, its leakage, its inventions |
| trained | the head only; backbone frozen |
| learns | *how this listener mishears* |

**Life 2: training the extractor.** The surrogate is frozen; the audio moves.

| | |
|---|---|
| input | `s_output`, the extractor's waveform, `(B, 64128)`, with gradient |
| target | the clean stem `s_target`, already in the loss call |
| trained | the extractor; surrogate weights frozen |
| asks | *now do not mishear*, and pushes that back into the audio |

The surrogate learns to mishear like Gemini; the extractor then makes audio that defeats the mishearing. The labels are Gemini transcripts; the extractor's target never is.

#### The backbone: `torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H`

**No new dependency: torchaudio 2.11.0 is installed and ships it.** It is 16 kHz native, matching `sample_rate`, and has a trained CTC head over a **29-symbol character vocabulary** (`'-', '|', A-Z, '`), so only the head is fine-tuned onto Gemini behaviour.

**Do not use Whisper for the in-loop term.** Its encoder takes a fixed 30 s window, so every 4.008 s crop would cost a 30 s forward pass, ~7.5x wasted compute per step. `faster-whisper` is worse: it is CTranslate2, an inference engine with no autograd. Whisper stays the offline stand-in listener.

Shapes, for the 4.008 s crop (conv stack strides 5,2,2,2,2,2,2 = 320, i.e. 50 Hz):

    waveform      (B, 64128)        B = 6 at batch_size 3 x both_directions
    features      (B, 200, 768)
    CTC logits    (B, 200, 29)

#### Two heads on the one frozen backbone

| head | output | trained against | used for |
|---|---|---|---|
| **T** transcription | `(B, 200, 29)` CTC logits | CTC vs **Gemini's transcript** | **the gradient path** |
| **S** score | one scalar | Huber vs measured per-trial LCF-WER | reading only, never in the gradient |

**Head S is the instrument, not the trainer.** It checks whether the surrogate predicts the judge (against the r^2 0.680 / MAE 21.7 floor already measured) and gives free checkpoint selection. It must never be backpropagated: a per-clip scalar is the textbook reward-hacking target.

#### The extractor's loss term, and why it needs no text alignment

**Option 1, ship first: "transcribe like the clean stem does".**

    L_gem = D( HeadT(G(s_output)) , HeadT(G(s_target)) )

`D` = KL over the frame-wise character posteriors, or L1 on the logits. **No ground-truth text is needed**, because the reference is the clean stem, which `losses.py` already receives. This matters: the loader yields 4 s crops with no text, so anything text-based first needs per-crop alignment.

**Option 2, later: "get the words right".**

    L_gem = CTC( HeadT(G(s_output)) , words spoken inside this crop )

More tolerant (any audio that transcribes correctly scores well, however it sounds), but it needs the crop's text. That is buildable: the loader returns `crop_start` and `trial_id`, and the manifest has `target_utts` and `target_onsets_s`. A middle route skips alignment: greedily decode `HeadT(G(s_target))` and use that as the crop's pseudo-label.

#### Where it goes: one term beside `L_MR`, taking the same two tensors

`scripts/train.py:723-731` already computes what is needed:

    s_output, mask = model(mixture, enrollment, return_mask=True)
    loss, parts = loss_fn(target, s_output.float(), mixture, crop_absent, ...)

So `L_gem` enters `LossBSRNN.__call__` in the **present branch**, next to `L_MR`, reading the same `s_target` / `s_output` pair and gated the same way. An absent crop has no clean stem to listen to, and `L_abs` already owns that case.

**The defensible framing.** `L_MR` already compares output to target in a feature space, which happens to be four STFT magnitudes. **`L_gem` is `L_MR` with a learned, Gemini-aligned feature space instead of a hand-chosen one.** One term, one weight; `w_gem = 0` reproduces today's objective exactly, and the weight is derived like `w_g` and `w_struct` (`derive_w_g.py`), never chosen by hand.

#### The extractor does not change, and that protects the latency result

**The surrogate is training-time only.** At inference the shipped model is the same 7.19 M parameters at RTF 0.528 and 162 ms, so the streaming and latency story is unchanged. This is the strongest practical argument for it over changing the architecture.

#### Registered before building — the things that will actually bite

1. **Step-time and memory cost are unmeasured.** A base backbone runs forward and backward every step (frozen weights still pass gradient to the input), against 0.674 s/step now at batch 3. **Measure with `scripts/profile_step.py` before committing**, and do not quote a projection as a measurement. If the T4's memory will not take it, use gradient checkpointing on the backbone; E5 already records activations as the binding constraint.
2. **The front end must be torch ops end to end**, or the gradient stops. The torchaudio bundle satisfies this; `faster-whisper` does not.
3. **Per-clip waveform normalisation in the backbone destroys level information.** That is wanted (`L_gem` cannot be gamed by getting louder), but `L_gain` must stay in the objective to cover level.
4. **Train the surrogate on 4 s crops**, the length it sees in the loop, not 19.6 s clips.
5. **Cache backbone features when training the head.** In Life 1 the audio is fixed, so features are computed once and head training takes minutes. In Life 2 the audio changes every step and nothing can be cached.
6. **Validate Head S on a held-out system, not a held-out trial.**

### G1 REFRAMED — 2026-09-16. A surrogate answers scarcity, and scarcity may not bind

**Raised by Grant: "how do I tune to a transcription tool that has its own biases and flaws, and how do I do it with Gemini if tokens are not scarce?"**

**Correction to G1 as proposed.** A differentiable surrogate exists for one reason: you cannot backpropagate through an API. That constrains *gradients*, not *tuning*. **A black-box listener's signal can reach a system by four routes, and only the last needs a surrogate.**

| level | what moves | needs the judge differentiable? | hacking risk |
|---|---|---|---|
| **1 selection** | which artefact you keep (epoch, config, trials) | no | none — no pressure on the audio |
| **2 parameter search** | non-learned knobs, scored directly by the judge | no | low — few parameters |
| **3 imitation of winners** | **the weights** — best-of-N, then retrain toward the winners | **no** | moderate |
| **4 differentiable surrogate** | the weights, through backprop | yes | high |

**Level 3 was missed.** Generate N candidates per trial, let Gemini rank them, and fine-tune the extractor toward the winning audio with the existing loss. The judge's preferences reach the weights without Gemini being differentiated. This is expert iteration (rejection-sampling fine-tuning). **G1's surrogate drops from "necessary" to "last resort".**

#### Tuning to a biased listener — the two things that make it defensible

1. **Keep an unoptimised second listener as the control.** `small.en` is untouched by training. If tuning to Gemini also moves Whisper, intelligibility improved; if not, we fitted Gemini's quirks. **Both are reportable, and this partly recovers what the 2026-09-15 withdrawal cost**: the holdout moved from the model to the data, and can also move to the listener. The baseline for that check exists: r^2 0.680, MAE 21.7 (`2026-09-15-judge-predictability`).
2. **Separate systematic bias from stochastic noise.** Bias is tunable; noise is not. One clip moved **16.0 points across five identical calls**, so a k=1 label is partly noise. **With a free budget, buy repeats, not more trials.** Aggregates already have SEM ~0.5 over 103, but every method above uses per-trial labels, which are the noisy ones.

#### What "no shortage of tokens" does and does not unlock

- **Does:** mass labelling, k>=3 on everything, best-of-N at large N.
- **Does not:** put Gemini inside the gradient loop. A call takes ~8 s against a 0.674 s training step; that is latency, not cost, and no budget removes it.
- **Unverified, and it sets the plan.** `judge_gate.yaml` records `tier: "free"`, 10 rpm / 1500 rpd, both annotated *"VERIFY in AI Studio -- third-party figure"*, while `project-state.md` says AI Studio **prepay**. At 10 rpm, 100k labels take 167 h; at 1000 rpm, 100 minutes. **Check the console, and whether batch mode exists for this model; it is built for this and usually runs far above the interactive limit.**

#### The experiment to run first: the alpha-oracle

**It attacks the project's biggest measured failure with Gemini as the teacher, needing no retraining for the diagnostic and no surrogate.**

The #1 finding against our model is that **it applies one transform to everything**: SIR/SAR is flat from easy to hard while the outcome swings −4.2 to +23.1 points (decisions-m3.md 2026-09-01). D11's mix-back is the knob: `s_alpha = alpha * s_hat + (1 - alpha) * x`, with **zero added latency, no retraining, and every alpha from one forward pass.** It is not yet built (`postprocess_mask.py` does hysteresis sharpening, not mix-back), but it is a multiply-add.

1. Sweep `alpha` over ~5 values on N trials (one forward pass, trivial DSP).
2. Score every `(trial, alpha)` through the judge at **k>=3**.
3. Take `alpha*(trial)`, the per-trial best.

**Reading it:**

- **`alpha*` roughly constant:** nothing trial-dependent to learn. Ship the global knob; "tuned to Gemini" is then one number, cheaply obtained.
- **`alpha*` varies with the trial:** **Gemini has handed us a supervised target for a per-frame gate.** That is D11 strategy 1 (learned, input-conditioned alpha), it directly attacks "the model cannot tell the cases apart", and the labels come from the judge.

**This is not the patch D11's objection rejected.** That objection is against a global constant presented as a fix, which trades hard-trial gains for easy-trial harm and adds no capability. Here the sweep is (a) a probe for what the judge wants and (b) a source of training targets for the principled per-frame version. Report it that way, or the objection applies again.

#### Revised order

1. **Verify the rate limit and batch availability.** Minutes. Gates everything.
2. **Judge test-retest noise, k=5 on ~100 clips.** Still the gate: if the judge disagrees with itself by ~20 points, there is nothing to tune to.
3. **The alpha-oracle sweep.** Either outcome is a result.
4. **Learned per-frame alpha head**, if step 3 says alpha is trial-dependent.
5. **Expert iteration** (best-of-N over alpha x hysteresis x checkpoints, retrain on winners), if time remains.
6. **G1c's surrogate last**, and only if a signal is needed inside the gradient.

**Carry to every claim: *optimised for Gemini*, never *generalises to live models*. The Whisper control says which of the two we achieved.**

### J6 — 2026-09-16. Tuning the LISTENER: a second instrument, never a parameter

**Raised by Grant: "can the Gemini side be tuned to best account for my system?"**

**Risk.** LCF-WER is *what the live model reports*, so the live model is the transcriber, and no separate frozen scorer exists to grade a tuned receiver. **Tuning the listener until it is kinder to our audio, then reporting one number, games the benchmark more completely than anything the 2026-09-15 withdrawal allowed.** That withdrawal kept the data held out; this would remove the project's last fixed reference point.

**Resolution, at no cost.** A listener configuration is an **instrument, not a parameter**. Changing it creates a second instrument, on which **every system is re-scored, floor and ceiling included**, and comparing the two tables is itself a finding. `metric-definitions.md` already treats the prompt, the normaliser and the gate this way; decoding configuration joins that list.

#### DEFECT, and it must be fixed before any of this is attempted

**The cache key does not include the generation config.**

    f"{model_id}@{backend}|{prompt_sha}|{trial}|{file}|{audio_sha}|r{repeat}"

Model ID and prompt hash are in it, so a prompt change correctly forces anchors to be re-bought. **A temperature change does not: the cache silently serves answers produced at the old setting.** Fix it backward-compatibly (fold the config into the key only when non-default), or all 653 paid-for responses are invalidated at once.

#### The instrument is currently running on server defaults

`Judge._call_once` sets **no temperature, seed, top_p or thinking budget**; the server picks everything.

**This is the likeliest source of the 16.0-point spread** across five identical calls on one ambiguous clip (2.9 on another, 0.0 on the ceiling). That spread blocks per-trial claims (project-state.md "Cannot": *per-trial needs k>=3 and averaging*).

| knob | now | expected effect | verdict |
|---|---|---|---|
| **temperature / seed** | **unset, server default** | likely large; the noise source | **a REPAIR, not tuning** |
| **k-sample consensus** (vote over k transcripts) | k=1 on estimates | attacks per-trial noise directly | legitimate as a second instrument |
| **thinking budget** | unset | unknown; plausibly large on degraded audio | legitimate as a second instrument |
| prompt wording | frozen `d118b7d3bf30` | **MEASURED LOW**: cross-prompt range 18.0 against same-prompt noise 16.0 | answered; do not spend time here |
| enrolment audio as context | no | probably large | **FORBIDDEN by J3**: hands the extractor's job to the judge |
| fine-tuning Gemini on our outputs | no | large | **destroys the premise**: a tuned Gemini is no longer the off-the-shelf live model the project claims to serve |

**Pinning temperature is a repair, not tuning**, because it applies identically to every system and reduces variance rather than shifting the mean in our favour. **Verify this, do not assume it:** re-score floor and ceiling first and report the mean shift alongside the variance drop. If the mean moves materially, it is a new instrument and every number moves with it.

#### The version that is a contribution rather than a problem

The project is a **pipeline**: the extractor feeds the live model. In deployment both ends get tuned, and **"where is the headroom, the front end or the receiver's configuration?" is a question nobody in TSE asks.**

Run it as systems x listener configurations, each cell re-scoring its own floor and ceiling. The ceiling is 1.05 % on clean audio, so listener tuning can buy almost nothing there. **The question is whether a better-configured listener is more robust to extractor artefacts**, which is `metric-definitions.md` 1's hypothesis from the other side.

**If a listener configuration closes more of the floor-to-ceiling gap than our extractor does, that is a finding about where effort belongs in live-model pipelines.** Uncomfortable, but cheap (scoring, no training).

#### Order

1. **Fix the cache key, backward-compatibly.** Nothing else is safe until then.
2. **Pin temperature and seed, re-score anchors, and report mean shift and variance.**
3. **k-sample consensus as a declared second instrument.**
4. Do not tune the prompt, send the enrolment, or fine-tune Gemini.

### 2026-09-16 — THE RECOMMENDED SEQUENCE. A recommendation, not a decision

**Four weeks to the 2026-10-14 freeze.** This ties together G1, G1c, the G1 reframe and J6. **Dropping D13 is a real decision and is not taken here.**

**`project-state.md` "Next" item 1 already names the mix-back sweep as the cheap route to M6's missing near-tie.** Free tokens promote that sweep from a *measuring instrument* to a *teacher*, because every `(trial, alpha)` cell can now be scored densely enough to read `alpha*` per trial.

#### Everything in this conversation bottlenecks on ONE number

Per-trial judge noise blocks per-trial claims (project-state "Cannot"), a surrogate (noise cannot be fitted), `alpha*(trial)`, best-of-N ranking and every delta's error bar. **It is unmeasured at n>=3 clips, and it may be a config line**, since J6 found the judge on server-default temperature. **Quieten the instrument first, and everything downstream gets cheaper.**

#### Tier 1 — this week. Cheap, unblocking, none of it optional

| | what | cost |
|---|---|---|
| 1 | verify the real rate limit + batch availability (J6, G1 reframe) | minutes |
| 2 | fix the judge cache key backward-compatibly (J6 defect) | small |
| 3 | pin temperature/seed; re-score anchors; report **mean shift AND variance** | ~600 calls |
| 4 | judge test-retest, k=5 on ~100 clips — the noise floor | ~$0.65 |
| 5 | **O1** (struct control on `sir0_privval`) and **O2** (write D17 up) | 1.2 h + 1 h |

#### Tier 2 — the one new experiment: the alpha-oracle sweep

**It beats every other candidate by doing three jobs at once:**

1. attacks the top measured failure, one transform applied to everything;
2. **produces M6's missing near-tie family** from one checkpoint, with no retraining;
3. if `alpha*` is trial-dependent, Gemini has handed us a supervised target for the per-frame gate (D11 strategy 1).

**Every member of the alpha family shares one set of weights.** So within-family comparisons carry no training noise, and **J5's unmeasured seed variance cannot confound them**, unlike every delta reported so far or any retraining arm.

#### Tier 3 — conditional. Learned per-frame alpha head, one training run

Only if Tier 2 says `alpha*` varies by trial. It then replaces D13 as M5's deliverable, better motivated because the target comes from the judge, not an oracle mask.

#### Tier 4 — in parallel, and it is the part that cannot be cut

M6's stratified tables (**O3**) and the five existing, unwritten findings. **The thesis risk is not a weak extractor (itself an evidenced result) but that M6, the stated central finding, never runs.**

#### Recommended DROP, each with its reason

| item | why |
|---|---|
| **G1c surrogate** | answers scarcity; scarcity may not bind. Last resort, not first |
| **D13 per-band gate** | Tier 2 delivers M6's near-tie family more cheaply and with no training noise |
| **D18 / D19** | diagnostics of the mask-shape line, which Tier 2 routes around |
| **J5 seed variance** | deferrable *only because* Tier 2 is retraining-free. Required again the moment Tier 3 runs |

**Carry regardless: *optimised for Gemini*, never *generalises to live models*, with `small.en` untouched as the control listener that tells the two apart.**

### AMENDMENT 2026-09-16 — the four routes are not four ways to do the SAME thing

**This corrects the G1 reframe above**, which said the surrogate "drops from necessary to last resort". That is right about sequence but misleading about sufficiency.

**The four routes differ in how much of the system the judge shapes.**

| route | what Gemini decides | parameters Gemini shapes | are the extractor's weights tuned? |
|---|---|---|---|
| selection | which artefact is kept | 0, a choice among N | **no** |
| alpha sweep, global | one scalar | **1** | **no** |
| alpha sweep -> learned per-frame head | the head's training target | ~thousands | partially |
| **expert iteration** | which of N candidate outputs is best | **all 7.19 M** | **yes, within the span of the candidate set** |
| **differentiable surrogate** | the gradient itself, every step | **all 7.19 M** | **yes, including directions no candidate set contains** |

**So the alpha sweep is barely "tuning to Gemini"**: it is one knob whose value Gemini picks. It was recommended because it is cheap, attacks the top measured failure and gives M6 its near-tie family. **Those are reasons of project risk, not of fidelity to the stated goal.** Say so when presenting it.

**Expert iteration's ceiling.** Candidates from mix-back and mask post-processing are all variants of one separation, so retraining on the winners only teaches the model post-processing it could already apply. **It cannot discover a separation outside the candidate set.** Widening the set (stochastic masks, several checkpoints, several architectures) raises the ceiling but does not remove it.

**This is why the surrogate exists, and why the supervisor proposed it.** A gradient explores continuously, so it can reach outputs no candidate set contains. **If the goal is "tune the extractor to Gemini", the surrogate is the only route that fully does it.**

#### The middle path not previously named: surrogate as RANKER, never as gradient

Train G1c's **Head S only** (scalar LCF-WER, the cheap half), use it to rank thousands of candidates locally at zero API cost, keep and retrain on the winners, and **verify the winners with real judge calls**.

- Gemini labels train the ranker; the ranker supplies candidate throughput the rate limit could never buy.
- **A wrong ranker is far safer here than in a gradient loop**: it only re-orders a fixed candidate set, and every winner is checkable against the real judge.
- It needs no differentiable front end, no backward pass through a backbone, and no training step-time cost.

**This is the best fidelity-per-risk point on the page**, and was missing from the sequence above.

---

## 2026-09-21 — O4, OBLIGATION: we ARE the challenge baseline, and we reproduced its consensus independently

**Report this. Our architecture is the SLT 2026 REAL-TSE Challenge baseline `BSRNN_TFMAP_CAUSAL`, and our three "negative" M5 results are that challenge's published consensus, reached without knowing it.** Writing them up as an independent reproduction, not project-specific failures, is a stronger claim at no cost.

### What we found on our own, and what the challenge found

| our result | date | the consensus it reproduces |
|---|---|---|
| 14.73 M bought **exactly zero** separation gain (-2.9020 vs -2.9000) | 09-21 | top teams used *nearly the same extractor* and won on data, recipe and post-processing, not on size |
| data scaling +0.32 dB/doubling with **no downstream conversion** | 09-04 | careful data simulation and real-data adaptation are the reported levers |
| SI-SDR moves the wrong way against content fidelity | 09-04, 09-12 | non-intrusive metrics were **gamed** mid-challenge and the organisers now recommend banning them during development |

**The third row matters most for the thesis.** The organisers changed their official perceptual metric mid-challenge after a team inflated SpkSim and DNSMOS with adversarial waveform perturbations without improving extraction. That is dated, attributable, first-hand evidence that a gaming-resistant content metric, which this project exists to build, is a real contribution. Cite it in the metric chapter, not as a footnote.

### VERIFIED LOCALLY on 2026-09-21 -- safe to report

`../wesep_pretrained/tfmap_context_causal_100/`, read directly:

- **Its speaker cue is `tfmap` + `context` (embed_dim 512, `fusion: multiply`), fed by a pretrained `ECAPA_TDNN_GLOB_c512`.** `spkemb` and `usef` are both disabled. **`src/models/conditioning.py` has only `TFMap` and `TFMapInjector`: no encoder, no contextual embedding, no identity path.**
- **Parameters: `sep_model` 18.868 M + `spk_ft` 14.597 M = 33.465 M**, against our 7.19 M. The separator *hyperparameters* match ours (`causal: true`, `feature_dim: 128`, `num_repeat: 6`, `win: 512`, `stride: 128`), but the realised separator is 2.7x ours, so **this is not a controlled ablation.**
- **It trained clean:** `data/clean/train-100`, `noise_prob: 0`, `noise_enroll_prob: 0`. No noise, no reverb, no low-SIR curriculum.

### The comparison has FOUR confounds. State all of them or state none

Ours (88.2 WER) vs WeSep (49.4) at SIR < -5 differs in capacity (4.7x total), causality (WeSep is **not streaming**: global normalisation, ~6 orders of magnitude of future leakage, `decisions-m3.md` 2026-09-03), training data (clean vs our noise+reverb) and conditioning. **That comparison alone cannot show the cue is the bottleneck, and must never be written as if it does.**

The narrower evidence for the cue hypothesis stands on its own: (a) WeSep's *own* published cue ablation, which varies only the cue with backbone, data and schedule fixed; (b) our direct measurement of our cue (78 % rank-1, `corr(alpha_t, mixture loudness) = 0.990`, no negative evidence anywhere in the path); (c) the injector's per-band `LayerNorm(bw)` deleting `alpha_t`, which mechanically explains D4a's near-zero gates.

### NOT YET VERIFIED -- do not put these in the report until checked

From a literature search on 2026-09-21, not read first-hand: the challenge overview (arXiv 2607.15198) and its verbatim conclusion; the baseline TER 0.808 on EVAL-2; the WeSep cue-ablation table (6.87 / 12.98 / 14.15 dB causal, accuracy 79.9 to 95.9 %); the gaming incident's details. **Read the two papers before citing any of it.**

### The standing rule still applies

Different data, metric and protocol: nothing of ours is comparable to a published REAL-TSE number. The licensed claim is "we independently reproduced their conclusion", never "we match their results".

## 2026-09-29 — IDEA, not decided: smart glasses as the deployment to compare against

Raised by Grant. **Not in scope** (CLAUDE.md: server-class compute assumed, on-device out of scope). Kept for the discussion/future-work section and as a comparison point for the introduction.

### Meta "Conversation Focus" (Ray-Ban Meta / Oakley Meta HSTN, Dec 2025) vs ours

| | Conversation Focus | Our system (e21) |
|---|---|---|
| Clue | direction: face the person, ≤ 6 ft | voice sample (enrolment) |
| Listener | the wearer's ears (open-ear speakers) | a live AI model (Gemini) |
| Delay need | near-zero: played alongside the direct sound | ~175 ms is fine for a live model |
| Target absent | not documented | falls silent on 46 % of stranger-only clips (`eval_public`, 2026-09-29) |
| Hardware | Snapdragon AR1 Gen 1; AI mostly on the paired phone | 7.28 M per chunk; ECAPA (20.8 M) only at enrolment, precomputable |
| Evidence | product docs only; no published metric | LCF-WER / ICR / FR on constructed two-speaker trials |

Sources: Meta newsroom 2025-12 (about.fb.com/news/2025/12/updates-to-meta-ai-glasses-conversation-focus-spotify-integration/), Meta help (meta.com/help/ai-glasses/1660846264628288/), Qualcomm AR1 Gen 1 page. **Unverified:** AR1+ Gen 1 running Llama 3.2 1B on-glass (third-party blog; check Qualcomm's own announcement before citing).

### What it would take to test, if ever taken up

1. Precompute the enrolment spectrogram as well as the ECAPA embedding (`bsrnn.py:195` re-runs `self.stft(enrollment)` every chunk today).
2. Export 8-bit and time it on a phone CPU/NPU. The LSTMs may not map well to an NPU (TF-MLPNet, Itani et al. 2025, replaced sequential RNNs for that reason).
3. Measure compute (MACs) and power, never yet measured in this project.

### Wording limits

- Never "runs on glasses": nothing is measured on that hardware.
- Conversation Focus is a product, not a benchmark: compare purpose and clue type, never quality.

## 2026-10-03 — CLOSED 2026-10-03: the hail-mary run's loss weights (w_interf, w_artif)

**Taken (Grant):** w_interf 2.0 only, w_artif dropped; config `bsrnn_interf_scratch.yaml`. Evidence: decisions-m2.md 2026-10-03 "AB-SDR RESULT". The text below is the open question as it stood.

**On hold until the 2x2 is judged.** `bsrnn_interf_artif_scratch.yaml` carries w_interf 2.0 / w_artif 4.0 provisionally. Do not launch it before this closes.

- **The 2 : 4 ratio was never chosen.** 4.0 is Ochiai et al.'s alpha = 2 (tuned on a conventional ASR backend); 2.0 is D10's. As energy weights, the distortion push is ~6x the leakage push (`interf_artif_ft` epoch 1, train: 0.455 x 3 = +137 % of the error vs 0.239 x 1 = +24 %).
- **Evidence so far, epochs 1–2 (rough, from epoch averages):** w_artif 4 alone lets ~59 % more of the other speaker through than the control and leans less on the enrolment; with w_interf 2 added, ~+30 %.
- **Decide on:** the judge's e5 (and e3) results for `artif_ft` and `interf_artif_ft` against the control and `interf_ft` (rule: decisions-m2.md 2026-10-03). If `interf_artif_ft` still leaks more than the control on the judge, lower w_artif (e.g. 2.0) for the hail mary.
- **Write the choice** into decisions-m2.md's hail-mary entry, and the config, before launch.

## 2026-10-14 — OPEN: the presence gate saturates on e21's features; beta and lam measured on Kaggle

**Found while preparing the gradient-share measurement** (`scripts/derive_presence_weights.py`, docs/extra/variational-presence-gate.md eq. 12):

| | mean $\lVert h_t \rVert$ | gate logit $w^\top h_t + b$ | frames with $\pi_t$ exactly 0 or 1 |
|---|---|---|---|
| random init | 2.2 | +0.11 to +0.36 | 0 % |
| e21 | 494–760 | +9.8 to +56.3 | 51–100 % |

Two `sir0_train` crops (0.5 s), the default `nn.Conv1d` init, CPU, 2026-10-14.

- **A gate on e21's features starts dead.** $\pi_t = 1$ passes no gradient: the KL's extractor gradient vanishes and the BCE's is inflated. Any fine-tune of the gate from e21 inherits this. The from-scratch config (`bsrnn_presence_gate.yaml`) starts unsaturated, but features grow during training.
- **Options, not decided:** normalise $h_t$ before the linear layer (changes eq. 2); scale the init to the feature size; or train from scratch only. Decide before any e21 fine-tune.
- **DECIDED 2026-10-14 (Grant): layer-normalise $h_t$ after the band mean, no learned affine** (`src/models/presence_gate.py`; eq. 2 updated in the note). Verified on e21's real features: logits went from +10 to +56 (91 % of frames pinned) to +0.59 to +0.86 (0 % pinned). Both anchors now unsaturated, so beta and lam are re-measured on Kaggle with the rebuilt code zip. Move this into decisions-m2.md with the gate's registration.
- **Measurement.** The laptop cannot run it: one 4 s crop at batch 1 exceeded 6 GiB (OOM-killed; the first attempt took the terminal with it). It now runs on Kaggle (`notebooks/kaggle_presence_weights.ipynb`), reports both anchors and flags saturation. A from-scratch run reads the **init** anchor.
- **Measured 2026-10-14 on Kaggle T4, full `sir0_train`, 8 batches of 3** (`experiments/results/2026-10-14-presence-weights/meta.yaml`). Gate unsaturated at both anchors (0 % of frames pinned).

  | anchor | KL share at weight 1 (mean) | ±SD | 95 % CI | beta | BCE share at weight 1 (mean) | ±SD | 95 % CI | lam |
  |---|---|---|---|---|---|---|---|---|
  | random init | 859 % | 581 pp | ±486 pp | 0.01747 | 185 % | 198 pp | ±166 pp | 0.08115 |
  | e21 | 942 % | 854 pp | ±714 pp | 0.01592 | 377 % | 404 pp | ±338 pp | 0.03974 |

  - **Superseded by the 64-batch run below.** Too noisy to set weights from: per-batch shares span 0.5x-17x (KL, init).
- **Re-measured 2026-10-15, 64 batches of 3, same seed** (`experiments/results/2026-10-15-presence-weights/`; batches 0-7 reproduce the 8-batch run exactly). Gate unsaturated at both anchors.

  | anchor | KL share at weight 1 (mean) | ±SD | 95 % CI | beta | BCE share at weight 1 (mean) | ±SD | 95 % CI | lam |
  |---|---|---|---|---|---|---|---|---|
  | random init | 1177 % | 734 pp | ±183 pp | **0.01274** | 320 % | 299 pp | ±75 pp | **0.04689** |
  | e21 | 839 % | 643 pp | ±161 pp | 0.01787 | 314 % | 249 pp | ±62 pp | 0.04773 |

  - **Set: beta 0.01274, lam 0.04689** in `bsrnn_presence_gate.yaml`; lam 0.04689, beta 0 in `bsrnn_presence_gate_b.yaml`. Init anchor (from-scratch configs), mean of per-batch shares as `derive_w_state.py` did for D14. Rule fixed before the 64-batch numbers were seen.
  - **Why this average:** grad_clip 5.0 sits far below the base norm (1,760 init), so every step is clipped to the same length. What a term does to a step is its share of that step, so the mean of per-batch shares is the quantity; a ratio of means would let a few large-gradient batches decide.
  - The 8-batch weights were 1.37x (beta) and 1.73x (lam) the 64-batch ones, outside the new 95 % ranges (beta 0.0110-0.0151, lam 0.0380-0.0612).
  - **Clipping.** grad_clip 5.0 sits below the base norm alone (3,030 init, 53 e21), so every step is clipped already. Both terms raise the global norm's upper bound by 12 % at init, but by 3.4x at e21, mostly on the gate's own weights. A fine-tune from e21 would spend most of its clipped step on the gate.
- **Arm A trained** (`experiments/results/2026-10-16-train-sir0-presence-gate-A/`): session 1, 13 epochs on Kaggle (10.4 h), ended "STILL IMPROVING"; session 2 resumed from its epoch-12 checkpoint with `EPOCHS = 26`. **Schedule artefact:** `train.py:545` spreads the gate temperature over `--epochs`, so tau went 1.0 -> 0.5 over epochs 0-12, then jumps to 0.74 at epoch 13 and anneals to 0.5 again by epoch 25. Report it with any session-2 number. A separate Colab run of the same config was cut off after 3 epochs (`2026-10-15-train-sir0-presence-gate-A-s1`) and is not continued.
- **Arm A epoch 13 (index 12), Gemini judge** (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in, 2026-10-17; `2026-10-17-eval-presence-gate-A-e12-judge`): LCF-WER **51.98 %** on 103 clips (no processing 63.01, target alone 1.14; 1 judge timeout, clip 000116). Paired on 102 clips (`2026-10-17-paired-presence-gate-A-e12/`): **+8.5 pp worse** than no-gate at epoch 13 (SD 3.5, 95 % CI +1.9 to +15.7) and +12.3 pp worse than e21 (SD 3.7, CI +5.2 to +19.7). The extra errors are substitutions (+5.3) and insertions (+5.3); deletions are not higher (-2.1, CI spans 0); interferer leakage +9.4 pp (CI +3.0 to +16.1). So the gate run lets more of the other speaker through. The "gate mutes the target" reading, from the 40-clip probe and one Whisper clip, is **not** supported by the judge. Interval covers test-set noise only, not seed variance.
- **Arm A session 2** (`experiments/results/2026-10-16-train-sir0-presence-gate-A-resume/`, idx 13-25, Kaggle T4, 10.3 h, 2026-10-17). The Whisper probe ended at its best: smoothed 58.91, raw 56.19 at idx 25, against e21's raw 55.00 on the same 40 clips, so by the C+S stopping rule a third session qualifies. Everything else on validation says the model is memorising and suppressing harder: target separation -3.34 (idx 14) to -1.10 at idx 25 (lower is better) while train went -3.04 to -5.75; the gate's log-loss 0.54 to 0.74 nats/frame, worse than always saying 50 % (0.69), while train went 0.43 to 0.18; artefact share of the error 0.59 to 0.68, interferer share 0.27 to 0.20. The probe rewards that pattern; the judge punished it at e27 (decisions-m2.md 2026-09-26). **Temperature may matter:** at the resume, lr unchanged, val separation jumped -1.41 to -3.03 and the gate's val log-loss 0.68 to 0.54 when tau restarted 0.50 to 0.74, then both drifted back as tau annealed to 0.50. One uncontrolled observation, confounded with the lr halvings (idx 19, 23) and epoch count. **Next, before choosing session 3's tau:** judge idx 18 (best kept val separation, -3.00, tau 0.64) and idx 25 (best probe, tau 0.50) against e21 on the 103 `sir0_val` clips.
- **Arm A session 2, Gemini judge** (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in, 2026-10-17; e21 answers 2026-09-25). Paired on 98 clips (5 gate-A timeouts dropped from every system), cache only: `experiments/results/2026-10-18-paired-presence-gate-A-s2/`. LCF-WER vs e21: idx 18 **+10.4 pp** (CI +4.6 to +16.6), idx 25 **+8.4 pp** (CI +2.6 to +14.6). Vs the no-gate run at the same index: idx 18 +8.4 (CI +2.8 to +14.1); idx 25 vs no-gate idx 27 +9.5 (CI +3.3 to +15.4). Idx 25 vs idx 18: word error -2.0 (CI spans 0), leakage **-11.1** (CI -16.5 to -6.0), so idx 25 leaks about as much as e21 (+4.9, CI spans 0). The remaining gap is wrong words: substitutions +6.5 per 100 reference words vs e21 (CI +3.5 to +9.4); deletions not higher. Same verdict as session 1: the gate run is ~8-10 points worse than no gate, beyond the noise. Analysis with Claude.
- **Arm A session 3 trained** (`experiments/results/2026-10-18-train-sir0-presence-gate-A-resume2/`, idx 26-38, 10.7 h). tau restarted 0.66 -> 0.50. Probe never beat idx 25 (best raw 57.92 at idx 30, vs 56.19); val separation best -2.77 (idx 27), never back to session 2's -3.34. Not judged yet.
- **Fallback, not used:** `scripts/make_presence_mini_bundle.py` (24-trial subset, 75 MB) and `notebooks/colab_presence_weights.ipynb`, built while the Kaggle session stalled attaching the 19 GB audio.
- **Not yet logged:** `bsrnn_presence_gate.yaml`'s header cites "decisions-m2.md 2026-10-14", which does not exist. Write the presence-gate registration (arms, predictions P1–P4, from scratch vs fine-tune) into decisions-m2.md before training.
- **2026-10-18, diagnosis and the next run:** the gated loss shielded the mask (arm A lost 2.4 dB of interferer suppression vs e21) and the gate fitted training crops far beyond validation (0.18 vs 0.53-0.74 nats/frame), failing where the extractor picks the wrong voice; the fix, "postfilter" mode on a frozen e21, is registered in decisions-m2.md 2026-10-18 (`bsrnn_presence_postfilter.yaml`). Arm A is closed as a negative result.

**AI-use.** Claude-originated: the saturation check and the measurement changes (device, graph freeing, zero-gradient skip, both anchors reported), the 24-trial subset with the Colab notebook, the spread and clipping analysis above, the arm B config file, the 64-batch re-measurement, and the clipping argument for the mean of per-batch shares.

## 2026-10-18 — OPEN: the offline ASR transcript cache can serve one clip's transcript for another

`src/live_model_metric/evaluate.py::_cache_key` is `model | trial dir | file name | mtime (whole seconds) | size`. No estimate directory, no content hash. Two `estimate.wav` files for the same trial, written in the same second at the same size, share a key. **Seen 2026-10-18:** `scripts/oracle_gate_ceiling.py` writes two variants per trial back to back, and the second directory's ASR run was served the first's transcripts for 9 of 13 differing clips (LCF-WER 49.39 reported; 53.14 after forcing fresh transcripts). Worked around by re-touching those 13 files. The judge is not affected: `judge.cache_key` hashes the bytes, and its `audio_fingerprint` docstring records the same bug fixed there. Past ASR rows are at risk only where two variants of a trial were written within one second (make_estimates writes one directory per run, minutes apart). **Fix, not applied:** add `judge.audio_fingerprint(path)` to the key. It invalidates every cached ASR transcript, so the next ASR run of each directory re-transcribes (~6 min per 103 clips). Grant to decide.

## 2026-10-28 — CLOSED 2026-10-29: target tracker, forward-only test (D1, D2), registered before running — no training

Idea (Grant): follow the target while it talks alone, then estimate how much of each overlapped frame is the target. Plan: `docs/extra/2026-10-28-target-tracker-plan.md`. Script: `scripts/diagnose_target_tracker.py`. No weights change; e21's stored outputs (`2026-09-25-est-cuecontext-wer-e21`) are edited band by band.

- **Fixed before any result.** Only frames where both speakers talk (Silero labels on the clean tracks, generator.yaml settings) are changed, and a band is never made louder.
  - D1, oracle energy: each band turned down to the clean target's band energy. Ceiling for D2.
  - D2, Kalman per band on log band energy: random-walk variance q_k fitted on 200 sir0_train trials with target speech (seed 42); observation variance 1e-4 when the target talks alone, no update when it is silent, r_overlap in overlap at three values, q̄, 10 q̄ and 100 q̄ (q̄ = median q_k). All three reported.
- **Judge:** `gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in. Paired bootstrap over the 103 trials against e21 (answers of 2026-09-25), 10,000 draws, seed 42.
- **Stop rule:** if no D2 setting lowers LCF-WER with a 95 % interval below 0, no training run.

**AI-use.** The idea is Grant's; the Kalman formulation, the D1/D2 design and the script are Claude's.
- **Signal check, before the judge** (`experiments/results/2026-10-28-tracker-diagnostics/signal_check.json`, prior in `2026-10-28-tracker-prior/`). Overlap is 26.7 % of frames. Median random-walk variance 0.656 (log energy per 8 ms step, about 3.5 dB). SI-SDR vs e21 (2.31 dB): D1 +1.88 dB (overlap turned down 3.26 dB); D2 q̄ −0.02, 10 q̄ −0.66, 100 q̄ −1.95 dB (Kalman gain in overlap 0.62, 0.28, 0.10). The more the filter trusts its prediction, the worse: a random-walk prediction is a worse guess than e21's output. Room exists (D1), the simple tracker does not find it. Gemini runs pending.
- **D1, Gemini** (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in, 2026-10-28; `2026-10-28-eval-e21-tracker-d1-oracle-energy-judge`; paired in `2026-10-28-tracker-diagnostics/paired.txt`, e21 rebuilt at its published 39.23). LCF-WER 26.34, change **−12.89 [−16.80, −9.14]**, 69 trials better, 16 worse, 18 tied. Deletions 9.40 → 4.57 %, mean leak 22.33 → 13.43 %. Oracle: it uses the clean target's band energy, so it is a ceiling, not a method. SI-SDR saw only +1.88 dB of it.
- **D2 at q̄, Gemini** (2026-10-28; `2026-10-28-eval-e21-tracker-d2-kalman-r1q-judge`; clip 000035 timed out, so paired on 102 clips with it dropped from every system, `paired.txt`). LCF-WER 42.79 against e21 39.52 on the same 102: change **+3.27 [−0.33, +7.37]**, inside the noise but pointing worse; 30 trials better, 40 worse, 32 tied. Run row (103, timeout scored as a non-response): mean leak 22.33 → 26.03 %, invented 2.15 → 2.82 per trial, deletions 9.40 → 7.36 %. The lightest setting, closest to e21 (Kalman gain 0.62), already leaks and invents more. D1 on the same 102: −12.89 [−16.90, −9.07].
- **All four, Gemini, paired on 95 clips** (8 judge timeouts across the runs dropped from every system: 000035, 054, 057, 067, 070, 116, 133, 134; `2026-10-28-tracker-diagnostics/paired.txt`). e21 37.28 on these 95. D1 **−11.05 [−14.98, −7.36]**. D2 q̄ +3.24 [−0.10, +7.01], 10 q̄ **+4.37 [+0.80, +8.20]**, 100 q̄ **+11.95 [+7.51, +16.57]**; deletions 6.2, 7.9, 12.3 % (D1 4.5 %). The more the filter trusts its random-walk prediction, the worse Gemini does, in step with the SI-SDR check.
- **Decision (Grant, 2026-10-29): stop rule met, no training run.** Band-level turn-down in overlap is worth up to 11–13 points to Gemini (D1), but predicting the target's band energy from its recent past cuts the target's onsets and leaves leak in its dips. A learned trust head would only learn how far to ignore that prediction. Open for future work: a learned predictor of the target's next-frame band energy.

## 2026-10-29 — CLOSED 2026-10-29: turn-down head on a frozen e21 (learn the D1 ceiling), registered before running — not better than e21

Follows the tracker test above. D1 showed that turning each band of e21's output down to the clean target's energy in overlap lowers LCF-WER by 11–13 points; this asks whether a model can learn that turn-down. Grant chose this option (2026-10-29).

- **Model.** e21 frozen (weights and ECAPA unchanged). A new head per band reads e21's separator state (128 channels) plus the log band energy of e21's estimate and of the mixture, and outputs a gain in (0, 1) per band per frame (norm, 1x1 conv to 64, tanh, 1x1 conv to 1, sigmoid; output bias +5 so it starts at about 0.99, close to e21). The gain multiplies e21's estimate, band by band. Per-frame and causal, so it streams.
- **Target.** The oracle gain g* = min(1, sqrt(E_target / E_estimate)) per band, in frames where the clean target is active (frame energy within 40 dB of the crop's loudest frame), and 1 elsewhere, so it never learns to blank the target's silences (the presence gate's failure, 2026-10-18). g* floored at -30 dB. Differs from D1 in using an energy rule for activity, not Silero, and in acting in every target-active frame, not only overlap.
- **Loss.** Mean squared error between predicted and oracle gain in dB, over bands with non-silent estimate energy. Only the head trains; AdamW, lr 1e-3, 6 epochs, seed 42, selection on sir0_val loss.
- **Test.** Gemini (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in) on sir0_val `both`, paired bootstrap against e21 (10,000 draws, seed 42). Matching ceiling: the same oracle gain applied by the same rule, forward-only. Success: LCF-WER change with a 95 % interval below 0. The share of the ceiling recovered is reported either way.

**AI-use.** Option chosen by Grant from Claude's list; the head design, target, loss and scripts are Claude's.
- **Built 2026-10-29, not trained.** `src/models/turn_down_head.py` (278,688 weights), `scripts/train_turn_down.py` (GPU-only; `--check` is forward-only), `experiments/configs/turn_down_head.yaml`, `scripts/make_turn_down_estimates.py`, `tests/test_turn_down_head.py` (4 pass). `bsrnn.extract()` split into `extract_spectrogram()` + inverse STFT; output bit-identical to e21's stored estimate, streaming tests pass. At init the head gives −0.06 dB against an oracle mean of −5.42 dB (val check, 2 batches). Matching ceiling estimates written (forward-only): `2026-10-29-est-e21-turn-down-oracle`, not yet judged. Kaggle bundle (110 MB, e21 weights + ECAPA) and `notebooks/kaggle_train.ipynb` ready.
- **Matching ceiling, Gemini** (2026-10-29, `2026-10-29-eval-e21-turn-down-oracle-judge`; 4 timeouts dropped from every system, 99 clips, `2026-10-29-turn-down-diagnostics/paired.txt`). LCF-WER 27.03 against e21 38.03: **−11.00 [−17.13, −4.74]**, 58 better, 22 worse, 19 tied. Deletions 8.2 % (D1, overlap only: 4.5 %), so acting in every target-active frame removes more of the target than acting in overlap only.
- **Kaggle training, epochs 0–4** (T4, log pasted by Grant): val loss 42.73, 41.89, 42.27, 41.61, 41.41 dB² (RMS error about 6.4 dB per band and frame); train 37.75 → 34.88; mean val gain predicted −3.34 to −3.75 dB against the oracle's −3.60. Epoch 0 took 36.4 min, then 23.6 min per epoch. Most of the learning happened in epoch 0; train falls while val is flat, which fits the per-session frozen crops (persistent workers, 2026-10-24).
- **Trained 2026-10-29** (Kaggle T4, 2.6 h, 1,546 s/epoch; `experiments/results/2026-10-29-train-turn-down/`, commit 3cc2a2c-dirty from the bundle). Val loss by epoch 42.73, 41.89, 42.27, 41.61, 41.41, **41.34** (best, epoch 5, kept as `models/model_sir0_turn-down-head-e5.pt`); mean val gain −3.36 dB against the oracle's −3.60. Head estimates being written; Gemini pending.
- **Head, SI-SDR check before the judge** (103 clips, `2026-10-29-est-e21-turn-down-head`). SI-SDR vs e21 (2.31 dB): head **+0.07 dB**, better on 51/103; ceiling +1.92 dB, better on 101/103. Overall output level: head −1.63 dB, ceiling −0.76 dB, so the head turns down more than the oracle overall without moving the output closer to the target. Gemini pending.
- **Head, Gemini** (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in, 2026-10-29; `2026-10-29-eval-e21-turn-down-head-judge`; 6 timeouts; paired on 94 clips with every system's timeouts dropped, `2026-10-29-turn-down-diagnostics/paired.txt`). e21 36.93 on these 94. Head 39.90: **+2.97 [−1.27, +7.19]**, inside the noise, pointing worse; 28 better, 49 worse, 17 tied. Ceiling on the same 94: −9.84 [−16.17, −3.38]. Run row (103, timeouts as non-responses): mean leak 22.33 → 17.47 %, invented 2.15 → 3.09 per trial, FR@2 45.10 → 59.14 %.
- **Reading.** Success criterion not met. The head lowers leaked words but raises invented ones, the same trade-off as the loss-weighted fine-tunes (2026-10-03), where the ceiling lowered both. Learning the turn-down from e21's own per-frame state lands on e21's trade-off, not on the oracle. Not retrained. Open: give the head past frames, or put the oracle's band-energy target into the mask's own loss.

## 2026-10-29 — OPEN: a small lookahead for context (L = 3 frames, 24 ms), registered before running

Grant's request (2026-10-29), after the turn-down head: give each mask decision a little future context, so a word's onset is seen before the frame is masked, rather than smoothing (the mask is already over-smooth, 2026-09-12 above).

- **Method: delayed output.** The network still runs causally; the mask for output frame t comes from the network's state after frame t + L. Whole clip: L silent frames are appended to the network's input, and output frame t reads features[t + L]. Streaming: each mixture frame waits until the network has run L frames past it; at the end, L silent frames are pushed through the network only. No new weights. `model.lookahead_frames`, default 0, so every existing model and checkpoint is unchanged.
- **Cost:** L x 8 ms = 24 ms more delay. e21's measured mean 175 ms and worst 196 ms become about 199 and 220 ms: inside the 200–300 ms budget, at its lower edge.
- **Training: matched to the 2026-10-03 control fine-tune.** From e21, lr 1.25e-4, 6 epochs, seed 42, no w_schedule, same data; the only difference is lookahead_frames 3. Needs warm-start (`training.init_weights`) and an optional w_schedule in train.py, both removed in the clean-up. Caveat: the control ran under the pre-refactor train.py.
- **Test:** Gemini (`gemini-3.7-flash`, prompt `d118b7d3bf30`, audio-in) on sir0_val `both`, epoch 5 against the control's epoch 5 (`models/model_sir0_interf-ft-control-e5-full.pt`, LCF-WER 38.24), paired bootstrap, 10,000 draws, seed 42. Success: LCF-WER change with a 95 % interval below 0. Leak, invented and deletions reported either way; streaming delay measured with `scripts/measure_rtf.py`.

**AI-use.** The lookahead idea is Grant's; the delayed-output design, the matched comparison and the code are Claude's.
- **History.** This re-adds a designed knob: `lookahead_frames` (0–16, default 0) was decided in decisions-m1.md (2026-08-18), and a `lookahead_shift()` was implemented and tested with an ablation of {0, 8, 16} planned (above, D-list), but no lookahead model was ever trained, and the 2026-10-22/23 clean-up removed the code. The control's checkpoint config records `lookahead_frames: 0`, `num_workers` 4 (the notebook's override), batch 3 and lr 1.25e-4, which the new config matches.
- **Built 2026-10-29, not trained.** `bsrnn.build_bsrnn_tfmap(lookahead_frames=0)` and `extract_spectrogram` (silent frames appended, features read L frames later); `streaming.py` (a queue of waiting frames, silent frames through the network only at flush); `train.py`: `lookahead_frames` from the config, `training.init_weights` warm start, optional `w_schedule`. e21 still bit-identical to its stored estimate. Tests: streamed equals whole clip at lookahead 0 and 3, and a new test confirms output frame t first changes when audio L frames past it changes, no earlier. Config `experiments/configs/bsrnn_lookahead_ft.yaml`; bundle (110 MB) checked: e21's weights load into the lookahead model, val loss finite. `notebooks/kaggle_train.ipynb` regenerated.
- **First Kaggle attempt, epoch 0 (2026-10-29, log pasted by Grant): 10,856 s, about 4.2x e21's 2,575 s/epoch.** Cause: the 3 silent frames made the network's sequence 507 frames, off the multiple of 8 that keeps fp16 LSTM kernels fast (504 at chunk_s 4.008, the 4.44x of 2026-08-28). Fix: pad to the next multiple of 8 (512) and still read features at t + 3; the network is causal, so outputs are unchanged (streamed, unpadded, still equals whole clip; 81 tests pass). Epoch 0's numbers before the restart: val total −4.72, Whisper probe 51.22 on 40 clips (e21's run read 53.11–63.83 at epochs 18–21, corrected 2026-10-30 from "53.11–55.00"; noisy, Whisper, not a result). Fixed bundle built; whether to stop and restart is Grant's call.
- **First attempt ended at Kaggle's 12 h cap after 3 of 6 epochs** (unpadded bundle, 10,836 s/epoch mean; log pasted by Grant 2026-10-30). Last checkpoint downloaded as `out/model_sir0_last.zip` (a torch file: idx 2, step 9,954, lookahead 3, md5 45ab14f0efaac1f3f2d7c23c78789b6d). Against the control at idx 0/1/2: train L_pres −6.81/−7.31/−7.68 vs −6.80/−7.29/−7.63; val L_pres −4.50/−4.53/−3.99 vs −4.33/−4.76/−4.05; enrol_sens −1.03/−1.09/−0.69 vs −0.82/−1.36/−0.59 dB; pres_abs_gap 21.4/21.2/20.0 vs 20.8/18.6/20.9 dB; Whisper probe 51.2/48.7/56.7 vs 54.8/47.7/63.6 (40 clips). No difference from the control at loss level. Both lose val after idx 1 while train keeps falling, the frozen-crops pattern (2026-10-24).
- **Recommendation (Claude, 2026-10-30): rerun from e21 with the padded bundle, do not resume.** A resume opens a new session, which draws new crops for idx 3–5 (persistent workers, 2026-10-24); the one-session control never had that, and a session's fresh crops moved val L_pres by about 1 dB (e21's −3.17 to the fine-tunes' −4.3/−4.5 at idx 0), more than any effect expected here. Grant's call.
- **Rerun trained 2026-10-30** (Kaggle T4, padded bundle, 5.1 h, 3,053 s/epoch, all 6 epochs; `experiments/results/2026-10-30-train-sir0-lookahead-ft/`, bundle 983f8de-dirty). Installed as `models/model_sir0_lookahead-ft-{e0,e3,e5-full}.pt`. Whisper probe (40 clips) by idx 49.3, 63.6, 61.5, 51.5, 50.2, 65.4, so the probe kept idx 0 as `model_sir0.pt`; val L_pres best at idx 3 (−4.59); lr halved at idx 5.
- **Decision (Grant, 2026-10-30): judge e5 only.** The registered test stands; the probe-selected e0 is not judged (40-clip probe noise, the 2026-10-03 rule). e0 estimates rendered (`2026-10-30-est-lookahead-ft-e0`) but not scored.
- `scripts/measure_rtf.py` fixed 2026-10-30: the delay now adds `lookahead_frames` x hop (24 ms here); every earlier model has 0 frames, so their figures are unchanged.
