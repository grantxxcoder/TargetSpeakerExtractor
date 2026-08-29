# Open decisions

**Written 2026-08-10.** Groups A–C were the pre-generation data decisions; all are
closed except C2. Full reasoning for each lives in `decisions-m0.md` under its
date. Group D holds open *modelling* questions, which have nowhere else to live;
decisions actually taken go to `decisions-m1.md`.

---

## Still open

- **C2 — how hard should the task be?** Floor word error rate on the raw mixture;
  current plan targets 60–80 %. Needs the supervisor. Blocks nothing meanwhile,
  but M0's floor/ceiling calibration is what answers it.
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
taken go to `decisions-m1.md`.*

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

### D3. Diagnose the conditioning failure before rebuilding anything

**Status: three measurements, all unrun, all cheap, all runnable on
`models/model_sir0.pt` on CPU — no contention with a live GPU run.**

**The one number.** Every run has failed the same way: swapping a stranger's
enrollment in changes the output by at most **15 %** (`val_enrol_sens_db`
-8.25, epoch 7, `2026-08-27-train-sir0`). Five-sixths of the output is decided
without reference to who was asked for. Architecture work should target that
number; everything below is ranked by how much it moves it per unit of cost.

**Why measure first.** The conditioning path has two halves — *build a cue* and
*make the network use it* — and no run so far distinguishes which one fails.
Rebuilding the wrong half is the expensive mistake available here.

- **D3a — is the cue itself speaker-discriminative?** Compute the TF-Map output
  for the true enrollment and for a rolled one, same mixture:
  `||tf_true - tf_swap||^2 / ||tf_true||^2`. Same roll trick as
  `diagnostic_accumulate()` in `scripts/train.py`, one layer upstream.
  **This partitions the problem.** Cue barely moves -> nothing downstream can
  help, fix the cue (D5). Cue moves a lot but the output does not -> the cue is
  fine and the *injection path* is discarding it (D4).
- **D3b — oracle-cue ceiling.** Replace the TF-Map channel with the clean
  target's magnitude spectrogram and train briefly. Still cannot extract -> the
  separator/mask is the bottleneck and conditioning is not the story. Extracts
  well -> conditioning is confirmed as the bottleneck and the run gives its
  ceiling. Upper-bound experiment, deliberately cheap.
- **D3c — stratify every val metric by `same_gender`.** Free: the column is in
  the manifest and `sir0_val` is 76 same-gender / 69 cross. `sir0_train` is
  balanced 50/50 (680/680), which **caps but does not remove** the gender
  shortcut: a model using pitch alone gets the cross-gender half right and coin
  flips the rest, i.e. ~75 % correct with no enrollment at all. If quality and
  sensitivity collapse on same-gender trials, the model is riding gender, not
  identity — and that is invisible in the pooled numbers we currently log.

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

### D7. Status correction — D2 has effectively been run

**`tfmap_scale: 16.0` in `bsrnn_baseline.yaml` is D2's temperature**, scale
being 1/tau, so tau ~ 0.0625 — sharper than D2's sharpest proposed arm (0.05 was
proposed as tau; 16 corresponds to 0.0625). The measurement in `TFMap`'s
docstring confirms it worked mechanically: 619.6/628 frames effectively used
before, top 50 frames carrying ~59 % after.

**But it did not solve the problem, and the gain cannot be attributed to it.**
Enrollment sensitivity went 2.6 % -> 15 %, which looks like a win, except that
`tfmap_scale`, `both_directions` and the `sir0` split all changed between those
two measurements. **Three variables, one number: unattributable.** An ablation
arm on `tfmap_scale` alone is what would close D2 honestly.

**Consequence for D1.** D2's stopping rule was "if sharpening the existing
mechanism does not help, a richer dictionary is unlikely to, and D1 should not be
scheduled". Sharpening helped but left 85 % of the output enrollment-blind, so
D1 is **not** cleanly ruled out — but it remains M5-scale against D4 and D6,
which are hours to days. Order by cost: D3, D6, D4, D5, then reconsider D1.

---

## E. Performance and memory — open

*Added 2026-08-28. Group D is modelling; this is engineering. Both are open
questions with nowhere else to live. Anything actually decided goes to
`decisions-m1.md`.*

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

### E3b. MEASURED 2026-08-28 — what the profile actually says

`scripts/profile_step.py`, batch 1, CPU x4, fp32. **CPU timings do not predict
GPU timings** (no kernel-launch overhead on CPU, different LSTM/conv balance),
so the shares below need re-running on the T4. The RNN dominance and the loader
result transfer; the band-loop share is the one that may not.

    11.381 s/step at batch 1, peak RSS 5100 MB
    forward = 34 % of wall (so backward ~ 66 %, i.e. ~2x forward -- normal)

    stft            0.002 s    0.0 %
    tfmap           0.005 s    0.1 %
    subband_norm    0.019 s    0.5 %
    separator       3.661 s   95.8 %
    estimator       0.134 s    3.5 %

      time_rnn  seq=497 batch=64    1.207 s   31.6 % of forward
      band_rnn  seq=32  batch=994   2.276 s   59.6 % of forward

**E2's prediction is confirmed.** Analytic said the band RNN would be ~67 % of
LSTM work; measured it is 65 % of the two RNNs (59.6 / 91.2). The "cheap"
across-frequency RNN really is the dominant cost.

**Three corrections to the estimates above.**

1. **The band loops are not the bottleneck.** `SubbandNorm` + `Estimator`
   together are **4.0 %** of forward. E4 lever 5 was ranked too high even at
   last place; on CPU it is not worth the refactor risk at all. Kernel-launch
   overhead could make it larger on the T4 — that is the one number worth
   re-checking there — but it will not be the headline.
2. **Data loading is not the bottleneck either.** `--loader-only`: **0.382
   s/batch** for 6 examples at `num_workers=0`, single-threaded and unoverlapped,
   i.e. the worst case. Against the measured 5.84 s/step that is 6.5 %, and on
   Kaggle with `num_workers=4` it overlaps compute and largely disappears.
   **This settles E5: audio compression would buy nothing.** Do not spend time
   on FLAC or on caching STFTs.
3. **E1's memory estimate is ~3x too low.** Analytic said ~1677 MB of LSTM
   activations at batch 1; measured peak RSS is **5100 MB**. The gap is the
   autograd graph, gradients, optimiser state, and — significant — `L_MR`'s
   **eight STFTs** (four window sizes x target and output), all retained for
   backward. Scale E1's table by ~3 when reasoning about what fits.

**So the remaining unexplained time is inside the RNNs, not around them.** The
levers that matter are the ones that make the RNNs cheaper or better utilised:
AMP, a bigger batch for occupancy, checkpointing to enable it, and cutting `T`.

### E3d. MEASURED 2026-08-28 on the T4 — AMP works, batch 3 is the ceiling

`scripts/profile_step.py`, Tesla T4 14.56 GiB, torch 2.10.0+cu128, 8 timed steps.

| batch 3 | s/step | peak GPU |
|---|---|---|
| fp32 | 4.741 | **12.20 GB** |
| AMP (fp16) | 2.968 | **6.57 GB** |

**AMP: 1.60x faster, 1.86x less memory.** Measured, not projected.

**Batch 3 is the fp32 ceiling.** Batches 4, 5, 6 and 12 all raised
`OutOfMemoryError`, every one inside `band_rnn`'s `_VF.lstm` — the term E2
identified. `bsrnn_baseline.yaml`'s comment promising batch 12 is wrong by 4x
and should be corrected: 12.20 GB of a 14.56 GB card leaves no room for a
fourth trial.

**Wall-clock effect is 1.44x, not 1.60x.** The profiled step is compute only.
The real loop measures 5.84 s/step (3875.2 s / 663), so ~1.1 s/step is loader,
validation and the extra diagnostic forward, none of which AMP touches:

    2.968 + 1.1 = 4.07 s/step -> 2698 s/epoch -> 10 epochs in 7.5 h (was 10.8 h)

**E1's analytic memory model is ~2.4x low on GPU** (5032 MB predicted at batch 3,
12.20 GB measured), consistent with the ~3x under-estimate on CPU. Multiply E1's
table by ~2.5 before using it to predict a ceiling.

**Attribution from that run is VOID.** It reported `estimator` 97.4 % and
`separator` 0.8 %, inverting the CPU result. Two bugs, both fixed 2026-08-28:
CUDA kernels are asynchronous, so unsynchronised `perf_counter` hooks measure
QUEUING and whichever module blocks last absorbs the whole queue; and the AMP
loop accumulated into the same counters as the fp32 loop (`calls=17` where 8
were expected). The fixed script syncs per boundary and runs attribution as its
own gated fp32 pass. **The GPU attribution still needs re-running** — E3b's CPU
split (separator 93-96 %, band_rnn 57-60 %, band loops 4 %) is the only valid one.

**Open, and it decides whether checkpointing is needed: the AMP batch ceiling.**
Unknown, because the sweep runs fp32 first and OOMs before reaching AMP.

**The first `--amp-only` attempt (2026-08-28) was VOID** and reported the fp32
ceiling a second time: the warmup step before the timing loops ran unconditionally
in fp32, so it allocated the exact footprint the mode exists to avoid and OOM'd at
batch 4 before AMP was ever exercised. Fixed by constructing the `GradScaler`
before the warmup and passing it in when `--amp-only` is set; the mode now also
refuses to run without CUDA. **Any `--amp-only` result from a bundle built before
2026-08-28 21:00 should be discarded.**

Re-run over batches 4-8. At 6.57 GB for batch 3, batch 6 is plausible and batch 8
is not.

### E3c. Operational: profiling locally can kill the terminal

On 2026-08-28 an earlier `profile_step.py` at the config's batch 3 triggered:

    systemd-oomd: Killed .../app-org.gnome.Terminal.slice/vte-spawn-*.scope
    due to memory pressure ... 57.19% > 50.00% for > 20s with reclaim activity
    -> killed 13 process(es) in this unit

**systemd-oomd kills the whole cgroup scope, not the offending process**, and it
fires on sustained PSI pressure rather than absolute exhaustion — swap thrash is
enough. At 5.1 GB per batch-1 step, batch 3 needs ~15 GB on a 15 GB laptop.
This is the same failure `measure_train_cost.py` records against VSCode on
2026-08-24.

**Protocol for any local profiling or training:**

    systemd-run --user --scope -p MemoryMax=5G -p MemorySwapMax=0 -- \
        ../tse_venv/bin/python scripts/profile_step.py --batch 1

`profile_step.py` now also refuses to start when its estimate exceeds half of
available RAM, and keeps `torch.profiler` behind `--deep` because it retains a
record per op.

### E3e. MEASURED 2026-08-28 — AMP ceiling is batch 6, and a bigger batch buys NOTHING

`scripts/profile_step.py --amp-only`, T4 14.56 GiB, 8 steps, fp16.

| batch | s/step | **s/trial** | peak GB | GB/trial |
|---|---|---|---|---|
| 3 | 2.971 | **0.990** | 6.57 | 2.19 |
| 4 | 0.926 | *0.232* | 8.72 | 2.18 |
| 5 | 4.969 | **0.994** | 10.87 | 2.17 |
| 6 | 6.028 | **1.005** | 13.02 | 2.17 |
| 12 | OOM | — | — | — |

**Memory is exactly linear: 0.12 GB fixed + 2.15 GB per trial**, three identical
2.15 GB increments. Batch 7 would need 15.17 GB against 14.56 available, so
**batch 6 is the fp16 ceiling** — confirmed by the batch-12 OOM.

**The finding that changes the plan: throughput per trial is FLAT.** Batches 3,
5 and 6 sit at 0.990, 0.994 and 1.005 s/trial — a 1.4 % spread across a 2x batch
range. **The T4 is already saturated at batch 3.**

**Therefore gradient checkpointing is NOT worth building (E4 lever 3 is
withdrawn).** Its entire justification was that batch 3 gives only 192 parallel
sequences, that an LSTM's batch is its only parallelism, and that a larger batch
would repay the ~33 % recompute cost through occupancy. **Measured, occupancy
does not improve.** Checkpointing would buy memory headroom we have no use for
and charge 33 % more compute for it. That prediction was wrong and the
measurement is the reason to drop it.

**Batch 4 (0.232 s/trial) is a 4.28x outlier and is NOT yet a result.** Its
memory sits exactly on the linear trend, so it did allocate the normal
footprint; only the time is anomalous. A plausible mechanism is a cuDNN LSTM
kernel switch — `time_rnn`'s batch is `B*K` = 256 at batch 4, the only power of
two in the sweep — but that should not yield 4x when `time_rnn` is ~1/3 of the
RNN work, so the mechanism does not explain the size of the effect.
**If real it is worth 8.4x on epoch time (3875 s -> 460 s) and dwarfs every
other lever in this group.** Verify before believing: re-run batches 3, 4, 5 at
`--steps 20`. Do not put it in the thesis on one observation.

### E3f. MEASURED 2026-08-28 — the 4x is REAL: fp16 tensor-core batch alignment

**Batch 4 reproduced six times across two sessions, interleaved with 3/5/6, at a
0.3 % spread.** It is not noise.

| batch | ex | runs | mean s/step | spread | **s/example** | `band_rnn` batch = ex*T | %8 |
|---|---|---|---|---|---|---|---|
| 3 | 6 | 4 | 2.993 | 0.9 % | 0.499 | 3018 | 2 |
| **4** | 8 | 6 | **0.977** | **0.3 %** | **0.122** | **4024** | **0** |
| 5 | 10 | 2 | 5.005 | 0.0 % | 0.501 | 5030 | 6 |
| 6 | 12 | 1 | 6.007 | — | 0.501 | 6036 | 4 |

**The mechanism.** NVIDIA tensor cores require the batch dimension to be a
multiple of 8 for fp16. `BSNet.forward` reshapes to `(B*T, N, K)` for
`band_rnn`, so its batch is `ex * T`. **T = 503 is odd** (prime), so
`ex * T % 8 == 0` requires `ex % 8 == 0`, i.e. **batch divisible by 4**. Miss it
and cuDNN falls back to a non-tensor-core kernel.

**The hypothesis predicts the data exactly, with nothing fitted:** of 3, 4, 5, 6
it says only 4 aligns, and only 4 is fast. The three unaligned sizes agree with
each other to 0.4 % (0.499 / 0.501 / 0.501 s per example) — the signature of a
shared fallback kernel. Aligned vs unaligned is **4.09x**.

**T was wrong in every earlier entry: it is 503, not 497.** `profile_step.py`
used `(n - n_fft)//hop + 1`, but `src/models/stft.py` pads by `n_fft - hop` on
*both* sides for overlap-add ramp room, so the formula under-counts by 6 frames.
Now measured from the real STFT. E3b/E3d/E3e's stated T is wrong; their timings
and memory figures are unaffected.

### The fix: `chunk_s` 4.0 -> 4.008, NOT a batch-size change

`T = 504` at 4.008 s, and 504 is a multiple of 8, so **every batch size aligns**
(verified for 3-6). Why this is better than moving to batch 4:

- **The crop is 0.2 % LONGER, not shorter** (4.008 s vs 4.000 s), so nothing is
  lost.
- **`chunk_s` is a crop parameter applied by `TrialDataset`, not a render
  parameter — no re-rendering, one config line.**
- **Batch stays 3, so training dynamics are untouched.** Changing batch 3 -> 4
  changes gradient noise and drops optimiser steps per epoch from 663 to 497,
  which is a real change to the optimisation and would need its own arm. A
  0.2 % longer crop is not.

**CONFIRMED 2026-08-28.** Profiler: 0.674 s/step at batch 3 with
`--chunk-s 4.008`, reproduced twice at 0.1 % spread, against a 0.73 s prediction.
Then the real 2-epoch `sir0` run measured **505.7 s/epoch against 3875.2 s =
7.66x**, beating even the optimistic end of the 3.9-7.0x range in E3d (the
~516 s of unaccounted overhead evidently scales with compute rather than being
fixed). 10 epochs is now **1.4 h instead of 10.8 h**.

**Losses are unchanged.** Against the fp32 / `chunk_s` 4.0 run at the same seed,
split and `w_g`, epoch 1 agrees on all twelve logged terms to within 3 % and is
marginally *better* on most (`val_total` -1.4 %, `val_L_pres` -0.9 %,
`val_L_MR` -2.0 %, gap +3.0 %). Epoch 0's *train* terms lag (`train_L_pres`
-1.96 -> -1.42) while its *val* terms match to 4 % -- exactly the signature of
`GradScaler` skipping its first optimiser steps while calibrating the loss
scale: fewer updates early in the epoch, caught up by the time val is measured.
Benign and expected.

**Still untested: both epochs ran at `w` = 0.0.** The absent branch, and the
mute pressure it creates, never engaged. fp16 behaviour through the `w` ramp
(epochs 4-6) is not yet evidence.

**Does not revive checkpointing.** Throughput per example was flat across the
three unaligned sizes, and we have only one aligned point, so nothing yet
suggests a larger batch helps. E4 lever 3 stays withdrawn.

**Loader settled.** Measured on Kaggle at **0.044 s/batch** (`num_workers=4`),
i.e. 4.5 % of a 0.977 s step. E5 stands: audio compression buys nothing.

`profile_step.py` now measures T from the real STFT, takes `--chunk-s`, and
prints an ALIGNED / NOT ALIGNED verdict with the aligned batch sizes for the
current T.

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
