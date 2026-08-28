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
