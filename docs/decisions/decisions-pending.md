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
