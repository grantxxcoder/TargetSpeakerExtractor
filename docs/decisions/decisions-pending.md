# Decisions needed before audio can be generated

**Written 2026-08-10. Status 2026-08-13: one item left open.**

- **C2** — how hard the task should be (floor word error rate on the raw mixture).
  Needs the supervisor. Blocks nothing that can be done meanwhile.

**Every other decision is made** and recorded in `docs/decisions/decisions-m0.md`. Group A
is closed, so the renderer is unblocked. What remains is implementation, not choices:
B12's two PRs, then the manifest rebuild carrying B9, B10, B4 and the interruption
column.

Plain wording first, the jargon term in brackets so it can be matched to
`docs/data/data-construction-parameters.md`.

Move each answer into `docs/decisions/decisions-m0.md` once made.

---

## A. Blocks the renderer

### A1. *Decided 2026-08-13 — full reverberant, "what the mic heard". See `decisions-m0.md`.*

Pending supervisor sign-off. Dereverberation is an ablation only, if time allows.

### A2. *Decided 2026-08-11 — wrap around. See `decisions-m0.md`.*

Numbers are kept in `decisions-m0.md`. The gap in numbering is deliberate: A3-A6
keep their identifiers so earlier references stay valid.

### A3. *Decided 2026-08-12 — BS.1770 integrated loudness. See `decisions-m0.md`.*

### A4. *Decided 2026-08-12 — no room on the enrollment. See `decisions-m0.md`.*

### A5. *Decided 2026-08-13 — yes, pad by `t60_s`. See `decisions-m0.md`.*

### A6. *Decided 2026-08-13 — common-gain rescale at 0.95. See `decisions-m0.md`.*

**Group A is closed.** The renderer is unblocked.

---

## B. Needed before the real sets are generated

### B1. *Closed 2026-08-13 — subsumed by the difficulty dial. See `decisions-m0.md`.*

Not a standalone decision: `overlap_ratio` is one of the 14 parameters ranked in
`docs/data/difficulty-dial.md`, adjustable on request once B12 lands. Recorded there
as the narrowing to do **last**, because its 0.7 ceiling is deliberately matched to
REAL-TSE and changing it diverges from the anchor.

### B2. *Decided 2026-08-13 — measure from detected speech. See `decisions-m0.md`.*

Detector to be named in the PR that adds it.

### B3. *Decided 2026-08-12 — fixed 5 s, kept configurable. See `decisions-m0.md`.*

### B4. *Decided 2026-08-13 — yes, same fraction as train, scored on their own row. See `decisions-m0.md`.*

The fraction itself follows B9.

### B5. *Decided 2026-08-13 — Whisper `EnglishTextNormalizer`. See `decisions-m0.md`.*

### B6. *Decided 2026-08-13 — 500 generated, 200 the minimum scored. See `decisions-m0.md`.*

### B7. *Decided 2026-08-13 — off for the main run, kept as a switch. See `decisions-m0.md`.*

### B8. *Decided 2026-08-11 — different book. See `decisions-m0.md`.*

### B9. *Decided 2026-08-13 — 50 % both / 25 % absent / 25 % target-only, and a variable target activity ratio. See `decisions-m0.md`.*

Blocks the manifest rebuild until implemented. Sets B4's eval fraction at 0.25.

### B10. *Decided 2026-08-13 — three enrollment tiers, recorded per trial; eval pools redrawn. See `decisions-m0.md`.*

Not a reversal of B8: B8's cost note specified this fallback and its trigger. Folds
into the PR3 rebuild.

### B11. *Decided 2026-08-13 — report a latency decay curve, never cap T60. See `decisions-m0.md`.*

Largely defused by A1: with a full reverberant reference the model is no longer asked
to suppress a tail it has not heard.

### B12. *Architecture decided 2026-08-13 — two regimes, sampler layer, no relational constraints. See `decisions-m0.md`.*

**PR1 and PR2 landed 2026-08-14.** `src/data/sampling.py` holds the sampler;
`build_manifest.py` draws a regime per trial and records it. PR3 (B9/B10/B4 rebuild)
is next.

One band was deliberately not applied: `overlap_ratio` keeps its full `[0.2, 0.7]`
in `base`, because narrowing it diverges from the REAL-TSE anchor and needs
supervisor agreement (`difficulty-dial.md` §3). `target_activity_ratio` likewise
stays fixed until B9 decides what varying it means.

Band values live in `docs/data/difficulty-dial.md` §2; the how-to is
`docs/data/changing-the-data.md`.

Two sub-questions the original entry raised, both now answered in `decisions-m0.md`:
beta is dropped (no use case), and the wall-absorption ambiguity is resolved by
**not** capping absorption — it is derived from `t60_s` and volume, so a cap would
be a rejection rule, and rejection is what bends distributions. Raising the `t60_s`
floor achieves the same realism gain without rejection (`difficulty-dial.md` §1).

### B13. *Decided 2026-08-13 — per condition, no combinations, 100 trials per bucket. See `decisions-m0.md`.*

One part deferred: the **interruption** condition. Nothing marks an interruption
today, and defining one needs the turn-taking trials B9 introduces, so it is fixed
during that rebuild rather than before it.

---

## C. Ask the supervisor

1. **A1 — decided, needs sign-off only.** Reference is what the mic heard from the
   target (full reverberant): separate and denoise, do not dereverberate. Removing a
   0.6 s tail from a 300 ms causal window is not possible, and attempting it trades
   residue for artefacts, which are what degrade recognition most. Dereverberation
   kept as an ablation if time allows. `decisions-m0.md` 2026-08-13.
2. **How hard should the task be?** Measured as how badly an off-the-shelf
   transcriber does on the raw mixture (**floor word error rate**). Too easy and
   nothing distinguishes systems; too hard and nothing can be ranked. The current
   plan targets 60–80 %. **Still open — nothing else here can settle it.**
3. *B10 — decided 2026-08-13. Three enrollment tiers, recorded per trial. Not a
   reversal of B8: B8's own cost note specified this fallback and the trigger for it,
   and 60.2 % of speakers dropping out met that trigger. Worth mentioning, not
   asking. See `decisions-m0.md`.*
4. *B4 — answered 2026-08-12 and now fully decided. See `decisions-m0.md`.*
5. *B11 — decided 2026-08-13, and largely defused by A1. See `decisions-m0.md`.*

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
