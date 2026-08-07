# LCF — Live-model Content Fidelity

**Status:** draft v0.1, 2026-08-07. Not yet reviewed by supervisors.
**Purpose:** the project's primary contribution. Defines what we measure,
how, and why it resists gaming.

---

## 1. What this metric is for

Conventional TSE is scored on how good the separated audio *is* — signal
quality (SI-SDR), perceptual quality (DNSMOS, PESQ), or transcribability by
an offline ASR system (WER/TER). The observation motivating this project is
that these can diverge from what we actually care about:

> Conventional TSE can improve offline transcription accuracy while making
> the audio *harder* for a live speech-to-speech model to understand.

Plausible mechanism: masking-based extraction leaves spectral holes,
musical noise and phase artefacts. An ASR model trained on clean read
speech tolerates these because they don't move the phonetic decision
boundaries much. A live speech-to-speech model consumes audio through a
learned audio encoder over a much wider distribution, and appears to be more
sensitive to the artefacts of the processing itself than to the interfering
speech it removed. This is a hypothesis, not established fact — testing it
is result #1 of the project.

So the metric must measure **what a live model actually recovered**, and it
must be blind to how nice the audio sounded.

---

## 2. Trial definition

A trial is a tuple:

| Element | Symbol | Notes |
|---|---|---|
| Mixture audio | `x` | Target + ≥1 interfering speaker + noise/reverb |
| Enrollment audio | `e` | ≥5 s of the target speaking alone, from a different recording |
| Target ground-truth text | `t` | Exact verbatim text of what the target said in `x` |
| Interferer ground-truth text | `d` | Exact verbatim text of what the interferer(s) said |

`d` is required — it is what makes the contamination score computable, and
without it the metric is one-sided and trivially gameable.

The system under test produces `ŝ = f(x, e)`, streaming, within the latency
budget.

---

## 3. Scoring procedure

1. Present `ŝ` to a live speech-to-speech model under a **fixed, published
   prompt** asking it to report what it heard.
2. If the model responds in audio, transcribe the response with a **fixed
   ASR system**, identical across all conditions. Record it as part of the
   protocol — it is a component of the measuring instrument, and changing it
   invalidates comparisons.
3. Score the response text `r` against `t` and `d`.

### 3.1 Primary score — LCF-WER

Word error rate of `r` against `t`. **Lower is better.**

This is the headline number: how much of what the target said did the live
model actually recover?

Normalisation (casing, punctuation, numbers, disfluencies) must be fixed and
published. Do not hand-tune it per system.

### 3.2 Secondary score — ICR (Interferer Content Rate)

Fraction of trials where `r` contains content attributable to `d` rather
than `t`. **Lower is better.**

Computed as content-word overlap between `r` and `d`, excluding words that
also appear in `t`, thresholded. The threshold must be fixed in advance and
its sensitivity reported.

**This is the score that makes the metric two-sided**, and it is the one an
offline WER-based metric structurally cannot see.

### 3.3 Secondary score — NRR (Non-Response Rate)

Fraction of trials where the model declines, reports hearing nothing,
returns silence, or produces a refusal. **Lower is better.**

Catches the degenerate "output silence" strategy, which would otherwise
score perfectly on ICR.

### 3.4 Reported anchors — mandatory

Every results table reports these two rows alongside the systems:

| Anchor | Audio presented | Interpretation |
|---|---|---|
| **Floor** | Unprocessed mixture `x` | Doing nothing. Any system must beat this or it is worthless. |
| **Ceiling** | Clean target only | The best any extraction could achieve on this judge. |

On constructed trials the ceiling is exact. On AMI it is **approximate**,
computed from the individual headset (IHM) channel, which carries cross-talk
bleed and a different channel response from the distant mixture mic. Always
label it as approximate. See `docs/decisions.md`, 2026-08-07 data decision.

---

## 4. Why this resists gaming

The REAL-TSE Challenge is the cautionary tale to design against: teams
over-optimised DNSMOS-OVRL badly enough — in one case with adversarial
waveform perturbations — that its correlation with human MOS on Track 1 was
essentially zero (LCC +0.003), and the organisers had to swap the official
metric after the fact.

Four properties, built in from the start:

**Semantic, discrete and non-differentiable.** The score depends on whether
the right words came back, not on a smooth perceptual predictor. There is no
gradient to hill-climb with waveform perturbations.

**Two-sided.** Suppress everything and NRR blows up. Pass everything through
and ICR blows up. Only genuine extraction moves LCF-WER without moving the
other two. A single-score metric would not have this property, which is why
all three are always reported together.

**The judge is held out from training.** Enforced by the different-model-
family rule in `docs/decisions.md`. The judge is also never used as a
training-data filter.

**Anchored.** The clean-target ceiling bounds what is achievable on this
judge, so an implausible score is visible immediately rather than being
mistaken for a breakthrough.

Additionally: keep a **private trial split**. Publish the protocol, the
construction code and a public split; hold back a private split for the
headline numbers. Prevents overfitting to specific utterances.

---

## 5. Methodological requirements

These are not optional — they are what makes the benchmark a scientific
instrument rather than an anecdote.

**Pin and date everything.** Record the exact judge model ID, the exact
prompt, the exact response-transcription ASR, and the run date for every
number. Closed live models change silently. Cross-date comparisons are
invalid unless re-run.

**At least one open-weight judge.** Alongside the closed API, evaluate with
an open-weight speech-to-speech model so the benchmark is reproducible by
someone without API access. This is the single most important decision for
the metric's shelf life — a benchmark only reproducible on a paid API whose
behaviour changes monthly is not a contribution. Candidate survey needed
before the protocol is frozen.

**Report variance.** Live models are stochastic. Run each trial `k` times
(k ≥ 3) and report mean with confidence intervals. A difference smaller than
the run-to-run spread is not a difference.

**Fix decoding parameters** where the API exposes them, and record them
where it doesn't.

**Budget the API cost before building the trial set.** Cost per trial × number
of trials × number of systems × number of judges × k repeats. Size the trial
set to the budget, on a spreadsheet, before writing the harness.

---

## 6. Relationship to existing metrics

We report LCF-WER/ICR/NRR as the primary result, and conventional metrics
alongside — SI-SDR, DNSMOS-P808, and offline ASR WER — **specifically to
show where they diverge.** The divergence is a result, not a nuisance:
demonstrating that a system can improve offline WER while worsening LCF-WER
is the empirical justification for the whole metric.

Prior art to position against, not duplicate:

- **PS4** (Ning et al., arXiv:2607.08111) — proxy-supervised training with
  differentiable ASR/speaker/VAD/quality objectives. Closest existing work
  on optimising TSE for what a downstream model needs. Difference: PS4's
  proxies are also its evaluation; we hold the judge out, and our judge is a
  live speech-to-speech model rather than a cascaded ASR.
- **Ma et al.** (arXiv:2501.14477) — joint generative TSE + target-speaker
  ASR for intelligibility. Directly addresses the transcribable-but-worse
  phenomenon.
- **REAL-TSE TER** — offline Zipformer transcription. This is the metric we
  are arguing is insufficient for the live-model use case.

---

## 7. Open questions

- Which open-weight speech-to-speech model is the reproducible anchor. Needs
  a survey before the protocol freezes.
- Prompt sensitivity — how much does the "report what you heard" wording
  move the scores? Needs a small ablation; if it dominates, the metric is
  fragile and needs redesign.
- Whether ICR's overlap threshold is stable across judges, or needs
  per-judge calibration (which would weaken cross-judge comparison).
- Whether to add a semantic-equivalence score alongside LCF-WER, since a
  live model may paraphrase correctly and be penalised by strict WER. Likely
  needed; decide after seeing pilot responses.
