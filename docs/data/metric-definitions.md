# LCF — Live-model Content Fidelity

**Status:** draft v0.2, 2026-08-07. Not yet reviewed by supervisors.
**Purpose:** the project's primary contribution. Defines what we measure,
how, and why it resists gaming.
**Changed in v0.2:** output modality (audio vs text into the judge) is now an
explicit, recorded property of every trial rather than an unstated assumption
that the judge is handed audio — see §3.5. Follows the amendment to spec
note 10.

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
budget. `ŝ` is **audio in the primary condition and text in the reference
condition** — see §3.5. Modality is a recorded property of every trial, never
an implicit assumption.

---

## 3. Scoring procedure

1. Present `ŝ` to a live speech-to-speech model under a **fixed, published
   prompt** asking it to report what it heard — via the model's audio input
   in the primary condition, via its text input in the reference condition
   (§3.5). The prompt wording is held fixed across modalities except for the
   minimum change needed to make it grammatical ("what you heard" vs "what
   you received"); both wordings are published verbatim.
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

**The normaliser, pinned (B5, decisions-m0.md 2026-08-13).** Whisper's
`EnglishTextNormalizer`, from `whisper-normalizer==0.1.15`, defined in Appendix C
of Radford et al. (2022) and borrowed unchanged. It lower-cases, strips
punctuation, expands contractions so *don't* becomes *do not*, and standardises
spelled-out numbers and dates. **Applied identically to both sides of every
comparison**, and never adjusted per system — adjusting it per system would
flatter a model without improving it. It is a component of the measuring
instrument, not preprocessing, and a change to it invalidates every previously
reported number.

**Trials where the target never speaks are EXCLUDED, not scored as zero (B4,
decisions-m0.md 2026-08-13).** With no reference text the rate is 0/0 —
undefined, not perfect. Folding such trials in as 0 % would reward a system for
saying nothing when nothing was said, and dilute the headline with trials the
metric cannot judge. They are counted, reported separately, and measured instead
by their own invented-speech row: on a trial where the target is silent the
correct output is silence, so what matters is how many words came out. Measured
2026-08-31, `eval_public`: **95.1 % of such trials produce speech from the
unprocessed mixture, against 0.0 % from the clean target.**

**A non-response counts as all-deletions, deliberately.** If the judge reports
nothing on a trial where the target did speak, every reference word is a deletion
and the trial scores ~100 %. That is the honest reading — a listener that said
nothing recovered nothing — and it means LCF-WER already punishes a muting
extractor. NRR (§3.3) therefore exists to protect ICR, not to catch the mute.

**A SPEECH GATE decides, before any listener, whether a clip contains speech at
all** (`src/live_model_metric/speech_gate.py`, added 2026-09-02). A clip with no
speech is answered locally as the empty hypothesis and never reaches the judge.

*Why the paragraph above needed it.* Measured 2026-09-02: handed digital silence
(RMS exactly 0), `gemini-3.7-flash` does **not** report nothing — it fabricates
17–42 words of fluent prose, once in French, and did so under all three prompt
variants tried. So "a non-response counts as all-deletions" was describing an
event that never occurred, and LCF-WER did **not** in fact punish a muting
extractor. The gate makes this paragraph true rather than aspirational: it is a
repair, not a new scoring rule. `decisions-m4.md` 2026-09-02.

*What decides what.* For the rendered anchors the answer follows from
CONSTRUCTION — `target.wav` carries speech only on `both`/`target_only`,
`interferer.wav` only on `both`/`interferer_only`, and `mixture.wav` on
everything except `noise_only` — so no detector runs and no moving part is
added. `estimate.wav` is the only clip whose content is not fixed by
construction, and it is decided by Silero VAD 6.2.1 (pinned under B2) at a
threshold of 0.10 s of detected speech.

*The gate is a component of the measuring instrument*, exactly like the
normaliser and the prompt: **a change to it invalidates every previously
reported number.** It is applied identically to the judge and to the offline
ASR — gate one listener and not the other and every difference between them on
a speech-free clip measures the gate rather than the listeners.

Every gate decision is logged to `experiments/results/speech_gate.csv`, blocks
and passes alike, so the denominator is recoverable. **A gate firing on a
target-present trial is a finding, not a measurement error** — it means the
system destroyed the speech — and must be reported as such.

### 3.2 Secondary score — ICR (Interferer Content Rate)

Fraction of trials where `r` contains content attributable to `d` rather
than `t`. **Lower is better.**

Computed as content-word overlap between `r` and `d`, excluding words that
also appear in `t`, thresholded.

**The threshold is `count >= 2`** — a trial fires when at least two
interferer-exclusive content words appear in the response. One shared content
word is coincidence at the rate English repeats nouns; two is signal. The
alternative rule, a *fraction* of the interferer-exclusive words available,
varies with the interferer's utterance length — a property of the trial, not of
the system — which makes it the worse primary. Both are computed; `count >= 2`
is what a headline ICR means unless stated otherwise.

**Sensitivity, as this section requires.** Swept 2026-08-31 on `eval_public`,
offline ASR stand-in (`faster-whisper small.en`), not a live-model result.
Bracketed figures are that k's own eligible count.

| set | ICR@1 | ICR@2 | ICR@3 | ICR@5 |
|---|---|---|---|---|
| present (`both`), floor | 57.0 (230) | **52.0** (229) | 49.3 (223) | 42.9 (212) |
| present (`both`), ceiling | 0.0 | **0.0** | 0.0 | 0.0 |
| absent (`interferer_only`), floor | 100.0 (123) | **100.0** (123) | 99.2 (123) | 97.5 (120) |
| absent, ceiling | 0.0 | **0.0** | 0.0 | 0.0 |

**The choice of k is not load-bearing.** The floor moves 57.0 -> 42.9 across the
whole range while the ceiling stays at 0.0, so every k separates the two anchors
completely and no conclusion in this document changes if k is 1, 3 or 5. The
sensitivity requirement is met by that fact, not by defending 2 over 3.

**Exclusion rule.** A trial where the interferer said nothing the target did not
also say carries no evidence of contamination either way. Such trials are
**excluded from ICR, not scored as clean** — scoring them clean would pull the
rate toward zero with trials that could never have fired. The exclusion count is
reported alongside the rate. On `eval_public` it is currently **0 in every
stratum**, so the rule is correct but has not yet had occasion to fire; it must
stay in the definition because a shorter or more repetitive interferer would
trigger it.

**The floor row's ICR is partly set by construction, and must be quoted that
way.** The judge never sees the enrolment, so on an unprocessed two-speaker
mixture it cannot know which speaker is the target and will pick one. The floor's
ICR therefore tends toward a coin flip. This is correct behaviour and it *is* the
finding — doing nothing gets you the wrong speaker much of the time — but it is a
property of the task, not a failure of the listener, and it must be stated when
the floor is quoted rather than discovered in a results table. Measured at k=2 on
the `eval_public` floor:

| which voice is louder | n | ICR@2 |
|---|---|---|
| interferer | 48 | **87.5 %** |
| balanced | 21 | 81.0 % |
| target | 160 | **37.5 %** |

**Consequence for the prompt, and it is binding.** The fixed prompt must **not**
instruct the judge to choose a speaker — no "the clearest voice", no "the loudest
speaker". Such an instruction hands the extractor's job to the judge and converts
a measurement into an instruction, and it would move the floor row for a reason
that has nothing to do with any system under test.

**This is the score that makes the metric two-sided**, and it is the one an
offline WER-based metric structurally cannot see.

### 3.3 Secondary score — NRR (Non-Response Rate)

Fraction of trials where the model declines, reports hearing nothing,
returns silence, or produces a refusal. **Lower is better.**

Catches the degenerate "output silence" strategy, which would otherwise
score perfectly on ICR.

**It catches a mute only because the speech gate exists** (§3.1, added
2026-09-02). A muting extractor produces audio with no detectable speech, the
gate answers it as the empty hypothesis, and NRR sees the non-response it is
built to detect. Without the gate the judge invents words on silence, NRR reads
near-zero, and the degenerate strategy passes. This is the mechanism by which
the metric set stays two-sided (§4) — a blocked trial scores a **clean ICR**,
and it is NRR that raises the alarm on the same trial. State that pairing
explicitly whenever a blocked trial is reported; the hole is obvious to a
reviewer otherwise.

**NRR detects a judge that DECLINES, not one that CONFABULATES.** This is a
measured limitation, not a theoretical one. `decisions-m4.md` (2026-08-31) gives
NRR's strongest purpose as detecting judge malfunction — "it declined on perfect
input". The judge measured on 2026-09-02 does not decline; it invents. NRR's
detector is an empty response, and a confabulating judge is never empty, so this
purpose is **not** served by NRR and must not be claimed for it.

**The judge's invention rate is therefore its own row**, measured against
known speech-free audio with the gate deliberately bypassed. It characterises
the judge, is run once, and is not part of the per-system protocol. Measured
2026-09-02 on `gemini-3.7-flash`: **0 of 6 silent clips returned `no_speech`;
17–42 words invented per clip across three prompt variants.**

### 3.4 Reported anchors — mandatory

Every results table reports these rows alongside the systems:

| Anchor | Presented to judge | Modality | Interpretation |
|---|---|---|---|
| **Floor** | Unprocessed mixture `x` | audio | Doing nothing. Any system must beat this or it is worthless. |
| **Ceiling** | Clean target only | audio | The best any extraction could achieve on this judge. |
| **Text ceiling** | Ground-truth text `t` | text | The best the text path could achieve with a perfect ASR. Bounds the reference condition, and separates "the ASR was wrong" from "the judge mishandled clean text". |

The **text floor** — an off-the-shelf ASR run on the unprocessed mixture `x`,
its output handed to the judge as text — is reported whenever any text-path
row is reported. Without it a text row has nothing to be better than.

On constructed trials the ceiling is exact. On AMI it is **approximate**,
computed from the individual headset (IHM) channel, which carries cross-talk
bleed and a different channel response from the distant mixture mic. Always
label it as approximate. See `docs/decisions/decisions-m0.md`, 2026-08-07 data decision.

### 3.5 Output modality — audio primary, text as a reference condition

Spec note 10 allows the extractor to hand the live model **either audio or
text**. The metric is defined so that both are scored by the same end-to-end
question — *what did the assistant recover of what the target said?* — and
LCF-WER, ICR and NRR are computed identically in both cases. What differs is
what the number means.

| | **Primary (audio)** | **Reference (text)** |
|---|---|---|
| What is built | Streaming TSE extractor, `ŝ` is a waveform | Same extractor + off-the-shelf streaming ASR |
| Judge input | audio | text |
| Judge's role | Listens and reports | Reads and echoes — close to a pass-through |
| What LCF-WER measures | Judge's recovery from processed audio | Mostly the front-end ASR's WER |
| Optimised for? | Yes — this is the build target | **No.** Measured and reported only |

**CUT 2026-09-03 — the text row is not scored, it is excluded on latency.** The
table above stands as the definition, but no text LCF number will be produced.
Extraction alone costs 162 ms mean / 176 ms p99 against a 200-300 ms budget, and
the off-the-shelf ASR is non-streaming: it must endpoint before it can decode, on
top of that. The text path therefore cannot meet the project's own latency
constraint, and scoring it would report content fidelity for a system that cannot
be deployed under the spec — inviting "text wins, why not use text?" when the
answer is that it does not run in time. **The modality question is answered with
latency, which is measured, instead of with LCF, which would need a caveat to be
read correctly.** decisions-m4.md 2026-09-03.

The two consequences below are retained: they are why a text row would have needed
heavy caveating even if it had been affordable, and they belong in the write-up as
the reasoning behind the exclusion.

Two consequences that must not be lost:

**A text-path score is not evidence about the judge's listening.** In the
text condition the judge is handed the answer in the modality it is best at,
so the score largely reflects the cascade's ASR. Cross-modality comparison is
legitimate as a statement about *pipelines* ("this way of getting content to
the assistant recovers more of it"), and illegitimate as a statement about
*the judge* ("the model understands text better than speech"). Every table
that mixes modalities must carry that caveat.

**The metric is blind to what the text path throws away.** LCF-WER, ICR and
NRR are all lexical. Prosody, emphasis, hesitation, emotion and speaker
identity vanish at the ASR boundary, and a speech-to-speech model uses them
both to interpret a turn and to shape its spoken reply. A text row can
therefore score well while being a worse assistant experience, and the metric
will not show it. This is a known blind spot, stated rather than fixed — it
is the strongest single argument against reading the text row as an upper
bound. If the text row wins decisively, the honest next step is a
paralinguistic probe (see §7), not a switch of build target.

Latency is reported per modality and is not comparable across them without
it: the text path pays for ASR decoding plus endpointing on top of
extraction. See `docs/decisions/decisions-m0.md`, 2026-08-07 output-modality decision.

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
family rule in `docs/decisions/decisions-m0.md`. The judge is also never used as a
training-data filter.

**Anchored.** The clean-target ceiling bounds what is achievable on this
judge, so an implausible score is visible immediately rather than being
mistaken for a breakthrough.

**Modality must be declared.** Converting to text and sending that instead is
an obvious way to sidestep the audio problem the metric exists to measure. It
is not forbidden — spec note 10 permits it — but it must be declared, scored
against the text floor and text ceiling, and reported with its own latency.
An undeclared text path would make two systems' numbers silently
incomparable, so this rule is what keeps the leaderboard meaningful rather
than what keeps anyone honest.

Additionally: keep a **private trial split**. Publish the protocol, the
construction code and a public split; hold back a private split for the
headline numbers. Prevents overfitting to specific utterances.

---

## 5. Methodological requirements

These are not optional — they are what makes the benchmark a scientific
instrument rather than an anecdote.

**Pin and date everything.** Record the exact judge model ID, the exact
prompt, the exact response-transcription ASR, the **input modality**, and the
run date for every number. Closed live models change silently. Cross-date
comparisons are invalid unless re-run.

**Two ASRs, kept separate.** The text condition introduces a *front-end* ASR
(part of the system under test, inside the latency budget). The judge harness
already has a *response-transcription* ASR (part of the measuring
instrument, outside the budget). These are different components with
different roles and must be logged separately — conflating them makes the
system look like it contains the instrument. They may be the same checkpoint,
but that must be stated, since a shared error profile could flatter the text
condition.

**At least one open-weight judge — waived 2026-09-03.** The intent stands: a
benchmark reproducible only on a paid API that changes silently has a short shelf
life. The anchor (Qwen3-Omni, chosen for encoder independence from our reference
ASR) was cut for time.

The metric is implemented judge-agnostically, so running this protocol against an
open-weight model is a configuration change rather than a redesign, and every
result records model ID, prompt hash, modality and run date with the raw responses
retained. Published numbers are therefore auditable but not re-runnable without
API access. Adding one open-weight judge over the same trials is the first thing
to do if the work continues. decisions-m4.md 2026-09-03.

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
alongside — SI-SDR, the SIR/SAR decomposition, **DNSMOS-P.835 and DNSMOS-P.808**
and offline ASR WER — **specifically to show where they diverge.**

**Amended 2026-09-01: report BOTH P.835 and P.808.** The original wording named
P.808 alone, which loses the point of §4, and the first amendment named P.835
alone, which loses the field's current convention. The history is why:

| variant | role in the REAL-TSE story |
|---|---|
| **DNSMOS-P.835** (Reddy et al., 2022) — returns `SIG`, `BAK`, `OVRL` | **`OVRL` is the score that got gamed** |
| **DNSMOS-P.808** (Reddy et al., 2021) — returns one score | **the replacement the organisers switched to** |

Reporting both lets the write-up say plainly: *here is the metric that was gamed,
and here is the metric that replaced it, measured on the same audio.* Neither
alone supports that sentence, and the second model costs one extra inference pass.

P.835 additionally returns `SIG` and `BAK` separately, which map onto the
signal-domain SIR/SAR split — `BAK` for interference removed, `SIG` for artefacts
introduced — giving a perceptual and a signal decomposition of the same trade.

Three rules fixed with it:

- **Absent trials are excluded.** "Speech quality" is undefined when nobody
  speaks, so those trials are dropped as they are from LCF-WER (B4), not scored.
- **Scores are averaged over segments.** DNSMOS operates on windows of about 9 s
  and trials run 15–20 s, so the per-trial score is the mean over its segments.
- **The ONNX model is pinned and snapshotted**, with its hash recorded. DNSMOS is
  a learned model and can drift between releases — the same failure mode the
  normaliser and the stopword list are pinned against.

**DNSMOS is reported as the exhibit, not as a metric we trust.** It is the only
score here with a gradient, and therefore the only one that can be attacked by
optimisation — which is precisely what §4 is about. It is included to be
disagreed with, and the write-up must say so, or a reader will reasonably ask why
we report a metric we spend §4 criticising. The divergence is a result, not a nuisance:
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
  phenomenon, and is the closest prior art to the text reference condition of
  §3.5: it is what a *properly built* text path would look like, versus our
  deliberately cheap extractor→off-the-shelf-ASR cascade. Cite it when
  reporting the text row, so the row is not mistaken for a serious attempt at
  the text path.
- **REAL-TSE TER** — offline Zipformer transcription. This is the metric we
  are arguing is insufficient for the live-model use case.

---

## 7. Open questions

- ~~Which open-weight speech-to-speech model is the reproducible anchor.~~
  **CUT 2026-09-03 on schedule.** Qwen3-Omni was selected and not run; §5
  records the waiver and the resulting limitation.
- Prompt sensitivity — how much does the "report what you heard" wording
  move the scores? Needs a small ablation; if it dominates, the metric is
  fragile and needs redesign.
- Whether ICR's overlap threshold is stable across judges, or needs
  per-judge calibration (which would weaken cross-judge comparison).
- Whether to add a semantic-equivalence score alongside LCF-WER, since a
  live model may paraphrase correctly and be penalised by strict WER. Likely
  needed; decide after seeing pilot responses.
- **Whether the text reference condition beats the audio path**, and under
  what conditions (SNR, overlap ratio, device mismatch). If text wins
  everywhere, the interesting question becomes *why* the audio path cannot
  close the gap — the answer is the thesis's contribution, not a reason to
  change build target.
- **Whether a lexical metric is enough**, given that the text path discards
  all paralinguistic content and LCF cannot see the loss (§3.5). A small
  probe — trials where the target's meaning depends on emphasis, question
  intonation or hesitation — would test it cheaply. Only worth building if
  the text row wins.
- Whether the judge's text and audio front-ends normalise numbers, dates and
  disfluencies differently. If they do, the shared normalisation in §3.1 is
  doing unequal work across modalities and must be re-checked before any
  cross-modality claim.
