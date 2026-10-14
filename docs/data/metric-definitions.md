# LCF — Live-model Content Fidelity

**Status:** draft v0.2, 2026-08-07. Not yet reviewed by supervisors.
**Purpose:** the project's primary contribution. Defines what we measure, how, and why it resists gaming.
**Changed in v0.2:** output modality (audio or text into the judge) is now an explicit, recorded property of every trial, not an unstated assumption of audio (§3.5). Follows the amendment to spec note 10.

---

## 1. What this metric is for

Conventional TSE is scored on how good the separated audio *is*: signal quality (SI-SDR), perceptual quality (DNSMOS, PESQ) or transcribability by an offline ASR (WER/TER). These can diverge from what we care about:

> Conventional TSE can improve offline transcription accuracy while making the audio *harder* for a live speech-to-speech model to understand.

Plausible mechanism: masking-based extraction leaves spectral holes, musical noise and phase artefacts. An ASR model trained on clean read speech tolerates these, as they barely move its phonetic decision boundaries. A live speech-to-speech model hears through a learned audio encoder over a much wider distribution, and appears more sensitive to processing artefacts than to the interfering speech removed. This is a hypothesis; testing it is result #1 of the project.

So the metric measures **what a live model actually recovered**, and is blind to how nice the audio sounded.

---

## 2. Trial definition

A trial is a tuple:

| Element | Symbol | Notes |
|---|---|---|
| Mixture audio | `x` | Target + ≥1 interfering speaker + noise/reverb |
| Enrollment audio | `e` | ≥5 s of the target speaking alone, from a different recording |
| Target ground-truth text | `t` | Exact verbatim text of what the target said in `x` |
| Interferer ground-truth text | `d` | Exact verbatim text of what the interferer(s) said |

`d` is required: without it there is no contamination score, and the metric is one-sided and trivially gameable.

The system under test produces `ŝ = f(x, e)`, streaming, within the latency budget. `ŝ` is **audio in the primary condition and text in the reference condition** (§3.5). Modality is recorded on every trial.

---

## 3. Scoring procedure

1. Present `ŝ` to a live speech-to-speech model under a **fixed, published prompt** asking it to report what it heard: through its audio input in the primary condition, its text input in the reference condition (§3.5). The wording differs across modalities only as grammar needs ("what you heard" vs "what you received"); both are published verbatim.
2. If the model responds in audio, transcribe it with a **fixed ASR system**, identical across all conditions and recorded in the protocol. It is part of the measuring instrument, so changing it invalidates comparisons.
3. Score the response text `r` against `t` and `d`.

### 3.1 Primary score — LCF-WER

Word error rate of `r` against `t`. **Lower is better.** The headline number: how much of what the target said did the live model recover?

Normalisation (casing, punctuation, numbers, disfluencies) is fixed and published.

**The normaliser, pinned (B5, decisions-m0.md 2026-08-13).** Whisper's `EnglishTextNormalizer` from `whisper-normalizer==0.1.15`, defined in Appendix C of Radford et al. (2022) and borrowed unchanged. It lower-cases, strips punctuation, expands contractions (*don't* to *do not*) and standardises spelled-out numbers and dates. **Applied identically to both sides of every comparison** and never adjusted per system, which would flatter a model without improving it. It is part of the measuring instrument, so changing it invalidates every previously reported number.

**Trials where the target never speaks are excluded, not scored as zero (B4, decisions-m0.md 2026-08-13).** With no reference text the rate is 0/0, undefined. Scoring them 0 % would reward saying nothing and dilute the headline with trials the metric cannot judge. They are counted and reported separately as an invented-speech row: how many words came out where silence was correct. Measured 2026-08-31, `eval_public`: **95.1 % of such trials produce speech from the unprocessed mixture, against 0.0 % from the clean target.**

**A non-response counts as all-deletions, deliberately.** If the judge reports nothing when the target did speak, every reference word is a deletion and the trial scores ~100 %, so LCF-WER already punishes a muting extractor. FR's empty-response count (§3.3) therefore protects ICR rather than catching the mute (NRR's job until 2026-09-03).

**A speech gate decides, before any listener, whether a clip contains speech** (`src/live_model_metric/speech_gate.py`, added 2026-09-02). A clip with no speech is answered locally as the empty hypothesis and never reaches the judge.

*Why.* Measured 2026-09-02: given digital silence (RMS exactly 0), `gemini-3.7-flash` does **not** report nothing. It fabricates 17–42 words of fluent prose (once in French) under all three prompt variants tried. So the all-deletions rule above described an event that never occurred, and LCF-WER did **not** punish a muting extractor. The gate is a repair, not a new scoring rule. `decisions-m4.md` 2026-09-02.

*What decides what.* For the rendered anchors, construction decides: `target.wav` has speech only on `both`/`target_only`, `interferer.wav` only on `both`/`interferer_only`, and `mixture.wav` on everything except `noise_only`. No detector runs. `estimate.wav`, the only clip not fixed by construction, is decided by Silero VAD 6.2.1 (pinned under B2) at a threshold of 0.10 s of detected speech.

*The gate is part of the measuring instrument*, like the normaliser and the prompt: **changing it invalidates every previously reported number.** It is applied identically to the judge and the offline ASR; otherwise any difference between them on a speech-free clip would measure the gate, not the listeners.

Every gate decision, block or pass, is logged to `experiments/results/speech_gate.csv`, so the denominator is recoverable. **A gate firing on a target-present trial is a finding, not a measurement error**: the system destroyed the speech, and it must be reported.

### 3.2 Secondary score — ICR (Interferer Content Rate)

Fraction of trials where `r` contains content attributable to `d` rather than `t`. **Lower is better.** Computed as thresholded content-word overlap between `r` and `d`, excluding words also in `t`.

**The threshold is `count >= 2`**: a trial fires when at least two interferer-exclusive content words appear in the response. One shared content word is coincidence at the rate English repeats nouns; two is signal. The alternative, a *fraction* of the available interferer-exclusive words, varies with the interferer's utterance length (a trial property, not a system one), so it is the worse primary. Both are computed; a headline ICR means `count >= 2` unless stated.

**Sensitivity.** Swept 2026-08-31 on `eval_public` with an offline ASR stand-in (`faster-whisper small.en`), not a live-model result. Bracketed figures are each k's eligible count.

| set | ICR@1 | ICR@2 | ICR@3 | ICR@5 |
|---|---|---|---|---|
| present (`both`), floor | 57.0 (230) | **52.0** (229) | 49.3 (223) | 42.9 (212) |
| present (`both`), ceiling | 0.0 | **0.0** | 0.0 | 0.0 |
| absent (`interferer_only`), floor | 100.0 (123) | **100.0** (123) | 99.2 (123) | 97.5 (120) |
| absent, ceiling | 0.0 | **0.0** | 0.0 | 0.0 |

**The choice of k does not matter.** The floor moves from 57.0 to 42.9 across the range while the ceiling stays at 0.0, so every k separates the anchors completely and no conclusion here changes for k = 1, 3 or 5. This meets the sensitivity requirement.

**Exclusion rule.** A trial where the interferer said nothing the target did not also say carries no evidence either way, so it is **excluded from ICR, not scored as clean** (which would pull the rate toward zero). The exclusion count is reported with the rate. On `eval_public` it is currently **0 in every stratum**, but the rule stays because a shorter or more repetitive interferer would trigger it.

**The floor row's ICR is partly set by construction, and must be quoted that way.** The judge never sees the enrolment, so on an unprocessed two-speaker mixture it cannot know which speaker is the target and picks one, pushing the floor's ICR toward a coin flip. This is correct behaviour and the finding (doing nothing often gets the wrong speaker), but it is a property of the task, not a listener failure. Measured at k=2 on the `eval_public` floor:

| which voice is louder | n | ICR@2 |
|---|---|---|
| interferer | 48 | **87.5 %** |
| balanced | 21 | 81.0 % |
| target | 160 | **37.5 %** |

**Consequence for the prompt (binding).** The fixed prompt must **not** tell the judge to choose a speaker (no "the clearest voice", no "the loudest speaker"). That would hand the extractor's job to the judge and move the floor row for a reason unrelated to any system under test.

**This is the score that makes the metric two-sided**, and one an offline WER-based metric structurally cannot see.

### 3.3 Secondary score — FR (Fabrication Rate)

Content words in the response that appear in **neither** speaker's script. **Lower is better.**

The third disjoint failure mode: LCF-WER measures content **recovered**, ICR content **leaked** from the other speaker, FR content **invented**. LCF-WER pools invented and leaked words as insertions, and ICR only sees words traceable to the interferer, so nothing previously isolated the invented residue.

**Two readings, and only one is safe across systems.**

| reading | definition | use |
|---|---|---|
| **`invented_per_trial`** | mean invented content words per trial | **compare systems on this** |
| `FR@k` | fraction of trials with ≥ k invented content words, k=2 headline | distribution, not magnitude |
| `mean_invented_percent` | invented as a share of the response's own content words | **within** one system only |

**The percentage is length-confounded; do not rank systems on it.** It divides by the response's own length, so a terser listener scores worse for saying less, not for inventing more. Measured 2026-09-03: our baseline and WeSep invent **1.83 and 1.80** words per trial (indistinguishable) but read **10.4 % and 13.8 %**, only because WeSep's responses average 15.1 content words against 17.9.

Every trial with a non-empty response is eligible at every k, as fabrication has no cap like ICR's (the interferer's exclusive word count). An **empty response is excluded, not scored 0 % invented**, otherwise a muting system lowers its rate by saying nothing. The empty count is reported alongside (see the anti-mute note below).

**FR is an upper bound, and its ceiling is not zero.** A response word can be in neither script because the listener invented it (the only true fabrication), misheard a real word as another, or because the reference does not match what was uttered. The ceiling row (the listener given the **clean target**) calibrates how much the measure fires on perfect input, and **only the excess above it is attributable to a system**. Measured 2026-09-03, the ceiling is 0.20 words per trial. Never quote FR without it. ICR's ceiling, by contrast, really is 0.0.

Measured on `sir0_val` `both`, n=103, `gemini-3.7-flash`, prompt `d118b7d3bf30` (`experiments/results/2026-09-03-fr-sweep-sir0_val/`):

| system | invented/trial | FR@2 |
|---|---|---|
| ceiling (clean target) | 0.20 | 1.0 % |
| floor (do nothing) | 1.24 | 32.0 % |
| baseline | 1.83 | 41.7 % |
| WeSep | 1.80 | 38.8 % |

**Both extractors raise fabrication ~48 % above doing nothing, and are indistinguishable despite a 25-point WER gap.** Fabrication is therefore its own axis, not a by-product of extraction quality. The ordering is unchanged at k = 1, 2, 3 and 5, so the headline k=2 is a convention, as J3 found for ICR.

**NRR (Non-Response Rate) was removed 2026-09-03.** It read 0.0 % on every judge row: its detector is an empty response, and this judge never returns one, it invents. Its own module called it "a tripwire, not a quality measure. It will not rank two working systems." **Its one live purpose is kept**: making a muting extractor, which scores a *clean* ICR because nothing came out, still trip an alarm. FR reports the empty-response count over the same trials; state it whenever a blocked or muted trial is reported. decisions-m4.md 2026-09-03.

**The judge's own invention rate stays a separate one-off row**, measured on known speech-free audio with the gate bypassed. It characterises the *judge*; FR characterises a *system*. Measured 2026-09-02 on `gemini-3.7-flash`: 0 of 6 silent clips returned `no_speech`; 17–42 words invented per clip across three prompt variants.

### 3.4 Reported anchors — mandatory

Every results table reports these rows alongside the systems:

| Anchor | Presented to judge | Modality | Interpretation |
|---|---|---|---|
| **Floor** | Unprocessed mixture `x` | audio | Doing nothing. Any system must beat this or it is worthless. |
| **Ceiling** | Clean target only | audio | The best any extraction could achieve on this judge. |
| **Text ceiling** | Ground-truth text `t` | text | The best the text path could achieve with a perfect ASR. Bounds the reference condition, and separates "the ASR was wrong" from "the judge mishandled clean text". |

The **text floor** (an off-the-shelf ASR run on the unprocessed mixture `x`, its output handed to the judge as text) is reported whenever any text-path row is. Without it a text row has nothing to beat.

On constructed trials the ceiling is exact. On AMI it is **approximate**, from the individual headset (IHM) channel, which carries cross-talk bleed and a different channel response from the distant mixture mic. Always label it approximate. See `docs/decisions/decisions-m0.md`, 2026-08-07 data decision.

### 3.5 Output modality — audio primary, text as a reference condition

Spec note 10 allows the extractor to hand the live model **either audio or text**. Both are scored by the same question (*what did the assistant recover of what the target said?*), with LCF-WER, ICR and FR computed identically. Only the meaning of the number differs.

| | **Primary (audio)** | **Reference (text)** |
|---|---|---|
| What is built | Streaming TSE extractor, `ŝ` is a waveform | Same extractor + off-the-shelf streaming ASR |
| Judge input | audio | text |
| Judge's role | Listens and reports | Reads and echoes, close to a pass-through |
| What LCF-WER measures | Judge's recovery from processed audio | Mostly the front-end ASR's WER |
| Optimised for? | Yes, this is the build target | **No.** Measured and reported only |

**Cut 2026-09-03: the text row is not scored, it is excluded on latency.** The table above still defines the condition. Extraction alone costs 162 ms mean / 176 ms p99 against a 200-300 ms budget, and the off-the-shelf ASR is non-streaming, so it must also endpoint before it can decode. The text path cannot meet the latency constraint, and scoring it would invite "text wins, why not use text?" for a system that cannot be deployed under the spec. **The modality question is answered with latency, which is measured, instead of with LCF, which would need a caveat to be read correctly.** decisions-m4.md 2026-09-03.

Kept for the write-up, as the reasons a text row would need heavy caveats even if affordable:

**A text-path score is not evidence about the judge's listening.** In the text condition the judge gets the answer in the modality it is best at, so the score mostly reflects the cascade's ASR. Cross-modality comparison is valid about *pipelines* ("this way of getting content to the assistant recovers more of it"), not about *the judge* ("the model understands text better than speech"). Every table mixing modalities must say so.

**The metric is blind to what the text path throws away.** LCF-WER, ICR and FR are all lexical. Prosody, emphasis, hesitation, emotion and speaker identity vanish at the ASR boundary, yet a speech-to-speech model uses them to interpret a turn and shape its spoken reply. A text row can therefore score well while being a worse assistant experience. This known blind spot is stated rather than fixed, and is the strongest argument against reading the text row as an upper bound. If the text row wins decisively, the next step is a paralinguistic probe (see §7), not a switch of build target.

Latency is reported per modality, and the modalities are not comparable without it: the text path adds ASR decoding plus endpointing to extraction. See `docs/decisions/decisions-m0.md`, 2026-08-07 output-modality decision.

---

## 4. Why this resists gaming

The REAL-TSE Challenge is the cautionary tale. Teams over-optimised DNSMOS-OVRL (once with adversarial waveform perturbations) until its correlation with human MOS on Track 1 was essentially zero (LCC +0.003), and the organisers had to swap the official metric after the fact.

Four properties, built in from the start:

**Semantic, discrete and non-differentiable.** The score depends on whether the right words came back, not on a smooth perceptual predictor, so there is no gradient to hill-climb with waveform perturbations.

**Two-sided.** Suppress everything and the empty-response count blows up (FR §3.3, formerly NRR). Pass everything through and ICR blows up. Only real extraction moves LCF-WER without moving the other two, which is why all three are always reported together.

**The judge is held out from training.** Enforced by the different-model-family rule in `docs/decisions/decisions-m0.md`. The judge is also never used as a training-data filter.

**Anchored.** The clean-target ceiling bounds what is achievable on this judge, so an implausible score is visible immediately rather than mistaken for a breakthrough.

**Modality must be declared.** Sending text instead sidesteps the audio problem the metric measures. Spec note 10 permits it, but it must be declared, scored against the text floor and text ceiling, and reported with its own latency. An undeclared text path would make two systems' numbers silently incomparable.

Additionally, keep a **private trial split**: publish the protocol, construction code and a public split, and hold back a private split for the headline numbers, to prevent overfitting to specific utterances.

---

## 5. Methodological requirements

These are not optional; they make the benchmark a scientific instrument rather than an anecdote.

**Pin and date everything.** Record the exact judge model ID, prompt, response-transcription ASR, **input modality** and run date for every number, so drift can be checked if suspected. Amended 2026-09-07: cross-date comparison is treated as valid within this project's timeframe. See `decisions/decisions-m4.md` 2026-09-07.

**Two ASRs, kept separate.** The text condition adds a *front-end* ASR (part of the system, inside the latency budget). The judge harness already has a *response-transcription* ASR (part of the instrument, outside the budget). Log them separately, or the system looks like it contains the instrument. They may be the same checkpoint, but say so, since a shared error profile could flatter the text condition.

**At least one open-weight judge: waived 2026-09-03.** The intent stands: a benchmark reproducible only on a paid, silently changing API has a short shelf life. The anchor (Qwen3-Omni, chosen for encoder independence from our reference ASR) was cut for time.

The metric is judge-agnostic, so an open-weight judge is a configuration change, not a redesign. Every result records model ID, prompt hash, modality and run date, with raw responses kept, so published numbers are auditable but not re-runnable without API access. If the work continues, add one open-weight judge over the same trials first. decisions-m4.md 2026-09-03.

**Report variance.** Live models are stochastic. Run each trial `k` times (k ≥ 3) and report the mean with confidence intervals. A difference smaller than the run-to-run spread is not a difference.

**Fix decoding parameters** where the API exposes them; record them where it doesn't.

**Budget the API cost before building the trial set.** Cost per trial × number of trials × number of systems × number of judges × k repeats. Size the trial set to the budget, on a spreadsheet, before writing the harness.

---

## 6. Relationship to existing metrics

LCF-WER/ICR/FR are the primary result. Conventional metrics (SI-SDR, the SIR/SAR decomposition, **DNSMOS-P.835 and DNSMOS-P.808** and offline ASR WER) are reported alongside **specifically to show where they diverge.**

**Amended 2026-09-01: report both P.835 and P.808.** The original named P.808 alone, losing the point of §4; the first amendment named P.835 alone, losing the field's current convention. The history:

| variant | role in the REAL-TSE story |
|---|---|
| **DNSMOS-P.835** (Reddy et al., 2022), returns `SIG`, `BAK`, `OVRL` | **`OVRL` is the score that got gamed** |
| **DNSMOS-P.808** (Reddy et al., 2021), returns one score | **the replacement the organisers switched to** |

Reporting both lets the write-up say: *here is the metric that was gamed, and here is the metric that replaced it, measured on the same audio.* Neither alone supports that, and the second model costs one extra inference pass.

P.835 also returns `SIG` and `BAK`, which map onto the signal-domain SIR/SAR split (`BAK` for interference removed, `SIG` for artefacts introduced), giving a perceptual and a signal decomposition of the same trade.

Three rules fixed with it:

- **Absent trials are excluded.** "Speech quality" is undefined when nobody speaks, so they are dropped, as from LCF-WER (B4).
- **Scores are averaged over segments.** DNSMOS works on windows of about 9 s and trials run 15–20 s, so the per-trial score is the mean over segments.
- **The ONNX model is pinned and snapshotted**, with its hash recorded. DNSMOS is a learned model and can drift between releases, the same risk the normaliser and stopword list are pinned against.

**DNSMOS is reported as the exhibit, not as a metric we trust.** It is the only score here with a gradient, so the only one optimisation can attack (§4). The write-up must say it is included to be disagreed with, or a reader will ask why we report a metric §4 criticises. The divergence is a result: a system that improves offline WER while worsening LCF-WER is the empirical justification for the metric.

Prior art to position against, not duplicate:

- **PS4** (Ning et al., arXiv:2607.08111): proxy-supervised training with differentiable ASR/speaker/VAD/quality objectives, the closest work on optimising TSE for a downstream model. Difference: PS4's proxies are also its evaluation; we hold the judge out, and our judge is a live speech-to-speech model, not a cascaded ASR.
- **Ma et al.** (arXiv:2501.14477): joint generative TSE and target-speaker ASR for intelligibility. It addresses the transcribable-but-worse phenomenon directly and is the closest prior art to the §3.5 text reference condition: a *properly built* text path, unlike our deliberately cheap extractor plus off-the-shelf ASR cascade. Cite it with the text row so the row is not mistaken for a serious attempt at the text path.
- **REAL-TSE TER**: offline Zipformer transcription, the metric we argue is insufficient for the live-model use case.

---

## 7. Open questions

- ~~Which open-weight speech-to-speech model is the reproducible anchor.~~ **Cut 2026-09-03 on schedule.** Qwen3-Omni was selected but not run; §5 records the waiver and resulting limitation.
- Prompt sensitivity: how much does the "report what you heard" wording move the scores? Needs a small ablation. If it dominates, the metric is fragile and needs redesign.
- Whether ICR's overlap threshold is stable across judges or needs per-judge calibration (which would weaken cross-judge comparison).
- Whether to add a semantic-equivalence score beside LCF-WER, since a live model may paraphrase correctly and be penalised by strict WER. Likely needed; decide after seeing pilot responses.
- **Whether the text reference condition beats the audio path**, and under what conditions (SNR, overlap ratio, device mismatch). If text wins everywhere, the question becomes *why* the audio path cannot close the gap. That answer is the thesis's contribution, not a reason to change build target.
- **Whether a lexical metric is enough**, given the text path discards all paralinguistic content and LCF cannot see the loss (§3.5). A small probe (trials whose meaning depends on emphasis, question intonation or hesitation) would test it cheaply. Only worth building if the text row wins.
- Whether the judge's text and audio front-ends normalise numbers, dates and disfluencies differently. If so, the shared normalisation in §3.1 does unequal work across modalities and must be re-checked before any cross-modality claim.
