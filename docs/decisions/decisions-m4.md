# Decision Log — M4 (the metric is computable end to end)

Covers **LCF (Live-model Content Fidelity)**, the primary contribution: the three scores (LCF-WER, ICR, NRR), the judge, the fixed prompt, the anchors and the audio/text modality split. `docs/data/metric-definitions.md` defines them; this log records what it does not fix.

Created 2026-08-31 when the log was split by milestone. See `milestones.md` M4.

**The conventional metrics are not here.** SI-SDR, DNSMOS, offline-WER and the offline ASR behind them are M3 (`decisions-m3.md`). Keeping them apart stops the divergence result of `metric-definitions.md` 6 from being one instrument measuring itself.

**Standing constraint (CLAUDE.md), amended 2026-09-15.** The judge is no longer held out from training (see 2026-09-15). Still binding: `sir0_privval` and `eval_private` are never touched by training, and every judge result records the exact model ID, prompt, input modality and run date, because closed models change silently.

---

## 2026-09-15 — The judge is no longer held out. Family separation withdrawn

**Decision: Gemini may be used inside the training loop** as a teacher, reward signal, data filter or checkpoint-selection criterion. The supervisor approved over-optimising for a single model family. This reverses the CLAUDE.md rule "never appears in the training loop" and the 2026-08-31 restatement of it further down this file.

**Cost.** The primary contribution was a *gaming-resistant* metric, which relied on the judge never seeing training. That no longer holds. A system trained against `gemini-3.7-flash` and scored on `gemini-3.7-flash` measures how well we fitted one listener, not how intelligible the audio is. **Every claim must now say *optimised for Gemini*, never *generalises to live models*.**

**Replacement: the holdout moves from the model to the data.** `sir0_privval` (n=1421) and `eval_private` are never scored, filtered or selected on during training. A number on these trials still shows generalisation *across trials and speakers*. This is the only holdout claim left, so it is now the binding constraint.

**2026-09-24:** `sir0_privval` is no longer scored at all (`decisions-m3.md` 2026-09-24), so the selection-free number now comes from `eval_public` alone.

**Unchanged:** Gemini cannot be backpropagated through, so it can never be a loss term. Every training-time Gemini call records model ID, prompt and date, as a judge call does.

---

## 2026-09-07 — Cross-date judge comparison is treated as valid

**Decision: results scored on different dates are compared directly.** The old rule (cross-date comparison is invalid unless every system is re-scored together) is withdrawn.

**Why.** All judge scoring falls within a few weeks. Enforcing the rule would mean re-scoring every system whenever one is added (four judge passes and the free-tier daily quota each time) and invalidating every judge number already in the report. The risk of a material model change in that window does not justify that.

**Kept.** Every judge result still records the exact model ID, prompt digest, input modality and run date, so the assumption stays falsifiable.

**Concession.** If `gemini-3.7-flash` changed silently between 2026-09-02 and the last scoring run, part of a difference between systems could be the model, and nothing in the current results would show it. Affected: the anchors and the 4,976-trial system (2026-09-02), WeSep (2026-09-03), and the 9,955-trial system (2026-09-06).

**Cheap check.** The floor and ceiling anchors are fixed audio, and their 2026-09-02 responses are cached in `experiments/results/judge_responses.csv`. Re-scoring them later and comparing with the cache measures drift directly, for 206 calls. Not run as of this entry.

**Consequences.**
- `CLAUDE.md`, `docs/data/metric-definitions.md`, `docs/project-state.md` and the appendix comment in `report/appendices/appendices.tex` were amended to match.
- The 2026-09-05 entry's cost note ("cross-date comparison stays invalid unless re-run") is partly superseded.
- The thesis limitations still carry the closed-model dependency (judge numbers are not reproducible without API access). That is a separate point.

---

## 2026-09-02 — The judge hallucinates on silence. Speech-gate added; NRR caveated

**Measured.** On the clean target of an absent trial (digital silence, RMS exactly 0), `gemini-3.7-flash` returns `status=speech` and fabricates fluent prose. Three prompt variants, six silent clips, **0 of 6 returned `no_speech`**:

| prompt | sha256[:12] | invented words |
|---|---|---|
| v1 | `d118b7d3bf30` | 42, 27 |
| v2a: silence licensed as an expected outcome | `a21169a651d6` | 25, 17 |
| v2b: adds "Be very careful... Do not hallucinate" | `64d4b994a9a2` | 29, 33 |

`gemini-3.7-flash`, audio in / text out, AI Studio, prepay tier, 2026-09-02.

**It is generation from the language prior, not misheard audio.** On trial `sir0_val-42-000003`, v2b answered **in French** ("Ah c'est comme ça qu'on fait ici...") on an English-only pipeline. The three answers per clip share no content with each other or with either reference. Stronger wording did not reduce the word count monotonically: v2b, which forbids hallucination, produced more words than v2a. **Prompting does not fix this, and further prompt attempts were abandoned.**

**Not a Gemini quirk.** `small.en` does the same, more mildly: it emits "you" on silence in 8 of 8 absent trials (`decisions-m3.md`). *Investigation of Whisper ASR Hallucinations Induced by Non-Speech Audio* (arXiv:2501.11378) compares Koenecke-style VAD gating, confidence thresholding, hallucination pattern matching and LM filtering. VAD and confidence thresholding both help, but **no single method eliminates it**. WhisperX established VAD pre-processing as the standard fix and names Silero.

**Confidence thresholding is unavailable**: logprobs are missing from the Interactions API (a documented parity gap with `generateContent`), and logprob support above Gemini 2.0 is reported as unreliable. Rejected on availability, not merit.

### Decision: a speech gate in front of BOTH listeners

`src/live_model_metric/speech_gate.py`. Every clip gets a speech/no-speech verdict before any listener sees it. A no-speech verdict returns the empty hypothesis without a call.

**Why it is not a workaround.** *Did the system emit speech?* is a **signal** question and never needed a language model. The judge is needed for what was **said**, and it is excellent at that: 0.0 % on clean targets, byte-identical five times out of five.

**Anchors are decided by construction, not by VAD**, since the renderer already fixed them. Verified against measured RMS on `sir0_val`, 2026-09-02:

| condition | n | target.wav | interferer.wav | mixture.wav |
|---|---|---|---|---|
| `both` | 103 | speech | speech | speech |
| `target_only` | 47 | speech | RMS 0 | speech |
| `interferer_only` | 42 | **RMS 0** | speech | speech |
| `noise_only` | 8 | RMS 0 | RMS 0 | **noise, no speech** |

`noise_only` was not in the original silence analysis and is the more realistic probe: real energy, no speech.

**`estimate.wav` is the only clip the VAD decides**, as its content is not fixed by construction. Collapse-to-silence is a measured failure mode of this model (2026-08-25), so this is where it matters. Silero VAD 6.2.1, already pinned under B2. Threshold 0.10 s of detected speech, about one short syllable.

**Symmetry is mandatory.** The same rule gates the judge **and** the offline ASR. Otherwise every judge-vs-ASR difference on a speech-free clip measures the gate. `gated()` makes symmetric use the default. It runs at scoring time for the ASR (transcripts already cached) and at call time for the judge (which also saves money).

**Verified on the 24 v1-prompt answers, 2026-09-02.** 2 of 24 clips blocked (the two silent targets): judge 42 and 27 words down to 0, `small.en` 1 to 0. Same block counts for both listeners. All 20 `both` clips passed untouched.

### What is logged, and why every decision and not only the blocks

`experiments/results/speech_gate.csv`: date, split, trial, clip, condition, listener, verdict, reason, detected speech seconds. Every decision is logged so the denominator is recoverable and a gate that never fires is visible.

**A gate firing on a target-present trial is a finding, not a measurement error**: the extractor destroyed the speech. It must be auditable, not folded into a score.

### The cost to NRR — CORRECTED later the same day

**The first version said "NRR cannot catch the degenerate muting extractor it was designed to catch". That is wrong**; it holds only for the ungated pipeline. Kept rather than edited away, as the metric is defended on the correction.

**With the gate, NRR works.** For a muting extractor the VAD finds no speech, the gate returns the empty hypothesis, `non_response_reason("")` returns `"silence"`, and **NRR fires.** Without the gate that non-response never occurred, which is why the mute went undetected.

**What NRR cannot see is a judge that confabulates rather than declines**, as measured here. The 2026-08-31 NRR entry gives its strongest purpose as detecting judge malfunction ("it declined on perfect input"). This judge's malfunction is invention, and NRR's detector (an empty response) never fires on it. That purpose must **not** be claimed for NRR.

**Consequence: the judge's invention rate becomes its own row**, measured once on known speech-free audio with the gate bypassed. It characterises the judge and is not part of the per-system protocol. Already measured: **0 of 6 silent clips returned `no_speech`, 17–42 words invented, across three prompts.**

**NRR is not redefined.** Rejected: widening it to "gate blocked or judge silent", which would quietly change a published score. Taken: keep 3.3 as written, and state that its mute detection *depends on the gate* and its judge-defect detection does not cover confabulation.

**State this whenever a blocked trial is reported.** A blocked trial scores a **clean ICR**: eligibility depends on the interferer's exclusive content, not the response, so an empty response is eligible with zero leakage. That hole is closed only because NRR fires on the same trial. This pairing is the two-sidedness §4 claims.

### Deviation from the spec, recorded

`metric-definitions.md` 3.1 sends every clip to the judge; it no longer does. The spec needs a §3.1 amendment naming the gate as part of the measuring instrument, so changing the gate invalidates comparisons, as for the normaliser and the prompt.

### This is a result, not only a fix

"A live model cannot be trusted to report absence" is a reportable metric-design finding, of the same shape as the DNSMOS-gaming story (`metric-definitions.md` §4). It is evidence that the metric had to be designed rather than borrowed.

**J2 stays open** until this entry is signed off. Readings 1 and 2 of the candidate gate pass decisively (ceiling 1.6 % vs the offline ASR's 5.8 %; floor 113 %, so ample dynamic range).

---

## 2026-08-31 — J1 CLOSED. The judge is audio-in / text-out, not full-duplex

**Decision: the judge is an audio-in / text-out model. Full duplex is not required.** Closes J1 in `decisions-pending.md` and unblocks J2, and so all of M4.

**Why.** LCF measures the judge's **audio encoder**, not its turn-taking. Nothing in LCF-WER, ICR or NRR reads duplexing, and `metric-definitions.md` 3.1 step 2 already makes audio *output* optional ("*if* the model responds in audio, transcribe the response"). An audio-in / text-out model has the learned encoder and wide training distribution the artefact hypothesis in 3.1 is about.

**Three gains.**

1. **The response-transcription ASR leaves the measuring instrument.** 3.1 step 2 calls it "a component of the measuring instrument, and changing it invalidates comparisons". A text response removes it. Only the *front-end* ASR of the text reference condition (3.5) remains, and it belongs to the system under test.
2. **A wider candidate field**, open and closed. Full duplex narrows the open field to roughly Moshi, whose task adherence (1.26/5 on FullDuplexBench) is below our own ceiling gate.
3. **A less noisy instrument.** A full-duplex model is tuned to *converse*, not *report*. Handed "report what you heard", it may answer or discuss the content instead of repeating it. That lands in **NRR**, where a degenerate judge looks like a degenerate extractor.

**Cost.** This deviates from the stated objective: CLAUDE.md and spec note 10 both say "live speech-to-speech model (Gemini Live and similar)". The write-up, not only this log, must say: *we used an audio-in model because the measurement depends on the judge's audio encoder rather than on its duplexing.* Without it, a reviewer can fairly say the thesis measured something else.

**Offset: a small full-duplex confirmation run.** The headline benchmark is audio-in / text-out. Alongside it, ~50 trials are scored on a genuine full-duplex live API, a few dollars at J2 prices. The claim then becomes "the ranking was confirmed on a real full-duplex model at n=50" rather than "we relaxed the objective and here is why that is acceptable". **It is part of the decision, not optional.**

**Unchanged.** The judge is still held out from training absolutely: never a proxy, never a data filter (CLAUDE.md). Modality is recorded for every trial (3.5). The open-weight reproducibility anchor is still required, for reproducibility rather than cost (J2). Every judge result records the exact model ID, prompt, modality and run date.

**Still open:** J2a (closed model for the headline) and J2b (open-weight anchor), now decided by the candidate gate rather than argument. J1 no longer blocks either.

---

## 2026-08-31 — ICR design: NLTK content words, ICR@k with per-k eligibility, absent trials on their own row

Four decisions, closing most of J3. `metric-definitions.md` 3.2 fixes the *mechanism* (thresholded content-word overlap between the response `r` and the interferer's text `d`, excluding words also in the target's text `t`) but not the content-word list, threshold or eligibility rule.

**Why ICR matters.** On the floor at n=230 (`decisions-m3.md` 2026-08-31), **insertions are 30.8 of the 57.4 points** of LCF-WER, the largest error type. WER counts an inserted word the same whether the other speaker leaked or the listener hallucinated from noise. **ICR diagnoses that largest component.**

### 1. Content words are defined by NLTK's English stopword list

**Decision: NLTK, chosen for comprehensiveness over a hand-written list.**

**The list is snapshotted into the repo**, with the NLTK version and retrieval date beside it. Otherwise:

1. `nltk.corpus.stopwords` needs `nltk.download('stopwords')`, a **network fetch at runtime**. Kaggle sessions run offline, so the metric would fail.
2. The list **changes between NLTK releases**, which would silently move every ICR number on upgrade (the failure mode the normaliser is pinned to avoid, B5).
3. Pinning is **1 line of code**, so it costs nothing.

**The stopword list must pass through the Whisper normaliser first.** NLTK's list has apostrophe forms (`don't`, `you're`, `she's`), which the normaliser expands and strips on `r` and `d`, so raw entries would never match.

**Residual risk, not fixed by NLTK.** The list lacks `one`, which was seen as an "interferer-exclusive content word" on `eval_public-42-000000`: a chance word that can fire a false positive. If false positives show up in the pilot, the fix is a documented project-specific addendum to the snapshot, never a silent edit.

### 2. The threshold is a family: ICR@k, with per-k eligibility

**Decision: report ICR@k for k in {1, 2, 3, 5}, with k=2 as the headline. Eligibility is per k.**

`ICR@k` = the fraction of eligible trials in which **at least k** interferer-exclusive content words appear in the response.

**Why per-k eligibility.** A trial whose interferer had only 1 exclusive content word *cannot* score ICR@2. Pooling it would drag ICR@3 and ICR@5 toward zero for a reason about the trial, not the system. So for ICR@k only trials with **≥ k available** exclusive content words are eligible, and **n_k is reported for every k.**

**Cost.** Each ICR@k has a different denominator, so the curve mixes threshold strictness with a changing subset. It is an interpretation aid, **not** the sensitivity analysis 3.2 asks for; decision 3's continuous measure carries that.

**Why k=2.** One shared content word is coincidence at the rate English repeats nouns. Two is signal.

### 3. A continuous leakage measure is reported alongside, and it is what systems are RANKED on

**Decision: also report the mean fraction of available interferer-exclusive content that leaked, over trials with ≥5 exclusive words available.**

A thresholded rate ("one trial in two leaks") is easy to read but discards information and has poor variance at n=230, which conflicts with 5's requirement for confidence intervals. **Rank on the mean; quote ICR@2 for interpretation.**

The ≥5 floor is because one leaked word out of one is 100 %, which is noise.

### 4. Absent trials get their own ICR row, and it is the SHARPEST leakage evidence

**Decision: score ICR separately on target-absent trials, and treat that row as the primary leakage measurement.**

**B4's exclusion rule does not apply to ICR.** B4 keeps target-absent trials out of the headline because `t` is empty and WER against nothing is undefined. **ICR's reference is `d`, not `t`**, and `d` is non-empty. These trials are the cleanest case:

- the target never speaks, so the correct output is **silence**
- `t` is empty, so **every** interferer content word is "exclusive"
- so any interferer content in the response is **unambiguous** leakage

**The number already exists.** Floor invented-speech rate on `interferer_only` trials is **95.1 %**, ceiling **0.0 %**: near-total dynamic range, against the 52 % / 0 % of the present-trial ICR.

**Only `interferer_only` qualifies.** On `noise_only`, `d` is empty, so those trials are ineligible. In `eval_public`: 123 eligible, 22 ineligible.

### Carried forward from J3, unchanged

**Floor ICR is partly set by construction and must be labelled so.** The judge never sees the enrolment, so on an unprocessed two-speaker mixture it picks a speaker, and floor ICR lands near a coin flip. That is correct, and is the finding: doing nothing gets the wrong speaker about half the time. State it wherever the floor row is quoted. **The fixed prompt must not tell the judge to choose a speaker** ("the clearest voice"), as that hands the extractor's job to the judge.

### What ICR does not measure

It measures **leaked content**, not **detected presence**. A judge that says "another voice was talking" without quoting it correctly scores 0, so the row must not be read as "the judge did not notice the interferer".

---

## 2026-08-31 — NRR design: an empty response is the signal, not pattern matching

**Decision: the prompt tells the judge *if you cannot identify any speech, return nothing*, and a non-response is an empty response.** No sentinel token, no pattern list. `metric-definitions.md` 3.3 defines NRR as the "fraction of trials where the model declines, reports hearing nothing, returns silence, or produces a refusal" but not how that is decided.

**Why not a sentinel token.** A sentinel (`######`, checked before normalisation) was drafted and rejected as needless complexity. **The offline ASR already returns empty on silence**, so emptiness is one rule for both the ASR stand-in and the judge, where a sentinel needs two rules and a token to document, pin and protect from API mangling. Telling a deliberate "no speech" apart from an empty response caused by an API error or truncation **belongs at the transport layer, not in the metric**: record the call's finish reason, and retry a failed call as a missing measurement rather than scoring it.

### What NRR is for, in order of actual strength

**1. It detects judge malfunction.** NRR on clean target audio is the judge's own defect rate (it declined on perfect input). Nothing else sees that, and J2 warns that a degenerate judge looks like a degenerate extractor in the numbers.

**2. It separates "declined" from "misheard".** An ASR always outputs something; a live model can refuse, hit a safety filter, or judge the audio unusable. LCF-WER lumps those in with a bad transcription.

**3. It stops a mute scoring perfectly on ICR.** This is 3.3's reason and the weakest, because **LCF-WER already scores a mute ~100 %** (all deletions) and the deletion rate flags it as suppression. Do not defend NRR on this alone.

**It is a tripwire, not a quality measure.** It will not rank two working systems, and a one-point difference means nothing.

### Why pattern matching was rejected — measured, not assumed

On the **ground-truth texts** of `eval_public`, refusal phrases match **38 of 500 trials (7.6 %)**:

| pattern | matches in reference text |
|---|---|
| `nothing` | 23 |
| `there is/was no` | 11 |
| `i cannot / can't` | 4 |
| `no speech/voice/sound` | 1 |

Ground truth a pattern detector would have scored as a refusal:

> *"nothing would satisfy him that could not stand cross-examination"*
> *"there was no man sir, his troubled blue eyes glanced…"*

**The speakers say these words.** A correct transcription would count as a non-response, at a false-positive rate above the metric's entire range.

**Correction to a number reported in conversation.** A pattern-matching draft gave NRR floor **4.78 %** / ceiling **3.91 %** on `condition=both`. Re-measured, both are **0.00 %**: all were pattern matches on real speech. Do not quote them.

### The rule

| response | verdict |
|---|---|
| empty after normalising | non-response (`silence`) |
| in the artefact set (`you`, `thank you`, …) | non-response (`artefact`) |
| anything else | a real report; score it |

**The artefact set exists only for the offline-ASR stand-in.** `small.en` cannot be instructed and emits `you` on digital silence (8 of 8 absent trials, `decisions-m3.md` 2026-08-28). A prompted judge returns nothing and lands in `silence`.

**NRR is per trial, not per second.** A trial where 2 of 30 words came back is a real report that scores badly on LCF-WER, not a non-response. How much was lost is LCF-WER's deletion rate.

### Absent trials are excluded, inside the metric

Where the target never speaks, reporting nothing is **correct**, so counting it as a non-response inverts the score. An earlier draft reported **32.8 % NRR on clean target audio** for this reason, all from the 145 absent trials. `compute_nrr` therefore **requires** the target texts and drops empty-reference trials itself, so a caller cannot forget. Absent trials are measured by the invented-speech row instead (B4).

### Two consequences to carry

**A prose refusal is deliberately not detected**; it scores as a bad transcription, and catching it needs the rejected pattern list. **Non-compliance is handled by disqualifying the judge**: `silence_compliance()` measures it on silent input and belongs in the J2 candidate gate beside ceiling LCF-WER.

**NRR must be read as a delta from the ceiling.** `system_attributable_nrr()` names the subtraction so it is not skipped.

### Status: dormant until a judge exists

With the offline ASR standing in, NRR is **0.00 % on all 355 present trials at both floor and ceiling**, since a transcriber always returns words on speech. **NRR carries no information until a real prompted judge can decline.** Built and tested; cannot be exercised before J2 closes. Read the zeros as "not yet measurable".

**Still open:** off-prompt chat, where the judge answers the content instead of reporting it ("what time is the meeting" answered with "I don't have access to your calendar"). It matches no rule, is not empty, and cannot be caught without the false-positive problem. Check by hand on the judge pilot and record as a known blind spot.

## 2026-09-03 — Judge run 2: the borrowed WeSep baseline. No rank inversion

The live judge did not reorder the two systems. All five instruments (SI-SDR, SIR, offline WER, DNSMOS `OVRL`, live judge) rank the borrowed WeSep checkpoint ahead of our baseline, the judge by the widest margin. M6's divergence test is answered negatively on n=2.

**Provenance** (mandatory). Closed models change silently, so this is not comparable to a judge result from another date unless re-run.

| field | value |
|---|---|
| model ID | `gemini-3.7-flash`, AI Studio prepay |
| modality | audio in / text out |
| prompt | `src/live_model_metric/judge_prompt.txt`, sha256 `d118b7d3bf30` |
| run date | 2026-09-03 |
| split / condition | `sir0_val` / `both`, 103 trials |
| speech gate | on, Silero 6.2.1 |
| estimates | `experiments/results/2026-09-03-est-wesep-tfmap-causal` |
| commit | `020c9698-dirty` |
| results | `experiments/results/2026-09-03-evaluate-wesep-judge/` |
| wall time | 11 min |

| system | LCF-WER | S | D | I | ICR@2 | mean leak | NRR |
|---|---|---|---|---|---|---|---|
| floor | 63.27 | 29.22 | 5.97 | 28.08 | 75.73 | 63.33 | 0.0 |
| WeSep | **26.40** | 10.62 | 6.78 | 8.99 | **25.24** | **12.92** | 0.0 |
| ceiling | 1.05 | 0.87 | 0.15 | 0.03 | 0.00 | 0.00 | 0.0 |

Baseline for comparison (2026-09-02, same prompt and anchors): 56.72 / 62.14 / 50.15.

**Run health:** 103 calls, 0 failed, 0 gate blocks, 1 transient safety-filter hit that passed on retry (~1-in-300, as on 2026-09-02). Floor and ceiling came from cache, identical to 2026-09-02, so both systems share anchors.

**WeSep wins by removing the interferer, not by preserving the target.** Insertions fall from 26.22 to 8.99 (the inserted words were the interferer's), while deletions *rise* from 3.43 to 6.78, above even the floor's 5.97. So LCF-WER on our two-speaker mixtures is dominated by interferer suppression, and a system can lose more target words than doing nothing while halving the score. Any claim built on LCF-WER must say so.

**Why this does not close M6.** The systems are 30 LCF-WER points apart, half the metric's range. A rank inversion concerns systems that are *close*, so this negative result is underpowered. Testing the hypothesis needs a near-tie pair, which is the case for building system 3 (per-band gate, D13).

Not comparable to any published REAL-TSE number: our data, metric and protocol. WeSep is borrowed (Wang et al., Interspeech 2024) and out of domain in every direction. See `project-state.md` § System 5.

---

## 2026-09-03 — J3 SIGNED OFF. ICR threshold is `count >= 2`, and k turns out not to matter

**Decision: the ICR threshold is `count >= 2`** (at least two interferer-exclusive content words in the response). Written into `metric-definitions.md` §3.2, which required a threshold "fixed in advance with its sensitivity reported" but supplied neither.

**Why `count >= 2` and not a fraction.** One shared content word is coincidence at the rate English repeats nouns; two is signal. The fraction rule (`>= θ` of the available interferer-exclusive words) scales with the interferer's utterance length, which varies per trial **by construction**, so it tracks the trial rather than the system. It is therefore the secondary. Both are computed.

**The sensitivity sweep shows the choice does not matter.** `eval_public`, offline ASR stand-in, 2026-08-31 (`experiments/results/2026-08-31-icr-sweep-eval_public/`):

| set | ICR@1 | ICR@2 | ICR@3 | ICR@5 |
|---|---|---|---|---|
| present (`both`), floor | 57.0 | **52.0** | 49.3 | 42.9 |
| present (`both`), ceiling | 0.0 | **0.0** | 0.0 | 0.0 |
| absent (`interferer_only`), floor | 100.0 | **100.0** | 99.2 | 97.5 |

The floor moves 14 points across the range while the ceiling stays at 0.0. **Every k separates the anchors completely**, so no conclusion changes for k of 1, 3 or 5. That fact meets §3.2's sensitivity requirement; no evidence could separate 2 from 3 anyway.

**Also settled.**

1. **Exclusion, not zero.** Trials where the interferer said nothing exclusive are excluded from ICR, not scored clean, which would dilute the rate with trials that could never fire. The count is reported. **Currently 0 in every `eval_public` stratum**: the rule is inert here and kept for correctness.
2. **The floor's ICR is partly by construction.** The judge never sees the enrolment, so on an unprocessed mixture it picks a speaker and the floor tends toward a coin flip. At k=2 on the `eval_public` floor: **87.5 %** when the interferer is louder, **37.5 %** when the target is louder. This is a property of the task, not the listener, and must be stated wherever the floor is quoted.

**Binding consequence for the prompt.** It must never tell the judge to choose a speaker (no "the clearest voice", no "the loudest speaker"). That hands the extractor's job to the judge and would move the floor row for reasons unrelated to any system. The frozen prompt (sha256[:12] `d118b7d3bf30`) already complies.

**Caveat carried.** The sweep scores `eval_public` through the offline ASR stand-in. It is **not** a live-model result and must not be quoted as one. The judge-measured ICR@2 figures are in `project-state.md`'s results table on `sir0_val`.

---

## 2026-09-03 — J2b cut: no open-weight judge

**Cut on schedule, not merit.** Qwen3-Omni was selected (2026-08-31) for encoder independence: its AuT encoder is trained from scratch, while Voxtral, Ultravox and Qwen2.5-Omni build on a Whisper encoder, sharing lineage with `faster-whisper small.en`. There is no room for a T4x2 session and a second scoring pass with 6 weeks to freeze and M5 unbuilt.

**Cost:** judge numbers are not reproducible without API access, and `gemini-3.7-flash` is a closed model whose weights we do not control. This goes in the thesis limitations next to A1 and the two-speaker boundary. **Superseded in part 2026-09-07:** cross-date comparison is no longer treated as invalid.

**Not lost:** the harness is judge-agnostic, so an open-weight listener is a later config change, not a redesign. The prompt is frozen by hash, provenance is recorded per result, and the 344 raw judge responses are kept in `judge_responses.csv`, so the numbers can be audited, though not regenerated.

`metric-definitions.md` §5 is downgraded from a requirement to a stated limitation, and §7's open question closed as cut. `research-plan.md`'s old contingency ("reduce judges to one — keep the open-weight one") ends up inverted.

---

## 2026-09-03 — Text reference condition cut: excluded on latency, not scored

**The text path is ruled out, not measured.** `metric-definitions.md` §3.5 defined it as a reference condition (extractor → off-the-shelf ASR → text → judge). No text LCF number will be produced.

**Why.** Extraction alone costs 162 ms mean / 176 ms p99 (measured 2026-09-01) against a 200-300 ms budget, and the non-streaming ASR also needs an endpoint before it decodes. The text path cannot meet the latency constraint. Scoring it would publish a content number for a system the spec cannot deploy, and a text row would likely *win* on content, inviting "text wins, why not use text?" with the answer in a footnote. **Answering the modality question with measured latency is cleaner than scoring LCF and then discounting it.**

This extends the 2026-08-07 output-modality decision: text would not be *built*, and now will not be *scored* either, for the same reason, stated as a measurement.

**Consequences.** M6's anchor item drops to audio floor + audio ceiling. §3.5's two caveats stay as the reasoning for the exclusion, and both belong in the write-up: the judge is close to a pass-through on text, so the score would mostly reflect the front-end ASR, and the metric is lexical, so prosody and speaker identity vanish at the ASR boundary.

---

## 2026-09-03 — FR added as the third score; NRR removed

**NRR is out of the protocol.** It read **0.0 % on every judge row**: its detector is an empty response, and `gemini-3.7-flash` never returns one; it invents instead. `nrr.py` itself says: "a tripwire, not a quality measure. It will not rank two working systems."

**FR (Fabrication Rate) replaces it**: response content words in **neither** speaker's script. It closes a hole §3.3 already named: the invention rate was measured once, *of the judge*, so a system whose artefacts *provoke* invention was invisible. `fabrication.py`, 17 tests, wired into `evaluate.py`. Sweep in `experiments/results/2026-09-03-fr-sweep-sir0_val/`, re-analysing cached responses with no new API calls.

`sir0_val` `both`, n=103, prompt `d118b7d3bf30`:

| system | invented/trial | FR@2 | mean % | resp. words |
|---|---|---|---|---|
| ceiling (clean target) | 0.20 | 1.0 % | 1.4 % | 14.8 |
| floor (mixture) | 1.24 | 32.0 % | 6.8 % | 18.3 |
| baseline | **1.83** | 41.7 % | 10.4 % | 17.9 |
| WeSep | **1.80** | 38.8 % | 13.8 % | 15.1 |

**Two findings.** Both extractors raise fabrication **~48 % above doing nothing** (1.24 to ~1.8). **Baseline and WeSep are indistinguishable** (1.83 vs 1.80) despite a 25-point WER gap and WeSep leaking a third as much. Fabrication is its own axis, not a by-product of extraction quality, so it is reported separately rather than left inside insertions.

**Pitfall found during implementation.** The first version reported the *percentage*, on which WeSep looks worse (13.8 % vs 10.4 %). That divides by the response's own length, and WeSep's responses are shorter (15.1 content words against 17.9), which inflates it. **Cross-system claims use `invented_per_trial`; the percentage is within-system only.** Enforced by `test_per_trial_count_is_immune_to_response_length`.

**FR is an upper bound and its ceiling is not zero** (0.20 words/trial). A word absent from both scripts may be invented, misheard as another real word, or a reference mismatch; only the first is fabrication. Quote FR only against its ceiling. ICR's ceiling is 0.0, so it has no such problem.

**Sensitivity.** Ordering is unchanged at k = 1, 2, 3, 5, so headline k=2 is a convention, as J3 found for ICR the same day.

**Kept from NRR.** A muting extractor scores a *clean* ICR because nothing came out, so something must still trip. FR excludes empty responses from its rate (otherwise muting would lower fabrication) and reports their count as `no_response`, keeping the pairing. `nrr.py` is kept and marked superseded: historical rows in `RESULTS.md`, `project-state.md` and the `results.json` files carry an `nrr` column produced by it.

---

## 2026-09-04 — `judge_prompt_v2.txt` CONSIDERED, NOT USED. The prompt is frozen

**The instrument stays `src/live_model_metric/judge_prompt.txt`, sha256[:12] `d118b7d3bf30`.** `judge_prompt_v2.txt` is in the tree, has never produced a recorded number, and is not to be adopted. Logged because an unused variant beside the live prompt is easily used by mistake.

### What is in use

    Write down the words spoken in this audio, exactly as spoken.

    If no words are audible, leave the transcript empty and set status to no_speech.

144 bytes, `d118b7d3bf30`. `judge_gate.yaml:19` points at it. Every judge result was run with it: the baseline row 2026-09-02, the WeSep row 2026-09-03, the ICR sweep and the FR sweep.

### What v2 added, and why it is not an improvement

385 bytes, sha256[:12] `64d4b994a9a2`. It adds a role line ("You are an audio transcriber"), expands the no-speech instruction, and appends "Be very careful to transcribe correctly. Do not hallucinate."

**Those last two lines are the problem.** FR measures fabrication (`decisions-m4.md` 2026-09-03: both extractors raise it ~48 % above doing nothing, and are indistinguishable on it despite a 25-point LCF-WER gap). Telling the judge not to hallucinate **tunes the instrument against the quantity it measures.** A later drop in FR could not be attributed to the front end or the prompt, which weakens the gaming-resistance claim (the primary contribution).

The no-speech wording is a real question, but J1's structured-field design handles it (`decisions-m4.md` 2026-08-31: the signal is a structured field, never in-band text, because the normaliser maps `####` and `[no speech]` both to `''`). v2 changes the wording, not the mechanism.

### The cost of switching, stated

The prompt's sha is in the judge cache key, so an edit cannot silently reuse stale answers. Pointing `judge_gate.yaml` at v2 would **re-buy every judge call at a prompt no results table refers to**: a full cache miss, a fresh bill, and numbers not comparable to anything recorded.

**M4 closed 2026-09-03. Reopening it for an untested prompt variant is not justified**; a late prompt change invalidates every judge row rather than adding one.

### Consequences to carry

- `judge_prompt_v2.txt` is kept as the record of what was considered (as with `nrr.py`). No config may ever point at it.
- `scripts/judge_smoke.py:52` uses v2 as the example in a `--prompt-file` help string. **Misleading**: change it to `judge_prompt.txt` so a copy-pasted smoke run cannot use the unused prompt.
- The appendix reproduces `judge_prompt.txt` verbatim **with its hash**, the machine-checkable form of CLAUDE.md's "record the exact prompt".

## 2026-09-26 — Judge repeats: e21 judged 3 times on the same 103 clips

**Decision (Grant):** e21's `sir0_val` `both` estimates get repeats 1 and 2 (repeat 0 is 2026-09-25), so every clip has 3 judge answers. Purpose: measure how much the judge's score on the same clip changes across identical calls, on our model's output.

- **103 × 3, not 50 × 6.** Same cost (309 vs 300 calls). Judge noise shrinks with total calls either way, clip-difficulty noise only with more clips, and 103 matches every existing row.
- **k=3, extendable.** Going from 1 to 3 cuts the judge-only band on one aggregate from ±2.94 to ±1.70 points (2026-09-22 retest SEM 1.50, divided by √k). Repeats 3–4 can be added later without re-buying 0–2.
- **Anchors not repeated.** Run-once: floor and ceiling serve their cached answer at any repeat index.
- **Settings unchanged**: server-default temperature, same model, prompt and backend. The cache key excludes generation config, so a mid-study change would silently mix two instruments.
- **Interval method:** resample clips carrying all 3 answers, not mean ± t·SD over 3 run totals (t = 4.30 at 2 df; a 3-value SD is unstable).
- **Code:** `scripts/evaluate.py --judge-repeat N`; `judge_repeat` recorded in `results.json` provenance. Partly closes weak-point A4.
- Dirs `2026-09-26-eval-cuecontext-wer-e21-judge-r1` / `-r2`. `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt sha `d118b7d3bf30`, run 2026-09-26.

**An inference, not a measurement.** The single-answer bootstrap on between-epoch gaps (±4.2–4.5 points, `decisions-m2.md` 2026-09-26) is about the size of the judge-only band on a paired difference (±4.16, 2026-09-22), so most of that spread is probably the judge, not clip choice. Caveat: the retest was on mixtures, where the judge is likely noisier than on estimates. The repeats test this.

### Results, 2026-09-27

`experiments/results/2026-09-27-judge-repeats-e21/`. Repeat 0 ran 2026-09-25, repeats 1–2 on 2026-09-26. No re-run needed: 102 of 102 judged clips answered on every run.

| lower is better | run 1 | run 2 | run 3 | mean | range |
|---|---|---|---|---|---|
| LCF-WER | 39.23 | 39.64 | 39.90 | **39.59** | 0.67 |
| ICR@2 | 33.01 | 32.04 | 35.92 | 33.66 | 3.88 |
| FR@2 | 45.10 | 46.53 | 49.02 | 46.88 | 3.92 |

- **The total is stable; single clips are not.** The total moved 0.67 points across 3 runs. Per clip: 2 of 102 answers word-for-word identical, 46 of 102 (45 %) moved by more than 10 points, 5 by more than 50, max 100. Yes/no verdicts changed on 15 of 102 clips (ICR@2) and 27 of 102 (FR@2).
- **The judge is steadier on our output than on raw mixtures**: pooled within-clip SD 12.85 vs 17.03 (2026-09-22 retest, both unbiased); judge-only SEM of one run 1.24 vs 1.50 points. This confirms the caveat above.
- **Repeats barely narrow e21's own interval**: 39.59 [33.10, 46.67], width 13.6, against 13.9 from one run. For one system's absolute score the width comes from which clips, not judge noise. Repeats pay on a paired gap between two systems, where clip difficulty cancels.
- One repeat-1 clip was refused by the safety filter on both retries and scored empty (`filter_blocked` 1). Kept as part of the instrument's run-to-run behaviour.
- **Against No processing (anchors one run so far), paired:** words wrong −22.87 [−31.10, −14.98]; 2+ other-speaker words through −41.10 [−50.81, −31.72]; **invented 2+ words +13.87 [+3.56, +24.14], worse, and outside the noise.** Gap closed 37.2 % [26.6, 47.0].
- **Run-to-run SD of one run, from per-clip variation:** LCF-WER 1.24, ICR@2 2.17, FR@2 2.91 points. The SD of the 3 totals (0.34 / 2.02 / 1.98) has 2 df, so report the per-clip figure. Speaker-block bootstrap (38 target speakers, Liu & Peng 2020) gives LCF-WER [33.3, 46.0] against plain [33.10, 46.67]: no understatement.
- **Report**: the judge's answer per clip changes on almost every call, but the corpus score over 103 clips moves under 1 point. Per-clip judge numbers are not usable alone (as 2026-09-22); the aggregate is.

## 2026-09-27 — Table 5.1 ± is the plain SD of the 3 run scores

**Decision (Grant):** every row of the report's judge table (Table 5.1) now has 3 runs on the same 103 clips, and its ± is the plain SD (ddof=1) of the three published whole-run scores. The clip-by-clip SD moves to a companion table (`tab:judge_runs`, LCF-WER only) beside each run's score and the spread. Supersedes "report the per-clip figure" (2026-09-26) for this table.

- **Why:** the clip-by-clip SD assumes clips vary independently. Baseline broke this: its 2026-09-27 runs differ by 6.4 LCF-WER points (paired sign-flip p < 0.01, below 0.05/3; 0.004–0.005 across Monte-Carlo draws), a whole-run shift. Clip-by-clip gave ±1.66; the runs spread ±3.22. Clip-by-clip cannot see a run shift.
- **Cost:** 3 values give an unstable SD (2 df). Floor reads ±0.37 against clip-by-clip 1.58, and Extension ±0.34 against 1.24, so the plain SD can understate by luck. The companion table carries both.
- **Anchors now repeated** (reverses "anchors not repeated" above): `--judge-repeat-anchors`, 2026-09-27. Run 0 was rebuilt from cache rows dated ≤ 2026-09-02, because `judge.py` `load_once_index` serves the last row per run-once clip whatever its repeat (open bug: a plain `evaluate.py` run now scores floor 62.98 / ceiling 1.14).
- Shifts tested (sign-flip, LCF-WER, each pair of runs): only Baseline r1–r2 is significant. Floor, ceiling, Extension (e21), WeSep: none (lowest p 0.06). `2026-09-27-judge-plain-sd/run_shift.txt`.
- `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt sha `d118b7d3bf30`. Run dates and dirs: `experiments/results/2026-09-27-judge-plain-sd/` and the `2026-09-27-judge-repeats-{anchors,wesep,baseline}` records.

## 2026-09-28 — Four-case judge results; a missing API key had passed for perfect silence

Judge `gemini-3.7-flash`, audio-in / text-out, prompt sha `d118b7d3bf30`, run 2026-09-28, one run. `sir0_val`, cases `target_only` (47), `interferer_only` (42), `noise_only` (8). `experiments/results/2026-09-28-case-suite-judge-{baseline-e6,e21,wesep}/`; e21 and WeSep renders `2026-09-28-est-*-cases`. `judge_failed` 0 in every row.

| | target only, LCF-WER % | other speaker only, words/trial | silenced (of 42) | suppression dB |
|---|---|---|---|---|
| No processing | 1.65 | 30.69 | 0 | 0.0 |
| Baseline | 4.00 | 25.26 | 7 | −11.0 |
| e21 | 3.56 | 16.50 | 18 | −27.4 |
| WeSep | 5.40 | 17.95 | 7 | −9.7 |
| Target alone | 1.46 | — | — | — |

- **Target only:** every system costs 2–4 words per 100 against doing nothing. One run, n=47: not a ranking.
- **Other speaker only is the largest remaining failure.** Baseline passes 82 % of the stranger's words, e21 54 %, WeSep 58 %. **e21 is all-or-nothing:** it silences 18 clips, and on the other 24 Gemini hears 28.9 words, nearly the raw mixture's 30.7. Its −27 dB mean comes from the silenced clips.
- **Noise only:** 0 words for every system, all 8 gated. The gate's verdict, not a listener's; n=8.
- **Instrument bug, fixed:** the first pass ran with `GEMINI_API_KEY` unset, so every uncached clip failed and was scored "". The absent-case path did not record failures, so this read as perfect silence. Now `MissingCredentials` aborts the run (`judge.py`, `evaluate._listen`), absent rows record `judge_failed`, and `results.txt` warns on any failure. Tests added.

## 2026-09-29 — Case 3 on `eval_public`: e21 beats WeSep, selection-free

Judge `gemini-3.7-flash`, audio-in / text-out, prompt sha `d118b7d3bf30`, run 2026-09-29, one run. `eval_public` `interferer_only`, n=123. Rendered via the new render-only split `sir0pub` (`scripts/train.py` refuses to train on it). `experiments/results/2026-09-29-{est-public-*-case3,case-suite-judge-public-*}/`.

| | words/trial | stranger's words passed | silenced (of 123) |
|---|---|---|---|
| No processing | 27.97 | 100 % | 0 |
| Baseline | 25.63 | 92 % | 10 |
| e21 | 15.91 | 57 % | 56 |
| WeSep | 19.55 | 70 % | 11 |

Paired bootstrap over clips, 10,000 draws, seed 42:
- **e21 − WeSep −3.64 [−5.67, −1.65]**, beyond noise. `sir0_val`'s −1.45 [−7.05, +4.10] was the same effect, under-powered at n=42.
- e21 − baseline −9.80 [−12.25, −7.41]; WeSep − baseline −6.24 [−8.45, −4.10] (n=122: baseline clip `eval_public-42-000149` got a 400 invalid-argument with normal audio, and was excluded rather than scored as silence).
- Silenced: 47 e21-only vs 2 WeSep-only, exact McNemar p = 4e-12. Deterministic (VAD), so no repeats needed.

- **First selection-free result for e21.** Word it that way.
- **Still all-or-nothing:** unsilenced e21 clips carry 29.2 words vs the floor's 28.0. The gain comes from silencing more often, not partial suppression.
- **`eval_public` is now seen.** Nothing may be selected or tuned on it.

## 2026-10-06 — AMI: WeSep and e21 through the judge
Judge `gemini-3.7-flash`, audio-in / text-out, prompt sha `d118b7d3bf30`, run 2026-10-06, one run. `ami_eval` `both`, n=300. Construction: decisions-m0.md 2026-10-06. Numbers and paths: `experiments/results/2026-10-06-ami-summary/`.

| | No processing | WeSep | e21 | Target's headset |
|---|---|---|---|---|
| LCF-WER | 115.5 | **80.6** | 91.3 | 72.5 |
| target words missed | 3.5 | 27.7 | 45.3 | 2.3 |
| other speaker passed (mean leak) | 82.8 | **19.6** | 25.2 | 52.0 |
| invented / trial | 1.48 | 2.65 | **1.83** | 1.13 |
| gate-blocked | 0 | 38 | 110 | 0 |

- **Ranking holds on real audio:** WeSep ahead of e21, as on `sir0_val`.
- **Floor > 100 %:** the judge transcribes both talkers, and the other speaker's words are insertions against a ~14-word target.
- **Headset is not a ceiling here:** median 20.9 dB bleed separation, 52 % leak. Do not report "share of headroom": the gap is small because the headset is poor.
- **e21 near-mutes real audio:** 110/300 outputs blocked, median 25.9 dB below its input, yet speech is still present (110/110 found after levelling). Real behaviour, not a gate artefact. Suspected cause: AMI input (~−48 dB RMS) is far quieter than training.
- **WeSep's 38 blocks are mostly a gate artefact:** ~7 dB below input, 38/38 speech after levelling. Its 80.6 is pessimistic.
- **e21 invents less:** +24 % over the floor vs +79 % for WeSep.
- **10 × 400 invalid-argument** across both passes, audio normal; all succeeded on retry. Final runs have 0 failures.

Open: meeting-clustered bootstrap on e21 − WeSep; e21 with the 110 inputs levelled (local diagnostic) to test the quiet-input explanation.

## 2026-10-27 — AMI: matched baseline e11 through the judge

Judge `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, run 2026-10-27, one run, speech gate on. `ami_eval` `both`, n = 300. Render `experiments/results/2026-10-27-est-ami-baseline-matched-e11/` (`models/model_sir0_baseline-matched-e11.pt`, seed 42). Judge: `...-baseline-matched-e11-judge{,-retry}/`; 1 client timeout, answered by the retry. Git commit `71614bf`, recorded by hand. Other systems are the 2026-10-06 runs.

Intervals: `experiments/results/2026-10-27-ami-paired-ci/` (58 meetings redrawn, 10,000 draws, seed 42). Reproduces the four published points and intervals of `2026-10-12-ami-ft-final/paired_ci.json` exactly.

| n = 300, lower is better | No processing | Baseline e11 | e21 | WeSep | Target's headset |
|---|---|---|---|---|---|
| LCF-WER | 115.51 | 101.50 | 91.26 | **80.56** | 72.53 |
| 95 % CI | [103.95, 128.89] | [94.40, 109.36] | [84.29, 98.87] | [73.49, 88.32] | [62.52, 83.72] |
| target words missed | 3.46 | 27.33 | 45.25 | 27.71 | 2.31 |
| mean leak | 82.80 | 31.63 | 25.19 | **19.64** | 51.96 |
| invented / trial | 1.48 | 3.35 | **1.83** | 2.65 | 1.13 |
| gate-blocked | 0 | 58 | 110 | 38 | 0 |

| Paired, first minus second | Difference | 95 % CI |
|---|---|---|
| Baseline − No processing | −14.01 | [−22.41, −6.42] |
| e21 − Baseline | −10.24 | [−15.84, −4.29] |

- **An empty output scores exactly 100 %** (pooled LCF-WER, every target word a deletion). The floor is above 100, so on AMI muting always beats doing nothing. The baseline's interval contains 100: it is not distinguishable from outputting silence. e21's lies below 100.
- The baseline's gain over no processing is mostly leak removal (82.8 → 31.6 %) paid for in missed target words (3.5 → 27.3 %) and more than double the invented words.
- `report/figures/set_comparison.pdf` panel (c) now shows the baseline.

## 2026-10-07 — eval_private scored ONCE for the selection-free headline

**Decided 2026-10-07 (Grant), after all selection on `sir0_val` is finished.** Baseline 10000-e6, e21 and WeSep are rendered on `eval_private` `both` (249 trials, 20 speakers never trained, tuned or selected on) and judged once each: `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, run date recorded per result. Driver `eval_private.sh` (render, then judge); render-only split `sir0priv` added to `scripts/train.py` and `scripts/make_estimates_wesep.py` (training on it is refused).

**Why.** Both blind reviews (`docs/reports/2026-10-07-blind-examiner-review*.md`) rank "checkpoints chosen on the reported trials" as the top problem. This is the protocol `select_by_wer.py` always stated: select on `sir0_val`, report on `eval_private`.

**Reporting rule, fixed now:** the methods say checkpoints were selected on `sir0_val` and the headline is reported on `eval_private`. `eval_private` uses the original level distribution (target usually louder), so it is easier than `sir0_val`. Report both, and per SIR band if space allows.

**Consequences:** `eval_private` is now seen. Nothing may be selected or tuned on it from here on (CLAUDE.md limit (a) still holds for training). The matched baseline, once selected on `sir0_val`, is scored here the same way.

## 2026-10-08 — eval_private results; trial 000372 removed for every system

Judge `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, run 2026-10-08, one run, speech gate on. `eval_private` `both`. First pass + retry: `experiments/results/2026-10-08-eval-private-*-judge{,-retry}/`.

**Removed (Grant):** `eval_private-42-000372`, for all systems and both anchors. Gemini refused the baseline and e21 outputs of that clip twice with HTTP 400 "Request blocked due to copyright/recitation content"; WeSep, mixture and target were answered. A filter decision cannot be retried, and scoring it empty for two systems only biases against them. Decided after seeing results, so report the n = 249 numbers alongside. Scored from cache, no new calls: `experiments/results/2026-10-08-eval-private-excl-000372/`.

| n = 248, lower is better | No processing | Baseline e6 | e21 | WeSep | Target alone |
|---|---|---|---|---|---|
| LCF-WER | 52.05 | 48.79 | 41.81 | **34.08** | 1.78 |
| mean leak | 47.89 | 38.20 | 21.15 | **18.99** | 0.00 |
| invented / trial | **1.48** | 1.97 | 2.26 | 2.07 | 0.29 |
| empty responses | 0 | 0 | 5 (gate) | 1 (gate) | 0 |

- Ranking unchanged from n = 249 (49.11 / 42.17 / 33.87): removal moves no system by more than 0.4 points.
- The recitation refusal reached the code as `judge_failed`, not `filter_blocked`, because a 400 bypasses the filter classifier. Open: classify recitation 400s as filter blocks.

## 2026-10-27 — matched baseline e11 scored on eval_private (closes the 2026-10-07 promise)

Judge `gemini-3.7-flash`, aistudio, audio-in / text-out, prompt `d118b7d3bf30`, run 2026-10-27, one run, speech gate on. Render `experiments/results/2026-10-27-est-private-baseline-matched-e11/` (`models/model_sir0_baseline-matched-e11.pt`, seed 42). Judge: `...-baseline-matched-e11-judge{,-retry,-retry2}/`; 6 client timeouts on the first pass, all answered by retry2. Git commit `71614bf` (clean tree), recorded by hand: the scripts no longer write it. Other systems are the 2026-10-08 runs.

Intervals: `experiments/results/2026-10-27-test-set-paired-ci/` (n = 248, 000372 removed, 20 target speakers redrawn, 10,000 draws, seed 42). The four existing systems reproduce their published values exactly.

| n = 248, LCF-WER, lower is better | No processing | Baseline e11 | e21 | WeSep | Target alone |
|---|---|---|---|---|---|
| point | 52.05 | 48.46 | 41.81 | **34.08** | 1.78 |
| 95 % CI | [44.81, 60.65] | [42.17, 55.58] | [35.64, 48.72] | [28.74, 40.06] | [1.33, 2.19] |

| Paired, first minus second | Difference | 95 % CI | Verdict |
|---|---|---|---|
| Baseline − No processing | −3.59 | [−9.33, +1.97] | not shown to help |
| e21 − Baseline | −6.65 | [−10.66, −3.18] | e21 better |
| e21 − No processing | −10.24 | [−18.06, −3.43] | e21 better |

- The baseline's gain over no processing shrinks from 9.7 points on `sir0_val` to 3.6 on test and is no longer distinguishable from zero. Two causes, not separated: e11 was the best of its judged epochs on `sir0_val` (selection optimism), and `eval_private` is easier (no processing 52.05 vs 62.94).
- n = 249 (000372 kept): baseline 48.16, no processing 51.78. Leak 33.22 % vs 47.68 %; invented / trial 2.24 vs 1.47.
- `report/figures/set_comparison.pdf` panel (b) now shows the baseline (`scripts/plot_set_comparison.py` reads the test panel from the new `ci.json`).
