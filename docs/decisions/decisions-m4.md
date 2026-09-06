# Decision Log — M4 (the metric is computable end to end)

Covers **LCF — Live-model Content Fidelity**, the project's primary
contribution: the three scores (LCF-WER, ICR, NRR), the judge, the fixed prompt,
the anchors, and the audio/text modality split. Defined in
`docs/data/metric-definitions.md`; this log records the decisions that document
does not fix.

Created 2026-08-31 when the log was split by milestone. See `milestones.md` M4.

**The conventional metrics are NOT here.** SI-SDR, DNSMOS, offline-WER and the
offline ASR that produces them are M3 (`decisions-m3.md`). Keeping them apart is
what stops the divergence result of `metric-definitions.md` 6 from being an
argument about one instrument measuring itself.

**Standing constraint on everything in this file (CLAUDE.md).** The judge model
never appears in the training loop, in any form, including as a proxy or a data
filter. Every judge result records the exact model ID, the exact prompt, the
input modality and the run date, because closed models change silently.

---

## 2026-09-02 — The judge hallucinates on silence. Speech-gate added; NRR caveated

**Measured, not suspected.** Fed the clean target of an absent trial — digital
silence, RMS **exactly 0** — `gemini-3.7-flash` returns `status=speech` and
fabricates fluent prose. Three prompt variants, six silent clips, **0 of 6 ever
returned `no_speech`**:

| prompt | sha256[:12] | invented words |
|---|---|---|
| v1 | `d118b7d3bf30` | 42, 27 |
| v2a — silence licensed as an expected outcome | `a21169a651d6` | 25, 17 |
| v2b — adds "Be very careful... Do not hallucinate" | `64d4b994a9a2` | 29, 33 |

`gemini-3.7-flash`, audio in / text out, AI Studio, prepay tier, 2026-09-02.

**It is not misheard audio — it is generation from the language prior.** The
clincher: on trial `sir0_val-42-000003`, v2b answered **in French** ("Ah c'est
comme ça qu'on fait ici...") on an English-only pipeline. The three answers per
clip share no content with each other or with either reference. The word count
did not fall monotonically with stronger wording — v2b, the variant that
explicitly forbids hallucination, produced MORE words than v2a. **Prompting does
not fix this and further prompt attempts were abandoned.**

**Not a Gemini quirk.** Whisper does the same, milder: `small.en` emits "you" on
silence in 8 of 8 absent trials (`decisions-m3.md`). The literature treats it as
a characterised failure mode of audio models and evaluates the mitigations
head to head — Koenecke-style VAD gating, confidence thresholding, hallucination
pattern matching, LM filtering — in *Investigation of Whisper ASR Hallucinations
Induced by Non-Speech Audio* (arXiv:2501.11378), which finds VAD and confidence
thresholding both give meaningful reductions and **no single method eliminates
it**. WhisperX established VAD pre-processing as the standard fix and names
Silero specifically.

**Confidence thresholding is unavailable to us**, so the choice was not free:
logprobs are missing from the Interactions API (documented parity gap with
`generateContent`), and logprob support above Gemini 2.0 is reported as
unreliable. Rejected on availability, not on merit.

### Decision: a speech gate in front of BOTH listeners

`src/live_model_metric/speech_gate.py`. Every clip gets a speech/no-speech
verdict before any listener sees it; a no-speech verdict returns the empty
hypothesis without a call.

**The argument, which is stronger than "workaround".** *Did the system emit
speech?* is a **signal** question and never needed a language model. The judge is
needed for what was **said**, and at that it is excellent — 0.0 % on clean
targets, byte-identical five times out of five. The gate is the right instrument
for the other question, not a patch over a broken one.

**Anchors are decided by CONSTRUCTION, not by VAD** — no new moving part in the
instrument for information the renderer already fixed. Verified against measured
RMS on `sir0_val`, 2026-09-02:

| condition | n | target.wav | interferer.wav | mixture.wav |
|---|---|---|---|---|
| `both` | 103 | speech | speech | speech |
| `target_only` | 47 | speech | RMS 0 | speech |
| `interferer_only` | 42 | **RMS 0** | speech | speech |
| `noise_only` | 8 | RMS 0 | RMS 0 | **noise, no speech** |

`noise_only` was not in the original silence analysis and is the more realistic
probe of the two: real energy, not one word of speech.

**`estimate.wav` is the only clip the VAD decides**, because it is the only one
whose content is not determined by construction — and it is where the question
matters, since collapse-to-silence is a measured failure mode of this model
(2026-08-25). Silero VAD 6.2.1, already pinned under B2. Threshold 0.10 s of
detected speech, about one short syllable.

**Symmetry is mandatory.** The identical rule gates the judge **and** the offline
ASR. Gate one and not the other and every judge-vs-ASR difference on a
speech-free clip measures the gate rather than the listeners. `gated()` exists so
the symmetric use is the natural one. Applied at scoring time for the ASR
(transcripts already cached) and at call time for the judge (where it also saves
money); the rule is the same at both points.

**Verified on the 24 v1-prompt answers, 2026-09-02.** 2 of 24 clips blocked — the
two silent targets — judge 42 → 0 and 27 → 0 words, `small.en` 1 → 0. Identical
block counts for both listeners. All 20 `both` clips passed untouched.

### What is logged, and why every decision and not only the blocks

`experiments/results/speech_gate.csv`: date, split, trial, clip, condition,
listener, verdict, reason, detected speech seconds. Every decision is recorded so
the denominator is recoverable and a gate that never fires is visible as such.

**A gate firing on a target-PRESENT trial is a finding, not a measurement
error** — it means the extractor destroyed the speech. It has to be auditable
rather than folded silently into a score.

### The cost to NRR — CORRECTED later the same day

**The first version of this entry said "NRR cannot catch the degenerate muting
extractor it was designed to catch". That is wrong.** It holds only of the
UNGATED pipeline. Recorded rather than edited away, because the corrected
statement is what the metric has to be defended on.

**With the gate, NRR works.** The gate intercepts *before* the judge, so a
muting extractor goes: VAD finds no speech → gate returns the empty hypothesis →
`non_response_reason("")` returns `"silence"` → **NRR fires.** The gate does not
bypass NRR, it *feeds* NRR the non-response NRR was built to detect. Without the
gate that non-response never occurred, which is exactly why the mute went
undetected.

**What NRR genuinely cannot see is narrower: a judge that CONFABULATES rather
than declines** — and this entry's own measurement is the example. The
2026-08-31 NRR entry gives its strongest purpose as detecting judge malfunction,
"it declined on perfect input". The malfunction this judge has is invention.
NRR's detector is an empty response; a confabulating judge is never empty. That
purpose is **not** served by NRR and must not be claimed for it.

**Consequence: the judge's invention rate becomes its own row**, measured on
known speech-free audio with the gate deliberately bypassed. It characterises
the judge, runs once, and is not part of the per-system protocol. The number
already exists: **0 of 6 silent clips returned `no_speech`, 17–42 words invented,
across three prompts.**

**NRR is NOT being redefined.** Rejected: widening it to "gate blocked or judge
silent", which would quietly change the meaning of a published score. Taken:
keep 3.3 exactly as written, and state that its mute detection *depends on the
gate* while its judge-defect detection does not extend to confabulation.

**One thing to state whenever a blocked trial is reported.** A blocked trial
scores a **clean ICR** — ICR eligibility is driven by how much exclusive content
the interferer had, not by the response, so an empty response is eligible with
zero leakage. On its own that is the degenerate-strategy hole; it is closed only
because NRR fires on the same trial. That pairing *is* the two-sidedness §4
claims, and a reviewer will go looking for it.

### Deviation from the spec, recorded

`metric-definitions.md` 3.1 sends every clip to the judge. It no longer does.
The spec needs a §3.1 amendment naming the gate as a component of the measuring
instrument — which means a change to it invalidates comparisons, exactly as for
the normaliser and the prompt.

### This is a result, not only a fix

"A live model cannot be trusted to report absence" is a metric-design finding
with the same shape as the DNSMOS-gaming story this project cites as its
cautionary tale (`metric-definitions.md` §4). It is reportable, and it is
evidence for why the metric had to be designed rather than borrowed.

**J2 stays open** until this entry is signed off; readings 1 and 2 of the
candidate gate pass decisively (ceiling 1.6 % vs the offline ASR's 5.8 %; floor
113 %, so ample dynamic range).

---

## 2026-08-31 — J1 CLOSED. The judge is audio-in / text-out, not full-duplex

**Decision: the judge is an audio-in / text-out model. Full duplex is not
required.** Closes J1 in `decisions-pending.md`. Unblocks J2 and therefore all
of M4.

**Why.** The property LCF measures is the judge's **audio encoder**, not its
turn-taking. Nothing in LCF-WER, ICR or NRR reads duplexing, and
`metric-definitions.md` 3.1 step 2 already makes audio *output* optional — "*if*
the model responds in audio, transcribe the response". An audio-in / text-out
model has exactly the learned encoder and the wide training distribution that
the artefact hypothesis in 3.1 is about.

**Three consequences that are gains, not compromises.**

1. **The response-transcription ASR disappears from the measuring instrument.**
   3.1 step 2 calls that ASR "a component of the measuring instrument, and
   changing it invalidates comparisons". A text response deletes it, and with it
   a whole class of invalidation. Only the *front-end* ASR of the text reference
   condition (3.5) remains, and it belongs to the system under test.
2. **A wider candidate field**, open and closed. Full duplex narrows the open
   field to roughly Moshi, whose task adherence is 1.26/5 on FullDuplexBench —
   below the bar our own ceiling gate sets.
3. **A less noisy instrument.** A full-duplex conversational model is tuned to
   *converse*, not to *report*. Handed "report what you heard" it may answer the
   content or chat about it instead of repeating it. That lands in **NRR**, where
   a degenerate judge is indistinguishable from a degenerate extractor. Removing
   the conversational tuning removes that confound.

**The cost, stated plainly.** This is a deviation from the project's stated
objective — CLAUDE.md and spec note 10 both say "live speech-to-speech model
(Gemini Live and similar)". The defensible sentence, which must appear in the
write-up and not merely in this log: *we used an audio-in model because the
measurement depends on the judge's audio encoder rather than on its duplexing.*
If that argument is not made explicitly, a reviewer is entitled to say the
thesis measured something other than what it set out to.

**How the deviation gets bought off rather than argued away: a small
full-duplex confirmation run.** The headline benchmark is audio-in / text-out.
Alongside it, ~50 trials are scored on a genuine full-duplex live API. At the
prices measured in J2 that is a few dollars. It converts the claim from "we
relaxed the objective and here is why that is acceptable" into "the ranking was
confirmed on a real full-duplex model at n=50". **Recorded here as part of the
decision, not as an optional extra**, because the argument above is weaker
without it.

**What this does NOT change.** The judge is still held out from training
absolutely — never a proxy, never a data filter (CLAUDE.md). Modality is still a
recorded property of every trial (3.5). The open-weight reproducibility anchor
is still required, and for reproducibility rather than cost (J2). Every judge
result still records the exact model ID, exact prompt, modality and run date.

**Still open after this:** J2a — which closed model for the headline; J2b —
which open-weight anchor. Both are now decided by the candidate gate rather than
by argument. J1 no longer blocks either.

---

## 2026-08-31 — ICR design: NLTK content words, ICR@k with per-k eligibility, absent trials on their own row

Four decisions taken, closing most of J3. `metric-definitions.md` 3.2 fixes the
*mechanism* — content-word overlap between the response `r` and the interferer's
text `d`, excluding words that also appear in the target's text `t`, thresholded
— but leaves the content-word list, the threshold and the eligibility rule
undefined. These are those.

**Why ICR is the second metric and not a nice-to-have.** Measured on the floor at
n=230 (`decisions-m3.md` 2026-08-31): **insertions are 30.8 of the 57.4 points**
of LCF-WER, the largest error type. Word error rate counts an inserted word
identically whether the other speaker leaked through or the listener hallucinated
from noise. **So the largest single component of the headline number is
undiagnosed, and ICR is what diagnoses it.**

### 1. Content words are defined by NLTK's English stopword list

**Decision: NLTK, chosen for comprehensiveness over a hand-written list.**

**Implementation requirement: the list is SNAPSHOTTED into the repo**, with the
NLTK version and the retrieval date recorded beside it. Three reasons, all of
which would otherwise break the metric rather than merely annoy:

1. `nltk.corpus.stopwords` needs `nltk.download('stopwords')`, a **network
   fetch at runtime**. Kaggle sessions run with networking off, so the metric
   would fail there, not degrade.
2. The list **changes between NLTK releases**. An unpinned word list inside the
   measuring instrument silently moves every ICR number on upgrade — the same
   failure mode the normaliser is pinned to avoid (B5).
3. It is **1 line of code either way**, so the pinned version costs nothing.

**The stopword list must itself be passed through the Whisper normaliser before
use.** NLTK's list contains apostrophe forms (`don't`, `you're`, `she's`); the
normaliser expands and strips those on both `r` and `d`, so unnormalised entries
would never match anything and would silently fail to filter.

**Known residual risk, not fixed by this choice.** NLTK's list does not contain
`one`, and `one` was observed as an "interferer-exclusive content word" on
`eval_public-42-000000` — a plausible-by-chance word that can fire a false
positive. Switching from a hand-written list to NLTK does not solve that. If
false positives show up in the pilot, the fix is a documented project-specific
addendum to the snapshot, never a silent edit.

### 2. The threshold is a family: ICR@k, with per-k eligibility

**Decision: report ICR@k for k in {1, 2, 3, 5}, with k=2 nominated as the
headline. Eligibility is evaluated PER k.**

`ICR@k` = the fraction of eligible trials in which **at least k**
interferer-exclusive content words appear in the response.

**Per-k eligibility is the part that makes this correct, and it came out of the
@k framing.** A trial whose interferer contributed only 1 exclusive content word
is *structurally incapable* of scoring ICR@2. Pooling it into the denominator
would drag ICR@3 and ICR@5 toward zero for a reason that is about the trial, not
the system. So for ICR@k, only trials with **≥ k available** exclusive content
words are eligible, and **n_k is reported for every k.**

**The cost of that, stated:** each ICR@k has a different denominator, so the
curve mixes threshold strictness with a changing trial subset. It is therefore an
interpretation aid, **not** the sensitivity analysis 3.2 asks for. The continuous
measure in decision 3 carries the sensitivity claim instead.

**Why k=2 is the headline.** One shared content word is coincidence at the rate
English repeats nouns. Two is signal.

### 3. A continuous leakage measure is reported alongside, and it is what systems are RANKED on

**Decision: also report the mean fraction of available interferer-exclusive
content that leaked, over trials with ≥5 exclusive words available.**

A thresholded rate of trials is interpretable — "one trial in two leaks" — but it
discards information and has poor variance at n=230, which fights 5's requirement
to report confidence intervals. The continuous mean has the statistical power, so
**rank on the mean and quote ICR@2 for interpretation.**

The ≥5 floor is because a fraction over a tiny denominator is not a fraction: one
leaked word out of one available is 100 %, which is noise, not a finding.

### 4. Absent trials get their own ICR row, and it is the SHARPEST leakage evidence

**Decision: score ICR separately on target-absent trials, and treat that row as
the primary leakage measurement.**

**B4's exclusion rule does not apply to ICR.** B4 keeps target-absent trials out
of the headline because `t` is empty and word error rate against nothing is
undefined. But **ICR's reference is `d`, not `t`** — and `d` is non-empty. So the
trials B4 excludes are not merely scoreable here, they are the cleanest case
available:

- the target never speaks, so the correct output is **silence**
- `t` is empty, so **every** interferer content word is "exclusive" by definition
- therefore any interferer content in the response is **unambiguous** leakage,
  with no attribution difficulty at all

**The number already exists in disguise.** Floor invented-speech rate on
`interferer_only` trials is **95.1 %**, ceiling **0.0 %** — near-total dynamic
range, against the 52 % / 0 % of the present-trial ICR.

**Only `interferer_only` qualifies, not all absent trials.** On `noise_only`
nobody speaks, so `d` is empty and there is no exclusive content to leak — those
trials are ineligible for ICR entirely. In `eval_public` that is 123 eligible
against 22 ineligible.

### Carried forward from J3, unchanged

**Floor ICR is partly set by construction and must be labelled as such.** The
judge never sees the enrolment, so on an unprocessed two-speaker mixture it
cannot know which speaker is the target and will pick one — which is why floor
ICR lands near a coin flip. That is the correct behaviour and it *is* the
finding: doing nothing gets you the wrong speaker about half the time. But it
must be stated wherever the floor row is quoted, and **the fixed prompt must not
instruct the judge to choose a speaker** ("the clearest voice"), which would hand
the extractor's job to the judge.

### What ICR does not measure

It measures **leaked content**, not **detected presence**. A judge that says
"another voice was talking" without quoting it scores 0, correctly — no content
leaked. That is intended, and worth stating so the row is not read as "the judge
did not notice the interferer".

---

## 2026-08-31 — NRR design: an empty response is the signal, not pattern matching

**Decision: the prompt instructs the judge — *if you cannot identify any speech,
return nothing* — and a non-response is an empty response.** No sentinel token,
no pattern list. `metric-definitions.md` 3.3 defines NRR as the "fraction of
trials where the model declines, reports hearing nothing, returns silence, or
produces a refusal" but does not say how that is decided. This is how.

**Why emptiness and not a sentinel token.** A sentinel (`######`, checked before
normalisation) was drafted first and rejected as needless complexity. **The
offline ASR already returns empty on silence**, so emptiness gives ONE detection
rule that serves both the ASR stand-in and the prompted judge; a sentinel would
have needed two rules and a token to document, pin and defend against API
mangling. The distinction a sentinel would have bought — telling a deliberate
"no speech" signal apart from an empty response caused by an API error or
truncation — **belongs at the transport layer, not in the metric**: record the
call's finish reason, and treat a failed call as a missing measurement to retry,
not as a trial that scored.

### What NRR is for, in order of actual strength

**1. It detects judge malfunction.** NRR on clean target audio *is* the judge's
own defect rate — it declined on perfect input. Nothing else in the metric set
can see that, and J2 warns that a degenerate judge is indistinguishable from a
degenerate extractor in the numbers.

**2. It separates "declined" from "misheard"** — a failure mode that exists only
because the judge is a conversational model rather than a transcriber. An ASR
always outputs something; a live model can refuse, hit a safety filter, or judge
the audio unusable. LCF-WER lumps those in with a bad transcription.

**3. It stops a mute scoring perfectly on ICR** — the reason 3.3 gives, and the
weakest of the three, because **LCF-WER already scores a mute ~100 %** (all
deletions) and the deletion rate identifies it as suppression. Do not defend NRR
on this ground alone; the obvious reply is "but WER already catches that".

**It is a tripwire, not a quality measure.** It will not rank two working systems
and must never be quoted as if a one-point difference means something.

### Why pattern matching was rejected — measured, not assumed

The obvious detector searches the response for refusal phrases. Run against the
**ground-truth texts** of `eval_public`, those phrases match **38 of 500 trials —
7.6 %**:

| pattern | matches in reference text |
|---|---|
| `nothing` | 23 |
| `there is/was no` | 11 |
| `i cannot / can't` | 4 |
| `no speech/voice/sound` | 1 |

Ground truth a pattern detector would have scored as a refusal:

> *"nothing would satisfy him that could not stand cross-examination"*
> *"there was no man sir, his troubled blue eyes glanced…"*

**The speakers say these words.** A correct, verbatim transcription would have
been recorded as a non-response, at a false-positive rate exceeding the metric's
entire range.

**CORRECTION TO A NUMBER REPORTED IN CONVERSATION.** A pattern-matching draft
gave NRR floor **4.78 %** / ceiling **3.91 %** on `condition=both`. Re-measured,
both are **0.00 %**. Those figures were entirely false positives — every one a
pattern match on real speech. Do not quote them.

### The rule

| response | verdict |
|---|---|
| empty after normalising | non-response — `silence` |
| in the artefact set (`you`, `thank you`, …) | non-response — `artefact` |
| anything else | a real report; score it |

**The artefact set exists only for the offline-ASR stand-in.** `small.en` cannot
be instructed and emits `you` on digital silence, measured 8 of 8 absent trials
(`decisions-m3.md` 2026-08-28). A prompted judge returns nothing and lands in
`silence`.

**NRR is per trial, not per second.** One trial, one yes/no. A trial where 2 of
30 words came back is a real report that scores badly on LCF-WER, not a
non-response. How much was lost is LCF-WER's deletion rate.

### Absent trials are excluded, inside the metric

Where the target never speaks, reporting nothing is the **correct** answer, so
counting it as a non-response inverts the score. An earlier draft reported
**32.8 % NRR on clean target audio** for exactly this reason, driven entirely by
the 145 absent trials. `compute_nrr` therefore **requires** the target texts and
excludes empty-reference trials itself, so a caller cannot forget. Absent trials
are measured by the invented-speech row instead (B4).

### Two consequences to carry

**A prose refusal is NOT detected, deliberately.** A judge that ignores the
instruction and declines in prose scores as a bad transcription. Catching it
needs the rejected pattern list. **Non-compliance is handled by disqualifying the
judge** — `silence_compliance()` measures it on silent input and belongs in the
J2 candidate gate beside ceiling LCF-WER.

**NRR must be read as a delta from the ceiling.** `system_attributable_nrr()`
names the subtraction so it is not skipped.

### Status: dormant until a judge exists

With the offline ASR standing in, NRR is **0.00 % on all 355 present trials at
both floor and ceiling** — a transcriber always returns words when the audio
contains speech. So unlike LCF-WER and ICR, **NRR carries no information until a
real prompted judge can decline.** Built and tested; cannot be exercised before
J2 closes. Read the zeros as "not yet measurable", not as a result.

**Still open:** off-prompt chat — a judge that answers the content instead of
reporting it ("what time is the meeting" → "I don't have access to your
calendar"). Matches no rule, is not empty, and is not detectable without
reintroducing the false-positive problem. To be checked by hand on the judge
pilot and recorded as a known blind spot.

## 2026-09-03 — Judge run 2: the borrowed WeSep baseline. No rank inversion

The live judge did not reorder the two systems. All five instruments (SI-SDR,
SIR, offline WER, DNSMOS `OVRL`, live judge) rank the borrowed WeSep checkpoint
ahead of our baseline, and the judge's margin is the widest. This is M6's
divergence test answered negatively on n=2.

**Provenance** — mandatory for any judge result. Closed models change silently,
so this is not comparable to a judge result from another date unless re-run.

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

Baseline for comparison (2026-09-02, same prompt and anchors): 56.72 / 62.14 /
50.15.

**Run health:** 103 calls, 0 failed, 0 gate blocks, 1 transient safety-filter hit
that passed on retry (the same ~1-in-300 rate as 2026-09-02). Floor and ceiling
served from cache and identical to 2026-09-02, so both systems are scored against
the same anchors.

**WeSep wins by removing the interferer, not by preserving the target.**
Insertions fall 26.22 → 8.99 (the inserted words were the interferer's) while
deletions *rise* 3.43 → 6.78, above even the floor's 5.97. Consequence: LCF-WER
on our two-speaker mixtures is dominated by interferer suppression, and a system
can lose more target words than doing nothing while halving the score. Any claim
built on LCF-WER must state that.

**Why this does not close M6.** The systems are 30 LCF-WER points apart, half the
metric's range. A rank inversion is a claim about systems that are *close*, so
this negative result is honest but underpowered. The test that exercises the
hypothesis is a near-tie pair — the argument for building system 3 (per-band
gate, D13).

Not comparable to any published REAL-TSE number: our data, metric and protocol.
WeSep is borrowed (Wang et al., Interspeech 2024) and out of domain in every
direction. See `project-state.md` § System 5.

---

## 2026-09-03 — J3 SIGNED OFF. ICR threshold is `count >= 2`, and k turns out not to matter

**Decision: the ICR threshold is `count >= 2`** — at least two
interferer-exclusive content words in the response. Written into
`metric-definitions.md` §3.2, which previously required a threshold "fixed in
advance with its sensitivity reported" and then supplied neither.

**Why `count >= 2` and not a fraction.** One shared content word between a
response and the interferer is coincidence at the rate English repeats nouns;
two is signal. The fraction rule (`>= θ` of the interferer-exclusive words
available) scales with the interferer's utterance length, which varies per trial
**by construction** — so it moves with a property of the trial rather than of the
system. That makes it the worse primary and the better secondary. Both are
computed.

**The sensitivity sweep says the choice is not load-bearing**, which is the
useful part. `eval_public`, offline ASR stand-in, 2026-08-31
(`experiments/results/2026-08-31-icr-sweep-eval_public/`):

| set | ICR@1 | ICR@2 | ICR@3 | ICR@5 |
|---|---|---|---|---|
| present (`both`), floor | 57.0 | **52.0** | 49.3 | 42.9 |
| present (`both`), ceiling | 0.0 | **0.0** | 0.0 | 0.0 |
| absent (`interferer_only`), floor | 100.0 | **100.0** | 99.2 | 97.5 |

The floor moves 14 points across the whole range while the ceiling sits at 0.0
throughout. **Every k separates the anchors completely**, so no conclusion
changes if k is 1, 3 or 5. §3.2's sensitivity requirement is discharged by that
fact rather than by arguing 2 over 3 — which is the honest way to close it,
because there is no evidence that would distinguish them.

**Two things settled at the same time.**

1. **Exclusion, not zero.** Trials where the interferer said nothing exclusive
   are excluded from ICR rather than scored clean; scoring them clean dilutes the
   rate toward zero with trials that could never have fired. The count is
   reported. **Currently 0 in every `eval_public` stratum** — the rule is inert on
   this data and is kept for correctness, not because it has bitten.
2. **The floor's ICR is partly by construction.** The judge never sees the
   enrolment, so on an unprocessed mixture it cannot know which speaker is the
   target and picks one; the floor tends toward a coin flip. Measured at k=2 on
   the `eval_public` floor: **87.5 %** when the interferer is louder against
   **37.5 %** when the target is louder. That is the task's property, not the
   listener's failure, and it must be stated wherever the floor is quoted.

**Binding consequence for the prompt.** It must never instruct the judge to
choose a speaker — no "the clearest voice", no "the loudest speaker". That hands
the extractor's job to the judge and turns a measurement into an instruction, and
it would move the floor row for reasons unrelated to any system under test. The
frozen prompt (sha256[:12] `d118b7d3bf30`) already complies; this records why it
must continue to.

**Caveat carried.** The sweep is `eval_public` scored through the offline ASR, a
STAND-IN. It establishes that the threshold choice does not change rankings; it
is **not** a live-model result and must not be quoted as one. The judge-measured
ICR@2 figures live in `project-state.md`'s results table on `sir0_val`.

---

## 2026-09-03 — J2b cut: no open-weight judge

**Cut on schedule, not on merit.** Qwen3-Omni was selected (2026-08-31) for
encoder independence — its AuT encoder is trained from scratch, while Voxtral,
Ultravox and Qwen2.5-Omni build on a Whisper encoder and would share lineage with
`faster-whisper small.en`. That reasoning still holds; there is just no room for
a T4x2 session and a second scoring pass with 6 weeks to freeze and M5 unbuilt.

**What it costs:** the judge numbers are not reproducible without API access, and
`gemini-3.7-flash` can change silently, so cross-date comparison stays invalid
unless re-run. Goes in the thesis limitations next to A1 and the two-speaker
boundary.

**What it doesn't cost:** the harness is judge-agnostic, so adding an open-weight
listener later is a config change, not a redesign. Prompt is frozen by hash,
provenance is recorded per result, and the 344 raw judge responses are kept in
`judge_responses.csv` — so the numbers can still be audited, just not regenerated.

`metric-definitions.md` §5 downgraded from a requirement to a stated limitation;
§7's open question closed as cut. Note that `research-plan.md`'s old contingency
("reduce judges to one — keep the open-weight one") ends up inverted.

---

## 2026-09-03 — Text reference condition cut: excluded on latency, not scored

**The text path is ruled out rather than measured.** `metric-definitions.md` §3.5
defined it as a reference condition (extractor → off-the-shelf ASR → text →
judge). No text LCF number will be produced.

**Why exclusion beats scoring.** Extraction alone costs 162 ms mean / 176 ms p99
(measured 2026-09-01) against a 200-300 ms budget, and the ASR is non-streaming —
it needs an endpoint before it decodes, on top of that. So the text path cannot
meet the project's own latency constraint. Scoring it would publish a content
number for a system that cannot be deployed under the spec, and a text row would
likely *win* on content — inviting "text wins, why not use text?" and forcing the
answer into a footnote. **Answering the modality question with latency, which is
measured, is cleaner than answering it with LCF and then discounting the LCF.**

This does not reverse the 2026-08-07 output-modality decision; it extends it. That
decision said text would not be *built*. This says it will not be *scored* either,
for the same reason, and states the reason as a measurement.

**Consequences.** M6's anchor item drops to audio floor + audio ceiling. §3.5's
two caveats are retained as the reasoning behind the exclusion — the judge is
close to a pass-through on text so the score would mostly reflect the front-end
ASR, and the metric is lexical so prosody and speaker identity vanish at the ASR
boundary. Both belong in the write-up.

---

## 2026-09-03 — FR added as the third score; NRR removed

**NRR is out of the protocol.** It read **0.0 % on every judge row**. Its detector
is an empty response and `gemini-3.7-flash` never returns one — it invents
instead. `nrr.py` said as much itself: "a tripwire, not a quality measure. It will
not rank two working systems."

**FR (Fabrication Rate) replaces it**: response content words appearing in
**neither** speaker's script. This closes a hole §3.3 already named — that the
invention rate was a property measured once *of the judge*, not per system, so a
system whose artefacts *provoke* invention was invisible. `fabrication.py`,
17 tests, wired into `evaluate.py`. Sweep in
`experiments/results/2026-09-03-fr-sweep-sir0_val/` — re-analysis of the cached
responses, no new API calls.

`sir0_val` `both`, n=103, prompt `d118b7d3bf30`:

| system | invented/trial | FR@2 | mean % | resp. words |
|---|---|---|---|---|
| ceiling (clean target) | 0.20 | 1.0 % | 1.4 % | 14.8 |
| floor (mixture) | 1.24 | 32.0 % | 6.8 % | 18.3 |
| baseline | **1.83** | 41.7 % | 10.4 % | 17.9 |
| WeSep | **1.80** | 38.8 % | 13.8 % | 15.1 |

**Two findings.** Both extractors raise fabrication **~48 % above doing nothing**
(1.24 → ~1.8). And **baseline and WeSep are indistinguishable** (1.83 vs 1.80)
despite a 25-point WER gap and WeSep leaking a third as much — so fabrication is
an axis of its own, not a by-product of extraction quality. That is the case for
reporting it separately rather than leaving it inside insertions.

**A trap found and closed during implementation.** The first cut reported the
*percentage*, on which WeSep looks worse (13.8 % vs 10.4 %). That is an artefact:
the percentage divides by the response's own length, and WeSep's responses are
shorter (15.1 content words against 17.9), so identical fabrication inflates.
**Cross-system claims use `invented_per_trial`; the percentage is a within-system
reading only.** Enforced by `test_per_trial_count_is_immune_to_response_length`.

**FR is an upper bound and its ceiling is not zero** (0.20 words/trial). A word
absent from both scripts may be invented, misheard as a different real word, or a
reference mismatch — only the first is fabrication. Quote FR only against its
ceiling; the excess is what a system is responsible for. ICR has no equivalent
problem: its ceiling genuinely is 0.0.

**Sensitivity.** Ordering is unchanged at k = 1, 2, 3, 5, so headline k=2 is a
convention rather than a load-bearing choice — the same conclusion J3 reached for
ICR the same day.

**What is preserved from NRR.** It existed so a muting extractor — which scores a
*clean* ICR because nothing came out — still trips an alarm. FR excludes empty
responses from its own rate (otherwise muting would lower fabrication) and reports
the count as `no_response`, so the pairing survives. `nrr.py` is kept and marked
superseded, not deleted: historical rows in `RESULTS.md`, `project-state.md` and
the `results.json` files carry an `nrr` column and this is the definition that
produced them.

---

## 2026-09-04 — `judge_prompt_v2.txt` CONSIDERED, NOT USED. The prompt is frozen

**The instrument stays `src/live_model_metric/judge_prompt.txt`,
sha256[:12] `d118b7d3bf30`.** `judge_prompt_v2.txt` exists in the tree, has never
produced a recorded number, and is not to be adopted. Logged because an unused
prompt variant sitting next to the live one is exactly the thing that gets picked
up by mistake.

### What is in use

    Write down the words spoken in this audio, exactly as spoken.

    If no words are audible, leave the transcript empty and set status to no_speech.

144 bytes, `d118b7d3bf30`. `judge_gate.yaml:19` points at it. Every judge result
in the project was run with it: the baseline row 2026-09-02, the WeSep row
2026-09-03, the ICR sweep and the FR sweep.

### What v2 added, and why it is not an improvement

385 bytes, sha256[:12] `64d4b994a9a2`. It adds a role line ("You are an audio
transcriber"), expands the no-speech instruction, and appends two imperatives:
"Be very careful to transcribe correctly. Do not hallucinate."

**Those last two are the problem, not the fix.** Fabrication is the thing FR
measures (`decisions-m4.md` 2026-09-03: both extractors raise it ~48 % above
doing nothing, and the two extractors are indistinguishable on it despite a
25-point LCF-WER gap). Instructing the judge not to hallucinate is **tuning the
measuring instrument against the quantity it measures.** Any drop in FR afterwards
would be unattributable between the front end and the prompt, and the metric's
gaming-resistance claim — the primary contribution — would be weaker for it. The
judge's listening behaviour must stay fixed while the extractor varies.

The no-speech wording is a real question, but it is J1's structured-field design
that carries it (`decisions-m4.md` 2026-08-31: the signal is a structured field,
never in-band text, because the normaliser maps `####` and `[no speech]` both to
`''`). v2 changes the wording, not the mechanism.

### The cost of switching, stated

The prompt's sha is part of the judge cache key, by design, so that editing it
cannot silently reuse stale answers. Pointing `judge_gate.yaml` at v2 therefore
**re-buys every judge call at a prompt no results table refers to** — no error,
just a full cache miss and a fresh bill, producing numbers not comparable to
anything already recorded.

**M4 closed 2026-09-03. Reopening it for an untested prompt variant is not
justified**, and a prompt change late in the project would invalidate every judge
row rather than add one.

### Consequences to carry

- `judge_prompt_v2.txt` is kept, not deleted, as the record of what was
  considered — same treatment as `nrr.py`. It must never be pointed at by a
  config.
- `scripts/judge_smoke.py:52` names v2 in a `--prompt-file` help string as its
  example. **Misleading**: change the example to `judge_prompt.txt` so a smoke
  run cannot be made against the unused prompt by copy-paste.
- The appendix reproduces `judge_prompt.txt` verbatim **with its hash**, which is
  the machine-checkable form of CLAUDE.md's "record the exact prompt".
