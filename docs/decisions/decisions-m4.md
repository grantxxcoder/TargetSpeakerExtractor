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
