# Decision Log — M3 (baseline evaluated conventionally)

Covers the **conventional** evaluation instrument and what it says about the
data: the offline ASR, SI-SDR / DNSMOS / offline-WER scoring, latency and RTF
measurement. The NEW metric and its judge are in `decisions-m4.md` — that split
is deliberate, because M3's whole purpose is to produce the "conventional
metrics" column that M6's divergence table compares the new metric against
(`metric-definitions.md` 6).

Created 2026-08-31 when the log was split by milestone. See `milestones.md` M3.

**Why an M0 decision lives here.** C2 (how hard the task should be) was raised as
a data question in M0 and could only be *answered* with an evaluation
instrument, which did not exist until 2026-08-28. `decisions-m0.md` closed on
2026-08-18, so the closure is logged here, where the evidence was produced.

---

## 2026-08-28 — Offline ASR pinned: `faster-whisper small.en`, int8 CPU, greedy

**Decision: `small.en`, int8 on CPU, greedy decoding, Whisper
`EnglishTextNormalizer` (B5).** Pinned string:
`faster-whisper==1.2.1:small.en:int8:cpu:greedy`.

**Logged here on 2026-08-31.** The choice was made and used on 2026-08-28 but
lived only in `experiments/results/RESULTS.md`. It is a pinned component of the
measuring instrument, so it belongs in a decision log — CLAUDE.md requires the
exact instrument on every reported number, and RESULTS.md is a results table, not
a decision record. No number changes; this entry records what was already done.

### The evidence, 15 `eval_public` trials

| model | ceiling (clean target) | floor (mixture) | s/clip |
|---|---|---|---|
| `tiny.en` | 8.9 % | **123.0 %** | 0.7 |
| **`small.en`** | **4.0 %** | **64.6 %** | **3.0** |
| `medium.en` | 1.6 % | 50.4 % | 8.2 |

**Why not `tiny.en`.** Its floor exceeds 100 %, which means it inserts more words
than the reference contains — it hallucinates, and a metric built on it cannot
rank systems because the errors are not the target's words going missing, they
are invented words being added.

**Why not `medium.en`.** Better on both ends, but 2.7x the cost of `small.en`,
and that cost multiplies across every pass that needs it: M3's offline WER, M4's
k>=3 judge repeats, M6's comparison, and the text reference condition. The
accuracy gain does not change any ranking; the cost does change what fits in the
schedule.

**The n=15 figures above are MODEL-SELECTION evidence, not results.** B6 sets 200
as the minimum scored trial count. The numbers that may be quoted are the n=230
rescoring in the C2 entry below — the 12-trial pilot's 76.4 % floor was wrong by
19 points, and quoting it was the error that rescoring caught.

### Two roles, kept separate

`metric-definitions.md` 5 requires this. The same checkpoint serves both, which
is permitted but must be *stated*, since a shared error profile could flatter the
text condition:

| role | belongs to | inside the latency budget? |
|---|---|---|
| **response ASR** | the measuring instrument | no |
| **front-end ASR** | the system under test, text condition only (3.5) | **yes** |

J1's closure on 2026-08-31 (`decisions-m4.md`) removes the *response* role
entirely — an audio-in / text-out judge replies in text, so there is nothing to
transcribe. The front-end role remains.

### Known artefact, must be filtered

`small.en` emits the word **"you"** on digital silence — 8 of 8 absent trials.
Filter it before counting invented words, or B4's invented-speech row reports a
Whisper quirk as the extractor hallucinating. **Re-measure the artefact set if
the ASR is ever changed.**

### Hard constraint this creates

`small.en` may **not** be used as the differentiable training proxy. CLAUDE.md
requires the proxy to be a different model family from the judge, and separately
the evaluation instrument must not be trained against. See
`decisions-pending.md` D9.
---

## 2026-08-30 — C2 CLOSED. Task difficulty measured at n=230 and accepted

The last open M0 item. Scored from `experiments/results/transcripts.csv`
(`faster-whisper small.en`, int8 CPU, greedy, Whisper `EnglishTextNormalizer`
per B5), which already held 1,220 trials transcribed clean and mixed. No new
ASR was run — only the scoring, which had never been done at scale.

### What C2 asks

How hard the task should be, as two numbers: the **floor**, an off-the-shelf
ASR's word error rate on the raw mixture (how much of the target is lost if you
do nothing), and the **ceiling**, its WER on the clean target (the best anyone
could do). The gap between them is the headroom the extractor works in. A floor
too low makes the task trivial and the metric unable to separate systems; too
high and everything scores badly and again nothing discriminates. The declared
target band was 60-80 %.

### Measured, `both` condition only

| set | n | ceiling (clean) | floor (mixture) | mean SIR | interferer louder |
|---|---|---|---|---|---|
| `eval_public` | 230 | 6.1 % | **57.4 %** | +4.9 dB | 26 % |
| `sir0_val` | 103 | 5.8 % | **65.2 %** | -0.7 dB | 54 % |

**Decision: the measured range 57.4-65.2 % is accepted as the task difficulty.**
It straddles the lower edge of the 60-80 % band, and the band was a target set
before any data existed, not a constraint. Nothing is re-rendered and
`overlap_ratio` stays un-narrowed (B1 says narrow it last).

Plain reading of 57.4 %: for every 100 words the target speaker said, about 57
come out wrong. The failure is not mush — inspected on
`eval_public-42-000132`, the ASR transcribes the target perfectly for 17 words
and then **switches to the interferer's sentence**. The number is measuring "the
machine listened to the wrong person", which is exactly what the extractor is
built to prevent. The 51-point gap from 6.1 % to 57.4 % is the room available.

### This corrects the number of record by 19 points

`RESULTS.md` carried **76.4 %** from a 12-trial pilot. At n=230 it is **57.4 %**.
B6's 200-trial minimum exists for exactly this reason and the pilot was always
labelled as model-selection evidence, not the answer — but 76.4 % had already
been quoted as "the task's real floor" and must not be used again.

### Two things that must travel with the number

1. **Never quote the pooled figure.** `eval_public` pooled is 40.7 %, dragged
   down by `target_only` (floor 7.1 %, because with no interferer the "mixture"
   is already near-clean). The task's floor is the `both` row, always.
2. **The eval set and the training set are not the same difficulty.** Training
   is on `sir0`, symmetric by construction; `eval_public` keeps the original
   distribution where the target is the louder voice 74 % of the time. That is
   an 7.8-point difference in floor and it is a train/eval mismatch, not a
   measurement artefact. **Which set defines the benchmark is still open** and
   is now the more important question than the difficulty itself. Rendering a
   symmetric eval set costs ~2 min for 500 trials if the answer is the second.

### Consequences

- **C2 moves to closed in `decisions-pending.md`.** Accepted 2026-08-30; the
  supervisor conversation the item called for should confirm it rather than
  re-open it, and the eval-set question above is what that conversation is
  actually about.
- `eval_private` is also fully transcribed (500 trials) and stays held back. It
  was not scored here and must not be used for calibration.
- Absent trials carry no reference text, so they are not WER at all — they are
  the invented-words check, where `small.en` emits "you" on digital silence
  (8/8, 2026-08-28).
- The ceiling is ~6 %, not ~3 %: `small.en` on reverberant LibriSpeech is worse
  than the pilot suggested. Any claim of the form "we recovered X % of the
  ceiling" must use 6.1 %.

---

## 2026-08-31 — Floor and ceiling broken into error types: the floor is insertion-dominated

**Measured, not decided.** Recorded because the error split changes what the
extractor's job actually is, and because a bare word error rate cannot support
the claim below.

Same data as the C2 entry above — `eval_public`, `condition=both`, n=230, 6,471
reference words, `faster-whisper small.en`, Whisper `EnglishTextNormalizer`,
scored from `experiments/results/transcripts.csv` with no new ASR run.

Rates are each error type as a share of reference words, so the three sum to the
word error rate.

| system | LCF-WER | substitutions | deletions | insertions |
|---|---|---|---|---|
| **floor** (raw mixture) | 57.4 % | 23.2 | **3.5** | **30.8** |
| **ceiling** (clean target) | 6.1 % | 2.7 | 0.4 | 2.9 |

### What it means

**Doing nothing means the listener hears too much, not too little.** Only 3.5 %
of the target's words are lost outright. The dominant damage — 30.8 points, more
than half the floor's total error — is words being **added**: the other speaker
being reported as if they were the target.

**This reframes the task.** At these signal-to-interference ratios (mean +4.9 dB
on `eval_public`) the target is rarely buried. The job is overwhelmingly
*removing the other voice*, not *recovering hidden speech*. An extractor that
improves audibility without suppressing the interferer cannot move this number
much.

**It is also the empirical case for ICR.** Word error rate counts an inserted
interferer word and an inserted hallucination identically. Since insertions are
the floor's largest error type, the metric most needed alongside LCF-WER is the
one that says *whether the inserted words were the other speaker's* — which is
ICR (`metric-definitions.md` 3.2). Without it, the largest component of the
headline number is undiagnosed.

### Caveat that must travel with these numbers

**The alignment is chosen to minimise total edits, so the split between
substitutions and insertions is the cheapest explanation, not necessarily the
true mechanism.** When the ASR transcribes the target for a stretch and then
switches to the other speaker's sentence — observed on inspection, 2026-08-30 —
whether those words score as substitutions or insertions depends on the relative
lengths of the two utterances, not on what physically happened. Treat the
decomposition as suggestive and directional, and do not build a mechanism claim
on it alone.

### Why it is worth recording anyway

The two error types move in **opposite** directions as masking is made more
aggressive: harder masking removes the interferer (fewer insertions) and
introduces artefacts (more substitutions). So the decomposition is a cheap probe
of the artefact-versus-residue trade-off that `metric-definitions.md` 1
hypothesises about, and it needs no metric beyond the one already defined.

---

## 2026-09-01 — FIRST system row on the project's own metrics, and the model hurts easy trials

**The first end-to-end measurement of a trained extractor on LCF-WER, ICR and
NRR.** Everything before this was floor/ceiling anchors. Offline ASR standing in
for the judge, so this is not a live-model result — but the pipeline is proven
end to end and the numbers are real.

Checkpoint `models/model_sir0_5000-e7.pt` (4,976-trial run, `decisions-m2.md`
2026-09-01). Estimates in `experiments/results/2026-09-01-est-sir0-5000/`, 200
trials, CPU, whole-clip single forward pass. `sir0_val`, `condition=both`, n=103.

| system | LCF-WER | sub | del | ins | ICR@2 | mean leak | NRR |
|---|---|---|---|---|---|---|---|
| floor (unprocessed mixture) | 65.2 % | 32.9 | 9.3 | 23.1 | 67.0 % | 51.3 % | 0.0 % |
| **the model** | **59.1 %** | 28.1 | **12.6** | 18.4 | **54.4 %** | **39.1 %** | 1.0 % |
| ceiling (clean target) | 5.8 % | 3.4 | 0.7 | 1.7 | 0.0 % | 0.0 % | 0.0 % |

**Headroom captured:** LCF-WER **10.3 %** of the 59.4-point band; ICR@2 **18.8 %**
of its 67-point band; mean leakage **23.8 %**.

**The model is better at removing the interferer than at making the target
intelligible** — it captures roughly twice as much ICR headroom as LCF-WER
headroom. That asymmetry is invisible to a word error rate on its own and is the
first concrete thing ICR has told us that LCF-WER could not.

### The finding: it helps hard trials and HURTS easy ones

| floor difficulty | n | mean change in LCF-WER | trials improved |
|---|---|---|---|
| easy, floor <25 % | 22 | **−4.2 pts (worse)** | 18 % |
| medium, 25–60 % | 27 | −0.6 pts | 44 % |
| hard, 60–100 % | 27 | +9.0 pts better | 37 % |
| very hard, >100 % | 27 | **+23.1 pts better** | 63 % |

`correlation(floor WER, improvement) = +0.33`. Overall: **43 trials improved
(mean +29 pts), 31 worsened (mean −16 pts), 29 unchanged.** The regressions
cancel most of the gains, which is why the aggregate moves only 6.1 points while
individual trials move by 100+.

**This is the artefact-versus-residue trade-off, measured.** On an easy trial
there is little interferer to remove, so nearly everything the extractor does is
introduce distortion. On a hard trial the interferer dominates and removing it
more than pays for the distortion. **The optimum is therefore not "always
filter"**, which is the direct motivation for the mixture/estimate interpolation
sweep.

**The error decomposition says the same thing from another angle.** Insertions
fell 23.1 -> 18.4 (interferer removed) and substitutions 32.9 -> 28.1, but
**deletions ROSE 9.3 -> 12.6** — the extractor removes some of the target along
with the interferer.

### A divergence result already exists, without a judge

On these same 103 trials, whole-clip SI-SDR improved by a **mean of +1.99 dB**
(median +1.40, better on 78 % of trials) — a clean win by the conventional
measure. The content metric says the model made **30 % of trials worse**, and
systematically the easy ones.

**Conventional signal quality and content fidelity disagree on this model, on
this data, today.** That is a miniature of the thesis's central claim, obtained
before the judge exists. It must be labelled as measured through an ASR rather
than a live model, and the SI-SDR figure here is whole-clip and unfloored, so it
is NOT the same quantity as the `L_pres` reported in training.

### What the good cases look like

Worth keeping for the write-up. `sir0_val-42-000152`, SIR +9.9 dB: the
unprocessed mixture had the transcriber report the interferer's entire sentence
before the target's, WER 195 %, 15 leaked words. The model's output transcribed
**word-for-word correct, WER 0 %, 0 leaked**.

`sir0_val-42-000050`, SIR +0.9 dB, is the more diagnostic one: near-equal
loudness and the interferer **interleaved throughout** rather than prepended.
WER 138 % -> 18 %, leaked 17 -> 0, and the residual errors are mishearings
(`snare` for `snake`) rather than leakage. The model could not have solved that
by keeping the louder voice or by taking the first speaker.

### One genuine defect

`sir0_val-42-000145`, floor WER 40 % -> 100 %. The output transcribes as
`the the the the the the the the the the the the the the` — a **degenerate
collapse on a single trial**, on a trial that was already easy. NRR caught it
(1.0 % against the floor's 0.0 %), which is the metric doing its job on the
first system it has ever scored. Listen to the file: if the output is mush
rather than speech this is a stability bug, not a quality issue.

### Status of the third metric

NRR is 0.0 % at the floor and 1.0 % for the model. It remains near-useless with
an ASR standing in for the judge, because a transcriber cannot decline — the one
non-zero entry is the degenerate trial above. Read it as "not yet measurable",
not as a result.
