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

**Headroom captured:** LCF-WER **10.4 %** of the 59.4-point band; ICR@2 **18.8 %**
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

---

## 2026-09-01 — SIR/SAR added. The artefact hypothesis is NOT supported in the form predicted

**Implemented** `src/live_model_metric/separation.py`, 12 tests. Splits what an
estimate contains beyond the target into **interference** (the model failed to
remove it) and **artefact** (the model invented it). Borrowed from the BSS_EVAL
decomposition of Vincent et al. (2006); scale-invariant framing after Le Roux et
al. (2019). Both cited as borrowed.

**Method.** The estimate is projected onto the span of the three true sources.
Whatever no scaled combination of them explains was not in the microphone signal,
so the model created it. **Artefact is defined by elimination, not subtraction** —
plain subtraction would give `estimate - mixture`, which is large for any working
extractor and measures nothing useful.

Requires all three clean sources (`noise = mixture - target - interferer`, as the
loader already derives it), so it is **computable on constructed mixtures only,
never on AMI**.

### Three design decisions, taken

1. **Scale-invariant**, so the scores are commensurable with the training
   objective's SI-SDR.
2. **Noise counts as a source, not as artefact.** Noise leaking through is a
   failure to remove something that was genuinely present; calling it invention
   would be wrong, and the hypothesis is about *processing* artefacts.
3. **Scaling only, no allowed filter.** BSS_EVAL variants permit a filter before
   calling the residual an artefact; the stricter scaling-only form is used, and
   the reference is already the reverberant target so the honest filtering is
   done.

**A bug the tests caught.** An absolute epsilon in the denominator made SAR
*not* scale-invariant — it swung 17 dB under a 7.5x gain, because when the
artefact part is near zero the score becomes a function of the estimate's gain
rather than its content. Fixed by flooring the denominator **relative** to the
numerator with `TAU = 1e-3`, the same value and reasoning as `tau_pres` in the
training objective. This caps every score at **+30 dB** and makes the scores
exactly scale-invariant across a 1000x gain range.

### Measured on the 5,000-trial checkpoint, sir0_val `both`, n=103

| | change vs doing nothing |
|---|---|
| SIR (interferer removed) | **+4.33 dB** |
| SAR (artefact avoided) | **-19.66 dB** |
| SDR (the net) | +1.98 dB |

Absolute SAR of the model output: **+10.34 dB**, where +30 dB is the artefact-free
ceiling. So roughly **9 % of the output's energy is invented** — the artefact is
real and substantial.

**Consistency check:** SDR +1.98 dB against +1.99 dB computed independently with
a separate SI-SDR implementation. Two code paths, 0.01 dB apart.

### THE PREDICTION FAILED, and this is the finding

The stated prediction was: if processing artefacts are what damage the listener,
word-error improvement will correlate with SAR.

| | correlation with LCF-WER improvement |
|---|---|
| delta SDR | **+0.20** (marginal at n=103, p about 0.04) |
| delta SIR | +0.16 (not significant) |
| **delta SAR** | **-0.05 (null)** |

**Trial-level artefact severity does not predict trial-level word errors.** The
artefact hypothesis of `metric-definitions.md` 1 is **not supported in that
form.** Recorded as a failed prediction rather than quietly dropped.

### What is actually happening: the model does not adapt

| bucket | dSIR | dSAR | dSDR | dLCF-WER |
|---|---|---|---|---|
| easy, floor <25 % | +3.80 | -17.45 | +1.34 | **-4.2 (worse)** |
| medium | +4.06 | -18.33 | +1.69 | -0.6 |
| hard | +5.06 | -21.05 | +2.53 | +9.0 |
| very hard >100 % | +4.32 | -21.41 | +2.24 | **+23.1 (better)** |

**The signal columns are flat; only the outcome swings.** The model applies the
same transform to easy and hard trials alike. The easy-trial regression is
therefore *not* the model producing worse artefacts there — **the same artefact
costs nothing when there was a lot of interference to remove, and costs dearly
when there was not.** The trade lives in the input, not the output.

It is not structurally incapable of adapting — it sees the mixture and the
enrolment — it empirically **does not**, because `L_pres` is an average over
crops in which suppression pays and artefact costs the same everywhere. No
per-trial decision ever enters. See `decisions-pending.md` D11.

### The result this actually produces

**No signal-domain measure meaningfully predicts what the listener recovered.**
The best, delta SDR, explains about **4 % of the variance** in word-error
improvement. That is the divergence claim quantified, and it is a more general
statement than a rank inversion.

Caveats that must travel with it: n=103, so only delta SDR is marginally
significant; and this is measured **through an ASR, not a live model**, so the
live-model correlation is a separate open question.

### Reading SAR without being misled

**Delta SAR always looks catastrophic and that is an artefact of the reference.**
The mixture is artefact-free *by construction* — it is exactly the sum of the
sources — so it sits at the +30 dB ceiling and any processing whatsoever drops
below it. **Quote the absolute SAR (+10.34 dB), not the delta.**

---

## 2026-09-01 — Mix-back sweep: no global optimum, but the per-difficulty optimum spans the FULL range

D11's screening test, run as an instrument rather than a fix. `sir0_val`
`condition=both`, n=103, alpha in {0, 0.25, 0.5, 0.75, 1}, offline ASR.
`experiments/results/sweep_alpha_rows.json`, transcripts cached in
`sweep_alpha_transcripts.csv`. 309 new transcriptions, ~15 min. **No retraining
and no extra forward passes** — every alpha is a linear blend of two signals
already on disk.

| alpha | LCF-WER | ICR@2 | SDR | SIR | SAR |
|---|---|---|---|---|---|
| 0.00 (do nothing) | 65.2 % | 67.0 % | −1.12 | −1.12 | +30.00 |
| 0.25 | 63.4 % | 66.0 % | −0.61 | −0.58 | +24.95 |
| 0.50 | 69.6 % | 67.0 % | −0.02 | +0.15 | +19.21 |
| 0.75 | 67.2 % | 62.1 % | +0.58 | +1.21 | +14.56 |
| **1.00 (current model)** | **59.1 %** | **54.4 %** | +0.86 | +3.21 | +10.34 |

**The instrument validated itself.** SDR, SIR and SAR are all perfectly
monotonic in alpha, exactly as a linear blend predicts. The blend is provably
linear — the iSTFT is linear, so waveform blending and mask interpolation are the
same operation — so nothing is confounded by the blending itself.

### Globally, alpha = 1 wins. There is no interior optimum.

The model is already at the best *global* aggressiveness. **This is evidence
against a global loss-side shift toward gentler masking**, i.e. against
`BETA > 1` as a standalone intervention.

Caveat on how far that generalises: the sweep is an **imperfect proxy** for
`BETA`. A model retrained at higher `BETA` learns a *different mask*; it is not
the same mask blended with its input. The sweep tests the direction — "is gentler
better globally?" — and the answer is no, but it does not rule `BETA` out.

### The decisive result: the optimum varies across the entire range

| difficulty | n | a=0 | a=0.25 | a=0.5 | a=0.75 | a=1 | best |
|---|---|---|---|---|---|---|---|
| easy <25 % | 22 | **9.6 %** | 11.2 % | 10.3 % | 11.7 % | 14.2 % | **0.00** |
| medium | 27 | 40.5 % | **36.8 %** | 58.6 % | 46.6 % | 40.9 % | **0.25** |
| hard | 27 | 81.9 % | 80.6 % | 82.7 % | 79.1 % | **73.4 %** | **1.00** |
| very hard >100 % | 27 | 137.4 % | 133.8 % | 131.4 % | 140.1 % | **113.6 %** | **1.00** |

**Easy trials want no filtering at all. Hard trials want full filtering.** A
single global constant therefore cannot be right, and the global answer is
alpha = 1 only because the hard buckets dominate the corpus: they gain 24 points
at alpha = 1, swamping the 5 points the easy trials lose.

### What an adaptive gate could buy

| | LCF-WER |
|---|---|
| do nothing | 65.2 % |
| current model | 59.1 % |
| **oracle, best alpha per difficulty bucket** | **56.9 %** |
| oracle per trial (cheats — uses the answer) | 52.5 % |

**A realistic gate is worth about 2.2 points**, with 6.6 as an unreachable
ceiling. Real but modest, and that is the number to weigh against a 6.2 h retrain.

### The WER curve is non-monotonic, and that is itself the finding

65.2, 63.4, **69.6**, 67.2, 59.1. alpha = 0.5 is *worse than doing nothing*,
which makes no physical sense against monotonic signal measures.

It is not a blending artefact — the blend is linear, verified. It is **n=103
noise plus transcriber nonlinearity**: the listener's behaviour is not a smooth
function of signal quality. **Trust the endpoints and the per-bucket pattern; do
not read individual interior alpha values.**

And note what it *is*: the signal moved perfectly smoothly across five settings
while the content outcome jumped around. That is another instance of this
project's central claim, obtained for free.

---

## 2026-09-01 — DNSMOS added, and it disagrees with the content metric

**Implemented** `src/live_model_metric/dnsmos.py`. Four scores per clip, all 1-5
and higher-is-better: `P808`, and `SIG` / `BAK` / `OVRL` from P.835. Ported from
`microsoft/DNS-Challenge`, `DNSMOS/dnsmos_local.py`, retrieved 2026-09-01.
Reddy et al. (2021) for P.808, Reddy et al. (2022) for P.835, both cited as
borrowed.

**Non-intrusive: it needs only the degraded audio.** No clean reference, no
transcript. **It is therefore the only quality metric in this project that can be
run on AMI**, which makes it load-bearing for the real-audio transfer check rather
than merely another column.

### Validated against the reference implementation

The reference script's own class was run against the port on the same clip:

| | reference | port | difference |
|---|---|---|---|
| P808 | 2.146801 | 2.146801 | 2.4e-07 |
| SIG | 3.196597 | 3.196597 | 4.4e-16 |
| BAK | 3.030237 | 3.030237 | 0 |
| OVRL | 2.196043 | 2.196043 | 4.4e-16 |

Exact on three, float32 rounding on the fourth, same segment count. **The numbers
are DNSMOS's, not an interpretation of DNSMOS.**

### Both variants, and why

`metric-definitions.md` 6 amended 2026-09-01 to report **both**. The original
named P.808 alone, losing the point of 4; a first amendment named P.835 alone,
losing the field's convention. The history is the reason:

| variant | role |
|---|---|
| **P.835** -> `SIG`, `BAK`, `OVRL` | **`OVRL` is the score that got gamed** |
| **P.808** -> one score | **the replacement the organisers switched to** |

### Three implementation facts that were easy to get wrong

1. **Personalised is a SEPARATE MODEL FILE**, not merely different coefficients:
   `pDNSMOS/sig_bak_ovr.onnx` against `DNSMOS/sig_bak_ovr.onnx`. Model and
   coefficient set must match. **Personalised is correct here, because target
   speaker extraction IS personalised speech enhancement** — the distinction
   exists precisely because in this task the right output removes a speaker, and
   the standard model can score that removal as damage.
2. **The polynomial correction applies to P.835 only.** P.808 is used raw. The
   raw P.835 outputs are not MOS scores, and skipping the correction produces
   plausible but wrong numbers with no error.
3. **Short clips are LOOPED, not zero-padded** — the reference doubles the audio
   until it fills one 9.01 s segment, because silence would be scored as bad
   audio. Does not trigger on 15-20 s trials but is replicated faithfully.

Models snapshotted to `src/live_model_metric/dnsmos_onnx/` (2.5 MB) with SHA-256
recorded, for the same reason the normaliser and stopword list are pinned: DNSMOS
is a learned model and drifts between releases. `librosa==1.0.0` added, needed to
match the reference mel filterbank exactly.

### Measured, sir0_val `both`, n=103, personalised

| system | P808 | SIG | BAK | OVRL |
|---|---|---|---|---|
| floor (do nothing) | 2.913 | **4.090** | 2.031 | **2.497** |
| **the model** | 2.937 | **3.366** | **2.266** | **2.237** |
| ceiling (clean target) | 3.550 | 4.175 | 3.592 | 3.429 |
| **change, floor -> model** | **+0.02** | **-0.72** | **+0.24** | **-0.26** |

### The SIG/BAK prediction held: two instruments, one conclusion

Predicted from the SAR result: `BAK` up (suppression works), `SIG` down
(artefacts). **Both happened, and SIG fell three times as far as BAK rose.**

So **`SIG` corresponds to artefacts introduced and `BAK` to interference
removed**, empirically, which is the perceptual analogue of the signal-domain
SAR/SIR split. The model invents ~9 % of its output energy (SAR +10.34 dB), and
an independent human-perception model agrees that the damage outweighs the
cleanup.

### THE DIVERGENCE, and it runs opposite to the prediction

| | direction |
|---|---|
| LCF-WER | 65.2 % -> **59.1 %**, **better by 6.1 points** |
| DNSMOS OVRL | 2.497 -> **2.237**, **worse by 0.26** |

**A human listener would say the model made the audio worse. The listener
recovered more of the words.**

The stated prediction was the reverse — DNSMOS rising while content fell.
**Recorded as another failed prediction.** The direction observed is arguably the
stronger result: **the conventional perceptual metric would have rejected a system
that measurably helps the downstream task**, which is exactly the failure mode
`metric-definitions.md` 1 argues conventional metrics have.

### Two further observations

**P808 is flat, +0.02.** The metric the organisers switched *to* is close to blind
to what this model does, while `OVRL` — the one that was gamed — moves. Worth
stating about both.

**The ceiling is 3.43 of 5, not near-perfect**, because the reference is the
*reverberant* target (A1) and DNSMOS was trained on denoising, not
dereverberation. Same lesson as the offline ASR ceiling at 6.1 % rather than 0 %:
**the instrument's own ceiling must be quoted with any score.**

---

## 2026-09-01 — RTF and end-to-end latency measured. It keeps up on a laptop CPU

Closes the M3 item "measured algorithmic latency + RTF against the ~200-300 ms
budget". `scripts/measure_rtf.py`, result in
`experiments/results/2026-09-01-rtf-cpu/rtf.json`.

### Two numbers, one measurement

| | definition | requirement |
|---|---|---|
| **RTF** | processing time / audio duration | **< 1**, else the backlog grows without bound |
| **latency** | chunk + lookahead + processing | < 200-300 ms |

**The RTF deadline is the tighter of the two.** Processing must finish inside the
chunk's own duration; at 80 ms that is stricter than the latency budget would
permit. **If RTF < 1 the latency budget is met automatically**, so RTF is the
number to lead with.

### Why chunks, and why 80 ms

**Timing a whole clip flatters streaming by up to 32x**, measured 2026-09-01: a
whole-sequence LSTM call is one batched matrix multiply over every frame, whereas
streaming is thousands of tiny matrix-vector products and becomes launch-latency
bound rather than compute bound. **Whole-clip RTF is not evidence of streaming
ability**, and the previously quoted 0.42 from `make_estimates.py` should not be
read as one.

The chunk-size sweep, one LSTM, us per frame:

| chunk | 8 ms | 40 ms | **80 ms** | 160 ms | 500 ms | whole clip |
|---|---|---|---|---|---|---|
| us/frame | 245.9 | 45.4 | **26.4** | 16.1 | 9.9 | 7.5 |
| speedup | 1.0x | 5.4x | **9.3x** | 15.2x | 24.9x | 32.6x |

**It saturates early.** 8 -> 80 ms buys 9.3x; 80 ms -> whole clip buys only 3.5x
more. And latency is `chunk + 40 + compute`, so 160 ms chunks would sit exactly
on the 200 ms limit while 80 ms leaves real margin. **80 ms chosen.**

### Measured, i5-1135G7, 4 threads, 500 chunks

| | value | requirement | met |
|---|---|---|---|
| per chunk, mean | 42.24 ms | — | |
| per chunk, **p99** | **56.50 ms** | **< 80 ms** | **yes** |
| per chunk, max | 58.24 ms | < 80 ms | yes |
| **RTF mean** | **0.528** | **< 1** | **yes** |
| RTF p99 | 0.706 | < 1 | yes |
| **latency mean** | **162.2 ms** | 200-300 ms | **yes** |
| latency p99 | 176.5 ms | 200-300 ms | yes |

= 80 ms chunk + 40 ms lookahead + 42.2 ms compute. **The model keeps up on a
laptop CPU and no chunk misses the deadline.**

### Caveats that must travel with these numbers

**This is an ESTIMATE, not a streaming measurement.** The model has no stateful
streaming path, so chunks are processed **independently** and the output is
discarded. The timing is close because passing a carried hidden state costs the
same as passing a fresh one, but two things are missed: the cost of state
management, and a difference in frames-per-chunk because `forward()` pads each
chunk where a streaming STFT would not. **10-20 % error.**

**The margin is real but not generous.** RTF p99 of 0.706 leaves 29 % headroom on
the worst chunks. On a loaded machine that could be exceeded — this project has
already seen a test suite go from 40 s to 16 min under memory pressure.

**Enrolment is not the bottleneck.** Per-chunk time is flat across enrolment
lengths of 0.5-5.0 s (41.7-43.1 ms), so there is no easy win from caching the
enrolment embedding.

**A prediction that was 8x wrong, recorded as such.** Extrapolating from a single
LSTM predicted ~5 ms per chunk; the truth is 42 ms. The extrapolation ignored the
32-band estimator trunks, the STFT/iSTFT and the conditioning. **Do not
extrapolate compute from one layer.**

### Still open

**The GPU figure.** No local GPU. The spec assumes server-class compute
(`decisions-m0.md` 2026-08-07), so **the GPU number is the one that supports the
claim** and CPU is the pessimistic case. One Kaggle cell.

**B11's latency decay curve is a different thing** and is expensive:
`lookahead_frames` shifts the feature sequence at TRAIN time, so a proper
quality-versus-latency curve needs one retrain per point. Evaluating a
lookahead-0 model at other lookaheads is not valid.

## 2026-09-03 — WeSep cannot stream. `causal: true` covers the separator, not the normaliser

**Measured, not inferred.** `scripts/probe_wesep_causality.py`, checkpoint
`tfmap_context_causal_100`, 8 s mixture / 2 s enrolment, seed 42, CPU.

| probe | cut 2 s | cut 4 s | cut 6 s | worst |
|---|---|---|---|---|
| determinism floor (same input twice) | — | — | — | 1.21e-05 |
| **A. scale-matched future** | 1.12e-02 (3.49 %) | 1.02e-02 (2.66 %) | 5.39e-03 (1.40 %) | **1.12e-02** |
| B. 5x louder future | 2.16e-02 (6.74 %) | 2.89e-02 (7.53 %) | 1.38e-02 (3.60 %) | 2.89e-02 |

Ours on the same protocol, 2026-08-24: **1.68e-08**. WeSep is ~6 orders of
magnitude worse, and ~900x its own determinism floor.

### The mechanism, so this is a finding rather than a number

`causal: true` in their config sits under `separator:` and is **not a false
claim** — the separator's RNNs are causal. But `SubbandNorm`
(`wesep/modules/separator/bsrnn.py:44`) builds `select_norm('GN', …)` =
`nn.GroupNorm(group=1, C)` and applies it to a `(B, C, T)` tensor, so it
normalises over all channels **and the whole time axis** before the causal
separator runs. That is global layer norm (`gLN`): every output frame is a
function of the entire clip's statistics, future included.

**Two checks that it is global normalisation and not architectural lookahead.**
The leak *shrinks* as the cut moves later (3.49 → 2.66 → 1.40 %) — a perturbation
covering less of the clip shifts the global statistic less, whereas a fixed
lookahead window would read ~0 once the cut is beyond it. And the 5x-louder probe
leaks only ~2.6x more, not unboundedly more.

### Consequences

- **WeSep is an offline system for our purposes.** Its 2026-09-03 judge and
  signal scores stand, as *offline* scores. Do not describe it as a streaming or
  causal baseline anywhere in the thesis.
- **Fixable in principle, not by us.** Cumulative layer norm is the standard
  causal replacement, but swapping it means retraining their checkpoint. Out of
  scope, and it is not our model to fix.
- **Our own causal claim is unaffected and now has a contrast.** 1.68e-08 vs
  1.12e-02 is a genuine architectural difference worth one line in the report:
  a model can declare causality per-module and still be non-causal end to end.
- **Latency measurement is now academic** but was taken anyway (below).

### Two caveats that travel with these numbers

**The model is not bit-reproducible.** Determinism floor 1.21e-05 on identical
input, most likely multi-threaded reduction order. The effect is ~900x that, so
the verdict holds, but never quote the probe values without the floor.

**Mode A equalises broadband RMS, not per-subband RMS**, so white noise at
matched RMS still presents very different per-band statistics to a per-band
normaliser. Mode A is therefore a better control than mode B but not a clean
isolation of lookahead from global normalisation — the `GroupNorm` reading above
is what actually separates them. A cleaner probe would perturb inside one band.

### Latency, taken with `scripts/measure_rtf_wesep.py` (new)

Protocol copied term for term from `measure_rtf.py` (80 ms chunks, 20 warmup,
same percentiles) so the rows are the same quantity. `measure_rtf.py` could not
be reused: it calls `build_model()`/`load_state_dict()` from `train.py`, and the
WeSep venv has none of our model code.

**Measured, 2250 chunks, 23 min wall**, same i5-1135G7 / 4 threads as the
baseline. `experiments/results/2026-09-03-rtf-wesep-cpu/rtf.json`.

| | WeSep | our baseline |
|---|---|---|
| RTF mean | **2.854** | 0.528 |
| RTF p99 | 5.300 | 0.706 |
| latency mean | **348.3 ms** | 162.2 ms |
| latency p99 | 544.0 ms | 176.5 ms |
| per-chunk mean / p99 / max | 228.3 / 424.0 / 552.9 ms | — |
| meets budget | **no** | yes |

**No chunk finishes inside its own 80 ms** (min observed well above the
deadline), so the input backlog grows without bound. A 20-chunk smoke run earlier
the same day read 4.85; that was too few samples — **quote 2.854**.

Two reasons for the gap: 27.2 M parameters in the timed forward against our
7.19 M, and WeSep re-embeds the 5 s enrolment through its speaker branch on every
80 ms chunk. Our model re-embeds per chunk too, so the protocol is matched, but
the cost is wildly asymmetric — a full speaker encoder against our TF-Map cue.
`--cache-fbank` gives a lower bound; neither mode hoists the speaker encoder.

**Parameter counts differ by basis and both are correct.** 33.46 M is the
`avg_model.pt` state-dict total (project-state.md, includes the jointly-trained
VoxCeleb ECAPA-TDNN); 27.2 M is the TSE model as instantiated, which is what was
timed. Label which one you are quoting.

Not comparable to any published REAL-TSE latency figure — different hardware,
chunking and latency convention.

---

## 2026-09-12 — the evaluation battery is one command, and it runs in two stages

`scripts/run_eval_suite.py`. Seven steps in a fixed order — estimates, signal,
asr, judge, rtf, cue, leakage — against one checkpoint and one `--tag` that
names every output directory.

**Why it exists.** Every evaluation until now was assembled by hand from
`docs/run_times.md`, which is how `2026-09-06-eval-10000-signal-perceptual` came
to sit on a different flag set from `2026-09-04-train-sir0-10000`. Three steps
consume the first step's output directory and all of them need the same
`--split` and `--condition` or the numbers are not comparable. This makes the
battery one recorded definition instead of a habit.

**It orchestrates and measures nothing.** Every number still comes from the
script that owns it, in that script's own results directory with its own meta.
No new metric, no changed default.

### Run it in two stages. The gate is `asr`

**Stage 1, `--steps estimates,asr`, ~33 min, no API budget.** ICR@2 and
`mean_leak` are the direct measures of interferer leakage, which is 58.5 % of
our content-word error mass and correlates +0.622 with per-trial WER
(decisions-pending.md D14, 2026-09-11). An arm aimed at leakage either moves
them or it has failed on its own logic.

The numbers to beat, `model_sir0_10000-e6.pt`, `sir0_val` `both`, n=103:

| system | LCF-WER | ICR@2 | mean leaked |
|---|---|---|---|
| floor, raw mixture | 65.22 | 66.99 % | 51.30 % |
| control, epoch 6 | **59.52** | **50.49 %** | **34.58 %** |
| ceiling, clean target | 5.85 | 0.00 % | 0.00 % |

**Stage 2, `--steps judge,rtf,cue`, ~31 min + API.** Only if stage 1 moved.

**`signal` is optional and least informative.** SI-SDR and DNSMOS score the one
axis this project does not optimise, and a leakage-targeted arm is expected to
lose on it. Run it for the write-up, not for a decision.

**Wall times printed by `--dry-run` are measured rows from `run_times.md`, not
estimates.** `diagnose_cue.py` has no recorded row and prints as unmeasured
rather than guessed.

`--limit N` is for smoke tests only and puts the limit in the output directory
name, so a two-trial run cannot be mistaken later for a published one.

---

## 2026-09-12 — MEASURED: what `sir0_val` at n=103 can and cannot resolve. It cannot resolve 2 points

Paired bootstrap over trials, 10,000 resamples, seed 42, on the per-trial rows
in `experiments/results/2026-09-12-leakage-share/per_trial.json`. Both systems
resampled on the SAME trial indices each draw, so this is the paired difference,
not two independent intervals.

| quantity | point | 95 % interval | resamples with the opposite sign |
|---|---|---|---|
| LCF-WER, state arm minus control | **+1.74** | **[−4.77, +10.85]** | **36.8 %** |
| mean leaked %, arm minus control | −1.93 | [−5.28, +1.30] | 12.0 % |

Per trial, the two systems are a coin flip: **WER better on 33, worse on 36,
tied on 34.** Leakage: better on 19, worse on 16, tied on 68.

**NEITHER the harm nor the benefit of the state-teacher arm is established.**
The +1.72 LCF-WER is deep inside trial-sampling noise, and so is the −3.88
ICR@2 that looked like the arm's success. Both readings from 2026-09-12 stand as
directions, not as effects.

### The number to plan with

**The 95 % interval on a system difference is about ±8 LCF-WER points at n=103.**
Halving that needs 4x the trials; resolving a 2-point effect needs roughly
(7.8/2)² ≈ **15x, about 1,570 trials**.

**Consequence, and it governs the rest of M5: an arm expected to move LCF-WER by
less than ~5 points is not measurable on this evaluation as constructed.** The
differences we CAN resolve are the big ones — our 59.5 against WeSep's 34.6
(25 points) and against the ceiling's 5.8 (54 points).

### This is a different noise source from the judge SEM, and they compose

`project-state.md` records **judge SEM ≈ 0.5 over 103** — how much the aggregate
moves if you re-ask the SAME judge about the SAME trials. That is listener
repeatability. What is measured here is TRIAL SAMPLING: how much a system
difference moves if you had drawn a different 103 trials from the same
distribution. They answer different questions and both are real.

**Which one applies depends on the claim being made, and the thesis needs both
stated.**

- *"System A beats system B on this benchmark"* — a fixed test set, leaderboard
  style. Listener repeatability (~0.5) is the relevant floor, and +1.72 clears it.
- *"This method reduces LCF-WER"* — a claim about the method, which is what a
  thesis argues. The trial-sampling interval applies and **nothing is
  established.**

Report the generalisation reading as the headline. A benchmark-local win on 103
trials that vanishes under resampling is not a method result.

### What is still NOT measured

Run-to-run training variance — two identical configs at different seeds. It sits
ON TOP of everything above and no same-config replicate exists anywhere in
`experiments/results/`. The intervals here are therefore a LOWER BOUND on the
scatter.

### Cost of this measurement

Seconds, on data already on disk. It should have been run before the first arm.

---

## 2026-09-12 — CORRECTION: the benchmark's resolution depends on how PAIRED the comparison is. Same-model interventions resolve ~3 points

Earlier today this file recorded "an arm expected to move LCF-WER by less than
~5 points is not measurable on this evaluation as constructed." **That is true
only for comparing two INDEPENDENTLY TRAINED models. It is wrong as a general
statement and the difference is large enough to change how M5 should be run.**

| comparison | n tied trials | measured | 95 % interval | verdict |
|---|---|---|---|---|
| state extension vs baseline (two trained models) | 34 / 103 | +1.72 | [−4.77, +10.85] | inside noise |
| mask floor 0.05 vs baseline (one model, one knob) | **53 / 103** | +3.81 | **[+1.09, +6.78]** | **REAL** |

**The interval halved because half the trials produce an IDENTICAL transcript.**
Two separately trained models differ on every trial, so trial-sampling noise
enters twice. One model with an inference-time setting changed differs only where
the setting bites, and the paired bootstrap sees that.

**Consequence for planning, and it is the useful half of today:** an idea that can
be tested as an inference-time intervention on a fixed checkpoint is measurable
at roughly 3 points. The same idea tested by retraining is not measurable below
~8. **Test at inference first, always, wherever the idea admits it.** That is
cheaper AND more sensitive, which is a rare combination.

### The mask floor is a MEASURED failure, with a measured mechanism

Floor 0.05 on `model_sir0_10000-e6.pt`, paired bootstrap, 10,000 draws, n=103:

| | measured | 95 % interval | |
|---|---|---|---|
| deletions | **−3.14** | [−6.93, −0.21] | **REAL, better** |
| substitutions | **+4.10** | [+1.56, +7.33] | **REAL, worse** |
| mean leaked % | **+5.39** | [+1.60, +9.67] | **REAL, worse** |
| insertions | +2.85 | [−0.33, +6.33] | inside noise |
| LCF-WER | **+3.81** | [+1.09, +6.78] | **REAL, worse** |

**Read in plain words: filling the mask's holes genuinely recovers target words
the model was deleting, and genuinely costs more misheard words and more of the
other speaker, and the second effect is bigger.**

**Both halves are real, and that is the point.** The holes were destroying target
speech — that is now measured, not inferred. And filling them with the raw
mixture is the wrong cure, because the mixture contains both speakers. What is
wanted is a target-selective fill.

### The full sweep, three settings

| floor | deletions | mean leaked % | LCF-WER |
|---|---|---|---|
| none | 11.79 | 34.58 | **59.52** |
| 0.05 | 8.64 | 40.87 | 63.33 |
| 0.10 | 9.98 | 41.32 | 61.87 |
| 0.20 | **8.09** | **44.56** | 62.34 |

**Leakage rises monotonically with the floor, with no exceptions** -- that is the
trend to trust. Deletions fall, best at 0.20 (-3.70). LCF-WER is worse at every
setting and is not monotonic among the three, which is what noise at this n looks
like: only the 0.05 deletion result and the substitution results clear the
interval.

The verdict is consistent across all three settings and rests on none of them
alone.

**Keep the floor as a knob, not as a setting.** It is the only mechanism in the
project that trades deletions against leakage at a measured exchange rate.

---

## 2026-09-12 — THE METRIC'S IRRELEVANCE FLOOR, measured by accident: 1.57 points for doing nothing

`report-todo.md` #9 has asked since M4 for a noise floor on the content metric:
"a system difference smaller than it cannot honestly be claimed." Here is one,
and it cost nothing because it fell out of a control arm.

**The null intervention.** `scripts/postprocess_mask.py` at `--down 1.0` is
arithmetically the identity: it takes the STFT of a finished estimate, multiplies
by exactly 1.0, and inverts. The only change is the analysis/synthesis round
trip. Measured over the 103 scored clips, the output differs from its source by a
**median of -61.7 dB** relative energy (worst -27.8, best -71.4). That is roughly
one part in a thousand of the amplitude: inaudible, and orders of magnitude below
any artefact the model itself produces.

**What it did to the metric.**

| | baseline | null intervention |
|---|---|---|
| LCF-WER | 59.52 | **57.95** |
| transcripts changed | — | **39 of 103** (22 better, 17 worse, 64 tied) |

**Doing nothing to the audio moved the headline number 1.57 points in the
model's favour and rewrote 38 % of the transcripts.**

### Consequences, and they are not small

1. **1.57 points is the irrelevance floor.** Any claimed improvement at or below
   it can be produced by a change that is definitionally meaningless. This is a
   SEPARATE and tighter bar than trial-sampling noise, and it applies even when
   the test set is held fixed.
2. **It calibrates the ASR stand-in, not the judge.** `small.en` is deterministic
   given its input; the instability is the input crossing decision boundaries,
   not sampling. The judge has its own floor (SEM ~ 0.5, `project-state.md`), and
   they do not substitute for each other.
3. **Report it beside every content number from now on.** Two of today's results
   sit near it: the state extension's +1.72 and this. One sits clearly above it:
   the mask floor's +3.81.

### It also validates the paired bootstrap, which is why this was worth the hour

The bootstrap was given a change known to mean nothing and correctly refused it:

| comparison | measured | 95 % interval | verdict |
|---|---|---|---|
| null intervention vs baseline | -1.57 | [-4.94, +1.75] | **inside the noise** |
| mask floor 0.05 vs baseline | +3.81 | [+1.09, +6.78] | **outside** |

A method that certified the null as real would have been worse than no method.
It did not. **`scripts/bootstrap_difference.py` is now a validated instrument and
should gate every content claim this project makes.**

### And it revises this morning's rule

The earlier entry today said same-model interventions resolve about 3 points
because half the clips tie. That stands, with the floor attached: **the usable
range for a same-model intervention is above ~1.6 points, not above zero.** Below
that, pairing buys precision around a number that is not measuring the
intervention.

---

## 2026-09-12 — the gain-following floor is UNTESTED, not refuted. The normaliser was wrong

The idea: the flat mask floor cuts deletions but fills holes with the raw
mixture, which carries both speakers, so leakage rises. Fill only where the
target is believed present -- and the model's own per-frame mean gain is 84 % of
its mask's information and is effectively its speech detector, so follow that.

**The implementation scaled the floor by `frame_gain / max_frame_gain`. The
maximum is an outlier**, so over 20 clips the median frame's multiplier is 0.151.
A nominal floor of 0.10 became about **0.015 in a typical frame** -- roughly a
seventh of nominal, and well under the **0.05** flat floor that produced the one
real deletion effect measured today.

Measured, and every delta is below the 1.57-point irrelevance floor:

| | baseline | flat 0.10 | gain-following 0.10 |
|---|---|---|---|
| LCF-WER | 59.52 | 61.87 | 59.58 |
| deletions | 11.79 | 9.98 | 11.20 |
| mean leaked % | 34.58 | 41.32 | 36.11 |

It avoided the flat floor's leakage penalty and lost the deletion benefit with
it, which is what applying almost no floor looks like.

**Recorded as UNTESTED.** Writing this up as "the targeted floor does not work"
would be false: it tested about a third of the dose that works flat. A fair test
needs a nominal value near 0.3-0.6, or a 90th-percentile normaliser rather than
the max (the 90th-percentile frame sits at 0.222 of the max, so the spread is
wide either way).

**The general lesson, worth more than the instance:** a variant that rescales an
intervention by a data-dependent quantity must have its EFFECTIVE strength
measured before its result is interpreted. The number that matters is what
reached the audio, not what was typed on the command line.

### The second setting made it worse, and suggests the variant is not just weak but wrong

| | baseline | gain-following 0.10 | gain-following 0.20 |
|---|---|---|---|
| LCF-WER | 59.52 | 59.58 | 61.93 |
| deletions | 11.79 | 11.20 | **12.02** |
| mean leaked % | 34.58 | 36.11 | 35.21 |

**Deletions are WORSE at the higher setting than at the lower one, and worse than
the baseline.** A floor cannot do that by filling holes, so the variant is doing
something else as well.

**The likely mechanism, and it should have been anticipated: a floor that varies
frame to frame adds a TIME-VARYING component to the audio.** A flat floor adds a
steady quiet copy of the mixture; this one adds a fluctuating one, and
fluctuation is the musical-noise mechanism. The variant may be manufacturing the
artefact the floor exists to avoid.

Not bootstrapped: the variant is both under-powered and confounded, so a
significance test on it would price a number that does not mean what it says.
**Redesign before retesting** -- a percentile normaliser AND a floor that is
smooth in time, or the two effects cannot be separated.

---

## 2026-09-12 — VOID: the internal-mask hysteresis runs measured a level explosion, not the idea

`2026-09-12-eval-hystint-sharp` returned LCF-WER **115.31**, deletions 80.21,
**65 % of clips with no transcript at all**, and leakage 2.53 %. The leakage
number looks like a triumph and means nothing: there was barely any usable audio
to leak into.

**Cause, and it is a bug in `apply_hysteresis`, not a property of the model.**
The function restored each frame's MEAN mask magnitude after sharpening.
Concentrating the same mean into fewer surviving bins multiplies the RMS, and the
output level follows the RMS. Measured output level of that run: **+7.38 dB above
the mixture**, against the baseline's -5.00 dB -- twelve decibels too loud. The
evaluation scored distortion.

**Why it did not show up in the output-domain runs.** Those sharpen the ratio
`|estimate| / |mixture|`, which is rough, so removing its small values barely
moves its mean and the restoration factor stays near 1. Our internal mask is
nearly flat (84 % of its variance is one number per frame), so removing bins
moves the mean a great deal. **The flatness that is this project's central
finding is exactly what detonated the experiment built to exploit it.**

**Fixed:** the invariant is now per-frame RMS, verified to hold level to x1.000
across flat, realistic and rough masks, plus a guard leaving a frame untouched
when nothing survives rather than dividing by ~0. Both settings are being re-run.

**Void, and to be deleted rather than reported:** `2026-09-12-est-hystint-*` and
`2026-09-12-eval-hystint-*`. The fixed runs are `*-hystfix-*`.

### The lesson, which is the reusable part

`tests/test_mask_postprocess.py` HAD a level-preservation test. It asserted the
MEAN, which is what the buggy code preserved, so it passed throughout. **A test
that asserts the wrong invariant is how a bug reaches a results table** -- it
supplies confidence without supplying a check. The test now asserts RMS and a
second test covers the empty-frame case.

The same question should be asked of any future post-processor here: *what
quantity does the audio's loudness actually follow, and is that the one being
held fixed?*

---

## 2026-09-13 — the expanded dev split `sir0_privval`, and eval_private released from holdout

**Decision (Grant): eval_private stops being a holdout. `eval_public` alone is the
final holdout.** Taken to buy evaluation resolution, with the cost accepted in
advance.

**Why it was needed.** 2026-09-12 measured `sir0_val`'s resolution at **+-8
LCF-WER points** over its 103 `both` trials. An arm moving the metric less than
~5 points is unreadable, and most realistic arms move it by 2. Resolving a
2-point effect needs ~15x the trials.

### What was NOT done, and why

**eval_private's 500 rendered trials were NOT concatenated onto `sir0_val`.**
Two blocking reasons, both measured rather than argued:

1. **Its renders predate `interferer.wav`.** Each eval_private trial directory
   holds only `enrollment/mixture/target/meta`. `evaluate.py:115` requires
   `interferer.wav` for EVERY leakage metric — ICR@2, `mean_leak`,
   `wrong_from_interferer`. Those 249 extra `both` trials could not have
   produced the project's primary diagnostic, which is 58.5 % of the error mass.
2. **It is a different distribution.** SIR spans [-5, +15] against `sir0_val`'s
   [-10, +10]; `regimes: null` (it predates the regime system by ten days);
   SNR mean 10.47 against 12.89; noise pool `tt` against `cv`; different
   `config_md5`. Merging would have silently made the benchmark ~5 dB easier and
   invalidated the 59.52 baseline, the 65.22 floor, the 5.85 ceiling and the
   1.57 irrelevance floor in one step.

Re-rendering was therefore mandatory, which made the distribution a free choice
rather than an inherited accident.

### What was done

**A new split, `sir0_privval`, from eval_private's 20 released speakers.**
`sir0_val` is untouched and byte-identical, so every prior result stays
reproducible.

| | sir0_val | sir0_privval |
|---|---|---|
| trials | 200 (103 `both`) | **2,800 (1,421 `both`)** |
| SIR | [-10, +10] | **[-10, +15]** |
| speakers | 40 (val) | 20 (eval_private) |
| noise | `cv` | `cv` |
| regimes | 0.6 base / 0.4 hard | identical |

**SIR IS WIDER, NOT EASIER.** [-10, +15] is a SUPERSET of [-10, +10]. Results on
this split are reported **per SIR band, never as one blended mean** — blending
re-hides the structure the widening was for.

**Noise is `cv`, not eval_private's `tt`.** `tt` is the pool `eval_public` draws
from; developing against those clips would leak noise into the final holdout.

**Verified, not assumed:** schema byte-compatible with `sir0_val`; 0 speaker
overlap with `sir0_train`; 0 speaker overlap with `eval_public`; 0 noise-clip
overlap with `eval_public`; all 2,800 rendered directories carry all six files.

### Why the easy end is worth having — measured, not assumed

Per-trial WER by SIR band on `sir0_val`, baseline `model_sir0_10000-e6.pt`
(n per band 23-33, so directional):

| SIR band | n | raw mixture | ours | WeSep | ceiling | our leaked fraction |
|---|---|---|---|---|---|---|
| < -5 | 33 | 89.5 | 88.2 | 49.4 | 3.3 | 0.651 |
| -5..0 | 23 | 95.6 | 87.5 | 38.9 | 7.5 | 0.351 |
| 0..+5 | 24 | 63.2 | 51.3 | 40.7 | 10.8 | 0.234 |
| >= +5 | 23 | 39.5 | 30.8 | 23.8 | 5.6 | **0.080** |

**The two ends are different problems.** At SIR < -5 we beat doing nothing by
1.3 WER (89.5 -> 88.2) while WeSep reaches 49.4 — we are effectively not working,
and leakage is 65 % of our wrong content words. At SIR >= +5 leakage is 8 % and
we still sit **25 points above the clean-target ceiling** (30.8 against 5.6), so
that error is our OWN damage — deletion and fabrication, not leakage.

**The easy band is therefore the only clean instrument for fabrication**, which
is 41.5 % of the error mass and has no other measurement in this project.

### Registered as `sir0ext` in `train.py`

`SPLIT_MANIFESTS["sir0ext"]` pairs `sir0_train` with `sir0_privval`, so a
checkpoint trained under `sir0` is scored on the wider set **without
retraining** and the two splits differ in exactly one axis.

### Carry this into every write-up

Anchors measured on `sir0_val` do NOT transfer to `sir0_privval`. The floor,
ceiling and irrelevance floor must be re-measured on the new split before any
number from it is compared to anything. Re-measurement was started 2026-09-13.

`>= +5` now spans +5..+15, wider than the other bands. Split it into +5..+10 and
+10..+15 when reporting, or the top band averages two different difficulties.

## 2026-09-13/14 — the anchors on `sir0_privval`, and they are NOT the sir0_val anchors

`experiments/results/2026-09-13-eval-privval-anchors`, `both` condition,
faster-whisper `small.en` STAND-IN, not a live-model result.

| | sir0_val (103 trials) | **sir0_privval** |
|---|---|---|
| floor, raw mixture | 65.22 | **53.49** |
| ceiling, clean target | 5.85 | **3.90** |
| floor ICR@2 | 66.99 | 60.49 |
| floor mean_leak | 51.30 | 43.92 |

**The new set's floor is 11.7 LCF-WER points easier, exactly as designed and
exactly as warned.** SIR runs to +15 dB here against +10 on `sir0_val`, so the
raw mixture is more often already intelligible. **Nothing measured on
`sir0_privval` may be compared to a `sir0_val` number**, and the per-SIR-band
reporting is what makes results from this split mean anything.

The usable range narrows too: floor-to-ceiling is 49.6 points here against 59.4
on `sir0_val`. More trials bought resolution; the wider SIR range spent some of
the dynamic range. Both are real and both must be stated.

---

## 2026-09-13 — w_struct DERIVED. The mask is 99.6 % of the way to a pure volume knob

`scripts/derive_w_struct.py`, `experiments/results/2026-09-13-wstruct-anchor-sir0`,
50 present crops, `model_sir0_10000-e6.pt`, 10 min CPU.

| anchor | mean L_struct |
|---|---|
| flat mask (pure volume knob) | 0.165366 |
| **our model's mask** | **0.164718** |
| oracle (ideal mask itself) | **0.000000** |

**Our mask sits 99.6 % of the way from the ideal mask to a flat one.**

The oracle reading exactly 0.000000 is the wiring check: the loss and its target
are on the same STFT grid. If that were nonzero the term would be supervising
against a misaligned object and nothing else would catch it.

**This is an INDEPENDENT confirmation of the 84.2 % finding**, measured a
different way and in the units the loss actually uses -- not "how much variance
is one number per frame" but "how far is this mask's frequency shape from flat".
Two methods, same conclusion.

**w_struct = 46.2981** at a 15 % gradient share on the PARAMETERS, re-measured at
0.1500. Large only because L_struct is numerically tiny beside the other terms.

### Two caveats that travel with the number

**Derived at chunk_s 1.0, training uses 4.008.** Memory forced it: one LSTM layer
in this stack allocates ~381 MB of activations at batch 1, twelve are retained
for backward, and the first version of the script took a 15 GB machine down. The
share is a ratio measured on the same crops so it should be stable, but **this
has not been checked**. Re-derive at 4.008 on a T4 before the number is quoted in
the write-up.

**Small-sample instability is real:** 17.4 at 2 crops, 124.6 at 4, 46.3 at 50.
Do not use a derivation under ~50 crops for anything.

---

## 2026-09-15 — Estimate rendering resumes; a resumed directory is provenance-locked

**Decision: `write_estimates` skips a trial whose `estimate.wav` is already on
disk and complete, and REFUSES to resume a directory whose recorded provenance
differs from the current run. `--force` re-renders and skips the guard.**

**Why resume.** A pass over `sir0_privval` is 3.4 h measured (`run_times.md`
2026-09-14). Before this, an interrupted pass restarted from zero, so the cost
of stopping a render was the whole render — which on 2026-09-15 threw away a
part-finished baseline render that was competing for CPU with a job needed
sooner. Inference runs under `no_grad` and is deterministic, so a kept file is
the file the pass would have written.

**Why refuse rather than warn on a provenance mismatch.** Pointing a second
checkpoint at an existing directory would blend two systems' audio into one set
of estimates labelled as one system. Nothing downstream can detect that:
`evaluate.py` reads wav files and believes `meta.yaml`. A warning in a terminal
scrollback is not a control, so the mismatch is fatal.

**Two files, two questions.** `meta.yaml` keeps its existing meaning untouched —
written last, so its presence still means the pass completed. The new
`run.provenance.yaml` is written first and deleted on success, so its presence
means a pass started here and did not finish. A resume checks the ledger if
present, else `meta.yaml`, so it is guarded against both an interrupted and a
completed prior run.

**A half-written wav is not reused.** Ctrl-C lands mid-write, and a truncated
wav opens without error while being silently short. A file is only kept if its
length matches its mixture within `LENGTH_WARN_S`. This is a completeness check,
not a checksum — it catches the interrupted write, and the provenance ledger,
not this, is what stops a different model's audio being trusted.

**`n_trials` is unchanged** — still trials in the directory — so every
`meta.yaml` written before this reads the same way. `n_written` and `n_reused`
are recorded alongside it, and the `run_times.md` row now names trials actually
rendered, so a resumed pass cannot log a per-trial rate for trials it skipped.

**Changes no number.** A full render with no prior output behaves exactly as
before, bit for bit.
