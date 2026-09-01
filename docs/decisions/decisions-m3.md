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
