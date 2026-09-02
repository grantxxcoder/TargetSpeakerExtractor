# Where the project is — 2026-09-02

Plain-language status. Numbers in the milestone decision logs
(`decisions/decisions-m1.md` architecture, `-m2` training, `-m3` conventional
evaluation, `-m4` the metric and judge); dates and checklists in
`decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6 weeks.**

---

## In one paragraph

**The judge exists and the metric has been measured through it. The primary
contribution is now a real measurement rather than a plan.** `gemini-3.7-flash`,
audio in / text out, scored floor, estimate and ceiling across all 103 `sir0_val`
`both` trials on 2026-09-02 for about 40 cents. **And the headline result is
negative: LCF-WER through the live judge captures 10.5 % of the available
headroom against the offline ASR's 10.4 %** — the two listeners agree, so on this
system the live model does not rank differently from conventional word error
rate. That was always a possible outcome and `milestones.md` M6 asks for it
explicitly ("or evidence that it doesn't"), and it cannot currently be otherwise:
**a rank inversion needs two systems to rank and there is one.** What the judge
*does* buy is a far better instrument — a **1.05 % ceiling against the ASR's
5.85 %** — plus two live-model behaviours an ASR cannot exhibit and which are
findings in their own right: it **fabricates 17–42 words when handed silence**,
and a **safety filter occasionally refuses the extractor's own output**. The two
earlier divergence results are untouched and remain the strongest evidence for
the metric: signal quality explains only ~4 % of the variance in what the
listener recovered, and perceptual quality moved *against* content.

## What changed on 2026-09-02 — the judge

1. **J2 answered: `gemini-3.7-flash`**, audio-in / text-out, AI Studio, prepay.
   Chosen over the `preview` Live models because it is *stable* and the metric
   only needs the audio encoder (J1). Prompt frozen at sha256[:12]
   **`d118b7d3bf30`**. `decisions-m4.md` 2026-09-02.
2. **The candidate gate passed on both readings that matter.** It reports clean
   speech (ceiling **1.05 %**, and byte-identical five times out of five) and it
   can fail (floor 63.27 %), so it has ample dynamic range to rank systems.
3. **Run-to-run spread measured — M4's gate criterion, and it passes by an order
   of magnitude.** Ceiling **0.0 points** across five identical calls; a mixture
   **2.9 points** on one clip and **16.0** on a more ambiguous one, against a
   62-point floor-to-ceiling range. Consequence to carry: the offline ASR is
   deterministic and the judge is not, so **per-trial** claims through the judge
   need k≥3 and averaging. Aggregates over 100+ trials are unaffected
   (SEM ≈ 0.5 points).
4. **The judge hallucinates on silence — 0 of 6 silent clips ever returned
   `no_speech`.** Survived three prompt variants including one that says "Do not
   hallucinate"; one answer came back in French on an English-only pipeline.
   Prompting does not fix it. A **speech gate** now decides speech/no-speech
   before any listener sees a clip. `decisions-m4.md` 2026-09-02.
5. **A safety filter refused an extractor output**, once in 309 calls, and
   **passed on retry** — so it is non-deterministic. Retried twice before being
   believed, and both counts (`filter_blocked`, `filter_transient`) are reported
   per system, because a filter that fires on one system's output and not
   another's is a bias in the benchmark.
6. **M4's prompt-sensitivity ablation is answered, by accident.** Across three
   prompt variants the cross-prompt range on one floor clip was 18.0 points and
   the *same-prompt* noise on that clip was 16.0. **The metric is not fragile to
   prompt wording** — the apparent prompt effect was sampling noise.
7. **§1's stated hypothesis is not supported, and its stated mechanism is
   inverted.** §1 predicts conventional TSE improves offline transcription while
   making audio *harder* for a live model, because an ASR "tolerates" artefacts.
   Measured: the ASR *deletes more* on the processed audio (9.28 → 12.60) while
   the judge deletes *less* (5.97 → 3.43). The two reach nearly the same total by
   opposite routes. Recorded as a prediction that failed.

## What changed since 2026-08-30

1. **The memorisation diagnosis was tested and confirmed.** `sir0_train` was
   re-rendered at **4,976 trials** (2.5x), `weight_decay` turned on at 1e-4 and the
   enrolment bank dropped to K=3. 18 epochs of a requested 25, early-stopped,
   6.2 h at 1,244 s/epoch. `decisions-m2.md` 2026-09-01.
2. **All three of the project's own metrics are built and tested.** LCF-WER, ICR
   and NRR, 51 tests, judge-agnostic with the transcriber swappable. Validated by
   reproducing the C2 floor and ceiling exactly. `decisions-m4.md` 2026-08-31.
3. **SIR/SAR added**, which splits residual interference from invented artefact.
   12 tests. `decisions-m3.md` 2026-09-01.
4. **The first system row exists.** The trained model has been scored end to end
   on its own metrics for the first time — see the results table below.
5. **J1 closed:** the judge is audio-in / text-out, not full-duplex.
6. **DNSMOS added, both variants**, validated against Microsoft's own
   implementation to 4e-16. Non-intrusive, so it is the only quality metric that
   can run on AMI. `decisions-m3.md` 2026-09-01.
7. **RTF and latency measured.** 80 ms chunks: RTF **0.528** mean / 0.706 p99,
   latency **162 ms** mean / 176 ms p99, no chunk misses the 80 ms deadline.
   Closes the M3 latency item. `decisions-m3.md` 2026-09-01.
8. **The mix-back sweep decided M5's ordering.** Globally α=1 already wins, but
   the per-difficulty optimum spans 0 to 1, so adaptation is motivated and a
   global gentleness shift is not. The per-band gate is now M5's primary
   experiment (D13); the artefact weight β is demoted to a secondary arm.
9. **Two spec gaps closed** — B4's absent-trial rule and B5's normaliser are now
   written into `metric-definitions.md` §3.1, not just the code.

## The 5,000-trial run

`experiments/results/2026-09-01-train-sir0-5000/`, checkpoint
`models/model_sir0_5000-e7.pt`. Best epoch **7**.

| | 1,989 trials (08-29) | **4,976 trials (09-01)** |
|---|---|---|
| best held-out separation | 2.14 dB | **2.58 dB** |
| **margin over doing nothing** (1.59 dB) | 0.55 dB | **0.99 dB** |
| gap at the best epoch | 1.24 dB | **1.08 dB** |
| gap at the last epoch | 5.68 dB (ep 24) | 4.16 dB (ep 18) |
| epochs to match the old best | 14 | **2** |

**The margin over doing nothing nearly doubled.** That is the honest framing:
2.58 sounds close to 2.14, but doing nothing already scores 1.59 dB, so what the
extractor *adds* is what matters. It also reaches in two epochs what previously
took fourteen.

**Still data-limited, not at capacity.** Train separation improved monotonically
all 18 epochs, 2.30 → 5.64 dB, and never plateaued, while held-out peaked at
epoch 7 and fell to 1.48 dB. That is the data-limited signature, not the
capacity-limited one — so more data would help again. **But 2.5x bought only
+0.44 dB, so a further doubling will buy less**, for ~5 h of preparation and a
33 GB upload. Decided against; the extractor is not the contribution.

**Conditioning did not improve.** Enrolment sensitivity −3.79 dB at the selected
epoch against −3.80 dB before. More data bought separation, not conditioning.
**Do not quote the −1.53 dB reached by epoch 17** — held-out separation was
collapsing over those same epochs, the same "headline moving for a bad reason"
pattern as 08-29.

## The finding that matters: it hurts trials that were already easy

Measured on the first system row, n=103. Grouped by how bad the unprocessed
mixture was:

| mixture difficulty | n | change in LCF-WER | ΔSIR | ΔSAR |
|---|---|---|---|---|
| easy, floor <25 % | 22 | **−4.2 pts WORSE** | +3.80 | −17.45 |
| medium, 25–60 % | 27 | −0.6 | +4.06 | −18.33 |
| hard, 60–100 % | 27 | +9.0 better | +5.06 | −21.05 |
| very hard, >100 % | 27 | **+23.1 better** | +4.32 | −21.41 |

**The two signal columns are flat while the outcome swings by 27 points.** The
model applies the same transform to every trial — it does not adapt. So the
easy-trial regression is *not* worse artefacts there: **the same artefact costs
nothing when there was a lot of interference to remove, and costs dearly when
there was not.** The trade lives in the input, not the output.

Overall: 43 trials improved (mean +29 pts), 31 worsened (mean −16), 29 unchanged.
The regressions cancel most of the gains, which is why the aggregate moves 6.1
points while single trials move by over 100.

**And no signal-domain measure predicts the outcome.** The best, ΔSDR, explains
about **4 %** of the variance in word-error improvement (n=103, through an ASR).
That is the divergence claim quantified — a more general statement than a rank
inversion. `decisions-m3.md` 2026-09-01.

**A prediction that failed, recorded as such.** ΔSAR correlates with word-error
improvement at **−0.05**, i.e. not at all. Trial-level artefact severity does not
predict trial-level word errors, so the artefact hypothesis of
`metric-definitions.md` §1 is not supported *in that form*.

## Superseded: the memorisation finding, 2026-08-29

Kept because it is what the 5,000-trial run was built to test. `1,989` trials,
best epoch 14, collapsed by epoch 24.

| | epoch 10 | epoch 14 (best) | epoch 24 |
|---|---|---|---|
| separation on **training** data (dB) | 2.97 | 3.38 | **5.51** |
| separation on **held-out** data (dB) | 1.52 | 2.14 | **−0.17** |
| gap | 1.45 | 1.24 | **5.68** |

By epoch 24 it was *worse than handing the mixture through untouched*. Diagnosis:
1,989 trials is not enough for a 7.2 M-parameter model. **Confirmed by the run
above.**

## Can

- Run causally, streaming-compatible. Measured, not assumed.
- **Identify the target from a 5 s sample.** 37.6 % enrolment sensitivity, on
  `sir0`, where "keep the louder voice" no longer works.
- **Output at roughly the right volume** — the mute is closed.
- Tell speech from silence: ~7 dB louder when the target is talking than when
  it is not, against 2.45 dB for the control.
- Train 10 epochs in 1.45 h, checkpoint, and resume without losing state.
- **Transcribe for evaluation, and say how hard the task is.** Offline ASR
  chosen (`faster-whisper small.en`) and C2 closed at n=230.
- **Score all three of its own metrics.** LCF-WER, ICR and NRR implemented and
  tested (51 tests), judge-agnostic, with the transcriber swappable. Validated
  by reproducing the C2 floor/ceiling exactly.
- **Beat doing nothing by ~1 dB** on held-out data, against 0.55 dB a week ago.
- **Run in real time, measured.** 80 ms chunks, RTF **0.528** mean / 0.706 p99 on
  a laptop CPU, end-to-end latency **162 ms** against a 200–300 ms budget, and no
  chunk misses the 80 ms deadline. An estimate rather than a true streaming
  measurement — there is no stateful path, so chunks are processed independently,
  10–20 % error. GPU figure outstanding.
- **Separate interference from artefact.** SIR/SAR decomposition implemented
  (`separation.py`, 12 tests). The model removes interferer well (**SIR +4.33 dB**)
  while inventing a lot (**absolute SAR +10.34 dB**, i.e. ~9 % of output energy is
  invented). Constructed data only — needs the clean sources, so never on AMI.
- **Be scored end to end on its own metrics.** First system row taken 2026-09-01:
  LCF-WER 65.2 → **59.1 %**, ICR@2 67.0 → **54.4 %** against a 5.8 % / 0.0 %
  ceiling. It captures ~10 % of the word-error headroom and ~19 % of the leakage
  headroom, and it **hurts trials that were already easy** — the
  artefact-versus-residue trade-off, measured. `decisions-m3.md` 2026-09-01.
- **BE SCORED ON A LIVE MODEL.** 2026-09-02, all 103 `sir0_val` `both` trials
  through `gemini-3.7-flash`: LCF-WER 63.27 → **56.72 %** against a **1.05 %**
  ceiling, ICR@2 75.73 → **62.14 %**. The primary contribution is a real
  measurement. About 40 cents. `decisions-m4.md` 2026-09-02.
- **Resume a judge run without ever paying twice.** target/mixture/interferer are
  judged once per instrument and never re-bought; estimates are keyed by audio
  content, so a retrained checkpoint is judged fresh but a byte-identical
  re-render is free. Every answer is written to disk before the call returns, so
  an interrupted run resumes exactly where it stopped.
- **Tell a speech-free clip from a speech-bearing one before spending anything.**
  Anchors by construction from the manifest, estimates by Silero VAD 6.2.1 (B2).
  Blocked clips are answered locally and logged with a reason.

## Cannot

- **Generalise fully.** Improved but not solved: the train/held-out gap still
  reaches 4.16 dB by the last epoch, against 5.68 dB before.
- **Separate well.** Best held-out separation 2.58 dB against 1.59 dB for doing
  nothing. The margin is ~1 dB — better than it was, still thin.
- **Match level per utterance.** `L_gain` fell only 3 % over its whole run. It
  works as a *constraint on going silent*, not as a level regression target.
- **Show a divergence between the live judge and offline WER — YET, and not for
  want of a judge.** Measured 2026-09-02: the two capture 10.5 % and 10.4 % of
  headroom respectively, i.e. they agree. **A rank inversion needs two systems to
  rank and there is one**, so the claim is untestable rather than refuted. M6's
  off-the-shelf TSE row is the cheapest way to make it testable.
- **Make per-trial claims through the judge.** The offline ASR is deterministic;
  the judge is not. Measured spread on one ambiguous mixture: **16.0 points**
  across five identical calls, and it sometimes transcribes one speaker and
  sometimes interleaves both. Aggregates over 100+ trials are safe
  (SEM ≈ 0.5 points); single-trial judge numbers need k≥3 and averaging.
- **Trust the judge to report absence.** It fabricates 17–42 words on digital
  silence, 0 of 6 clips ever returning `no_speech`, under three prompt variants.
  Mitigated by the speech gate, not fixed — and NRR is therefore blind to a
  *confabulating* judge, though the gate restores its ability to catch a
  *muting extractor*.
- **Adapt to how hard the trial is.** Measured 2026-09-01: SIR and SAR change are
  essentially FLAT across easy-to-hard trials (+3.80 to +4.32, −17.45 to −21.41)
  while the word-error outcome swings −4.2 to +23.1. It applies one transform to
  everything, so the same artefact is a bad trade on easy trials and a good one on
  hard ones. `decisions-pending.md` D11 carries the fixes.

## Not started

None of these exist, and none can be cut:

- **A SECOND SYSTEM. This is now the critical path.** With one system there is
  nothing to rank, so the divergence claim cannot be tested however good the
  judge is. M6 names the cheapest route: **≥1 off-the-shelf pretrained TSE
  system**, no training cost. That is worth more to the thesis right now than
  M5's per-band gate.
- **The absent rows.** B4's invented-speech row and the `noise_only` cases have
  not been scored through the judge. ~200 calls, and the only place the gate does
  real work. Also where the judge and the ASR are known to differ most.
- **J2b, the open-weight anchor.** Qwen3-Omni, chosen for encoder independence:
  its AuT encoder is trained from scratch, whereas Voxtral, Ultravox and
  Qwen2.5-Omni are all built on a Whisper encoder and would share lineage with
  our own reference ASR. Needed for reproducibility, not for cost.
- **`eval_public` / `eval_private` anchors** through the judge. ~$5 once, then
  reusable forever. `eval_private` is scored last and once.
- **AMI untouched** — the only real-audio check in the project.

**Judge-side work that is DONE and no longer blocks anything:** J2a, the prompt,
the harness, the candidate gate, the spread study, and the prompt-sensitivity
ablation.

**Closed 2026-08-31 — J1: the judge is audio-in / text-out.** LCF measures the
judge's audio encoder, not its turn-taking, so full duplex is not required. This
also deletes the response-transcription ASR from the measuring instrument.
Carries a ~50-trial full-duplex confirmation run so the deviation from the stated
objective is bought off rather than argued away. `decisions-m4.md` 2026-08-31.

## Milestone scoreboard

| | status | open items |
|---|---|---|
| **M0** data | closed | — |
| **M1** architecture | closed | — |
| **M2** baseline trained | functionally complete | 1 (band-plan / `w_m` ablations) |
| **M3** conventional evaluation | **2 of 3 done** | 1 (listen to the outputs) |
| **M4** the metric | **15 of 18 done** | 3 (J3 sign-off; J2b anchor; text reference condition) |
| **M5** second model | designed, not built | 23 |
| **M6** the comparison | **blocked on a second system, not on the metric** | 9 |

**M4 is nearly closed.** The judge, the prompt, the harness, the gate, the spread
study and the prompt-sensitivity ablation are all done; what remains is J3's ICR
threshold sign-off (a free re-score from stored text) and the text reference
condition.

**The bottleneck has moved.** It is no longer the metric — it is having a second
thing to measure. M6 cannot produce its comparison table from one system, and its
own checklist already names the fix: an off-the-shelf pretrained TSE system,
scored alongside, at no training cost.

## The results table

**Every metric value the project has, in one place.** Add columns as metrics are
built and rows as models are made. `sir0_val`, `condition=both`, **n=103**.

**Content metrics, THROUGH THE LIVE JUDGE** (lower is better). The project's
primary contribution, measured. `gemini-3.7-flash`, audio in / text out, via AI
Studio, prompt sha256[:12] `d118b7d3bf30`, speech gate on, **run 2026-09-02**.
Closed models change silently — a comparison against a different date is invalid
unless re-run.

| system | LCF-WER | ICR@2 | mean leak | NRR |
|---|---|---|---|---|
| **1. Floor** — do nothing | 63.27 % | 75.73 % | 63.33 % | 0.0 % |
| **2. Baseline** — `model_sir0_5000-e7.pt` | **56.72 %** | **62.14 %** | **50.15 %** | 0.0 % |
| **3. Extension** — per-band gate (D13) | — | — | — | — |
| **4. Ceiling** — clean target | **1.05 %** | 0.00 % | 0.00 % | 0.0 % |

**Content metrics, through the OFFLINE ASR** (`faster-whisper small.en`). Kept as
the conventional comparison, not as a stand-in any more. Same 103 trials.

| system | LCF-WER | ICR@2 | mean leak | NRR |
|---|---|---|---|---|
| **1. Floor** | 65.22 % | 66.99 % | 51.30 % | 0.0 % |
| **2. Baseline** | **59.05 %** | **54.37 %** | **39.13 %** | 1.0 % |
| **4. Ceiling** | 5.85 % | 0.00 % | 0.00 % | 0.0 % |

**THE HEADLINE COMPARISON, and it is a negative result.**

| listener | floor | baseline | ceiling | range | gain | **headroom captured** |
|---|---|---|---|---|---|---|
| live judge | 63.27 | 56.72 | 1.05 | 62.22 | 6.55 | **10.5 %** |
| offline ASR | 65.22 | 59.05 | 5.85 | 59.37 | 6.17 | **10.4 %** |

**The two listeners agree to within a tenth of a point.** On this system, LCF-WER
does not rank differently from offline word error rate. `milestones.md` M6 asks
for exactly this evidence — "or evidence that it doesn't" — and **it cannot
currently be otherwise: a rank inversion needs two systems to rank and there is
one.** Do not read this as the divergence claim failing; read it as the divergence
claim being untestable until a second system exists (M6's off-the-shelf TSE row
is the cheapest route).

**What the judge does buy, on the same data.**

1. **A far better instrument.** Ceiling **1.05 % against 5.85 %** — the judge
   hears clean audio 5.6x better, so the measurable range is 62.2 points instead
   of 59.4, and the ceiling is nearly perfect rather than imposing a 6-point cap
   on achievable performance.
2. **Leakage that is visible.** ICR@2 runs ~9 points higher throughout (floor
   75.73 vs 66.99), because the judge *reports* interferer content the ASR
   discards. ICR was near-structurally-blind through an ASR.
3. **The opposite failure mechanism** — see below.

**THE MECHANISM DIVERGES EVEN THOUGH THE TOTAL DOES NOT.** This is why the error
split is reported and not just the rate:

| | judge D | judge I | ASR D | ASR I |
|---|---|---|---|---|
| floor | 5.97 | 28.08 | 9.28 | 23.05 |
| baseline | **3.43** ↓ | 26.22 | **12.60** ↑ | 18.39 ↓ |

**On the processed audio the ASR deletes MORE and the judge deletes LESS.** The
two reach nearly the same total by opposite routes. **This inverts §1's stated
mechanism**, which has the ASR tolerating artefacts and the live model being
sensitive to them. Measured, it is the ASR that is brittle.

**NRR: judge 0.0 %, ASR 1.0 %.** The ASR produced one non-response; the judge
produced none, and never will — it invents rather than declining. See the
hallucination row below.

**Live-judge behaviours an offline ASR cannot exhibit**, both measured
2026-09-02 and both reportable findings:

| behaviour | measured | consequence |
|---|---|---|
| **invents speech on silence** | **0 of 6** silent clips returned `no_speech`; 17–42 words each, one in French | NRR could not catch a mute; a speech gate now answers speech-free clips locally |
| **safety filter refuses output** | 1 in 309 calls, on an *extractor output*; **passed on retry** | non-deterministic, so retried twice before being recorded; both counts reported per system as a bias check |

**Cost of the whole live-judge programme so far: about 40 cents**, ~309 clips.

**Signal metrics** (higher is better, dB, ceiling +30 by construction):

| system | SDR | SIR | SAR |
|---|---|---|---|
| **1. Floor** | −1.12 | −1.12 | +30.00 |
| **2. Baseline** | +0.86 | **+3.21** | **+10.34** |
| **3. Extension** | — | — | — |
| **4. Ceiling** | +30.00 | +30.00 | +30.00 |

**Perceptual metrics** (higher is better, 1–5, DNSMOS personalised):

| system | P808 | SIG | BAK | OVRL |
|---|---|---|---|---|
| **1. Floor** | 2.913 | **4.090** | 2.031 | **2.497** |
| **2. Baseline** | 2.937 | **3.366** | **2.266** | **2.237** |
| **3. Extension** | — | — | — | — |
| **4. Ceiling** | 3.550 | 4.175 | 3.592 | 3.429 |

**Latency and throughput** (a property of the model, so the floor and ceiling
rows do not apply — they are not models). 80 ms chunks, i5-1135G7, 4 threads:

| system | RTF mean | RTF p99 | latency mean | latency p99 | keeps up |
|---|---|---|---|---|---|
| **2. Baseline** | **0.528** | 0.706 | **162.2 ms** | 176.5 ms | **yes** |
| **3. Extension** | — | — | — | — | — |

Requirements: **RTF < 1** (else the input backlog grows without bound) and
**latency < 200–300 ms**. Latency = 80 ms chunk + 40 ms model lookahead + 42.2 ms
compute. Per-chunk max was 58.2 ms against the 80 ms deadline, so no chunk
overran. **The RTF deadline is the tighter constraint** — meeting it satisfies the
latency budget automatically.

Two caveats. **This is an estimate, not a streaming measurement**: the model has
no stateful path, so chunks are processed independently and the output discarded,
giving 10–20 % error. And the **GPU figure is outstanding** — the spec assumes
server-class compute, so the GPU number is the one that supports the claim and
CPU is the pessimistic case.

**Headroom captured by the baseline:** LCF-WER **10.4 %**, ICR@2 **18.8 %**, mean
leakage **23.8 %**.

**Ceilings are not perfect, in any of the three families.** LCF-WER's ceiling is
5.8 % not 0 %, DNSMOS `OVRL`'s is 3.43 not 5, because the reference is the
*reverberant* target (A1) and neither the ASR nor DNSMOS was built for
reverberation. **Never quote a score without its ceiling.**

**How to read the three signal columns.** +30 dB is the ceiling in all of them,
set by `TAU = 1e-3`. The floor's **SAR is +30 by construction** — the mixture is
exactly the sum of the sources, so it invents nothing — which means **ΔSAR always
looks catastrophic and must never be quoted.** Use the absolute value: the
baseline's +10.34 dB means about **9 % of its output energy is invented**.

**What the row 2 numbers say in one line.** The model removes the interferer
reasonably well (SIR −1.12 → +3.21) at the cost of inventing a lot (SAR +30 →
+10.34), and the net signal gain (+1.98 dB) buys only 6.1 points of word error.

**THE DIVERGENCE, measured 2026-09-01.** Content improved by **6.1 points** of
LCF-WER while perceptual quality got **worse** — DNSMOS `OVRL` 2.497 → 2.237.
**A human listener would say the model damaged the audio; the listener recovered
more of the words.** The conventional perceptual metric would have rejected a
system that measurably helps the downstream task, which is precisely the failure
mode this project exists to demonstrate.

**Two instruments agree on the mechanism.** `SIG` fell 0.72 while `BAK` rose only
0.24 — perceptual confirmation of the signal-domain finding that the artefact
introduced outweighs the interference removed. `SIG` behaves as the perceptual
analogue of SAR, `BAK` of SIR.

**And `P808` is flat (+0.02)** — the metric the REAL-TSE organisers switched *to*
is nearly blind to what this model does, while `OVRL`, the one that was gamed,
moves. Worth reporting about both.

**Caveats that travel with this table.**

- The **signal and perceptual rows are ASR-independent**; only the content rows
  have a listener, and those now exist in both versions above.
- `sir0_val` is symmetric by construction, so it is **harder than
  `eval_public`** — the two differ by 7.8 points on the floor, and which set
  defines the benchmark is still undecided (C2's open consequence).
- **NRR is near-zero for a reason that is now understood, not an artefact.** It
  was expected to lift once a real judge could decline. It cannot: this judge
  **never declines, it invents** (0 of 6 silent clips returned `no_speech`). NRR
  detects a declining judge and is structurally blind to a confabulating one.
  Its mute-detection works only because the speech gate manufactures the empty
  response NRR looks for. `decisions-m4.md` 2026-09-02.
- **Judge numbers are dated, not permanent.** `gemini-3.7-flash` is a closed,
  silently-updated model. Every judge figure above is valid for prompt
  `d118b7d3bf30` on 2026-09-02 and must be re-run to be compared against any
  other date.
- **Aggregates only.** Per-trial judge numbers carry up to 16 points of
  run-to-run noise; these are means over 103 trials (SEM ≈ 0.5 points).

`decisions-m3.md` 2026-09-01, `decisions-m4.md` 2026-09-02.

---

## Offline ASR — chosen 2026-08-28

`small.en`, int8 on CPU, greedy, Whisper `EnglishTextNormalizer`. `tiny.en`'s
floor exceeds 100 % (it invents words); `medium.en` is better but 2.7x the cost
across every pass.

**C2 is closed, 2026-08-30.** Scored at n=230 on `both` trials — the condition
that has an interferer to remove, and the only row that should ever be quoted:

| set | ceiling (clean) | floor (raw mixture) |
|---|---|---|
| `eval_public` (n=230) | 6.1 % | **57.4 %** |
| `sir0_val` (n=103) | 5.8 % | **65.2 %** |

**Plain reading:** of every 100 words the target says, ~57 come out wrong if you
do nothing, against ~6 wrong on clean audio. **That 51-point gap is the room the
extractor has to work in.** And the errors are not mush: inspected on one trial,
the ASR transcribes the target perfectly for 17 words and then switches to the
*other speaker's* sentence — which is exactly the failure the model exists to fix.

Accepted at this range. **This replaces the 76.4 % that was quoted from a
12-trial pilot; it was wrong by 19 points.**

**The open consequence is bigger than the number.** Training is on `sir0`, which
is symmetric by construction, while `eval_public` keeps the original
distribution where the target is the louder voice 74 % of the time. Which set
defines the benchmark is undecided, and that is the supervisor conversation.

Known artefact: `small.en` emits the word "you" on digital silence, 8 of 8 absent
trials. Filter it before counting invented words.

## Compute

**Measured, 2026-09-01: 1,244 s/epoch at 4,976 trials** on a Kaggle T4, batch 3.
18 epochs took 6.2 h, so ~25 epochs fits the ~12 h cap. The earlier projection of
~1,315 s/epoch was 6 % high. 19,938 trials would still not fit.

Local preparation costs, all measured on 2026-08-31 rather than projected:
render 4,976 trials **1.2 h**, enrolment bank at K=3 **1.5 h**, bundle and zip
**~47 min**, and the split occupies **15 GB**. Note `run_times.md` projected
1.26 MB/trial and the real figure is **3.25 MB/trial**, because sir0 renders the
interferer stem and two enrolment banks.

**A render is cheaper than it looks.** Raising `n_trials` *appends* trials rather
than resampling — the first 1,989 came back byte-identical — so `render_trials.py`
skips what exists and only the new trials cost anything. Its `config_md5` refusal
is a conservative guard, not evidence the audio changed.

Batch size is still 3 where the config's comment says 12 — untested, and it must
stay pinned at 3 for any resume, because `train.py` refuses a resume whose config
differs from the checkpoint's.

## Next

1. ~~**Choose the judge (J2).**~~ **DONE 2026-09-02.** `gemini-3.7-flash`, and
   the first live-judge row exists. See the results table.
2. **GET A SECOND SYSTEM. This is the critical path now.** The judge is built and
   the metric works, and neither can produce a divergence result from one system.
   The cheapest route is M6's own: **an off-the-shelf pretrained TSE system**, no
   training cost, scored through the same harness. It is worth more right now
   than M5's per-band gate, because it converts an untestable claim into a
   testable one for a few hours of work and ~16 cents of judge calls.
   *Second cheapest:* the mix-back sweep (`decisions-pending.md` D11) turns one
   checkpoint into a family of systems with no retraining at all.
3. **Score the absent rows through the judge.** ~200 calls, `--condition ""`.
   B4's invented-speech row plus the `noise_only` cases — the only place the
   speech gate does real work, and where the judge and the ASR differ most.
4. **Two results already exist and should be written up.** (a) No signal-domain
   measure predicts what the listener recovered: the best, ΔSDR, explains ~4 % of
   the variance in word-error improvement. (b) The live judge and the offline ASR
   **agree** on headroom captured (10.5 % vs 10.4 %) while failing by **opposite
   mechanisms** — the ASR deletes more on processed audio, the judge deletes
   less. The second is a negative result on §1's headline and an inversion of its
   stated mechanism, and both belong in the write-up as such.
5. **Do NOT render more data.** It would still help — the model is data-limited,
   not at capacity — but 2.5x bought +0.44 dB and a further 2x will buy less,
   for ~5 h of rendering and a 33 GB upload. `decisions-m2.md` 2026-09-01.

**Watch the train/held-out gap, not the total.** Both totals fell the whole way
through the run that overfitted.

## The open architecture question

**Unchanged on size, but there is now a live question about the OUTPUT.**

On size the answer still points *against* changing anything: the model has enough
capacity to memorise its training set, so a bigger or richer one overfits sooner.
Conditioning changes (D1/D4a in `decisions-pending.md`) stay cheap and additive.
Replacing the backbone means retraining from zero. That distinction holds — it is
not "change the architecture", it is which half.

**What is new is the masking question.** The model invents ~9 % of its output
energy (absolute SAR +10.34 dB), and **a mask can only attenuate bins that
already exist; it cannot synthesise.** The artefacts are therefore imperfect
attenuation — spectral holes, musical noise, phase damage — which is a property of
the parameterisation, not of the weights.

M5's artefact-penalty retrain is the experiment that tests it, and **both outcomes
are results.** If artefact falls without suppression falling, masking was merely
being applied too aggressively. **If SAR cannot improve without giving up SIR,
that is evidence masking is the wrong output parameterisation for this task** —
the argument for a mapping or generative output, reportable even though building
the replacement is out of scope before the freeze. `milestones.md` M5.
