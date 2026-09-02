# Where the project is — 2026-09-01

Plain-language status. Numbers in the milestone decision logs
(`decisions/decisions-m1.md` architecture, `-m2` training, `-m3` conventional
evaluation, `-m4` the metric and judge); dates and checklists in
`decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6 weeks.**

---

## In one paragraph

**Every measurement instrument the project needs is now built, and none of it has
seen a live model.** Six metrics exist and are tested — the three LCF scores, the
SIR/SAR decomposition, DNSMOS in both variants, and RTF/latency. Two divergence
results already exist without a judge: signal quality explains only ~4 % of the
variance in what the listener recovered, and perceptual quality moved *against*
content (DNSMOS OVRL fell 0.26 while LCF-WER improved 6.1 points). The model
keeps up in real time (RTF 0.53 on a laptop CPU). **The single gap is the judge:
until one is chosen, LCF-WER is arithmetically identical to offline ASR word error
rate, so the project's primary contribution does not yet exist as a distinct
measurement.** That is one ~1 hour experiment and about $25.

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

## Cannot

- **Generalise fully.** Improved but not solved: the train/held-out gap still
  reaches 4.16 dB by the last epoch, against 5.68 dB before.
- **Separate well.** Best held-out separation 2.58 dB against 1.59 dB for doing
  nothing. The margin is ~1 dB — better than it was, still thin.
- **Match level per utterance.** `L_gain` fell only 3 % over its whole run. It
  works as a *constraint on going silent*, not as a level regression target.
- **Be scored on a LIVE model.** The three metrics now produce real system
  numbers, but through an offline ASR standing in for the judge. No live-model
  measurement exists.
- **Adapt to how hard the trial is.** Measured 2026-09-01: SIR and SAR change are
  essentially FLAT across easy-to-hard trials (+3.80 to +4.32, −17.45 to −21.41)
  while the word-error outcome swings −4.2 to +23.1. It applies one transform to
  everything, so the same artefact is a bad trade on easy trials and a good one on
  hard ones. `decisions-pending.md` D11 carries the fixes.

## Not started

None of these exist, and none can be cut:

- No metric harness, no benchmark, no comparison table.
- No judge MODEL picked (J2a closed / J2b open-weight anchor). No longer an open
  argument — it is now a ~1-hour candidate gate.
- AMI untouched — the only real-audio check in the project.

They need *a* trained model, not a good one. **They are not blocked on
generalisation and should not wait for it.**

**Unblocked 2026-08-31 — J1 closed: the judge is audio-in / text-out.** LCF
measures the judge's audio encoder, not its turn-taking, so full duplex is not
required. This also deletes the response-transcription ASR from the measuring
instrument. Carries a ~50-trial full-duplex confirmation run so the deviation
from the stated objective is bought off rather than argued away.
`decisions-m4.md` 2026-08-31.

## Milestone scoreboard

| | status | open items |
|---|---|---|
| **M0** data | closed | — |
| **M1** architecture | closed | — |
| **M2** baseline trained | functionally complete | 1 (band-plan / `w_m` ablations) |
| **M3** conventional evaluation | **2 of 3 done** | 1 (listen to the outputs) |
| **M4** the metric | **6 of 13 done** | **7 — all of them behind the judge** |
| **M5** second model | designed, not built | 23 |
| **M6** the comparison | not started | 9 |

**M3 is effectively closed.** M4's seven remaining items are almost entirely
consequences of one decision: choose the judge.

## The results table

**Every metric value the project has, in one place.** Add columns as metrics are
built and rows as models are made. `sir0_val`, `condition=both`, n=103, scored
2026-09-01. **The listener is an offline ASR standing in for the judge — no
live-model number exists yet.**

**Content metrics** (lower is better):

| system | LCF-WER | ICR@2 | mean leak | NRR |
|---|---|---|---|---|
| **1. Floor** — do nothing | 65.2 % | 67.0 % | 51.3 % | 0.0 % |
| **2. Baseline** — `model_sir0_5000-e7.pt` | **59.1 %** | **54.4 %** | **39.1 %** | 1.0 % |
| **3. Extension** — per-band gate (D13) | — | — | — | — |
| **4. Ceiling** — clean target | 5.8 % | 0.0 % | 0.0 % | 0.0 % |

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

**Caveats that travel with this table.** Offline ASR, not a judge. `sir0_val` is
symmetric by construction, so it is harder than `eval_public`. NRR is structurally
near-zero until a real judge can decline. `decisions-m3.md` 2026-09-01.

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

1. **Choose the judge (J2).** It is the only thing blocking the project's actual
   result, and the choice is now a ~1-hour candidate gate rather than an open
   argument: score the ceiling, the floor and a few silent trials on each
   candidate and read whether it can report clean speech, whether it can fail,
   and whether it stays quiet when there is nothing to hear.
2. ~~**Score the 5,000-trial checkpoint on the metrics that exist.**~~ **DONE
   2026-09-01** — see the results table. Next in this line is the mix-back sweep
   (`decisions-pending.md` D11), which turns one checkpoint into a family of
   systems for the divergence curve and needs no retraining.
3. **A result already exists without a judge, and should be written up.** No
   signal-domain measure predicts what the listener recovered: the best, ΔSDR,
   explains ~4 % of the variance in word-error improvement (n=103, through an
   ASR). That is the divergence claim quantified.
4. **Do NOT render more data.** It would still help — the model is data-limited,
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
