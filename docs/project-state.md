# Where the project is — 2026-09-03

Status only. Numbers and rationale live in the milestone logs
(`decisions/decisions-m1.md` architecture, `-m2` training, `-m3` conventional
evaluation, `-m4` metric and judge); dates in `decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6 weeks.**

## Status

A borrowed WeSep checkpoint now sits alongside our baseline as system 5. It beats
us on every metric, including the live judge. There is **no rank inversion**: all
five instruments agree, so M6's divergence question is answered negatively on
n=2. The pair is 30 LCF-WER points apart, which cannot test a claim about close
systems — the per-band gate (system 3) is now the experiment that matters.

Our baseline is not competitive with an off-the-shelf checkpoint on our own data.
The thesis has to say so. This is the data-limited diagnosis arriving as a number.

## Results

All rows: `sir0_val`, `condition=both`, n=103. Lower is better except where said.

**Content through the live judge** — `gemini-3.7-flash`, audio in / text out,
prompt sha256[:12] `d118b7d3bf30`, run 2026-09-02 (baseline) and 2026-09-03
(WeSep). Closed model: comparisons across dates are invalid unless re-run.

| system | LCF-WER | ICR@2 | mean leak | invented/trial | FR@2 |
|---|---|---|---|---|---|
| 1. Floor — do nothing | 63.27 % | 75.73 % | 63.33 % | 1.24 | 32.0 % |
| 2. Baseline — `model_sir0_5000-e7.pt` | 56.72 % | 62.14 % | 50.15 % | **1.83** | 41.7 % |
| 3. Extension — per-band gate (D13) | — | — | — | — | — |
| 5. WeSep — `tfmap_context_causal_100`, borrowed | **26.40 %** | **25.24 %** | **12.92 %** | **1.80** | 38.8 % |
| 4. Ceiling — clean target | 1.05 % | 0.00 % | 0.00 % | 0.20 | 1.0 % |

**FR (fabrication) replaced NRR on 2026-09-03.** NRR read 0.0 % on every judge row
— its detector is an empty response and this judge invents instead of declining.
FR counts response words in **neither** speaker's script.
`experiments/results/2026-09-03-fr-sweep-sir0_val/`, decisions-m4.md 2026-09-03.

**Read fabrication on `invented/trial`, never on a percentage of the response.**
The percentage divides by the response's own length, so a terser listener scores
worse for saying less: WeSep reads 13.8 % against the baseline's 10.4 % while
inventing *fewer* words, because its responses average 15.1 content words to 17.9.

**Two things this column shows that nothing else does.** Both extractors raise
fabrication **~48 % above doing nothing** (1.24 → ~1.8). And **the baseline and
WeSep are indistinguishable on it** (1.83 vs 1.80) despite a 25-point LCF-WER gap
and WeSep leaking a third as much — **fabrication is its own axis, not a
by-product of extraction quality.** The ceiling is 0.20, not 0, because a word
absent from both scripts may be misheard rather than invented; only the excess
over the ceiling belongs to a system.

**Content through the offline ASR** (`faster-whisper small.en`).

| system | LCF-WER | ICR@2 | mean leak | invented/trial |
|---|---|---|---|---|
| 1. Floor | 65.22 % | 66.99 % | 51.30 % | not yet scored |
| 2. Baseline | 59.05 % | 54.37 % | 39.13 % | not yet scored |
| 5. WeSep | **34.60 %** | **15.53 %** | **9.02 %** | not yet scored |
| 4. Ceiling | 5.85 % | 0.00 % | 0.00 % | not yet scored |

FR exists for both listeners but has only been computed on the judge so far; the
ASR column fills on the next `evaluate.py --metrics content` run. The `NRR`
column that stood here is removed — it read 0.0 % everywhere except the baseline's
single 1.0 %.

> **The WeSep offline-ASR row has no results file.** `run_times.md` records that
> `evaluate.py` pass as failed; there is no `2026-09-03-evaluate-wesep-asr/`.
> Re-run before quoting those four numbers:
> `python scripts/evaluate.py --split sir0_val --condition both --est experiments/results/2026-09-03-est-wesep-tfmap-causal --metrics content --out experiments/results/2026-09-03-evaluate-wesep-asr`

**Headroom captured** — fraction of the floor-to-ceiling gap each system closes.
Higher is better.

| system | listener | LCF-WER | ICR@2 | mean leak |
|---|---|---|---|---|
| 2. Baseline | judge | 10.5 % | 17.9 % | 20.8 % |
| 5. WeSep | judge | **59.3 %** | **66.7 %** | **79.6 %** |
| 2. Baseline | ASR | 10.4 % | 18.8 % | 23.7 % |
| 5. WeSep | ASR | **51.6 %** | **76.8 %** | **82.4 %** |

WeSep captures about six times the content headroom our baseline does, on both
listeners.

**Error split** — S/D/I rates.

| system | judge S | judge D | judge I | ASR S | ASR D | ASR I |
|---|---|---|---|---|---|---|
| floor | 29.22 | 5.97 | 28.08 | — | 9.28 | 23.05 |
| 2. baseline | 27.07 | 3.43 | 26.22 | 28.13 | 12.60 | 18.39 |
| 5. WeSep | 10.62 | 6.78 | 8.99 | 18.51 | 8.91 | 7.19 |
| ceiling | 0.87 | 0.15 | 0.03 | — | — | — |

Two mechanism findings:

- **The ASR deletes more on the baseline's audio while the judge deletes less**
  (9.28 → 12.60 vs 5.97 → 3.43). Same total by opposite routes. This inverts
  `metric-definitions.md` §1, which has the ASR tolerating artefacts. Measured,
  the ASR is the brittle one.
- **WeSep wins by silencing the interferer, not by preserving the target.**
  Judge insertions fall 26.22 → 8.99 (those words were the other speaker);
  deletions *rise* to 6.78, worse than the baseline's 3.43 and worse than doing
  nothing. So LCF-WER on two-speaker mixtures is dominated by interferer
  suppression, and a system can lose target words while halving the score. State
  that wherever LCF-WER is quoted.

**Signal** (higher better, dB, ceiling +30 by construction, `TAU = 1e-3`).

| system | SDR | SIR | SAR |
|---|---|---|---|
| 1. Floor | −1.12 | −1.12 | +30.00 |
| 2. Baseline | +0.86 | +3.21 | +10.34 |
| 3. Extension | — | — | — |
| 5. WeSep | **+4.68** | **+10.34** | 8.26 |
| 4. Ceiling | +30.00 | +30.00 | +30.00 |

- **SAR is the only column our baseline wins** (+10.34 vs 8.26): WeSep removes
  7.1 dB more interferer and pays in artefact.
- **Never quote ΔSAR.** The floor's +30 is by construction, so ΔSAR always looks
  catastrophic. Use the absolute: +10.34 dB means ~9 % of output energy invented.
- WeSep beats the baseline on 79 % of trials by SI-SDR and beats doing nothing on
  83 % (baseline 78 %). Quote **+4.68**, not the pre-flight's +4.71 (floored vs
  unfloored SI-SDR).

**Perceptual** (higher better, 1–5, DNSMOS personalised).

| system | P808 | SIG | BAK | OVRL |
|---|---|---|---|---|
| 1. Floor | 2.913 | 4.090 | 2.031 | 2.497 |
| 2. Baseline | 2.937 | 3.366 | 2.266 | 2.237 |
| 3. Extension | — | — | — | — |    
| 5. WeSep | **3.010** | 3.450 | **2.640** | **2.510** |
| 4. Ceiling | 3.550 | 4.175 | 3.592 | 3.429 |

- **WeSep is the only system that does not make quality worse than doing
  nothing** (2.510 vs 2.497). Our baseline drops to 2.237.
- **The divergence, 2026-09-01:** the baseline improved content by 6.1 points
  while `OVRL` fell 2.497 → 2.237. A human would say it damaged the audio; the
  listener recovered more words. A perceptual metric would have rejected a system
  that helps the task.
- `SIG` fell 0.72 while `BAK` rose 0.24 — perceptual confirmation that artefact
  added outweighs interference removed. `P808` is flat (+0.02): the metric the
  REAL-TSE organisers switched *to* is nearly blind here, while `OVRL`, the one
  that was gamed, moves.

**Latency and throughput** (a model property, so floor/ceiling do not apply).
80 ms chunks, i5-1135G7, 4 threads.

| system | RTF mean | RTF p99 | latency mean | latency p99 | meets budget |
|---|---|---|---|---|---|
| 2. Baseline | 0.528 | 0.706 | 162.2 ms | 176.5 ms | yes |
| 3. Extension | — | — | — | — | — |
| 5. WeSep | **2.854** | 5.300 | **348.3 ms** | 544.0 ms | **no — and cannot stream at all** |

Measured 2026-09-03, 2250 chunks, 23 min wall, same i5-1135G7 / 4 threads as the
baseline. `experiments/results/2026-09-03-rtf-wesep-cpu/rtf.json`. Per chunk:
mean 228.3 ms, p50 207.9, p95 338.6, p99 424.0, max 552.9 — **no chunk finishes
inside its own 80 ms**, so the backlog grows without bound.

**It misses the RTF deadline by 2.9x and the latency budget by ~1.2-1.8x**, where
our baseline sits at 0.528 and 162.2 ms. Two reasons: 27.2 M parameters in the
timed forward against our 7.19 M, and WeSep re-embeds the 5 s enrolment through
its speaker branch on **every 80 ms chunk**. Our model re-embeds per chunk too, so
the protocol is matched, but the *cost* is wildly asymmetric — a full speaker
encoder against our cheap TF-Map cue. `--cache-fbank` gives a lower bound; it has
not been run. Either way the causality result above makes the RTF academic: a
model that needs the whole clip cannot stream at any speed.

Requirements: RTF < 1 and latency < 200–300 ms. Latency = 80 ms chunk + 40 ms
lookahead + 42.2 ms compute; per-chunk max 58.2 ms against an 80 ms deadline.
RTF is the tighter constraint. Two caveats: no stateful path exists, so chunks
are processed independently and this is an estimate with 10–20 % error; and the
GPU figure is outstanding, which is the one that supports the server-class claim.

**Ceilings are not perfect in any family** — LCF-WER 5.85 % not 0, `OVRL` 3.43
not 5 — because the reference is the *reverberant* target (A1). Never quote a
score without its ceiling.

**Caveats that travel with these tables.** Signal and perceptual rows are
ASR-independent; only content has a listener. `sir0_val` is symmetric by
construction so it is harder than `eval_public` (7.8 points apart on the floor);
which set defines the benchmark is undecided. All judge figures are aggregates —
per-trial judge numbers carry up to 16 points of noise (SEM ≈ 0.5 over 103).

## System 5 — the borrowed WeSep baseline

The REAL-TSE Challenge baseline toolkit's published checkpoint (Wang et al.,
Interspeech 2024, "WeSep"; fork `REAL-TSE/wesep-real-tse`), run on our data with
our metric. **No number from it is comparable to any published REAL-TSE result** —
different data, metric and protocol.

| | |
|---|---|
| checkpoint | `tfmap_context_causal_100`, sha256 `97c01b79b0cf1a5b…` |
| parameters | 33.46 M (ours: 7.19 M) |
| trained on | Libri2Mix `train-100` **clean**, anechoic |
| training loss | SI-SDR against the **dry** source |
| augmentation | `noise_prob 0`, `reverb_enroll_prob 0`, `noise_enroll_prob 0` |
| declared causal | `true`, win 512 / stride 128 @ 16 kHz |
| rendered by | `scripts/make_estimates_wesep.py`, 103 trials, 34 min CPU |
| commit | `020c9698` |

**Out of domain in every direction.** Our mixtures are reverberant (T60
0.25–0.6 s) with real recorded noise; its training data had neither, and it was
optimised for the dry source while we score against the reverberant target (A1).

**Both systems render through the same `src/estimates/runner.py`** — whole clip,
one forward pass, float32 unnormalised — so the rows differ only by model.
WeSep's own Silero 5.1.2 gate and output normalisation are off: our gate is
Silero 6.2.1 applied identically to every system by `speech_gate.py`, and a
second gate inside a system under test would mute clips before ours saw them.

Render health, all passed: no non-finite samples; −4.9 dB re the mixture so not
muted; SI-SDR vs mixture only +1.80 dB so not a pass-through; worst length delta
−126 samples (7.9 ms).

**Streaming status: RESOLVED 2026-09-03 — it CANNOT stream as called.**
Scale-matched probe (`scripts/probe_wesep_causality.py`): replacing the future at
matched loudness still moves the output *before* the cut by **1.12e-02**, about
3.5 % of the signal, against our own model's 1.68e-08. Every output frame depends
on the whole clip.

**Mechanism found, and `causal: true` is not a false claim — it is narrower than
it reads.** That flag sits under `separator:` only, and the separator's RNNs are
causal. But `SubbandNorm` (`wesep/modules/separator/bsrnn.py:44`) applies
`nn.GroupNorm(group=1, …)` over a `(B, C, T)` tensor, which normalises over all
channels **and all time frames** — a whole-utterance statistic computed *before*
the causal separator sees anything. This is global layer norm in the `gLN` sense.

Two things corroborate global normalisation rather than architectural lookahead:
the leak *shrinks* as the cut moves later (3.49 % → 2.66 % → 1.40 % for cuts at
2/4/6 s of an 8 s clip), which is the signature of a statistic diluted by a
smaller perturbed fraction, not of a fixed lookahead window; and the 5x-louder
probe leaks only ~2x more (2.89e-02), not catastrophically more.

**Fixable in principle, not by us.** The standard causal replacement is
cumulative layer norm, but swapping it requires retraining their checkpoint. For
our purposes WeSep is an offline system, and its scores stand as offline scores.

**Caveat to carry:** the determinism floor was **1.21e-05**, not zero, so the
model is not bit-reproducible on repeated identical input (most likely
multi-threaded reduction order). The effect above is ~900x that floor, so the
conclusion holds, but quote the floor alongside it. And the probe's matched mode
equalises *broadband* RMS, not per-subband RMS, so it does not fully isolate
lookahead from global normalisation — the mechanism above is what separates them.

Whole-clip throughput RTF **1.128** against our 0.398 by the same route; that is
batch, not streaming, and must not go in the RTF column.

**Judge run health, 2026-09-03:** 103 calls, 0 failed, 0 gate blocks, 1 transient
safety-filter hit that passed on retry. Floor and ceiling served from cache and
identical to 2026-09-02, so both systems are scored against the same anchors.
`experiments/results/2026-09-03-evaluate-wesep-judge/`.

## The judge

`gemini-3.7-flash`, audio in / text out, AI Studio prepay. Chosen over the
`preview` Live models because it is stable and the metric only needs the audio
encoder (J1 closed 2026-08-31: LCF measures the encoder, not turn-taking, which
also deletes the response-transcription ASR from the instrument). Prompt frozen
at `d118b7d3bf30`. Cost to date ~53 cents over ~412 clips. J1 carries a
~50-trial full-duplex confirmation run, so the deviation from the stated
objective is bought off rather than argued away.

What it buys over the offline ASR:

1. **A better instrument** — ceiling 1.05 % vs 5.85 %, so 62.2 points of range
   instead of 59.4, and no 6-point cap on achievable performance.
2. **Visible leakage** — ICR@2 runs ~9 points higher throughout because the judge
   reports interferer content the ASR discards.
3. **Run-to-run spread passes M4's gate by an order of magnitude** — ceiling 0.0
   points across five identical calls, a mixture 2.9 and an ambiguous one 16.0,
   against a 62-point range.

Three live-model behaviours an ASR cannot exhibit, all reportable findings:

- **It fabricates on silence.** 0 of 6 silent clips returned `no_speech`; 17–42
  words each, one in French, under three prompt variants including one saying
  "do not hallucinate". Prompting does not fix it; a speech gate now answers
  speech-free clips locally.
- **A safety filter refused an extractor output**, 1 in 309 calls, and passed on
  retry. Non-deterministic, so `filter_blocked` and `filter_transient` are
  reported per system — a filter that fires on one system and not another is a
  benchmark bias.
- **It never declines, so NRR was structurally blind to it** — which is why NRR
  was removed on 2026-09-03 and replaced by FR. NRR detects a
  declining judge, not a confabulating one. Its mute-detection works only because
  the gate manufactures the empty response NRR looks for.

**The prompt-sensitivity ablation is answered, by accident.** Cross-prompt range
on one floor clip was 18.0 points; same-prompt noise on that clip was 16.0. The
metric is not fragile to wording — the apparent effect was sampling noise.

## Our baseline — the 5,000-trial run

`experiments/results/2026-09-01-train-sir0-5000/`, checkpoint
`models/model_sir0_5000-e7.pt`, best epoch 7.

| | 1,989 trials (08-29) | 4,976 trials (09-01) |
|---|---|---|
| best held-out separation | 2.14 dB | **2.58 dB** |
| margin over doing nothing (1.59 dB) | 0.55 dB | **0.99 dB** |
| gap at the best epoch | 1.24 dB | **1.08 dB** |
| gap at the last epoch | 5.68 dB (ep 24) | 4.16 dB (ep 18) |
| epochs to match the old best | 14 | **2** |

- **The margin over doing nothing nearly doubled**, which is the honest framing:
  2.58 looks close to 2.14, but doing nothing already scores 1.59.
- **Still data-limited, not at capacity.** Train separation rose monotonically
  all 18 epochs (2.30 → 5.64 dB) and never plateaued while held-out peaked at
  epoch 7 and fell to 1.48. More data would help, but 2.5x bought only +0.44 dB
  so a further doubling buys less, for ~5 h and a 33 GB upload. Decided against.
- **Conditioning did not improve** — enrolment sensitivity −3.79 dB vs −3.80 dB
  before. Do not quote the −1.53 dB at epoch 17: held-out separation was
  collapsing over those epochs, the same headline-moving-for-a-bad-reason pattern
  as 08-29.

**It hurts trials that were already easy.** n=103, grouped by how bad the
unprocessed mixture was:

| mixture difficulty | n | Δ LCF-WER | ΔSIR | ΔSAR |
|---|---|---|---|---|
| easy, floor <25 % | 22 | **−4.2 pts worse** | +3.80 | −17.45 |
| medium, 25–60 % | 27 | −0.6 | +4.06 | −18.33 |
| hard, 60–100 % | 27 | +9.0 better | +5.06 | −21.05 |
| very hard, >100 % | 27 | **+23.1 better** | +4.32 | −21.41 |

The signal columns are flat while the outcome swings 27 points: the model applies
one transform to everything. So the easy-trial regression is not worse artefacts
there — **the same artefact costs nothing when there was a lot to remove and
costs dearly when there was not.** The trade lives in the input.

43 trials improved (mean +29 pts), 31 worsened (−16), 29 unchanged. The
regressions cancel most of the gains, which is why the aggregate moves 6.1 points
while single trials move over 100.

**No signal-domain measure predicts the outcome.** The best, ΔSDR, explains ~4 %
of the variance in word-error improvement (n=103, through the ASR) — the
divergence claim quantified, and more general than a rank inversion. ΔSAR
correlates at **−0.05**, so trial-level artefact severity does not predict
trial-level word errors and §1's artefact hypothesis is unsupported in that form.
`decisions-m3.md` 2026-09-01.

**Superseded: the memorisation finding, 2026-08-29.** Kept because the run above
was built to test it. 1,989 trials, best epoch 14, collapsed by 24.

| | epoch 10 | epoch 14 (best) | epoch 24 |
|---|---|---|---|
| separation on training data (dB) | 2.97 | 3.38 | 5.51 |
| separation on held-out data (dB) | 1.52 | 2.14 | **−0.17** |
| gap | 1.45 | 1.24 | **5.68** |

By epoch 24 it was worse than passing the mixture through untouched. 1,989 trials
is not enough for a 7.2 M-parameter model. Confirmed by the 4,976 run.

## Can

- Run causally, streaming-compatible. Measured, not assumed.
- Identify the target from a 5 s sample — 37.6 % enrolment sensitivity on `sir0`,
  where "keep the louder voice" no longer works.
- Output at roughly the right volume; the mute is closed.
- Tell speech from silence — ~7 dB louder when the target talks, vs 2.45 dB for
  the control.
- Train 10 epochs in 1.45 h, checkpoint, and resume without losing state.
- Run in real time — RTF 0.528 mean on a laptop CPU, latency 162 ms against a
  200–300 ms budget, no chunk missing the 80 ms deadline.
- Score all three of its own metrics — LCF-WER, ICR, FR (NRR until 2026-09-03),
  judge-agnostic with the transcriber swappable, validated by reproducing the C2
  floor and ceiling exactly.
- Separate interference from artefact — SIR/SAR (`separation.py`, 12 tests).
  Needs clean sources, so never on AMI.
- Beat doing nothing by ~1 dB held-out, against 0.55 dB a week ago.
- Be scored on a live model, and against a second system.
- Resume a judge run without paying twice. Anchors are judged once per
  instrument; estimates are keyed by audio content, so a retrained checkpoint is
  judged fresh but a byte-identical re-render is free. Every answer hits disk
  before the call returns.
- Tell a speech-free clip from a speech-bearing one before spending anything —
  anchors from the manifest, estimates by Silero VAD 6.2.1 (B2).

## Cannot

- **Beat an off-the-shelf checkpoint.** WeSep leads on every metric.
- **Generalise fully** — the train/held-out gap still reaches 4.16 dB by the last
  epoch, against 5.68 before.
- **Separate well** — 2.58 dB held-out against 1.59 for doing nothing. Better,
  still thin.
- **Match level per utterance.** `L_gain` fell only 3 % over its whole run; it
  works as a constraint against going silent, not as a level regression target.
- **Show a rank inversion between the judge and conventional metrics.** Measured
  2026-09-03 on n=2: no inversion, all five instruments agree, judge margin the
  widest. Not yet a conclusive refutation — the pair is 30 points apart and an
  inversion is a claim about close systems.
- **Make per-trial claims through the judge.** 16.0 points of spread across five
  identical calls on one ambiguous mixture; it sometimes transcribes one speaker
  and sometimes interleaves both. Per-trial needs k≥3 and averaging.
- **Trust the judge to report absence** — see the fabrication finding above.
- **Adapt to how hard the trial is.** SIR and SAR change are flat across
  easy-to-hard (+3.80 to +4.32, −17.45 to −21.41) while the outcome swings −4.2
  to +23.1. `decisions-pending.md` D11 carries the fixes.

## Not started

- **A near-tie third system.** The per-band gate (D13) is now the experiment that
  matters, because n=2 at 30 points apart cannot test ranking. The mix-back sweep
  (D11) is the cheaper route — it turns one checkpoint into a family with no
  retraining.
- **The absent rows.** B4's invented-speech row and the `noise_only` cases, ~200
  calls, `--condition ""`. The only place the gate does real work, and where the
  judge and ASR differ most.
- ~~**J2b, the open-weight anchor.**~~ **CUT 2026-09-03** for time. Judge results
  are auditable (raw responses kept) but not re-runnable without API access —
  a stated limitation. decisions-m4.md 2026-09-03.
- **`eval_public` / `eval_private` anchors** through the judge. ~$5 once, then
  reusable. `eval_private` is scored last and once.
- **AMI** — the only real-audio check in the project.
- **The text reference condition** (spec note 10), and J3's ICR threshold
  sign-off (a free re-score from stored text).
- **J1's ~50-trial full-duplex confirmation run.**

B4's absent-trial rule and B5's normaliser are written into
`metric-definitions.md` §3.1, not just the code.

## Milestone scoreboard

| | status | open items |
|---|---|---|
| M0 data | closed | — |
| M1 architecture | closed | — |
| M2 baseline trained | functionally complete | 1 (band-plan / `w_m` ablations) |
| M3 conventional evaluation | 2 of 3 done | 1 (listen to the outputs) |
| **M4 the metric** | **closed 2026-09-03** | — J3 signed off; J2b and the text condition both cut, each with its reason logged |
| M5 second model | designed, not built | 23 |
| M6 the comparison | table exists, no near-tie in it | 9 |

The bottleneck is no longer measurement. M6's comparison table exists; what it
lacks is two systems close enough to rank meaningfully.

## Offline ASR — chosen 2026-08-28

`small.en`, int8 CPU, greedy, Whisper `EnglishTextNormalizer`. `tiny.en`'s floor
exceeds 100 % (it invents words); `medium.en` is better at 2.7x the cost.

**C2 closed 2026-08-30**, n=230 on `both` trials — the only row to quote:

| set | ceiling (clean) | floor (raw mixture) |
|---|---|---|
| `eval_public` (n=230) | 6.1 % | **57.4 %** |
| `sir0_val` (n=103) | 5.8 % | **65.2 %** |

Of every 100 words the target says, ~57 come out wrong if you do nothing, against
~6 on clean audio. That 51-point gap is the room the extractor has. The errors
are not mush: on one inspected trial the ASR transcribes the target perfectly for
17 words then switches to the *other speaker's* sentence. This replaces the
76.4 % quoted from a 12-trial pilot, which was wrong by 19 points.

**Open consequence, bigger than the number:** training is on `sir0`, symmetric by
construction, while `eval_public` keeps the original distribution where the target
is the louder voice 74 % of the time. Which set defines the benchmark is
undecided — a supervisor conversation.

Known artefact: `small.en` emits "you" on digital silence, 8 of 8 absent trials.
Filter before counting invented words.

## Compute

**1,244 s/epoch at 4,976 trials** on a Kaggle T4, batch 3 (measured 2026-09-01;
the earlier ~1,315 s projection was 6 % high). 18 epochs took 6.2 h, so ~25 fit
the 12 h cap. 19,938 trials would not.

Local preparation, all measured 2026-08-31: render 4,976 trials **1.2 h**,
enrolment bank at K=3 **1.5 h**, bundle and zip **~47 min**, split occupies
**15 GB**. `run_times.md` projected 1.26 MB/trial; the real figure is
**3.25 MB/trial**, because sir0 renders the interferer stem and two banks.

**A render is cheaper than it looks.** Raising `n_trials` appends rather than
resamples — the first 1,989 came back byte-identical — so `render_trials.py` skips
what exists. Its `config_md5` refusal is a conservative guard, not evidence the
audio changed.

Batch size is pinned at 3 where the config comment says 12. It must stay 3 for
any resume: `train.py` refuses a resume whose config differs from the checkpoint's.

## Next

1. **Build a near-tie system** — the per-band gate (D13), or the mix-back sweep
   (D11) for a no-retraining family. This is what converts M6's negative result
   into a real test.
2. **Resolve WeSep's streaming status** — run the scale-matched control,
   `scripts/probe_wesep_causality.py`. Until then its RTF row and any streaming
   claim stay open.
3. **Re-run the WeSep offline-ASR pass** so that row has a results artefact.
4. **Score the absent rows through the judge** — ~200 calls, `--condition ""`.
5. **Write up three results that already exist.** (a) No signal-domain measure
   predicts what the listener recovered (ΔSDR ~4 % of variance). (b) Judge and
   ASR agree on ranking but fail by opposite mechanisms. (c) LCF-WER on
   two-speaker mixtures is dominated by interferer suppression.
6. ~~**Do NOT render more data.**~~ **ANSWERED 2026-09-04, and it was worth it
   once.** 9,955 trials gave best held-out separation **2.900 dB** against 2.584,
   margin over pass-through **0.991 -> 1.307 dB (+32 %)**. Three points now make
   the curve log-linear — 0.55 / 0.99 / 1.31 dB of margin at 1,989 / 4,976 /
   9,955 — so **~+0.32 dB per doubling, still paying, and the next one costs
   ~20,000 trials** (~30 GB rendered, ~10 h upload, 21 h train). The original
   advice was directionally right about diminishing returns and is now measured
   rather than projected. **Do not render a fourth point** unless the downstream
   metrics show this one moved LCF-WER, which is not yet known.
   `decisions-m2.md` 2026-09-04.

Watch the train/held-out gap, not the total. Both totals fell the whole way
through the run that overfitted.

## The open architecture question

**On size, the answer still points against changing anything:** the model has
enough capacity to memorise its training set, so bigger or richer overfits
sooner. Conditioning changes (D1/D4a) stay cheap and additive; replacing the
backbone means retraining from zero.

**On the output there is a live question.** The model invents ~9 % of its output
energy (absolute SAR +10.34 dB), and a mask can only attenuate bins that already
exist — it cannot synthesise. The artefacts are therefore imperfect attenuation
(spectral holes, musical noise, phase damage), a property of the
parameterisation, not the weights.

M5's artefact-penalty retrain tests it, and both outcomes are results. If
artefact falls without suppression falling, masking was merely being applied too
aggressively. If SAR cannot improve without giving up SIR, that is evidence
masking is the wrong output parameterisation for this task — the argument for a
mapping or generative output, reportable even though building it is out of scope
before the freeze. `milestones.md` M5.
