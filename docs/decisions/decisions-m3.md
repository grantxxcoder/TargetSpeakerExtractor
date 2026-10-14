# Decision Log — M3 (baseline evaluated conventionally)

Covers the **conventional** evaluation instrument and what it says about the data: the offline ASR, SI-SDR / DNSMOS / offline-WER scoring, and latency and RTF. The new metric and its judge are in `decisions-m4.md`, because M3 exists to produce the "conventional metrics" column that M6's divergence table compares the new metric against (`metric-definitions.md` 6).

Created 2026-08-31 when the log was split by milestone. See `milestones.md` M3.

**Why an M0 decision lives here.** C2 (how hard the task should be) was an M0 data question, but answering it needed an evaluation instrument, which only existed from 2026-08-28. `decisions-m0.md` closed on 2026-08-18, so it is logged here.

---

## 2026-08-28 — Offline ASR pinned: `faster-whisper small.en`, int8 CPU, greedy

**Decision: `small.en`, int8 on CPU, greedy decoding, Whisper `EnglishTextNormalizer` (B5).** Pinned string: `faster-whisper==1.2.1:small.en:int8:cpu:greedy`.

**Logged here on 2026-08-31.** Chosen and used on 2026-08-28, but only recorded in `experiments/results/RESULTS.md`. CLAUDE.md requires the exact instrument on every reported number. No number changes.

### The evidence, 15 `eval_public` trials

| model | ceiling (clean target) | floor (mixture) | s/clip |
|---|---|---|---|
| `tiny.en` | 8.9 % | **123.0 %** | 0.7 |
| **`small.en`** | **4.0 %** | **64.6 %** | **3.0** |
| `medium.en` | 1.6 % | 50.4 % | 8.2 |

**Why not `tiny.en`.** Its floor exceeds 100 %: it hallucinates extra words rather than missing target words, so a metric built on it cannot rank systems.

**Why not `medium.en`.** Better on both ends, but 2.7x the cost of `small.en`, multiplied across M3's offline WER, M4's k>=3 judge repeats, M6's comparison and the text reference condition. The accuracy gain changes no ranking; the cost changes what fits in the schedule.

**The n=15 figures above are model-selection evidence, not results** (B6 minimum: 200 scored trials). Quote the n=230 rescoring below, which showed the 12-trial pilot's quoted 76.4 % floor was 19 points wrong.

### Two roles, kept separate

Required by `metric-definitions.md` 5. One checkpoint serves both. This is permitted but must be *stated*, since a shared error profile could flatter the text condition:

| role | belongs to | inside the latency budget? |
|---|---|---|
| **response ASR** | the measuring instrument | no |
| **front-end ASR** | the system under test, text condition only (3.5) | **yes** |

J1's closure on 2026-08-31 (`decisions-m4.md`) removes the *response* role: an audio-in / text-out judge replies in text, so there is nothing to transcribe. The front-end role remains.

### Known artefact, must be filtered

`small.en` emits **"you"** on digital silence (8 of 8 absent trials). Filter it before counting invented words, or B4's invented-speech row reports a Whisper quirk as extractor hallucination. **Re-measure the artefact set if the ASR is ever changed.**

### Hard constraint this creates

`small.en` may **not** be the differentiable training proxy. CLAUDE.md requires the proxy to be a different model family from the judge, and the evaluation instrument must not be trained against. See `decisions-pending.md` D9.

---

## 2026-08-30 — C2 CLOSED. Task difficulty measured at n=230 and accepted

The last open M0 item. Scored from `experiments/results/transcripts.csv` (`faster-whisper small.en`, int8 CPU, greedy, Whisper `EnglishTextNormalizer` per B5), which already held 1,220 trials transcribed clean and mixed. Only the scoring was new; it had never been done at scale.

### What C2 asks

How hard the task should be, as two numbers. The **floor** is an off-the-shelf ASR's word error rate on the raw mixture (what is lost if you do nothing). The **ceiling** is its WER on the clean target (the best possible). The gap is the extractor's headroom. A floor too low or too high leaves the metric unable to separate systems. The declared target band was 60-80 %.

### Measured, `both` condition only

| set | n | ceiling (clean) | floor (mixture) | mean SIR | interferer louder |
|---|---|---|---|---|---|
| `eval_public` | 230 | 6.1 % | **57.4 %** | +4.9 dB | 26 % |
| `sir0_val` | 103 | 5.8 % | **65.2 %** | -0.7 dB | 54 % |

**Decision: the measured range 57.4-65.2 % is accepted as the task difficulty.** It straddles the lower edge of the 60-80 % band, which was a pre-data target, not a constraint. Nothing is re-rendered and `overlap_ratio` stays un-narrowed (B1 says narrow it last).

About 57 of every 100 target words come out wrong. On `eval_public-42-000132` the ASR transcribes the target perfectly for 17 words, then **switches to the interferer's sentence**. The number measures "the machine listened to the wrong person", which the extractor exists to prevent. The 51-point gap from 6.1 % to 57.4 % is the room available.

### This corrects the number of record by 19 points

`RESULTS.md` carried **76.4 %** from a 12-trial pilot. At n=230 it is **57.4 %**, which is why B6 sets a 200-trial minimum. Despite its model-selection label, 76.4 % had been quoted as "the task's real floor". Do not use it again.

### Two things that must travel with the number

1. **Never quote the pooled figure.** `eval_public` pooled is 40.7 %, dragged down by `target_only` (floor 7.1 %, since with no interferer the "mixture" is near-clean). The task's floor is always the `both` row.
2. **The eval and training sets differ in difficulty.** Training is on `sir0`, symmetric by construction; `eval_public` keeps the original distribution, with the target louder 74 % of the time. The 7.8-point floor difference is a train/eval mismatch, not a measurement artefact. **Which set defines the benchmark is still open**, and now matters more than the difficulty. If the answer is the second, rendering a symmetric eval set costs ~2 min for 500 trials.

### Consequences

- **C2 moves to closed in `decisions-pending.md`**, accepted 2026-08-30. The supervisor conversation it called for should confirm it, not re-open it; it is really about the eval-set question above.
- `eval_private` is also fully transcribed (500 trials) but held back: not scored here, and not to be used for calibration.
- Absent trials have no reference text, so no WER. They are the invented-words check (`small.en` emits "you" on digital silence, 8/8, 2026-08-28).
- The ceiling is ~6 %, not ~3 %: `small.en` does worse on reverberant LibriSpeech than the pilot suggested. Any "we recovered X % of the ceiling" claim must use 6.1 %.

---

## 2026-08-31 — Floor and ceiling broken into error types: the floor is insertion-dominated

**Measured, not decided.** The error split changes what the extractor's job is, which a bare WER cannot show.

Same data as the C2 entry: `eval_public`, `condition=both`, n=230, 6,471 reference words, `faster-whisper small.en`, Whisper `EnglishTextNormalizer`, scored from `experiments/results/transcripts.csv` with no new ASR run. Each rate is an error type as a share of reference words, so the three sum to the WER.

| system | LCF-WER | substitutions | deletions | insertions |
|---|---|---|---|---|
| **floor** (raw mixture) | 57.4 % | 23.2 | **3.5** | **30.8** |
| **ceiling** (clean target) | 6.1 % | 2.7 | 0.4 | 2.9 |

### What it means

**Doing nothing means the listener hears too much, not too little.** Only 3.5 % of target words are lost outright. Over half the floor's error (30.8 points) is **added** words: the other speaker reported as the target.

**This reframes the task.** At these SIRs (mean +4.9 dB on `eval_public`) the target is rarely buried. The job is mostly *removing the other voice*, not *recovering hidden speech*, so improving audibility without suppressing the interferer cannot move this number much.

**It is also the empirical case for ICR.** WER counts an inserted interferer word and an inserted hallucination the same. Insertions dominate the floor, so LCF-WER most needs ICR (`metric-definitions.md` 3.2), which says *whether the inserted words were the other speaker's*. Without it, the largest part of the headline number is undiagnosed.

### Caveat that must travel with these numbers

**The alignment minimises total edits, so the substitution/insertion split is the cheapest explanation, not necessarily the true mechanism.** When the ASR follows the target and then switches to the other speaker (seen on inspection, 2026-08-30), whether those words count as substitutions or insertions depends on the utterances' relative lengths. Treat the split as directional, never as a mechanism claim on its own.

### Why it is worth recording anyway

The two error types move **oppositely** as masking gets harder: fewer insertions (interferer removed), more substitutions (artefacts). The split is therefore a cheap probe of the artefact-versus-residue trade-off hypothesised in `metric-definitions.md` 1.

---

## 2026-09-01 — FIRST system row on the project's own metrics, and the model hurts easy trials

**The first end-to-end measurement of a trained extractor on LCF-WER, ICR and NRR**, after only floor/ceiling anchors. An offline ASR stands in for the judge, so this is not a live-model result, but the pipeline works end to end.

Checkpoint `models/model_sir0_5000-e7.pt` (4,976-trial run, `decisions-m2.md` 2026-09-01). Estimates in `experiments/results/2026-09-01-est-sir0-5000/`, 200 trials, CPU, whole-clip single forward pass. `sir0_val`, `condition=both`, n=103.

| system | LCF-WER | sub | del | ins | ICR@2 | mean leak | NRR |
|---|---|---|---|---|---|---|---|
| floor (unprocessed mixture) | 65.2 % | 32.9 | 9.3 | 23.1 | 67.0 % | 51.3 % | 0.0 % |
| **the model** | **59.1 %** | 28.1 | **12.6** | 18.4 | **54.4 %** | **39.1 %** | 1.0 % |
| ceiling (clean target) | 5.8 % | 3.4 | 0.7 | 1.7 | 0.0 % | 0.0 % | 0.0 % |

**Headroom captured:** LCF-WER **10.4 %** of the 59.4-point band; ICR@2 **18.8 %** of its 67-point band; mean leakage **23.8 %**.

**The model removes the interferer better than it makes the target intelligible**, capturing about twice as much ICR headroom as LCF-WER headroom. This is the first thing ICR has shown that LCF-WER could not.

### The finding: it helps hard trials and HURTS easy ones

| floor difficulty | n | mean change in LCF-WER | trials improved |
|---|---|---|---|
| easy, floor <25 % | 22 | **−4.2 pts (worse)** | 18 % |
| medium, 25–60 % | 27 | −0.6 pts | 44 % |
| hard, 60–100 % | 27 | +9.0 pts better | 37 % |
| very hard, >100 % | 27 | **+23.1 pts better** | 63 % |

`correlation(floor WER, improvement) = +0.33`. Overall **43 trials improved (mean +29 pts), 31 worsened (mean −16 pts), 29 unchanged.** The regressions cancel most gains, so the aggregate moves 6.1 points while single trials move by 100+.

**This is the artefact-versus-residue trade-off, measured.** On easy trials there is little interferer to remove, so the extractor mostly adds distortion. On hard trials, removing the interferer more than pays for it. **The optimum is therefore not "always filter"**, which motivates the mixture/estimate interpolation sweep.

**The error split agrees.** Insertions fell from 23.1 to 18.4 and substitutions from 32.9 to 28.1, but **deletions rose from 9.3 to 12.6**: some of the target goes out with the interferer.

### A divergence result already exists, without a judge

On the same 103 trials, whole-clip SI-SDR improved by a **mean of +1.99 dB** (median +1.40, better on 78 % of trials). Yet on content the model made **30 % of trials worse**, systematically the easy ones.

**Conventional signal quality and content fidelity disagree on this model, on this data, today**: a miniature of the thesis's central claim, before the judge exists. Label it as measured through an ASR, not a live model. This SI-SDR is whole-clip and unfloored, so it is not the same quantity as `L_pres` in training.

### What the good cases look like

For the write-up. `sir0_val-42-000152`, SIR +9.9 dB: on the mixture the transcriber reported the interferer's whole sentence before the target's (WER 195 %, 15 leaked words). The model's output transcribed **word-for-word correct, WER 0 %, 0 leaked**.

`sir0_val-42-000050`, SIR +0.9 dB, is more diagnostic: near-equal loudness, with the interferer **interleaved throughout** rather than prepended. WER fell from 138 % to 18 % and leaked words from 17 to 0. The remaining errors are mishearings (`snare` for `snake`). Keeping the louder voice or the first speaker could not have done this.

### One genuine defect

`sir0_val-42-000145`: WER rose from 40 % (floor) to 100 %. The output transcribes as `the the the the the the the the the the the the the the`, a **degenerate collapse on a single, already easy trial**, which NRR caught (1.0 % against the floor's 0.0 %). Listen to the file: mush rather than speech means a stability bug, not a quality issue.

### Status of the third metric

NRR is 0.0 % at the floor and 1.0 % for the model. With an ASR as judge it is near-useless, because a transcriber cannot decline; its one non-zero entry is the trial above. Read it as "not yet measurable".

---

## 2026-09-01 — SIR/SAR added. The artefact hypothesis is NOT supported in the form predicted

**Implemented** `src/live_model_metric/separation.py`, 12 tests. It splits what an estimate holds beyond the target into **interference** (not removed) and **artefact** (invented by the model). Borrowed from the BSS_EVAL decomposition of Vincent et al. (2006), with the scale-invariant framing of Le Roux et al. (2019).

**Method.** The estimate is projected onto the span of the three true sources. Anything no scaled combination of them explains was not in the microphone signal, so the model created it. **Artefact is defined by elimination, not subtraction**, since `estimate - mixture` is large for any working extractor.

It needs all three clean sources (`noise = mixture - target - interferer`, as the loader derives it), so it is **computable on constructed mixtures only, never on AMI**.

### Three design decisions, taken

1. **Scale-invariant**, to match the training objective's SI-SDR.
2. **Noise counts as a source, not as artefact.** Leaked noise was present, not invented, and the hypothesis is about *processing* artefacts.
3. **Scaling only, no allowed filter.** BSS_EVAL variants allow a filter before calling the residual artefact. The stricter form is used because the reference is already the reverberant target.

**A bug the tests caught.** An absolute epsilon in the denominator broke scale invariance: SAR swung 17 dB under a 7.5x gain, because with near-zero artefact the score tracked gain, not content. Fixed by flooring the denominator **relative** to the numerator with `TAU = 1e-3`, as `tau_pres` does in the training objective. Every score is now capped at **+30 dB** and exactly scale-invariant across a 1000x gain range.

### Measured on the 5,000-trial checkpoint, sir0_val `both`, n=103

| | change vs doing nothing |
|---|---|
| SIR (interferer removed) | **+4.33 dB** |
| SAR (artefact avoided) | **-19.66 dB** |
| SDR (the net) | +1.98 dB |

The output's absolute SAR is **+10.34 dB** against the +30 dB artefact-free ceiling: roughly **9 % of its energy is invented**.

**Consistency check:** SDR +1.98 dB against +1.99 dB from a separate SI-SDR implementation, 0.01 dB apart.

### THE PREDICTION FAILED, and this is the finding

Prediction: if processing artefacts damage the listener, word-error improvement will correlate with SAR.

| | correlation with LCF-WER improvement |
|---|---|
| delta SDR | **+0.20** (marginal at n=103, p about 0.04) |
| delta SIR | +0.16 (not significant) |
| **delta SAR** | **-0.05 (null)** |

**Trial-level artefact severity does not predict trial-level word errors.** The artefact hypothesis of `metric-definitions.md` 1 is **not supported in that form**, and is recorded as a failed prediction.

### What is actually happening: the model does not adapt

| bucket | dSIR | dSAR | dSDR | dLCF-WER |
|---|---|---|---|---|
| easy, floor <25 % | +3.80 | -17.45 | +1.34 | **-4.2 (worse)** |
| medium | +4.06 | -18.33 | +1.69 | -0.6 |
| hard | +5.06 | -21.05 | +2.53 | +9.0 |
| very hard >100 % | +4.32 | -21.41 | +2.24 | **+23.1 (better)** |

**The signal columns are flat; only the outcome swings.** The model applies the same transform everywhere, so easy trials do *not* get worse artefacts. **The same artefact costs nothing when there was a lot of interference to remove, and costs dearly when there was not.**

The model sees the mixture and the enrolment, so it could adapt, but it **does not**: `L_pres` averages over crops in which suppression pays and artefact costs the same everywhere, so no per-trial decision ever enters. See `decisions-pending.md` D11.

### The result this actually produces

**No signal-domain measure meaningfully predicts what the listener recovered.** The best, delta SDR, explains about **4 % of the variance** in word-error improvement, a more general divergence result than a rank inversion.

Caveats: n=103, so only delta SDR is marginally significant; and this is **through an ASR, not a live model**, so the live-model correlation is still open.

### Reading SAR without being misled

**Delta SAR always looks catastrophic.** The mixture is the exact sum of the sources, so it is artefact-free *by construction* and sits at the +30 dB ceiling; any processing drops below it. **Quote the absolute SAR (+10.34 dB), not the delta.**

---

## 2026-09-01 — Mix-back sweep: no global optimum, but the per-difficulty optimum spans the FULL range

D11's screening test, run as an instrument rather than a fix. `sir0_val` `condition=both`, n=103, alpha in {0, 0.25, 0.5, 0.75, 1}, offline ASR. `experiments/results/sweep_alpha_rows.json`, transcripts cached in `sweep_alpha_transcripts.csv`. 309 new transcriptions, ~15 min. **No retraining and no extra forward passes**: every alpha is a linear blend of two signals already on disk.

| alpha | LCF-WER | ICR@2 | SDR | SIR | SAR |
|---|---|---|---|---|---|
| 0.00 (do nothing) | 65.2 % | 67.0 % | −1.12 | −1.12 | +30.00 |
| 0.25 | 63.4 % | 66.0 % | −0.61 | −0.58 | +24.95 |
| 0.50 | 69.6 % | 67.0 % | −0.02 | +0.15 | +19.21 |
| 0.75 | 67.2 % | 62.1 % | +0.58 | +1.21 | +14.56 |
| **1.00 (current model)** | **59.1 %** | **54.4 %** | +0.86 | +3.21 | +10.34 |

**The instrument validated itself.** SDR, SIR and SAR are perfectly monotonic in alpha, as a linear blend predicts. The iSTFT is linear, so waveform blending equals mask interpolation and the blending confounds nothing.

### Globally, alpha = 1 wins. There is no interior optimum.

The model is already at the best *global* aggressiveness. **This is evidence against a global loss-side shift toward gentler masking**, i.e. against `BETA > 1` as a standalone intervention.

However, the sweep is an **imperfect proxy** for `BETA`: retraining at higher `BETA` learns a *different mask*, not the same mask blended with its input. So it does not rule `BETA` out.

### The decisive result: the optimum varies across the entire range

| difficulty | n | a=0 | a=0.25 | a=0.5 | a=0.75 | a=1 | best |
|---|---|---|---|---|---|---|---|
| easy <25 % | 22 | **9.6 %** | 11.2 % | 10.3 % | 11.7 % | 14.2 % | **0.00** |
| medium | 27 | 40.5 % | **36.8 %** | 58.6 % | 46.6 % | 40.9 % | **0.25** |
| hard | 27 | 81.9 % | 80.6 % | 82.7 % | 79.1 % | **73.4 %** | **1.00** |
| very hard >100 % | 27 | 137.4 % | 133.8 % | 131.4 % | 140.1 % | **113.6 %** | **1.00** |

**Easy trials want no filtering; hard trials want full filtering.** No global constant is right. Alpha = 1 wins globally only because the hard buckets dominate: they gain 24 points at alpha = 1, swamping the 5 points the easy trials lose.

### What an adaptive gate could buy

| | LCF-WER |
|---|---|
| do nothing | 65.2 % |
| current model | 59.1 % |
| **oracle, best alpha per difficulty bucket** | **56.9 %** |
| oracle per trial (cheats — uses the answer) | 52.5 % |

**A realistic gate is worth about 2.2 points** (6.6 is an unreachable ceiling). Weigh that against a 6.2 h retrain.

### The WER curve is non-monotonic, and that is itself the finding

65.2, 63.4, **69.6**, 67.2, 59.1. alpha = 0.5 is *worse than doing nothing*, which makes no physical sense against monotonic signal measures.

The blend is verified linear, so this is **n=103 noise plus transcriber nonlinearity**. **Trust the endpoints and the per-bucket pattern, not individual interior alpha values.**

It is another instance of the project's central claim: the signal moved smoothly across five settings while the content outcome jumped around.

---

## 2026-09-01 — DNSMOS added, and it disagrees with the content metric

**Implemented** `src/live_model_metric/dnsmos.py`. Four scores per clip, all 1-5, higher is better: `P808`, and `SIG` / `BAK` / `OVRL` from P.835. Ported from `microsoft/DNS-Challenge`, `DNSMOS/dnsmos_local.py`, retrieved 2026-09-01. Reddy et al. (2021) for P.808, Reddy et al. (2022) for P.835, both cited as borrowed.

**Non-intrusive: it needs only the degraded audio**, so it is **the only quality metric here that can run on AMI** and is essential to the real-audio transfer check.

### Validated against the reference implementation

The reference script's own class was run against the port on the same clip:

| | reference | port | difference |
|---|---|---|---|
| P808 | 2.146801 | 2.146801 | 2.4e-07 |
| SIG | 3.196597 | 3.196597 | 4.4e-16 |
| BAK | 3.030237 | 3.030237 | 0 |
| OVRL | 2.196043 | 2.196043 | 4.4e-16 |

Exact on three, float32 rounding on the fourth, same segment count.

### Both variants, and why

`metric-definitions.md` 6 was amended 2026-09-01 to report **both**. The original named only P.808, losing the point of 4. A first amendment named only P.835, losing the field's convention. The history:

| variant | role |
|---|---|
| **P.835** -> `SIG`, `BAK`, `OVRL` | **`OVRL` is the score that got gamed** |
| **P.808** -> one score | **the replacement the organisers switched to** |

### Three implementation facts that were easy to get wrong

1. **Personalised is a separate model file**, not just other coefficients: `pDNSMOS/sig_bak_ovr.onnx` against `DNSMOS/sig_bak_ovr.onnx`. Model and coefficients must match. **Personalised is right here, because target speaker extraction is personalised speech enhancement**: the correct output removes a speaker, which the standard model can score as damage.
2. **The polynomial correction applies to P.835 only.** P.808 is used raw. Raw P.835 outputs are not MOS scores, and skipping the correction silently gives plausible but wrong numbers.
3. **Short clips are looped, not zero-padded**, to fill one 9.01 s segment, because silence would score as bad audio. It does not trigger on 15-20 s trials but is replicated faithfully.

Models are snapshotted to `src/live_model_metric/dnsmos_onnx/` (2.5 MB) with SHA-256 recorded, like the normaliser and stopword list, because DNSMOS is a learned model that drifts between releases. `librosa==1.0.0` was added to match the reference mel filterbank exactly.

### Measured, sir0_val `both`, n=103, personalised

| system | P808 | SIG | BAK | OVRL |
|---|---|---|---|---|
| floor (do nothing) | 2.913 | **4.090** | 2.031 | **2.497** |
| **the model** | 2.937 | **3.366** | **2.266** | **2.237** |
| ceiling (clean target) | 3.550 | 4.175 | 3.592 | 3.429 |
| **change, floor -> model** | **+0.02** | **-0.72** | **+0.24** | **-0.26** |

### The SIG/BAK prediction held: two instruments, one conclusion

The SAR result predicted `BAK` up (suppression works) and `SIG` down (artefacts). **Both happened, and SIG fell three times as far as BAK rose.**

So **`SIG` tracks artefacts introduced and `BAK` interference removed**, the perceptual analogue of the SAR/SIR split. An independent human-perception model agrees with SAR (+10.34 dB, ~9 % of energy invented) that the damage outweighs the cleanup.

### THE DIVERGENCE, and it runs opposite to the prediction

| | direction |
|---|---|
| LCF-WER | 65.2 % -> **59.1 %**, **better by 6.1 points** |
| DNSMOS OVRL | 2.497 -> **2.237**, **worse by 0.26** |

**A human listener would say the model made the audio worse. However, the listener recovered more of the words.**

The prediction was the reverse, so this is **another failed prediction.** It is arguably the stronger result: **the conventional perceptual metric would have rejected a system that measurably helps the downstream task**, the failure mode `metric-definitions.md` 1 describes.

### Two further observations

**P808 is flat, +0.02.** The metric the organisers switched *to* is nearly blind to this model, while `OVRL`, the gamed one, moves. State this about both.

**The ceiling is 3.43 of 5**, because the reference is the *reverberant* target (A1) and DNSMOS was trained on denoising, not dereverberation. As with the 6.1 % ASR ceiling, **quote the instrument's own ceiling with any score.**

---

## 2026-09-01 — RTF and end-to-end latency measured. It keeps up on a laptop CPU

Closes the M3 item "measured algorithmic latency + RTF against the ~200-300 ms budget". `scripts/measure_rtf.py`, result in `experiments/results/2026-09-01-rtf-cpu/rtf.json`.

### Two numbers, one measurement

| | definition | requirement |
|---|---|---|
| **RTF** | processing time / audio duration | **< 1**, else the backlog grows without bound |
| **latency** | chunk + lookahead + processing | < 200-300 ms |

**The RTF deadline is the tighter one**: at 80 ms, finishing within the chunk's own duration is stricter than the latency budget. **If RTF < 1 the latency budget is met automatically**, so lead with RTF.

### Why chunks, and why 80 ms

**Timing a whole clip flatters streaming by up to 32x** (measured 2026-09-01). A whole-sequence LSTM call is one batched matrix multiply. Streaming is thousands of tiny matrix-vector products, bound by launch latency rather than compute. **Whole-clip RTF, including the 0.42 previously quoted from `make_estimates.py`, is not evidence of streaming ability.**

The chunk-size sweep, one LSTM, us per frame:

| chunk | 8 ms | 40 ms | **80 ms** | 160 ms | 500 ms | whole clip |
|---|---|---|---|---|---|---|
| us/frame | 245.9 | 45.4 | **26.4** | 16.1 | 9.9 | 7.5 |
| speedup | 1.0x | 5.4x | **9.3x** | 15.2x | 24.9x | 32.6x |

**It saturates early.** 8 to 80 ms buys 9.3x; 80 ms to the whole clip buys only 3.5x more. Latency is `chunk + 40 + compute`, so 160 ms chunks would sit exactly on the 200 ms limit, while 80 ms leaves margin. **80 ms chosen.**

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

Mean latency = 80 ms chunk + 40 ms lookahead + 42.2 ms compute. **No chunk misses the deadline.**

### Caveats that must travel with these numbers

**This is an estimate, not a streaming measurement.** With no stateful streaming path, chunks are processed **independently** and the output discarded. Timing is close, since a carried hidden state costs the same as a fresh one, but it misses state management and a frames-per-chunk difference (`forward()` pads each chunk; a streaming STFT would not). **10-20 % error.**

**The margin is not generous.** RTF p99 of 0.706 leaves 29 % headroom on the worst chunks, which a loaded machine could use up (a test suite here once went from 40 s to 16 min under memory pressure).

**Enrolment is not the bottleneck.** Per-chunk time is flat across enrolment lengths of 0.5-5.0 s (41.7-43.1 ms), so caching the enrolment embedding gains nothing.

**A prediction that was 8x wrong.** Extrapolating from one LSTM predicted ~5 ms per chunk; the truth is 42 ms. It ignored the 32-band estimator trunks, the STFT/iSTFT and the conditioning. **Do not extrapolate compute from one layer.**

### Still open

**The GPU figure.** No local GPU. The spec assumes server-class compute (`decisions-m0.md` 2026-08-07), so **the GPU number supports the claim** and CPU is the pessimistic case. One Kaggle cell.

**B11's latency decay curve is separate and expensive.** `lookahead_frames` shifts the feature sequence at train time, so each point on a quality-versus-latency curve needs a retrain. Evaluating a lookahead-0 model at other lookaheads is not valid.

## 2026-09-03 — WeSep cannot stream. `causal: true` covers the separator, not the normaliser

**Measured, not inferred.** `scripts/probe_wesep_causality.py`, checkpoint `tfmap_context_causal_100`, 8 s mixture / 2 s enrolment, seed 42, CPU.

| probe | cut 2 s | cut 4 s | cut 6 s | worst |
|---|---|---|---|---|
| determinism floor (same input twice) | — | — | — | 1.21e-05 |
| **A. scale-matched future** | 1.12e-02 (3.49 %) | 1.02e-02 (2.66 %) | 5.39e-03 (1.40 %) | **1.12e-02** |
| B. 5x louder future | 2.16e-02 (6.74 %) | 2.89e-02 (7.53 %) | 1.38e-02 (3.60 %) | 2.89e-02 |

Ours on the same protocol, 2026-08-24: **1.68e-08**. WeSep is ~6 orders of magnitude worse, and ~900x its own determinism floor.

### The mechanism, so this is a finding rather than a number

`causal: true` sits under `separator:` in their config, and the separator's RNNs are causal. However, `SubbandNorm` (`wesep/modules/separator/bsrnn.py:44`) applies `select_norm('GN', …)` = `nn.GroupNorm(group=1, C)` to a `(B, C, T)` tensor, normalising over all channels **and the whole time axis** before the separator runs. This is global layer norm (`gLN`): every output frame depends on the whole clip, future included.

**Two checks that it is global normalisation, not architectural lookahead.** The leak *shrinks* as the cut moves later (3.49, 2.66, then 1.40 %), whereas a fixed lookahead window would read ~0 beyond the cut. And the 5x-louder probe leaks only ~2.6x more, not unboundedly more.

### Consequences

- **WeSep is an offline system for our purposes.** Its 2026-09-03 judge and signal scores stand as *offline* scores. Never call it a streaming or causal baseline in the thesis.
- **Fixable in principle, not by us.** The standard causal replacement, cumulative layer norm, means retraining their checkpoint, which is out of scope.
- **Our own causal claim is unaffected and now has a contrast** (1.68e-08 vs 1.12e-02), worth one line in the report: a model can declare causality per-module and still be non-causal end to end.
- **Latency measurement is now academic** but was taken anyway (below).

### Two caveats that travel with these numbers

**The model is not bit-reproducible.** Determinism floor 1.21e-05 on identical input, most likely multi-threaded reduction order. The effect is ~900x that, so the verdict holds, but never quote probe values without the floor.

**Mode A equalises broadband RMS, not per-subband RMS**, so matched-RMS white noise still shifts per-band statistics a lot. It is a better control than mode B but does not cleanly separate lookahead from global normalisation; the `GroupNorm` reading does. A cleaner probe would perturb inside one band.

### Latency, taken with `scripts/measure_rtf_wesep.py` (new)

Protocol copied term for term from `measure_rtf.py` (80 ms chunks, 20 warmup, same percentiles). `measure_rtf.py` itself could not run: it calls `build_model()`/`load_state_dict()` from `train.py`, which the WeSep venv lacks.

**Measured, 2250 chunks, 23 min wall**, on the same i5-1135G7 / 4 threads as the baseline. `experiments/results/2026-09-03-rtf-wesep-cpu/rtf.json`.

| | WeSep | our baseline |
|---|---|---|
| RTF mean | **2.854** | 0.528 |
| RTF p99 | 5.300 | 0.706 |
| latency mean | **348.3 ms** | 162.2 ms |
| latency p99 | 544.0 ms | 176.5 ms |
| per-chunk mean / p99 / max | 228.3 / 424.0 / 552.9 ms | — |
| meets budget | **no** | yes |

**No chunk finishes inside its own 80 ms** (minimum well above the deadline), so the input backlog grows without bound. **Quote 2.854**, not the 4.85 from a 20-chunk smoke run earlier that day (too few samples).

Two reasons for the gap: 27.2 M parameters in the timed forward against our 7.19 M, and WeSep re-embeds the 5 s enrolment through a full speaker encoder on every 80 ms chunk. Ours also re-embeds per chunk (protocol matched), but only through our TF-Map cue. `--cache-fbank` gives a lower bound; neither mode hoists the speaker encoder.

**Parameter counts differ by basis; both are correct.** 33.46 M is the `avg_model.pt` state-dict total (project-state.md, includes the jointly-trained VoxCeleb ECAPA-TDNN); 27.2 M is the TSE model as instantiated, which is what was timed. Label which one you are quoting.

Not comparable to any published REAL-TSE latency figure: hardware, chunking and latency convention all differ.

---

## 2026-09-12 — the evaluation battery is one command, and it runs in two stages

`scripts/run_eval_suite.py`. Seven steps in a fixed order (estimates, signal, asr, judge, rtf, cue, leakage) against one checkpoint and one `--tag` that names every output directory.

**Why.** Evaluations assembled by hand from `docs/run_times.md` left `2026-09-06-eval-10000-signal-perceptual` on a different flag set from `2026-09-04-train-sir0-10000`. Three steps consume the first step's output directory and need the same `--split` and `--condition`. The battery is now one recorded definition, not a habit.

**It orchestrates and measures nothing.** Each number still comes from its owning script, in that script's results directory, with its own meta. No new metric, no changed default.

### Run it in two stages. The gate is `asr`

**Stage 1, `--steps estimates,asr`, ~33 min, no API budget.** ICR@2 and `mean_leak` directly measure interferer leakage: 58.5 % of our content-word error mass, correlating +0.622 with per-trial WER (decisions-pending.md D14, 2026-09-11). A leakage arm that does not move them has failed.

The numbers to beat, `model_sir0_10000-e6.pt`, `sir0_val` `both`, n=103:

| system | LCF-WER | ICR@2 | mean leaked |
|---|---|---|---|
| floor, raw mixture | 65.22 | 66.99 % | 51.30 % |
| control, epoch 6 | **59.52** | **50.49 %** | **34.58 %** |
| ceiling, clean target | 5.85 | 0.00 % | 0.00 % |

**Stage 2, `--steps judge,rtf,cue`, ~31 min + API.** Only if stage 1 moved.

**`signal` is optional and least informative.** SI-SDR and DNSMOS score the axis we do not optimise, where a leakage arm is expected to lose. Run it for the write-up only.

**Wall times printed by `--dry-run` are measured rows from `run_times.md`, not estimates.** `diagnose_cue.py` has no recorded row and prints as unmeasured.

`--limit N` is for smoke tests. It puts the limit in the output directory name, so a two-trial run cannot pass as a published one.

---

## 2026-09-12 — MEASURED: what `sir0_val` at n=103 can and cannot resolve. It cannot resolve 2 points

Paired bootstrap over trials, 10,000 resamples, seed 42, on the per-trial rows in `experiments/results/2026-09-12-leakage-share/per_trial.json`. Both systems are resampled on the same trial indices each draw (a paired difference, not two independent intervals).

| quantity | point | 95 % interval | resamples with the opposite sign |
|---|---|---|---|
| LCF-WER, state arm minus control | **+1.74** | **[−4.77, +10.85]** | **36.8 %** |
| mean leaked %, arm minus control | −1.93 | [−5.28, +1.30] | 12.0 % |

Per trial, the two systems are a coin flip: **WER better on 33, worse on 36, tied on 34.** Leakage: better on 19, worse on 16, tied on 68.

**Neither the harm nor the benefit of the state-teacher arm is established.** The +1.72 LCF-WER and the −3.88 ICR@2 (which looked like success) are deep inside trial-sampling noise: directions, not effects.

### The number to plan with

**At n=103 the 95 % interval on a system difference is about ±8 LCF-WER points.** Halving it needs 4x the trials; resolving a 2-point effect needs roughly (7.8/2)² ≈ **15x, about 1,570 trials**.

**This governs the rest of M5: an arm expected to move LCF-WER by less than ~5 points is not measurable on this evaluation as constructed.** Only big gaps resolve: our 59.5 against WeSep's 34.6 (25 points) and the ceiling's 5.8 (54 points).

### This is a different noise source from the judge SEM, and they compose

`project-state.md` records **judge SEM ≈ 0.5 over 103**: how much the aggregate moves when the same judge is re-asked about the same trials (listener repeatability). This entry measures trial sampling: how much a system difference moves with a different draw of 103 trials from the same distribution.

**Which applies depends on the claim; the thesis must state both.**

- *"System A beats system B on this benchmark"*: a fixed, leaderboard-style test set. Listener repeatability (~0.5) is the floor, and +1.72 clears it.
- *"This method reduces LCF-WER"*: the claim a thesis makes. The trial-sampling interval applies and **nothing is established.**

Report the generalisation reading as the headline. A win on 103 trials that vanishes under resampling is not a method result.

### What is still NOT measured

Run-to-run training variance (two identical configs, different seeds) sits on top of all this, and no same-config replicate exists anywhere in `experiments/results/`. These intervals are therefore a lower bound on the scatter.

### Cost of this measurement

Seconds, on data already on disk. It should have been run before the first arm.

---

## 2026-09-12 — CORRECTION: the benchmark's resolution depends on how PAIRED the comparison is. Same-model interventions resolve ~3 points

Earlier today this file recorded "an arm expected to move LCF-WER by less than ~5 points is not measurable on this evaluation as constructed." **That holds only for two independently trained models. In general it is wrong, by enough to change how M5 should be run.**

| comparison | n tied trials | measured | 95 % interval | verdict |
|---|---|---|---|---|
| state extension vs baseline (two trained models) | 34 / 103 | +1.72 | [−4.77, +10.85] | inside noise |
| mask floor 0.05 vs baseline (one model, one knob) | **53 / 103** | +3.81 | **[+1.09, +6.78]** | **REAL** |

**The interval halved because half the trials produce an identical transcript.** Two separately trained models differ on every trial, so trial-sampling noise enters twice. One model with one inference setting changed differs only where the setting bites.

**Consequence for planning:** tested at inference time on a fixed checkpoint, an idea is measurable at about 3 points; by retraining, not below ~8. **Test at inference first, always, wherever the idea allows it.** It is also cheaper.

### The mask floor is a MEASURED failure, with a measured mechanism

Floor 0.05 on `model_sir0_10000-e6.pt`, paired bootstrap, 10,000 draws, n=103:

| | measured | 95 % interval | |
|---|---|---|---|
| deletions | **−3.14** | [−6.93, −0.21] | **REAL, better** |
| substitutions | **+4.10** | [+1.56, +7.33] | **REAL, worse** |
| mean leaked % | **+5.39** | [+1.60, +9.67] | **REAL, worse** |
| insertions | +2.85 | [−0.33, +6.33] | inside noise |
| LCF-WER | **+3.81** | [+1.09, +6.78] | **REAL, worse** |

**In plain words, filling the mask's holes recovers target words the model was deleting, but adds misheard words and more of the other speaker, and the second effect is bigger.**

**Both halves are real.** The holes were destroying target speech. But the raw mixture contains both speakers, so it is the wrong fill; a target-selective fill is needed.

### The full sweep, three settings

| floor | deletions | mean leaked % | LCF-WER |
|---|---|---|---|
| none | 11.79 | 34.58 | **59.52** |
| 0.05 | 8.64 | 40.87 | 63.33 |
| 0.10 | 9.98 | 41.32 | 61.87 |
| 0.20 | **8.09** | **44.56** | 62.34 |

**Leakage rises monotonically with the floor, without exception**: the trend to trust. Deletions fall, best at 0.20 (-3.70). LCF-WER is worse at every setting but not monotonic, as noise at this n would be: only the 0.05 deletion result and the substitution results clear the interval.

The verdict holds across all three settings and rests on none alone.

**Keep the floor as a knob, not as a setting.** It is the only mechanism in the project that trades deletions against leakage at a measured exchange rate.

---

## 2026-09-12 — THE METRIC'S IRRELEVANCE FLOOR, measured by accident: 1.57 points for doing nothing

`report-todo.md` #9 has asked since M4 for a noise floor on the content metric: "a system difference smaller than it cannot honestly be claimed." A control arm produced one at no cost.

**The null intervention.** `scripts/postprocess_mask.py` at `--down 1.0` is arithmetically the identity: STFT of a finished estimate, multiply by exactly 1.0, invert. Only the analysis/synthesis round trip changes the audio: by a **median of -61.7 dB** relative energy over the 103 scored clips (worst -27.8, best -71.4), roughly one part in a thousand of the amplitude. That is inaudible, and orders of magnitude below any artefact the model produces.

**What it did to the metric.**

| | baseline | null intervention |
|---|---|---|
| LCF-WER | 59.52 | **57.95** |
| transcripts changed | — | **39 of 103** (22 better, 17 worse, 64 tied) |

**Doing nothing to the audio moved the headline number 1.57 points in the model's favour and rewrote 38 % of the transcripts.**

### Consequences, and they are not small

1. **1.57 points is the irrelevance floor.** A meaningless change can produce any improvement at or below it. It is a separate, tighter bar than trial-sampling noise, and applies even on a fixed test set.
2. **It calibrates the ASR stand-in, not the judge.** `small.en` is deterministic given its input; the instability is the input crossing decision boundaries. The judge has its own floor (SEM ~ 0.5, `project-state.md`); neither substitutes for the other.
3. **Report it beside every content number from now on.** Today, the state extension's +1.72 and this one sit near it; the mask floor's +3.81 sits clearly above it.

### It also validates the paired bootstrap, which is why this was worth the hour

The bootstrap correctly refused a change known to mean nothing:

| comparison | measured | 95 % interval | verdict |
|---|---|---|---|
| null intervention vs baseline | -1.57 | [-4.94, +1.75] | **inside the noise** |
| mask floor 0.05 vs baseline | +3.81 | [+1.09, +6.78] | **outside** |

**`scripts/bootstrap_difference.py` is now a validated instrument and should gate every content claim this project makes.**

### And it revises this morning's rule

The earlier entry said same-model interventions resolve about 3 points because half the clips tie. That stands, with the floor attached: **the usable range for a same-model intervention is above ~1.6 points, not above zero.** Below that, pairing gives precision on a number that does not measure the intervention.

---

## 2026-09-12 — the gain-following floor is UNTESTED, not refuted. The normaliser was wrong

The idea: the flat mask floor cuts deletions but fills holes with the raw mixture, which carries both speakers, so leakage rises. Instead, fill only where the target is believed present, following the model's own per-frame mean gain (84 % of its mask's information, and effectively its speech detector).

**The implementation scaled the floor by `frame_gain / max_frame_gain`, and the maximum is an outlier.** Over 20 clips the median frame's multiplier is 0.151, so a nominal floor of 0.10 became about **0.015 in a typical frame**: roughly a seventh of nominal, and well under the **0.05** flat floor behind today's one real deletion effect.

Every delta is below the 1.57-point irrelevance floor:

| | baseline | flat 0.10 | gain-following 0.10 |
|---|---|---|---|
| LCF-WER | 59.52 | 61.87 | 59.58 |
| deletions | 11.79 | 9.98 | 11.20 |
| mean leaked % | 34.58 | 41.32 | 36.11 |

It avoided the flat floor's leakage penalty and lost its deletion benefit, as almost no floor would.

**Recorded as untested**, not as "the targeted floor does not work": it tested about a third of the dose that works flat. A fair test needs a nominal value near 0.3-0.6, or a 90th-percentile normaliser instead of the max (that frame sits at 0.222 of the max, so the spread is wide either way).

**General lesson:** when a variant rescales an intervention by a data-dependent quantity, measure the *effective* strength that reached the audio before interpreting the result.

### The second setting made it worse, and suggests the variant is not just weak but wrong

| | baseline | gain-following 0.10 | gain-following 0.20 |
|---|---|---|---|
| LCF-WER | 59.52 | 59.58 | 61.93 |
| deletions | 11.79 | 11.20 | **12.02** |
| mean leaked % | 34.58 | 36.11 | 35.21 |

**Deletions are worse at the higher setting than the lower, and worse than baseline.** Filling holes cannot do that, so something else is happening.

**Likely mechanism (it should have been anticipated): a floor that varies frame to frame adds a time-varying component.** A flat floor adds a steady quiet copy of the mixture; this one adds a fluctuating copy, which is the musical-noise mechanism. The variant may create the artefact the floor exists to avoid.

Not bootstrapped: the variant is under-powered and confounded, so its number does not mean what it says. **Redesign before retesting**, with a percentile normaliser and a floor smooth in time, or the two effects stay entangled.

---

## 2026-09-12 — VOID: the internal-mask hysteresis runs measured a level explosion, not the idea

`2026-09-12-eval-hystint-sharp` returned LCF-WER **115.31**, deletions 80.21, **65 % of clips with no transcript at all**, and leakage 2.53 %. The leakage number means nothing: there was barely any usable audio to leak into.

**Cause: a bug in `apply_hysteresis`, not the model.** It restored each frame's mean mask magnitude after sharpening. Packing the same mean into fewer bins multiplies the RMS, which the output level follows: **+7.38 dB above the mixture**, against the baseline's -5.00 dB, twelve decibels too loud. The evaluation scored distortion.

**Why the output-domain runs did not show it.** They sharpen the rough ratio `|estimate| / |mixture|`, whose mean barely moves when small values go, so the restoration factor stays near 1. Our internal mask is nearly flat (84 % of its variance is one number per frame), so removing bins moves the mean a lot.

**Fixed:** the invariant is now per-frame RMS, verified to hold level to x1.000 across flat, realistic and rough masks. A guard leaves a frame untouched when nothing survives, instead of dividing by ~0. Both settings are being re-run.

**Void, and to be deleted rather than reported:** `2026-09-12-est-hystint-*` and `2026-09-12-eval-hystint-*`. The fixed runs are `*-hystfix-*`.

### The lesson, which is the reusable part

`tests/test_mask_postprocess.py` had a level-preservation test, but it asserted the mean, which the buggy code preserved. **A test that asserts the wrong invariant is how a bug reaches a results table.** It now asserts RMS, and a second test covers the empty-frame case.

Ask of any future post-processor: *what quantity does the audio's loudness actually follow, and is that the one being held fixed?*

---

## 2026-09-13 — the expanded dev split `sir0_privval`, and eval_private released from holdout

**Decision (Grant): eval_private stops being a holdout; `eval_public` alone is the final holdout.** This buys evaluation resolution, with the cost accepted in advance.

**Why.** 2026-09-12 measured `sir0_val`'s resolution at **+-8 LCF-WER points** over its 103 `both` trials. Arms moving the metric less than ~5 points are unreadable, most realistic arms move it by 2, and resolving 2 points needs ~15x the trials.

### What was NOT done, and why

**eval_private's 500 rendered trials were not appended to `sir0_val`.** Two blocking reasons, both measured:

1. **Its renders predate `interferer.wav`.** Each trial directory holds only `enrollment/mixture/target/meta`, but `evaluate.py:115` needs `interferer.wav` for every leakage metric (ICR@2, `mean_leak`, `wrong_from_interferer`). Its 249 extra `both` trials could not give the primary diagnostic (58.5 % of the error mass).
2. **It is a different distribution.** SIR spans [-5, +15] against `sir0_val`'s [-10, +10]; `regimes: null` (it predates the regime system by ten days); SNR mean 10.47 against 12.89; noise pool `tt` against `cv`; different `config_md5`. Merging would silently have made the benchmark ~5 dB easier, invalidating the 59.52 baseline, 65.22 floor, 5.85 ceiling and 1.57 irrelevance floor at once.

Re-rendering was therefore mandatory, which freed the choice of distribution.

### What was done

**A new split, `sir0_privval`, from eval_private's 20 released speakers.** `sir0_val` is untouched and byte-identical, so prior results stay reproducible.

| | sir0_val | sir0_privval |
|---|---|---|
| trials | 200 (103 `both`) | **2,800 (1,421 `both`)** |
| SIR | [-10, +10] | **[-10, +15]** |
| speakers | 40 (val) | 20 (eval_private) |
| noise | `cv` | `cv` |
| regimes | 0.6 base / 0.4 hard | identical |

**SIR is wider, not easier**: [-10, +15] is a superset of [-10, +10]. Report this split **per SIR band, never as one blended mean**, or blending re-hides the structure the widening was for.

**Noise is `cv`, not eval_private's `tt`.** `eval_public` draws from `tt`, so developing on it would leak noise into the final holdout.

**Verified, not assumed:** schema byte-compatible with `sir0_val`; 0 speaker overlap with `sir0_train`; 0 speaker overlap with `eval_public`; 0 noise-clip overlap with `eval_public`; all 2,800 rendered directories carry all six files.

### Why the easy end is worth having — measured, not assumed

Per-trial WER by SIR band on `sir0_val`, baseline `model_sir0_10000-e6.pt` (n per band 23-33, so directional):

| SIR band | n | raw mixture | ours | WeSep | ceiling | our leaked fraction |
|---|---|---|---|---|---|---|
| < -5 | 33 | 89.5 | 88.2 | 49.4 | 3.3 | 0.651 |
| -5..0 | 23 | 95.6 | 87.5 | 38.9 | 7.5 | 0.351 |
| 0..+5 | 24 | 63.2 | 51.3 | 40.7 | 10.8 | 0.234 |
| >= +5 | 23 | 39.5 | 30.8 | 23.8 | 5.6 | **0.080** |

**The two ends are different problems.** At SIR < -5 we beat doing nothing by only 1.3 WER (89.5 to 88.2), while WeSep reaches 49.4: we are effectively not working, and leakage is 65 % of our wrong content words. At SIR >= +5 leakage is 8 %, yet we sit **25 points above the clean-target ceiling** (30.8 against 5.6). That is our own damage (deletion and fabrication).

**The easy band is therefore the only clean instrument for fabrication** (41.5 % of the error mass, measured nowhere else).

### Registered as `sir0ext` in `train.py`

`SPLIT_MANIFESTS["sir0ext"]` pairs `sir0_train` with `sir0_privval`, so a `sir0` checkpoint is scored on the wider set **without retraining**, changing one axis only.

### Carry this into every write-up

`sir0_val` anchors do not transfer to `sir0_privval`. Re-measure the floor, ceiling and irrelevance floor there before comparing any of its numbers. Re-measurement started 2026-09-13.

`>= +5` now spans +5..+15, wider than the other bands. Report +5..+10 and +10..+15 separately, or the top band mixes two difficulties.

## 2026-09-13/14 — the anchors on `sir0_privval`, and they are NOT the sir0_val anchors

`experiments/results/2026-09-13-eval-privval-anchors`, `both` condition, faster-whisper `small.en` stand-in, not a live-model result.

| | sir0_val (103 trials) | **sir0_privval** |
|---|---|---|
| floor, raw mixture | 65.22 | **53.49** |
| ceiling, clean target | 5.85 | **3.90** |
| floor ICR@2 | 66.99 | 60.49 |
| floor mean_leak | 51.30 | 43.92 |

**The new set's floor is 11.7 LCF-WER points easier, as designed and warned.** SIR runs to +15 dB here against +10 on `sir0_val`, so the raw mixture is more often intelligible already. **Nothing measured on `sir0_privval` may be compared to a `sir0_val` number**; report it per SIR band.

The usable range narrows too: floor-to-ceiling is 49.6 points here against 59.4 on `sir0_val`. More trials bought resolution, the wider SIR range spent some dynamic range, and both must be stated.

---

## 2026-09-13 — w_struct DERIVED. The mask is 99.6 % of the way to a pure volume knob

`scripts/derive_w_struct.py`, `experiments/results/2026-09-13-wstruct-anchor-sir0`, 50 present crops, `model_sir0_10000-e6.pt`, 10 min CPU.

| anchor | mean L_struct |
|---|---|
| flat mask (pure volume knob) | 0.165366 |
| **our model's mask** | **0.164718** |
| oracle (ideal mask itself) | **0.000000** |

The oracle reading exactly 0.000000 is the wiring check: loss and target share the STFT grid. A nonzero value would mean the term supervises a misaligned object, which nothing else would catch.

**This independently confirms the 84.2 % finding**, in the units the loss uses: instead of "how much variance is one number per frame", it asks "how far is this mask's frequency shape from flat".

**w_struct = 46.2981** at a 15 % gradient share on the parameters, re-measured at 0.1500. It is large only because L_struct is numerically tiny beside the other terms.

### Two caveats that travel with the number

**Derived at chunk_s 1.0; training uses 4.008.** Memory forced it: one LSTM layer here allocates ~381 MB of activations at batch 1, twelve are kept for backward, and the script's first version took a 15 GB machine down. The share is a ratio on the same crops, so should be stable, but **this has not been checked**. Re-derive at 4.008 on a T4 before quoting it in the write-up.

**Small-sample instability is real:** 17.4 at 2 crops, 124.6 at 4, 46.3 at 50. Do not use a derivation under ~50 crops for anything.

---

## 2026-09-15 — Estimate rendering resumes; a resumed directory is provenance-locked

**Decision: `write_estimates` skips a trial whose `estimate.wav` is already on disk and complete, and refuses to resume a directory whose recorded provenance differs from the current run. `--force` re-renders and skips the guard.**

**Why resume.** A `sir0_privval` pass takes 3.4 h (measured, `run_times.md` 2026-09-14), and an interrupted pass restarted from zero. On 2026-09-15 that discarded a part-finished baseline render competing for CPU with a more urgent job. Inference is deterministic under `no_grad`, so a kept file is the one the pass would have written.

**Why refuse, not warn, on a provenance mismatch.** A second checkpoint pointed at an existing directory would blend two systems' audio under one label, and nothing downstream could tell: `evaluate.py` reads wav files and believes `meta.yaml`. A terminal warning is not a control.

**Two files, two questions.** `meta.yaml` is written last, so its presence still means the pass completed. The new `run.provenance.yaml` is written first and deleted on success, so its presence means a pass started and did not finish. A resume checks the ledger if present, else `meta.yaml`, guarding against both an interrupted and a completed prior run.

**A half-written wav is not reused.** Ctrl-C lands mid-write, and a truncated wav opens without error. A file is kept only if its length matches its mixture within `LENGTH_WARN_S`. This checks completeness, not identity; the provenance ledger stops a different model's audio being trusted.

**`n_trials` is unchanged** (trials in the directory), so older `meta.yaml` files read the same. `n_written` and `n_reused` are recorded too, and the `run_times.md` row counts only trials actually rendered, so a resumed pass cannot log a per-trial rate for skipped trials.

**Changes no number.** A full render with no prior output is bit-for-bit as before.

---

## 2026-09-21 — THE OTHER HALF OF THE DATA, scored for the first time

`scripts/eval_by_case.py`, `experiments/results/2026-09-21-case-suite-baseline-e6`, `sir0_val`, baseline `model_sir0_10000-e6.pt`, faster-whisper `small.en` stand-in (not a live-model result), 8 min.

**Every content number reported so far was on `both` (51.5 % of the split). The other 48.5 % had never been scored.**

| case | n | floor | ours | ceiling | reading |
|---|---|---|---|---|---|
| `target_only` | 47 | 4.00 | **5.34** | 2.80 | we make easy speech *worse* |
| `both` | 103 | 65.22 | 59.52 | 5.85 | reproduces the known numbers exactly |

Absent cases, where word error is undefined (words emitted per trial; zero is correct):

| case | n | floor | ours | median suppression |
|---|---|---|---|---|
| `interferer_only` | 42 | 30.81 | **24.81** | **-6.14 dB** |
| `noise_only` | 8 | 0.00 | 0.00 | -26.33 dB |

### The finding: we pass a whole stranger through as if it were the target

**On the 42 trials where only another person spoke, our output still yields 1,042 transcribable words against the raw mixture's 1,294: we remove just 19 % of a stranger's speech.** 83 % of those clips still produce words a downstream consumer would attribute to the target. Median suppression is **-6.14 dB**; the mean of -10.97 is carried by a few clips and should not be quoted.

**The model produced no clean silences of its own.** Non-response was 16.7 % (7 of 42) and the speech gate blocked exactly 7: every correct silence came from Silero.

**`noise_only` says nothing about the model.** The gate blocked 8/8 clips for *both* mixture and output. Only the -26.33 dB suppression is the model's, a signal-domain number with no listener.

### We damage speech that needed no separation

On `target_only` the mixture is nearly intelligible already (4.00) and we move it to 5.34. **+1.33 is below the 1.57-point irrelevance floor, so "worse" is not established**: report unchanged-to-worse, never a loss. Fabrication moves more and is not bounded by that floor: FR@2 from 21.28 to **29.78**, invented words per trial from 0.85 to 1.06, against a ceiling of 14.89. Substitutions drive it (3.18 to 4.26).

**`both` shows the same, unreported until now: FR@2 from 57.28 to 68.00.** Word error falls 5.70 points while fabrication rises 10.7: the headline improves while a separate failure mode grows.

### `both` by SIR band — read as headroom captured, not as WER

| SIR band | n | floor | ours | ceiling | **headroom captured** |
|---|---|---|---|---|---|
| < -5 | 33 | 80.29 | 76.94 | 3.27 | **4.4 %** |
| -5..0 | 23 | 79.05 | 72.55 | 6.10 | **8.9 %** |
| 0..+5 | 24 | 58.92 | 49.12 | 9.42 | 19.8 % |
| >= +5 | 23 | 33.56 | 29.42 | 5.80 | 14.9 % |

**Where the interferer is louder we capture under 9 % of the available headroom; elsewhere, 15-20 %.** This is the collapse, in the one unit comparable across bands of different difficulty.

**Not comparable to the 2026-09-12 table**, which averaged per-trial WER; this aggregates corpus-wide over the same trials (89.5/88.2 there, 80.29/76.94 here). Never quote one against the other.

### Consequences

1. **Leakage on target-absent trials is now a measured headline, not an inference.** It is the purest test of the conditioning: with no target, anything transcribable is a non-target speaker the model failed to reject.
2. **Score every future arm on all four cases.** Improving `both` while worsening `interferer_only` moves leakage around rather than removing it, and the old protocol could not see that.
3. **`noise_only` needs the gate disabled** (`--no-gate`) or a larger n to say anything about the model. 8 trials is too few either way.

---

## 2026-09-24 — `sir0_privval` DROPPED from evaluation

**Decision (Grant): no further estimates or scoring on `sir0_privval`.** Development numbers come from `sir0_val` through the judge; the only selection-free check left is `eval_public`.

**Why.** Its only listener was the offline ASR, which does not track the judge across our arms, and one system costs 3.6 h of CPU to render (`run_times.md` 2026-09-14) for a number no decision rests on.

**Evidence.** Rank agreement between the ASR and the judge over our 8 judged arms (`sir0_val` `both`, n=103):

| ASR scoring | Spearman | p |
|---|---|---|
| as reported | 0.16 | 0.71 |
| loop guard on (`content_probe.cap_errors_at_spoken`) | 0.75 | 0.03 |

Most of the disagreement is Whisper looping on 3 clips of the WER-selected e13 (ASR 63.24, last; judge 42.93, first), not a different hearing of the audio. Per-trial r = 0.825 (`2026-09-15-judge-predictability`).

### Consequences to carry

- **No selection-free number exists for any current checkpoint.** 1a e15 and 1c e12 were picked after seeing `sir0_val`; wer e13 by the in-loop probe on `sir0_val` clips. The one time this was measured, the bias was most of the effect: 1a e15 vs baseline **−6.76** on `sir0_val`, **−1.51 [−4.07, +1.04]** on `sir0_privval` (`decisions-m2.md` 2026-09-24).
- **That check must now come from `eval_public`**, easier than training (SIR +4.76 dB, 75.1 % target-louder; weak-point B1). Report it per SIR band.
- **CLAUDE.md limit (a) is unchanged.** `sir0_privval` is still never scored, filtered or selected on during training.
- **Exception, 2026-10-02 (Grant):** one SI-SDR voice-swap test, 1a e15 vs 1c e12, on all 1,421 `both` trials. No listener, no selection. Result in `decisions-m2.md` 2026-10-02.

### Kept, not deleted

Recorded `sir0_privval` results stay citable: baseline 48.90, struct e8 46.99 / e12 47.37, 1a e15 47.38. 1a e15's leakage cut, **−13.46 [−15.01, −11.90]**, is the split's only selection-free finding.

The wer e13 render (`2026-09-24-est-privval-cuecontext-wer-e13`) is abandoned at 215 of 1,421 trials: resumable (2026-09-15 entry), not to be scored. On disk: rendered split 5.4 GB, estimate dirs 3.4 GB + 247 MB partial. Delete only on request.

## 2026-09-26 — e21 through the conventional suite, next to e27

Same four checks and flags as e27 (2026-09-25), `sir0_val` `both`, n=103 unless stated. Run one at a time, latency first on an idle machine. Dirs `2026-09-26-*-cuecontext-wer-e21`; e27 in `2026-09-25-*-cuecontext-wer-e27`.

| | better | No processing | e21 | e27 |
|---|---|---|---|---|
| SI-SDR (dB) | higher | −1.12 | 2.29 | 2.29 |
| SIR, other speaker removed (dB) | higher | −1.12 | 9.17 | 10.21 |
| SAR, artefacts (dB) | higher | 30.00 | 6.28 | 5.44 |
| DNSMOS overall | higher | 2.50 | 2.31 | 2.34 |
| DNSMOS speech (SIG) | higher | 4.09 | 2.96 | 2.91 |
| DNSMOS background (BAK) | higher | 2.03 | 3.05 | 3.22 |
| latency mean / slowest 1 % (ms) | lower | – | 162.9 / 180.1 | 173.6 / 197.3 |
| real-time factor, slowest 1 % | < 1 | – | 0.75 | 0.97 |
| Whisper LCF-WER (stand-in, not the verdict) | lower | 65.22 | 56.72 | 53.64 |
| right voice, same / cross gender (76 / 78 decisions) | higher | – | 72.4 / 89.7 % | 75.0 / 91.0 % |

- **e21 has fewer artefacts than e27** (SAR +0.84 dB) and removes slightly less of the other speaker (SIR −1.04 dB), consistent with e27 cutting harder.
- **Both score below No processing on DNSMOS overall**: cleaner background, more distorted target speech. Not optimised for; report it as the cost of suppression. With no interval computed, the 0.03 overall gap is not claimed.
- **The latency gap is not a model difference** (same architecture, same compute). 43 vs 54 ms per chunk is the run-to-run spread of `measure_rtf.py`, itself an estimate (chunks processed independently, no state carried, 10–20 % error). Report the range across runs, never one value as the model's latency. e27's 0.97 is within that error of 1.
- **Whisper prefers e27 by 3.1 points; the judge ties them.** The selection stands on the judge (`decisions-m2.md` 2026-09-26).
- **Right voice: e27 leads by 2 of 76 same-gender and 1 of 78 cross-gender decisions.** Not a difference; no test was run.

## 2026-09-28 — Report's offline ASR is `large-v3-turbo`; `small.en` stays in the training loop

**Decision (Grant):** every offline-ASR figure in the report comes from `faster-whisper==1.2.1:large-v3-turbo:int8:cpu:greedy` (CTranslate2 build `mobiuslabsgmbh/faster-whisper-large-v3-turbo`, snapshot `0a363e9161cb`). The in-loop probe (`content_probe.py`) and the `evaluate.py` default stay `small.en`. Config `experiments/configs/eval_offline_asr_turbo.yaml`, seed 42. Evidence: `experiments/results/2026-09-28-eval-asr-turbo-{baseline,e21,wesep}/`, `sir0_val` `both`, n=103, commit `7afbd610f358-dirty` (the `--asr-model` change, committed in `f54d2bd`).

**Why:**
- `small.en` was pinned 2026-08-28 for CPU cost, not accuracy; `medium.en` was better on both anchors then. "Doesn't change any ranking" was asserted, never run.
- e21's lr schedule and checkpoint shortlist were tuned against `small.en`, so it is not an independent listener for the extension.
- Published extraction work scores with a large Whisper (SoloSpeech: `large-v3-turbo`). REAL-TSE itself uses Zipformer, so this is not its standard.

**Result, words wrong (LCF-WER, lower is better):**

| | small.en | large-v3-turbo | judge (3-run mean) |
|---|---|---|---|
| No processing | 65.22 | 63.04 | — |
| Baseline | 59.52 | 52.71 | — |
| Extension (e21) | 56.72 | 41.01 | — |
| WeSep | 34.60 | 29.13 | — |
| Target alone | 5.85 | 1.98 | 1.19 |
| Baseline → e21 | −2.80 | **−11.70** | −15.59 |

- **Ranking unchanged** (WeSep < e21 < baseline < floor), but turbo sees 4x the extension gain `small.en` saw, closer to the judge.
- **Turbo hears the other speaker more:** floor mean leak 51.30 against 62.78, ICR@2 66.99 against 76.70. It invents less: floor 2.61 against 1.59 words/trial.
- **This reverses a claim.** `small.en` deleted more on the baseline than on the floor (9.28 to 11.79), whereas turbo deletes less (6.93 to 5.70), like the judge (5.97 to 3.00). "The ASR is the brittle listener" (`results.tex` comment, claim a) was a property of `small.en`. Do not write it.
- Repeatable: 6/6 fresh re-transcriptions (2 trials x mixture, baseline, e21) matched the cached text. ~14 s/clip on this CPU, against ~3.

**Consequences to carry:**
- Never compare a `small.en` number with a turbo number. Every figure logged before today is `small.en`, including the 1.57-point irrelevance floor (2026-09-12) and the 57.4 / 6.1 `eval_public` anchors.
- The irrelevance floor has not been re-measured for turbo.
- Signal, perceptual and latency rows do not depend on the listener.

## 2026-09-28 — Latency re-timed: baseline and extension are indistinguishable

`measure_rtf.py`, 80 ms chunks, 4 threads, CPU, 2,250 chunks per run; baseline (`model_sir0_10000-e6.pt`) and e21 alternated, 3 runs each, one session. `experiments/results/2026-09-28-rtf-{baseline,e21}-r{1,2,3}/`, commit `f54d2bd-dirty`.

| | RTF mean | RTF p99 | latency mean (ms) | latency p99 (ms) |
|---|---|---|---|---|
| baseline | 0.79–0.98 | 1.38–1.70 | 183–198 | 230–256 |
| e21 | 0.64–0.89 | 0.72–1.73 | 172–191 | 178–259 |

- **The ranges overlap on every column**, and one model's own spread (e21 mean RTF 0.64–0.89) exceeds any gap between the models. The report's earlier 0.783 vs 0.536 compared runs 20 days apart: drift.
- **Both keep pace on average** (mean RTF < 1 in 6/6 runs). **The slowest 1 % do not** in 5/6 runs (only e21 r3 < 1), so a backlog can form under bursts.
- **The machine was not fully idle:** `report.pdf` was rebuilt at 15:17, inside baseline r3. The tails (p99 ~2x mean in 5/6 runs, 1.1x in e21 r3) point to background load, so the ranges are an upper bound on the real spread. WeSep was not re-run: its 2.85 is far outside this drift.
- **Report:** the extension's "yes" becomes "mean only". Claim only that the two are within run-to-run variation; e21 adds 0.09 M parameters.

**Update, same day: the idle re-run supersedes the table above.** Same command, nothing else open, `powerprofilesctl set performance`, 60 s settle. `experiments/results/2026-09-28-rtf-idle-{baseline,e21}-r{1,2,3}/`. All six clean (p99 / mean 1.13–1.30, against ~2x in the first set). Median (range):

| | RTF mean | RTF p99 | latency mean (ms) | latency p99 (ms) |
|---|---|---|---|---|
| baseline | 0.69 (0.66–0.70) | 0.79 (0.79–0.86) | 175 (173–176) | 183 (183–189) |
| e21 | 0.69 (0.68–0.71) | 0.78 (0.77–0.82) | 175 (175–177) | 182 (182–185) |

- **Identical speed:** medians 0.692 vs 0.687, inside each other's range, as +1.3 % parameters predicts.
- **Both keep up at p99 in 6/6 runs** and stay under 200 ms delay. The earlier "p99 > 1" was background load and the balanced profile, not the models.
- **The p99 margin is 14–23 %**, about the method's stated 10–20 % error. Claim "keeps up on an idle CPU", not "with room to spare"; a loaded machine breaks it (first set: 5/6 runs over real time at p99).
- Report table: median (range), "meets budget" yes for both.
