# Research Plan

**Written:** 2026-08-06 · **Rewritten:** 2026-08-07 for the re-scoped objective
**Horizon:** ~3 months to final submission (2026-11-05), *including* write-up
**Supersedes:** the metric-first-but-baseline-replication plan of 2026-08-06.
Decisions are logged in `docs/decisions.md`.

---

## 0. The thesis in one paragraph

Conventional target speaker extraction is optimised and scored on how good
the separated audio *is*. For a voice assistant, what matters is what the
downstream live speech-to-speech model *understands*. These are not the same
thing, and our supervisors have observed them diverging. This project
defines a metric for the second thing, uses it to characterise the
divergence across existing systems, and then trains a streaming TSE model
with differentiable proxies aligned to it — with the live model held out as
an independent judge throughout.

Three claims, in decreasing order of confidence that they will land:

1. **A metric.** Live-model content fidelity is definable, gaming-resistant
   and measurable. (`docs/metric-definitions.md`)
2. **A finding.** Signal quality, perceptual quality and offline WER do not
   predict live-model content fidelity — and conventional TSE can improve
   the former while degrading the latter.
3. **A model.** Training against differentiable proxies aligned to the
   objective improves live-model content fidelity over a
   conventionally-trained streaming baseline.

Claim 1 is the deliverable that cannot be cut. Claim 2 needs no training at
all. Claim 3 is where the compute goes and it is the one that can fail.

---

## 1. Three constraints that shape everything

### 1.1 The judge is a black box

A live speech-to-speech API cannot be backpropagated through, is stochastic,
costs money per call, and changes silently over time. Everything follows
from this:

- Training uses **differentiable proxies**, never the judge.
- The proxy must be a **different model family** from the judge, or the
  benchmark measures overfitting to one evaluator. This is the single most
  important methodological rule in the project.
- Every number carries a model ID and a date.
- Trial-set size is bounded by **API budget**, not by data availability.

### 1.2 Compute is the binding constraint

Confirmed: Kaggle GPUs. University HPC probably, not yet secured. No local
GPU.

Kaggle in practice: ~30 GPU-hours/week, hard 12-hour session limit, single
P100 or dual T4, constrained working disk. Every training run must
checkpoint and resume **from day one** — retrofitting that after a session
dies at hour 11 costs a week.

A causal BSRNN-class model on LibriSpeech-derived data is on the order of
**40–70 wall-clock hours** for one run on a P100. Budget for **one base
training run plus two to three fine-tunes**, not ten. Any plan implying more
is fiction.

This is why claim 3 is structured as *train once, then fine-tune variants* —
fine-tuning from a converged base is a fraction of the cost and lets the
proxy-objective comparison be a controlled A/B rather than two independent
runs.

### 1.3 Ten technical weeks, not thirteen

Roughly **10 weeks technical, 3 weeks writing**, and the writing weeks are
not compressible — they are what gets graded. **2026-10-14 is the hard
freeze on new experiments.**

---

## 2. The four legs

Strict priority order. Everything shares one evaluation harness so the
infrastructure is paid for once.

| # | Leg | Claim | GPU cost | Cut if behind? |
|---|-----|-------|----------|----------------|
| 1 | Metric definition + scoring harness + trial sets | 1 | ~0 | **Never** |
| 2 | Benchmark study: existing systems × judges | 2 | ~0 | **Never** |
| 3 | Streaming TSE baseline, conventionally trained | 3 | high | No — shrink it |
| 4 | Proxy-objective fine-tune + ablation | 3 | medium | **Cut first** |

Legs 1 and 2 are CPU/API work. They do not compete with training for the
Kaggle quota, and they constitute a complete thesis on their own. **If every
training run fails, legs 1 and 2 still submit.** That property is the reason
for this ordering — invert it and a training failure leaves nothing.

Leg 3 exists to give leg 4 something to fine-tune from, and to give leg 2 a
system we fully control and understand.

### What leg 4 is not

It is not "a fundamentally new architecture." In ten weeks on Kaggle that is
not on the table, and claiming it invites a bad viva. Leg 4 is a controlled
comparison: same architecture, same data, same base checkpoint, different
training objective. That is a clean, defensible, publishable experiment, and
it is directly aligned with spec note 8.

---

## 3. Leg 1 — the metric and its harness

Full specification in `docs/metric-definitions.md`. Summary of what has to
get built:

**Trial construction — constructed set (primary).** LibriSpeech-derived
mixtures with a target, ≥1 interferer, real noise (WHAM!-style) and
reverberation (WHAMR!-style RIRs). Gives exact verbatim ground truth for
both target and interferer, a true clean-target ceiling, and controllable
overlap ratio, SNR and device mismatch. Speaker-disjoint splits.

**Trial construction — AMI set (secondary).** REAL-T-style construction (Li
et al., Interspeech 2025): use existing diarisation annotations to find
naturally overlapping segments as mixtures, and ≥5 s non-overlapping
same-speaker segments as enrollment. Distant mic as mixture, headset (IHM)
as the **approximate** ceiling. This is the real-audio transfer check, and
without it we only ever measure ourselves in training conditions.

**Judge harness.** Fixed prompt, fixed response-transcription ASR, pinned
model IDs, k≥3 repeats per trial, cost accounting. At least one open-weight
judge alongside the closed API, for reproducibility. **Two input paths to the
judge — audio and text — with the modality recorded on every result**, since
spec note 10 permits either as the extractor's output. Build both from the
start: retrofitting a second path through a harness that assumes audio is
more work than allowing for it now.

**Conventional metrics alongside.** SI-SDR, DNSMOS-P808, offline ASR WER —
computed on the same trials, specifically so leg 2 can show where they
diverge from LCF.

---

## 4. Leg 2 — the benchmark study

This is claim 2, and it is the highest value-per-GPU-hour work in the
project. No training required: run existing systems over the trial sets and
score them.

Systems to include:
- Unprocessed mixture (floor)
- Clean target (ceiling)
- ≥2 off-the-shelf pretrained TSE models, ideally one streaming and one
  offline, to separate the effect of causality from the effect of extraction
- A conventional speech enhancer, as a "denoising without extraction" control
- **Text reference condition**: extractor → off-the-shelf streaming ASR →
  text → judge, with its own text floor (ASR on the raw mixture) and text
  ceiling (ground-truth text). Spec note 10 permits a text output; this is
  how we measure it without building a second system. No training, one extra
  harness path, and it tells us how much content is recoverable at all.
  See `docs/decisions.md`, 2026-08-07 output-modality decision, and
  `docs/metric-definitions.md` §3.5 for why it is a reference condition
  rather than a rival build target — and why it is *not* an upper bound.
- Later: our own leg-3 and leg-4 models

Judges: ≥1 closed live API + ≥1 open-weight speech-to-speech model.

**The result this is designed to produce:** a table where the ranking of
systems by SI-SDR / DNSMOS / offline-WER differs from the ranking by
LCF-WER. If that divergence exists, the metric is justified and the thesis
has its central finding. If it doesn't — if conventional metrics predict
live-model fidelity perfectly well — that is *also* a publishable negative
result, and it is better to discover it in week 5 than week 11. Either way
this leg de-risks the project.

**Week-1 check with a large payoff:** find out which pretrained streaming
TSE checkpoints are publicly available (WeSep/WeSep-family releases,
HuggingFace, USEF-TSE's released code). A usable public checkpoint would
strengthen leg 2 immediately and reduce leg 3 from "train from scratch" to
"fine-tune from a released model", which changes the entire compute picture.
Ten minutes of searching, potentially weeks saved.

---

## 5. Legs 3 and 4 — the model

**Output modality: audio.** Settled — see `docs/decisions.md`, 2026-08-07.
Text output is measured as a benchmark reference condition (§4), not built.
Nothing in legs 3 and 4 changes because of it: the extractor produces a
waveform, the proxy losses are computed on that waveform, and the text row is
produced by bolting an existing ASR onto the same checkpoint at eval time.

**Architecture.** Causal BSRNN-family extractor with TF-Map + speaker-
embedding conditioning. Rationale: it is the best-understood strong
streaming TSE design, the challenge evidence shows the causal TF-Map variant
is unusually strong for an online system, and reference implementations
exist to read. We are borrowing it as a well-characterised instrument, not
replicating it as a result — see `docs/decisions.md`.

Cite: Luo & Yu (TASLP 2023) for BSRNN; Zhang et al. (ICASSP 2025) for
TF-Map / multi-level speaker representation.

**Leg 3 training (conventional).** SI-SDR + multi-resolution STFT, with two
cheap components borrowed from CARTSE (Li & Seki, 2026):
- **target-absent training** — a substantial fraction of examples with no
  target present, split loss (masked SI-SDR when present, push-to-silence
  when absent). Addresses false alarms and the interruption goal.
- **channel-gap enrollment augmentation** — random RMS-preserving EQ curves
  on the enrollment, so conditioning learns device invariance. Addresses the
  microphone-mismatch concern in the spec.

Both are ~a day of work each and are separable, citable components.

**Leg 4 training (proxy-aligned).** Fine-tune the leg-3 checkpoint with
added differentiable proxies:

1. **Frozen-encoder feature matching** (primary proxy) — match intermediate
   activations of a frozen ASR/SSL encoder between output and clean target.
   Cheaper than full ASR cross-entropy, precedented by CARTSE's Zipformer
   feature-matching loss and PS4's proxy objectives.
2. **ASR cross-entropy** (stretch) — full differentiable ASR in the loop.
   Roughly doubles memory and step time; only attempt if leg 4 is on
   schedule.
3. Speaker-similarity and target-activity terms as auxiliaries.

**The rule, restated because it is easy to violate accidentally:** the proxy
encoder must be a different model family from the judge. Record which, and
why, in the experiment config.

**Ablation.** Base vs +feature-matching vs +ASR-CE, each scored on LCF and
on conventional metrics. This is the controlled experiment that supports
claim 3, and it is why leg 3 and leg 4 share a base checkpoint.

**Latency.** ~200–300 ms streaming budget (`docs/decisions.md`). Measured
algorithmic latency and RTF reported for every system, **and separately per
output modality** — the text reference condition pays for ASR decoding and
endpointing on top of extraction, and reporting one latency figure across
both modalities would hide that. Note the budget is currently a stated
assumption — see §8.

---

## 6. Timeline (2026-08-07 → 2026-11-05)

Hard freeze on new experiments: **2026-10-14.**

| Week | Dates | Work | Done when |
|------|-------|------|-----------|
| 1 | Aug 7–13 | **De-risk.** Secure HPC. Survey public streaming-TSE checkpoints (§4) and open-weight speech-to-speech judges. Estimate API cost. Pilot: 20 trials by hand through one live model to sanity-check the whole idea — **through both the audio and the text input**, since the same 20 trials answer both questions and the text path is a handful of extra API calls. | You have seen real judge responses and know roughly what a trial costs |
| 2 | Aug 14–20 | Constructed trial-set generation. Scoring implementation (LCF-WER, ICR, NRR) + conventional metrics. | Floor and ceiling measured on the constructed set — a real result, log it |
| 3 | Aug 21–27 | Judge harness: fixed prompt, k-repeats, variance, cost accounting, second (open-weight) judge. Prompt-sensitivity ablation. | Metric is stable enough that run-to-run spread is smaller than system differences |
| 4 | Aug 28–Sep 3 | **Leg 2 benchmark** on constructed set, including the text reference condition and its two text anchors. AMI download + REAL-T-style construction in parallel. | The divergence table exists — claim 2 stands or falls here |
| 5 | Sep 4–10 | AMI trial set finished, leg 2 extended to it. Training infra: data generation, YAML config, **checkpoint/resume proven across a session kill**. | A 1-epoch run completes, dies, resumes cleanly |
| 6–7 | Sep 11–24 | **Train leg-3 baseline.** Metric protocol write-up and supervisor review in parallel (no GPU). | Converged checkpoint; protocol signed off |
| 8 | Sep 25–Oct 1 | Score leg-3 model on both trial sets, both judges. Latency + RTF verification. | Our model is in the benchmark table |
| 9–10 | Oct 2–14 | **Leg 4:** proxy-objective fine-tune + ablation. | Ablation table, or an honest "cut for time" |
| 11–13 | Oct 15–Nov 5 | Write-up. Buffer. | Submitted |

Note weeks 1–4 produce a defensible result with zero GPU time. That is
deliberate.

### Cut list, in order

Record every cut in `docs/decisions.md`.

1. **ASR cross-entropy proxy** → feature-matching only. Costs one ablation row.
2. **Leg 4 entirely** → weeks 9–10 become writing buffer. The thesis becomes
   claims 1 and 2, which is still a complete thesis.
3. **Shrink leg 3**: train a deliberately smaller model and report it as
   such. A small baseline you fully understand beats a large one that never
   converged.
4. **Reduce judges to one** — keep the **open-weight** one, not the API.
   Reproducibility beats prestige.
5. **Never cut:** the metric definition, the constructed trial set, the
   floor/ceiling anchors, or the leg-2 divergence table.

The AMI leg sits between items 3 and 4 in priority: cut it only if the
alternative is not finishing, and if cut, say plainly in the thesis that
real-audio transfer is untested.

The **text reference condition** is not on the cut list in any useful sense —
it is an off-the-shelf ASR and an extra harness path, costing well under a
day, and it directly answers a question the spec asks. If it is somehow cut,
say in the thesis that the text output permitted by spec note 10 was
considered, argued against on latency grounds, and left unmeasured.

---

## 7. What to tell your supervisors

> The goal is to improve what a live speech-to-speech model understands, not
> how good the audio sounds — and those two can point in opposite
> directions. So the first contribution is a metric for the former, designed
> to be hard to game, with the live model held out as an independent judge.
> The second is a benchmark showing where conventional metrics mispredict
> it. Then I train a streaming extractor with differentiable proxies aligned
> to that objective — I can't backprop through Gemini Live, so the proxies
> are frozen encoders from a different model family, which also keeps the
> judge honest. I'm not replicating the challenge; I'm borrowing its
> data-construction methods and its lessons about metric gaming.

Get explicit sign-off on two things:

1. **Is the metric allowed to be the primary contribution**, with the model
   work as the second half rather than the centrepiece? Spec notes 1 and 8
   say yes; get it minuted.
2. **Is the ~200–300 ms latency budget acceptable**, and does anyone have
   evidence for what live models actually tolerate? (§8)
3. **Is audio-out the agreed build target**, with text-out measured as a
   reference condition rather than developed? Spec note 10 leans this way
   ("it would probably make more sense to have audio output") but stops short
   of deciding. Worth minuting, because building a text path later is a
   different project, not an extension of this one.

---

## 8. Open questions

- **The 200–300 ms latency budget is an assumption, not a result.** We have
  no evidence yet for the turn-taking tolerance of live speech-to-speech
  models. Find published evidence or measure it; until then present it as a
  stated assumption.
- **Which open-weight speech-to-speech judge.** Needs a survey in week 1,
  before the protocol freezes. The metric's reproducibility depends on it.
- **API budget** — unestimated. Must be a spreadsheet before week 2.
- **HPC access** — unresolved; materially changes what leg 4 can be.
- **Does the divergence actually exist?** The entire motivation rests on the
  supervisors' observation. Week 1's 20-trial pilot is the cheapest possible
  test of it. If it doesn't reproduce, re-scope immediately rather than
  building a harness for a phenomenon that isn't there.
- **Prompt sensitivity.** If judge scores move more with prompt wording than
  with system quality, the metric needs redesign. Week 3.
- **Does the text path beat the audio path?** Cheap to find out in week 1's
  pilot. If text wins by a wide margin the honest framing of the thesis
  shifts — the contribution becomes "here is how much artefact sensitivity
  costs the audio path, measured against a text bypass" rather than "here is
  a better audio path." That is still a good thesis, but it is better to know
  in week 1 than week 8. It does not change the build target (see §5).
- **Does the judge expose a text input with comparable latency and pricing?**
  Assumed, unverified. Check in week 1 alongside the API cost estimate — the
  text row's cost model may differ enough to matter for trial-set sizing.
