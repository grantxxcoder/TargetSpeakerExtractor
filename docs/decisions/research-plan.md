# Research Plan

**Written:** 2026-08-06 · **Rewritten:** 2026-08-07 for the re-scoped objective · **§2 and §6 re-ordered:** 2026-08-07 (evening) to a build-first sequence
**Horizon:** ~3 months to final submission (2026-11-05), *including* write-up
**Supersedes:** the metric-first-but-baseline-replication plan of 2026-08-06. Decisions are logged in `docs/decisions/decisions-m0.md`.

> **Schedule authority is `docs/decisions/milestones.md`.** This file holds the reasoning; the milestone file holds dates and checklists. §2 and §6 now agree with it. Sections 3–5 are organised by *workstream*, not schedule; §2's table gives the order.

---

## 0. The thesis in one paragraph

Conventional target speaker extraction is optimised and scored on how good the separated audio *is*. A voice assistant needs what the downstream live speech-to-speech model *understands*, and our supervisors have observed the two diverging. This project defines a metric for the second, uses it to characterise the divergence across existing systems, then trains a streaming TSE model with differentiable proxies aligned to it. The live model is held out as an independent judge throughout.

Three claims, in decreasing order of confidence:

1. **A metric.** Live-model content fidelity is definable, gaming-resistant and measurable. (`docs/data/metric-definitions.md`)
2. **A finding.** Signal quality, perceptual quality and offline WER do not predict live-model content fidelity. Conventional TSE can improve the former while degrading the latter.
3. **A model.** Training against differentiable proxies aligned to the objective improves live-model content fidelity over a conventionally-trained streaming baseline.

Claim 1 cannot be cut. Claim 2 needs no training. Claim 3 is where the compute goes, and it can fail.

---

## 1. Three constraints that shape everything

### 1.1 The judge is a black box

A live speech-to-speech API cannot be backpropagated through, is stochastic, costs money per call and changes silently over time. Therefore:

- Training uses **differentiable proxies**, never the judge.
- The proxy must be a **different model family** from the judge, or the benchmark measures overfitting to one evaluator. This is the most important methodological rule in the project.
- Every number carries a model ID and a date.
- Trial-set size is bounded by **API budget**, not data availability.

### 1.2 Compute is the binding constraint

Confirmed: Kaggle GPUs. University HPC probable, not yet secured. No local GPU.

Kaggle in practice: ~30 GPU-hours/week, hard 12-hour session limit, single P100 or dual T4, constrained working disk. Every training run must checkpoint and resume **from day one**; retrofitting it after a session dies at hour 11 costs a week.

One run of a causal BSRNN-class model on LibriSpeech-derived data is **40–70 wall-clock hours** on a P100. Budget for **one base training run plus two to three fine-tunes**, not ten.

Claim 3 is therefore *train once, then fine-tune variants*: a fraction of the cost, and the proxy-objective comparison becomes a controlled A/B, not two independent runs.

### 1.3 Ten technical weeks, not thirteen

Roughly **10 weeks technical, 3 weeks writing**. The writing weeks cannot be compressed; they are what gets graded. **2026-10-14 is the hard freeze on new experiments.**

---

## 2. The six workstreams, in build order

Ordering changed 2026-08-07 (evening) from risk-first to **build-first**, at the researcher's direction. All workstreams share one evaluation harness.

| Order | Leg | Claim | GPU cost | Cut if behind? |
|---|-----|-------|----------|----------------|
| 1 | Data construction (training + eval, one generator) | — | ~0 | **Never** |
| 2 | Streaming TSE baseline, conventionally trained | 3 | high | No, shrink it |
| 3 | Conventional evaluation of that baseline | — | ~0 | **Never** |
| 4 | Metric definition + scoring harness + trial sets | 1 | ~0 | **Never** |
| 5 | Proxy-objective fine-tune (the "second model") | 3 | medium | **Cut first** |
| 6 | Benchmark / divergence table | 2 | ~0 | **Never** |

### Why this order, and what it costs

**For:** you cannot design a good instrument for a phenomenon you have not yet heard. Building the extractor first means the metric is frozen *after* someone has listened to real masked-extraction output and characterised its artefacts. The claim-2 divergence is then measured within a system we built and fully understand, not across off-the-shelf models whose training data and objectives we cannot inspect, which strengthens claim 2.

**Against, accepted with mitigations:** the previous ordering put the metric and benchmark first so a total training failure still left a submittable thesis. Both now depend on a training run converging on Kaggle. Two mitigations, both in `docs/decisions/milestones.md`:

1. **The public-checkpoint survey moves to week 1.** If a causal BSRNN + TF-Map checkpoint is public, the baseline becomes a fine-tune and its failure mode largely disappears. Already a week-1 item (§4), it is now the most valuable hour in the schedule.
2. **The metric is designed on paper during the baseline's training weeks**, while the GPU is busy: designed by end of week 6, implemented in week 8.

The second model is still fine-tuned *from* the baseline, and the baseline still gives the benchmark a system we control.

### What the second model is not

It is not "a fundamentally new architecture." That is not possible in ten weeks on Kaggle, and claiming it invites a bad viva. It is a controlled comparison: same architecture, data and base checkpoint, different training objective. A clean, publishable experiment, directly aligned with spec note 8.

---

## 3. The metric and its harness

Full specification in `docs/data/metric-definitions.md`. What has to be built:

**Trial construction: constructed set (primary).** LibriSpeech-derived mixtures with a target, ≥1 interferer, real noise (WHAM!-style) and reverberation (WHAMR!-style RIRs). This gives exact verbatim ground truth for target and interferer, a true clean-target ceiling, and controllable overlap ratio, SNR and device mismatch. Speaker-disjoint splits.

**Trial construction: AMI set (secondary).** REAL-T-style construction (Li et al., Interspeech 2025): existing diarisation annotations give naturally overlapping segments as mixtures, and ≥5 s non-overlapping same-speaker segments as enrollment. The distant mic is the mixture and the headset (IHM) the **approximate** ceiling. This is the real-audio transfer check.

**Judge harness.** Fixed prompt, fixed response-transcription ASR, pinned model IDs, k≥3 repeats per trial, cost accounting. At least one open-weight judge alongside the closed API, for reproducibility. **Two input paths to the judge (audio and text), with the modality recorded on every result**, since spec note 10 permits either output. Build both from the start, as retrofitting a second path costs more.

**Conventional metrics alongside.** SI-SDR, DNSMOS-P808, offline ASR WER (amended 2026-09-01 to P.835, see metric-definitions.md §6), computed on the same trials so the benchmark study can show where they diverge from LCF.

---

## 4. The benchmark study

This is claim 2, the highest value-per-GPU-hour work in the project. No training: run existing systems over the trial sets and score them.

Systems to include:
- Unprocessed mixture (floor)
- Clean target (ceiling)
- ≥2 off-the-shelf pretrained TSE models, ideally one streaming and one offline, to separate the effect of causality from that of extraction
- A conventional speech enhancer, as a "denoising without extraction" control
- **Text reference condition**: extractor → off-the-shelf streaming ASR → text → judge, with its own text floor (ASR on the raw mixture) and text ceiling (ground-truth text). It measures the text output spec note 10 permits without building a second system: no training, one extra harness path, and it shows how much content is recoverable at all. See `docs/decisions/decisions-m0.md` (2026-08-07 output-modality decision) and `docs/data/metric-definitions.md` §3.5 for why it is a reference condition, not a rival build target, and why it is *not* an upper bound.
- Later: our own baseline and second model

Judges: ≥1 closed live API + ≥1 open-weight speech-to-speech model.

**Target result:** a table where the ranking by SI-SDR / DNSMOS / offline-WER differs from the ranking by LCF-WER. If so, the metric is justified and the thesis has its central finding. If conventional metrics predict live-model fidelity perfectly well, that is *also* a publishable negative result, better found in week 5 than week 11.

**Week-1 check with a large payoff:** find which pretrained streaming TSE checkpoints are public (WeSep/WeSep-family releases, HuggingFace, USEF-TSE's released code). A usable one strengthens the benchmark study immediately and turns the baseline from "train from scratch" into "fine-tune from a released model". Ten minutes of searching could save weeks.

---

## 5. The model

**Output modality: audio.** Settled (`docs/decisions/decisions-m0.md`, 2026-08-07). Text output is measured as a benchmark reference condition (§4), not built. The model work is unchanged: the proxy losses use the output waveform, and the text row attaches an existing ASR to the same checkpoint at eval time.

**Architecture.** Causal BSRNN-family extractor with TF-Map + speaker-embedding conditioning. It is the best-understood strong streaming TSE design, challenge evidence shows the causal TF-Map variant is unusually strong online, and reference implementations exist. We borrow it as a well-characterised instrument, not to replicate it as a result (see `docs/decisions/decisions-m0.md`).

Cite: Luo & Yu (TASLP 2023) for BSRNN; Zhang et al. (ICASSP 2025) for TF-Map / multi-level speaker representation.

**Baseline training (conventional).** SI-SDR + multi-resolution STFT, with two cheap components borrowed from CARTSE (Li & Seki, 2026):
- **target-absent training**: a substantial fraction of examples with no target, with a split loss (masked SI-SDR when present, push-to-silence when absent). Addresses false alarms and the interruption goal.
- **channel-gap enrollment augmentation**: random RMS-preserving EQ curves on the enrollment, so conditioning learns device invariance. Addresses the spec's microphone-mismatch concern.

Each is ~a day of work and a separable, citable component.

**Second-model training (proxy-aligned).** Fine-tune the baseline checkpoint with added differentiable proxies:

1. **Frozen-encoder feature matching** (primary proxy): match intermediate activations of a frozen ASR/SSL encoder between output and clean target. Cheaper than full ASR cross-entropy; precedent in CARTSE's Zipformer feature-matching loss and PS4's proxy objectives.
2. **ASR cross-entropy** (stretch): full differentiable ASR in the loop. Roughly doubles memory and step time; only attempt if the second model is on schedule.
3. Speaker-similarity and target-activity terms as auxiliaries.

**Proxy family (§1.1, easy to break by accident):** record which family the proxy encoder is from, and why, in the experiment config.

**Ablation.** Base vs +feature-matching vs +ASR-CE, each scored on LCF and conventional metrics. This is the controlled experiment behind claim 3, hence the shared base checkpoint.

**Latency.** ~200–300 ms streaming budget (`docs/decisions/decisions-m0.md`). Algorithmic latency and RTF are reported for every system **and separately per output modality**, since the text reference condition also pays for ASR decoding and endpointing. The budget is a stated assumption (§8).

---

## 6. Timeline (2026-08-07 → 2026-11-05)

Hard freeze on new experiments: **2026-10-14.**

Week-by-week dates and acceptance criteria are in `docs/decisions/milestones.md`, which is authoritative for scheduling.

**Changed risk profile.** The previous ordering gave a defensible result in weeks 1–4 with zero GPU time. Now the first result *part of the thesis argument* arrives in week 7, and everything from week 8 depends on a training run converging (mitigations in §2). Weeks 9 and 10 have no slack: if M5 slips it is cut, not delayed, because M6 must still run before the freeze.

### Cut list, in order

Record every cut in `docs/decisions/decisions-m0.md`.

1. **ASR cross-entropy proxy**, leaving feature-matching only. Costs one ablation row.
2. **The second model entirely**, so weeks 9–10 become writing buffer. The thesis becomes claims 1 and 2, still a complete thesis.
3. **Shrink the baseline**: train a deliberately smaller model and report it as such. A small baseline you fully understand beats a large one that never converged.
4. **Reduce judges to one**, keeping the **open-weight** one, not the API. Reproducibility beats prestige.
5. **Never cut:** the metric definition, the constructed trial set, the floor/ceiling anchors, or the benchmark's divergence table.

The AMI leg sits between items 3 and 4: cut it only if the alternative is not finishing, and then state in the thesis that real-audio transfer is untested.

The **text reference condition** is effectively not on the cut list: an off-the-shelf ASR and an extra harness path, well under a day, answering a question the spec asks. If it is cut, say in the thesis that the text output permitted by spec note 10 was considered, argued against on latency grounds, and left unmeasured.

---

*§7 ("What to tell your supervisors") removed 2026-08-12, superseded by `docs/meetings/meetings.md`. §8 keeps its number so existing references hold.*

## 8. Open questions

- **The 200–300 ms latency budget is an assumption, not a result.** There is no evidence yet for the turn-taking tolerance of live speech-to-speech models. Find published evidence or measure it; until then, present it as an assumption.
- **Which open-weight speech-to-speech judge.** Survey in week 1, before the protocol freezes; the metric's reproducibility depends on it.
- **API budget**: unestimated. Must be a spreadsheet before week 2.
- **HPC access**: unresolved, and materially changes what the second model can be.
- **Does the divergence actually exist?** The motivation rests entirely on the supervisors' observation. Week 1's 20-trial pilot is the cheapest test. If it does not reproduce, re-scope immediately.
- **Prompt sensitivity.** If judge scores move more with prompt wording than with system quality, the metric needs redesign. Week 3.
- **Does the text path beat the audio path?** Cheap to find out in week 1's pilot. If text wins by a wide margin, the thesis framing shifts: the contribution becomes "here is how much artefact sensitivity costs the audio path, measured against a text bypass" rather than "here is a better audio path." Still a good thesis, but better known in week 1 than week 8. It does not change the build target (see §5).
- **Does the judge expose a text input with comparable latency and pricing?** Assumed, unverified. Check in week 1 with the API cost estimate; the text row's cost model may differ enough to affect trial-set sizing.
