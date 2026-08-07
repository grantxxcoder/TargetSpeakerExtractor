# Research Plan & Route Recommendation

**Written:** 2026-08-06
**Horizon:** 3 months to final submission (~2026-11-05), *including* write-up
**Status:** proposal — supersedes nothing; log the chosen route in `docs/decisions.md` once agreed with supervisors

---

## 0. The short version

**Do not build the winning challenge system. Do not do a broad multi-paper
replication. Do both of the following instead:**

1. **Reimplement exactly one baseline** — the causal BSRNN + TF-Map online
   baseline — and use it as an *instrument*, not as a result.
2. **Make the live-model metric the thesis.** It is the only genuinely novel
   thing in the brief, it is the thing your supervisor said matters most
   (meeting note 1), and it is almost free in GPU-hours.

Then, if and only if time remains, add a third leg: an **accuracy-vs-compute
Pareto study under the 100 ms latency budget**, which is a real and documented
hole in the literature.

The rest of this document argues why, rules out the alternatives, names the
specific papers, and lays out a week-by-week plan with a cut list.

---

## 1. Three constraints that decide everything

Before comparing routes, be honest about the box you are in. Every
recommendation below falls out of these three facts.

### 1.1 You have no evaluation data, so you cannot "replicate" anything

`docs/decisions.md` already records this: challenge registration closed before
we could register, so we have no DEV/EVAL audio and no baseline checkpoints.

This has a consequence that is easy to miss and fatal if missed late:

> **A replication you cannot validate is not a replication — it is a
> reimplementation.**

If you frame the thesis as "I replicated the REAL-TSE online baselines," an
examiner will ask what number you matched. The honest answer is *none, because
the eval audio differs*. That is a weak position to defend for a whole thesis,
and it is entirely avoidable by framing the reimplementation as infrastructure
supporting an evaluation-methodology contribution.

Frame it as: **"a new evaluation methodology for real-time TSE, instrumented
with a faithfully reimplemented online baseline."** The baseline then only has
to be *credible* and *correctly described*, not *numerically identical* — which
is achievable, whereas the alternative is not.

### 1.2 Compute is the binding constraint, and it is tighter than it looks

Confirmed available: Kaggle GPUs now, university HPC probably (not yet secured).
No local GPU.

Kaggle in practice means: ~30 GPU-hours/week, hard 12-hour session limit,
single P100 or dual T4, and a working-disk quota that Libri2Mix will strain.
Every training run must checkpoint and resume from step one of the project —
retrofitting that after a session dies at hour 11 costs you a week.

Rough order-of-magnitude on a single P100: a 27 M-parameter causal BSRNN on
Libri2Mix-100 at a normal recipe length is on the order of **40–70 wall-clock
hours** for one run. That is two to three weeks of your Kaggle quota, for *one*
model, assuming nothing breaks.

**Budget realistically for two to four full training runs across the entire
project.** Not ten. Any plan that implies more than four is fiction.

### 1.3 Ten technical weeks, not thirteen

Three months including write-up means roughly **10 weeks of technical work and
3 weeks of writing**, and the writing weeks are not compressible — they are what
you are actually graded on. Treat 2026-10-14 as the hard freeze date for new
experiments.

---

## 2. The routes you asked about, assessed

### Route A — "Just recreate the papers." Which papers? — **Rejected as the primary plan**

Reimplementing 3–4 architectures (BSRNN, TF-Map, USEF-TSE, SA-Mamba) means 3–4
training runs, which §1.2 says you cannot afford, and it produces **zero
novelty**. Worse, per §1.1, you cannot verify any of them against their
published numbers, so you would spend your entire compute budget generating
results whose only honest caption is "not comparable to the source paper."

Breadth of *reading* is required and you already have it. Breadth of
*reimplementation* is the single worst use of your GPU quota.

There is one exception, and it is the plan in §3: reimplement **one** baseline,
deeply, because you need a working online TSE system as an instrument regardless.

### Route B — "Implement the winning solution (CARTSE) + one metric." — **Rejected, and it is the worst fit of the three**

This is the intuitive choice and it is the trap. Your own literature review
(`review_synthesis.md`, finding 2 and entry #2) already contains the reason:

> the top Track 1 entries were nearly all built on BSRNN-style backbones
> architecturally close to the provided baseline, and the large score gains came
> from data simulation, real-data adaptation, pseudo-label generation and
> filtering, multi-objective loss design, and latency control — not from new
> architectures.

So "implement the winner" does not mean implementing a clever architecture. It
means reproducing **a data pipeline**, and specifically:

- an entire *second* system (their offline Track-2 teacher) to generate
  pseudo-labels — i.e. double the training cost before you start;
- pseudo-labelling ~38 h of real audio and filtering it on four automatic
  metrics;
- a two-stage fine-tune with five auxiliary losses, including DNSMOS-in-the-loop
  and a multi-layer Zipformer feature-matching loss.

That is comfortably a 500+ GPU-hour project and it needs the real conversational
training data and the eval set to steer against. You have neither. On Kaggle it
is not a stretch goal, it is impossible — and if you attempt it you will spend
ten weeks with nothing finished.

There is also a subtler reason to avoid it: CARTSE **explicitly optimises against
DNSMOS**, and the organisers subsequently found DNSMOS-OVRL had been so heavily
over-optimised that its correlation with human MOS on Track 1 was approximately
zero. Reproducing a metric-gaming pipeline is a strange centrepiece for a thesis
whose novel contribution is a *better metric*.

**What to take from CARTSE instead:** two components that are cheap, need no
teacher, and are separable —

- **target-absent training** (~38 % of training examples have no target present;
  split loss: masked SI-SDR when present, push-to-silence when absent), and
- **channel-gap enrollment augmentation** (random RMS-preserving EQ curves on the
  enrollment so conditioning learns device invariance).

Both are a day of work each, both address goals in your brief (interruptions;
microphone mismatch), and both are legitimately citable as "adopted from Li &
Seki (2026)." That is the right dose of CARTSE for this project.

### Route C — "Add on one metric to a replication." — **Right ingredients, wrong emphasis**

This is close to correct, but the word *add-on* is backwards, and getting the
emphasis right is the single highest-leverage decision in this document.

The metric is not a garnish on a replication. It is:

- **the only novel contribution in the brief** — the spec says the challenge
  does not measure this and the project should define how;
- **explicitly prioritised by your supervisor** — meeting note 1: *"the actual
  score values from my defined metric do not matter — the metric itself matters
  more"*;
- **nearly compute-free** — it is inference and API calls over audio you already
  have, so it does not compete with training for your Kaggle quota;
- **the de-risking leg** — if every training run fails, you still have a thesis.
  If you invert the priorities and training fails, you have nothing.

So: same ingredients as Route C, inverted weighting. That is Route D.

### Route D — Metric-first, baseline-as-instrument — **Recommended**

Three legs, in strict priority order, sharing one evaluation pipeline so the
infrastructure cost is paid once:

| # | Leg | Brief deliverable | GPU cost | Drop if behind? |
|---|-----|-------------------|----------|-----------------|
| 1 | Eval pipeline + AMI-based REAL-T-style test set | "replicate the evaluation pipeline" | ~0 | **Never** |
| 2 | Live-model utility metric + benchmark study | "define the live-model measurement" | ~0 | **Never** |
| 3 | One reimplemented causal BSRNN+TF-Map baseline | "replicate the online-track baselines" | high | No, but shrink it |
| 4 | Accuracy-vs-compute Pareto under 100 ms latency | "explore a new architecture" | high | **Yes — cut this first** |

Legs 1 and 2 are CPU/API work and are the thesis. Leg 3 is the instrument that
makes leg 2 interesting (you need *something* to compare against unprocessed
audio). Leg 4 is the stretch.

Note what leg 4 is *not*: it is not "a fundamentally new architecture." In ten
weeks on Kaggle, that is not on the table, and claiming it would be the kind of
overreach that gets punished in a viva. What leg 4 *is* — per your review's own
closing section — is the documented gap that **no REAL-TSE entry reported a
parameter or MAC budget anywhere near on-device class** (smallest online system:
15.89 M params / 30 GMAC/s). Since you settled on a paper compute budget rather
than real hardware, you can populate that gap with a scaling study, cheaply, and
frame it honestly as an efficiency characterisation rather than a new model.

---

## 3. Which specific papers, and what to do with each

Direct answer to your question. Four tiers, by *action*, not by importance.

**Tier 1 — reimplement in code (this is your only build):**

- **#4 BSRNN** (Luo & Yu, TASLP 2023) — the backbone.
- **#3 Multi-Level Speaker Representation / TF-Map** (Zhang et al., ICASSP 2025)
  — the conditioning.

These are one codebase, not two: the causal `BSRNN_TFMAP_CAUSAL` baseline is the
intersection of both papers, and it is the variant your review notes actually
*beats the non-causal embedding baseline on TER* — a genuinely useful result for
an online-track project. Read both papers against the `wesep-real-tse` source.

**Tier 2 — reproduce methodologically, no model training:**

- **#6 REAL-T** (Li et al., Interspeech 2025) — the trial-construction pipeline.
  This is your test set and, per your review, "probably the single most useful
  piece of infrastructure you can build in your first month." Agreed.
- **#1 REAL-TSE overview** (Wang et al., arXiv:2607.15198) — the scoring
  pipeline, the metric definitions, the latency verification protocol.

**Tier 3 — read deeply, borrow two components, do not replicate:**

- **#2 CARTSE** (Li & Seki, 2026) — take target-absent simulation and
  channel-gap EQ augmentation. Leave the pseudo-label teacher pipeline. See §2
  Route B.

**Tier 4 — prior art for the metric (read, cite, do not implement):**

- **PS4** (Ning et al., arXiv:2607.08111) — the closest published prior art to
  "optimise TSE for what a downstream model needs." Your review is right that
  this is the most important Tier-2 paper for your contribution; read it in the
  first fortnight, not later.
- **Ma et al., arXiv:2501.14477** — joint TSE + target-speaker ASR; directly
  addresses the "more transcribable yet worse for a live model" phenomenon in
  your spec.
- **Delcroix et al., Interspeech 2022, "Listen only to me!"** — target-absent
  behaviour and false alarms; underpins both the interruption goal and the
  distractor-leakage half of your metric.

**Background chapter only:** #5 Žmolíková survey. Do not cite as current SOTA —
your review already flags that it predates everything that matters here.

**Explicitly out of scope — name them in the thesis as future work:**
#10 SA-Mamba, #9 USEF-TSE, #7 causal TF-GridNet, #8 TF-MLPNet as a *build*.
Each is a defensible project on its own and none fits in the remaining budget.
Saying so deliberately is much stronger than leaving them unmentioned.

**Housekeeping:** the spec cites TF-MLPNet as Interspeech 2025; per your review
it is the 6th Clarity Workshop (Clarity 2025). Fix before the proposal is marked.

---

## 4. The metric — a concrete proposal

This is the contribution, so it deserves a real design rather than "define a
metric." Treat the following as a starting draft to argue with, not a spec.

### 4.1 What it must measure

The spec's observation: *traditional TSE can improve transcription accuracy
while making the audio harder for live speech-to-speech models to understand.*
So the metric must be sensitive to something WER is blind to. That rules out any
metric computed on a transcript.

### 4.2 Proposed design: task-completion, not signal quality

Measure whether a **live speech-to-speech model does the right thing**, not
whether the audio sounds good.

Construct trials where the target speaker utters an instruction with a
**verifiable answer**, while an interfering speaker utters a **conflicting
distractor instruction**. Feed the audio to the live model. Score:

- **TIC — Target Instruction Compliance:** fraction of trials where the response
  correctly follows the target's instruction.
- **DLR — Distractor Leakage Rate:** fraction where it follows the interferer's
  instruction instead. *This is the number a WER-based metric structurally
  cannot see, and the one your supervisors' observation predicts will behave
  strangely under TSE.*
- **NRR — Non-Response Rate:** "sorry, I didn't catch that" / refusal / silence.

Headline score = TIC, reported always alongside DLR and NRR. Baselines:
unprocessed mixture (floor), and target-speaker-only clean audio (ceiling).

### 4.3 Why this design resists gaming — build this in from day one

Your review's caution is the strongest argument available for the whole
contribution: DNSMOS-OVRL was gamed hard enough (in one case with adversarial
waveform perturbations) that its human-MOS correlation on Track 1 was ~0.003,
and the organisers had to swap metrics after the fact. Design against that
explicitly, and say so in the thesis:

- **Semantic, end-to-end, and discrete.** The score depends on whether a model
  *did the right thing*, not on a differentiable perceptual score. There is no
  smooth signal to hill-climb with waveform perturbations.
- **Two-sided.** Suppressing everything scores well on DLR but destroys TIC.
  Passing everything through scores well on TIC but blows up DLR. Only genuine
  extraction moves both.
- **Held-out instruction bank.** Publish the protocol and a public split; keep a
  private split for scoring. Prevents overfitting to specific prompts.
- **Report the ceiling.** Clean-target TIC bounds what is achievable and makes
  an implausible score obvious.

### 4.4 Two methodological requirements — do not skip these

- **Pin and date every model.** Closed live models change silently. Record exact
  model IDs and run dates for every number, and treat cross-date comparisons as
  invalid unless re-run.
- **Include at least one open-weight speech-to-speech model** alongside the
  closed API. If the benchmark can only be reproduced by someone paying for an
  API whose behaviour changes monthly, it is not a scientific contribution. This
  is the single most important design decision for the metric's shelf life.

**Budget:** API calls cost real money. Estimate cost per trial × number of trials
× number of systems × number of live models *before* week 7, and size the trial
set to fit. Cheaper to discover this on a spreadsheet than at 2 a.m.

---

## 5. Data plan

**Training:** Libri2Mix-100 + WHAM!, matching what the baselines used.
Generation is a multi-hour CPU job producing a large corpus — you have ~146 GB
free. Generate **only** the sample rate and mode you need (16 kHz, one of
min/max — decide and record it in `decisions.md`); generating everything will
exhaust the disk.

**Evaluation:** AMI first, per `decisions.md`. One point worth exploiting:

> AMI has per-speaker **headset (IHM)** microphones alongside the distant array
> (SDM). So unlike REAL-T — which has no clean target and is therefore forced
> into reference-free metrics — you can use SDM/array as the mixture and the
> corresponding IHM channel as an **approximate** clean target.

That lets you additionally report intrusive metrics (SI-SDR etc.) that the
official protocol cannot compute. That is a small genuine methodological
advantage and worth a paragraph in the thesis. **Caveat it properly** — IHM has
cross-talk bleed and a completely different channel response from the array, so
it is an approximate reference, not ground truth. Per CLAUDE.md this caveat goes
in a code comment *and* in `docs/decisions.md`.

Download only what you need: AMI in full is hundreds of GB.

**Still worth chasing:** email the organisers again about academic access to
DEV/EVAL and the baseline checkpoints, and separately check whether any
`wesep-real-tse` checkpoint is public (HuggingFace / the WeSep repo). **A
released checkpoint would change this entire plan** — it would remove leg 3's
training cost outright and free those weeks for legs 2 and 4. It is a ten-minute
check with a very large payoff; do it in week 1.

---

## 6. Timeline (2026-08-06 → 2026-11-05)

Hard freeze on new experiments: **2026-10-14.**

| Week | Dates | Work | Done when |
|------|-------|------|-----------|
| 1 | Aug 6–12 | **De-risk.** Secure HPC. Check for public baseline checkpoints (§5). Re-email organisers. Official scoring repo running on dummy audio. Start AMI download. Fix TF-MLPNet citation. | Scoring pipeline produces all four metrics on a toy file |
| 2 | Aug 13–19 | REAL-T-style trial construction from AMI. Score **unprocessed mixtures** through the pipeline. | A trial set exists + the "do nothing" floor is measured on every metric — this is a real result, log it |
| 3 | Aug 20–26 | Training infra: `wesep-real-tse` running, Libri2Mix-100 + WHAM! generated, YAML config committed, **checkpoint/resume proven across a session kill** | A 1-epoch run completes, dies, and resumes cleanly |
| 4–5 | Aug 27–Sep 9 | Train causal BSRNN+TF-Map. **In parallel (no GPU):** design the metric, write the protocol doc, build the instruction bank | Checkpoint exists; metric protocol reviewed by supervisors |
| 6 | Sep 10–16 | Evaluate baseline on AMI set. Run the official latency-verification script. | First full comparison table: unprocessed vs baseline, all four metrics + verified latency |
| 7–8 | Sep 17–30 | **Metric benchmark.** Unprocessed / baseline / ≥1 public TSE model × ≥2 live models (1 API + 1 open-weight). Gaming-resistance analysis. | The thesis's central result table |
| 9–10 | Oct 1–14 | **Stretch:** Pareto study — 2–3 shrunk baseline variants, report params/MACs/RTF/latency vs metrics | Accuracy-vs-compute curve, or an honest "cut for time" |
| 11–13 | Oct 15–Nov 5 | Write-up. Buffer. | Submitted |

### Cut list, in order

If you are behind at the week-6 checkpoint, cut in this order and record each
cut in `decisions.md`:

1. **Leg 4 entirely** (weeks 9–10 become writing buffer). Costs nothing
   essential — it was always the stretch.
2. **Shrink the metric benchmark**: one live model instead of two, fewer trials.
   Keep the open-weight one, not the API one — reproducibility beats prestige.
3. **Shrink the baseline**: train a deliberately smaller BSRNN and report it as
   such. A well-documented small baseline you understand beats a large one that
   never finished.
4. **Never cut:** the eval pipeline, the AMI test set, the unprocessed-mixture
   floor, or the metric *definition*. Those four are the thesis.

Note that item 3 and leg 4 point the same direction — if training goes badly,
the small-model path is both the fallback *and* the contribution. That is a
useful property of this plan: its failure mode degrades into a different valid
result rather than into nothing.

---

## 7. What to tell your supervisors

A one-paragraph pitch to take to the next meeting:

> The challenge is over and the organisers' own conclusion is that the winning
> online systems were baseline-like architectures lifted by data pipelines, not
> new architectures. Reproducing the winner therefore means reproducing a
> pipeline that needs an offline teacher, real conversational training data and
> the eval set to steer against — none of which we have, and which does not fit
> the compute budget. So: I will reimplement one online baseline as an
> instrument, build a REAL-T-style eval set from AMI, and spend the bulk of the
> project on the live-model measurement, which is the part of the brief nobody
> in the challenge addressed. If it goes well, I will add an
> accuracy-versus-compute characterisation under the 100 ms latency budget,
> since no challenge entry reported a compute budget anywhere near on-device
> class.

Two things to get explicit sign-off on, because they change the plan:

1. **Is a reimplemented-but-unvalidated baseline acceptable** for the
   "replicate the baselines" deliverable, given the eval data is unavailable?
   (§1.1 — this is the biggest framing risk in the project.)
2. **Is the metric allowed to be the primary contribution**, with architecture
   work demoted to a stretch goal? Meeting notes 1 and 7 suggest yes, but get it
   said out loud and minuted.

---

## 8. Open questions

- HPC access — unresolved, and it materially changes what leg 4 can be.
- API budget for the live-model benchmark — unestimated (§4.4).
- Which open-weight speech-to-speech model to use as the reproducible anchor —
  needs a survey in week 4, before the protocol is frozen.
- Whether any `wesep-real-tse` checkpoint is public — a week-1 check that could
  reshape the whole schedule (§5).
