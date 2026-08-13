# Milestones

**Rewritten 2026-08-07** to a build-first ordering: data → baseline model →
evaluate it conventionally → define the metric → second model → compare on
the metric → report.

Submission 2026-11-05; hard freeze on new experiments **2026-10-14**.

Each milestone names the artefact that proves it is done. "Reviewed by
supervisors" is not a milestone — a thing that exists is.

> **Supersedes** the metric-first milestone set of 2026-08-07 (morning).
> This file is authoritative for scheduling. `docs/decisions/research-plan.md`
> was reconciled with it on 2026-08-12: §6 points here for dates, and its §2
> build-order table agrees with this file.

---

## How this ordering works

The spine is sequential, but two things run alongside it rather than after it,
because they are not GPU-bound:

- **The metric definition and harness (M3)** is design, protocol and CPU/API
  work. It is drafted during M2's training weeks, when you are waiting on
  Kaggle sessions and have no GPU work to do. It is *finished* after the
  conventional evaluation, so it can be informed by what the baseline's output
  actually sounds like — which is the main argument for this ordering over
  metric-first.
- **The AMI trial set** is download-and-construct work with no GPU cost. Start
  it whenever there is a gap.

Everything else is a genuine dependency chain: no data → no model → nothing to
evaluate → nothing to compare.

---

## M0 — Data exists · target Aug 20 (weeks 1–2)

Both training and evaluation draw from the same construction code, so this is
built once.

**Status 2026-08-13.** Manifests exist for all six splits and have been audited
(`src/exploratory/data_setup.ipynb`). **No audio exists yet**, but A1 is now
decided (full reverberant reference), so the renderer is unblocked. The audit found
two shortcuts serious enough to require regenerating the manifests before any audio
is made.

Done:
- [X] ~~Speaker-disjoint train / val / eval splits~~
- [X] ~~Seed set and logged; generation config in `experiments/configs/`~~
- [X] ~~Manifest generator: speakers, timing, levels, rooms, noise, enrollment~~
- [X] ~~Enrollment segments ≥5 s, from a different recording than the mixture~~
- [X] ~~Manifest audit §1–§7: timing, levels, rooms, enrollment, noise, absent trials~~

Blocked on a decision:
- [X] ~~**A1 — reference signal.** Decided 2026-08-13: full reverberant. Supervisor
      sign-off outstanding but the renderer no longer waits on it~~
- [ ] **B12 — generator controllability.** Decide before the rebuild, or the
      rebuild happens twice.

Manifest rebuild — one pass covering all of:
- [ ] **B9** — add `target_only_fraction`; let `target_activity_ratio` vary.
      Silent-target trials are currently detectable at AUC 1.000 without
      listening, and a target speaking uninterrupted never occurs
- [ ] **B10** — book-preferred, chapter-fallback enrollment, recording which rule
      fired per trial. 60 % of speakers can never be a present target today
- [ ] **B4** — silent-target trials in the eval splits (fraction still open)
- [ ] Re-run the §7.5 leak scoreboard afterwards and show the AUCs dropped

Still unimplemented from `data-construction-parameters.md`:
- [ ] **`noise_speech_rejection`** — the docs call it critical. Speech in the
      noise bed enters as an unlabelled third talker and the metric scores those
      words as hallucination. Needs audio, so it lands in the renderer
- [ ] `length_mode`

Renderer (unblocked 2026-08-13):
- [ ] Manifest row → audio: RIRs, levels at BS.1770, noise wrap, clip ceiling
- [ ] Five stems per trial: mixture, clean target, enrollment, both texts
- [ ] Clean target = target × its own RIR, no interferer, no noise (A1). Record the
      RIR per trial so the direct+early ablation stem can be rendered later without
      re-drawing rooms
- [ ] Tail padding past the last speech by at least the room's decay (A5) — now
      mandatory, since the reference contains the tail
- [ ] Exact verbatim ground-truth text for **both** target and interferer
      (`d` is required by `docs/data/metric-definitions.md` §2 — without it ICR is
      not computable, and regenerating later to add it costs a week)
- [ ] Guards: stem <400 ms breaks BS.1770; assert on a silent stem
- [ ] Transcripts cut to match any audio truncation

Notebook and verification:
- [ ] §7.5 — leak scoreboard, **before** the rebuild so it is a before/after
- [ ] §8 — `enrollment_eq` rate vs config, `same_gender` rate vs config
- [ ] **EDA per parameter** — plot each parameter's realised distribution against
      the intended one (raised 2026-08-12, needed to verify B12)
- [ ] Revise §2, §7 and the final health checks after the rebuild; §3–§6 survive

Pilot calibration, before freezing any range:
- [ ] Listen to 40 trials with transcripts in hand
- [ ] Floor and ceiling WER measured; aim for a 60–80 % floor
- [ ] Confirm conditions separate — bin by SIR and by overlap

Housekeeping:
- [ ] `requirements.txt` / `pyproject.toml` — none exists, and `pyloudnorm` is
      now a real dependency

**Proof:** a generated set on disk with a manifest, plus a config + commit hash
+ seed in `experiments/results/`.

**Also this fortnight, because it changes everything downstream:**
- [ ] Survey public streaming-TSE checkpoints (WeSep family, HuggingFace). A
      usable causal BSRNN + TF-Map checkpoint turns M2 from a 40–70 h training
      run into a fine-tune. Ten minutes of searching, potentially weeks saved.
- [ ] HPC access resolved either way

---

## M1 — BSRNN implemented and training infrastructure trustworthy · target Sep 3 (weeks 3–4)

- [ ] Causal BSRNN + TF-Map extractor implemented; Luo & Yu (TASLP 2023) and
      Zhang et al. (ICASSP 2025) cited in-file
- [ ] STFT window/hop chosen against the ~200–300 ms budget, not the
      challenge's 100 ms cap, and the choice justified in `decisions.md`
- [ ] Model deliberately sized down from challenge scale (fewer blocks, smaller
      feature dim) and reported as such
- [ ] Target-absent training and channel-gap enrollment augmentation in
      (Li & Seki, 2026)
- [ ] YAML config committed — no hardcoded hyperparameters
- [ ] Seed set and logged
- [ ] **Checkpoint/resume proven across a deliberate session kill**

**Proof:** a 1-epoch run that completes, is killed, and resumes cleanly.
**Why resume is a checklist item and not an implementation detail:**
discovering it is broken at hour 11 of a 12-hour Kaggle session costs a week.

---

## M2 — Baseline trained · target Sep 17 (weeks 5–6)

- [ ] Converged checkpoint from conventional training (SI-SDR +
      multi-resolution STFT)
- [ ] Training curves and final losses logged with config, commit hash, seed,
      date

**Proof:** a checkpoint in `experiments/results/` that reproduces its own
reported numbers from its config.

**Run in parallel (no GPU):** draft `docs/data/metric-definitions.md` to v1 — fixed
prompt wording, response-transcription ASR chosen and pinned, judge shortlisted,
k-repeat protocol, cost model. See M3.

---

## M3 — Baseline evaluated conventionally · target Sep 24 (week 7)

Evaluated with the metrics that already exist, because the new one does not yet.
This is deliberate: these numbers become the "conventional metrics" column of
the divergence table in M5.

- [ ] SI-SDR, DNSMOS-P808, offline ASR WER on the held-out constructed set
- [ ] Measured algorithmic latency + RTF against the ~200–300 ms budget
- [ ] Listen to the outputs. Characterise the artefacts qualitatively — this is
      what tells you whether the artefact hypothesis in
      `metric-definitions.md` §1 is even plausible, and it should inform the
      final metric design

**Proof:** a results row with config + commit hash + seed + date.

---

## M4 — The metric is computable end to end · target Oct 1 (week 8)

Drafted during M2, finished here now that there is a real system to point it at.

- [ ] LCF-WER, ICR, NRR implemented
- [ ] **A scoring rule for silent-target trials.** B4 is decided (eval carries
      them) but `metric-definitions.md` defines nothing for a trial with no
      reference text, so the main score is not computable on them. Standing
      proposal: a separate false-alarm row, never folded into the headline
- [ ] Judge harness: fixed prompt, fixed response ASR, pinned model IDs, k≥3
      repeats, **input modality recorded per trial**, cost/compute logging
- [ ] Judge decided and its cost model resolved — closed API (money) or
      self-hosted open-weight (GPU-hours contending with training quota).
      **Currently unresolved**; this is the gating question for the whole
      milestone
- [ ] Trial-set size fixed to that budget, on a spreadsheet, before the harness
      is finalised
- [ ] **Floor and ceiling measured** (unprocessed mixture; clean target)
- [ ] Text reference condition wired: extractor → off-the-shelf ASR → text →
      judge, with its text floor and text ceiling
- [ ] Prompt-sensitivity ablation run

**Proof:** floor/ceiling numbers logged with config, commit hash, seed, date.
**Gate:** run-to-run spread must be smaller than the floor-to-ceiling gap. If
not, the metric is too noisy to detect system differences — fix before M5.

---

## M5 — Second model · target Oct 14 (week 10) · CUTTABLE

Proxy-objective fine-tune from the M2 checkpoint. Same architecture, same data,
same base checkpoint, different training objective — a controlled A/B, not a new
architecture.

*(Assumption: "second model" means the proxy-trained variant. If you meant a
genuinely different backbone, this milestone needs rewriting and almost
certainly does not fit before the freeze — say so and I'll redo it.)*

- [ ] Frozen-encoder feature-matching proxy implemented
- [ ] Proxy model family confirmed **different** from the judge, and recorded
      in the config
- [ ] Fine-tuned from the M2 checkpoint
- [ ] Speaker-similarity and target-activity auxiliaries if time
- [ ] ASR cross-entropy proxy only if comfortably ahead of schedule

**This is the first thing to cut.** If it goes, M6 compares the M2 baseline
against the anchors and the off-the-shelf system, which is still a result.

---

## M6 — The comparison · target Oct 14 (week 10)

The thesis's central finding. Runs immediately as M5 checkpoints land — it is
scoring, not training, so it does not need its own week.

- [ ] Baseline and second model both scored on LCF-WER / ICR / NRR
- [ ] Both scored on SI-SDR / DNSMOS-P808 / offline WER on the same trials
- [ ] All anchors present: audio floor, audio ceiling, text floor, text ceiling
- [ ] **≥1 off-the-shelf pretrained TSE system** scored alongside. No training
      cost, and it is what stops the divergence claim from resting on n=2
- [ ] AMI trial set built and the benchmark extended to it, if it survived
- [ ] Latency reported per modality

**Proof:** a table where ranking by SI-SDR / DNSMOS / offline-WER differs from
ranking by LCF-WER — or evidence that it doesn't.
**A negative result here is still a result**, and far better found now than in
week 12.

---

## M7 — Submitted · 2026-11-05

- [ ] Experiment freeze honoured (Oct 14)
- [ ] Every result traceable to config + commit hash + seed + date
- [ ] Every borrowed method cited
- [ ] Every deviation and cut recorded in `docs/decisions/decisions.md`
- [ ] Approximate-ceiling caveat stated wherever AMI numbers appear
- [ ] Modality recorded on every judge result, and the cross-modality caveat
      (`docs/data/metric-definitions.md` §3.5) stated wherever audio and text rows
      appear in the same table
- [ ] Written, reviewed, submitted

---

## Risks this ordering accepts

Recorded so they are accepted deliberately rather than discovered late.

**1. There is no zero-GPU fallback deliverable.** The previous ordering front-
loaded the metric and benchmark specifically so that a total training failure
still left a submittable thesis. Here, M4 onwards depends on M2 converging. If
training fails, the metric harness exists but has only floor, ceiling and one
off-the-shelf system to score.
*Mitigation:* the M0 checkpoint survey. If a public checkpoint exists, M2's
failure mode largely disappears. Do that search in week 1, not week 5.

**2. The primary contribution lands late.** Spec notes 1 and 8 make the metric
the deliverable that cannot be cut, and here it is finished in week 8 with M5
and M6 downstream of it. Any slip in M0–M2 pushes directly into it.
*Mitigation:* the M2-parallel drafting above. The metric must be *designed* on
paper by end of week 6 even though it is not *implemented* until week 8.

**3. Two evaluation passes over the baseline.** M3 scores it conventionally,
M6 scores it again with LCF. That is duplicated harness work, and it is the
price of defining the metric after building the model.
*Accepted:* the harness is shared, and having listened to real outputs before
freezing the metric is worth more than the duplicated pass costs.

**4. The freeze and M5/M6 land on the same date.** There is no slack. If M5
slips at all, it is cut rather than delayed — M6 must still run.

---

## Dropped from earlier milestone sets

See `docs/decisions/decisions.md`, 2026-08-07:

- Replication of REAL-TSE online-track baselines
- Use of the official challenge scoring pipeline as primary evaluation
- On-device / parameter / MAC budget study and quantisation work
- Matching any published challenge number
