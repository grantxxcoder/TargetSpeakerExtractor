# Milestones

Aligned to `docs/research-plan.md` (2026-08-07 re-scope). Submission
2026-11-05; hard freeze on new experiments **2026-10-14**.

Each milestone names the artefact that proves it is done. "Reviewed by
supervisors" is not a milestone — a thing that exists is.

---

## M0 — Re-scope agreed · target Aug 13 (week 1)

- [ ] Supervisors sign off that the metric is the primary contribution
- [ ] Supervisors sign off on the ~200–300 ms latency budget
- [ ] **Pilot: 20 trials by hand through one live model** — does the
      quality-vs-intelligibility divergence actually reproduce?
- [ ] Public streaming-TSE checkpoint survey done
- [ ] Open-weight speech-to-speech judge candidates shortlisted
- [ ] API cost per trial estimated; trial-set size sized to budget
- [ ] HPC access resolved either way

**Proof:** a short written pilot note with real judge responses in it.
**Gate:** if the divergence does not reproduce, stop and re-scope. Do not
build a harness for a phenomenon that isn't there.

---

## M1 — Metric computable end to end · target Aug 27 (week 3)

- [ ] Constructed trial set generated (target + interferer + noise + reverb,
      exact ground-truth text for both, speaker-disjoint splits)
- [ ] LCF-WER, ICR, NRR implemented
- [ ] Conventional metrics (SI-SDR, DNSMOS-P808, offline WER) on same trials
- [ ] Judge harness: fixed prompt, fixed response ASR, pinned model IDs,
      k≥3 repeats, cost logging
- [ ] Second (open-weight) judge wired in
- [ ] Prompt-sensitivity ablation run
- [ ] **Floor and ceiling measured** on the constructed set

**Proof:** floor/ceiling numbers logged in `experiments/results/` with config,
commit hash, seed and date.
**Gate:** run-to-run spread must be smaller than the floor-to-ceiling gap. If
not, the metric is too noisy to detect system differences — fix before
proceeding.

---

## M2 — The divergence table · target Sep 10 (week 5)

This is the thesis's central finding and the point of no-GPU de-risking.

- [ ] ≥2 off-the-shelf pretrained TSE systems scored
- [ ] Enhancement-only control scored
- [ ] Both judges, both anchors
- [ ] AMI trial set built (REAL-T-style construction, IHM approximate ceiling)
- [ ] Benchmark extended to AMI

**Proof:** a table where system ranking by SI-SDR / DNSMOS / offline-WER
differs from ranking by LCF-WER — or evidence that it doesn't.
**Note:** a negative result here is still a result, and is far better found
now than in week 11.

---

## M3 — Training infrastructure trustworthy · target Sep 10 (week 5)

- [ ] Training data generated (16 kHz; mode recorded in `decisions.md`)
- [ ] YAML config committed — no hardcoded hyperparameters
- [ ] Seed set and logged
- [ ] **Checkpoint/resume proven across a deliberate session kill**

**Proof:** a 1-epoch run that completes, is killed, and resumes cleanly.
**Why this is its own milestone:** discovering resume is broken at hour 11 of
a 12-hour Kaggle session costs a week.

---

## M4 — Baseline model trained and scored · target Oct 1 (week 8)

- [ ] Causal BSRNN + TF-Map extractor implemented, papers cited in-file
- [ ] Target-absent training and channel-gap enrollment augmentation in
- [ ] Converged checkpoint
- [ ] Scored on both trial sets, both judges
- [ ] Measured algorithmic latency + RTF, against the ~200–300 ms budget

**Proof:** our model appears as a row in the M2 benchmark table.

---

## M5 — Proxy-objective result · target Oct 14 (week 10) · CUTTABLE

- [ ] Frozen-encoder feature-matching proxy implemented
- [ ] Proxy model family confirmed *different* from every judge, and recorded
- [ ] Fine-tuned from the M4 checkpoint
- [ ] Ablation: base vs +feature-matching (vs +ASR-CE if time)
- [ ] Scored on LCF and conventional metrics

**Proof:** ablation table showing whether proxy alignment moves LCF-WER.
**This is the first thing to cut.** Claims 1 and 2 are a complete thesis
without it.

---

## M6 — Submitted · 2026-11-05

- [ ] Experiment freeze honoured (Oct 14)
- [ ] Every result traceable to config + commit hash + seed + date
- [ ] Every borrowed method cited
- [ ] Every deviation and cut recorded in `docs/decisions.md`
- [ ] Approximate-ceiling caveat stated wherever AMI numbers appear
- [ ] Written, reviewed, submitted

---

## Dropped from the previous milestone set

Recorded so the change is deliberate rather than silent — see
`docs/decisions.md`, 2026-08-07:

- Replication of REAL-TSE online-track baselines
- Use of the official challenge scoring pipeline as primary evaluation
- On-device / parameter / MAC budget study and quantisation work
- Matching any published challenge number
