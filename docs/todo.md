# TODO

Re-scoped 2026-08-07. Ordered by what unblocks the most. See
`docs/milestones.md` for dates and `docs/research-plan.md` for why.

## Week 1 — de-risk (do these before anything else)

- [ ] **Pilot the core premise.** ~20 trials by hand through one live model:
      unprocessed mixture vs an off-the-shelf TSE output vs clean target.
      Does the divergence actually reproduce? Everything depends on this.
- [ ] Survey public pretrained **streaming** TSE checkpoints (WeSep family,
      HuggingFace, USEF-TSE). A usable one changes the compute picture.
- [ ] Shortlist open-weight speech-to-speech judges (the reproducibility
      anchor — the metric's shelf life depends on having one)
- [ ] Estimate API cost per trial; size the trial set to budget on a spreadsheet
- [ ] Resolve HPC access
- [ ] Take the re-scope to supervisors — get notes 1 and 8 minuted, and the
      latency budget agreed

## Literature — read for the new objective

Priority reflects the re-scope; see `literature/review_synthesis.md`.

- [ ] **PS4** — Ning et al., arXiv:2607.08111. Closest prior art to the whole
      project. Read first.
- [ ] **Ma et al.** — arXiv:2501.14477. Joint generative TSE + TS-ASR for
      intelligibility; addresses the transcribable-but-worse phenomenon.
- [ ] **Delcroix et al.**, Interspeech 2022, "Listen only to me!" —
      target-absent behaviour and false alarms.
- [ ] **REAL-T** — Li et al., Interspeech 2025. Needed for the AMI trial
      construction method.
- [ ] Žmolíková et al., IEEE SPM 2023 — background chapter only, not SOTA
- [ ] Evidence on live-model turn-taking latency tolerance, to justify the
      200–300 ms budget (currently an unsupported assumption)

Already read: REAL-TSE overview, CARTSE, Multi-Level Speaker Representation,
BSRNN.

## Metric and harness

- [ ] Constructed trial generation (target + interferer + noise + reverb,
      verbatim text for both, speaker-disjoint)
- [ ] LCF-WER / ICR / NRR scoring
- [ ] Conventional metrics on the same trials, for the divergence table
- [ ] Judge harness: fixed prompt, fixed response ASR, pinned IDs, k≥3 repeats
- [ ] Prompt-sensitivity ablation
- [ ] Decide: add a semantic-equivalence score alongside strict WER?
      (live models paraphrase; strict WER may penalise correct answers)

## Model

- [ ] Causal BSRNN + TF-Map implementation (cite Luo & Yu 2023; Zhang et al. 2025)
- [ ] Target-absent training + channel-gap enrollment augmentation (cite Li & Seki 2026)
- [ ] Checkpoint/resume proven across a session kill — before any real run
- [ ] Frozen-encoder feature-matching proxy; **confirm different model family
      from every judge** and record it in the config

## Housekeeping

- [ ] Fix TF-MLPNet citation in `docs/specification.md`: it is the 6th Clarity
      Workshop (Clarity 2025), not Interspeech 2025
- [ ] Fix duplicate numbering in `docs/specification.md` — there are two
      items numbered 9
- [ ] Replace the placeholder commit hashes in `README.md`
- [ ] Create `experiments/configs/` and `experiments/results/` (required by
      CLAUDE.md, don't yet exist)
