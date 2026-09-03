# Milestones

**Rewritten 2026-08-07** to a build-first ordering: data → baseline model →
evaluate it conventionally → define the metric → second model → compare on
the metric → report.

Submission 2026-11-05; hard freeze on new experiments **2026-10-14**.

**Current status lives in `docs/project-state.md`.** This file holds the
schedule and the per-milestone checklists.

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

## M0 — Data exists · target Aug 20 (weeks 1–2) · CLOSED

**Closed.** All six manifests audited, all 21,208 trials rendered — 63,624 files,
27 GB, 105.4 h, 0 failures, 40 checked by ear. Every Group A/B/C decision closed,
C2 last on 2026-08-30 (floor 57.4 %, ceiling 6.1 %, n=230, accepted). B12's
sampler and the B2 VAD pass are wired in; the manifest rebuild (PR3, 2026-08-14)
cut the speaker-identity leak 0.795 → 0.508. Rationale and every closed decision
are in `decisions-m0.md`.

**Proof:** a generated set on disk with a manifest, plus config + commit + seed
in `experiments/results/`.

Still open:
- [~] ~~**Narrow `overlap_ratio` in the `base` regime.**~~ **CUT 2026-09-03.**
      B1 said narrow it last; "last" has arrived and the answer is not to. Every
      trial is rendered and `sir0_train` is mid-extension to 9,955 — changing the
      sampler now invalidates the lot and forfeits comparability with every result
      to date, to buy a difficulty shift nothing is asking for. The 0.7 ceiling
      stays, matched to REAL-TSE. decisions-m0.md 2026-09-03
- [X] ~~**Write the two-speaker limitation into the thesis**, and decide whether the
      AMI check is restricted to ≤2 active speakers or reported as an
      out-of-scope probe. `decisions-m0.md` 2026-08-14.~~ **CLOSED 2026-09-03 as
      duplicated** — thesis text is tracked in M7 ("Two-speaker boundary stated as a
      declared limit"), which is the right home. The AMI half is moot while the AMI
      set is unbuilt (M6); if AMI is cut, so is the question.
- [~] ~~**Enrollment offset (B2 PR3, optional).**~~ **CUT 2026-09-03.** Real but
      marked optional at the outset: `enroll_offset` is drawn uniformly and
      `long_enough` filters on file duration, so a 5.2 s file with 1.2 s of leading
      silence yields a "5 s enrollment" holding 4 s of voice, against B10/A1's ≥5 s.
      Fixing it means re-rendering every enrolment and its 3-variant bank — ~1.5 h
      plus a re-upload, and it breaks comparability with every trained checkpoint.
      **Report as a known enrolment-quality caveat instead.** decisions-m0.md 2026-09-03
- [~] ~~**`length_mode`**~~ **CUT 2026-09-03.** Never implemented; every split is
      rendered without it and nothing downstream references it. Left documented as
      not-implemented in `data-construction-parameters.md` rather than carried as
      an open task. decisions-m0.md 2026-09-03
- [ ] **Three renderer points were interpreted, not decided** — noise covering
      A5's tail, the `noise_only` level anchor, and the enrollment's level. Worth
      a supervisor glance. `decisions-m0.md` 2026-08-16.
- [ ] **EDA per parameter** — realised vs intended distribution for each, to
      verify B12.
- [ ] **Revise notebook §2, §7 and the final health checks** after the rebuild;
      §3–§6 survive. The §7.5 leak scoreboard ran standalone, so those cells
      still print pre-rebuild figures.
- [X] ~~Survey public streaming-TSE checkpoints — **done 2026-09-03**: the WeSep
      `tfmap_context_causal_100` checkpoint is rendered and scored as system 5.~~
- [X] ~~**HPC access resolved either way.**~~ **RESOLVED 2026-09-03: no HPC, and
      it no longer matters.** Never secured; every training run has been Kaggle T4.
      `research-plan.md` flagged it as "materially changes what the second model can
      be" — that ceased to be true when M5 became a per-band gate fine-tuned from an
      existing checkpoint (tens of thousands of parameters, hours not days). Compute
      is not the constraint on the second model; time is.

---

## M1 — BSRNN implemented, training infrastructure trustworthy · target Sep 3 · CLOSED

**Functionally complete 2026-08-24, ten days early.** Causal BSRNN + TF-Map
implemented (`src/models/`), cited in-file: Luo & Yu (TASLP 2023), Yu et al.
(Interspeech 2023), Zhang et al. (ICASSP 2025). STFT 512/128 `center=False`;
future dependency *measured* at 23.9 ms. Model deliberately sized down to
**7.19 M** against the REAL-TSE causal baselines' 25–27 M — quote 7.19 M in the
thesis. Config in `experiments/configs/`, seed set before the model is built.
Rationale in `decisions-m1.md` (08-18 to 08-24); the objective is
`decisions-m2.md` 2026-08-20.

Three things proven by measurement rather than assumed:

- **Resume works.** Scheduler `_last_lr` stayed 0.00025 rather than resetting to
  0.0005, and AdamW's `step` read 480 = 464 + 16, so state was restored, not
  reinitialised.
- **Causality: 1.68e-08.** Appending later audio leaves earlier output unchanged,
  so one full-length pass is what streaming emits and no chunk-stitching is
  needed. Concatenating independent 4 s chunks is *worse* — it reinjects the
  384-sample (23.4 ms) overlap-add tail at every seam (4.37e-03 max,
  rel L2 1.04e-02).
- **Loss floor is exactly −30** for any `w` in [0,1] and any `w_m`; only `tau`
  moves it. Do-nothing anchor −2.24, so the usable range is −2.24 → −30.

**Proof:** a 1-epoch run that completes, is killed, and resumes cleanly. Resume is
a checklist item because discovering it is broken at hour 11 of a 12-hour Kaggle
session costs a week.

Still open, neither blocking M2:
- [ ] **Unit tests for the model modules.** `tests/test_losses.py` has 30; there
      are none for `stft.py`, `bands.py`, `modules.py`, `conditioning.py` or
      `bsrnn.py`. "Training infrastructure trustworthy" is not met by a loop that
      runs. The causality property is the obvious first test to lift.
- [ ] **Train/inference context mismatch, found 2026-08-24.** Causal is not
      context-free: training only feeds cold-start 4 s crops while deployment
      hands the model unbounded warm state. On the epoch-4 smoke checkpoint the
      same window scores 30.6 % rel L2 apart cold vs warm, though `total` moves
      only 0.005. Re-measure on a converged checkpoint: if the gap grows it
      argues for warm-state or longer-context training; if it shrinks it retires
      with evidence. Needs a `decisions-m2.md` entry either way.

---

## M2 — Baseline trained · target Sep 17 (weeks 5–6)

**Status 2026-08-30. M2's proof is met and the milestone is functionally
complete — but the checkpoint it produced memorises its training set.** A run on
`sir0` (1,989 trials) converged and early-stopped of its own accord at epoch 24,
best at 14. Held-out separation peaked at 2.14 dB and then fell to −0.17 dB,
below pass-through, while training separation improved every single epoch. The
diagnosis is data volume, not architecture: **a 7.19 M-parameter model has enough
capacity to memorise 1,989 scenes**, so adding capacity would make it worse.
Two augmentations were built in response (enrolment bank D8a, SIR/SNR remix D8b,
both `decisions-m2.md` 2026-08-30) and are running as separate arms. Compute,
batch size, early stopping and the `L_gain` re-run are all closed below. Still
open in M2: the band-plan and `w_m` ablations, and `w`'s re-derivation at
batch 3.

Superseded status, 2026-08-24, kept for the record —
**Started early: the loop works on `smoke`, nothing is
converged, and `--split full` cannot run on this laptop.** 31 epochs total on the
50-trial smoke split — 30 in one run, then epochs 29-30 via `--resume`.
`models/model_smoke.pt` is at epoch 29, `best_val -12.9296`.
`ReduceLROnPlateau` fired once, at epoch 16; lr has been 0.00025 since.
Artefacts: `experiments/results/2026-08-24-train-smoke/{history.csv,loss_plot.png}`
and `-train-smoke-resume/` for the resumed epochs. Treat it as a wiring proof,
not a result — and see the collapse item below, which is now measured rather
than suspected.

**The total plateaued around epoch 20.** Slope over the last 10 epochs is
-0.0222/epoch at t = -0.43, i.e. indistinguishable from flat, against
-0.1285/epoch (t = -3.33) over the last 15. It is not converged, it is
oscillating: val sd 0.423 and range 1.416 over the last 10 epochs on a **fixed**
val set (`random_crop=False`, so that spread is the weights moving, not crop
noise). At batch 3 over 50 trials with `drop_last=True` that is 16 optimiser
steps per epoch.

- [X] ~~**Converged checkpoint from conventional training** (SI-SDR +
      multi-resolution STFT). **2026-08-29**,
      `experiments/results/2026-08-29-train-sir0-e50-resume/`: sir0, seed 42,
      requested 50 epochs, **early-stopped at 24 on a fixed val set, best epoch
      14** (`val_total` −2.178). Convergence is real and the proof below is met.
      **But it converged to an OVERFITTED optimum** — train separation went
      2.97 → 5.51 dB while held-out fell 1.52 → −0.17 dB, i.e. worse than
      pass-through. It is the M2 baseline and it memorises. decisions-m2.md
      2026-08-29~~
- [X] ~~Training curves and final losses logged with config, commit hash, seed,
      date — `log_results()` writes `meta.yaml` + per-epoch wide `history.csv`
      (train and val on one row, plus `lr`, which is what distinguishes a plateau
      from a scheduler step). Never raises: a logging bug must not discard a
      finished run~~
- [X] ~~**Compute resolved — Kaggle T4, and the step is 7.2x faster.** 523 s/epoch
      at batch 3 over 1,989 trials (`run_times.md` 2026-08-29) against 3,773
      before the fp16 tensor-core frame alignment and AMP, so 10 epochs is
      **1.45 h, was 10.5 h**. The laptop is still GPU-less and is used for
      rendering and CPU evaluation only. decisions-m2.md 2026-08-28. Original
      note kept below for the record:~~
- [X] ~~**Compute is the blocker, not the code.** 15.7 GB RAM with VSCode open is
      not enough — `systemd-oomd` killed the editor on 2026-08-24 before training
      started. `requirements.txt` pins a CPU torch. The one measured row in
      `run_times.md` is **277 s/epoch** at batch 3 over 50 trials on CPU (2.3 h
      for 30 epochs — the 243 s/epoch row is a 1-epoch run and includes startup,
      so prefer the 30-epoch figure). Smoke timing: **must not be extrapolated**
      to 19,938 trials. Server-class compute or Kaggle is required, which makes the M1 resume proof urgent~~
- [X] ~~**`batch_size` — MEASURED, and 3 is the answer.** 12 OOMs on a 14.6 GiB
      T4; the Kaggle notebook's probe steps down until one fwd+bwd+step fits and
      writes the winner into the config that trains. So 3 is a measured ceiling,
      not a laptop compromise, and the 2026-08-18 choice of 12 is superseded.
      **Consequence that survives and is still open:** `w` = 0.458 was calibrated
      against an absent-crop rate derived at batch 12 and has never been
      re-derived at 3~~
- [X] ~~**Early stopping DID fire — the 2026-08-24 prediction was wrong.**
      2026-08-29 requested 50 epochs, ran 15 (epochs 10–24), `early_stopped:
      true`, patience 10, best epoch 14. No minimum-delta threshold was needed:
      once `L_gain` closed the mute, val stopped creeping downward and began
      genuinely degrading, so patience had a real signal to detect. The original
      analysis, correct for the smoke run it was made on, is kept below:~~
- [X] ~~**Early stopping will not fire as configured.** `patience: 10` resets on
      *any* improvement, and best-val keeps creeping down by less than the noise
      (blocks of 5 epochs: -8.636, -9.495, -10.276, -12.301, -12.706, -12.900,
      then -12.9296 on resume — a 0.03 gain against a 0.42 sd). Left alone this
      run grinds to epoch 99 chasing `L_abs` to its floor. Needs a minimum-delta
      threshold before `--split full`, or the epoch budget is the only stop
- [ ] **Two ablations are declared but unrun** — the band plan (six candidates,
      `decisions-m1.md` 2026-08-18) and `w_m` (`ablate_w_m: [0.0, 2.89, 9.62]`,
      the 0 arm required). Both need the converged baseline first
- [X] ~~**CONFIRMED 2026-08-24: the smoke model attenuates rather than separates.**
      93 % of the total's movement was the absent half; output RMS −24.9 dB below
      the mixture on present crops; `L_MR` flat for 30 epochs. **Superseded by the
      2026-08-27 sir0 run**, which reproduced it at 40x the data (95 %, −22.4 dB)
      and traced the cause to the objective. See `decisions-m2.md` 2026-08-27/28~~
- [X] ~~**Re-run with `L_gain` on — done 2026-08-28**, exactly as specified:
      `experiments/results/2026-08-28-train-sir0-e10/`, fresh (not a resume),
      `--epochs 10`, `w_g` 1.69. **The mute closed and conditioning followed** —
      an enrolment swap moved the output **37.6 %** against 14.9 % for the
      `w_g`=0 control, and the output sits at −4.2 dB where the target is at
      −3.9, against the control's −22.4 dB. decisions-m2.md 2026-08-28~~

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

- [X] ~~**SI-SDR, SIR/SAR, DNSMOS-P.835 + P.808 and offline ASR WER on the
      held-out set.** 2026-09-01 on `sir0_val` `both`, n=103, all three reference
      points. See the results table in `project-state.md`. decisions-m3.md
      2026-09-01~~
- [X] ~~**Measured algorithmic latency + RTF against the ~200–300 ms budget.**
      2026-09-01: 80 ms chunks, i5-1135G7 4 threads — **RTF 0.528 mean / 0.706
      p99**, latency **162 ms mean / 176 ms p99**, no chunk misses the 80 ms
      deadline. Lookahead 40 ms by the STFT convention, 23.9 ms measured as
      effective future dependency. **An estimate, not a streaming measurement**:
      chunks are processed independently since there is no stateful path,
      10–20 % error. GPU figure still outstanding. decisions-m3.md 2026-09-01~~
- [ ] Listen to the outputs. Characterise the artefacts qualitatively — this is
      what tells you whether the artefact hypothesis in
      `metric-definitions.md` §1 is even plausible, and it should inform the
      final metric design

**Proof:** a results row with config + commit hash + seed + date.

---

## M4 — The metric is computable end to end · target Oct 1 (week 8)

Drafted during M2, finished here now that there is a real system to point it at.

- [X] ~~**LCF-WER, ICR, NRR implemented.** 2026-08-31, `src/live_model_metric/`,
      63 tests, judge-agnostic with the transcriber swappable. Validated by
      reproducing the C2 floor and ceiling exactly. decisions-m4.md 2026-08-31~~
- [X] ~~**J3 — the ICR overlap threshold signed off.** `count>=2` declared with a
      sensitivity sweep; the exclusion rule and the floor row's
      by-construction ICR both need stating. decisions-pending.md J3~~
      **SIGNED OFF 2026-09-03**, written into `metric-definitions.md` §3.2:
      threshold `count>=2`; sensitivity table showing the floor moves 57.0 -> 42.9
      across k=1/2/3/5 while the ceiling stays 0.0, so **the choice of k changes no
      conclusion**; exclusion rule stated (currently 0 exclusions in every
      `eval_public` stratum); the floor's by-construction ICR stated with the
      87.5 % / 37.5 % interferer-louder split; and the binding prompt constraint
      that the judge must never be told to pick a speaker. decisions-m4.md 2026-09-03
- [X] ~~**B4's scoring rule written into `metric-definitions.md`** — done
      2026-09-01, §3.1: absent trials excluded not scored zero, with the
      invented-speech row and the measured 95.1 % / 0.0 % figures. Originally
      decided
      2026-08-13: absent trials are excluded from the main score and reported as
      their own invented-speech row, never folded into the headline. The decision is
      made; the document still defines nothing for a trial with no reference text
- [X] ~~**B5's normaliser pinned in the spec** — done 2026-09-01, §3.1:
      `whisper-normalizer==0.1.15`, Radford et al. Appendix C, applied identically
      to both sides, frozen before the first judge result and never adjusted per
      system (decisions-m0.md 2026-08-13)
- [X] ~~**Judge harness built.** 2026-09-02. Prompt frozen at sha256[:12]
      `d118b7d3bf30` and its hash is in every cache key; model ID, serving
      backend, input modality and run date recorded in the provenance of every
      result; call counts and cache hits logged per system. No response ASR —
      J1 deleted it. `src/live_model_metric/judge.py`, `--listener judge` in
      `scripts/evaluate.py`, 63 tests~~
      **AMENDED: k≥3 is NOT applied to the anchors.** target/mixture/interferer
      are judged ONCE per instrument (the run-once rule) because no model can
      change that audio, so a second answer buys nothing but cost. Repeats are
      opt-in and used only for the deliberate spread study
      (`scripts/judge_spread.py`). Consequence: **per-trial** judge claims need
      k≥3 and averaging; aggregates over 100+ trials do not.
      decisions-m4.md 2026-09-02
- [X] ~~**Judge CLASS decided — audio-in / text-out, not full-duplex.**
      2026-08-31, J1 closed: LCF measures the judge's audio encoder, not its
      duplexing, so full duplex is not required. Deletes the
      response-transcription ASR from the measuring instrument. Carries a
      ~50-trial full-duplex confirmation run as part of the decision.
      decisions-m4.md 2026-08-31~~
- [X] ~~**Cost model resolved** — not a budget question. ~$4 for the audio
      condition at 200 trials x k=3 x 4 systems, ~$25 including the
      prompt-sensitivity ablation. decisions-pending.md J2~~
- [X] ~~**J2a chosen: `gemini-3.7-flash`**, audio-in / text-out, AI Studio,
      prepay tier. 2026-09-02. Picked over the `preview` Live models because it
      is STABLE — the metric needs the audio encoder, not duplexing (J1) — and
      the Live socket added a deprecation risk for nothing. Price recorded with
      its expiry: $0.75/1M audio tokens through 2026-12-31.
      decisions-m4.md 2026-09-02~~
- [~] **J2b — the open-weight anchor. CUT 2026-09-03**, for time not merit. Was
      Qwen3-Omni, chosen for encoder independence from our reference ASR. Costs
      reproducibility: judge numbers need API access to regenerate, so it goes in
      the thesis limitations. The harness is judge-agnostic, so adding one later is
      a config change. decisions-m4.md 2026-09-03
- [X] ~~**Candidate gate run, 2026-09-02, before anything was committed to.**
      Three readings: (1) it reports clean speech — ceiling **1.05 %** at n=103,
      against the offline ASR's 5.85 %, and byte-identical five times out of
      five; (2) it can FAIL — floor **63.27 %**, so 62 points of dynamic range;
      (3) **it does NOT stay quiet on silence** — 0 of 6 silent clips returned
      `no_speech`, 17-42 words invented, one in French, under three prompt
      variants. Reading 3 failed and produced the speech gate rather than a
      different judge. decisions-m4.md 2026-09-02~~
- [X] ~~**Speech gate built** because of that failure.
      `src/live_model_metric/speech_gate.py`, 22 tests. Anchors decided by
      construction from the manifest condition; `estimate.wav` by Silero VAD
      6.2.1 (B2) at 0.10 s. Applied identically to the judge AND the offline ASR
      — gate one and not the other and every difference between them on a
      speech-free clip measures the gate. Every decision logged to
      `experiments/results/speech_gate.csv`. It is a component of the measuring
      instrument: metric-definitions.md §3.1~~
- [X] ~~**Judge failure modes characterised**, both reportable findings an
      offline ASR cannot exhibit: invents speech on silence (above), and a
      **safety filter refused an extractor output** once in 309 calls and then
      PASSED ON RETRY. Non-deterministic, so retried twice before being believed,
      with `filter_blocked` and `filter_transient` reported per system because a
      filter that fires on one system's output and not another's is a bias.
      decisions-m4.md 2026-09-02~~
- [X] ~~**Trial-set size fixed and it is no longer a budget question.**
      2026-09-02: the full `sir0_val` row (103 `both` trials x floor/estimate/
      ceiling) cost **about 16 cents**, and the whole live-judge programme to
      date about **40 cents** — against the ~$25 the cost model budgeted. B6's
      200-scored floor and B13's 100-per-bucket stand unchanged, and 500 are
      generated per eval split. Scoring more extends the set rather than
      replacing it, because the cache is keyed per clip. decisions-m4.md
      2026-09-02~~
- [X] ~~**Floor and ceiling measured** (unprocessed mixture; clean target) —
      **2026-08-30**, n=230 `both` trials on `eval_public`: floor **57.4 %**,
      ceiling **6.1 %**; `sir0_val` 65.2 % / 5.8 % at n=103. C2 accepted at this
      range. Scored from `transcripts.csv`, no new ASR run. decisions-m3.md
      2026-08-30~~
- [~] ~~Text reference condition wired: extractor → off-the-shelf ASR → text →
      judge, with its text floor and text ceiling~~ **CUT 2026-09-03, ruled out on
      LATENCY rather than scored on content.** Extraction alone is 162 ms mean /
      176 ms p99 of a 200-300 ms budget, and the ASR is non-streaming: it needs an
      endpoint before it can decode. The text path cannot meet the budget, so an
      LCF row for it would score a system inadmissible under the project's own
      constraints. Reported as a latency exclusion in the modality section.
      decisions-m4.md 2026-09-03
- [X] ~~**Prompt-sensitivity ablation run — and the metric is NOT fragile to
      prompt wording.** 2026-09-02, three variants (`d118b7d3bf30`,
      `a21169a651d6`, `64d4b994a9a2`) over the same clips. On one floor clip the
      cross-prompt range was 18.0 points while the SAME-prompt run-to-run spread
      on that clip was 16.0 points — indistinguishable. The apparent prompt
      effect was sampling noise. v1 stays frozen. decisions-m4.md 2026-09-02~~
- [X] ~~**Run-to-run spread measured — the milestone Gate below, and it passes
      by an order of magnitude.** `scripts/judge_spread.py`, 2026-09-02: ceiling
      **0.0 points** across five identical calls (byte-identical answers);
      mixture **2.9 points** on one clip and **16.0** on a more ambiguous one,
      against a 62-point floor-to-ceiling range. The variance mechanism is
      visible: on the noisy clip it sometimes transcribes one speaker and
      sometimes interleaves both. decisions-m4.md 2026-09-02~~
- [X] ~~**FIRST LIVE-JUDGE ROW SCORED.** 2026-09-02, all 103 `sir0_val` `both`
      trials: floor 63.27 %, baseline **56.72 %**, ceiling **1.05 %**; ICR@2
      75.73 → 62.14 %. **Headline is NEGATIVE: 10.5 % of headroom captured
      against the offline ASR's 10.4 %** — the two listeners agree, so on this
      system LCF-WER does not rank differently from offline WER. M6 asks for
      exactly this evidence. It cannot currently be otherwise: a rank inversion
      needs two systems and there is one. project-state.md, results table~~

**Proof — MET 2026-09-02.** `experiments/results/2026-09-02-evaluate-sir0_val/`
carries the numbers with the git commit, the date, the model ID, the prompt hash,
the modality and the gate state in its provenance.

**Gate — PASSED 2026-09-02, by an order of magnitude.** Run-to-run spread 0.0
points on the ceiling and 2.9-16.0 on mixtures, against a 62.2-point
floor-to-ceiling range. The metric is not too noisy to detect system differences.
*Carried consequence:* the offline ASR is deterministic and the judge is not, so
single-trial judge claims need k≥3 and averaging.

**WHAT M4 DID NOT AND CANNOT DELIVER.** A divergence between LCF-WER and
conventional metrics. Not because the metric failed — it works, and it is a
better instrument than the ASR on every axis except agreement — but because
**one system cannot be ranked against anything.** That is M6's problem and M6
already names the fix: an off-the-shelf pretrained TSE system, no training cost.
**The bottleneck has moved from the metric to having a second thing to measure.**

---

## M5 — Second model · target Oct 14 (week 10) · CUTTABLE

**Scope re-ordered 2026-09-01 after the mix-back sweep. The second model is now
an INPUT-CONDITIONED GATE, with the artefact weight `BETA` demoted to a
secondary arm.**

### Why the order changed

The sweep (`decisions-m3.md` 2026-09-01) screened both ideas for free:

- **Globally, alpha = 1 wins.** No interior optimum. The model is already at the
  best global aggressiveness, which is **evidence against `BETA > 1` as a
  standalone intervention** — a global shift toward gentler masking is the wrong
  direction.
- **Per difficulty, the optimum spans the entire range.** Easy trials want
  alpha = 0 (no filtering), hard trials want alpha = 1. **A single constant
  cannot be right**, so making the model adapt is the intervention the data
  supports.
- **An oracle gate is worth ~2.2 points** of LCF-WER (59.1 -> 56.9), with 6.6 as
  an unreachable per-trial ceiling.

### The gate, and it is the K=2 case of a mixture of experts

Predict the blend per frame and per band from features the model already computes:

```
alpha[t,b] = sigmoid( w_b . h[t,b] + b_b )
S_hat[t,b] = alpha[t,b] * ( m[t,b] * X[t,b] ) + (1 - alpha[t,b]) * X[t,b]
```

Fully differentiable with no trick, since
`d S_hat / d alpha = m*X - X` and the sigmoid derivative is standard.

**Algebraically this is a two-expert mixture whose second expert is the identity
mask:**

```
alpha (m * X) + (1 - alpha) X  ==  [ alpha*m + (1-alpha)*1 ] * X
```

So it is not an alternative to the mixture-of-experts idea — **it is that idea's
base case at near-zero parameter cost**, and the natural first step before
scaling K. See `decisions-pending.md` D12.

**Cost.** On the order of `B x (|h_b| + 1)` parameters — tens of thousands
against 7.19 M, so **capacity risk is negligible on a data-limited model.** The
backbone is untouched, so the 2026-08-28 architecture freeze survives. No latency
change: output frame `t` needs only input frame `t`.

**Why per-band rather than per-frame.** The band-split architecture already
carries per-band features, and artefacts concentrate in the high bands while
speech energy sits in the low ones. A scalar gate cannot express that; a per-band
gate can, for the same order of cost.

### Checklist

- [X] ~~Screening test: does the optimum differ by difficulty? **Yes, across the
      full range.** `decisions-m3.md` 2026-09-01~~
- [ ] Per-band gate implemented; verify that forcing `alpha = 1` reproduces the
      current model **bit for bit** before training anything
- [ ] Fine-tuned from `models/model_sir0_5000-e7.pt`, everything else held fixed
- [ ] Gate behaviour inspected against trial difficulty — **does it actually
      learn to back off on easy trials?** That is the hypothesis, and a gate that
      saturates at 1 everywhere is a negative result worth reporting
- [ ] Scored into row 3 of the results table in `project-state.md`
- [ ] Compared against the oracle's 56.9 % to say how much of the available gain
      was captured

### Secondary arm: the artefact weight `BETA`

Demoted, not dropped. The sweep argues against a global gentleness shift, but it
is an imperfect proxy — a retrained model learns a different mask rather than a
blended one. Worth one arm **if the gate lands and time allows**, and it remains
the cleanest test of the masking question below.

### The term: one weight inside `L_pres`, not a second loss

**`L_pres`'s denominator is ALREADY the sum of the two piles**, exactly and
verifiably. The three components are mutually orthogonal, so

```
||s_hat - alpha*s||^2  ==  ||e_interf||^2 + ||e_artif||^2
```

checked numerically to six decimal places on 2026-09-01, with all cross inner
products at 1e-12. `L_pres`'s `alpha*s` IS the projection onto the target.

So the two piles currently cost **exactly the same per unit of energy, 1:1**. That
is the thing to change, and the change is a single ratio:

```
L_sep = -10 log10(  ||alpha*s||^2
                  / ( ||e_interf||^2 + BETA*||e_artif||^2 + tau*||alpha*s||^2 ) )
```

`BETA` states how much more damaging invented energy is than leaked energy.

**Why this beats adding a separate `L_artif` term.** Four reasons, the second
decisive:

1. **`BETA = 1` recovers `L_pres` EXACTLY**, by the algebra above. The control arm
   is the existing baseline with no re-derivation, and every previous run stays
   comparable.
2. **It removes the degenerate attractor.** A separate artefact term scores
   *perfectly* on pass-through — the mixture invents nothing, so `s_hat = x` wins
   it outright, a new pull toward doing nothing structurally identical to the mute
   of 2026-08-25. With `BETA` inside the ratio, pass-through's interference term
   is enormous and it can never win. **The pass-through collapse risk largely
   disappears.**
3. **No new anchor derivation.** `w_m` and `w_g` needed break-even tables because
   they reconcile terms in *different units*. `BETA` is dimensionless inside an
   already-calibrated ratio, so it needs none.
4. **One number, still in dB**, same `tau` behaviour, same scale as every
   `L_pres` ever reported.

**What it actually encourages.** The model's only lever is the mask, and a
gentler, smoother mask means less artefact and more residual interference.
`BETA` sets where on that curve the optimum sits. Unlike an inference-time
mix-back gain, it lets the model choose *where in time and frequency* to be
gentle.

**Differentiable and cheap.** A 3x3 normal-equations solve per crop;
`torch.linalg.solve` carries gradients and the cost is negligible beside the
BSRNN forward pass. **No new data** — `interferer.wav` is rendered and the loader
already derives `noise = mixture - target - interferer`.

### Choosing BETA — screen it for free before spending training time

**It is one new hyperparameter, and the only one.** But it does not have to be
guessed, and it must not be swept blind at 6.2 h per arm.

**Step 1, free: use the D11 mix-back sweep as a screening test.** Blending the
mixture back at inference walks the *same* trade-off curve — more mixture means
more interference and less artefact — with no retraining. **If the sweep's optimum
sits at `alpha < 1`, gentler is better and `BETA > 1` is worth training. If the
optimum is `alpha = 1`, `BETA > 1` will probably not help and roughly 25 h of
Kaggle time has been saved.** Run this first.

**Step 2, if screening says go: ablate, as `w_m` and `w_g` were.** `BETA` in
{1, 2, 4, 8}, with `BETA = 1` the free control. Select on held-out LCF-WER, which
is the quantity actually being optimised for. Four arms at 6.2 h is ~25 h, so
three or four Kaggle sessions — budget it before starting.

**Step 3, as a sanity check only: estimate BETA from the measurement.** The
instrument now exists to regress word-error against the two energies separately
and read `BETA` off as the ratio of their coefficients. **Do not trust it yet:**
the 2026-09-01 correlations were weak — delta SAR against word-error improvement
was **-0.05**, i.e. null — so a naive regression would return `BETA` near or below
1, saying "artefacts do not matter". That may be true, may be an artefact of
n=103, and may be an artefact of scoring through an ASR rather than the judge.
**Use it to check the ablation's answer, never to replace it.**

### Open design questions

- **Rank-deficient crops.** On `target_only` and `noise_only` crops the source
  basis is degenerate. Apply the decomposition to present crops with a genuine
  interferer only; elsewhere fall back to `BETA = 1`, which is exactly today's
  `L_pres`.
- **Absent crops.** Excluded, as the other present-branch terms are.
- **Reporting.** `L_sep` at `BETA = 1` must be logged alongside whatever `BETA`
  trained, so the curve stays readable against the historical `L_pres`.

### Checklist

- [ ] **D11 mix-back sweep run first as the screening test.** No `BETA` training
      until it says gentler is better
- [ ] `L_sep` implemented with `BETA = 1` verified to reproduce `L_pres` bit for
      bit on a fixed crop — the ablation is worthless without that
- [ ] Applied to present crops with a genuine interferer only
- [ ] `BETA` ablated over {1, 2, 4, 8}, selected on held-out LCF-WER
- [ ] Fine-tuned from `models/model_sir0_5000-e7.pt`, everything else held fixed
- [ ] Scored into row 3 of the results table in `project-state.md`
- [ ] Reported as **evidence about masking**, not merely as a better model

### Why this is a rebuke of the masking parameterisation

Worth stating as the argument, because it is the reason this is interesting rather
than merely corrective. **A mask can only attenuate time-frequency bins that
already exist; it cannot synthesise.** The artefacts are the by-product of
imperfect attenuation — spectral holes, musical noise, phase damage. A term that
penalises invention specifically therefore pressures the model toward gentler,
smoother masks.

**So the experiment has two possible outcomes and both are results.** If artefact
falls without suppression falling, masking was simply being applied too
aggressively. **If SAR cannot be improved without giving up SIR, that is evidence
masking is the wrong output parameterisation for this task** — which is the
argument for a mapping or generative output, and a finding worth reporting even
though building that replacement is out of scope before the freeze.

### Checklist

- [ ] `w_a` derived from measured anchors, **including pass-through**, before any
      training run
- [ ] `L_artif` implemented with the double-count stated in the config comment
- [ ] Applied to present crops with a genuine interferer only
- [ ] Fine-tuned from `models/model_sir0_5000-e7.pt`, everything else held fixed
- [ ] Scored on all metrics into row 3 of the results table in `project-state.md`
- [ ] Reported as **evidence about masking**, not merely as a better model

*(Superseded plan, kept for the record: a frozen-encoder feature-matching proxy
fine-tune. Dropped because the artefact penalty targets a cause measured in this
project's own data, whereas the proxy was borrowed from PS4 and would have needed
its own anchor work regardless. Speaker-similarity and target-activity
auxiliaries remain available if time allows.)*

Original wording follows for the record.

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

**UNBLOCKED 2026-09-03, AND THE FIRST ANSWER IS NEGATIVE.** The WeSep pretrained
checkpoint was rendered and scored on `sir0_val` `both` (n=103), giving the second
system this milestone was waiting for. **There is no rank inversion.** All five
instruments — SI-SDR, SIR, offline WER, DNSMOS `OVRL` and the live judge — rank
WeSep above our baseline, and the judge's margin is the widest of them: 59.3 % of
judge headroom captured against our 10.5 %. The only dissenting column anywhere is
SAR, which favours our baseline (10.34 against 8.26) because WeSep removes far more
interferer and pays in artefact.

**What this does and does not settle.** It settles that the metric does not invert
conventional rankings on a pair separated by ~25 points of WER — no metric would.
It does not settle the claim the thesis actually makes, which is about systems close
enough for the instruments to disagree. So M5's per-band gate is now the experiment
that matters, and it carries a measurement risk that must be stated before it runs:
its ORACLE ceiling is 2.2 points of LCF-WER (59.1 -> 56.9) against a judge whose
aggregate SEM is ~0.5 points. A realistic gate may not clear the metric's own noise
floor, in which case a null is uninterpretable rather than informative. Decide the
acceptable effect size and the repeat count k BEFORE running it.

**Superseded (2026-09-02): "blocked on a second system."** Kept because it was the
reasoning that prioritised the WeSep row, and that prioritisation was correct.

- [ ] Baseline and second model both scored on LCF-WER / ICR / NRR
- [ ] Both scored on SI-SDR / DNSMOS-P.835 + P.808 / offline WER on the same trials
- [ ] All anchors present: **audio floor and audio ceiling** (text anchors cut
      2026-09-03 — the text path fails the latency budget, so it is excluded rather
      than scored; see M4)
- [X] ~~**≥1 off-the-shelf pretrained TSE system** scored alongside. No training
      cost, and it is what stops the divergence claim from resting on n=2~~
      **DONE 2026-09-03: WeSep `tfmap_context_causal_100`** (REAL-TSE fork, Wang et
      al. 2024), 103 trials, rendered through the shared `src/estimates/runner.py`
      so it differs from ours by its model and nothing else. Out of domain by its
      own config (clean anechoic Libri2Mix, `noise_prob 0`, SI-SDR against the dry
      source) and **it still beats our baseline on every metric except SAR.**
      Scored into row 5 of the results table in `project-state.md`.
- [ ] AMI trial set built and the benchmark extended to it, if it survived
- [ ] Latency reported per modality
- [ ] **B13 — every number broken out condition by condition, no combinations.**
      Primary: which voice is louder, how much they overlap, whether the target
      speaks. Secondary: T60 above/below budget, gender, `enrollment_guard` tier.
      100 trials per bucket minimum. **A headline aggregate must never appear alone**
- [ ] **B11 — latency decay curve**, the same model scored at 100/200/300/400/500 ms
      rather than a single pass/fail at 300 ms
- [ ] **A1 ablation, if time allows** — the same architecture trained against a
      direct+early reference instead of the full reverberant one, to measure the
      artefact-versus-residue trade rather than assume it. Needs the second reference
      stem rendered, which is why M0 records the RIR per trial. Cut before M5 is cut

**Proof:** a table where ranking by SI-SDR / DNSMOS / offline-WER differs from
ranking by LCF-WER — or evidence that it doesn't.
**A negative result here is still a result**, and far better found now than in
week 12.

---

## M7 — Submitted · 2026-11-05

- [ ] Experiment freeze honoured (Oct 14)
- [ ] Every result traceable to config + commit hash + seed + date
- [ ] Every borrowed method cited
- [ ] Every deviation and cut recorded in the milestone decision logs
      (`decisions-m0.md` through `decisions-m4.md`)
- [ ] Approximate-ceiling caveat stated wherever AMI numbers appear
- [ ] Modality recorded on every judge result, and the cross-modality caveat
      (`docs/data/metric-definitions.md` §3.5) stated wherever audio and text rows
      appear in the same table
- [ ] **Reverberation stated as a known limitation** (A1). The reference is what the
      mic heard, so late reverb is never removed; it costs recognition accuracy and
      the write-up must say so rather than omit it
- [ ] **Divergence from WHAMR!'s direct-path reference noted** wherever the reference
      signal is described — different task, not a better choice
- [ ] Written, reviewed, submitted

---

## Report — chapter status

Added 2026-08-21. LaTeX skeleton builds (`cd report && latexmk -pdf report.tex`).
Structure is fixed; what follows is writing, not decisions. One line per item,
tick as written.

**Front matter**
- [ ] Title page: exact degree wording and submission month
- [ ] Declaration: official SU wording, copied not paraphrased
- [ ] Abstract, written last; decide whether the Afrikaans Opsomming is required
      (`frontmatter/abstract.tex` currently has two `Abstract` blocks)
- [ ] Nomenclature: extend as symbols and abbreviations are introduced

**Ch 1 Introduction**
- [ ] Problem statement; the live-model objective, not signal quality (spec note 10)
- [ ] Two-speaker boundary stated as a declared limit (`decisions-m0.md` 2026-08-14)
- [ ] Contributions list, metric first (spec notes 1 and 8)

**Ch 2 Literature Review** — source: `literature/review_synthesis.md`
- [X] ~~BSRNN lineage: MSS to PSE to TSE conditioning~~ draft in `litreview.tex`
- [ ] TSE conditioning survey; where TF-Map sits among the alternatives
- [ ] REAL-TSE Challenge as the anchor benchmark, and why we do not replicate it
      (spec note 8) — never present our numbers as comparable
- [ ] Metric critique: DNSMOS over-optimisation as the cautionary tale
      (`metric-definitions.md` §4)
- [ ] Novelty positioning — source: `literature/novelty-review-contrastive-phonetic-cue.md`

**Ch 3 Methodology** — source: `decisions-m0.md`, `decisions-m1.md`, `decisions-m2.md`
- [ ] Data construction: mixtures, rooms, levels, absent trials, VAD-measured overlap
- [ ] Reference signal is the full reverberant target (A1), with the consequence stated
- [ ] Architecture: the nine component subsections in `architecture.tex`
- [ ] Causal adaptation and the latency convention (`decisions-m1.md` 2026-08-18)
- [ ] Sizing: **7.19 M** against challenge scale 25-27 M, reported as deliberate
      (corrected from 7.16 M on 2026-08-24 — see M1)
- [ ] Objective: three terms, six deviations from CARTSE, DNSMOS rejection recorded
- [ ] **`L_gain`, the fourth term — Deviation 7, OURS not CARTSE's.** Four points,
      in order: (a) why needed — sir0 muted to 22.4 dB below the mixture with 95 %
      of its improvement from the absent half, because `L_pres` structurally cannot
      see a mute; (b) why it does not undo Deviation 1 — that bug was *unbounded
      one-directional* reward, this is symmetric and minimised at correct level;
      (c) ±3 dB deadzone, and why percent is the wrong unit (10 % = 0.83 dB);
      (d) per-trial anchor, and why a dataset mean was rejected (automatic gain
      control, and it contradicts A1). Caveats to state, not bury: RMS not BS.1770,
      and present crops only. `decisions-m2.md` 2026-08-27.
- [ ] Training setup — objective, chunk, batch, seed and schedule now exist in
      `bsrnn_baseline.yaml`, `decisions-m1.md` (chunk, batch) and `decisions-m2.md`
      (objective, schedule); still unlogged are the
      `batch_size` 3-vs-12 resolution and the compute actually used
- [ ] Metric definition: LCF-WER, ICR, NRR; judge protocol and modality recording

**Ch 4 Experiments**
- [ ] Protocol: config, commit hash, seed and date on every run
- [ ] Causality verified by measurement, and why no chunk-stitching is used
      (2026-08-24) — with the cold-vs-warm context gap stated as a limitation
- [ ] Band-plan ablation (six candidates)
- [ ] `w_m` ablation, the 0 arm required
- [ ] **`w_g` ablation, 0 arm required** — the control is what proves the term did
      the work rather than the extra epochs
- [ ] **The sir0 conditioning result** — enrolment swap moved the output 18 % → 39 %.
      Say why it is only interpretable on `sir0`: on `mid`, 90 % of trials had the
      target louder, so a model could look conditioned while tracking the loud voice
- [ ] **`L_gain`'s derivation, not just its value** — `w_m`'s "30 % of `|L_pres|`"
      rule does not transfer (`L_gain` is ~0 at pass-through). Show the four anchors
- [ ] **`L_MR` rewards the mute** (measured 2026-08-28). Correct the 2026-08-20
      claim that it "pins the output gain" at source — the correction is why a
      fourth term was needed
- [ ] Latency decay curve at 100/200/300/400/500 ms (B11)
- [ ] Judge harness: exact model ID, exact prompt, input modality, run date

**Ch 5 Results**
- [ ] Conventional metrics (M3), then LCF metrics (M6), on the same trials
- [ ] Anchors: audio floor, audio ceiling, text floor, text ceiling
- [ ] Per-condition breakdown, no combinations, >=100 trials per bucket (B13).
      A headline aggregate must never appear alone
- [ ] At least one off-the-shelf pretrained TSE system scored alongside

**Ch 6 Conclusion**
- [ ] Limitations: reverberation never removed (A1); two-speaker mixtures, never
      "conversation"; approximate-ceiling caveat wherever AMI appears; the
      cross-modality caveat wherever audio and text rows share a table

**Bibliography** (`report/mybib.bib`)
- [X] ~~`luo2023music`, `yu2023high`, `zhang2025multi` — verified against the
      published venues, not the arXiv preprints~~
- [ ] `zhang2025multi` page range, from IEEE Xplore
- [ ] Still to add: REAL-TSE overview, CARTSE, PS4, Zmolikova et al. 2023,
      LibriSpeech, WHAM!/WHAMR!, Silero VAD, Whisper text normaliser
- [ ] Two entries inherited from an earlier project remain (`hunter2007matplotlib`,
      `chatgpt2024`) — keep only if actually cited

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

See `docs/decisions/decisions-m0.md`, 2026-08-07:

- Replication of REAL-TSE online-track baselines
- Use of the official challenge scoring pipeline as primary evaluation
- On-device / parameter / MAC budget study and quantisation work
- Matching any published challenge number
