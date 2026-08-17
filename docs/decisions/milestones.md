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
(`src/exploratory/data_setup.ipynb`). **No audio exists yet.** **Every data decision is
now made** — all of group A and all thirteen B items. The only open question is **C2**,
how hard the task should be, which needs the supervisor and blocks nothing meanwhile.
What is left is implementation: B12's two PRs, then one manifest rebuild, then the
renderer.

**Schedule reality check.** The target is Aug 20 and no code is written yet. What
makes it survivable is that manifest rebuilds take 58 s. **Superseded 2026-08-15:**
this paragraph assumed training audio was generated on the fly, so only `val` and
the two eval splits (~1,200 trials) needed rendering. All ~21,200 trials are now
rendered to disk (~26 GB) — B7 had already turned off the per-epoch variety that
justified on-the-fly, so it was recomputing identical audio every epoch. Render
*after* B2's rebuild, never before. The pilot calibration below is still the item
most likely to slip, and it depends on C2.

Done:
- [X] ~~Speaker-disjoint train / val / eval splits~~
- [X] ~~Seed set and logged; generation config in `experiments/configs/`~~
- [X] ~~Manifest generator: speakers, timing, levels, rooms, noise, enrollment~~
- [X] ~~Enrollment segments ≥5 s, from a different recording than the mixture~~
- [X] ~~Manifest audit §1–§7: timing, levels, rooms, enrollment, noise, absent trials~~

Decisions — all closed except C2, which needs the supervisor:
- [X] ~~**A1** reference signal, **A5** tail padding, **A6** clipping. Group A closed~~
- [X] ~~**B1** overlap range is a dial setting, **B2** VAD, **B4** eval absent trials,
      **B5** text normalisation, **B6** trial count, **B7** resampling, **B11** latency
      reporting, **B13** stratified reporting~~
- [X] ~~**B9** — 50 % both / 25 % absent / 25 % target-only, and a variable
      `target_activity_ratio`. Sets B4's eval fraction at 0.25~~
- [X] ~~**B12 — generator controllability.** Architecture decided 2026-08-13.
      Implementation outstanding: PR1 sampler, PR2 wire-in~~
- [X] ~~**B10** — three enrollment tiers (`book` / `chapter` / `utterance`) recorded
      per trial; eval pools redrawn to balance the tier mix. Executes B8's own
      documented contingency rather than reversing it~~
- [ ] **C2** — how hard the task should be (floor WER); needs the supervisor. Blocks
      nothing that can be done meanwhile

B12 implementation, before the rebuild:
- [X] ~~**PR1** — `src/data/sampling.py`: `draw()`, `resolve()`, unit tests. No wiring~~
- [X] ~~**PR2** — wire into `build_manifest.py`, add the `regime` column, raise the
      `t60_s` floor to 0.25, eval splits skip regimes. Acceptance test passed: with
      the config unchanged, all six splits (21,270 rows) are byte-identical once the
      new `regime` column is ignored — `scripts/check_manifest_parity.py`~~
- [ ] **Narrow `overlap_ratio` in the `base` regime** — left out of PR2 on purpose.
      `difficulty-dial.md` §3 puts it last and requires supervisor agreement plus a
      `decisions.md` entry, because its 0.7 ceiling is deliberately matched to
      REAL-TSE. One config line once approved
- [X] ~~Decide whether `data/manifests/` is tracked in git — still open, and now
      overdue: PR2 changed the schema. `.gitignore:222` claims manifests are
      tracked, `/data/` on line 223 untracks them; none are in git~~

Manifest rebuild — **done 2026-08-14 (PR3)**. Speaker-identity leak 0.795 -> 0.508,
P(absent | no overlap) 1.000 -> 0.500, all 1,172 speakers now reach the present pool:
- [X] ~~**B9** — add `target_only_fraction`; let `target_activity_ratio` vary.
      Silent-target trials were detectable at AUC 1.000; a target speaking
      uninterrupted never occurred. Both fixed~~
- [X] ~~**B10** — three-tier enrollment guard with `enrollment_guard` recorded per
      trial; assert enrollment and mixture never share an utterance~~
- [X] ~~**B10** — `make_splits.py`: eval pools redrawn, stratified by guard tier as
      well as sex. Weakest tier was 8/20 vs 3/20, now 6 vs 5~~
- [X] ~~**B4** — eval splits carry the same composition as train (absent 0.25)~~
- [X] ~~Define the **interruption** condition and add it as a column — an interferer
      onset strictly inside a target utterance. 53.2 % of both-speaking trials~~
- [X] ~~Re-run the leak scoreboard **per regime as well as pooled** — done; 0.497 /
      0.505 within base / hard~~
- [ ] **Scope decision 2026-08-14 consequences**: decide whether the AMI check is
      restricted to <=2 active speakers or reported as an out-of-scope probe, and
      write the two-speaker limitation into the thesis. See `decisions.md`

Still unimplemented from `data-construction-parameters.md`:
- [X] ~~**`noise_speech_rejection`** — done 2026-08-16, and *not* in the renderer as
      planned: the screening pass already measured every clip, and the manifest is
      what names the noise clip, so the pool is filtered before selection. Drop any
      clip whose longest unbroken speech run reaches 0.5 s — 4.1 % of tr, 2.0 % of
      cv, 1.3 % of tt. Takes effect at PR2's rebuild~~
- [X] **B2 — voice-activity detection pass over the corpus** — **closed 2026-08-16**
      except the optional PR3 below. Cached alongside the
      utterance index, so overlap is measured from where speech actually is. Detector:
      Silero VAD, `silero-vad` 6.2.1, pinned. Measured, reproducibly, 2026-08-16:
      files are 86.2 % speech, overlap overstated ~25 %, per-trial error up to 0.270
      so no correction factor can fix it. Changes every overlap figure, so it belongs
      *before* the rebuild, not after
  - [X] ~~**PR1** — `src/data/vad.py`, `scripts/build_vad_index.py`, `vad:` config
        block, 30 unit tests, and `scripts/measure_vad_impact.py`. Result written
        2026-08-16 (16 min, sanity passed) to
        `experiments/results/2026-08-15-vad-impact/`. `build_manifest.py` untouched~~
  - [X] ~~**Run the index**: `scripts/build_vad_index.py` — done 2026-08-15, 2.1 h,
        137,876 utterances, 0 failures, 86.4 % speech. `data/index/vad_segments.csv`.
        `scripts/screen_noise_speech.py` ran the same day, 25 min, 28,000 clips~~
  - [X] ~~**PR2** — done 2026-08-16, branch `m0-b2-pr2-vad-wire`. Wired into
        `build_manifest.py`; all six manifests rebuilt (train 2 min). Overlap labels
        were wrong on 95.5 % of both-trials (mean 0.073, max 0.485) and are now exact.
        Real speech overlap rose 0.212 -> 0.275 because `best_onset` finally hits the
        requested amount. Leak scoreboard unchanged (AUC 0.648 -> 0.651, speaker prior
        0.508 -> 0.503). `n_failed` 50 -> 62 of 20,000, so the 0.78 ceiling stands.
        `check_manifest_parity.py` fails on all six, as designed~~
  - [X] ~~**`target_activity_ratio` ceiling.** Lowered 0.85 -> 0.78 on 2026-08-16 in
        both the global band and `base`, with a decisions.md entry. 0.85 of *speech*
        needs 0.988 of the window filled with audio and realised footprint tops out
        at 0.946, so it was unreachable; unreachable draws are dropped silently and
        would have thinned the talkative end of the band. REAL-TSE's ~0.75 still sits
        inside. **0.78 is near the edge, not inside it — check PR2's `n_failed` and
        lower again if it is material**~~
  - [ ] **PR3 (optional)** — enrollment offset. `enroll_offset` is drawn uniformly
        anywhere in the file and `long_enough` filters on file duration, so a 5.2 s
        file with 1.2 s of leading silence yields a "5 s enrollment" holding 4 s of
        voice, against B10/A1's >=5 s. Separate PR: it changes enrollment quality,
        not overlap measurement
- [ ] `length_mode`

Renderer — **written 2026-08-16** (`src/data/render.py` + `scripts/render_trials.py`),
**not yet run at full scale**. 100 trials measured at 23.4 s / 8 workers, so the
whole set is **~83 min and ~27 GB** — the "unknown" row in `run_times.md` is closed
and it is an hour and a half, not the overnight job feared here:
- [X] ~~Manifest row → audio: RIRs, levels at BS.1770, noise wrap, clip ceiling~~
- [X] ~~Five stems per trial: mixture, clean target, enrollment, both texts~~
- [X] ~~Clean target = target × its own RIR, no interferer, no noise (A1). Room
      columns are already per trial and the RIR is a pure function of them, so the
      direct+early ablation stem re-renders without re-drawing rooms~~
- [X] ~~Tail padding past the last speech by at least the room's decay (A5)~~
- [X] ~~Exact verbatim ground-truth text for **both** target and interferer, in
      each trial's `meta.json`~~
- [X] ~~Guards: stem <400 ms breaks BS.1770; assert on a silent stem~~
- [X] ~~Transcripts cut to match any audio truncation — no truncation occurs, and
      `lay_track` raises if audio would run past the window~~
- [ ] **Run it.** ~83 min for all six splits. Resumable; re-issue the same command
- [ ] Three under-specified points were interpreted, not decided — noise covering
      A5's tail, the `noise_only` level anchor, and the enrollment's level. Worth a
      supervisor glance. See `decisions.md` 2026-08-16

Notebook and verification:
- [X] ~~§7.5 — leak scoreboard, **before** the rebuild so it is a before/after — done
      2026-08-16 for B2 PR2, but run standalone against the backed-up pre-rebuild
      manifests rather than in the notebook. Numbers in `decisions.md`. The notebook
      cells themselves still print the old figures~~
- [ ] **EDA per parameter** — plot each parameter's realised distribution against
      the intended one (raised 2026-08-12, needed to verify B12)
- [ ] Revise §2, §7 and the final health checks after the rebuild; §3–§6 survive

Pilot calibration, before freezing any range:
- [X] ~~Listen to 40 trials — done 2026-08-17, all 40 judged correct, nothing
      re-rendered. Rules out a systematic renderer fault, which is what the measured
      A1–A6 properties could not do. Weight it honestly: subjective, one listener,
      40 of 21,208 trials (0.19 %), and the trial ids were not recorded, so the same
      40 cannot be re-listened to~~
- [ ] Floor and ceiling WER measured; aim for a 60–80 % floor. **Now the M0 blocker**
      and what C2 needs, since the listen is closed

Housekeeping:
- [X] ~~`requirements.txt` — added 2026-08-15 (`e1e2436`), 38 lines, versions pinned
      exactly rather than loosely, since the VAD weights define what "overlap" means.
      No `pyproject.toml`; not needed while this is a scripts-and-modules repo~~
- [X] ~~`.gitignore`: `data/` was unanchored and matched `docs/data/` too, so all 8
      files there were untracked. Anchored to `/data/` 2026-08-13~~

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
- [ ] **Write B4's scoring rule into `metric-definitions.md`** — decided
      2026-08-13: absent trials are excluded from the main score and reported as
      their own invented-speech row, never folded into the headline. The decision is
      made; the document still defines nothing for a trial with no reference text
- [ ] **Pin B5's normaliser** — Whisper `EnglishTextNormalizer`, applied identically
      to both sides, frozen before the first judge result and never adjusted per
      system (decisions.md 2026-08-13)
- [ ] Judge harness: fixed prompt, fixed response ASR, pinned model IDs, k≥3
      repeats, **input modality recorded per trial**, cost/compute logging
- [ ] Judge decided and its cost model resolved — closed API (money) or
      self-hosted open-weight (GPU-hours contending with training quota).
      **Currently unresolved**; this is the gating question for the whole
      milestone
- [ ] Trial-set size fixed to that budget, on a spreadsheet, before the harness
      is finalised. **Floor is 200 scored trials** (B6/B13: 100 per bucket across a
      two-way split); 500 are generated, and scoring more later extends the set
      rather than replacing it
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
- [ ] Every deviation and cut recorded in `docs/decisions/decisions.md`
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
