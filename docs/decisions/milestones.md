# Milestones

**Rewritten 2026-08-07** to a build-first ordering: data → baseline model →
evaluate it conventionally → define the metric → second model → compare on
the metric → report.

Submission 2026-11-05; hard freeze on new experiments **2026-10-14**.

**Where the project is, 2026-08-24.** M0 data is built (21,208 trials, 27 GB) and
M1 is functionally complete ten days early — model, objective, training loop and
a proven resume. M2 has started ahead of schedule but only on the 50-trial smoke
split, where the model has found a degenerate attenuate-and-gate solution rather
than separating. **The two things actually blocking progress are compute** (no
usable GPU; `--split full` will not run on this laptop) **and M0's floor/ceiling
WER calibration**, which C2 needs. Neither is a code problem.

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

**Status 2026-08-24** (was 2026-08-13, which said "no audio exists yet" — it
does). Manifests exist for all six splits and have been audited
(`src/exploratory/data_setup.ipynb`), and **all 21,208 trials are rendered**:
63,624 files, 27 GB, 105.4 h of audio, 0 render failures, 40 trials checked by
ear. **Every data decision is made** — all of group A and all thirteen B items.
The only open decision is **C2**, how hard the task should be, which needs the
supervisor. What remains is the floor/ceiling WER calibration below, the
per-parameter EDA, and the notebook revision.

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
      `decisions-m0.md` entry, because its 0.7 ceiling is deliberately matched to
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
      write the two-speaker limitation into the thesis. See `decisions-m0.md`

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
        both the global band and `base`, with a decisions-m0.md entry. 0.85 of *speech*
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

Renderer — **written 2026-08-16** (`src/data/render.py` + `scripts/render_trials.py`)
and **run at full scale 2026-08-16/17: 3.2 h, 27 GB, 0 failures**. The ~83 min
projected from a 100-trial sample was out by 2.3x, which is why that sample is
now treated as too small to extrapolate from rather than as a measurement — the
same caution now applied to the smoke epoch times in M2:
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
- [X] ~~**Run it.** ~83 min for all six splits. Resumable; re-issue the same command~~
- [ ] Three under-specified points were interpreted, not decided — noise covering
      A5's tail, the `noise_only` level anchor, and the enrollment's level. Worth a
      supervisor glance. See `decisions-m0.md` 2026-08-16

Notebook and verification:
- [X] ~~§7.5 — leak scoreboard, **before** the rebuild so it is a before/after — done
      2026-08-16 for B2 PR2, but run standalone against the backed-up pre-rebuild
      manifests rather than in the notebook. Numbers in `decisions-m0.md`. The notebook
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

**Status 2026-08-24. M1 is functionally complete, ten days ahead of the Sep 3
target.** Every architecture decision is logged in `decisions-m1.md`
(2026-08-18 to 08-20). `scripts/train.py` runs, early-stops, checkpoints, plots,
writes `history.csv` + `meta.yaml`, and **resume is proven** — see below. The
proof item that gates this milestone is therefore closed.

One checklist item and one open question remain, neither blocking M2: unit tests
for the model modules (the loss has 30, nothing else has any), and the
cold-vs-warm context mismatch.

- [X] ~~Causal BSRNN + TF-Map extractor implemented — `src/models/{stft,bands,
      modules,conditioning,bsrnn}.py`. Cited in-file: Luo & Yu (TASLP 2023) in
      `bands.py` and `modules.py`, Yu et al. (Interspeech 2023) in `bands.py`,
      Zhang et al. (ICASSP 2025) in `conditioning.py` and `bsrnn.py`~~
- [X] ~~STFT window/hop chosen against the ~200–300 ms budget — 512/128,
      `center=False`, manual overlap-add, `lookahead_frames` knob. Latency
      convention and the rejection of the challenge's 100 ms cap in
      `decisions-m1.md` 2026-08-18. Effective future dependency then *measured*
      at 23.9 ms rather than assumed (2026-08-19)~~
- [X] ~~Model deliberately sized down and reported as such — `decisions-m1.md`
      2026-08-19, against the REAL-TSE causal baselines' 25–27 M. **Realised
      figure 2026-08-24: 7,189,644 (7.19 M), not the pre-conditioning 7,156,234
      (7.16 M) — `decisions-m1.md` 2026-08-24.** The whole 33,410 gap is
      `SubbandNorm` (104,326 vs 70,916), and the 08-19 entry *predicted* it at
      ~104,000 / ~33 k / under 0.5 % — measured +326 off, 0.46 %. The 3.5x
      reduction claim is unaffected. **Quote 7.19 M in the thesis**~~
- [X] ~~Target-absent training and channel-gap enrollment augmentation
      (Li & Seki, 2026) — target-absent is `L_abs` + the loader's `crop_absent`
      (`decisions-m1.md` 2026-08-20); channel-gap enrollment EQ is
      `src/data/render.py:152`, cited in-file~~
- [X] ~~YAML config committed — `experiments/configs/bsrnn_baseline.yaml`. No
      training hyperparameter is a command-line flag; `--epochs` overrides only~~
- [X] ~~Seed set and logged — `train.py` seeds torch and numpy from the config
      *before* the model is built, so weight init is reproducible too, and the
      seed lands in every `meta.yaml`~~
- [X] ~~**Checkpoint/resume proven — 2026-08-24.** Resumed the 30-epoch smoke
      run from epoch 28 and ran on. State verified restored, not reinitialised:
      scheduler `_last_lr` stayed at **0.00025** rather than resetting to the
      config's 0.0005, and AdamW's `step` counter read **480** = 464 through
      epoch 28 plus 16 for epoch 29. A silent failure would have shown `step`
      restarting near 16. `best_val` carried across and improved to −12.9296,
      so the resumed run wrote its own checkpoint~~
- [ ] **Unit tests for the model modules.** `tests/test_losses.py` collects 30;
      there are **none** for `stft.py`, `bands.py`, `modules.py`,
      `conditioning.py` or `bsrnn.py`. "Training infrastructure trustworthy" is not met by a loop that
      runs. The causality property below is the obvious first test to lift

Built alongside, not on the original list:
- [X] ~~Objective implemented and its anchor measured — `src/models/losses.py`,
      three terms and six deviations from CARTSE (`decisions-m1.md` 2026-08-20);
      do-nothing anchor over 300 crops in
      `experiments/results/2026-08-20-loss-anchor/`, which is where `w_m = 9.62`
      comes from~~
- [X] ~~`scripts/measure_train_cost.py` — per-batch-size peak RSS and step time
      in a subprocess each, after `systemd-oomd` killed VSCode on 2026-08-24~~
- [X] ~~`scripts/pass_a_test_case_through.py` — one val trial through a
      checkpoint, estimate audio + full provenance, loss beside the
      pass-through anchor for that same crop~~
- [X] ~~**Loss floor derived and verified 2026-08-24: exactly −30.** All three
      terms are floored (`L_pres`, `L_abs` at `10log10(tau)`; `L_MR` at 0) and
      the outer weights are convex, so the bound is −30 for **any** `w` in [0,1]
      and any `w_m` — only `tau` moves it. Do-nothing anchor is −2.24, so the
      usable range is −2.24 → −30~~
- [X] ~~**Causality verified by measurement 2026-08-24, not assumed.** Appending
      later audio leaves earlier output unchanged (1.68e-08), so one full-length
      pass is what streaming emits and **no chunk-stitching is needed** —
      concatenating independent 4 s chunks is worse, reinjecting the
      `n_fft - hop` = 384-sample (23.4 ms) overlap-add tail at every seam
      (4.37e-03 max, rel L2 1.04e-02)~~
- [ ] **Train/inference context mismatch, found 2026-08-24 — open question.**
      Causal is not context-free: training only ever feeds cold-start 4 s crops,
      while deployment hands the model unbounded warm state. On the epoch-4 smoke
      checkpoint the same window scores 30.6 % rel L2 apart cold vs warm, though
      `total` moves only 0.005 (−2.4197 vs −2.4150). Re-measure on a converged
      checkpoint: if the gap grows it argues for warm-state or longer-context
      training; if it shrinks it is retired with evidence. Needs a
      `decisions-m1.md` entry either way

**Proof:** a 1-epoch run that completes, is killed, and resumes cleanly.
**Why resume is a checklist item and not an implementation detail:**
discovering it is broken at hour 11 of a 12-hour Kaggle session costs a week.

---

## M2 — Baseline trained · target Sep 17 (weeks 5–6)

**Status 2026-08-24. Started early: the loop works on `smoke`, nothing is
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

- [ ] **Converged checkpoint from conventional training** (SI-SDR +
      multi-resolution STFT). Smoke only so far
- [X] ~~Training curves and final losses logged with config, commit hash, seed,
      date — `log_results()` writes `meta.yaml` + per-epoch wide `history.csv`
      (train and val on one row, plus `lr`, which is what distinguishes a plateau
      from a scheduler step). Never raises: a logging bug must not discard a
      finished run~~
- [ ] **Compute is the blocker, not the code.** 15.7 GB RAM with VSCode open is
      not enough — `systemd-oomd` killed the editor on 2026-08-24 before training
      started. `requirements.txt` pins a CPU torch. The one measured row in
      `run_times.md` is **277 s/epoch** at batch 3 over 50 trials on CPU (2.3 h
      for 30 epochs — the 243 s/epoch row is a 1-epoch run and includes startup,
      so prefer the 30-epoch figure). Smoke timing: **must not be extrapolated**
      to 19,938 trials. Server-class
      compute or Kaggle is required, which makes the M1 resume proof urgent
- [ ] **`batch_size` is still 12-on-paper, 3-in-config.** `decisions-m1.md`
      2026-08-18 chose 12; the config says 3 with a comment saying it should be
      12, pending the GPU-memory measurement `measure_train_cost.py` exists to
      make. The absent-crop rate that sets `w` was derived at batch 12
- [ ] **Early stopping will not fire as configured.** `patience: 10` resets on
      *any* improvement, and best-val keeps creeping down by less than the noise
      (blocks of 5 epochs: -8.636, -9.495, -10.276, -12.301, -12.706, -12.900,
      then -12.9296 on resume — a 0.03 gain against a 0.42 sd). Left alone this
      run grinds to epoch 99 chasing `L_abs` to its floor. Needs a minimum-delta
      threshold before `--split full`, or the epoch budget is the only stop
- [ ] **Two ablations are declared but unrun** — the band plan (six candidates,
      `decisions-m1.md` 2026-08-18) and `w_m` (`ablate_w_m: [0.0, 2.89, 9.62]`,
      the 0 arm required). Both need the converged baseline first
- [ ] **CONFIRMED 2026-08-24: the smoke model attenuates, it does not separate.**
      Not a watch item any more — measured. Over 30 epochs on `smoke_val`:

      | | epoch 0 | epoch 29 | change | trend/epoch, last 10 |
      | --- | --- | --- | --- | --- |
      | `L_pres` | −4.181 | −5.273 | −1.092 | **+0.0365** (worsening) |
      | `L_MR` | 0.279 | 0.279 | −0.000 | +0.0004 (flat) |
      | `L_abs` | −5.753 | −23.905 | **−18.153** | −0.0963 (still improving) |

      **93 % of the total's movement is the absent half.** `L_MR` has not moved
      at all in 30 epochs. Paired on 200 identical crops, epoch 4 -> 28 gained
      only 0.462 total, and the *present half got worse by 0.320* while the
      absent half improved by 0.781. `L_pres` is now **+0.157 dB worse than
      passing the mixture through**, against −0.374 dB better at epoch 4.

      Mechanism: output RMS is **−24.9 dB below the input mixture** on present
      crops (−12.6 dB at epoch 4, so the attenuation is deepening). `L_pres` is
      scale-invariant *by design* (Deviation 1) so it cannot see attenuation at
      all; only `L_MR` penalises it, at weight `(1-w)*w_m = 5.21`, against
      `w = 0.458` on an absent branch whose optimum is reachable by outputting
      zero. Turning the volume down is therefore strongly net-positive.

      Where it sits among non-separating strategies on the fixed val set:

      | strategy | total | `L_pres` |
      | --- | --- | --- |
      | pass the mixture through | −2.288 | −6.152 |
      | all silence | −11.732 | 0.000 |
      | **model @ epoch 28** | **−12.900** | **−6.151** |
      | oracle-gated mixture (perfect VAD, zero separation) | −16.030 | −6.152 |

      The model's `L_pres` equals pass-through's to three decimals, which is
      what "attenuated mixture, no separation" looks like through a
      scale-invariant term. It has passed all-silence and is climbing toward the
      oracle-gated solution: **3.13 total still available with no separation at
      all.** Do not read further loss improvement as separation until `L_MR`
      starts falling — that is the term that cannot be fooled by gain.

      Open, and not to be fixed on smoke: 50 trials is too few to require
      separation, so a degenerate solution is the expected outcome here. Re-test
      on `full` before touching `w` or `w_m` — both are derived numbers
      (`decisions-m1.md` 2026-08-20) and changing them needs its own entry

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
      system (decisions-m0.md 2026-08-13)
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
- [ ] Every deviation and cut recorded in the milestone decision logs
      (`decisions-m0.md`, `decisions-m1.md`, and any later ones)
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

**Ch 3 Methodology** — source: `decisions-m0.md`, `decisions-m1.md`
- [ ] Data construction: mixtures, rooms, levels, absent trials, VAD-measured overlap
- [ ] Reference signal is the full reverberant target (A1), with the consequence stated
- [ ] Architecture: the nine component subsections in `architecture.tex`
- [ ] Causal adaptation and the latency convention (`decisions-m1.md` 2026-08-18)
- [ ] Sizing: **7.19 M** against challenge scale 25-27 M, reported as deliberate
      (corrected from 7.16 M on 2026-08-24 — see M1)
- [ ] Objective: three terms, six deviations from CARTSE, DNSMOS rejection recorded
- [ ] Training setup — objective, chunk, batch, seed and schedule now exist in
      `bsrnn_baseline.yaml` and `decisions-m1.md`; still unlogged are the
      `batch_size` 3-vs-12 resolution and the compute actually used
- [ ] Metric definition: LCF-WER, ICR, NRR; judge protocol and modality recording

**Ch 4 Experiments**
- [ ] Protocol: config, commit hash, seed and date on every run
- [ ] Causality verified by measurement, and why no chunk-stitching is used
      (2026-08-24) — with the cold-vs-warm context gap stated as a limitation
- [ ] Band-plan ablation (six candidates)
- [ ] `w_m` ablation, the 0 arm required
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
