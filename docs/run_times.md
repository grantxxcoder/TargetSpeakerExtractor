# Run times

Wall-clock for every job slow enough to plan around. **Anything over a minute gets
a row.** Purpose: knowing in advance whether a step is a coffee break or an
overnight run, and having a real number for the write-up instead of a guess.

Rows below the marker are **written automatically** by `src/run_log.py` when a
script finishes — newest first. Runs under a minute write nothing, which is what
keeps this file short enough to be worth reading. Set `RUN_LOG=0` to suppress.

Machine unless stated: laptop, Intel i5-1135G7, **4 physical cores / 8 threads**,
15 GB RAM, NVMe. No usable GPU (`torch.cuda.is_available() == False`).
Hyperthreading buys ~10 % here — measured, 4 workers 111 s vs 8 workers 99 s.

| date | command | scope | wall | rate |
|---|---|---|---|---|
<!-- rows appended below by src/run_log.py -->
| 2026-08-30 | `scripts/train.py --split sir0` | 1,989 trials x 22 epochs (early-stopped, best 11), sir0 | 3.4 h | batch 3, cuda (T4), **549 s/epoch**. Arm C: bank + remix. Copied by hand from the Kaggle session's own run_times.md before deleting the bundle. |
| 2026-08-30 | `scripts/train.py --split sir0` | 1,989 trials x 25 epochs, sir0 | 3.9 h | batch 3, cuda (T4), **568 s/epoch**. Arm A: remix only. Copied by hand from the Kaggle session's own run_times.md before deleting the bundle. |
| 2026-08-30 | `scripts/render_enrollment_bank.py --split sir0_train --variants 4` | 1,889 trials x 4 variants (100 already done, skipped) | 46 min | 8 workers, 16 kHz PCM_16. Full split. |
| 2026-08-30 | `scripts/make_estimates.py` | 200 trials, sir0 | 25 min | cpu, whole-clip |
| 2026-08-30 | `scripts/diagnose_cue.py` | 200 crops, sir0 | 14 min | cpu, batch 4 |
| 2026-08-30 | `scripts/diagnose_cue.py` | 24 crops, sir0 | 80 s | cpu, batch 4 |
| 2026-08-30 | `scripts/diagnose_cue.py` | 24 crops, sir0 | 81 s | cpu, batch 4 |
| 2026-08-30 | `scripts/render_enrollment_bank.py --split sir0_train --variants 4` | 1,889 trials x 4 variants | 46 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-30 | `scripts/render_enrollment_bank.py --split sir0_train --variants 4` | 100 trials x 4 variants | 2 min | 8 workers, 16 kHz PCM_16. TIMING RUN: `--limit 100` of 1,989. |
| 2026-08-29 | `scripts/train.py --split sir0 --epochs 50 --resume` | 1,989 trials x 15 epochs (resumed at 10, early-stopped at 24), sir0 | 2.20 h | batch 3, cuda (Tesla T4), **527.8 s/epoch**. AMP. Requested 50, ran 15: early stop, patience 10, best epoch 14. Kaggle; expanded by hand from the session's own run_times.md, which recorded the same run as `15 epochs / 2.2 h / 528 s per epoch`. |
| 2026-08-29 | `scripts/train.py --split sir0` | 1,989 trials x 10 epochs, sir0 | 1.45 h | batch 3, cuda (Tesla T4), **523.0 s/epoch**. AMP + `chunk_s` 4.008. **7.21x** the fp32 run's 3773 s/epoch. |
| 2026-08-28 | `scripts/train.py --split sir0` | 1,989 trials x 10 epochs, sir0 | 10.5 h | batch 3, cuda (Tesla T4), 3773 s/epoch. The `w_g`=1.69 run. fp32, pre-speed-fix. |
| 2026-08-28 | `scripts/train.py --split sir0` | 1,989 trials x 2 epochs, sir0 | 17 min | batch 3, cuda (Tesla T4), **505.7 s/epoch**. First run with `chunk_s` 4.008 + `amp: true`. **7.66x** the 2026-08-27 row's 3875 s/epoch, same batch, same GPU. Kaggle. |
| 2026-08-28 | `scripts/make_estimates.py` | 200 trials, sir0 | 33 min | cpu, whole-clip |
| 2026-08-28 | `scripts/make_estimates.py` | 20 trials, smoke | 2 min | cpu, whole-clip |
| 2026-08-28 | `scripts/derive_w_g.py` | 200 crops x 4 systems, sir0 | 6 min | cpu, batch 4 |
| 2026-08-27 | `scripts/train.py --split sir0` | 1,989 trials x 8 epochs, sir0 | 8.6 h | batch 3, cuda (Tesla T4), 3875 s/epoch. Kaggle. Added by hand: the row was written into the notebook log, not this file. |
| 2026-08-27 | `scripts/render_trials.py --split sir0_val` | 200 trials rendered | 2 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-27 | `scripts/render_trials.py --split sir0_train` | 264 trials rendered **(failed)** | 3 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-26 | `scripts/render_trials.py --split sir0_val` | 200 trials rendered | 3 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-26 | `scripts/render_trials.py --split sir0_train` | 1,989 trials rendered | 31 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-26 | `scripts/train.py --split sir0` | 1,989 trials x 10 epochs, sir0 | 5.2 h | batch 6, cuda, 1869 s/epoch. w warmup 4+3, tfmap_scale 16. Kaggle T4; row copied from the session's own run_times.md. |
| 2026-08-26 | `scripts/render_trials.py --split sir0_val` | 128 trials rendered | 81 s | 8 workers, 16 kHz PCM_16 |
| 2026-08-26 | `scripts/render_trials.py --split sir0_train` | 1,989 trials rendered | 18 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-25 | `scripts/train.py --split mid` | 2,000 trials x 10 epochs, mid | 5.4 h | batch 6, cuda, 1950 s/epoch. w warmup 4+3. Kaggle T4; row copied by hand from the session output. |
| 2026-08-25 | `scripts/train.py --split smoke` | 50 trials x 2 epochs, smoke | 11 min | batch 3, cpu, 337 s/epoch |
| 2026-08-25 | `scripts/train.py --split mid` | 2,000 trials x 2 epochs, mid | 58 min | batch 6, cuda, 1739 s/epoch |
| 2026-08-25 | `scripts/train.py --split smoke` | 50 trials x 2 epochs, smoke | 10 min | batch 3, cpu, 303 s/epoch |
| 2026-08-25 | `scripts/make_kaggle_bundle.py` | 2,200 trials staged + zipped, `mid` | 3.5 min | zip-only: audio was already staged (0 copied, 8,800 current). A cold run adds the ~2.7 GB copy. Added by hand: the script does not use `run_log.timed`. |
| 2026-08-24 | `scripts/train.py --split smoke` | 50 trials x 70 epochs, smoke | 4.8 h | batch 3, cpu, 246 s/epoch |
| 2026-08-24 | `scripts/train.py --split smoke` | 50 trials x 2 epochs, smoke | 9 min | batch 3, cpu, 263 s/epoch |
| 2026-08-24 | `scripts/train.py --split smoke` | 50 trials x 30 epochs, smoke | 2.3 h | batch 3, cpu, 277 s/epoch |
| 2026-08-24 | `scripts/train.py --split smoke` | 50 trials x 5 epochs, smoke | 22 min | batch 3, cpu, 268 s/epoch |
| 2026-08-24 | `scripts/train.py --split smoke` | 50 trials x 1 epochs, smoke | 4 min | batch 3, cpu, 243 s/epoch |
| 2026-08-18 | `src/models/workbook.ipynb` — `measure_empty_crops()` | 3,000 trials x 3 epochs = 9,000 target crops, `train` | 3 min | ~21 ms/crop, 1 windowed read per crop, single-threaded. Added by hand: notebook cell, not a `run_log.py` script |
| 2026-08-16 | `scripts/render_trials.py --split eval_public` | 500 trials rendered | 2 min | 8 workers, 16 kHz PCM_16 |
| 2026-08-16 | `scripts/measure_vad_impact.py` | 2,000 utts x 8 settings + 400 trials | 14 min | 8 workers |
| 2026-08-16 | `scripts/measure_vad_impact.py` | 2,000 utts x 8 settings + 400 trials **(failed)** | 14 min | 8 workers |
| 2026-08-16 | `scripts/build_manifest.py --split train` | 19,938 trials | 2 min | headers only, no audio |
| 2026-08-16 | `scripts/measure_vad_impact.py` | 2,000 utts x 8 settings + 400 trials | 16 min | 8 workers |
| 2026-08-15 | `scripts/screen_noise_speech.py` | 28,000 clips / 82 h | 25 min | 193x realtime, 8 workers |
| 2026-08-15 | `build_vad_index.py` | 137,876 utts / 475 h | 2.1 h | 222x realtime, 8 workers |
| 2026-08-14 | `build_manifest.py --split train` | 20,000 trials | 58 s | headers only, no audio |
| 2026-08-15 | `pytest tests/ -q` | 74 tests | 5 s | under threshold, kept for reference |

## Not yet run

Projections, kept deliberately separate from the measurements above. **Never quote
one of these as a measured figure.**

| command | scope | projected | basis |
|---|---|---|---|
| `render_trials.py --split train` | 19,938 trials | ~78 min | 100 trials measured at 23.4 s, 8 workers |
| `render_trials.py`, all six splits | 21,208 trials / ~27 GB | ~83 min | same rate, 1.26 MB per trial measured |

Move a row up to the table above once it has actually run — the script does that
for itself; delete the projection by hand.
