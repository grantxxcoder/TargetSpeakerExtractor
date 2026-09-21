# Run times

Wall-clock for every job slow enough to plan around. **Anything over a minute gets
a row.** Purpose: knowing in advance whether a step is a coffee break or an
overnight run, and having a real number for the write-up instead of a guess.

Rows below the marker are **written automatically** by `src/run_log.py` when a
script finishes — newest first. Runs under a minute write nothing, which is what
keeps this file short enough to be worth reading. Set `RUN_LOG=0` to suppress.

Machine unless stated: laptop, Intel i5-1135G7, **4 physical cores / 8 threads**,
15 GB RAM, **5400 rpm SATA HDD** (Toshiba MQ04ABF1) -- corrected 2026-09-03, this
file previously said NVMe. The NVMe in the machine is a 238 GB Samsung holding
Windows (NTFS) and is not mounted on Linux; `/` and all project data live on the
spinning disk. Measured 81 MB/s sequential read, but small-file work is
seek-bound -- the 2026-09-03 bundle zipped 130,619 files at ~6.5 MB/s. **Every
row in this file was measured on the HDD**, so none of them are NVMe numbers.
No usable GPU (`torch.cuda.is_available() == False`).
Hyperthreading buys ~10 % here — measured, 4 workers 111 s vs 8 workers 99 s.

| date | command | scope | wall | rate |
|---|---|---|---|---|
| 2026-09-21 | `pytest tests/ -q` | 515 tests | 11 min | full suite after adding eval_by_case.py. Added by hand: pytest does not use run_log.timed |
<!-- rows appended below by src/run_log.py -->
| 2026-09-21 | `scripts/diagnose_cue.py` | 200 crops, sir0 | 17 min | cpu, batch 4 |
| 2026-09-21 | `scripts/measure_rtf.py` | --checkpoint models/model_sir0_wesepref-e10.pt --config experiments/configs/bsrnn_wesep_ref.yaml --chunk-ms 80 --threads 4 --device cpu --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-rtf-wesepref-e10 | 3 min |  |
| 2026-09-21 | `scripts/evaluate.py` | --split sir0_val --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-est-wesepref-e10 --metrics content --listener judge --judge-rpm 10 --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-eval-wesepref-e10-judge | 15 min |  |
| 2026-09-21 | `scripts/evaluate.py` | --split sir0_val --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-est-wesepref-e10 --metrics content --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-eval-wesepref-e10-asr | 6 min |  |
| 2026-09-21 | `scripts/evaluate.py` | --split sir0_val --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-est-wesepref-e10 --metrics signal,perceptual --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-21-eval-wesepref-e10-signal | 23 min |  |
| 2026-09-21 | `scripts/make_estimates.py` | 103 trials rendered, sir0 | 28 min | cpu, whole-clip |
| 2026-09-21 | `scripts/train.py --split sir0` | 9,955 trials x 16 epochs (no early stop, best idx 10), sir0 | 5.5 h | batch 10, cuda (**T4 x2, DataParallel**), **1,231 s/epoch**. 14.73 M params. 1.9x FASTER than the 7.19 M single-card run's 2,364 s/epoch at 2.05x the model. Copied by hand from the Kaggle session before deleting kaggle_out. |
| 2026-09-21 | `scripts/eval_by_case.py --split sir0_val --cases target_only,both,interferer_only,noise_only --listener asr` | 4 cases, sir0_val, listener asr | 8 min |  |
| 2026-09-21 | `scripts/make_kaggle_bundle.py --split sir0 --code-only` | 22 code files staged + verified + zipped to 84 MB | 18 min | laptop HDD. Data untouched: `sir0_train` is unchanged at 9,955 trials and already a Kaggle dataset, so no 30 GB re-upload. Most of the wall time is the 85 MB dereferenced teacher backbone and the in-place verification batch, not the code. Added by hand: the script does not use `run_log.timed`. |
| 2026-09-21 | `pytest tests/ -q` | 507 tests, whole suite | 11 min | laptop, 4 threads. Added by hand: pytest does not use `run_log.timed`. Was 74 tests / 5 s on 2026-08-15. |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_privval --condition both --est experiments/results/2026-09-14-est-privval-control --metrics content --out experiments/results/2026-09-15-eval-privval-control-asr | 1.5 h |  |
| 2026-09-15 | `scripts/measure_effective_mask_flatness.py` | 3 systems x 103 trials | 2 min | cpu, whole-clip, no model inference |
| 2026-09-15 | `scripts/measure_rtf.py` | --checkpoint models/model_sir0_struct-e12.pt --config experiments/configs/bsrnn_struct.yaml --chunk-ms 80 --threads 4 --device cpu --out experiments/results/2026-09-15-rtf-struct-e12 | 3 min |  |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-15-est-struct-e12-sir0val --metrics signal,perceptual --out experiments/results/2026-09-15-eval-struct-e12-sir0val-signal | 24 min |  |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-15-est-struct-e12-sir0val --metrics content --listener judge --judge-rpm 10 --out experiments/results/2026-09-15-eval-struct-e12-sir0val-judge | 14 min |  |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_val --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-15-est-struct-e12-sir0val --metrics content --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-15-eval-struct-e12-sir0val-asr | 7 min |  |
| 2026-09-15 | `scripts/make_estimates.py` | 103 trials, sir0 | 16 min | cpu, whole-clip |
| 2026-09-15 | `scripts/make_estimates.py` | 0 trials, sir0ext **(failed)** | 34 min | cpu, whole-clip |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_privval --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-15-est-struct-e8 --metrics content --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-15-eval-struct-e8-asr | 1.2 h |  |
| 2026-09-15 | `scripts/make_estimates.py` | 1421 trials, sir0ext | 2.9 h | cpu, whole-clip |
| 2026-09-15 | `scripts/evaluate.py` | --split sir0_privval --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-14-est-struct-e12 --metrics content --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-14-eval-struct-e12-asr | 1.2 h |  |
| 2026-09-15 | `scripts/make_estimates.py` | 1421 trials, sir0ext | 3.4 h | cpu, whole-clip |
| 2026-09-14 | `scripts/measure_mask_flatness.py` | 4 checkpoints x 50 trials | 51 min | cpu, whole-clip |
| 2026-09-14 | `scripts/make_estimates.py` | 1421 trials, sir0ext | 3.6 h | cpu, whole-clip |
| 2026-09-13 | `scripts/evaluate.py` | --split sir0_privval --condition both --systems floor,ceiling --metrics content --out experiments/results/2026-09-13-eval-privval-anchors | 1.4 h |  |
| 2026-09-13 | `scripts/derive_w_struct.py` | 50 crops, sir0 | 10 min | cpu, two backward passes per batch |
| 2026-09-13 | `scripts/derive_w_struct.py` | 2 crops, sir0 | 2 min | cpu, two backward passes per batch |
| 2026-09-13 | `scripts/evaluate.py` | --split sir0_privval --condition both --systems floor,ceiling --metrics content --out experiments/results/2026-09-13-eval-privval-anchors **(failed)** | 4.6 h |  |
| 2026-09-13 | `scripts/measure_mask_flatness.py` | 5 checkpoints x 50 trials | 1.6 h | cpu, whole-clip |
| 2026-09-13 | `scripts/make_estimates.py` | 0 trials, sir0ext **(failed)** | 2.2 h | cpu, whole-clip |
| 2026-09-13 | `scripts/make_estimates.py` | 0 trials, sir0ext **(failed)** | 36 min | cpu, whole-clip |
| 2026-09-13 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-13-est-noresidual --metrics content --out experiments/results/2026-09-13-eval-noresidual | 22 min |  |
| 2026-09-13 | `scripts/make_estimates.py` | 103 trials, sir0 | 47 min | cpu, whole-clip |
| 2026-09-13 | `scripts/diagnose_mask_structure.py` | 12 trials, sir0_val | 16 min | cpu, whole-clip, contended with two other jobs (load avg 16) |
| 2026-09-13 | `scripts/make_estimates.py` | 0 trials, sir0ext **(failed)** | 8 min | cpu, whole-clip |
| 2026-09-13 | `scripts/evaluate.py` | --split sir0_privval --condition both --systems floor,ceiling --metrics content --limit 2 --out /home/grant/.claude/jobs/c82a808a/tmp/anchor-smoke | 2 min |  |
| 2026-09-13 | `scripts/diagnose_residual.py` | 103 trials, sir0 | 20 min | cpu, whole-clip |
| 2026-09-13 | `scripts/render_trials.py --split sir0_privval` | 2,800 trials rendered | 30 min | 8 workers, 16 kHz PCM_16 |
| 2026-09-13 | `pytest tests/ -q` | 492 tests | 23 min | added by hand; the suite crossed the 1-minute threshold long ago and was never logged again after 08-15 |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hystfix-mild --metrics content --out experiments/results/2026-09-12-eval-hystfix-mild | 5 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 12 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hystfix-sharp --metrics content --out experiments/results/2026-09-12-eval-hystfix-sharp | 5 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 12 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hystint-mild --metrics content --out experiments/results/2026-09-12-eval-hystint-mild | 5 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 13 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hystint-sharp --metrics content --out experiments/results/2026-09-12-eval-hystint-sharp | 3 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 12 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-follow-020 --metrics content --out experiments/results/2026-09-12-eval-follow-020 | 5 min |  |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-follow-010 --metrics content --out experiments/results/2026-09-12-eval-follow-010 | 5 min |  |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hyst-hard --metrics content --out experiments/results/2026-09-12-eval-hyst-hard | 5 min |  |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hyst-mid --metrics content --out experiments/results/2026-09-12-eval-hyst-mid | 5 min |  |
| 2026-09-12 | `pytest -q` | full suite, 474 tests, after the Estimator band-gather refactor | 5 min | run alongside the mask sweeps |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-hyst-control --metrics content --out experiments/results/2026-09-12-eval-hyst-control | 6 min |  |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-floor0.20 --metrics content --out experiments/results/2026-09-12-eval-floor0.20 | 5 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 12 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-floor0.10 --metrics content --out experiments/results/2026-09-12-eval-floor0.10 | 8 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 14 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-12-est-floor0.05 --metrics content --out experiments/results/2026-09-12-eval-floor0.05 | 9 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 14 min | cpu, whole-clip |
| 2026-09-12 | `scripts/evaluate.py` | --split sir0_val --condition both --est /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-12-est-state-e6 --metrics content --out /home/grant/Documents/University/Masters/Project/TargetSpeakerExtractor/experiments/results/2026-09-12-eval-state-e6-asr | 6 min |  |
| 2026-09-12 | `scripts/make_estimates.py` | 103 trials, sir0 | 13 min | cpu, whole-clip |
| 2026-09-12 | `pytest -q` | full suite, 474 tests, after the crop-alignment fix | 9 min |  |
| 2026-09-12 | `pytest tests/test_crop_alignment.py` | 7 tests, ramp fixtures | 2 min |  |
| 2026-09-12 | `scripts/diagnose_state_teacher.py` | state teacher suppression ladder, sir0_val | 8 min |  |
| 2026-09-11 | `pytest -q` | full suite, 467 tests (D4a added 20) | 3 min |  |
| 2026-09-11 | `pytest -q` | full suite, 447 tests (head A added 25) | 5 min |  |
| 2026-09-11 | `scripts/derive_w_state.py` | sir0, 6 batches | 18 min |  |
| 2026-09-11 | `scripts/derive_w_state.py` | sir0, 6 batches | 17 min |  |
| 2026-09-06 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-04-train-sir0-10000/ --metrics content --listener judge --judge-rpm 10 --out experiments/results/2026-09-06-evaluate-10000-judge | 12 min |  |
| 2026-09-06 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-04-train-sir0-10000/ --metrics content --listener judge --judge-rpm 10 --out experiments/results/2026-09-06-evaluate-10000-judge **(failed)** | 85 s |  |
| 2026-09-06 | `scripts/measure_rtf.py` | --checkpoint models/model_sir0_10000-e6.pt --chunk-ms 80 --threads 4 --device cpu --out experiments/results/2026-09-06-rtf-10000-cpu | 2 min |  |
| 2026-09-06 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-04-train-sir0-10000/ --metrics signal,perceptual --out experiments/results/2026-09-06-eval-10000-signal-perceptual | 17 min |  |
| 2026-09-04 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-04-train-sir0-10000/ --metrics content --out experiments/results/2026-09-04-train-sir0-10000/ | 6 min |  |
| 2026-09-04 | `scripts/make_estimates.py` | 200 trials, sir0 | 27 min | cpu, whole-clip |
| 2026-09-04 | `scripts/train.py --split sir0` | 9,955 trials x 16 epochs (no early stop, best epoch 6), sir0 | 10.5 h | batch 3, cuda (T4), **2364 s/epoch**. Third point on the data-scaling curve; `w_schedule` step-indexed for the first time (warmup 6632 / ramp 4974). 2364 s/epoch against 1244 at 4,976 trials = 1.90x for 2.00x the data. Copied by hand from the Kaggle session's own run_times.md before deleting kaggle_out. |
| 2026-09-03 | `scripts/make_kaggle_bundle.py --split sir0` | 9,955 train + 200 val trials staged + zipped, `sir0` | 2.1 h | 4,979 new trials copied (4,976 already current); 30.3 GB zip at a sustained 6.5 MB/s, 130,619 entries. Staging + verify 42 min, data zip 1.4 h. Added by hand: the script does not use `run_log.timed`. |
| 2026-09-03 | `scripts/render_enrollment_bank.py --split sir0_train --variants 3` | 4,979 trials x 3 variants | 1.2 h | 8 workers, 16 kHz PCM_16 |
| 2026-09-03 | `scripts/measure_rtf_wesep.py` | --pretrain ../wesep_pretrained/tfmap_context_causal_100/ | 23 min |  |
| 2026-09-03 | `scripts/render_trials.py --split sir0_train` | 4,979 trials rendered | 1.1 h | 8 workers, 16 kHz PCM_16 |
| 2026-09-03 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-03-est-wesep-tfmap-causal --metrics content --listener judge --out experiments/results/2026-09-03-evaluate-wesep-judge | 11 min |  |
| 2026-09-03 | `scripts/evaluate.py` | --split sir0_val --condition both --est experiments/results/2026-09-03-est-wesep-tfmap-causal --out experiments/results/2026-09-03-evaluate-wesep-asr **(failed)** | 17 min |  |
| 2026-09-03 | `scripts/make_estimates_wesep.py` | 103 trials, sir0, tfmap_context_causal_100 | 34 min | cpu, whole-clip |
| 2026-09-03 | `scripts/make_estimates.py` | 103 trials, sir0 | 12 min | cpu, whole-clip |
| 2026-09-02 | `scripts/evaluate.py` | --est experiments/results/2026-09-01-est-sir0-5000 --metrics content --listener judge | 29 min |  |
| 2026-09-02 | `scripts/evaluate.py` | --limit 20 --est experiments/results/2026-09-01-est-sir0-5000 --metrics content --listener judge | 2 min |  |
| 2026-09-02 | `scripts/evaluate.py` | --limit 20 --est experiments/results/2026-09-01-est-sir0-5000 --metrics content --listener judge **(failed)** | 2 min |  |
| 2026-09-02 | `scripts/evaluate.py` | --limit 20 --est experiments/results/2026-09-01-est-sir0-5000 --metrics content --listener judge **(failed)** | 2 min |  |
| 2026-09-01 | `scripts/evaluate.py` | --split sir0_val --est experiments/results/2026-09-01-est-sir0-5000 | 14 min |  |
| 2026-09-01 | `scripts/make_estimates.py` | 200 trials, sir0 | 29 min | cpu, whole-clip |
| 2026-09-01 | `scripts/train.py --split sir0` | 4,976 trials x 18 epochs (early-stopped, best epoch 7), sir0 | 6.2 h | batch 3, cuda (T4), **1244 s/epoch**. 5,000-trial split, `weight_decay` 1e-4, bank K=3. Copied by hand from the Kaggle session's own run_times.md before deleting kaggle_out. |
| 2026-08-31 | `scripts/render_enrollment_bank.py --split sir0_train --variants 3` | 4,976 trials x 3 variants | 1.5 h | 8 workers, 16 kHz PCM_16 |
| 2026-08-31 | `scripts/render_trials.py --split sir0_train` | 4,976 trials rendered | 1.2 h | 8 workers, 16 kHz PCM_16 |
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
