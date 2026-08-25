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
