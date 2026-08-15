# Run times

Wall-clock for every job slow enough to plan around. **Add a row for anything over
a minute.** Purpose: knowing in advance whether a step is a coffee break or an
overnight run, and having a real number for the write-up instead of a guess.

Machine unless stated: laptop, Intel i5-1135G7, **4 physical cores / 8 threads**,
15 GB RAM, NVMe. No usable GPU (`torch.cuda.is_available() == False`).
Hyperthreading buys ~10 % here — measured, 4 workers 111 s vs 8 workers 99 s.

| date | command | scope | wall | rate |
|---|---|---|---|---|
| 2026-08-14 | `build_manifest.py --split train` | 20,000 trials | 58 s | headers only, no audio |
| 2026-08-15 | `build_vad_index.py` | 137,876 utts / 475 h | *running* | 189x realtime, 8 workers |
| 2026-08-15 | `pytest tests/ -q` | 74 tests | 5 s | |

## Not yet run

| command | scope | projected | basis |
|---|---|---|---|
| `measure_vad_impact.py` | 2,000 utts + 400 trials | ~20 min | 8 settings x 2,000 decodes |
| WHAM! speech screen | 28,000 clips / 81.7 h | ~26 min | same 189x rate |
| `render_trials.py` | ~21,200 trials | **unknown** | 2 RIRs/trial, cost unmeasured |
