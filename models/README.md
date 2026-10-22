# Checkpoints

`.pt` files are gitignored, so this file is the only record of which is which. Each checkpoint carries its own `config` dict: `torch.load(p, weights_only=False)`, then `["config"]` and `["best_row"]`. **full** = model + optimiser + scheduler (resumable); otherwise weights only.

**Pruned 2026-10-17 to the thesis set: 125 files (6.1 GB) to 22 (1.3 GB).** Kept: every checkpoint the report rests on, their resume parents, and the live runs. The 103 deleted files (superseded arms, unselected epochs, and weights-only files whose weights were identical to a `-full` file) are listed with md5 in `deleted-2026-10-17.md5`. Not recoverable without retraining. Their histories, configs and judge results stay in `experiments/results/`; their notes are in `decisions-m2.md` and in this file's git history.

**Not in this folder.** Ext. (C+S+G) session 2 (idx 13-25, selected idx 25) is in `kaggle_out/session2/models/`; see `experiments/results/2026-10-16-train-sir0-presence-gate-A-resume/NOTES.txt`. WeSep is in `../wesep_pretrained/`.

| file | run (`experiments/results/`) | why it is kept |
|---|---|---|
| `model_sir0_baseline-matched-e11.pt` | `2026-10-07-train-sir0-baseline-matched-s1` | **Report: Baseline.** Lowest judge LCF-WER of its judged epochs. |
| `model_sir0_baseline-matched-e13-full.pt` | same | last epoch, full: the resume point. |
| `model_sir0_cuecontext-wer-e13.pt` | `2026-09-24-train-sir0-cuecontext-wer` | full. End of session 1, the parent of the resume that produced e21. Loaded by `demo_live.yaml`. |
| `model_sir0_cuecontext-wer-e18.pt` | `2026-09-24-train-sir0-cuecontext-wer-resume` | full. The in-loop Whisper pick, overruled by the judge. |
| `model_sir0_cuecontext-wer-e21.pt` | same | **Report: Ext. (C+S).** Judge pick 2026-09-26. Parent of every fine-tune below. |
| `model_sir0_cuecontext-wer-e27.pt` | same | full. Last epoch; the e21-vs-e27 appendix. |
| `model_sir0_interf-ft-{e3,e5-full}.pt` | `2026-10-02-train-sir0-interf-ft` | **Report: leak-weight fine-tune** (`tab:metric_evidence`). e5 primary, e3 secondary (D10 rule). |
| `model_sir0_artif-ft-{e3,e5-full}.pt` | `2026-10-03-train-sir0-artif-ft` | **Report: artefact-weight fine-tune.** Same rule. |
| `model_sir0_interf-ft-control-{e3,e5-full}.pt` | `2026-10-02-train-sir0-interf-ft-control` | the control for both fine-tunes, and for wm-ft. |
| `model_sir0_interf-artif-ft-{e3,e5-full}.pt` | `2026-10-03-train-sir0-interf-artif-ft` | the both-weights cell of the 2x2. |
| `model_sir0_presence-gate-A-e12.pt` | `2026-10-16-train-sir0-presence-gate-A` | Ext. (C+S+G) session 1 pick; parent of session 2. |
| `model_sir0_wm-ft-{e3,e4,e5,e5-full}.pt` | `2026-10-17-train-sir0-wm-ft` | C6 `w_m` re-derivation, live. `e5.pt` duplicates `e5-full`'s weights; kept because `eval_wm_ft.sh` loads it. |
| `model_sir0_10000-e6.pt` | `2026-09-04-train-sir0-10000` | the report baseline until 2026-10-07; most decision-log numbers before then use it. Default `--checkpoint` of `probe_state_features.py`, `diagnose_mask_structure.py`, `plot_mask_grid.py`. |
| `state_detector_{notebook,tuned}.pt` | `2026-09-11-train-sir0-state` | the speaker-state teacher (`state_teacher` in `bsrnn_state.yaml`). 3 MB. |

**Default checkpoint deleted, pass `--checkpoint`:** `pass_a_test_case_through.py` (`model_sir0.pt`), `measure_rtf.py` (`model_sir0_5000-e7.pt`).
