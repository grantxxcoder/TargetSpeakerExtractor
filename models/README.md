# Checkpoints

`.pt` files are gitignored, so this file is the only record of which is which. Each checkpoint carries its own `config` dict: `torch.load(p, weights_only=False)`, then `["config"]` and `["best_row"]`. **full** = model + optimiser + scheduler (resumable); otherwise weights only.

**Pruned to the report's set** on 2026-10-17 (125 files to 22) and again on 2026-10-22 (to 11): the final baseline and extension, and the checkpoints the report's tables rest on. Not recoverable without retraining. The deleted files' histories, configs and judge results stay in `experiments/results/`; their notes are in `decisions-m2.md` and in this file's git history. WeSep is in `../wesep_pretrained/`.

| file | run (`experiments/results/`) | why it is kept |
|---|---|---|
| `model_sir0_baseline-matched-e11.pt` | `2026-10-07-train-sir0-baseline-matched-s1` | **Report: Baseline.** Lowest judge LCF-WER of its judged epochs. |
| `model_sir0_cuecontext-wer-e21.pt` | `2026-09-24-train-sir0-cuecontext-wer-resume` | **Report: Extension.** Judge pick 2026-09-26. Parent of every fine-tune below. Loaded by `demo_live.yaml`. |
| `model_sir0_cuecontext-wer-e27.pt` | same | full. Last epoch; the e21-vs-e27 appendix. |
| `model_sir0_interf-ft-{e3,e5-full}.pt` | `2026-10-02-train-sir0-interf-ft` | **Report: leak-weight fine-tune** (`tab:metric_evidence`). e5 primary, e3 secondary (D10 rule). |
| `model_sir0_artif-ft-{e3,e5-full}.pt` | `2026-10-03-train-sir0-artif-ft` | **Report: artefact-weight fine-tune.** Same rule. |
| `model_sir0_interf-ft-control-{e3,e5-full}.pt` | `2026-10-02-train-sir0-interf-ft-control` | the control for both fine-tunes. |
| `model_sir0_interf-artif-ft-{e3,e5-full}.pt` | `2026-10-03-train-sir0-interf-artif-ft` | the both-weights cell of the 2x2. |

The fine-tunes' training options were removed on 2026-10-22 (neither final model uses them), so `train.py` cannot reproduce them. They still load as models, so `make_estimates.py` can render them.
