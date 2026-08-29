# Checkpoints

`.pt` files are gitignored, so this file is the only record of which is which.
Every one carries its own `config` dict — `torch.load(p, weights_only=False)`
then read `["config"]` and `["best_row"]` if this file goes stale.

| file | run | epoch | `w_g` | enrolment sensitivity | what it is |
|---|---|---|---|---|---|
| `model_sir0.pt` | `2026-08-28-train-sir0-e10` | 9 | 1.69 | **-4.25 dB (37.6 %)** | **current best.** The `L_gain` run. Default for `pass_a_test_case_through.py`. |
| `model_sir0_last.pt` | same | 9 | 1.69 | same | last epoch; identical to best here, since best_val WAS the last epoch |
| `model_sir0_amp-e10.pt` | `2026-08-29-train-sir0-e10-amp` | 9 | 1.69 | -4.96 dB (31.9 %) | the AMP + `chunk_s` 4.008 rerun. Trains 7.2x faster; scores within noise of `model_sir0.pt` (see below). |
| `model_sir0_amp-e10_last.pt` | same | 9 | 1.69 | same | last epoch |
| `model_sir0_wg0-e7.pt` | `2026-08-27-train-sir0` | 7 | none | -8.25 dB (14.9 %) | **the control.** No `L_gain`, and it MUTED — output sits ~22 dB below the mixture where the target is at ~-3.9. Keep: it is the comparison arm for the `L_gain` claim, not a spare. |
| `model_sir0_wg0-e7_last.pt` | same | 7 | none | same | last epoch of the control |

Both runs: `sir0`, seed 42, batch 3, `chunk_s` 4.0, fp32 (pre-speed-fix).
Neither used AMP. decisions-m1.md 2026-08-28.

**`model_sir0.pt` vs `model_sir0_amp-e10.pt`.** The fp32 run scores better on
every headline (val_total -1.673 vs -1.413, enrolment 37.6 vs 31.9 %), so it stays
the default. But the gap is **inside run noise** — mean between-run difference 2.9
points against a mean within-run epoch-to-epoch swing of 8.3, and the sign flips
(AMP ahead at epochs 6-8, behind at 5 and 9). Do not claim either is the better
model on this evidence.
