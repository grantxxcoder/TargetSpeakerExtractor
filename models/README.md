# Checkpoints

`.pt` files are gitignored, so this file is the only record of which is which.

**Every checkpoint here is load-bearing — none is a spare.** The `_last.pt`
duplicates were deleted 2026-08-30 (248 MB): for those three runs best_val WAS the
last epoch, so `_last` held the same weights as the best file. `model_sir0_e50es_last.pt`
is NOT such a duplicate (best 14, last 24) and stays.
Every one carries its own `config` dict — `torch.load(p, weights_only=False)`
then read `["config"]` and `["best_row"]` if this file goes stale.

| file | run | epoch | `w_g` | enrolment sensitivity | what it is |
|---|---|---|---|---|---|
| `model_sir0.pt` | `2026-08-28-train-sir0-e10` | 9 | 1.69 | **-4.25 dB (37.6 %)** | **current best.** The `L_gain` run. Default for `pass_a_test_case_through.py`. |
| `model_sir0_amp-e10.pt` | `2026-08-29-train-sir0-e10-amp` | 9 | 1.69 | -4.96 dB (31.9 %) | the AMP + `chunk_s` 4.008 rerun. Trains 7.2x faster; scores within noise of `model_sir0.pt` (see below). |
| `model_sir0_wg0-e7.pt` | `2026-08-27-train-sir0` | 7 | none | -8.25 dB (14.9 %) | **the control.** No `L_gain`, and it MUTED — output sits ~22 dB below the mixture where the target is at ~-3.9. Keep: it is the comparison arm for the `L_gain` claim, not a spare. |
| `model_sir0_e50es.pt` | `2026-08-29-train-sir0-e50-resume` | 14 | 1.69 | -3.80 dB (41.7 %) | best epoch of the resumed run, and the lowest `val_total` on record (-2.178). **Overfit — read the note below before using it.** |
| `model_sir0_e50es_last.pt` | same | 24 | 1.69 | -0.98 dB (79.9 %) — **not a win, see note** | last epoch before early stop. Val extraction ended WORSE than passing the mixture through. |

Both runs: `sir0`, seed 42, batch 3, `chunk_s` 4.0, fp32 (pre-speed-fix).
Neither used AMP. decisions-m1.md 2026-08-28.

**`model_sir0.pt` vs `model_sir0_amp-e10.pt`.** The fp32 run scores better on
every headline (val_total -1.673 vs -1.413, enrolment 37.6 vs 31.9 %), so it stays
the default. But the gap is **inside run noise** — mean between-run difference 2.9
points against a mean within-run epoch-to-epoch swing of 8.3, and the sign flips
(AMP ahead at epochs 6-8, behind at 5 and 9). Do not claim either is the better
model on this evidence.

## `model_sir0_e50es*` — the overfitting run

Resumed at epoch 10, early-stopped at 24, best epoch 14. Requested 50 epochs, ran 15.

**Both of these memorised the training set.** Train extraction improved the whole
way (`L_pres` -2.97 -> -5.51, i.e. 4.8 % -> 13.8 % of the way from do-nothing to
oracle) while val went backwards (-1.52 -> **+0.17**). Do-nothing scores -1.59, so
by epoch 24 the model was a worse estimate of the target than the untouched
mixture. Val speakers are properly held out (1 of 41 shared with train), so this
is a real generalisation failure, not a split leak.

**Do not read the enrolment-sensitivity column as progress on this run.** It climbs
-3.66 -> -0.98 dB (43 % -> 80 % output movement on an enrolment swap), which looks
like the conditioning finally working. It is not. The diagnostic saturates: an
ideal extractor handed a stranger's enrolment outputs silence and scores exactly
**0.00 dB**, and an output that depends on the enrolment *arbitrarily* scores
**+3.01 dB**. Everything from 0 dB up is ambiguous, and -0.98 dB sits in it.
`val_L_pres` is the tiebreaker and says the output is worse than doing nothing:
strongly enrolment-dependent and strongly wrong.

`model_sir0.pt` is still the default for `pass_a_test_case_through.py`. `e50es` has
the better `val_total`, but `val_total` is not a safe ranking here — most of the
gap is the absent branch, not extraction. Decide deliberately before switching.

## Why each survivor is kept — read before deleting any of these

| file | why it cannot go |
|---|---|
| `model_sir0.pt` | the default checkpoint `scripts/pass_a_test_case_through.py` loads. Deleting it breaks that script. |
| `model_sir0_amp-e10.pt` | **the parent of the current baseline.** `model_sir0_e50es.pt` was produced by resuming from this file at epoch 9 (`best_val -1.4126`, see `session.log`). Without it the baseline's provenance chain is broken and the run cannot be reproduced. |
| `model_sir0_wg0-e7.pt` | **the control arm for the `L_gain` claim** — the last run without the term. The "`L_gain` closed the mute" result is a comparison against this and nothing else. |
| `model_sir0_e50es.pt` | the current baseline, epoch 14. |
| `model_sir0_e50es_last.pt` | epoch 24, the fully-overfit endpoint. The *evidence* for the overfitting result: the numbers are in `history.csv`, but demonstrating a model that scores worse than pass-through needs the weights. |
