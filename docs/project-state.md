# Where the project is — 2026-08-28

Plain-language status. Numbers in `decisions/decisions-m1.md`, dates and
checklists in `decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6.5 weeks.**

---

## In one paragraph

Data is built. The model runs on a GPU and has learned **who** to listen for, but
not **how** to output them — it turns everything down instead of separating. That
was traced to the training objective, not the model or the data, and a fourth
loss term (`L_gain`) was added 2026-08-27 and weighted 2026-08-28. It has not
been trained with yet. Nothing downstream of the model — metric, benchmark,
comparison table — has been started, and those cannot be cut.

## Can

- Run causally, streaming-compatible. Measured, not assumed.
- **Identify the target from a 5 s sample**: a stranger's sample moves the output
  39 %, up from 18 %. And on `sir0`, where "keep the louder voice" no longer works.
- Beat doing nothing: +3.33 dB against the raw mixture's +1.59 dB, so **1.73 dB
  of real separation**.
- Tell speech from silence — 94 % of the way to the best score that earns.
- Train, checkpoint and resume without losing optimiser/scheduler state.

## Cannot

- **Separate well.** 1.73 dB is small, and only 0.81 dB of it came in 8 epochs.
- **Output at the right volume.** ~21 dB below where the target should be, even
  when the target is speaking. It whispers the mixture.
- **Be trusted by its own loss curve.** Total fell −1.14 → −8.52, but **95 % of
  that was the silence half**.

## Why

**Nothing in the objective told it to stay loud.** The separation term is blind
to volume by design (it was written that way to stop an amplification exploit),
the silence term's best score is "output nothing", and `L_MR` — the one term
believed to police volume — was **measured to reward the mute**. Going quiet was
a free win. The model optimised the loss correctly; the loss was wrong.

That is why the fix is not architectural.

## The fix

`L_gain`: *"be about as loud as the target actually was, ±3 dB."* Symmetric (no
runaway volume direction), deadzone (no gradient on inaudible errors), anchored
per trial (a dataset average would just teach constant-volume output).

**`w_g = 1.69`, derived not guessed** (`scripts/derive_w_g.py`, 200 sir0_val
crops). Below 1.24 the mute still pays. An independent marginal argument gives
0.85, agreeing in magnitude.

**It does not cause separation — it removes the alternative to it.**

## Not started

None of these exist, and all are un-cuttable:

- No offline ASR chosen (`src/eval/`, `src/baselines/`, `src/live_model_metric/`
  are empty).
- No floor/ceiling WER — the last open M0 item, and what the supervisor needs for
  "how hard should the task be".
- No judge decided. Gates the whole metric milestone.
- No metric harness, benchmark or comparison table.
- AMI untouched — the only real-audio check in the project.

They need *a* trained model, not a good one. They are not blocked on separation.

## Compute

8 epochs on 1,989 trials = 8.6 h on a T4 at batch 3 (2026-08-27). The full
19,938-trial set is ~10x that per epoch, which does not fit a 12 h session.
Decision: use a smaller training set. Batch size is still 3 where the config says
12 — an untested lever.

## Next

1. ~~Derive `w_g`~~ — done 2026-08-28, 1.69.
2. Fresh run on Kaggle, `--split sir0 --epochs 10`. **Not** a resume: resuming at
   epoch 8 skips the warm-up, which is where `L_gain` runs at full strength with
   the silence pressure off, and the ablation arms would not be comparable.
   8 epochs gives only 1 at full objective; 10 gives 3.
3. Then decide about the architecture.

**Watch `pres_abs_gap_db`, not the total.** It should widen — loud when the target
speaks, quiet when it does not. Both ends rising together is a pass-through.

## The open architecture question

The instinct that the architecture is wrong is worth taking seriously but cannot
be tested yet: **it has never been trained against a working objective.** Step 2
is that test.

Justifies a change: separation still flat once the volume cheat is closed **and**
the gap widening (so the term worked and the problem is elsewhere). Does not: a
high loss, or slow progress in 8 epochs.

Note the cost first. The research plan says a fundamentally new architecture on
Kaggle in this timeframe "invites a bad viva". Changing the *conditioning* (D1,
`decisions-pending.md`) is cheap and additive; changing the *backbone* means
retraining from zero with 6.5 weeks left and no metric harness. Make that
distinction, not "change the architecture".
