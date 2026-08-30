# Where the project is — 2026-08-30

Plain-language status. Numbers in `decisions/decisions-m1.md`, dates and
checklists in `decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6.4 weeks.**

---

## In one paragraph

The objective is fixed and it worked. The model no longer mutes, it listens to
the voice sample, and training is 7.2x faster than a week ago. But the run that
was supposed to settle "just train it longer" instead found the next wall:
**the model memorises the 1,989 training trials.** It keeps getting better on
data it has seen and steadily worse on data it has not. That is a data-volume
problem, not an architecture problem, and it means adding capacity would make
things worse, not better. Nothing downstream of the model — metric, judge,
benchmark, comparison table — has been started, and none of it can be cut.

## What changed since 2026-08-28

1. **`L_gain` worked.** Blocking the mute taught the model to use the enrolment.
   Swapping in a stranger's voice sample now moves the output **37.6 %**, against
   14.9 % for the no-`L_gain` control. Output level sits at −4.2 dB where the
   target actually is (−3.9 dB); the control sat at −22.4 dB, i.e. it had gone
   silent and called that separation.
2. **Training is 7.2x faster.** 523 s/epoch against 3,773. Ten epochs is now
   **1.45 h, was 10.5 h.** Three changes; the one that mattered was lengthening
   the crop by 8 ms so fp16 tensor-core kernels align (4.44x on its own).
   Validated over a full 10 epochs in fp16 — no NaNs, and the score difference
   against fp32 is inside run-to-run noise.
3. **The "train longer" experiment ran and answered the question — badly.**
   See below.

## The finding that matters: it memorises

`2026-08-29-train-sir0-e50-resume`. Resumed at epoch 10, ran to 24, early-stopped
on patience, 2.2 h. Best epoch **14**.

| | epoch 10 | epoch 14 (best) | epoch 24 |
|---|---|---|---|
| separation on **training** data (`L_pres`, dB) | 2.97 | 3.38 | **5.51** |
| separation on **held-out** data (dB) | 1.52 | 2.14 | **−0.17** |
| gap between them | 1.45 | 1.24 | **5.68** |

**Read the two rows in opposite directions.** On audio it has already seen the
model improves every single epoch, all fourteen of them. On audio it has not seen
it improves for four epochs, then falls apart — by epoch 24 it is *worse than
handing the mixture through untouched*. The level-matching term shows the same
split: better on train (2.92 → 1.51), worse on held-out (3.77 → 5.79).

That is textbook overfitting, and it has a specific cause: **1,989 trials is not
enough data for a 7.2 M-parameter model.** Random crops already vary per epoch,
so the effective set is roughly 24,000 four-second examples — still small.

**A headline that improved for a bad reason.** Enrolment sensitivity kept
climbing right through the collapse, 43 % → 80 %. That is not conditioning
getting better. A model whose output has stopped resembling the target will move
a lot when you change its input, for no useful reason. **Do not quote the 80 %.**
The trustworthy figure is 37.6 % at epoch 9, and 41.7 % at epoch 14.

## Can

- Run causally, streaming-compatible. Measured, not assumed.
- **Identify the target from a 5 s sample.** 37.6 % enrolment sensitivity, on
  `sir0`, where "keep the louder voice" no longer works.
- **Output at roughly the right volume** — the mute is closed.
- Tell speech from silence: ~7 dB louder when the target is talking than when
  it is not, against 2.45 dB for the control.
- Train 10 epochs in 1.45 h, checkpoint, and resume without losing state.
- **Transcribe for evaluation, and say how hard the task is.** Offline ASR
  chosen (`faster-whisper small.en`) and C2 closed at n=230.

## Cannot

- **Generalise.** The single biggest open problem, and it is new as of yesterday.
- **Separate well.** Best held-out separation ~2.1 dB, against ~1.6 dB for doing
  nothing on the same data. The margin is thin.
- **Match level per utterance.** `L_gain` fell only 3 % over its whole run. It
  works as a *constraint on going silent*, not as a level regression target.
- **Be scored on the thing this project is about.** Every number above is a
  training diagnostic. There is still no measurement on the live-model metric.

## Not started

None of these exist, and none can be cut:

- **No judge decided.** Gates the entire metric milestone (M4).
- No metric harness, no benchmark, no comparison table.
- AMI untouched — the only real-audio check in the project.

They need *a* trained model, not a good one. **They are not blocked on
generalisation and should not wait for it.**

## Offline ASR — chosen 2026-08-28

`small.en`, int8 on CPU, greedy, Whisper `EnglishTextNormalizer`. `tiny.en`'s
floor exceeds 100 % (it invents words); `medium.en` is better but 2.7x the cost
across every pass.

**C2 is closed, 2026-08-30.** Scored at n=230 on `both` trials — the condition
that has an interferer to remove, and the only row that should ever be quoted:

| set | ceiling (clean) | floor (raw mixture) |
|---|---|---|
| `eval_public` (n=230) | 6.1 % | **57.4 %** |
| `sir0_val` (n=103) | 5.8 % | **65.2 %** |

**Plain reading:** of every 100 words the target says, ~57 come out wrong if you
do nothing, against ~6 wrong on clean audio. **That 51-point gap is the room the
extractor has to work in.** And the errors are not mush: inspected on one trial,
the ASR transcribes the target perfectly for 17 words and then switches to the
*other speaker's* sentence — which is exactly the failure the model exists to fix.

Accepted at this range. **This replaces the 76.4 % that was quoted from a
12-trial pilot; it was wrong by 19 points.**

**The open consequence is bigger than the number.** Training is on `sir0`, which
is symmetric by construction, while `eval_public` keeps the original
distribution where the target is the louder voice 74 % of the time. Which set
defines the benchmark is undecided, and that is the supervisor conversation.

Known artefact: `small.en` emits the word "you" on digital silence, 8 of 8 absent
trials. Filter it before counting invented words.

## Compute

523 s/epoch at 1,989 trials on a Kaggle T4, batch 3. Scaling the training set is
the obvious response to overfitting, and it is affordable: **5,000 trials is
~1,315 s/epoch, so 20 epochs in 7.3 h** — inside Kaggle's ~12 h cap. 19,938
trials would be ~2.9 h/epoch and does not fit. Rendering a larger `sir0` set is
~31 min per 2,000 trials locally.

Batch size is still 3 where the config's comment says 12 — untested, and it must
stay pinned at 3 for any resume, because `train.py` refuses a resume whose config
differs from the checkpoint's.

## Next

1. **More data, not more model.** Render a larger `sir0` split (~5,000 trials)
   and retrain from scratch. This is the direct test of the diagnosis: if the
   train/held-out gap narrows, the finding is confirmed and the ceiling moves.
2. **The two free regularisers, in the same run or before it.** `weight_decay`
   is 0.0 and there is no dropout. Both are config-level and cost nothing.
3. **Start the metric work in parallel, on the epoch-14 checkpoint.** It does not
   need a better model, and it is the project's actual contribution. Decide the
   judge first — everything in M4 queues behind it.

**Watch the train/held-out gap, not the total.** Both totals fell the whole way
through the run that overfitted.

## The open architecture question

Still open, and the answer has moved *against* changing anything. The instinct
was "the architecture is too weak." The measurement says the opposite: the model
has enough capacity to memorise its training set. **A bigger or richer model
overfits sooner.** Conditioning changes (D1/D4a in `decisions-pending.md`) are
cheap and additive and remain reasonable; replacing the backbone means
retraining from zero with six weeks left and no metric harness. Keep that
distinction — it is not "change the architecture", it is which half.
