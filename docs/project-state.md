# Where the project is — 2026-09-01

Plain-language status. Numbers in the milestone decision logs
(`decisions/decisions-m1.md` architecture, `-m2` training, `-m3` conventional
evaluation, `-m4` the metric and judge); dates and checklists in
`decisions/milestones.md`.

**Submission 2026-11-05. Experiment freeze 2026-10-14 — about 6 weeks.**

---

## In one paragraph

The memorisation diagnosis was tested and **confirmed**: 2.5x the training data
lifted held-out separation from 2.14 to 2.58 dB and **nearly doubled the margin
over doing nothing, 0.55 to 0.99 dB**. The model is still data-limited rather
than at capacity, but the return is diminishing and the extractor is not the
contribution. **All three metrics are now built and tested** — LCF-WER, ICR and
NRR — and two of them already produce real numbers using an offline ASR in place
of the judge. The one thing still missing is the judge itself: no live-model
measurement has been taken, and that is now the only thing between here and the
project's actual result.

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
- **Score all three of its own metrics.** LCF-WER, ICR and NRR implemented and
  tested (51 tests), judge-agnostic, with the transcriber swappable. Validated
  by reproducing the C2 floor/ceiling exactly.
- **Beat doing nothing by ~1 dB** on held-out data, against 0.55 dB a week ago.
- **Be scored end to end on its own metrics.** First system row taken 2026-09-01:
  LCF-WER 65.2 → **59.1 %**, ICR@2 67.0 → **54.4 %** against a 5.8 % / 0.0 %
  ceiling. It captures ~10 % of the word-error headroom and ~19 % of the leakage
  headroom, and it **hurts trials that were already easy** — the
  artefact-versus-residue trade-off, measured. `decisions-m3.md` 2026-09-01.

## Cannot

- **Generalise fully.** Improved but not solved: the train/held-out gap still
  reaches 4.16 dB by the last epoch, against 5.68 dB before.
- **Separate well.** Best held-out separation 2.58 dB against 1.59 dB for doing
  nothing. The margin is ~1 dB — better than it was, still thin.
- **Match level per utterance.** `L_gain` fell only 3 % over its whole run. It
  works as a *constraint on going silent*, not as a level regression target.
- **Be scored on a LIVE model.** The three metrics now produce real system
  numbers, but through an offline ASR standing in for the judge. No live-model
  measurement exists.

## Not started

None of these exist, and none can be cut:

- No metric harness, no benchmark, no comparison table.
- No judge MODEL picked (J2a closed / J2b open-weight anchor). No longer an open
  argument — it is now a ~1-hour candidate gate.
- AMI untouched — the only real-audio check in the project.

They need *a* trained model, not a good one. **They are not blocked on
generalisation and should not wait for it.**

**Unblocked 2026-08-31 — J1 closed: the judge is audio-in / text-out.** LCF
measures the judge's audio encoder, not its turn-taking, so full duplex is not
required. This also deletes the response-transcription ASR from the measuring
instrument. Carries a ~50-trial full-duplex confirmation run so the deviation
from the stated objective is bought off rather than argued away.
`decisions-m4.md` 2026-08-31.

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

1. **Choose the judge (J2).** It is the only thing blocking the project's actual
   result, and the choice is now a ~1-hour candidate gate rather than an open
   argument: score the ceiling, the floor and a few silent trials on each
   candidate and read whether it can report clean speech, whether it can fail,
   and whether it stays quiet when there is nothing to hear.
2. **Score the 5,000-trial checkpoint on the metrics that exist.** Render
   estimates, transcribe them, and produce the first system row for LCF-WER and
   ICR. Costs nothing and needs no judge.
3. **Do NOT render more data.** It would still help — the model is data-limited,
   not at capacity — but 2.5x bought +0.44 dB and a further 2x will buy less,
   for ~5 h of rendering and a 33 GB upload. `decisions-m2.md` 2026-09-01.

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
