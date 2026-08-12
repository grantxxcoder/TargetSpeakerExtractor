# Decisions needed before audio can be generated

**Written 2026-08-10.** The trial table (`data/manifests/`) exists. The code that
turns a row into audio does not, and cannot be written until group A is answered.

Plain wording first, the jargon term in brackets so it can be matched to
`docs/data/data-construction-parameters.md`.

Move each answer into `docs/decisions/decisions.md` once made.

---

## A. Blocks the renderer

### A1. What should the "correct answer" audio sound like? (**reference signal / training target**)

The model is trained to reproduce one specific signal. Choosing it chooses the
task.

| Option | The model must learn to |
|---|---|
| The original recording, no room at all (**dry source**) | Separate the voices *and* remove all echo |
| Only the sound arriving straight from mouth to mic (**direct path**) | Separate and remove all echo, but keep the delay and quietening from distance |
| Straight path plus the first echoes off nearby walls (**direct + early reflections**, ~50 ms) | Separate, and remove only the long lingering echo |
| Exactly what the mic heard from that person (**full reverberant**) | Separate only, leave the room as it is |

**Recommended: straight path plus early echoes.** Removing lingering echo is a
second hard job, and it is where the audio starts sounding artificial — which is
the exact thing this project claims live models mishear.

### A2. *Decided 2026-08-11 — wrap around. See `decisions.md`.*

Numbers are kept in `decisions.md`. The gap in numbering is deliberate: A3-A6
keep their identifiers so earlier references stay valid.

### A3. *Decided 2026-08-12 — BS.1770 integrated loudness. See `decisions.md`.*

### A4. *Decided 2026-08-12 — no room on the enrollment. See `decisions.md`.*

### A5. Does the mixture keep running after the last person stops talking? (**tail padding**)

Echo continues after speech stops. If the file is cut at the last word, the echo
is cut too, and the correct-answer audio and the mixture stop matching.

**Recommended: yes, keep the tail.** Extend the window past the last speech by
at least as long as the room's echo lasts.

### A6. What happens if the summed audio is too loud to store? (**clipping**)

**Recommended: if anything exceeds 0.95 of full scale, quieten the mixture and
every separate track by the same amount.** Quietening only one track would
silently change the loudness differences the trial was built to have.

---

## B. Needed before the real sets are generated

### B1. How much of the mixture has both people talking at once? (**overlap ratio**)

Currently requested between 20 % and 70 % of the mixture, and recorded per trial
as an experimental variable. **Confirm the range is wide enough to show an effect
and narrow enough to be realistic.** 70 % is a hard ceiling while the target
talks 75 % of the time — both cannot exceed the quieter one.

### B2. Is "both talking" counted from where the recordings sit, or from where speech actually is? (**utterance boundaries vs voice activity detection**)

A read sentence contains pauses. Counting from the start and end of each
recording therefore overstates how much genuine talking overlaps.

| Option | Cost |
|---|---|
| Count from the recording boundaries (current) | Free, but the number is inflated and not comparable to the challenge's figure |
| Detect where speech actually is first (**VAD**) | An extra pass over the corpus, cached once |

**Recommended: detect it.** It is a one-off cost and it makes the number
defensible.

### B3. *Decided 2026-08-12 — fixed 5 s, kept configurable. See `decisions.md`.*

### B4. Should some evaluation trials contain no target speech at all? (**target-absent trials**)

Roughly a third of training trials have the target never speaking, so the model
learns to output silence rather than invent words.

**Answered 2026-08-12 (supervisor): yes — eval must include them, because what is
being measured is intelligibility.** Two things that answer does not settle, and
both block implementation:

1. `eval_public` and `eval_private` are `target_absent_fraction: 0.0` in
   `experiments/configs/generator.yaml`. Changing them forces an eval rebuild —
   fold it into the B9/B10 rebuild rather than rebuilding twice. **The fraction
   to use is still undecided**; train and val are at 0.35.
2. There is no correct text to compare against, so the main score cannot be
   computed on these trials, and `docs/data/metric-definitions.md` currently
   defines no rule for them. Standing proposal: **a separate reported row
   measuring only how often the system invents speech that was not there**,
   never folded into the main score. Not yet confirmed.

Stays open until both are settled.

### B5. How should the written text be tidied before comparison? (**text normalisation**)

The corpus transcripts are all capitals with no punctuation. The live model will
reply in normal sentences with numbers written as digits. Comparing them
unchanged makes every system look worse than it is.

**Decide one recipe — capitals, punctuation, numbers, filler words — and apply it
identically to both sides.** Never adjust it per system.

### B6. How many evaluation trials? (**trial count / judge budget**)

Cost is trials × repeats × systems × judges. Currently set to 500 per evaluation
split, which is a placeholder.

**Recommended: leave 500 generated and decide later how many to score.** Rows are
in a fixed order, so scoring the first 100 gives a set contained inside the first
300 — comparable as the budget grows.

### B7. Should the training mixtures be rebuilt differently each pass? (**per-epoch resampling**)

Rebuilding with fresh loudness and room draws each time gives the model far more
variety. It also means the run can no longer be reproduced from the table alone.

**Recommended: off for the main run, available as a switch.** Reproducibility of
the headline result is worth more than the extra variety.

### B8. *Decided 2026-08-11 — different book. See `decisions.md`.*

### B9. Silent-target trials can be spotted without listening to the voice (**target-absent detectability**) — **NB**

> **NB. This one changes what the model learns, not just how the data looks.**
> Silent-target trials are 35 % of every training batch and exist to teach one
> behaviour: stay quiet when the enrolled voice is not there. If absence can be
> read off the mixture some other way, that behaviour is never actually trained,
> and the failure will not show in any curve until the model meets audio built
> differently. Measured on `train` in §7 of the manifest notebook.

#### The evidence

A present trial always has the target talking for 0.75–0.85 of the window
(`target_activity_ratio` is fixed at 0.75 + up to 0.1 tolerance) and always has
at least 0.2 of the window double-talked (`overlap_ratio` starts at 0.2). A
silent-target trial has one voice and zero overlap. The two classes never come
closer than 0.200 against 0.000:

```
train   overlap_achieved   present 0.200..0.729   absent 0.000   AUC 1.0000
        total activity     present 0.98..1.72     absent 0.22..1.00
val     overlap_achieved   present 0.213..0.714   absent 0.000   AUC 1.0000
```

So *"did two people ever talk at the same time"* answers the question on **every
trial in the set**, without consulting the enrollment. Not an artefact of
measuring from file boundaries rather than voice activity (B2): trimming 500 ms
off both ends of every utterance still leaves present-trial overlap at 0.091
minimum, and no present trial anywhere near zero.

Compare with B10's speaker-identity leak at AUC 0.795. This one is worse in every
way — no memorisation needed, works from the first batch, and generalises to
nothing. In deployment the target often speaks with nobody interrupting, and a
model that learned "no overlap → emit silence" will talk over them.

#### What is actually missing

| target | interferer | overlap | in `train`? |
|---|---|---|---|
| speaks | speaks | 0.2–0.7 | 65 % |
| silent | speaks | 0 | 35 % |
| **speaks** | **silent** | **0** | **never generated** |
| **speaks** | **speaks, taking turns** | **~0** | **never generated** |

Two missing cases, and they need different fixes — widening `overlap_ratio` alone
cannot produce either.

**Why widening `overlap_ratio` does not work.** Overlap is floored by
`overlap >= target_activity + interferer_activity - 1`. With the target pinned at
0.75–0.85, zero overlap needs an interferer at ≤ 0.20 activity — about 3.5 s
against the current mean of 10.0 s. The interferer is also one contiguous block,
so it must fit inside a gap in the target's timeline, and those gaps are small:

```
total silence in the target's timeline : 3.49 s (20 % of the window)
largest single gap                     : 2.71 s (median 2.68, max 4.84)
  interferer block of 2 s fits in 82.7 % of trials
  interferer block of 3 s fits in 33.1 % of trials
  interferer block of 5 s fits in  0.0 % of trials
```

The result would be a narrow corner of trials with a near-silent interferer —
which is a *new* giveaway — plus heavy rejection, which is what produced the
`interferer_activity` bias in the first place (see below). **The binding
constraint is `target_activity_ratio`, not `overlap_ratio`.**

Found while checking this, and worth recording separately: **the target speaks in
a mean of 1.1 utterances per trial.** It is effectively one unbroken ~14 s
monologue filling 80 % of the window. Whatever is decided here, that is not what
conversation looks like.

#### Recommendation — two parameters, one rebuild

**1. Add `target_only_fraction` — do this regardless.** A new branch in
`build_trial` mirroring the silent-target branch: target speaks, no interferer,
noise bed as usual. It makes zero overlap *ambiguous* — "either the target alone
or the interferer alone" — so the only way to tell them apart is to compare the
voice against the enrollment, which is the behaviour being trained. It also fills
the largest realism gap in the set: **a target speaking uninterrupted is the most
common condition in deployment and currently appears zero times.**

On proportions, what matters is the odds *within* the zero-overlap trials.
Roughly 30 % silent-target / 15 % target-only / 55 % both leaves a third of
zero-overlap trials being target-only — enough that the shortcut is unlearnable.
Token amounts will not do: the model will absorb the loss and keep the shortcut.

**2. Let `target_activity_ratio` vary, down to a single short utterance.** This is
the larger change — it touches `pick_run`, `best_onset` and the overlap bounds
together — and it is what unlocks genuine turn-taking. It also fixes the residual
`interferer_activity` leak (§7.3 of the notebook, AUC 0.614) at source: that leak
exists because present trials are rejected when a contiguous interferer block
cannot hit the requested overlap, and silent-target trials never face that test.
A less talkative target widens the achievable overlap range so the tolerance test
stops firing asymmetrically. Balancing the rejection instead — running a phantom
target layout through `best_onset` on silent-target trials — would also work, but
only adds code whose purpose is to discard trials.

**Optional while rebuilding:** trials where *neither* speaker talks (noise bed
only). Teaches silence under pure noise, which is otherwise never demonstrated.
Cheap alongside 1, less urgent.

**A full manifest rebuild is required**, so do these together with B10 rather than
in separate passes. §2, §7 and the final health checks of the manifest notebook
assume the current timing model and will need revising after; §3–§6 will not.

### B10. The different-book enrollment guard split the speaker pool in two (**absent-only speakers**) — **NB**

> **NB. This is a side effect of B8, found by the check B8 itself asked for.**

B8 requires the enrollment clip to come from a different *book* than the mixture.
A present trial therefore needs a target who owns two or more books. A
silent-target trial skips the target's chapter search entirely and needs no such
thing, and when the retry loop draws a one-book speaker it silently redraws.

In LibriSpeech most readers read one book: **705 of the 1172 speakers** in
`train-clean-100` + `train-clean-360` have exactly one, 60.2 % (58 % across the
whole corpus). So the 467 two-book speakers are *precisely* the 467 that ever
appear as a present target, and the other 705 appear only in silent-target
trials — 21.3 % of `train`, at 5.9 trials each against 33.0 for the two-book
speakers. Scoring each trial by how often its target speaker is silent elsewhere
gives **AUC 0.795** on `train` and 0.756 on `val`; the interferer's identity gives
0.502, so this is specifically about the voice the model is told to listen for.

No reported number is contaminated — the eval splits are speaker-disjoint and
carry no silent-target trials — but 35 % of training batches can have the silence
decision made by voice recognition instead of by listening.

| Option | Cost |
|---|---|
| Draw silent-target speakers from the two-book pool as well | Closes it exactly, no rebuild of the guard. Target voices drop from 1170 to 467, and 705 speakers become interferer-only — a real loss of enrollment variety for a speaker-conditioned model |
| Book-preferred, chapter-fallback enrollment (the fallback B8 anticipated) | All 1172 speakers become present-capable, so the pools match and every voice is used. Reopens same-book enrollment for the 705 one-book speakers, which B8 judged indefensible |
| Reject enrollment clips that share too much rare vocabulary with the target's chapter | Keeps both the content guard and the full speaker pool. Most work, and needs a threshold nobody has justified yet |

**Recommended: the second, book-preferred with chapter fallback.** B8's argument
was about *content* leaking between enrollment and mixture; losing 60 % of the
speaker pool is a larger price than same-book enrollment for the speakers who
leave no alternative, and the fallback is recorded per trial so the effect can be
measured rather than assumed. Reversing B8 in part — take it to the supervisor.

### B11. Reverberation runs longer than the model is allowed to look (**T60 vs the latency budget**)

`t60_s` is sampled from [0.15, 0.6] s. With A1 settled as direct + early
reflections to 50 ms, the model is asked to suppress a tail reaching up to 600 ms
from a streaming window of 200–300 ms. **66.8 % of trials have a T60 longer than
that window.** It cannot cancel what it has not heard yet, so on those trials it
can only learn an average prior over rooms.

| Option | Cost |
|---|---|
| Leave the range, state the limit | Nothing to rebuild. Two thirds of the set trains a room prior rather than cancellation, and the write-up has to say so |
| Cap T60 at roughly the latency budget (~0.3 s) | Task becomes well-posed throughout. Throws away the reverberant half of the set and makes the benchmark easier than real rooms |
| Keep the data, move latency into the metric: score the same model at 100/200/300/400/500 ms and report the decay | Keeps every hard trial and costs no data. Several evaluation passes instead of one |

**Leaning (GB, 2026-08-12): the third — change the metric, never the data.**
Capping T60 would make the set less like a real room, which is the one thing the
constructed data is supposed to get right, and latency is a *secondary* objective
in the spec. So: train on the full range for the most accurate model available,
then let the metric show how the model degrades as the window shrinks, as a decay
curve rather than a single pass/fail at 300 ms.

This also reframes the finding as an opportunity rather than a defect. Learning a
prior over rooms is the intended first model contribution — varied room
dimensions so the model learns how sound propagates — so trials whose tail
outruns the window are the ones that will show whether the prior is doing
anything.

**Still open — take to the supervisor before pinning.** Depends on A1; revisit if
the reference signal changes.

---

## C. Ask the supervisor

1. **A1** — straight path only, or straight path plus early echoes?
2. **How hard should the task be?** Measured as how badly an off-the-shelf
   transcriber does on the raw mixture (**floor word error rate**). Too easy and
   nothing distinguishes systems; too hard and nothing can be ranked. The current
   plan targets 60–80 %.
3. **B4** — silent-target trials in evaluation as well as training?
4. **B9/B10** — silent-target trials are currently detectable without listening
   to the enrolled voice, structurally (no overlap ever) and by speaker identity
   (AUC 0.795). B10 means partly reversing B8. Both need a manifest rebuild.
5. **B11** — two thirds of trials have reverberation longer than the streaming
   window allows the model to see. Cap it, or report split by it?
