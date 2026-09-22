# Building a TSE front-end for a live speech-to-speech system

Transferable conclusions from this project, for a team starting their own.
Written 2026-09-22.

**Scope this before you use it.** Two-speaker mixtures (target + at most one
interferer + noise), constructed from LibriSpeech/WHAM!, 16 kHz, server-class
compute assumed. Evaluation is optimised for one judge family (Gemini). None of
our numbers are comparable to published REAL-TSE results — different data,
metric and protocol. Where a number is quoted it is ours, on our data.

---

## 1. Evaluation — read this section first

Our strongest results are about measurement, not modelling. Most of the cost
below was paid discovering that the obvious instruments are wrong.

**Do not tune post-processing on a local ASR.** Same audio, same 103 trials,
three settings, two independent systems: `faster-whisper small.en` and
`gemini-3.7-flash` picked *different* winners in both. Whisper overstated the
cost of one setting by **10.7x** on our model and got the **sign wrong** on the
other system (+14.93 vs the judge's −0.32). Anyone tuning on the cheap listener
ships the wrong configuration.

**Do not use perceptual quality (DNSMOS, MOS) as your objective.** Our model
made audio a human would rate *worse* (OVRL 2.497 → 2.237) while the listener
recovered **6.1 points more words**. A conventional perceptual metric would have
rejected a system that measurably helps the downstream task.

**Measure your metric's irrelevance floor before trusting any result.** An
arithmetically null intervention (multiply the spectrogram by exactly 1.0)
moved our headline **1.57 points in the model's favour** and rewrote **38 % of
transcripts**. Any claim at or below that floor is unclaimable.

**Test ideas at inference time before retraining.** Paired comparison on one
checkpoint with one knob resolves ~3 points; two independently trained models
need ~8. Cheaper *and* more sensitive — rare combination.

**Put a VAD gate in front of every listener.** Fed digital silence (RMS exactly
0), Gemini returned `status=speech` and fabricated fluent prose in **6 of 6**
clips across three prompts — once in French, on an English-only pipeline.
Prompting does not fix it; the variant explicitly forbidding hallucination
produced *more* invented words. Whisper does the same, milder. "Did it emit
speech?" is a signal question and never needed a language model.

**Score all four trial cases, not just the overlapped one.** We reported only
`both` (51.5 % of the split) for months. The other half showed the model passes
a stranger through as the target: on 42 trials where the target never spoke, our
output still yielded **1,042 transcribable words** against the mixture's 1,294 —
19 % suppression, and **83 % of clips still produce attributable words**.

**Track fabrication separately from word error.** Our headline improved 5.70
points while invented-word rate rose **10.7** (FR@2 57.28 → 68.00). A single
number hid a change of failure mode.

**Report headroom captured, not raw error.** Bands with different amounts of
room to win are not comparable otherwise. Ours: **under 9 %** captured where the
interferer is louder, 15–20 % where it is not — that is the collapse stated
honestly.

**Quote your instrument's own ceiling.** Our offline ASR ceiling is 6.1 %, not
0; DNSMOS's is 3.43 of 5, not 5. A score without its ceiling is unreadable.

**Pin and hash every learned component of the metric** — ASR, VAD, DNSMOS
models, text normaliser, stopword list. They drift between releases. Record
model ID, exact prompt, modality and date on every live-model call.

**Expect the live model to be robust where the ASR is brittle.** The judge's
spread across three settings was 2.44 points against a ±8 interval — it could
not tell them apart. Robustness means *less* exploitable structure. Measure with
it; be careful about training against it.

---

## 2. Data construction

**Centre your target/interferer loudness ratio on 0 dB.** This is the single
highest-value data decision we made. With 90 % of trials target-louder, "keep
the loud voice" is right ~81.5 % of the time for free, against ~58 % for a
learned speaker cue — so the model ignored the enrolment entirely. Swapping in a
stranger's voice sample changed the output by **2.6 %**.

**Do not fix a loudness shortcut by making the distribution asymmetric the other
way.** You create a new shortcut that looks like improvement.

**Train every mixture twice, once per speaker.** Same audio, both enrolments,
both ground truths. Free doubling of conditioning supervision.

**Include target-absent trials** (ours: ~50 % both / 25 % target-only / 20 %
interferer-only / 5 % noise-only). They are the only clean instrument for
whether the model rejects a non-target speaker — there is no target to extract,
so anything transcribable is a failure.

**Draw enrolment from a different recording session, not just a different
utterance.** Otherwise the model learns content, not voice.

**Give the enrolment no room acoustics.** Matching the target's *exact source
position* bought +4.9 points — positional fingerprinting, not speaker identity.
Both talkers share the room, so room-matching cannot separate them; position can.

**5 seconds of enrolment is enough.** 5 → 10 s bought 0.8 points; 5 → 20 s bought
the same 0.8. Do not re-render for longer enrolment.

**Enrolment EQ augmentation is near-free but near-worthless** — 0.3 points, 1 %
of the gap. Cheap to keep, not a lever.

**Use the reverberant target image as the training reference**, not the dry
signal — "what the mic heard". Otherwise you are asking for dereverberation you
never specified.

**Measure levels as BS.1770 integrated loudness**, and rescale the whole mixture
together to fix clipping.

**Measure overlap from detected speech, not file boundaries.** LibriSpeech is
86 % speech; naive overlap is overstated by ~25 %.

**Reject noise beds containing speech.** Ours are rejected at 0.5 s of detected
speech.

---

## 3. Conditioning and architecture

**Get identity into the network, then stop optimising that path.** Swapping a
stranger's enrolment moved our cue by 28.6 % and the output by **48.2 %** — the
network amplifies the cue rather than discarding it. Conditioning was not our
bottleneck; the separator and objective were. Measure this before spending
months on conditioning variants.

**Check what your cue actually encodes.** Ours turned out to be ~99 % correlated
with frame loudness, because it projects the mixture magnitude onto a unit
template — **architectural, not learned, and no data change fixes it**. 78 % of
its energy is rank-1. Hand the network the cue's *parts* (direction, normalised
similarity, unexplained residual), not their product.

**A cue built only from the target's spectra can say "target-like" and can never
say "that bin belongs to the other person."** That single mechanism explains
three separate symptoms for us. Negative evidence needs its own path.

**Watch normalisation for silent cue destruction.** Applying LayerNorm *within*
each narrow band left two degrees of freedom on our 3-bin bands and deleted the
cue's magnitude outright.

**Check softmax temperature after L2-normalising.** Normalising bounds cosines
to [−1,1], so the best-matching enrolment frame could outweigh the worst by only
e¹ ≈ 2.7x. Ours was averaging **619.6 of 628 frames** — a long-term mean
spectrum, not a selection. A `sqrt(F)` scale fixed it, worth +3.5 to +5.3 points.

**A frozen pretrained speaker encoder cannot memorise your training speakers.**
The standard objection to learned embeddings does not apply to a frozen one.

**Mask-only output cannot recover a bin the interferer dominates** —
multiplication only scales what is there. An additive residual branch is
standard; note it then becomes unidentifiable from the mask under most losses,
so constrain it or train a no-residual arm before making claims about "the mask".

---

## 4. Training

**Your data scaling curve is log-linear and will still be paying when you
stop.** Ours: each doubling bought ~0.32 dB of margin over pass-through
(1,989 → 4,976 → 9,955 trials). Fit two points, extrapolate, and decide whether
the next doubling is worth the render and upload cost — for us the next step was
~20,000 trials, ~30 GB, another day of upload, for the same increment.

**More data delays memorisation and raises the peak; it does not remove the
ceiling.** Train loss falling monotonically while validation peaks at epoch 7 is
the overfitting signature, not under-training.

**Separate "the model is under-trained" from "the model is data-limited" before
you buy either GPUs or data.** We have a capacity arm that doubled parameters
and bought nothing — and it is **confounded**, because 63 % of the added
parameters went into a `kernel_size=1` module that cannot improve separation
even in principle, at a learning rate held constant across a 3x batch change.
Record confounded arms as confounded, not negative.

**Check that your LR scheduler and your checkpoint selection watch the same
metric.** Ours stepped on a total loss that selection explicitly refuses to rank
on — for three weeks.

**A loss term whose target is unpredictable has a flat optimum.** Our
mask-shape term scored a perfectly flat mask **better** than the trained mask
(0.1717 vs 0.1803) — an L1 to an unpredictable target is minimised by the
conditional median. It was applying standing pressure toward exactly the failure
it was built to prevent.

**Test loss terms against a plausible imperfect prediction, not against the
oracle.** Comparing flat-vs-oracle is the one comparison that cannot fail, and
it is what our unit test did.

**Watch for headline metrics improving for bad reasons.** Our first full run's
total loss fell the entire way while the model collapsed to silence. Two of our
conditioning diagnostics reached their best values of the run at the epoch where
the model was worse than doing nothing.

**Index loss-weight warmup schedules in gradient steps, not epochs**, or the
schedule is not invariant to dataset size and your runs are not comparable.

---

## 5. Latency and deployment

**Real-time factor is the binding constraint, not latency.** If each chunk
finishes inside its own duration, the latency budget is met automatically. If
not, the backlog grows without bound.

**Never time a whole clip and call it streaming.** Whole-clip timing flattered
ours by **32x** — one batched matmul over every frame versus thousands of tiny
matrix-vector products that are launch-latency bound.

**80 ms chunks were our sweet spot.** 8 → 80 ms buys 9.3x; 80 ms → whole clip
buys only 3.5x more. At 160 ms chunks latency sits exactly on the 200 ms limit.

**"causal: true" in a config is not a causal system.** A well-known open TSE
checkpoint declares causal RNNs, then applies GroupNorm over the whole time axis
before them — every output frame depends on the entire clip. Measured leak
**1.12e-2** against our 1.68e-8, ~6 orders of magnitude, and ~900x its own
determinism floor. **Probe end-to-end causality yourself**: perturb the future,
measure the change in the past.

**Do not extrapolate compute from one layer.** We predicted ~5 ms per chunk from
a single LSTM; the truth was 42 ms — 8x wrong, because it ignored the per-band
estimator trunks, the STFT/iSTFT and the conditioning.

**Enrolment length is not a latency lever** — per-chunk time was flat from 0.5
to 5.0 s, so there is no easy win from caching the embedding.

**A 27 M-parameter model with a full speaker encoder re-embedded per chunk did
not meet the budget** (RTF 2.854, no chunk finishing in time) where our 7.19 M
model with a parameter-free cue did (RTF 0.528). Conditioning cost is asymmetric
and worth designing for.

---

## 6. Things we would tell you to check early

- **Whether your latency knob has ever run.** Ours crashed on any non-zero
  setting for a month — a 4-D tensor given a 2-long pad spec — so the ablation
  was silently un-runnable while being cited as future work.
- **Whether your headline split is the whole split.** Ours was 51.5 % of it.
- **Whether a "magnitude" diagnostic is actually directional.** "Output moves
  48.2 % on an enrolment swap" is fully consistent with the output only changing
  *volume*. We demoted a promising line on that evidence and had to reopen it.
- **Whether SI-SDRi is safe on your hardest slice.** It *rises* as the input
  gets worse, so it improves while the audio degrades.
- **Your run-to-run training variance.** Ten runs, one seed, no replicate — we
  still cannot put an error bar on a training-time intervention, and that is the
  gap we would close first if starting again.

---

## 7. What transfers least

- **Two-speaker only.** Our eval cannot detect a model exploiting "two
  overlapping voices means the target is present."
- **Read speech, not conversation.** No natural turn-taking; overlap is
  simulated.
- **One judge family.** Our metric is optimised for Gemini, and the benchmark is
  no longer held out from the system. Claims say *optimised for Gemini*, never
  *generalises to live models*.
- **Shoebox rooms, English, 16 kHz.**
