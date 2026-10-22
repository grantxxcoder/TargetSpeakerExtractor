# Building a TSE front-end for a live speech-to-speech system

Transferable conclusions from this project, for a team starting their own. Written 2026-09-22.

**Scope this before you use it.** Two-speaker mixtures (target + at most one interferer + noise), built from LibriSpeech/WHAM!, 16 kHz, server-class compute. Evaluation is optimised for one judge family (Gemini). Every number is ours, on our data, and none is comparable to published REAL-TSE results (different data, metric and protocol).

---

## 1. Evaluation — read this section first

Our strongest results are about measurement, not modelling: the obvious instruments are wrong.

**Do not tune post-processing on a local ASR.** Same audio, 103 trials, three settings, two systems: `faster-whisper small.en` and `gemini-3.7-flash` picked *different* winners in both. Whisper overstated one setting's cost by **10.7x** on our model and got the **sign wrong** on the other system (+14.93 vs the judge's −0.32).

**Do not use perceptual quality (DNSMOS, MOS) as your objective.** Our model made audio a human would rate *worse* (OVRL 2.497 → 2.237) while the listener recovered **6.1 points more words**.

**Measure your metric's irrelevance floor first.** A null intervention (spectrogram × exactly 1.0) moved our headline **1.57 points in the model's favour** and rewrote **38 % of transcripts**. Nothing at or below that floor is claimable.

**Test ideas at inference time before retraining.** One checkpoint with one knob, paired, resolves ~3 points; two independently trained models need ~8.

**Put a VAD gate in front of every listener.** On digital silence (RMS exactly 0), Gemini returned `status=speech` and fabricated fluent prose in **6 of 6** clips across three prompts, once in French on an English-only pipeline. A prompt forbidding hallucination produced *more* invented words. Whisper does the same, milder.

**Score all four trial cases, not just the overlapped one.** We reported only `both` (51.5 % of the split) for months. On 42 trials where the target never spoke, our output still yielded **1,042 transcribable words** against the mixture's 1,294 (19 % suppression), and **83 % of clips still produce attributable words**: the model passes a stranger through as the target.

**Track fabrication separately from word error.** Our headline improved 5.70 points while the invented-word rate rose **10.7** (FR@2 57.28 → 68.00).

**Report headroom captured, not raw error,** so bands with different room to win are comparable. Ours: **under 9 %** where the interferer is louder, 15–20 % where not (the collapse).

**Quote your instrument's ceiling.** Our offline ASR ceiling is 6.1 %, not 0; DNSMOS's is 3.43 of 5, not 5.

**Pin and hash every learned metric component** (ASR, VAD, DNSMOS models, text normaliser, stopword list); they drift between releases. Record model ID, exact prompt, modality and date on every live-model call.

**Expect the live model to be robust where the ASR is brittle.** The judge's spread across three settings was 2.44 points against a ±8 interval: it could not tell them apart. Robustness means *less* exploitable structure. Measure with it; be careful training against it.

---

## 2. Data construction

**Centre your target/interferer loudness ratio on 0 dB.** Our highest-value data decision. With 90 % of trials target-louder, "keep the loud voice" is right ~81.5 % of the time for free, against ~58 % for a learned speaker cue, so the model ignored the enrolment: a stranger's voice sample changed the output by **2.6 %**.

**Do not fix a loudness shortcut by skewing the distribution the other way.** That creates a new shortcut that looks like improvement.

**Train every mixture twice, once per speaker** (same audio, both enrolments, both ground truths). Free doubling of conditioning supervision.

**Include target-absent trials** (ours: ~50 % both / 25 % target-only / 20 % interferer-only / 5 % noise-only). They are the only clean test of rejecting a non-target speaker: with no target, anything transcribable is a failure.

**Draw enrolment from a different recording session, not just a different utterance.** Otherwise the model learns content, not voice.

**Give the enrolment no room acoustics.** Matching the target's *exact source position* bought +4.9 points by positional fingerprinting, not speaker identity. Both talkers share the room, so room-matching cannot separate them; position can.

**5 seconds of enrolment is enough.** 10 s bought 0.8 points; 20 s the same 0.8.

**Enrolment EQ augmentation is near-free but near-worthless:** 0.3 points, 1 % of the gap.

**Use the reverberant target image ("what the mic heard") as the training reference**, not the dry signal. Otherwise you ask for dereverberation you never specified.

**Measure levels as BS.1770 integrated loudness**, and rescale the whole mixture together to fix clipping.

**Measure overlap from detected speech, not file boundaries.** LibriSpeech is 86 % speech, so naive overlap is overstated by ~25 %.

**Reject noise beds containing speech** (ours: at 0.5 s of detected speech).

---

## 3. Conditioning and architecture

**Get identity into the network, then stop optimising that path.** A stranger's enrolment moved our cue by 28.6 % and the output by **48.2 %**: the network amplifies the cue. Our bottleneck was the separator and objective, not conditioning.

**Check what your cue encodes.** Ours was ~99 % correlated with frame loudness because it projects the mixture magnitude onto a unit template: **architectural, not learned, and no data change fixes it**. 78 % of its energy is rank-1. Give the network the cue's *parts* (direction, normalised similarity, unexplained residual), not their product.

**A cue built only from the target's spectra can say "target-like" but never "that bin belongs to the other person."** This one mechanism explains three of our symptoms. Negative evidence needs its own path.

**Watch normalisation for silent cue destruction.** LayerNorm *within* each narrow band left two degrees of freedom on our 3-bin bands and deleted the cue's magnitude.

**Check softmax temperature after L2-normalising.** Normalised cosines lie in [−1,1], so the best enrolment frame outweighed the worst by only e¹ ≈ 2.7x. Ours averaged **619.6 of 628 frames**: a long-term mean spectrum, not a selection. A `sqrt(F)` scale fixed it (+3.5 to +5.3 points).

**A frozen pretrained speaker encoder cannot memorise your training speakers.**

**Mask-only output cannot recover a bin the interferer dominates**; multiplication only scales what is there. An additive residual branch is standard, but is then unidentifiable from the mask under most losses. Constrain it or train a no-residual arm before claiming anything about "the mask".

---

## 4. Training

**Your data scaling curve is log-linear and will still be paying when you stop.** Each doubling bought us ~0.32 dB of margin over pass-through (1,989 → 4,976 → 9,955 trials). Fit two points and extrapolate before paying for the next doubling (for us ~20,000 trials, ~30 GB, another day of upload, same increment).

**More data delays memorisation and raises the peak, but does not remove the ceiling.** Train loss falling monotonically while validation peaks at epoch 7 is overfitting, not under-training.

**Separate "under-trained" from "data-limited" before you buy GPUs or data.** Our capacity arm doubled parameters and bought nothing, but is **confounded**: 63 % of the added parameters went into a `kernel_size=1` module that cannot improve separation, at a learning rate held constant across a 3x batch change. Record confounded arms as confounded, not negative.

**Check that your LR scheduler and checkpoint selection watch the same metric.** For three weeks ours stepped on a total loss that selection refuses to rank on.

**A loss term with an unpredictable target has a flat optimum.** Our mask-shape term scored a perfectly flat mask **better** than the trained one (0.1717 vs 0.1803): L1 to an unpredictable target is minimised by the conditional median.

**Test loss terms against a plausible imperfect prediction, not the oracle.** Flat-vs-oracle cannot fail. Our unit test used it.

**Watch for headline metrics improving for bad reasons.** Our first full run's total loss fell throughout while the model collapsed to silence. Two conditioning diagnostics peaked at the epoch where the model was worse than doing nothing.

**Index loss-weight warmup in gradient steps, not epochs**, or the schedule depends on dataset size and runs are not comparable.

---

## 5. Latency and deployment

**Real-time factor is the binding constraint, not latency.** If each chunk finishes within its own duration, the latency budget is met; otherwise the backlog grows without bound.

**Never time a whole clip and call it streaming.** Whole-clip timing flattered ours by **32x**: one batched matmul over all frames versus thousands of tiny matrix-vector products bound by launch latency.

**80 ms chunks were our sweet spot.** From 8 to 80 ms buys 9.3x; from 80 ms to the whole clip only 3.5x more. At 160 ms, latency sits exactly on the 200 ms limit.

**"causal: true" in a config is not a causal system.** A well-known open TSE checkpoint declares causal RNNs but applies GroupNorm over the whole time axis before them, so every output frame depends on the entire clip. Measured leak **1.12e-2** vs our 1.68e-8 (~6 orders of magnitude), ~900x its own determinism floor. **Probe end-to-end causality yourself**: perturb the future, measure the change in the past.

**Do not extrapolate compute from one layer.** We predicted ~5 ms per chunk from a single LSTM; it was 42 ms (8x wrong), because we ignored the per-band estimator trunks, STFT/iSTFT and conditioning.

**Enrolment length is not a latency lever.** Per-chunk time was flat from 0.5 to 5.0 s, so caching the embedding gains nothing.

**A 27 M-parameter model re-embedding with a full speaker encoder per chunk missed the budget** (RTF 2.854, no chunk in time); our 7.19 M model with a parameter-free cue met it (RTF 0.528). Design for conditioning cost.

---

## 6. Things we would tell you to check early

- **Whether your latency knob has ever run.** Ours crashed on any non-zero setting for a month (a 4-D tensor given a 2-long pad spec), while the ablation was cited as future work.
- **Whether your headline split is the whole split.** Ours was 51.5 % of it.
- **Whether a "magnitude" diagnostic is directional.** "Output moves 48.2 % on an enrolment swap" is consistent with only *volume* changing. We demoted a promising line on it and had to reopen it.
- **Whether SI-SDRi is safe on your hardest slice.** It *rises* as the input gets worse, so it improves while the audio degrades.
- **Your run-to-run training variance.** Ten runs, one seed, no replicate: no error bar on any training-time intervention. We would close this gap first.

---

## 7. What transfers least

- **Two-speaker only.** Our eval cannot detect a model exploiting "two overlapping voices means the target is present."
- **Read speech, not conversation.** No natural turn-taking; overlap is simulated.
- **One judge family.** Our metric is optimised for Gemini, and the benchmark is no longer held out from the system. Claims say *optimised for Gemini*, never *generalises to live models*.
- **Shoebox rooms, English, 16 kHz.**
