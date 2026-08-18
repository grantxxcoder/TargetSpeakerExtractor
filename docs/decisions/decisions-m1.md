# Decision Log — M1 (BSRNN implementation)

Decisions taken from M1 onward. Earlier milestones are in `decisions-m0.md`,
which is closed. Open questions still live in `decisions-pending.md`.

M1 scope: causal BSRNN + TF-Map extractor implemented, training infrastructure
trustworthy. See `milestones.md`.

---

## 2026-08-18 — Training chunk length: 4 s

**Decision: `chunk_s = 4.0` (64,000 samples at 16 kHz).**

What the literature uses, all verified in the source PDFs/configs:

| Source | Task | Chunk |
| --- | --- | --- |
| Luo & Yu, TASLP 2023 (music BSRNN) | Music separation | 3 s (T), drawn from 6 s salient segments |
| Yu et al., Interspeech 2023 (high-fidelity BSRNN) | Fullband SE + PSE | 6 s |
| Zhang et al., ICASSP 2025 (TF-Map) | TSE | 3 s |
| wesep `tse_bsrnn_spk.yaml` (REAL-TSE baseline) | TSE | 3 s (`chunk_len: 48000`) |
| CARTSE Track 1 (challenge winner) | **Online** TSE | **4 s** mixture, 3 s enrolment |

Why 4 s rather than the 3 s consensus:

1. **CARTSE is the only online, causal, TSE system in the set** — our exact task,
   and the system that won the track. It is the most defensible anchor we have.
   The 6 s outlier is a speech *enhancement* paper with no enrolment and no
   target-absent case, so its constraint is not ours.
2. **It reduces the empty-crop rate.** `target_activity_ratio` reaches down to
   0.15, so on an ~18 s clip a 3 s window can land entirely in target silence and
   silently turn a `both` trial into an absent example. A longer window catches
   speech more often. This mitigates the problem; it does not solve it — see the
   measurement required below.
3. **Our model is causal with an LSTM.** At inference the recurrent state persists
   across the whole stream, so training on short chunks means the state never
   accumulates as much context as it will see in deployment. Longer chunks narrow
   that train/inference gap. This argument does not apply to the offline models
   above, which is part of why they could afford 3 s.

Cost: ~33 % more frames per example than 3 s, so proportionally less batch at
equal memory.

**Required measurement: done 2026-08-18** — see the empty-crop entry below. The
drift is +0.043, which does not change this decision. `chunk_s = 4.0` stands.

Enrolment length is unchanged at 5 s (`decisions-m0.md` 2026-08-12). CARTSE used
3 s; ours sits inside the 5–10 s range of Yu et al. No change proposed.

---

## 2026-08-18 — Batch size: 12

**Decision: `batch_size = 12`, provisional on GPU memory.**

Matches CARTSE at the same 4 s chunk length ("trained on 4 s chunks with batch
size 12 for 200 epochs"), so the chunk/batch pair is taken as a unit from a system
that trained successfully on this task rather than tuned independently.

Provisional because batch size is hardware-bound and M2 runs on Kaggle sessions,
not the machine this was chosen on. If it must change, record the new value and
the memory limit that forced it — the effective batch is a training-dynamics
parameter, not just a throughput knob, and a silent change makes M2's curves
incomparable to this baseline.

---

## 2026-08-18 — Empty-crop rate measured: uniform cropping retained

**Decision: crop offsets stay uniform. VAD-aware cropping is NOT required.**
**The split loss must weight against the effective absent rate of 0.297, not the
designed 0.254.**

An "empty crop" is a chunk drawn from a trial where the target *does* speak, but
not during the window we cropped. `target_activity_ratio` reaches down to 0.15, so
a 4 s window can land entirely in the gaps. `crop_absent` is computed from the
cropped target stem, so these are labelled correctly — the risk was never
mislabelling, it was that the composition the model trains on drifts away from the
composition the generator was configured to produce.

Measured on `train`, 3,000 trials x 3 epochs = 9,000 crops, chunk 4 s, seed 42:

| quantity | value |
| --- | --- |
| designed absent rate (clip level) | 0.254 |
| effective absent rate (crop level) | **0.297** |
| drift | **+0.043** |
| leakage (present trial -> silent crop) | 0.058 |

Leakage by `target_activity`:

| band | leakage | n |
| --- | --- | --- |
| (0.0, 0.2] | 0.410 | 39 |
| (0.2, 0.3] | 0.347 | 354 |
| (0.3, 0.4] | 0.277 | 408 |
| (0.4, 0.5] | 0.114 | 537 |
| (0.5, 0.6] | 0.038 | 1584 |
| (0.6, 0.8] | 0.004 | 3426 |

Why uniform cropping is kept:

1. **The drift is small.** 0.254 -> 0.297 is +4.3 points. Not enough to move the
   model's silence prior meaningfully.
2. **Leakage is confined to the `hard` regime.** It is negligible above activity
   0.5 (0.4-3.8 %), where 75 % of the data sits, and only becomes large below 0.4
   — a band `base` does not reach at all (`base` draws [0.45, 0.78], `hard` draws
   [0.15, 0.78]). The existing escape hatch of filtering to `regime == base` is
   therefore also a leakage fix, so no new machinery is warranted.
3. **These crops are wanted.** A `both` trial cropped where the target has stopped
   talking is the interruption case: target present in the exchange, silent right
   now. That is a condition we care about, not noise.

Two sanity checks passed. Per-epoch effective rate was 0.291 / 0.298 / 0.302 —
varying but stable, confirming `set_epoch` reaches the offset draw. Leakage by
condition was `both` 0.059 vs `target_only` 0.055, indistinguishable at these
sample sizes, which is correct: leakage depends only on where the *target* speaks,
and the target stem is unaffected by whether an interferer exists.

Consequence to carry into the split loss: **the effective absent rate is 0.297.**
Any weighting between the target-present (SI-SDR) and target-absent
(push-to-silence) halves must use the crop-level rate, because that is what the
model sees. Using the designed 0.254 would under-weight the silence half by ~17 %
relative.

Caveat on scope: measured at `chunk_s = 4.0` on `train` only. Changing the chunk
length invalidates these numbers.

---
