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

## 2026-08-18 — STFT: 512/128, causal framing, and a lookahead knob

**Decision: `n_fft = 512`, `hop = 128`, Hanning window, `center=False` with
symmetric `n_fft - hop` padding at both ends. Algorithmic latency 40 ms.
`lookahead_frames` is a config knob, 0-16, default 0.**

### Window and hop

512/128 at 16 kHz is 32 ms window / 8 ms shift, taken from Yu et al. (Interspeech
2023) §4.2, which specifies exactly these for its 16 kHz model (the paper's other
figures — the 33-subband split, the 20 ms/10 ms window — belong to its 48 kHz
fullband model and do not transfer; see `decisions-m0.md` on not importing
fullband settings wholesale).

Chosen against **our** ~200-300 ms streaming budget, not the REAL-TSE Track 1
100 ms cap. At 40 ms we are using well under a quarter of the budget, which is the
headroom the lookahead knob below exists to spend. This is the M1 milestone's
required justification.

### Framing: `center=False`

`center=True` is the PyTorch default and is wrong for us. It centres frame *t* on
its timestamp, so the frame consumes `n_fft/2` = 256 samples = **16 ms of audio
from after *t***. Offline that is invisible — the future samples are on disk, the
model trains and scores normally — and it only surfaces when the system is
streamed. `center=False` frames cover `[t, t + n_fft)` and are emitted once that
window has filled, so nothing depends on the future.

The `n_fft - hop` = 384-sample left pad is not a numerical trick: it reproduces the
cold start of a live system, whose input ring buffer begins zeroed and fills over
the first 384 samples.

### Synthesis is a manual overlap-add, not `torch.istft`

`torch.istft` **refuses `center=False`** — it checks the overlap-add envelope for
edge coverage and raises `window overlap add min: 1`. Left-padding does not rescue
it. Synthesis is therefore `F.fold` (vectorised overlap-add) with explicit
envelope normalisation: divide by the summed squared window so each output sample
is corrected for how many frames covered it.

**The tail pad is load-bearing and was found by a bug.** A first implementation
padded only enough to complete the final frame, which for a 64,000-sample input
gives `right = 0`. The last real samples then sit where the envelope has decayed
to ~1e-9, and dividing by that destroys them: measured max reconstruction error
**6.6e-3, concentrated entirely in the final samples** (head and interior were
7e-7). Padding the tail by `n_fft - hop` as well, so the real signal ends before
the ramp-out, gives **9.5e-7 across lengths 12,345 / 16,000 / 63,999 / 64,000 /
64,001** with a flat envelope of 1.5 over the whole real signal.

Regression test keeps an assertion on the **envelope** (`env[pad:pad+T].min() >
1.0`), not only on reconstruction error. The error says something is wrong; the
envelope says what, and it generalises to any window/hop tried during the latency
ablation.

### Latency convention

40 ms = `n_fft + hop` = window fill plus one hop to emit. **State this convention
whenever the number is quoted:** CARTSE reports `window - hop` = 24 ms plus 8 ms
buffering for the same framing. Neither is wrong; they count different things. The
number that actually goes in the report is the *measured* effective future
dependency, not the formula.

### Lookahead

`lookahead_frames` (0-16, one frame = one hop = 8 ms) lets the mask head see *k*
frames of future context. Total algorithmic latency 40 ms at k=0 to 168 ms at
k=16 — all inside budget before compute and the live-model API round trip.

**Implementation note, non-obvious and worth defending:** lookahead cannot be
implemented by shifting the training target. The model emits a complex mask
multiplied into mixture frame *t*, and mixture frame *t* carries the target's
content at time *t*; a per-frame multiplicative mask cannot move energy in time.
The output must stay aligned to the mixture, and the lookahead is instead a shift
of the *feature* sequence between the sequence stack and the mask head
(`lookahead_shift`), so the mask for frame *t* is computed from a hidden state
that has consumed frames up to *t + k*.

Train k=0 first: the causal baseline is what the ablation is measured against.

Planned ablation: k in {0, 4, 8, 16} against LCF. No published work plots the
accuracy/latency trade-off against a live-model content-fidelity metric, because
no such metric exists yet — this is ours to produce.

---

## 2026-08-18 — Band plan: inherited for the baseline, six candidates for ablation

**Decision: `wesep_16k` is the baseline plan. It is inherited, not justified —
that is recorded here deliberately, because the band plan is an open question in
this literature and one of the few cheap contributions available to us.**

### The plan is unjustified everywhere, not just here

The 16 kHz table (`[3]*15 + [6]*10 + [16]*5 + [64] + [8]`, 32 bands over 257 bins)
does **not** come from Yu et al. (Interspeech 2023). That paper specifies 33
subbands — 20x200 Hz + 6x500 Hz + 7x2 kHz — for its **48 kHz** model, which covers
~21 kHz and does not transfer to our 8 kHz ceiling. The 16 kHz table is the
wesep / REAL-TSE baseline's own adaptation, and it is not published or motivated
anywhere. `review_synthesis.md` already noted SA-Mamba and CARTSE quietly use 36
and 32 bands with no stated reason.

So: every system in this space inherited a table from a *music* separation paper
(Luo & Yu, TASLP 2023), hand-adapted it, and nobody wrote down why. Given that
per-band normalisation is a large part of why the architecture works — speech
energy falls ~6 dB/octave, so without splitting the high bands are numerically
invisible — the placement of those boundaries is not a detail.

### Candidates, all verified expressible at n_fft=512 / 16 kHz

| preset | bands | min bins | max bins | motivation |
| --- | --- | --- | --- | --- |
| `wesep_16k` | 32 | 3 | 64 | inherited baseline; the thing being tested |
| `uniform_32` | 32 | 8 | 9 | control — if this matches, non-uniform splitting does no work |
| `yu_truncated` | 27 | 6 | 41 | Yu et al. §4.2 progression truncated to 0-8 kHz; closest to a published plan |
| `bark` | 22 | 3 | 41 | Bark critical bands (Zwicker, 1961) — boundaries where the ear integrates energy and masking occurs |
| `mel_32` | 32 | 2 | 22 | mel spacing: the filterbank our proxy losses and every ASR front-end consume |
| `f0_focused` | 26 | 3 | 35 | maximum resolution across F0 and the first two formants, coarse above |

All six sum to 257 with no empty bands.

`mel_32` is the motivation most specific to this project and has no precedent in
the TSE literature: our objective is what a live model *understands*, so aligning
band structure with the representation that model listens through is an argument
only available to us, because only we have the metric.

### Two constraints on the experiment, both found by measuring

**1. Band count confounds capacity.** Band count ranges 22-32 across the
candidates. `SubbandNorm` holds one normalisation and projection per band and
`BandMasker` one MLP per band, so band count drives parameter count directly.
Comparing `bark` (22) against `wesep_16k` (32) varies boundary placement *and*
model size together, and a difference could not be attributed to either. **Any
band-plan ablation must either match band counts across arms or report parameter
counts alongside every result.**

**2. The band-plan and window-size questions are entangled.** At n_fft=512 one bin
is 31.25 Hz, and `mel_32`'s narrowest band is 2 bins — expressible, but thin. An
ERB-spaced 32-band plan is *not* expressible at all: its lowest edge lands near
27 Hz, under one bin, and is correctly rejected by the `min(widths) >= 1` guard.
At n_fft=1024 a bin is 15.6 Hz and both become comfortable. So perceptual plans
argue for the larger window independently of the latency headroom argument
(40 ms used of a 200-300 ms budget). Treat these as one experiment grid, not two.

### Implementation

`band_plan(sample_rate, n_fft, spec)` is a pure function — no state, no I/O —
taking a preset name or an explicit spec, returning bin counts. Four spec kinds:
`segments` (bandwidth + count), `edges_hz` (natural for perceptual scales),
`uniform`, `mel`. It asserts widths sum to `n_fft // 2 + 1` and that no band is
empty; both failure modes are config errors rather than code bugs, so segment
overrun and sub-bin bandwidths raise named errors rather than surfacing as shape
mismatches downstream.

Kept pure specifically so the plan can be swapped from YAML and regenerated at any
`n_fft` — the wesep reference computes it inline in the `BSRNN` constructor, which
is exactly what makes this ablation impossible there.

---

## 2026-08-18 — Normalisation: channel-wise LayerNorm, not BatchNorm and not GroupNorm

**Decision: per-band normalisation is channel-wise LayerNorm — `nn.LayerNorm` over
the channel axis at each time step. `GroupNorm(1, d)` is retained for a
non-causal/offline config only. We do NOT use the paper's BatchNorm.**

### This is a deviation from the paper, deliberately

Yu et al. (Interspeech 2023) §4.2 state: "We use layer normalization for offline
configuration and batch normalization for online configuration." So BatchNorm is
*their* online choice, and departing from it needs a reason.

The reason is not causality — BatchNorm is causal at inference, because it uses
frozen running statistics that depend on neither future frames nor other examples.
It is that BatchNorm couples examples within a batch during training and then
switches to population statistics at inference, so training and deployment
normalise differently. In a streaming model that mismatch compounds with
everything else we are already managing about chunk boundaries and recurrent
state. Channel-wise LayerNorm has no train/inference discrepancy at all.

### Why channel-wise rather than cumulative

| option | pools over | causal | stateful |
| --- | --- | --- | --- |
| `GroupNorm(1, d)` | channels **and time** | **no** | - |
| **channel-wise LN** | channels, current frame | yes | **no** |
| cumulative LN | channels, start to now | yes | yes |
| BatchNorm | batch (frozen at inference) | yes | running stats |

Channel-wise is chosen because it is **stateless**. Cumulative layer norm's
statistics depend on how much audio has been seen, so a 4 s training chunk would
normalise differently from a 60 s deployed stream — a real train/inference
mismatch — and it adds state to carry in the streaming cache. Channel-wise has
neither problem: frame *t* is normalised using only its own channels.

**Naming trap, worth getting right in the write-up.** wesep calls this `"cLN"`,
but its implementation is `ChannelWiseLayerNorm`: a transpose, `nn.LayerNorm` over
channels, transpose back. Conv-TasNet's cLN (Luo & Mesgarani, 2019) means
*cumulative* layer norm, a different operation. Same abbreviation, different
thing. **Cite ours as channel-wise LayerNorm, never as Conv-TasNet's cLN.**

### Measured: GroupNorm leaks the future

Perturbing the input at frame 50 of 100 and finding the earliest output frame that
changes:

| normalisation | first changed frame | verdict |
| --- | --- | --- |
| channel-wise LN | 50 | causal |
| `GroupNorm(1, d)` | **0** | leaks the future |

GroupNorm contaminates frame 0 — the *entire* output — because it pools statistics
over channels and time jointly. This is the silent failure mode: such a model
trains normally and scores normally offline, and only fails when streamed. The
test is kept as a regression check and must be re-run after any change to the
normalisation or the modelling stack.

### Why per-band at all

Speech energy falls roughly 6 dB per octave, so the lowest bins carry orders of
magnitude more energy than the highest. Normalising all 257 bins jointly would
leave the high bands numerically invisible and contributing almost nothing to the
gradient. Each band instead gets its own normalisation and its own 1x1 projection
to N=128, so a quiet 6 kHz band arrives at the sequence model with the same
representational budget as a loud 200 Hz one. This is a large part of why the
band-split architecture works, and it is the mechanism the band-plan ablation
(entry above) is really testing.

### Implementation note

`SubbandNorm.forward` uses `reshape`, not `view`, when flattening `(B, C, BW, T)`
to `(B, C*BW, T)`. `BandSplit` slices with `narrow()`, which returns a
non-contiguous view, and `view()` raises on it (verified). The wesep reference
calls `.contiguous()` inside its band split to make `view` legal; `reshape` avoids
needing that.

Output shape `(B, K, N, T)` = `(12, 32, 128, 503)` at our config; 70,916
parameters. This is the module that turns 32 unequal-width bands into one
stackable tensor, which is what allows a single RNN to run across bands.

---

## 2026-08-19 — Model sizing: 7.16 M parameters, deliberately below challenge scale

> **Headline figure superseded 2026-08-24 — quote 7,189,644 (7.19 M).** Every
> number below is the *pre-conditioning* count, as this entry's own "Conditioning
> is nearly free" section states. The decision itself stands unchanged; only the
> total moved, and by the amount predicted here. See 2026-08-24 below.

**Decision: `feature_dim = 128`, `hidden_dim = 192`, `num_repeat = 6`,
`mlp_hidden = 384`, `n_hidden = 1`. Total 7,156,234 parameters against the
REAL-TSE causal baselines' 25-27 M — a 3.5x reduction.** This satisfies the M1
requirement that the model be deliberately sized down and reported as such.

### Where the parameters are

| component | params | note |
| --- | --- | --- |
| `SubbandNorm` | 70,916 | per-band norm + 1x1 projection to 128 |
| `BandSequenceModel` (6 x BSNet) | 4,898,304 | 816,384 per block: 272,256 time + 544,128 band |
| `Estimator` (384 x 1) | 2,187,014 | 32 per-band trunks + mask and residual heads |
| **total** | **7,156,234** | |

`STFT` and `BandSplit` hold no parameters (the Hann window is a non-persistent
buffer).

### Two deviations from the reference, both toward smaller

**`hidden_dim = 192`, not 256.** 192 is the paper's stated LSTM width (Yu et al.,
Interspeech 2023 §4.2). The wesep reference uses `feature_dim * 2` = 256 instead,
which would put the separator at ~7.7 M rather than 4.90 M. We follow the paper.

**`n_hidden = 1`, not 2.** The paper states the estimation module's *width* (384)
but not its *depth*; wesep uses two hidden layers. We keep the paper's width and
use one hidden layer, so the deviation falls on the axis the paper left
unspecified rather than on a number it actually gives. This is where the saving
is: the second 384->384 convolution costs 147,840 per band, i.e. 4.7 M across 32
bands, for what is a per-band readout rather than sequence modelling.

### The scaling ladder, if the baseline underfits

Measured, so the cost of each step is known in advance:

| `mlp_hidden` | `n_hidden` | Estimator | total model |
| --- | --- | --- | --- |
| 256 | 1 | 1.46 M | 6.43 M |
| **384** | **1** | **2.19 M** | **7.16 M** (chosen) |
| 256 | 2 | 3.57 M | 8.54 M |
| 384 | 2 | 6.92 M | 11.89 M (paper width and wesep depth) |
| 512 | 2 | 11.32 M | 16.29 M (wesep) |

Rationale for starting at the small end: a baseline that trains is worth more than
a baseline that is faithful, and raising capacity is a one-line config change
whereas debugging a model too large to iterate on is not. If it underfits, step up
this ladder and record which rung and why.

Note the Estimator dominates at every rung — at the paper's 384 x 2 it would be
58 % of the whole model, larger than the six-layer separator. Capacity added there
buys per-band readout richness, not temporal or cross-band modelling.

### Conditioning is nearly free

TF-Map (M1, still to build) raises `SubbandNorm`'s input channels from 2 to 3,
taking it from 70,916 to ~104,000 — about 33 k, under 0.5 % of the model. It needs
no speaker encoder at all, unlike the embedding path. So the sizing above will not
move materially when conditioning lands.

---

## 2026-08-19 — Measured effective future dependency: 23.9 ms

**Decision: report 23.9 ms measured effective future dependency, with the
`n_fft - hop` convention stated. Supersedes the derived 40 ms figure in the STFT
entry above as the number to quote.**

The STFT entry required the reported latency be measured rather than derived.
Method: perturb the input waveform from sample 32,000 of 64,000 onward, run the
full model, find the earliest output sample that differs.

| quantity | value |
| --- | --- |
| first changed output sample | 31,618 |
| lead over the perturbation | 382 samples = **23.9 ms** |
| `n_fft - hop` | 384 samples = 24.0 ms |
| `n_fft + hop` (earlier, conservative) | 640 samples = 40.0 ms |

The measured dependency tracks `n_fft - hop`, not `n_fft + hop`. The 40 ms figure
quoted earlier was a conservative over-accounting: it added a hop of buffering to
the window fill, whereas the true dependency is set by how far forward the
overlap-add reaches, which is `n_fft - hop`.

**Independent agreement with the published system.** CARTSE reports 24 ms
algorithmic latency and a *measured* effective future dependency of 22.2-23.7 ms
(mean 22.9 ms). We measure 23.9 ms from an independently written implementation.
That corroborates both the framing and the measurement method, and it is the
convention to use when the number appears next to theirs.

For a deployed total, add one hop of buffering (8 ms) as CARTSE does: ~32 ms.
Against a 200-300 ms budget that leaves roughly 170-270 ms for compute, the
lookahead knob, and the live-model API round trip.

Test to keep: this measurement is the causal-correctness check for the whole
model, not merely a latency figure. It must be re-run after any change to the
STFT framing, the normalisation, or the sequence stack; a result materially below
382 samples means something has started reading the future.

---

## 2026-08-19 — TF-Map conditioning: Spectral Similarity, not Embedding Similarity

**Decision: TF-Map uses the Spectral Similarity variant (eq. 2). The Embedding
Similarity variant (eq. 3) is rejected because it requires encoding the live
mixture, which is not causal at any acceptable latency.**

Source: K. Zhang, J. Li, S. Wang, Y. Wei, Y. Wang, Y. Wang, H. Li, "Multi-Level
Speaker Representation for Target Speaker Extraction", Proc. ICASSP 2025,
doi:10.1109/ICASSP49660.2025.10889409, arXiv:2410.16059, sec II-A and Fig. 2.
(Local copy is filed as `...TSE2024.pdf` — the preprint is 2024, the venue is
ICASSP 2025. Cite the venue.) That paper uses BSRNN as its own backbone, so
combining the two follows its architecture rather than stitching together
unrelated work.

This is what makes the model a target speaker *extractor*. Everything before it
is a speech enhancer: it would clean up a two-voice mixture without any notion of
which voice to keep.

### What it does

`F_tfmap = B_e H` with `H = Softmax(B_e^T B_x)`. `B_e` is the enrollment
magnitude spectrogram used directly as NMF-style basis vectors — every enrollment
frame is a basis vector, rather than a learned dictionary. For each mixture frame,
cosine similarity against every enrollment frame is softmaxed into weights, and the
weighted sum of enrollment frames is "what the target's spectrum probably looks
like at this instant, reconstructed only from the enrollment". Energy is then
recovered by projecting the mixture magnitude onto the unit TF-Map frame, per the
paper. The result is concatenated as a third input channel beside real and
imaginary, so `in_channels` goes 2 -> 3.

**In both variants the output is `B_e H` and the basis vectors are always the
enrollment magnitude spectrogram.** Only the computation of the weights `H`
differs. The encoder in eq. 3 does not produce the cue; it only produces a better
similarity measure for choosing which enrollment frames to blend.

### Why not Embedding Similarity

The honest weakness of the spectral variant: **magnitude spectra are dominated by
phonetic content rather than speaker identity.** An interferer saying "ah" and the
target saying "ah" in the enrollment have similar spectral shape, and cosine
similarity in 257-dim magnitude space will match them. A speaker-embedding space
is trained to be invariant to what is said and sensitive to who says it, so it
would not.

The reason we still reject it: **eq. 3 needs `E_x`, frame-level embeddings of the
live mixture.** The enrollment side (`E_e`) is free, computed once offline. The
mixture side is not. ECAPA-TDNN's frame-level layers are symmetric dilated
convolutions whose receptive field extends on the order of 100+ ms in each
direction (order-of-magnitude; verify against the specific checkpoint before
quoting). That is roughly half of our 200-300 ms budget, spent on a better
similarity measure rather than on the model.

**Freezing the encoder does not help.** Frozen or trainable, a symmetric
convolution still reads the future. Making it causal would mean retraining the
encoder with causal convolutions — a separate project.

Evidence that the cheap variant is not a compromise: the paper's own finding is
that the spectral-level feature is the main driver of improved generalisation, and
in the REAL-TSE baselines the *causal* TF-Map variant achieves the best TER
(0.652 DEV / 0.801 EVAL1 / 0.808 EVAL2), beating even the non-causal
speaker-embedding baselines. TER is the published metric closest to our LCF
objective. See `literature/review_synthesis.md`.

**Recorded as a deviation, not an omission:** Zhang et al.'s full system uses a
pretrained ECAPA-TDNN. We use only their eq. 2. The justification is the streaming
budget, and any claim about our results must not be presented as reproducing their
full multi-level system.

### Where the encoder does still belong

An *utterance-level* speaker embedding of the enrollment is computed entirely
offline and therefore costs zero latency on the mixture side. It supplies the
identity information TF-Map's phonetic confound lacks, and remains the natural
ablation arm.

| conditioning path | causal | mixture-side latency | provides |
| --- | --- | --- | --- |
| Spectral TF-Map (chosen) | yes | **0 ms** | time-varying spectral hint |
| Utterance-level embedding | yes | 0 ms | speaker identity |
| Embedding-similarity TF-Map | **no** | ~100+ ms | better weights only |

### Measured

| quantity | value |
| --- | --- |
| parameters added | **33,410** (0.47 % of the model) |
| parameters in the TFMap module itself | **0** |
| enrollment frames (5 s) vs mixture frames (4 s) | 628 vs 503 |
| compute (TF-Map + enrollment STFT) | 6.2 ms, **0.4 %** of the forward pass |
| causality: perturb mixture frame 250 | first changed TF-Map frame **250** |

The 33 k comes entirely from `SubbandNorm`'s input growing from 2 channels to 3.
TF-Map itself is normalise, matmul, softmax, matmul, rescale — no weights.

Causal by construction: each mixture frame attends only over enrollment frames,
and the enrollment is fixed and fully available before the stream starts. No
mixture frame ever sees another mixture frame.

### Test to keep: the conditioning must be live

Substituting a *different speaker's* enrollment and measuring the relative change:

| quantity | relative change |
| --- | --- |
| TF-Map feature | 0.8123 |
| model output | **0.4785** |

**This is the most important test on the conditioning path.** The standard silent
failure in TSE is a model that quietly learns to be a plain enhancer while the
conditioning input is ignored — it trains, it converges, it produces clean audio,
and it extracts the wrong speaker. No shape assertion detects that. Re-run this
after any change to the conditioning path or to `in_channels`, and treat a
relative output change near zero as a failure.

---

## 2026-08-20 — M2 training objective: three terms, six deviations from CARTSE

**Decision: the M2 loss is**

```
L = (1 - w) * mean_present[ L_pres + w_m * L_MR ]  +  w * mean_absent[ L_abs ]
```

```
                            ||s_proj||^2
L_pres  =  -10 log10  ---------------------------------
                      ||s_hat - s_proj||^2 + tau*||s_proj||^2

                     ||s_hat||^2 + tau*||x||^2
L_abs   =   10 log10 -------------------------
                              ||x||^2

                <s_hat, s>
s_proj  =  ---------------- * s
                 ||s||^2

            1
L_MR    =  --- SUM_i [ || |S_i|^p - |S_hat_i|^p ||_1  +  || S_i - S_hat_i ||_1 ]
            I

tau = 1e-3        p = 0.3    I = 4, windows [128, 256, 512, 1024]
w = 0.458         w_m = set by measurement (below)
```

The present/absent switch is `crop_absent` from `dataset_loader.py`, **never the
manifest condition label** — 5.8 % of `both`/`target_only` crops land in target
silence (2026-08-18 entry), and branching on the label sends ~1 crop in 17 down
the `L_pres` path with an all-zero target, which is a `NaN`.

### Provenance

| term | source |
| --- | --- |
| `L_pres` floored SI-SDR | CARTSE Track 1 (Li & Seki, 2026) eq (1). SI-SDR itself: Le Roux et al., ICASSP 2019; as a separation objective: Luo & Mesgarani, TASLP 2019 |
| `L_abs` push-to-silence | CARTSE eq (2). Target-absent/false-alarm framing: Delcroix et al., Interspeech 2022 |
| `L_MR` multi-resolution | Yu et al., Interspeech 2023 eq (3) (`p = 0.3`, windows 10-40 ms). Multi-resolution STFT loss: Yamamoto et al., ICASSP 2020. Compression exponent: Braun & Tashev, 2021 (verify venue string before citing) |

CARTSE applied eqs (1)-(2) to real pseudo-labelled conversational audio; we apply
them to constructed LibriSpeech mixtures. **Same formulae, different data — no
number produced under this loss is comparable to a published REAL-TSE result.**

### Deviation 1 — `L_pres` floors on `||s_proj||^2`, not `||s||^2`

**Found by running the sanity test, not by reading the paper.** CARTSE eq (1)
floors the denominator on `tau*||s||^2`, tied to the *target's* energy. The
numerator `||s_proj||^2` scales with the output gain `g`; the floor does not. So
eq (1) as written is **not scale-invariant and has no lower bound**: a
perfect-shape output scaled by `g` scores `-20 log10(g) - 30`.

Measured, perfect-shape output, sweeping `g`:

| `g` | floor `tau*||s||^2` (eq 1) | floor `tau*||s_proj||^2` (ours) |
| --- | --- | --- |
| 0.05 | -3.98 | **-30.00** |
| 0.2 | -16.02 | **-30.00** |
| 1.0 | -30.00 | **-30.00** |
| 5.0 | **-43.98** | **-30.00** |
| 100 | **-70.00** | **-30.00** |

So eq (1) pays **unlimited reward for amplifying the output**, and its 30 dB
ceiling exists only at unity gain. On an imperfect output the drift is smaller
but present (-11.85 at `g=0.2` rising to -13.95 as `g -> inf`, converging on
plain unfloored SI-SDR).

Flooring on `||s_proj||^2` makes numerator and floor scale together, so they
cancel: **flat -30 dB at every gain, and the range really is `[-30, inf)`.**

Why fix it at the source rather than leaving it to `L_MR`: `L_MR` compares
magnitudes directly and so does pin the output gain — but the `w_m = 0` ablation
arm is required, and in that arm nothing else bounds it.

Cost of the fix: on total collapse (`s_hat = 0`) numerator and denominator are
both exactly 0, so `0/0` is `NaN` where eq (1) would have given `+inf`. `NaN`
survives any clamp, so **eps is added to both sides**, making collapse read
0.0 dB — finite, and worse than the ~-6 dB of passing the mixture through, so
not an attractor.

Not a criticism of CARTSE: their objective carries a mel-filterbank L1 and a
DNSMOS term that both pin the output gain, so the defect is masked in their
system. It is exposed in ours only because a `w_m = 0` arm exists.

### Deviation 2 — `L_abs` is normalised by `||x||^2`

CARTSE eq (2) is `eta * 10 log10(||s_hat||^2 + tau*||x||^2)`, which is **not
scale-invariant**: scale `x` and `s_hat` by `g` and the value shifts by
`20 log10 g`. Two loudness-matched absent trials that are both perfectly silent
therefore receive different losses and different gradients. `L_pres` is already
scale-invariant, so the pair was mismatched.

Dividing by `||x||^2` fixes it and buys a free anchor:

| `L_abs` | meaning |
| --- | --- |
| `0` | emitted the mixture unchanged — **did nothing** |
| `-10` | suppressed 10 dB |
| `-30` | floor: at or below 30 dB down, i.e. silent |
| `> 0` | **amplifying. A bug, not a bad score** — flag it in the run log |

`0` = do-nothing on every trial regardless of loudness, so the term reads without
knowing the trial's level. CARTSE's form has no such anchor.

### Deviations 3-5 — how the halves combine

**Deviation 3: masked means per half, not one batch-wide mean.** Averaging each
term over its own subset. Under a single batch-wide mean the absent half's share
of the gradient is whatever the batch happened to draw — 15 % at 1 absent crop in
12, 67 % at 6 — so the present/absent balance fluctuates step to step on sampling
luck.

**Deviation 4: `eta` is removed and folded into `w`.** Under the masked-mean form
`w` and `eta` appear only as the product `w * eta`, so `(0.297, 2.0)` and
`(0.594, 1.0)` give identical gradients. Two dials, one degree of freedom.
CARTSE needed `eta` because they used a batch-wide mean and it was their only
weight; we do not.

**Deviation 5: `w = 0.458`, not `0.297`.** The 2026-08-18 entry requires the
weighting use the measured crop-level absent rate 0.297. It does — inside the
calculation, not as the weight itself:

```
present coefficient = 1 - p       = 0.703
absent  coefficient = p * eta     = 0.297 * 2.0 = 0.594
                                     total mass = 1.297
w = 0.594 / 1.297 = 0.458
```

`w = 0.458` reproduces CARTSE eqs (1)+(2) with `eta = 2.0` at *our* measured
0.297. `eta = 2.0` was their deliberate choice to weight silence **above** its
data frequency; `w = 0.297` would silently discard that and is logged as the
data-frequency-neutral ablation arm instead.

Note the coefficients sum to 1, scaling the whole loss by `1/1.297` relative to a
batch-wide mean. Interacts with learning rate only — but CARTSE's `1e-4` is not
directly transferable because of it.

### Deviation 6 — `L_MR` window set straddles the model's own framing

**Windows `[128, 256, 512, 1024]` samples (8/16/32/64 ms), hop = window/4.**

CARTSE used `[512, 1024, 2048]`, all at or above our `n_fft`. Yu et al. used
10-40 ms. Ours brackets 512 deliberately: an STFT with a 32 ms window averages
everything inside 32 ms into one number per band, and the model builds its output
by masking 32 ms frames and overlap-adding, so its characteristic artefacts —
frame-boundary discontinuities, per-frame gain jumps, warble at the frame rate —
have exactly the structure a 32 ms analysis integrates away. The 8 and 16 ms
windows resolve them; 64 ms catches harmonic structure the short ones blur.

Powers of two throughout, unlike Yu et al.'s `[160, 320, 480, 640]`, so no
zero-padded windows.

**`L_MR` is applied to present crops only.** With an all-zero reference both L1
terms reduce to "minimise output energy", which duplicates `L_abs`'s job in
non-dB, unnormalised units and makes the effective silence weight unknowable.

**Reduction is `mean`, pinned in config and in the code comment.** `||.||_1` in
Yu et al. eq (3) literally means a sum over ~257 x 500 = ~128,000 coefficients,
while `auraloss` and the ParallelWaveGAN reference use a mean — a factor of
~1e5. Under sum-reduction with `w_m = 1.0`, `L_MR ~ 1e4` against
`L_pres ~ 1e1` and **Term 1 becomes numerically invisible with no error
message.** Published `w_m` values do not transfer unless the reduction matches.

`p = 0.3` is an empirical convention, not derived. It compresses ~60 dB of
in-frame dynamic range to ~8:1 so quiet high-frequency bins (fricatives,
sibilants, stop bursts) can compete for gradient.

**Measured, on a real chunk** (`train-42-010130`, 4 s at the highest-energy
offset; top of the spectrum ablated, low band and phase kept exact):

| output | energy kept | `L_pres` | `L_MR` |
| --- | --- | --- | --- |
| perfect (`s_hat = s`) | 100.00 % | **-30.00** | **0.0000** |
| >6 kHz deleted | 99.50 % | -22.28 | 0.0560 |
| >4 kHz deleted | 98.07 % | -16.86 | 0.1263 |
| >2 kHz deleted | 97.67 % | -16.06 | 0.1896 |
| >1 kHz deleted | 96.69 % | **-14.61** | **0.2314** |
| unprocessed mixture | - | -5.60 | 0.2535 |

**This is the justification for the term, and it is reportable as a result.**
A signal with everything above 1 kHz destroyed - a muffled mumble, every
consonant gone - keeps 96.7 % of the energy, so `L_pres` scores it **-14.61**,
i.e. 9 dB *better* than doing nothing. `L_MR` scores the same signal at 0.2314
against the do-nothing mixture's 0.2535, i.e. 91 % of the way to "you achieved
nothing" - the correct judgement, and one `L_pres` cannot reach at any weight.

Supersedes the earlier order-of-magnitude estimate in this entry (~1 % of energy
above 4 kHz, "SI-SDR still reads ~20 dB"). Measured: 1.93 % and -16.86 dB. The
argument holds; the numbers are now measured rather than projected.

Consequence: the `w_m = 0` ablation arm is not a formality. It is the arm in
which this blindness is live.

### `w_m` is set by measurement, not by sweep

**Measure both terms at `s_hat = x` (the mixture passed through) before
training.** Model-free, seed-independent, and roughly what the model does after a
few hundred steps. An untrained model is *not* a valid anchor — its output depends
on the random init.

Target `w_m * L_MR ~ 0.3 * |L_pres|` at that anchor. First measurement, one
real chunk (`train-42-010130`): `L_pres = -5.605`, `L_MR = 0.2535`, so
`w_m ~ 6.6` -- **not** CARTSE's 1.0, which would have put `L_MR` at ~4.5 % of the
present branch. Provisional until run over a few hundred crops and medianed;
`L_MR` varies with a trial's spectral content far more than `L_pres` does.

**The ratio drifts monotonically during training.** `|L_pres|` grows as the model
improves (-5.6 -> -30) while `L_MR` shrinks (0.25 -> 0), so `L_MR`'s share of the
loss value falls throughout. An early-strong / late-weak spectral term is
defensible, but it must be a stated choice rather than an accident, and it is a
further reason to log both terms every step. `L_pres` defines the task;
`L_MR` prices what it cannot see. Ablate `w_m` at `{0, 0.3x, 1.0x}` — the `0` arm
is required to show the term earns its place, and is direct thesis material if
`L_MR` moves LCF without moving SI-SDR.

Caveat to state when reporting: matching loss *values* is a proxy for matching
*gradients*. Record gradient norms for each term once at the anchor
(`torch.autograd.grad`). No paper in `review_synthesis.md` reports this.

### Not in the M2 loss

| term | source | status |
| --- | --- | --- |
| scenario-aware frame-level split | CARTSE eq (3) | **deferred to M4.** Needs frame-level `y`; see below |
| speaker consistency | CARTSE eq (4) | deferred. Using it forfeits SpkSim as a held-out number |
| mel-filterbank L1 | CARTSE eq (5) | deferred to M4 |
| frozen-encoder feature matching | CARTSE eq (7), PS4 (Ning et al., 2026) | deferred to M4 — the primary proxy |
| **DNSMOS maximisation** | CARTSE eq (6) | **rejected, permanently** |

**DNSMOS is rejected on two independent grounds.** The organisers found
DNSMOS-OVRL over-optimised to the point of ~zero human-MOS correlation on Track 1
(LCC +0.003) and swapped the official metric post hoc; `metric-definitions.md` §4
names this as the cautionary tale this project designs against, and CARTSE
explicitly trains on it. Second, it optimises perceptual quality, which is
explicitly not the objective (`CLAUDE.md`). Recorded as a rejection with the
citation, not an omission.

**Why M2 is conventional at all**, given the thesis argues conventional
objectives are the wrong target: the divergence between conventional metrics and
LCF *is* the finding, and it needs a competently-trained conventional arm to
diverge from; `research-plan.md` §5 requires the proxy models share a base
checkpoint with the baseline or the ablation is unattributable; SI-SDR carries
calibration a proxy loss does not, so a bad number means a bad model rather than
an ambiguity; and every proxy paper in the set (CARTSE, PS4, Ma et al.)
fine-tunes from a signal-loss checkpoint — none trains from random init.

### Consequences to carry

1. **`L_MR` reduction, and the complex-term convention, must be pinned in the
   config.** L1 on a complex tensor is ambiguous: `L1(real) + L1(imag)` or
   `sum |S - S_hat|` (modulus). They are different numbers. Yu et al. eq (3)
   leaves the complex term **uncompressed** — some of the literature compresses
   both. Follow the paper; state the choice in the code comment.
2. **Frame-level `y` for M4 is not free.** `target.wav` is exactly zero only
   where no utterance was *placed*. Within-utterance pauses (LibriSpeech carries
   ~0.331 s leading silence; 86.0 % of a file is speech, 2026-08-15) are room
   noise and reverb tail after RIR convolution, not zeros. Frame-accurate `y`
   needs `vad_segments.csv` mapped through `target_onsets_s`, and a rule for the
   reverb tail: up to `t60` (<= 0.6 s) of the target's own energy follows the
   last word, is present in the reference, and is rewarded by `L_pres` — a strict
   VAD label would mark those frames silent and have `L_TS` punish correct
   behaviour. Either extend active regions by `t60` or threshold on stem energy.
3. **Supervision is against the *reverberant* target.** `render.py` returns "the
   target through its own room, alone", so this loss asks the model to preserve
   the room, not dereverberate. Our SI-SDR is therefore not comparable to
   dry-target-supervised numbers.
4. Every weight (`tau`, `w`, `w_m`, `p`, the window list, the reduction) lives in
   `experiments/configs/`. None in the loss module.
5. Compute the loss in float32 even under AMP — the squared-norm sums and the
   logs are unreliable in fp16.

### Tests to keep

Implemented as `tests/test_losses.py` (30 tests, synthetic seeded signals
only, no corpora read). The gain-invariance test is verified to FAIL on
CARTSE eq (1) as published, which is what makes it worth keeping.

| assertion | expected |
| --- | --- |
| `L_pres(s, s)` | exactly `-30.0` at `tau = 1e-3`. Validates the whole implementation in one line |
| `L_pres` gain invariance | `L_pres(s, g*s_hat) == L_pres(s, s_hat)` for `g` in [0.05, 100]. **This is the test that caught Deviation 1** — it fails on CARTSE eq (1) as written |
| `L_pres(s, 0)` | `0.0`, finite. Total collapse must be neither `NaN` nor `inf` |
| `L_pres(s, x)` | the mixture's own SI-SDR, ~ `sir_db`. The floor anchor |
| `L_MR(s, s)` | `0.0` |
| `L_abs(0, x)` | `-30.0`, on any `x` |
| `L_abs(x, x)` | `10 log10(1 + tau)` = **`0.00434`**, not `0.0`. On any `x` |
| masked means | do not divide by zero when a batch holds 0 present or 0 absent crops |
| silent target | `L_pres(0, x)` is `NaN` **by design**. Assert it, so the masking requirement is pinned by a test rather than a comment |

**Log every term separately from step 1, and keep absent trials in `val`**
(0.35, `decisions-m0.md` 2026-08-11). A total loss that falls while `L_abs` sits
flat near `0` is a model passing the interferer straight through whenever the
target is silent — invisible in the total, invisible in SI-SDR, and visible at
eval only as a blown-up ICR.

---

## 2026-08-24 — Sizing figure: 7.19 M realised, and the conditioning prediction confirmed

**Quote 7,189,644 parameters (7.19 M) wherever the model size appears — thesis,
README, `meta.yaml` prose. Not 7,156,234 (7.16 M).** The 2026-08-19 entry's
figure was measured before TF-Map conditioning was built, which that entry says
explicitly. This is a realised-versus-planned update, not a correction of an
error.

Measured from the config that trains, via `build_model()`:

```
../tse_venv/bin/python -c "import sys,yaml;from pathlib import Path;\
sys.path[:0]=['.','scripts'];from train import build_model;\
m=build_model(yaml.safe_load(Path('experiments/configs/bsrnn_baseline.yaml').read_text()));\
print(sum(p.numel() for p in m.parameters()))"
```

### The 08-19 prediction was right

That entry predicted the delta rather than leaving it to be discovered:

| | predicted 2026-08-19 | measured 2026-08-24 |
| --- | --- | --- |
| `SubbandNorm` | ~104,000 | **104,326** (+326) |
| increase over the 2-channel count | ~33 k | **33,410** |
| share of the whole model | under 0.5 % | **0.46 %** |

`BandSequenceModel` 4,898,304 and `Estimator` 2,187,014 match the 08-19 table
exactly, so the entire difference is `SubbandNorm` and the cause is not in doubt:
TF-Map raises the band-split projection's input from 2 channels (`Xri`) to 3
(`Xri` + the TF-Map plane), and that projection is inside `SubbandNorm`.

### What does not change

- **The 3.5x reduction claim.** 25-27 M / 7.19 M is 3.5x, same as before.
- **The two deviations toward smaller** (`hidden_dim = 192`, `n_hidden = 1`).
- **The scaling ladder**, whose `total model` column is likewise pre-conditioning.
  Add 33,410 to every rung: the chosen rung is 7.19 M, and the paper-width /
  wesep-depth rung is 11.92 M rather than 11.89 M. The ordering and the argument
  for starting at the small end are untouched.
- **"Conditioning is nearly free"** — now demonstrated rather than projected, at
  0.46 % of the model and with no speaker encoder.

### Why this is logged rather than edited in place

The 08-19 count is what the sizing *decision* was taken against, and a decision
log that silently rewrites its own numbers cannot be audited. The stale figure
stays where it is with a pointer here.

---

## 2026-08-25 — The first full run collapsed to a mute; `tau` split, `mid` split added

**Decision: split `tau` into `tau_pres` (0.001) and `tau_abs` (0.01), and add a
`mid` split — 2,000 trials subset from the already-rendered `train` — as the next
training target. `w_m` is NOT changed.**

The 100-epoch smoke run (`experiments/results/2026-08-24-train-smoke-resume`)
drove total loss to -15.44 while `L_MR` got steadily *worse*, 0.279 -> 0.318.
Diagnosed on the saved checkpoint over the 20 val crops.

### What the model actually learned

| measurement | value |
| --- | --- |
| output energy vs mixture, present crops | -34.11 dB |
| output energy vs mixture, absent crops | -35.46 dB |
| present-minus-absent discrimination | **+1.34 dB** |
| gain that minimises `L_MR` | **30x** (-29.5 dB too quiet) |
| `L_MR` at that gain | 0.224, vs 0.319 as trained |
| output change when the enrolment is swapped | **-17.15 dB** |
| `L_pres` cost of a swapped enrolment | **+0.62 dB** |

A uniform mute that nearly ignores the enrolment. `L_pres = -7.18` looks healthy
only because SI-SDR is scale-invariant *and* satisfiable by generic
speech-shaped output on a two-speaker mixture — it is flattering the model twice.

### Why `L_MR` was the term that paid

`dL/d(log g)` at the operating point: `L_pres` **0.00000**, `L_MR` -0.031,
`L_abs` +1.999. `L_MR` is the *only* term that can see output gain, so as the
model learned silence `L_MR` recorded the bill. Over epochs 3-99,
`corr(val_L_MR, val_L_abs) = -0.967` — one variable seen twice.

Loss units delivered over the run: `L_abs` **-10.55**, `L_pres` -1.64,
`L_MR` **+0.20**. A *perfect* `L_MR` is worth 1.66 units, ~3 % of the range.

### Neither hypothesis on the table was right

- **Not `w_m` miscalibration.** 9.62 is correct for what it was calibrated to:
  a `L_MR`-vs-`L_pres` ratio at the do-nothing anchor. Two things make it miss —
  `grad_norms.csv` never measured the absent branch, and the anchor sits at
  `s_hat = x` where the gain is already right, so the attenuated region where the
  trade-off bites was never sampled. But no `w_m` fixes it: 243 would be needed
  at g=10, and at g=30 the required value goes *negative*.
- **Not capacity.** `train_L_MR` 0.278 vs `val_L_MR` 0.319 — a 0.04 gap on 48
  crops against 7.19 M parameters. A capacity-bound model memorises; this one
  does not even try. And the failure is one global scalar.

### Root cause

No enrolment conditioning -> cannot tell present from absent -> one shared gain
serves both branches -> correct level costs **+24 dB** on absent crops
(x `w` = ~+11 loss units) -> the mute is genuinely optimal.

**The objective is not broken.** With present/absent gains free it already
prefers the right answer: -16.47 at (g_pres=30, g_abs=0) vs -15.44 achieved. The
model cannot use it.

### `tau_abs` is a knob, not the fix — logged so it is not retried

Raising `tau_abs` was the first proposal and it is **measured powerless**: the
argmin along the shared-gain diagonal is g\*=0.3 for every `tau_abs` from 0.001
to 0.1. `L_abs` at correct gain is -5.90 dB at *all* of them. `tau` floors the
quiet end; what pins the model quiet is the penalty at the loud end.

Giving `L_pres` gain authority (plain SNR instead of SI-SDR) does work as
intended — spread over the gain range goes from 0.0000 dB to 6.92 dB — but the
diagonal optimum still lands on the mute. Not adopted; it is a real deviation
from CARTSE eq (1) and it does not solve this. Revisit only if conditioning
works and the gain is still wrong.

The split is kept anyway because a single `tau` for two differently-scaled halves
was conflating two things, and `-20 dB` is already inaudible suppression.
Consequence: the plot's total floor is no longer `10log10(tau)` but
`(1-w)*10log10(tau_pres) + w*10log10(tau_abs)` = **-25.42**, now computed by
`total_loss_floor()` in `scripts/train.py`. The hardcoded do-nothing anchor line
moves -2.240 -> -2.222; left as -2.24, noted in the code.

### The `mid` split

Speaker diversity is what conditioning needs, and smoke has **20** speakers.
`mid_train` is 2,000 trials subset from `train`'s 19,938 — **940 target
speakers**, condition mix held to within 0.03 % by proportional stratification
(the absent rate is what `w` was calibrated against, so it must not drift).
`mid_val` is `val` unchanged, 200 trials over 40 unseen speakers.

No new audio: `TrialDataset`'s `split` is only the directory under
`data/rendered/`, so `SPLIT_MANIFESTS` now carries `(manifest, audio_dir)`
separately and `mid` reads the already-rendered `train`/`val` trials. Widening
smoke's 20 speakers instead would have cost a re-render and still not tested the
hypothesis.

**Not run yet.** ~2.7 h/epoch on the laptop (projection from the measured
4.92 s/trial-epoch, batch 3, CPU) — i.e. ~3.4 days for 30 epochs. Intended for
Kaggle at batch 12.

### What to watch, instead of `L_MR`

`L_pres` is scale-invariant so it **cannot** show a mute, and `L_MR` shows it
only as a lagging side-effect. The leading indicators are enrolment sensitivity
(dB change when the enrolment is swapped) and the present-minus-absent output
energy gap. Neither is in `history.csv` yet.

### Caveats

15 present + 5 absent val crops, one checkpoint, one seed. The gain sweep is a
1-D slice holding the learned mask shape fixed.

---

## 2026-08-25 — Turning off the silence reward did not help. The model still ignores the voice sample

**Decision: stop changing the loss. The next thing to investigate is how the
voice sample is fed into the model (`src/models/conditioning.py`), not the
scoring.** Ran 10 epochs on `mid` with a warm-up schedule; the result was a
clean negative and it rules out the loss as the cause.

Result: `experiments/results/2026-08-25-train-mid-warmup/`. 5.4 h on a Kaggle
T4, batch 6, 1,950 s/epoch.

### The job, and what the model is actually doing

Every clip has two people talking over each other plus background noise. We also
hand the model a short sample of the voice we want. It should output only that
person.

It is ignoring the sample. The test: run the same clip twice, once with the
right person's sample and once with a stranger's. If the model were listening,
the two outputs would sound like different people. The output changes by
**2.6 %**. Same answer either way — so it is doing something generic to the
audio rather than picking out a person.

### Why it can score well without doing the job

The score rewards two separate things: sound like the target while she is
talking, and stay silent while she is not. That leaves two shortcuts, and
neither one needs the voice sample.

  1. **Say nothing at all.** Lose points on the first half, max out the second.
     This is what the 2026-08-24 run did (entry above).
  2. **Hand the recording back nearly unchanged.** The target is usually the
     louder of the two voices, so the original mixture already resembles her.
     Decent score for doing almost nothing.

### What was tried

Switch off the silence reward for the first 4 epochs (`w = 0`), then ramp it in
over 3. With shortcut 1 unavailable the model should be forced to actually learn
to pick the person out. Config: `loss.w_schedule`, implemented in
`w_at_epoch()`.

Thresholds were fixed **before** the run so the result could not be argued into
whatever we hoped for. Measured on how much the output moves when the sample is
swapped: better than -6 dB = worked; -6 to -10 = partial; worse than -10 = the
schedule is not the answer.

### What happened

**At the end of the warm-up (epoch 3): -15.86 dB.** It started at -15.98. Flat
across all four epochs. It never began listening to the sample.

It took shortcut 2 instead. Compared with just handing the recording back
unchanged (measured 2026-08-20: `L_pres` -5.909, `L_MR` 0.1842):

| | end of warm-up (epoch 3) | vs handing it back unchanged |
| --- | --- | --- |
| how close to the target | -6.657 | only **0.75 dB better** |
| second quality measure | 0.1900 | **worse** (+0.0058) |

Four epochs bought three quarters of a dB over doing nothing, and the second
measure never beat doing nothing at all.

Then the silence reward came back and so did shortcut 1: output on
target-silent clips fell to -18.3 dB below the mixture, and the second quality
measure went back up, 0.190 -> 0.236.

### The trap: the headline number improved the whole way

Total loss fell from -3.40 to **-10.74** across the run, and the best score was
the very last epoch. That reads as a successful run. It is not — the score
improved *because the model got quieter*. The thing making the number look good
is the thing making the model useless.

This is the second time that has happened, and it is why the two extra columns
now exist (`val_enrol_sens_db`, `val_pres_abs_gap_db`). They are the only
numbers in the log that told the truth. **Never report `val_total` from this
objective without checking them.**

### One thing that did improve

Against the 20-speaker run, this one had 940 and ends slightly less deaf to the
voice sample: **5.5 % vs 2.5 %**, and the loud/quiet gap reached 3.09 dB against
1.34. Real, small, pointing the right way. More speakers helps a little; it is
not the main problem.

### What this rules out

Two runs, two different schedules for the silence reward, same blindness to the
voice sample. Combined with the 2026-08-24 measurements — no value of `w_m`
works (it would need ~243 and flips sign at the correct volume), and `tau_abs`
does nothing (the score at correct volume is -5.90 dB at 0.001, 0.01 and 0.1
alike) — **the loss is not what is stopping the model from using the sample.**

The remaining suspect is the path the sample takes into the model. If swapping
it for a stranger's changes the output by a few percent, that connection may be
too weak to influence what comes out, and no amount of rebalancing the score
will fix that. See the 2026-08-19 entry on Spectral Similarity conditioning.

### Kept, even though the warm-up did not work

`loss.w_schedule` stays in the config, defaulting to a schedule but returning
the constant `w` when the block is deleted. It is cheap, it is tested
(`tests/test_w_schedule.py`), and it is the arm this entry reports — removing it
would make the result unreproducible.

One implementation note worth keeping: every `total` is computed at the **final**
`w`, never the epoch's own `w`. Otherwise the number means something different
each epoch, and two things that read it break — the learning-rate scheduler sees
the ramp as improvement and never steps down, and best-checkpoint selection
picks whichever epoch had the largest `w` rather than the best model.

### Caveats

200 validation clips, one seed, one run. `meta.yaml` and the final checkpoint
were not downloaded before the Kaggle session ended, so this result carries no
config hash or commit — see the `NOTE.md` beside it. The checkpoint on disk is
epoch 3, the end of the warm-up, which happens to be the state the decision
turned on.

---

## 2026-08-25 — The model ignores the voice sample because the data lets it. New `sir0` split

**Decision: stop changing the model and the loss. Build `sir0` -- `mid` with the
target/interferer loudness ratio centred on zero instead of 90 % target-louder --
and retrain. `mid` is kept as the control arm.**

A day of measurement, mostly ruling things out. Everything below is measured on
`mid_train` / `mid_val`, on CPU, with no training.

### The job, and what the model actually does

Each clip has two people talking over each other plus background noise. We hand
the model a 5 s sample of the voice we want. It should output only that person.

It ignores the sample. Run the same clip twice, once with the right person's
sample and once with a stranger's: the output changes by **2.6 %**. Same answer
either way, so it is doing something generic to the audio rather than picking out
a person.

### Why: the data answers the question without the sample

**90 % of two-speaker trials have the target LOUDER than the interferer**, median
+6 dB, because `regimes.base` narrows `sir_db` to [0, 12]. So "keep the loud
voice" is right ~90 % of the time -- and the model sees the mixture, so it gets
loudness for free.

Measured as a hit rate on "is the target the dominant voice at this moment?":

| who is louder | n | speaker cue | loudness (free) |
| --- | --- | --- | --- |
| interferer louder (sir < 0) | 66 | 59.2 % | 54.8 % |
| target louder 0-6 dB | 302 | 61.4 % | 66.9 % |
| target louder 6+ dB | 379 | 58.5 % | **81.5 %** |

The cue is flat across all three -- it tracks *who*, not *how loud*, which is the
right behaviour. Loudness climbs to 81.5 % where the target dominates, and 379 of
747 trials live in that bin. Faced with an 81 %-accurate free strategy and a
58 %-accurate one that must be learned, the model picks the free one. It is
behaving correctly; the task barely requires the enrollment.

`difficulty-dial.md` (2026-08-13) ranked `sir_db` #1 of 14 dials and stated the
mechanism exactly: "At -5 dB the interferer is louder than the target, so nothing
but the enrollment can identify which voice to keep." It framed the narrowing as
**difficulty** relief. It is also **relevance** relief. Those are different: a
task can be easy and still require the enrollment. That distinction is the thing
this entry adds.

### One real bug in the cue, found and fixed

`TFMap` compares every mixture frame against every enrollment frame and softmaxes
the scores. Softmax compares logits by DIFFERENCE, not ratio:
`w_i / w_j = exp(s_i - s_j)`. `F.normalize` (needed -- we want spectral shape,
not loudness) bounds every cosine to [-1, 1], so the largest achievable
difference was ~1 and the best-matching frame could never outweigh the worst by
more than `e^1 = 2.7x`. Spread over 628 enrollment frames that is nothing.

Measured: **619.6 of 628 frames effectively used**; the top frame held 0.22 % of
the weight against 0.16 % for a flat average. The softmax was averaging, not
selecting, so the cue was the enrollment's long-term mean spectrum -- varying
only 4.7 % over time.

Zhang et al. eq (2) is written on UN-normalised products, measured here at
0..932, a range that selects sharply on its own. Normalising removed the range;
`model.tfmap_scale` restores it. Default `sqrt(F)` ~ 16 at F=257. Worth **+3.5
to +5.3 points** on the hit-rate table above. Pinned by
`tests/test_tfmap_scale.py`, which tests the mechanism (`w_i/w_j == exp(scale *
(s_i - s_j))`) and not just the symptom.

### Ruled out, with numbers

- **The loss, and the silence reward.** Two runs, two schedules. The `w = 0`
  warm-up moved enrollment sensitivity by 0.1 dB over four epochs; the model
  switched from the mute to passthrough. See the entry above.
- **`w_m` and `tau_abs`.** `w_m` would need ~243 and flips sign at the correct
  volume; `tau_abs` is identical at 0.001, 0.01 and 0.1.
- **Enrollment length.** 5 s -> 10 s buys **0.8** points. 5 s -> 20 s buys the
  same 0.8. Four times the audio, no further gain. **Do not re-render for a
  longer enrollment.**
- **Enrollment EQ.** 0.3 points (1 % of the gap), and 49.2 % of trials carry it.
- **Reverb mismatch.** Giving the sample the trial's room at a mirrored source
  position: **0.0** points.
- **The cue's ability to recognise voices.** Given two clean clips it separates
  "same person, different words" from "different person" in **95.3 %** of 319
  pairs (97.4 % at scale 8). Spectral Similarity is a strong speaker
  discriminator; the 2026-08-19 choice of eq (2) over eq (3) is NOT the problem.

### A4 is doing real work -- keep it

The only enrollment arm that helped was giving the sample the target's **exact**
source position: +4.9 points, 14 % of the gap. That is positional fingerprinting,
which is precisely what A4 (2026-08-12, "the enrollment carries NO room") exists
to prevent. Worth recording that A4's stated reason -- matching on room instead
of voice -- is subtly wrong for a two-speaker trial, since both talkers share the
room and room-matching cannot separate them. The real cheat is POSITION, and A4
blocks it. Right decision, slightly wrong justification.

### The finding that keeps the problem open

The cue scores **95 % on clean clips and 56-61 % on mixtures**. An overlapped
frame is the SUM of two people, so it resembles neither alone; judging whether
the target dominates a frame almost requires having separated it first. So even
with the loudness shortcut removed the per-frame signal is weak. It may still be
enough -- the network has 7.19 M parameters and six LSTM layers, and the cue only
has to say WHICH voice to favour while the network separates -- but that is
reasoning, not evidence, and `sir0` is what tests it.

### `sir0`

`sir_db: [-10.0, 10.0]`, symmetric, so the shortcut becomes a coin flip. Width
kept wide rather than [-6, 6] deliberately: it leaves easy trials (target +10 dB)
to bootstrap on instead of making every trial equally hard. Realism cost is real
and acknowledged -- difficulty-dial.md calls -10 dB "plausible but uncommon".
The protocol gates only `overlap_ratio` behind supervisor agreement, not
`sir_db`, so this is a logged decision rather than an escalation.

Both regimes resolve to [-10, 10]: the split-level value is what `hard`
inherits, and the split's own `regimes` block omits `base.sir_db` so `base`
cannot re-narrow it. Every other parameter matches `train`, so the split differs
in exactly one axis.

`speakers_from: train` (new, `scripts/build_manifest.py`) borrows train's 1,172
speakers so `splits.yaml` -- generated and pinned before any data existed --
needs no hand edit. Speaker-disjointness is inherited from the borrowed split.

**`sir0_val` carries the same symmetric range as `sir0_train`.** Training on one
loudness distribution and scoring on another would measure neither. This departs
from the B4 note in `generator.yaml` that eval composition matches train, which
is why it is logged here.

### Caveats

One seed, one run per arm. The 66-trial interferer-louder cell is the population
the whole argument rests on and its cue-vs-loudness margin (+4.4) came back at
`mean/se = +1.1`, i.e. **not distinguishable from noise** -- so "the cue beats
loudness where loudness fails" is NOT established. What is established is the
shortcut's size (81.5 % vs 58 %) and its prevalence (90 %).

An attempt to confirm the model literally follows loudness was **inconclusive**:
the only checkpoint on disk is epoch 1 of the warm-up run (`w = 0`, 333 steps),
where passthrough behaviour is what the schedule was designed to produce. It
tracked the mixture at 0.937 against the target at 0.761, but at that epoch that
shows nothing. The epoch-9 checkpoint was not downloaded.

---

## 2026-08-27 — sir0 run: conditioning works, the mute does not go away. Adding `L_gain`

`sir0`, 1,989 train / 200 val, 8 epochs, Tesla T4, batch 3, 8.6 h, 3,875 s/epoch.
Seed 42, config md5 `2da2d7a9...`, bundle commit `67de944...-dirty`. `kaggle_out/`.

**The model learned who to listen for and still will not speak up.** It attenuates
rather than separates.

**Good, and real.** Enrolment sensitivity -14.82 -> -8.25 dB: a stranger's
enrolment now moves the output **39 %, up from 18 %**. That is what `a6baf77`'s
unbounded conditioning scaling was for. Only interpretable *because* of the
2026-08-25 `sir0` change -- on `mid`, 90 % of trials had the target louder, so a
model could look conditioned while tracking the loud voice.

**Bad.** 95.0 % of the total's improvement is the absent half (smoke was 93 %),
decomposed at the reporting w of 0.458:

| branch | ep 0 | ep 7 | change | share |
| --- | --- | --- | --- | --- |
| present `(1-w)(L_pres + w_m·L_MR)` | +0.437 | +0.069 | -0.369 | 5.0 % |
| absent `w·L_abs` | -1.582 | -8.592 | -7.010 | **95.0 %** |

`L_MR` ended **5.5 % worse than it started** (0.2407 -> 0.2540). It improved to
0.2231 across epochs 0-3 -- exactly the warm-up epochs where `w = 0` -- then
reversed the epoch the ramp began. Confound named: 0-3 is also early training.
But it is a reversal, not a slowdown, and it lands on the ramp.

Derived, not logged: `L_abs` -18.76 puts absent crops 24.8 dB below the mixture;
with the logged 2.45 dB gap, present crops sit at **-22.4 dB** (smoke: -24.9).

`L_abs` is at -18.76 against its -20 floor -- **94 % spent**. The cheap direction
is nearly exhausted.

**Correction to the first reading:** `L_pres` is scale-invariant SI-SDR, so -2.32
is **+2.32 dB** and the term improved 0.81 dB. Smoke went *backwards*. "Not
separating at all" was too strong.

### Decision: add `L_gain`, deadzone level match, default OFF

    L = (1-w) · mean_present[L_pres + w_m·L_MR + w_g·L_gain] + w · mean_absent[L_abs]
    L_gain = max(0, |20·log10(RMS_out / RMS_target)| - delta_db),  delta_db = 3.0

**Objective, not architecture or data.** The architecture just learned
conditioning; the data got harder in the right way and it learned anyway. The
rest is arithmetic: `L_pres` is scale-invariant by design (Deviation 1) and
cannot see a mute, `L_abs`'s optimum is zero output at weight 0.458, and the only
push-back is `L_MR` at effective weight 5.21. The model optimised the objective
correctly; the objective was wrong.

**Does not undo Deviation 1.** That bug was *unbounded one-directional* reward --
a correct output scaled by g scored -20log10(g) - 30, so amplifying paid forever.
`L_gain` is symmetric in log-level and minimised AT the correct level. Scale
variance was never the hazard; unbounded monotone reward was. `L_pres` stays
scale-invariant and measures shape alone -- two terms, two jobs.

**Deadzone ±3 dB** (amplitude 0.71x-1.41x): no gradient on sub-dB errors, and it
puts `.abs()`'s kink inside the zeroed region. **dB, not percent** -- 10 %
amplitude is 0.83 dB, *stricter* than a listener can resolve.

**Dataset-mean anchor rejected.** Trial levels vary by construction (BS.1770,
`sir_db` in [-10, 10], varying SNR and `target_activity_ratio`). A global mean
rewards making quiet targets louder and loud ones quieter -- automatic gain
control, a new degenerate solution -- and contradicts A1, whose reference is what
the mic heard, level included.

**Deviations to carry:** Deviation 7 is ours, not CARTSE's. RMS not BS.1770 (not
differentiable; both signals measured identically so the comparison stays
symmetric). Present crops only. Floor unchanged at -25.42: `L_gain` is 0 at
perfect reconstruction.

**Shipped OFF (`w_g = 0.0`)** but still computed and logged -- that is what the
anchor run reads. `test_wg_defaults_to_zero_and_reproduces_the_three_term_total`
pins that the default reproduces the 2026-08-20 objective, so the ablation's
control arm is a real control.

**Watch `pres_abs_gap_db`, not the total.** If it works the gap *widens*. If it
stays flat while both ends rise, the model traded a mute for a pass-through.

**Incidental:** during warm-up `w = 0`, so `L_gain` runs at full strength while
the silence pressure is off.

Code: `losses.py::_loss_gain_match`; `train.py` (`build_loss_fn`,
`HISTORY_FIELDS`, `add_parts`, `epoch_report`); `bsrnn_baseline.yaml`. Seven
tests. **`history.csv` schema changed** -- `train_L_gain`/`val_L_gain` are new,
so older histories cannot be concatenated without filling them.

---

## 2026-08-28 — `w_g` = 1.69 derived. And `L_MR` does not do its documented job

`scripts/derive_w_g.py`, 200 `sir0_val` crops, 6 min CPU, seed 42.
`experiments/results/2026-08-28-wg-anchor-sir0/`.

### The finding that matters more than the weight

**`L_MR` rewards the mute.** The 2026-08-20 entry and `losses.py` both called it
the term that "pins the output gain". Holding the audio fixed and changing only
volume:

| anchor | `L_pres` | `L_MR` | `L_gain` | `L_abs` |
| --- | --- | --- | --- | --- |
| oracle (= clean target) | -30.000 | 0.0000 | 0.0000 | -20.000 |
| pass-through (mixture) | -1.593 | 0.2735 | 4.901 | 0.043 |
| mixture muted to the checkpoint's level | -1.593 | **0.2438** | 17.960 | -18.818 |
| epoch-7 checkpoint | -3.326 | 0.2417 | 17.960 | -18.818 |

Muting ~21 dB **improves** `L_MR`, 0.2735 -> 0.2438. Nothing opposed the mute:
the absent branch paid 8.639, `L_MR` a further 0.155, and the only other
present-branch term is scale-invariant. **There was no fight to lose.** This
supersedes the 2026-08-27 wording that `L_MR` was "losing".

Why the earlier reading was wrong: `w_m = 9.62` was derived on `train` data where
the anchor read `L_MR` 0.1842 / `L_pres` -5.909; on `sir0` it reads 0.2735 /
-1.593. The magnitude was calibrated on a distribution that no longer applies and
the *sign* against attenuation was never tested.

**`passthrough_muted`** is the anchor that made this visible -- the mixture scaled
per crop to the checkpoint's level. Model-vs-pass-through moves gain *and*
separation and would price both. Two wiring checks passed: oracle `L_gain` =
0.000000, and `L_pres` drifted 1.19e-06 pass-through -> muted, which is
Deviation 1's scale-invariance confirmed by measurement rather than by reading code.

### Anchors the 2026-08-27 entry said were missing

Pass-through SI-SDR **+1.59 dB**; the checkpoint **+3.33 dB**; so the model beats
doing nothing by **1.73 dB** -- modest, real, and anchored rather than inferred.
Not comparable to `history.csv`'s -2.317 (different crop protocol); the
comparison *within* this run is valid, both rows using identical crops.

`L_gain` 17.960 puts the output **20.96 dB** off the target's level, consistent
with the -22.4 dB re-mixture figure inferred on 2026-08-27.

### The derivation

    buys on absent   :  w * (0.043 - (-18.818))            = +8.639
    costs via L_MR   : (1-w) * w_m * (0.2438 - 0.2735)     = -0.155  (a BENEFIT)
    L_gain headroom  :  17.960 - 4.901                     = +13.060
    break-even w_g   = (8.639 + 0.155) / ((1-w) * 13.060)  =  1.242

**Chosen: 1.69**, geometric midpoint of [1.242, 2.297], clear of both edges.

**The 2.297 ceiling is a heuristic, not a bound** -- where `w_g` x headroom
reaches `L_pres`'s full 30 dB range. Stated as heuristic because the *achievable*
range is nearer 10-13 dB, which would put it lower. 1.69 clears break-even by 36 %.

**Cross-check.** Differentiating w.r.t. a global attenuation g while both terms
are linear gives `d(total)/dg = -w + (1-w)·w_g`, zero at `w/(1-w) = 0.845`. That
marginal bound and the measured integrated break-even agree in magnitude; they
differ because `L_abs` saturates at -20 and `L_MR` supplies a bonus. Two routes,
same order.

### Consequences

- **Re-derive, never inherit.** Assumes the mute is global (justified at a 2.45 dB
  gap). A checkpoint with a real gate invalidates it.
- **`ablate_w_m` gains a reason.** Its 0 arm now tests a term measured to be mildly
  counterproductive.
- **Write-up:** "L_MR pins the output gain" must be corrected at source in the
  2026-08-20 entry, not quietly dropped. The correction is *why* a fourth term
  was needed.
- `--n-crops 300` yields 200: `sir0_val` has 200 trials and this script builds the
  dataset single-direction.

---

## 2026-08-28 — `L_gain` works. Current architecture FROZEN as the baseline

**Decision: `BSRNN_TFMAP` as it stands at `38bf48f` — TF-Map concatenated once as
a third input channel, no per-layer injection — is the baseline architecture for
M2.** Every architecture change from here is measured against it. Recorded
because the next change (D4a, per-block TF-Map re-injection) would otherwise
leave no fixed point to compare to.

Config `experiments/configs/bsrnn_baseline.yaml`, seed 42, split `sir0`,
7,189,644 parameters, 32 bands. Run in progress on Kaggle T4; `w_g` = 1.69,
`gain_delta_db` = 3.0, warmup 4 + ramp 3.

### Why now: the mute is fixed, and it is measured at a matched epoch

`w` is on the same schedule in both runs, so epoch 5 (`w` = 0.30533) compares
like for like against `2026-08-27-train-sir0`, whose only difference is
`w_g` = 0.

| epoch 5, `w` = 0.305 | control (`w_g`=0) | this run (`w_g`=1.69) |
|---|---|---|
| output level, present crops | **-19.4 dB** re mixture | **-4.6 dB** |
| `val_pres_abs_gap_db` | 2.06 | **2.63** |
| `val_enrol_sens_db` | -10.41 (9.1 %) | **-9.81 (10.5 %)** |
| `val_L_pres` | -1.942 | **-2.212** |
| `val_L_MR` | 0.2490 | **0.2080** |

**In plain terms: under the same silence pressure the old model went quiet and
this one did not.** The target sits ~3.9 dB below the mixture in `sir0`
(SIR ~ U[-10, 10]); this run's output sits at -4.6 dB, i.e. about right, while
the control sat 15 dB below where the target actually is and called it
separation. It also wins on reconstruction and detail, which `L_gain` was not
designed to touch, and its selectivity gap already exceeds the control's own
epoch-7 best of 2.45.

**Caveats, both load-bearing.** The two level figures are *reconstructed* from
`val_L_abs` and `val_pres_abs_gap_db`, not measured — they mix mean-of-dB with
dB-of-mean and are worth about +-1 dB. Quote them as indicative; measure
directly before they go in the thesis. And epoch 6 is the first at full
`w` = 0.458, so this is 2/3 pressure, not full. The control had already
collapsed by epoch 4-5, so the comparison holds, but confirm at epoch 7.

### What this baseline is, and is not

- **It is a strong extraction baseline.** No mute, correct output level,
  improving reconstruction.
- **It is a weak conditioning baseline.** 10.5 % enrolment sensitivity: roughly
  seven eighths of the output is still decided without reference to who was
  asked for. This is deliberate as a starting point — it makes D4a's effect
  measurable — but it must never be described as "conditioning works".
- **It is not yet the baseline of record.** Every number above is a training
  diagnostic. The project's primary metric is live-model content fidelity
  (`docs/data/metric-definitions.md`) and this checkpoint has never been scored
  on it. **A baseline that exists only as a loss curve is not a baseline.**
  Scoring it through the eval harness is the blocking next step.

### Consequences

- D4a (per-block TF-Map re-injection, parameter-free) is the first change
  measured against this. See `decisions-pending.md` D4.
- The `ablate_w_g` 0 arm is now also the architecture-baseline arm.
- `2026-08-27-train-sir0` remains the control for the `L_gain` claim
  specifically, not a general baseline: it is a muted model.

## 2026-08-28 — Training made 7x faster. Three changes, one of them the whole story

**Measured on the T4, batch 3: 4.741 -> 0.674 s/step.** Nothing about the model
changed. Evidence and the full sweep in `decisions-pending.md` E3d-E3f.

| # | change | where | gain |
|---|---|---|---|
| 1 | `chunk_s` 4.0 -> **4.008** | `bsrnn_baseline.yaml` | **4.44x** |
| 2 | mixed precision (`amp: true`) | `bsrnn_baseline.yaml`, `train.py` | 1.58x, 1.86x less memory |
| 3 | `pin_memory=True` | `get_data_loaders()` | small |

### 1. The 8 ms that bought 4.44x

**fp16 tensor cores require the batch dimension to be a multiple of 8.**
`BSNet.forward` reshapes to `(B*T, N, K)` for `band_rnn`, so its batch is
`examples x T`. At `chunk_s` 4.0 the STFT yields **T = 503, which is prime**, so
only batch sizes divisible by 4 aligned — and cuDNN silently fell back to a
non-tensor-core kernel for everything else. **4.008 s gives T = 504 = 8 x 63, so
every batch size aligns.**

Found by accident: batch 4 profiled 4x faster than 3, 5 and 6, reproduced six
times at 0.3 % spread, and was the only size whose `examples x T` divided by 8.
The three slow sizes agreed with each other to 0.4 % — one shared fallback kernel.

**The crop is 0.2 % LONGER, not shorter, `chunk_s` is a crop applied by
`TrialDataset` so nothing was re-rendered, and batch stays 3 so the optimisation
is untouched.** Moving to batch 4 would have worked equally well but changes
gradient noise and drops optimiser steps per epoch 663 -> 497; that is a
modelling change and this is not.

**T was mis-stated as 497 for three days.** `profile_step.py` used
`(n - n_fft)//hop + 1`, but `src/models/stft.py` pads by `n_fft - hop` on *both*
sides for overlap-add ramp room. It now measures T from the real STFT and prints
an ALIGNED / NOT ALIGNED verdict.

### 2. Mixed precision

fp16 for the model forward, fp32 for master weights and **the entire loss** --
`L_pres`/`L_abs`/`L_gain` carry 1e-12 epsilons inside `log10` and fp16's smallest
normal is ~6e-5, so an fp16 loss returns NaN. `amp_ctx()` is the single
definition used by train, val and the diagnostic so they cannot drift apart.

Two correctness details that are easy to get wrong: **`scaler.unscale_()` must
precede `clip_grad_norm_`** or the clip compares a ~65536x inflated norm against
`grad_clip` and crushes every gradient to zero — training looks stable and learns
nothing; and the diagnostic's swapped forward is cast back with `.float()` before
its sums of squares, which would otherwise overflow fp16's 65504 ceiling.

Val runs in the same precision as training deliberately: a metric measured in a
precision the model was not trained in describes a model that does not exist.

### Consequences

- **Epoch time is 3.9-7.0x better, not 7x.** The compute step is 7.03x, but of
  the measured 3875 s/epoch only ~3143 s was training; ~516 s is unaccounted
  overhead that may not scale. Projected 550-1000 s/epoch, so 10 epochs in
  **1.5-2.8 h against 10.8 h**. First real run settles it — do not quote 7x for
  an epoch.
- **`amp` is config-driven, not hardcoded**, because it changes training numerics:
  a `history.csv` is only readable next to the flag that produced it. Runs before
  and after this entry are not numerically comparable.
- **Gradient checkpointing withdrawn** and the band-loop refactor dropped: E3e
  measured throughput per example flat across batch sizes, and E3b measured the
  32-band loops at 4 % of forward.
- **Audio compression is settled as worthless** -- loader measured at 0.044
  s/batch on Kaggle, 4.5 % of a step.
- 189 tests pass. The 4.008 s crop and `amp: true` are both live in
  `bsrnn_baseline.yaml`, so the next run uses them.

## 2026-08-28 — The 10-epoch `L_gain` run: conditioning finally works

`experiments/results/2026-08-28-train-sir0-e10/`. `sir0`, seed 42, `w_g`=1.69,
warmup 4+3, batch 3, fp32, 10.5 h on a T4 (pre-speed-fix).

### In plain words

**Blocking the model's escape route taught it to listen to the voice sample.**
Swapping in a stranger's enrolment now changes the output by **37.6 %**, against
2.9 % at the start of this run and 14.9 % for the control that had no `L_gain`.
The model also finally knows when to speak: it is **7.1 dB louder on crops where
the target is talking** than on crops where it is not, against 2.45 dB for the
control.

### Head-to-head at epoch 7, both at full `w` = 0.458

| | control (`w_g`=0) | this run (`w_g`=1.69) |
|---|---|---|
| enrolment sensitivity | 14.95 % | **16.88 %** |
| present/absent gap | 2.45 dB | **2.79 dB** |
| **output level, present crops** | **-22.4 dB** re mixture | **-4.2 dB** |
| `val_L_MR` | 0.254 | **0.212** |

The target sits at ~-3.9 dB in `sir0`. **The control was 18 dB below where the
target actually is — it had muted and called it separation.** This run sits at
-4.2 dB, i.e. right.

### The trajectory, and why it matters

| epoch | `w` | enrol % | gap dB | e_pres dB |
|---|---|---|---|---|
| 3 | 0.000 | 5.3 | 1.63 | -3.77 |
| 5 | 0.305 | 10.5 | 2.63 | -4.58 |
| 7 | 0.458 | 16.9 | 2.79 | -4.22 |
| 9 | 0.458 | **37.6** | **7.10** | -4.72 |

**Conditioning improved as a CONSEQUENCE of removing the mute, not because
anything in the conditioning path changed.** With silence available it was free
— `L_pres` is scale-invariant, `L_MR` was measured to reward muting, `L_abs`
rewards silence — so going quiet paid and using the enrolment did not. `L_gain`
priced the escape; the model then had to actually discriminate. **The
architecture was never the problem here; the objective was.** That is the
defensible claim, and it is the strongest result the project has.

### Three things this does NOT say

1. **It is not converged.** `best_val` is epoch **9, the last one**, and
   `early_stopped=False`. Enrolment sensitivity went 28.0 -> 37.6 % on the final
   epoch alone; the curve is still climbing steeply. **Training longer is now the
   cheapest available experiment** — 8.4 min/epoch after the speed fix.
2. **`L_gain` did not achieve its literal objective.** It fell only 3.902 ->
   3.782 (-3.1 %), so per-crop level error is still ~6.8 dB. The output level is
   right *on average* (-4.72 vs -3.9 ideal) while individual crops remain badly
   off — the model is not tracking level per utterance, it has just stopped
   muting. The term worked as a *constraint*, not as a *regression target*.
3. **Still no number on the project metric.** Every figure here is a training
   diagnostic.

### Consequences

- **D4a drops in priority.** It was proposed when enrolment sensitivity read
  10.5 % at epoch 5 of this same run and the diagnosis was "the network is
  throwing the cue away". At 37.6 % that diagnosis no longer holds. 62 % of the
  output is still enrolment-independent so there is headroom, but this is no
  longer the emergency it looked like. Order now: **train longer, then score on
  the metric, then reconsider D4a.**
- The `w` warmup schedule is vindicated: every jump in enrolment sensitivity
  tracks a `w` increase (5.7 -> 10.5 -> 21.0 % across epochs 4-6).
- Epoch 7 is a visible regression on every diagnostic (21.0 -> 16.9 %, gap
  3.69 -> 2.79) before recovering. Single-epoch noise at this scale; do not read
  a trend into any one epoch.

## 2026-08-29 — AMP validated over a full 10 epochs. Adopt it

`experiments/results/2026-08-29-train-sir0-e10-amp/`. Same seed (42), split and
`w_g` as the fp32 run; `amp: true` and `chunk_s` 4.008 are the two differences.

**523.0 s/epoch against 3772.6 = 7.21x. Ten epochs in 1.45 h, was 10.5 h.**

### The thing that needed testing, and passed

The 2-epoch speed check only reached `w` = 0. **Epochs 4-9 — the `w` ramp and
full silence pressure, where `L_abs` starts driving and the earlier model
collapsed — had never run in fp16.** They now have: all 10 epochs completed,
`mixed precision: ON (fp16 forward, fp32 loss)` confirmed in the log, and **zero
NaN, inf or error mentions across the whole session**. The `1e-12` epsilons are
safe behind the fp32 loss cast.

### AMP vs fp32, enrolment sensitivity %, epochs 5-9

|  | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|
| AMP | 8.5 | 24.1 | 20.4 | 28.2 | 31.9 |
| fp32 | 10.5 | 21.0 | 16.9 | 28.0 | 37.6 |

fp32 finishes ahead (37.6 vs 31.9 %, `val_total` -1.673 vs -1.413), **but the
difference is inside the noise**:

- mean |between-run gap| **2.9** points, against mean |within-run epoch-to-epoch
  swing| **8.3**;
- **the sign flips** — AMP is ahead at epochs 6, 7 and 8, behind at 5 and 9.

A systematic precision penalty would not change sign three times. **Do not claim
either run produced the better model.**

**Caveat on that claim.** Epoch-to-epoch variance within one run is a *proxy*
for run-to-run variance, not the same quantity, and n=2 runs supports no real
statistics. Two variables also differ, not one (`amp` and `chunk_s`), though both
are tiny perturbations. Settling it properly needs several seeds — which now
costs 1.45 h each rather than 10.5.

### Consequences

- **Adopt AMP.** It is already the config default; this is the evidence for it.
- Both runs have `best_val` at epoch **9, the last**, and neither early-stopped.
  **Still not converged, in either precision.** Training longer remains the
  cheapest experiment on the board.
- `model_sir0.pt` stays the fp32 checkpoint on best `val_total`; the AMP one is
  `model_sir0_amp-e10.pt`. See `models/README.md` — the choice is arbitrary
  within noise.

## 2026-08-29 — Trained longer. It memorises the training set

`experiments/results/2026-08-29-train-sir0-e50-resume/`. Resumed the AMP epoch-9
checkpoint, `sir0`, seed 42, `w_g`=1.69, batch 3, `amp: true`, target epoch 50.
Ran epochs 10-24 and **early-stopped on patience 10**; best `val_total` at epoch
**14**, −2.178. 15 epochs in 2.20 h at 527.8 s/epoch.

### In plain words

Both `L_gain` runs stopped with their best score on the last epoch and neither
early-stopped, so **"just train it longer" was the cheapest experiment on the
board** and this run was it. It answered the question, and the answer is that
there is nothing left to gain from more epochs on this data: the model spends
them learning the 1,989 training trials by heart. It gets better every epoch on
audio it has already seen and steadily worse on audio it has not.

### The two curves that diverge

Separation, as SI-SDR in dB (`L_pres` negated, higher better):

| epoch | train | held-out | gap |
|---|---|---|---|
| 10 | 2.97 | 1.52 | 1.45 |
| 12 | 3.24 | 2.18 | 1.06 |
| **14** | **3.38** | **2.14** | **1.24** |
| 17 | 3.70 | 1.46 | 2.24 |
| 20 | 4.53 | 0.98 | 3.55 |
| 24 | **5.51** | **−0.17** | **5.68** |

Train improves monotonically across all 15 epochs. Held-out improves for four,
then falls the rest of the way — and **by epoch 24 it is below the pass-through
anchor**, i.e. worse than not running the model at all. `L_gain` splits the same
way: 2.92 → 1.51 on train, 3.77 → 5.79 held-out. The generalisation gap widens
by a factor of four.

### The headline that moved for a bad reason

`val_enrol_sens_db` went −3.66 → −0.98 dB, i.e. **43 % → 80 %** output movement
on an enrolment swap, climbing all the way through the collapse. Taken alone that
reads as conditioning tripling. It is not. **An output that has stopped
resembling the target moves a lot when you perturb its input, and the diagnostic
cannot tell that apart from discrimination.**

**Consequence for the write-up: `val_enrol_sens_db` is only interpretable
alongside held-out `L_pres`.** Quote 37.6 % (epoch 9) and 41.7 % (epoch 14).
Never quote the 80 %.

### What this rules out, and what it rules in

**Rules out an under-capacity architecture.** A 7.19 M-parameter model that can
drive training separation to 5.51 dB on 1,989 trials is not too small for the
task; it is too large for the data. This inverts the standing instinct that the
backbone needs replacing — **a bigger or richer model overfits sooner, not
later.** It does not rule out a *better-shaped* model; it rules out a bigger one
as the response to this particular result.

**Rules in data volume as the binding constraint.** Random cropping already
varies the window per epoch (`_crop_offset_start` keys on `(seed, epoch, idx)`)
and `both_directions` already doubles the examples, so the effective set is about
24,000 four-second crops and it still memorises. 19,938 trials are rendered but
only 1,989 are in `sir0`, which is the split where the loudness shortcut is
closed — so the larger set has to be re-rendered symmetric, not just pointed at.

### Consequences

- **`models/model_sir0.pt` should now be the epoch-14 checkpoint**, not epoch 9
  of the e10 run. Everything after epoch 14 is a worse model on held-out data.
- **Next training run is more data, not more epochs and not more parameters.**
  ~5,000 `sir0` trials is ~1,315 s/epoch, so 20 epochs in 7.3 h — inside Kaggle's
  cap. Retrain from scratch: resuming would carry the memorisation forward.
- **Two regularisers are free and currently off.** `weight_decay` is 0.0 by
  explicit choice (`bsrnn_baseline.yaml`) and there is no dropout anywhere. Both
  are config-level. Neither substitutes for data, but both are one line.
- **Early stopping fired, contradicting the M2 checklist note** that it would not
  fire as configured. `patience: 10` on `val_total` stopped the run at 24 from a
  best of 14. That checklist item can be closed.
- **The held-out margin over doing nothing is thin.** Best held-out separation is
  2.14 dB. The pass-through anchor of +1.59 dB was measured by a different
  protocol (`derive_w_g.py`, 200 fixed crops) than the validation loop, so the
  two are not directly subtractable — **re-run the anchor script on the epoch-14
  checkpoint before quoting any "beats doing nothing by X dB" figure.**
- Not logged as an ablation arm: the run is a resume of the AMP run, so it shares
  its seed and its history. It is one trajectory, not an independent sample.

## 2026-08-30 — Enrollment bank: rotate the identity cue per epoch (D8a)

Response to the 2026-08-29 overfitting entry. Implemented, tested, unrun.

### The specific hole it closes

`enrollment.wav` is rendered once and read **in full** on every epoch
(`dataset_loader._example`, no crop on the enrollment path). So across the 24
epochs of the 2026-08-29 run, every trial presented the model with the *same
5 s waveform* as its identity cue. Random cropping does not touch this: it
rotates the mixture window, i.e. it resamples the same acoustic scene, while
the thing the model is supposed to generalise over — who is speaking — stayed
bit-identical.

**That makes "this exact waveform -> this exact voice" a lookup table over 1,989
entries, and fitting it is sufficient to explain the observed failure**: training
separation 2.97 -> 5.51 dB while held-out fell 1.52 -> -0.17 dB. Val speakers are
disjoint by construction (`splits.yaml`), so the table transfers nothing.

### What was built

`scripts/render_enrollment_bank.py` renders K enrollment recordings per trial per
direction, `enrollment_v00.wav` .. `v{K-1}.wav`; `dataset_loader` picks one per
`(seed, epoch, idx, direction)`; `data.enrollment_variants` in the config selects
K. **Additive only** — mixtures, targets and interferers are untouched, so no
existing audio, manifest or checkpoint is invalidated.

Variants are distinct **utterances** by the same speaker, not distinct windows of
one utterance: different sentences, usually a different chapter or book, and a
different recording session. A window of the same utterance would leave the
channel and the session identical and is the weak version of this fix.

### Five properties that are deliberate, not incidental

1. **`v00` reproduces `enrollment.wav` byte for byte** — same utterance, same
   offset, same EQ seed. Verified over 24 banks. Without this, `K=1` vs `K=4`
   would compare two different renders and nothing could be attributed to the
   augmentation.
2. **Every variant is levelled to the trial's own `target_loudness_lufs`**, so
   which variant is in play cannot be read off loudness. Verified: spread
   under 3 dB across a bank. Level is a cue closed everywhere else and this does
   not reopen it.
3. **The B8/B10 guard tiers are applied per variant**, through the *same*
   `pick_enrollment` the manifest builder uses, re-evaluated at each draw with
   the already-taken utterances removed. Measured on 12 trials: 32 book-tier,
   40 chapter-tier, 24 utterance-tier draws. The content-leak guarantee holds
   variant by variant rather than only for the first.
4. **Each variant gets its own EQ curve** (the CARTSE channel-gap augmentation,
   Li & Seki 2026, already in the renderer), seeded from `trial_id#v{k}`. So the
   bank varies channel as well as content.
5. **Validation never rotates.** `random_crop=False` forces
   `enrollment_variants` to 1, so val reads `enrollment.wav` itself. A val set
   that moved would make every val number in the project's history incomparable,
   and it is the held-out curve that this whole change is trying to move.

### Cost, and why K defaults to 4

`(K-1) x 2 x ~160 kB` per trial. K=4 on `sir0_train` is **+1.9 GB, ~45 % on top
of the split**, and **~38 min to render** (measured: 100 trials in 2 min, 8
workers). K=8 would double the addition. 4 gives each variant roughly 4-5
appearances over a 20-epoch run, which is enough to break a fixed lookup; the
binding constraint on raising it is the Kaggle dataset upload, not the renderer.

Every `sir0_train` speaker has at least 10 utterances of >= 5 s (median 109), so
no speaker is short of candidates at K=4. Speakers who cannot fill a bank get a
shorter one and the loader falls back to `v00`; the script reports the count
rather than padding with repeats, because a silent repeat would weaken the
augmentation exactly where the speaker is rarest.

### What this is NOT

**Not a substitute for more scenes.** It varies the identity cue and nothing
else: the same 1,989 speaker pairs, rooms, noise beds and SIR/SNR draws remain.
It attacks one memorisation route, the one that is cheapest to close and most
specific to the conditioning failure this project has been chasing since
2026-08-25. Expect it to narrow the train/held-out gap, not to eliminate it.

**Not yet evidence of anything.** 9 unit tests pass and the renderer is verified
on real data; no training run has used it.

### How it will be judged

Run `K=1` and `K=4` from scratch on the same seed and split. The claim is
supported if the **train/held-out separation gap at a matched epoch narrows**.
Held-out `L_pres` improving is the outcome that matters; `val_enrol_sens_db`
alone is not admissible evidence here, for the reason recorded on 2026-08-29 —
it rose through the last collapse.

## 2026-08-30 — Per-epoch SIR/SNR remix (D8b). No re-render needed

Second response to the 2026-08-29 overfitting entry, and it composes with the
enrollment bank above. Implemented, tested, unrun.

### The idea in one line

Every trial's loudness balance was a random draw made once, on 2026-08-26, and
then frozen into `mixture.wav`. **The ingredients are still on disk, so the draw
can be made again at load time** — same voices, same words, same room, different
difficulty, every epoch.

### Why it needs no new audio

`render_trial` sums three signals, and two of them are written out, so the third
is recoverable:

    noise = mixture - target - interferer

**Exactly, on every trial**, including the 69 of 1,989 (3.5 %) where A6's clip
guard fired — because A6 applies its common gain to the mixture *and* both stems
(`render_trial` step 4), so the equality survives it. An earlier draft of this
entry claimed those trials needed dividing by `common_gain` first; they do not,
and no `meta.json` read is required.

The rebuild is then two scalar multiplies:

    interferer *= 10 ** ((sir_rendered - sir_new) / 20)
    noise      *= 10 ** ((snr_rendered - snr_new) / 20)
    mixture     = target + interferer + noise

**Only the mixture changes.** `target.wav` is the training reference and is
returned untouched, unless the new sum would clip — in which case it takes the
same common gain the mixture does, which is A6's own rule and keeps the
output/reference level relationship `L_gain` measures intact. Applied to the
crop rather than the clip, because the crop is all the loader has; recorded as a
deviation from the renderer, which guards per clip.

### The new levels are RESAMPLED FROM THE MANIFEST, not from generator.yaml

Each trial borrows another trial's `(sir_db, snr_db)` from **the same difficulty
regime**. Three reasons, and they are the argument for the design:

1. Every value is one the generator actually produced, so no epoch can train on
   an out-of-distribution mixture. Sampling from declared ranges could.
2. The regime mix and any within-regime correlation between SIR and SNR survive
   for free.
3. It needs no second copy of the sampling config to drift from the one the
   manifest was built with. `sir0_train` overrides `sir_db` at the split level to
   `[-10, 10]` while the `hard` regime is derived rather than declared, so a
   re-implementation of that resolution is exactly the kind of duplicate that
   goes stale.

**This does not reopen the loudness shortcut.** `sir0` is a *symmetric* range,
not a pinned value; the shortcut came from the `base` regime's asymmetric
`[0, 12]`, where the target was louder 90 % of the time. Drawing from the same
symmetric pool preserves the property the split exists to enforce.

### Which trials are eligible, and why the rest are not

| condition | n | SIR redrawn | SNR redrawn |
|---|---|---|---|
| `both` | 984 | yes | yes |
| `target_only` | 513 | no interferer to rebalance | yes |
| `interferer_only` | 376 | no | no |
| `noise_only` | 116 | no | no |

**1,497 of 1,989 trials (75.3 %) get the augmentation.** The target-absent
quarter passes through as rendered: on those the loudness anchor is the
interferer or the noise itself (2026-08-11), so the recorded numbers are not
target-relative and re-applying them would be meaningless arithmetic.

### The I/O cost is negative, not positive

The remix needs both speaker stems, so it adds one windowed read per trial. But
`__getitem__` was restructured to read the shared mixture crop **once** instead
of once per direction — the two directions share the crop by construction, so
the second read was always a duplicate.

| | windowed reads per trial |
|---|---|
| before | 6 (mixture x2, target, interferer, enrollment x2) |
| after, remix on | **5** |

Unmeasured for this change. The recorded loader cost is 0.044 s/batch, 4.5 % of
a step (2026-08-28), and `scripts/profile_step.py` is what would settle it.

### Consequences

- `data.remix_gains`, default `false` = the pre-2026-08-30 behaviour. Pinned off
  on any fixed set: `random_crop=False` forces it off, so validation never
  changes difficulty and its curve stays readable.
- **`meta["sir_db"]` and `meta["snr_db"]` now report the REALISED levels**, which
  differ from the manifest whenever the remix fired. Any stratified diagnostic
  therefore describes the audio the model heard, not the audio on disk.
- `CLIP_CEILING` is duplicated in `dataset_loader.py` rather than imported, to
  keep scipy and pyloudnorm out of every DataLoader worker. A test pins the two
  values together.
- Restructuring `__getitem__` is behaviour-preserving when the remix is off, and
  a test asserts the returned tensors are unchanged.

### How it will be judged

Same rule as the bank: `remix_gains` on vs off, same seed and split, and the
claim is supported if the **train/held-out separation gap at a matched epoch
narrows**. The two are separable arms and should be run as such before being
combined, or a joint improvement cannot be attributed.

**What it does not do.** It varies difficulty, not diversity: still 1,172
speakers, still 1,989 rooms, still 1,989 pairs of sentences. It is not a
substitute for rendering more trials, and the learning curve over dataset size
is still the measurement that decides whether more are needed.

## 2026-08-30 — C2 CLOSED. Task difficulty measured at n=230 and accepted

The last open M0 item. Scored from `experiments/results/transcripts.csv`
(`faster-whisper small.en`, int8 CPU, greedy, Whisper `EnglishTextNormalizer`
per B5), which already held 1,220 trials transcribed clean and mixed. No new
ASR was run — only the scoring, which had never been done at scale.

### What C2 asks

How hard the task should be, as two numbers: the **floor**, an off-the-shelf
ASR's word error rate on the raw mixture (how much of the target is lost if you
do nothing), and the **ceiling**, its WER on the clean target (the best anyone
could do). The gap between them is the headroom the extractor works in. A floor
too low makes the task trivial and the metric unable to separate systems; too
high and everything scores badly and again nothing discriminates. The declared
target band was 60-80 %.

### Measured, `both` condition only

| set | n | ceiling (clean) | floor (mixture) | mean SIR | interferer louder |
|---|---|---|---|---|---|
| `eval_public` | 230 | 6.1 % | **57.4 %** | +4.9 dB | 26 % |
| `sir0_val` | 103 | 5.8 % | **65.2 %** | -0.7 dB | 54 % |

**Decision: the measured range 57.4-65.2 % is accepted as the task difficulty.**
It straddles the lower edge of the 60-80 % band, and the band was a target set
before any data existed, not a constraint. Nothing is re-rendered and
`overlap_ratio` stays un-narrowed (B1 says narrow it last).

Plain reading of 57.4 %: for every 100 words the target speaker said, about 57
come out wrong. The failure is not mush — inspected on
`eval_public-42-000132`, the ASR transcribes the target perfectly for 17 words
and then **switches to the interferer's sentence**. The number is measuring "the
machine listened to the wrong person", which is exactly what the extractor is
built to prevent. The 51-point gap from 6.1 % to 57.4 % is the room available.

### This corrects the number of record by 19 points

`RESULTS.md` carried **76.4 %** from a 12-trial pilot. At n=230 it is **57.4 %**.
B6's 200-trial minimum exists for exactly this reason and the pilot was always
labelled as model-selection evidence, not the answer — but 76.4 % had already
been quoted as "the task's real floor" and must not be used again.

### Two things that must travel with the number

1. **Never quote the pooled figure.** `eval_public` pooled is 40.7 %, dragged
   down by `target_only` (floor 7.1 %, because with no interferer the "mixture"
   is already near-clean). The task's floor is the `both` row, always.
2. **The eval set and the training set are not the same difficulty.** Training
   is on `sir0`, symmetric by construction; `eval_public` keeps the original
   distribution where the target is the louder voice 74 % of the time. That is
   an 7.8-point difference in floor and it is a train/eval mismatch, not a
   measurement artefact. **Which set defines the benchmark is still open** and
   is now the more important question than the difficulty itself. Rendering a
   symmetric eval set costs ~2 min for 500 trials if the answer is the second.

### Consequences

- **C2 moves to closed in `decisions-pending.md`.** Accepted 2026-08-30; the
  supervisor conversation the item called for should confirm it rather than
  re-open it, and the eval-set question above is what that conversation is
  actually about.
- `eval_private` is also fully transcribed (500 trials) and stays held back. It
  was not scored here and must not be used for calibration.
- Absent trials carry no reference text, so they are not WER at all — they are
  the invented-words check, where `small.en` emits "you" on digital silence
  (8/8, 2026-08-28).
- The ceiling is ~6 %, not ~3 %: `small.en` on reverberant LibriSpeech is worse
  than the pilot suggested. Any claim of the form "we recovered X % of the
  ceiling" must use 6.1 %.

## 2026-08-30 — D3a: the cue carries identity and the network amplifies it. Conditioning is NOT the bottleneck

`experiments/results/2026-08-30-cue-diag-sir0/`, `scripts/diagnose_cue.py`.
200 fixed crops of `sir0_val`, `model_sir0_e50es.pt` (epoch 14), CPU, 14 min.

### The measurement

Roll the enrollment within the batch and measure the SAME statistic at two
points one layer apart — `||a_true - a_swap||^2 / ||a_true||^2` — so the two
stages are comparable rather than merely similar.

| stratum | n | cue moves | output moves |
|---|---|---|---|
| all | 200 | **28.6 %** | **48.2 %** |
| same-gender trials | 131 | 26.8 % | 44.4 % |
| cross-gender trials | 69 | 31.0 % | 56.1 % |

### The partition, and it is the third branch

The conditioning path has two halves and no previous run distinguished them.
**Neither half is failing.** Swapping a stranger's enrollment moves the cue by
28.6 %, so the cue plainly carries identity; the output then moves by 48.2 %,
i.e. **the network amplifies the cue rather than discarding it.**

The standing diagnosis since 2026-08-25 — "the network is throwing the cue
away" — is wrong, measured. The ~2.3 dB held-out ceiling is in the **separator
or the objective**, not in getting speaker identity into the network.

### Consequences for the D-list

- **D4a (re-inject the TF-Map at every block) — DROP.** It exists to fix
  dilution of the cue across six BSNet blocks. There is no dilution to fix: the
  cue survives and is amplified. Its ranking as "cheapest large change" was
  conditional on D3a, and D3a says no.
- **D5 (speaker encoder + auxiliary speaker-ID loss) — DEMOTE.** The argument
  was that no parameter is devoted to identity. True, and it turns out not to be
  the binding constraint. Not refuted, but no longer indicated.
- **D2 (attention temperature) — ANSWERED WITHOUT RUNNING IT.** The softmax now
  blends **138 of 628** enrollment frames, max weight 0.067 against a uniform
  0.00159. The 2026-08-25 measurement that motivated D2 had it blending ~620 of
  628. `tfmap_scale` = 16 already did what the temperature sweep was going to
  test. Close it.
- **D1 (learned dictionary) — DROP the premise.** D1 and D2 shared one
  hypothesis: more selective matching against enrollment content improves
  extraction. The matching is already selective and extraction is still capped.

### D3c, and a correction

Sensitivity is **higher on cross-gender trials (56.1 %) than same-gender
(44.4 %)**. That is the direction a pitch shortcut produces: where the two
speakers differ in gender the enrollment's identity matters more to the output.
It is a weak signal, not proof — but it is not nothing, and `sir0_train`'s 50/50
gender balance caps rather than removes the shortcut (a model using pitch alone
scores ~75 % correct with no enrollment at all).

**Correction:** a 24-crop smoke run of the same script read the opposite
ordering (same-gender 44.9 % vs cross-gender 37.1 %) and was briefly described
as showing no pitch reliance. At n=200 the ordering reverses. The 24-crop figure
was noise and must not be quoted.

### Caveats

- One checkpoint, one split. The cue half needs no trained model (TFMap is
  parameter-free) so it generalises across checkpoints; the output half does not.
- `same_gender` describes whether a trial's TARGET and INTERFERER share a
  gender, not the gender of the rolled-in enrollment. It is the right
  stratification for detecting a pitch shortcut and is not a statement about the
  swap itself.
- This says where the ceiling is NOT. It does not say where it is.
