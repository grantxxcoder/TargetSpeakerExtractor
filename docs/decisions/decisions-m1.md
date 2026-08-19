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
