# Decision Log — M1 (BSRNN implementation)

**Closed 2026-08-31.** Covers the architecture and training infrastructure: the causal BSRNN + TF-Map extractor, its STFT, band plan, normalisation, sizing and conditioning. See `milestones.md` M1.

Split by milestone on 2026-08-31: training objective and runs moved to `decisions-m2.md`, the evaluation instrument to `decisions-m3.md`, the metric and judge to `decisions-m4.md`. Repo references were updated in the same commit. Earlier milestones are in `decisions-m0.md`; open questions in `decisions-pending.md`.

---

## 2026-08-18 — Training chunk length: 4 s

**Decision: `chunk_s = 4.0` (64,000 samples at 16 kHz).**

Literature, verified in the source PDFs/configs:

| Source | Task | Chunk |
| --- | --- | --- |
| Luo & Yu, TASLP 2023 (music BSRNN) | Music separation | 3 s (T), drawn from 6 s salient segments |
| Yu et al., Interspeech 2023 (high-fidelity BSRNN) | Fullband SE + PSE | 6 s |
| Zhang et al., ICASSP 2025 (TF-Map) | TSE | 3 s |
| wesep `tse_bsrnn_spk.yaml` (REAL-TSE baseline) | TSE | 3 s (`chunk_len: 48000`) |
| CARTSE Track 1 (challenge winner) | **Online** TSE | **4 s** mixture, 3 s enrolment |

Why 4 s rather than the 3 s consensus:

1. **CARTSE is the only online, causal TSE system in the set.** It is our task and won the track. The 6 s outlier is a speech *enhancement* paper with no enrolment and no target-absent case, so it does not apply.
2. **Fewer empty crops.** `target_activity_ratio` goes down to 0.15, so on an ~18 s clip a 3 s window can land entirely in target silence and turn a `both` trial into an absent example. A longer window reduces this but does not remove it (measured below).
3. **The model is causal with an LSTM.** At inference the recurrent state persists across the whole stream, so short training chunks never build up the context it sees in deployment. Longer chunks narrow this gap; the offline models above do not have it.

Cost: ~33 % more frames per example than 3 s, so proportionally less batch at equal memory.

**Required measurement: done 2026-08-18** (empty-crop entry below). Drift +0.043, so `chunk_s = 4.0` stands.

Enrolment stays at 5 s (`decisions-m0.md` 2026-08-12). CARTSE used 3 s; ours is inside the 5–10 s range of Yu et al.

---

## 2026-08-18 — Batch size: 12

**Decision: `batch_size = 12`, provisional on GPU memory.**

Matches CARTSE at the same 4 s chunk ("trained on 4 s chunks with batch size 12 for 200 epochs"), so the chunk/batch pair is taken from them as a unit.

Provisional because M2 runs on Kaggle, not the machine this was chosen on. If it changes, record the new value and the memory limit that forced it. The effective batch affects training dynamics, so a silent change makes M2's curves incomparable.

---

## 2026-08-18 — Empty-crop rate measured: uniform cropping retained

**Decision: crop offsets stay uniform. VAD-aware cropping is not required. The split loss must weight against the effective absent rate of 0.297, not the designed 0.254.**

An "empty crop" is a 4 s window from a trial where the target speaks, but not inside the window (`target_activity_ratio` goes down to 0.15). `crop_absent` is computed from the cropped target stem, so these are labelled correctly. The risk is that the training mix drifts from what the generator was configured to produce.

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

1. **The drift is small.** 0.254 to 0.297 is +4.3 points, too little to shift the model's silence prior.
2. **Leakage is confined to the `hard` regime.** It is negligible above activity 0.5 (0.4-3.8 %), where 75 % of the data sits, and only large below 0.4, which `base` never reaches (`base` draws [0.45, 0.78], `hard` draws [0.15, 0.78]). The existing escape hatch, filtering to `regime == base`, therefore also fixes leakage.
3. **These crops are wanted.** A `both` trial cropped after the target stops talking is the interruption case: the target is in the exchange but silent right now.

Two sanity checks passed. The per-epoch effective rate was 0.291 / 0.298 / 0.302, so `set_epoch` reaches the offset draw. Leakage was `both` 0.059 vs `target_only` 0.055, indistinguishable at these sample sizes, as expected, since leakage depends only on the *target* stem.

**Consequence for the split loss:** weighting between the target-present (SI-SDR) and target-absent (push-to-silence) halves must use the crop-level rate of 0.297. The designed 0.254 would under-weight the silence half by ~17 % relative.

Scope: `chunk_s = 4.0` on `train` only. A different chunk length invalidates these numbers.

---

## 2026-08-18 — STFT: 512/128, causal framing, and a lookahead knob

**Decision: `n_fft = 512`, `hop = 128`, Hanning window, `center=False` with symmetric `n_fft - hop` padding at both ends. Algorithmic latency 40 ms. `lookahead_frames` is a config knob, 0-16, default 0.**

### Window and hop

512/128 at 16 kHz is a 32 ms window / 8 ms shift, from Yu et al. (Interspeech 2023) §4.2 for its 16 kHz model. The paper's other figures (the 33-subband split, the 20 ms/10 ms window) are for its 48 kHz fullband model and do not transfer (see `decisions-m0.md` on fullband settings).

Chosen against **our** ~200-300 ms streaming budget, not the REAL-TSE Track 1 100 ms cap. 40 ms uses under a quarter of it, leaving headroom for the lookahead knob below. This is the justification M1 requires.

### Framing: `center=False`

`center=True`, the PyTorch default, is wrong for us. It centres frame *t* on its timestamp, so the frame consumes `n_fft/2` = 256 samples = **16 ms of audio from after *t***. This is invisible offline and only surfaces when streamed. `center=False` frames cover `[t, t + n_fft)` and are emitted once the window has filled, so nothing depends on the future.

The `n_fft - hop` = 384-sample left pad reproduces the cold start of a live system, whose input ring buffer starts zeroed and fills over the first 384 samples.

### Synthesis is a manual overlap-add, not `torch.istft`

`torch.istft` **refuses `center=False`**: it checks the overlap-add envelope for edge coverage and raises `window overlap add min: 1`, even with left-padding. Synthesis is therefore `F.fold` (vectorised overlap-add), divided by the summed squared window so each output sample is corrected for how many frames covered it.

**The tail pad is required (found through a bug).** Padding only enough to complete the final frame gives `right = 0` for a 64,000-sample input. The last real samples then sit where the envelope has decayed to ~1e-9, and dividing by it destroys them: max reconstruction error **6.6e-3, concentrated entirely in the final samples** (head and interior 7e-7). Also padding the tail by `n_fft - hop` gives **9.5e-7 across lengths 12,345 / 16,000 / 63,999 / 64,000 / 64,001**, with a flat envelope of 1.5 over the whole real signal.

The regression test asserts on the **envelope** (`env[pad:pad+T].min() > 1.0`) as well as reconstruction error. The error says something is wrong; the envelope says what, for any window/hop in the latency ablation.

### Latency convention

40 ms = `n_fft + hop` = window fill plus one hop to emit. **State this convention whenever the number is quoted:** CARTSE reports `window - hop` = 24 ms plus 8 ms buffering for the same framing. Neither is wrong; they count different things. The report quotes the *measured* effective future dependency, not the formula.

### Lookahead

`lookahead_frames` (0-16, one frame = one hop = 8 ms) lets the mask head see *k* future frames. Algorithmic latency runs from 40 ms at k=0 to 168 ms at k=16, all inside budget before compute and the live-model API round trip.

**Implementation note:** lookahead cannot be done by shifting the training target. The mask is multiplied into mixture frame *t*, which carries the target's content at time *t*, and a per-frame mask cannot move energy in time. Instead, the *feature* sequence is shifted between the sequence stack and the mask head (`lookahead_shift`), so the mask for frame *t* comes from a hidden state that has seen frames up to *t + k*.

Train k=0 first, as the causal baseline for the ablation.

Planned ablation: k in {0, 4, 8, 16} against LCF. No published work plots this accuracy/latency trade-off against a live-model content-fidelity metric, since no such metric exists yet.

---

## 2026-08-18 — Band plan: inherited for the baseline, six candidates for ablation

**Decision: `wesep_16k` is the baseline plan. It is inherited, not justified. This is recorded because the band plan is an open question in this literature and one of the few cheap contributions available to us.**

### The plan is unjustified everywhere, not just here

The 16 kHz table (`[3]*15 + [6]*10 + [16]*5 + [64] + [8]`, 32 bands over 257 bins) is **not** from Yu et al. (Interspeech 2023). That paper specifies 33 subbands (20x200 Hz + 6x500 Hz + 7x2 kHz) for its **48 kHz** model (~21 kHz), which does not transfer to our 8 kHz ceiling. Every system here hand-adapted a table from a *music* separation paper (Luo & Yu, TASLP 2023) without saying why. The 16 kHz table is the wesep / REAL-TSE baseline's own unpublished adaptation, and `review_synthesis.md` notes that SA-Mamba and CARTSE use 36 and 32 bands with no stated reason.

The boundaries matter, because per-band normalisation is a large part of why the architecture works (without splitting, the high bands are numerically invisible).

### Candidates, all verified expressible at n_fft=512 / 16 kHz

| preset | bands | min bins | max bins | motivation |
| --- | --- | --- | --- | --- |
| `wesep_16k` | 32 | 3 | 64 | inherited baseline; the thing being tested |
| `uniform_32` | 32 | 8 | 9 | control: if this matches, non-uniform splitting does no work |
| `yu_truncated` | 27 | 6 | 41 | Yu et al. §4.2 progression truncated to 0-8 kHz; closest to a published plan |
| `bark` | 22 | 3 | 41 | Bark critical bands (Zwicker, 1961): boundaries where the ear integrates energy and masking occurs |
| `mel_32` | 32 | 2 | 22 | mel spacing: the filterbank our proxy losses and every ASR front-end consume |
| `f0_focused` | 26 | 3 | 35 | maximum resolution across F0 and the first two formants, coarse above |

All six sum to 257 with no empty bands.

`mel_32` is the most project-specific motivation and has no precedent in the TSE literature. It matches band structure to the representation a live model listens through, and only we have a metric for what that model *understands*.

### Two constraints on the experiment, both found by measuring

**1. Band count confounds capacity.** Band count ranges 22-32. `SubbandNorm` has one normalisation and projection per band and `BandMasker` one MLP per band, so band count drives parameter count. `bark` (22) vs `wesep_16k` (32) varies boundary placement *and* model size together. **Any band-plan ablation must match band counts across arms or report parameter counts with every result.**

**2. The band-plan and window-size questions are entangled.** At n_fft=512 one bin is 31.25 Hz, and the narrowest band of `mel_32` is 2 bins (expressible, but thin). A 32-band ERB plan is *not* expressible: its lowest edge lands near 27 Hz, under one bin, and the `min(widths) >= 1` guard rejects it. At n_fft=1024 a bin is 15.6 Hz and both are comfortable. Perceptual plans therefore argue for the larger window, independently of the latency headroom (40 ms used of a 200-300 ms budget). Treat these as one experiment grid.

### Implementation

`band_plan(sample_rate, n_fft, spec)` is a pure function (no state, no I/O): preset name or explicit spec in, bin counts out. Four spec kinds: `segments` (bandwidth + count), `edges_hz` (natural for perceptual scales), `uniform`, `mel`. It asserts that widths sum to `n_fft // 2 + 1` and that no band is empty. Segment overrun and sub-bin bandwidths raise named config errors, not shape mismatches downstream.

Kept pure so the plan can be swapped from YAML and regenerated at any `n_fft`. The wesep reference computes it inline in the `BSRNN` constructor, which makes this ablation impossible there.

---

## 2026-08-18 — Normalisation: channel-wise LayerNorm, not BatchNorm and not GroupNorm

**Decision: per-band normalisation is channel-wise LayerNorm (`nn.LayerNorm` over the channel axis at each time step). `GroupNorm(1, d)` is kept for a non-causal/offline config only. We do not use the paper's BatchNorm.**

### This is a deviation from the paper, deliberately

Yu et al. (Interspeech 2023) §4.2: "We use layer normalization for offline configuration and batch normalization for online configuration."

The reason is not causality: BatchNorm is causal at inference, since its frozen running statistics depend on neither future frames nor other examples. The problem is that BatchNorm couples examples within a batch in training and switches to population statistics at inference, so training and deployment normalise differently. In a streaming model this adds to the chunk-boundary and recurrent-state issues. Channel-wise LayerNorm has no train/inference discrepancy.

### Why channel-wise rather than cumulative

| option | pools over | causal | stateful |
| --- | --- | --- | --- |
| `GroupNorm(1, d)` | channels **and time** | **no** | - |
| **channel-wise LN** | channels, current frame | yes | **no** |
| cumulative LN | channels, start to now | yes | yes |
| BatchNorm | batch (frozen at inference) | yes | running stats |

Channel-wise is chosen because it is **stateless**. Cumulative layer norm depends on how much audio has been seen, so a 4 s training chunk normalises differently from a 60 s deployed stream, and it adds state to the streaming cache. Channel-wise uses only frame *t*'s own channels.

**Naming trap.** wesep calls this `"cLN"`, but its implementation is `ChannelWiseLayerNorm`: transpose, `nn.LayerNorm` over channels, transpose back. Conv-TasNet's cLN (Luo & Mesgarani, 2019) is *cumulative* layer norm, a different operation. **Cite ours as channel-wise LayerNorm, never as Conv-TasNet's cLN.**

### Measured: GroupNorm leaks the future

Perturbing the input at frame 50 of 100 and finding the earliest output frame that changes:

| normalisation | first changed frame | verdict |
| --- | --- | --- |
| channel-wise LN | 50 | causal |
| `GroupNorm(1, d)` | **0** | leaks the future |

GroupNorm changes frame 0, i.e. the *entire* output, because it pools over channels and time jointly. Such a model trains and scores normally offline and only fails when streamed. Keep the test as a regression check and re-run it after any change to the normalisation or the modelling stack.

### Why per-band at all

Speech energy falls roughly 6 dB per octave, so the lowest bins carry orders of magnitude more energy than the highest. Normalising all 257 bins jointly would leave the high bands contributing almost nothing to the gradient. Each band instead gets its own normalisation and 1x1 projection to N=128, so a quiet 6 kHz band reaches the sequence model with the same representational budget as a loud 200 Hz one. This is a large part of why band-split works, and is the mechanism the band-plan ablation (above) tests.

### Implementation note

`SubbandNorm.forward` uses `reshape`, not `view`, to flatten `(B, C, BW, T)` to `(B, C*BW, T)`. `BandSplit` slices with `narrow()`, which returns a non-contiguous view, and `view()` raises on it (verified). The wesep reference calls `.contiguous()` in its band split instead.

Output shape `(B, K, N, T)` = `(12, 32, 128, 503)` at our config; 70,916 parameters. This module turns 32 unequal-width bands into one stackable tensor, so a single RNN can run across bands.

---

## 2026-08-19 — Model sizing: 7.16 M parameters, deliberately below challenge scale

> **Headline figure superseded 2026-08-24 — quote 7,189,644 (7.19 M).** Every number below is the *pre-conditioning* count, as this entry's own "Conditioning is nearly free" section states. The decision itself stands unchanged; only the total moved, and by the amount predicted here. See 2026-08-24 below.

**Decision: `feature_dim = 128`, `hidden_dim = 192`, `num_repeat = 6`, `mlp_hidden = 384`, `n_hidden = 1`. Total 7,156,234 parameters against the REAL-TSE causal baselines' 25-27 M, a 3.5x reduction.** This meets the M1 requirement that the model be deliberately sized down and reported as such.

### Where the parameters are

| component | params | note |
| --- | --- | --- |
| `SubbandNorm` | 70,916 | per-band norm + 1x1 projection to 128 |
| `BandSequenceModel` (6 x BSNet) | 4,898,304 | 816,384 per block: 272,256 time + 544,128 band |
| `Estimator` (384 x 1) | 2,187,014 | 32 per-band trunks + mask and residual heads |
| **total** | **7,156,234** | |

`STFT` and `BandSplit` hold no parameters (the Hann window is a non-persistent buffer).

### Two deviations from the reference, both toward smaller

**`hidden_dim = 192`, not 256.** 192 is the paper's LSTM width (Yu et al., Interspeech 2023 §4.2). The wesep reference uses `feature_dim * 2` = 256, which would put the separator at ~7.7 M rather than 4.90 M.

**`n_hidden = 1`, not 2.** The paper gives the estimation module's *width* (384) but not its *depth*; wesep uses two hidden layers. We keep the paper's width with one hidden layer, so the deviation is on the axis the paper left unspecified. The saving: the second 384-to-384 convolution costs 147,840 per band, i.e. 4.7 M across 32 bands, for a per-band readout rather than sequence modelling.

### The scaling ladder, if the baseline underfits

Measured, so each step's cost is known in advance:

| `mlp_hidden` | `n_hidden` | Estimator | total model |
| --- | --- | --- | --- |
| 256 | 1 | 1.46 M | 6.43 M |
| **384** | **1** | **2.19 M** | **7.16 M** (chosen) |
| 256 | 2 | 3.57 M | 8.54 M |
| 384 | 2 | 6.92 M | 11.89 M (paper width and wesep depth) |
| 512 | 2 | 11.32 M | 16.29 M (wesep) |

Start small: a baseline that trains is worth more than a faithful one, raising capacity is a one-line config change, and a model too large to iterate on is hard to debug. If it underfits, step up the ladder and record which rung and why.

The Estimator dominates at every rung (58 % of the model at the paper's 384 x 2, larger than the six-layer separator). Capacity there buys per-band readout, not temporal or cross-band modelling.

### Conditioning is nearly free

TF-Map (M1, still to build) raises the input channels of `SubbandNorm` from 2 to 3, taking it from 70,916 to ~104,000: about 33 k, under 0.5 % of the model. Unlike the embedding path it needs no speaker encoder, so the sizing will not move materially when conditioning lands.

---

## 2026-08-19 — Measured effective future dependency: 23.9 ms

**Decision: report 23.9 ms measured effective future dependency, with the `n_fft - hop` convention stated. This replaces the derived 40 ms figure in the STFT entry above as the number to quote.**

Method: perturb the input waveform from sample 32,000 of 64,000 onward, run the full model, and find the earliest output sample that differs.

| quantity | value |
| --- | --- |
| first changed output sample | 31,618 |
| lead over the perturbation | 382 samples = **23.9 ms** |
| `n_fft - hop` | 384 samples = 24.0 ms |
| `n_fft + hop` (earlier, conservative) | 640 samples = 40.0 ms |

The dependency tracks `n_fft - hop`, not `n_fft + hop`. The earlier 40 ms over-counted by adding a hop of buffering to the window fill; the true dependency is how far forward the overlap-add reaches.

**Agreement with the published system.** CARTSE reports 24 ms algorithmic latency and a *measured* effective future dependency of 22.2-23.7 ms (mean 22.9 ms). Our independently written implementation measures 23.9 ms. This corroborates the framing and the measurement method; use this convention beside their number.

For a deployed total, add one hop of buffering (8 ms) as CARTSE does: ~32 ms. That leaves roughly 170-270 ms of the 200-300 ms budget for compute, the lookahead knob and the live-model API round trip.

**Test to keep:** this is the causal-correctness check for the whole model. Re-run it after any change to the STFT framing, the normalisation or the sequence stack. A result materially below 382 samples means something is reading the future.

---

## 2026-08-19 — TF-Map conditioning: Spectral Similarity, not Embedding Similarity

**Decision: TF-Map uses the Spectral Similarity variant (eq. 2). The Embedding Similarity variant (eq. 3) is rejected because it requires encoding the live mixture, which is not causal at any acceptable latency.**

Source: K. Zhang, J. Li, S. Wang, Y. Wei, Y. Wang, Y. Wang, H. Li, "Multi-Level Speaker Representation for Target Speaker Extraction", Proc. ICASSP 2025, doi:10.1109/ICASSP49660.2025.10889409, arXiv:2410.16059, sec II-A and Fig. 2. (The local copy is filed as `...TSE2024.pdf`; the preprint is 2024, the venue ICASSP 2025. Cite the venue.) The paper uses BSRNN as its own backbone, so combining the two follows its architecture.

TF-Map makes the model a target speaker *extractor*. Without it the model is a speech enhancer with no notion of which voice to keep.

### What it does

`F_tfmap = B_e H` with `H = Softmax(B_e^T B_x)`. `B_e`, the enrollment magnitude spectrogram, is used directly as NMF-style basis vectors (one per enrollment frame, not a learned dictionary). For each mixture frame, cosine similarities against all enrollment frames are softmaxed into weights, and the weighted sum of enrollment frames is "what the target's spectrum probably looks like at this instant, reconstructed only from the enrollment". Energy is then recovered by projecting the mixture magnitude onto the unit TF-Map frame, per the paper. The result is a third input channel beside real and imaginary, so `in_channels` goes from 2 to 3.

**In both variants the output is `B_e H` and the basis vectors are always the enrollment magnitude spectrogram.** Only the weights `H` differ: the encoder in eq. 3 only gives a better similarity measure for choosing which enrollment frames to blend.

### Why not Embedding Similarity

The spectral variant's weakness: **magnitude spectra are dominated by phonetic content rather than speaker identity.** An interferer saying "ah" and the target saying "ah" in the enrollment have similar spectra, and cosine similarity in 257-dim magnitude space will match them. A speaker-embedding space, trained to ignore what is said and capture who says it, would not.

We still reject it because **eq. 3 needs `E_x`, frame-level embeddings of the live mixture.** The enrollment side (`E_e`) is computed once offline; the mixture side is not free. ECAPA-TDNN's frame-level layers are symmetric dilated convolutions whose receptive field extends on the order of 100+ ms in each direction (order-of-magnitude; verify against the specific checkpoint before quoting). That is roughly half of our 200-300 ms budget, spent on a better similarity measure.

**Freezing the encoder does not help**: a symmetric convolution still reads the future. Making it causal means retraining the encoder, a separate project.

The cheap variant is not a compromise: the paper finds the spectral-level feature is the main driver of improved generalisation, and in the REAL-TSE baselines the *causal* TF-Map variant has the best TER (0.652 DEV / 0.801 EVAL1 / 0.808 EVAL2), beating even the non-causal speaker-embedding baselines. TER is the published metric closest to our LCF objective. See `literature/review_synthesis.md`.

**Recorded as a deviation, not an omission:** Zhang et al.'s full system uses a pretrained ECAPA-TDNN; we use only their eq. 2, because of the streaming budget. Our results must not be presented as reproducing their full multi-level system.

### Where the encoder does still belong

An *utterance-level* speaker embedding of the enrollment is computed offline, so it costs zero mixture-side latency. It supplies the identity information TF-Map lacks and remains the natural ablation arm.

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

The 33 k comes entirely from the input of `SubbandNorm` growing from 2 to 3 channels. TF-Map itself (normalise, matmul, softmax, matmul, rescale) has no weights.

Causal by construction: each mixture frame attends only over enrollment frames, which are fixed and available before the stream starts. No mixture frame sees another.

### Test to keep: the conditioning must be live

Substituting a *different speaker's* enrollment and measuring the relative change:

| quantity | relative change |
| --- | --- |
| TF-Map feature | 0.8123 |
| model output | **0.4785** |

**This is the most important test on the conditioning path.** The standard silent failure in TSE is a model that learns to be a plain enhancer and ignores the conditioning. It trains, converges and produces clean audio of the wrong speaker, and no shape assertion catches it. Re-run after any change to the conditioning path or `in_channels`; a relative output change near zero is a failure.

---

## 2026-08-24 — Sizing figure: 7.19 M realised, and the conditioning prediction confirmed

**Quote 7,189,644 parameters (7.19 M) wherever the model size appears (thesis, README, `meta.yaml` prose), not 7,156,234 (7.16 M).** The 2026-08-19 figure was measured before TF-Map was built, as that entry says. This is a realised-versus-planned update, not an error correction.

Measured from the config that trains, via `build_model()`:

```
../tse_venv/bin/python -c "import sys,yaml;from pathlib import Path;\
sys.path[:0]=['.','scripts'];from train import build_model;\
m=build_model(yaml.safe_load(Path('experiments/configs/bsrnn_baseline.yaml').read_text()));\
print(sum(p.numel() for p in m.parameters()))"
```

### The 08-19 prediction was right

That entry predicted the delta:

| | predicted 2026-08-19 | measured 2026-08-24 |
| --- | --- | --- |
| `SubbandNorm` | ~104,000 | **104,326** (+326) |
| increase over the 2-channel count | ~33 k | **33,410** |
| share of the whole model | under 0.5 % | **0.46 %** |

`BandSequenceModel` 4,898,304 and `Estimator` 2,187,014 match the 08-19 table exactly, so the whole difference is the band-split projection inside `SubbandNorm`, whose input TF-Map raises from 2 channels (`Xri`) to 3 (`Xri` + the TF-Map plane).

### What does not change

- **The 3.5x reduction claim.** 25-27 M / 7.19 M is still 3.5x.
- **The two deviations toward smaller** (`hidden_dim = 192`, `n_hidden = 1`).
- **The scaling ladder**, whose `total model` column is also pre-conditioning. Add 33,410 to every rung: the chosen rung is 7.19 M, and the paper-width / wesep-depth rung 11.92 M rather than 11.89 M. The ordering and the case for starting small hold.
- **"Conditioning is nearly free"**, now demonstrated at 0.46 % of the model, with no speaker encoder.

### Why this is logged rather than edited in place

The 08-19 count is what the sizing *decision* was taken against, and a log that silently rewrites its numbers cannot be audited. The stale figure stays, with a pointer here.
