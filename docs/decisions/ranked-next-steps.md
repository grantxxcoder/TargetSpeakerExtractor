# Ranked next steps — written 2026-09-21

Working document for discussion, not a decision log. Decisions taken from it
move to the milestone log they belong to. Ordered by **expected value**, not by
cost: a cheap item appears high only where it changes what the next item means.

Sources: five parallel investigations on 2026-09-21 (objective design, causal
TSE architectures, sequential Monte Carlo, in-model difficulty estimation, and
an independent code audit), plus local verification of
`../wesep_pretrained/tfmap_context_causal_100/`.

---

## The diagnosis in one paragraph

**The speaker cue is a loudness meter with a smoothed spectrum attached.**
Measured on 20 `sir0_val` trials: 78 % of the TF-Map's energy is rank-1 (one
static template x one number per frame), and that number correlates **0.990**
with the mixture frame's loudness. The softmax blends 138 of 628 enrolment
frames, which averages the harmonic comb away, so the template is a speaker
*envelope*. The template is a non-negative combination of the target's spectra
only, so the cue can say "target-like" and can **never** say "that bin belongs
to the other person". One mechanism, three symptoms: below 0 dB SIR the mixture
frame's shape *is* the interferer's, so the cue tracks the wrong speaker
(the collapse); a per-frame scalar cue can only teach a per-frame gain (the flat
internal mask); and capacity on a starved cue overfits faster (the null 14.7 M
result).

**Correction carried from the same audit: the flat-mask symptom was
misdiagnosed.** The output's effective ratio `|M(x)X+R|/|X|` is **0.148**
volume-knob share and **1.610** freq/time -- *more* frequency-varying than the
ideal mask (0.530 / 1.012). The residual branch `R` supplies essentially all of
it. The 84 % figure describes `|M|`, one of two factors, and the one that does
not carry the structure. D14, D15 and D17 were aimed at a property of the output
that was never wrong. The note at line ~3143 that "R carries 8.6 % of the energy
and removing it changes nothing" measured in the wrong domain: R is **148 %** of
the masked path's *compressed* magnitude, which is what `L_MR` and mask shape
actually see.

---

## Tier 0 — preconditions. Nothing below is interpretable without these.

**0a. Fix the lookahead crash.** `modules.py:467` calls
`F.pad(h[..., k:], (0, k), mode="replicate")` on a 4-D tensor. Reproduced
2026-09-21: `NotImplementedError: Padding size 2 is not supported for 4D input
tensor`. **Any non-zero `lookahead_frames` crashes**, so the latency ablation
documented since 2026-08-18 has been silently un-runnable. Fix: `(0, k, 0, 0)`.
One line, plus a test, plus a `run_times.md` row when the ablation runs.

**0b. Settle the capacity confound before quoting it.** `lr: 0.0005` is held at
batch 3, 6 and 10 (6 / 12 / 20 examples per step with `both_directions`). The
hold was deliberate and reasoned (`decisions-m2.md` 2026-09-21) and that entry
names raising lr as the first follow-up if the run underfits. It did. Two
further candidates, both from the audit: **63 % of the added parameters went
into the estimator** (2.187 -> 6.918 M), which is `kernel_size=1` throughout and
so cannot improve separation even in principle; and `n_hidden: 2` stacks two
`Tanh` layers with no normalisation between them. Record the capacity arm as
**confounded, not negative**.

*Disputed:* whether the baseline is under-trained. Train `L_pres` -5.50 at
epoch 14 alongside val -1.282, with train falling monotonically while val peaks
at epoch 7, is the **overfitting / data-limited** signature already diagnosed
2026-09-01 -- not under-optimisation. Do not record it as under-trained.

---

## 1. Give the model a discriminative speaker cue

**Three steps, increasing in cost. This is the highest-value line available.**

- **1a, one line. Hand the network the cue's PARTS, not their product.**
  `conditioning.py:57-59` returns `alpha_t * u_t` where
  `alpha_t = <|X_t|, u_t>`. Pass instead: `u_t` (the direction, already
  loudness-invariant because `sim` is computed on L2-normalised spectra);
  `cos_t = alpha_t / |||X_t|||`, the **normalised** similarity, i.e. how
  target-like this frame's *shape* is with loudness divided out; and the
  unexplained residual `|X_t| - alpha_t * u_t`, the negative evidence the path
  has none of. To recover `cos_t` today the network would have to divide by
  `|||X_t|||`, which it is not given in usable form because `SubbandNorm` has
  already jointly normalised the channels.

  **THE LOUDNESS CONFOUND IS ARCHITECTURAL, NOT LEARNED, AND NO DATA CHANGE
  FIXES IT.** `alpha_t` is the mixture magnitude projected onto a unit vector,
  so it scales with frame energy by construction. Verified 2026-09-21:
  `sir0_train` SIR is mean **+0.08 dB**, median +0.11, range -10..+10 --
  symmetric and centred on zero, so "the target is the louder one" is already a
  coin flip in training and the model gains nothing from it. The cue is
  nonetheless 99 % loudness. Distribution cannot be the cause.

  **Consequence: do NOT make the level distribution asymmetric to fix this.**
  Asymmetric SIR would *create* a loudness shortcut where none exists, and it
  would look like improvement. Asymmetric SNR does not bear on speaker
  selection at all and would move every anchor. Level jitter is already
  supplied by `remix_gains: true`.

  **Acceptance test for the whole of item 1 is `corr(cue, frame loudness)`, not
  LCF-WER alone.** It is free, it is what the intervention is meant to move,
  and an arm that does not move it has failed on its own logic. Free companion
  diagnostic: `eval_public` keeps the target-louder distribution while `sir0`
  is symmetric (C2, they differ by 7.8 points), so scoring one checkpoint on
  both, split by SIR band, measures residual level bias with no training.
- **1b, one line.** `conditioning.py:158` applies `ChannelWiseLayerNorm(bw)`
  *within each band*. For the fifteen 3-bin bands that leaves two degrees of
  freedom and **deletes `alpha_t`**. Normalise across the whole TF-Map or not
  at all. This is the mechanical cause of D4a's gates sitting near zero.
- **1c, the real arm.** A frozen pretrained speaker encoder as a contextual
  embedding, multiplicative fusion. **ECAPA is already downloaded and wired in**
  for the state teacher (`state_teacher.py:245-276`, `../ecapa_pretrained`).
  It helps for a specific reason worth stating: the embedding is computed from
  the **enrolment only**, so it is constant per trial and carries zero
  per-frame loudness -- it cannot be confounded by the mixture's level. But
  multiply-fusing a constant embedding onto a loudness-scaled cue MODULATES the
  scaling rather than removing it, so **1c does not subsume 1a**. Run 1a first
  or the confound survives the fix.

**The memorisation objection does not apply to a frozen encoder.** D5 was
demoted partly because a learned embedding over 251 speakers can memorise. An
encoder pretrained on VoxCeleb and frozen never sees the training set.

**Register before running:** the effect size, and that the acceptance test is
content fidelity on `both` split by SIR band, not SI-SDR. Noise floor is 1.57
points (null intervention) and ~3 points (trial sampling), with run-to-run
training variance still unmeasured (J5).

## 2. The directional enrolment-swap test

**Run before and after item 1. Zero training.** `diagnose_cue.py:83-87`
computes `||a-b||^2/||a||^2` -- a magnitude with no direction. "Output moves
48.2 % on an enrolment swap" is fully consistent with the output only changing
*level*, which is the behaviour under investigation. **D5 was demoted on that
evidence.** The test that settles it: swap in the *interferer's* enrolment and
score SI-SDR against the interferer stem. Does the output move *toward* the
other speaker, or just change volume?

Pair it with speaker-confusion-vs-SIR: score each output against both speakers'
embeddings, plot against SIR. Three different failures currently hide inside one
88.2 % WER -- wrong speaker, near-silence, passthrough -- and they need opposite
fixes.

**Do not use SI-SDRi on the low-SIR slice.** It rises as the input gets worse,
so it improves while the audio degrades.

## 3. Set `w_struct` to 0 and close the mask-shape line

**D18 Test 1 is run and confirmed.** On the baseline a perfectly flat mask
scores **better** on `L_struct` than the trained mask (0.1717 vs 0.1803). The
term is an energy-weighted L1 to an unpredictable target, and L1's uninformed
optimum is the conditional median -- zero deviation, i.e. flat. At
`w_struct: 46.3` that is a standing ~0.40 loss units of pure flattening
pressure, and the measured outcome matches (the struct arm's `d_freq` halved).

`tests/test_mask_structure_loss.py:97` cannot catch this: it compares flat
against the *oracle*, the one comparison that cannot fail. Add the missing
comparison -- flat against a plausible imperfect prediction.

Any successor must supervise `|M(x)X+R|/|X|`, not `|M|` -- the output's
structure lives in R. But D19 already says this is the wrong tree.

## 4. Decide whether `R` is allowed to do this

`R` is a raw unbounded `Conv1d`, no activation, no normalisation, no
conditioning (`modules.py:350`). **`M` and `R` are unidentifiable given the
loss**, so no statement about "the mask" is well-posed until R is constrained or
a no-R arm is trained. The inference-time ablation cannot settle it, because a
model trained with R has a mask that depends on it.

Related and cheap: `p=0.3` compression with an **unweighted mean over all bins**
(`losses.py:144-145`) puts most of `L_MR`'s gradient on near-silent bins, and R
is the branch serving them. Energy-weighting that reduction is the defensible
version.

## 5. The artefact-weighted objective (AB-SDR)

Ochiai et al., IEEE/ACM TASLP 32:3589-3602 (2024). Decompose the error into
interference / noise / artefact by the Vincent projection, boost only the
artefact term: `L = -10log10(||s_target||^2 / ||e_interf + e_noise + a*e_artif||^2)`,
`a = 2.0` multi-talker, `L = 1` delay tap. Reported 19.6 -> 14.9 % WER.
Differentiable; `fast_bss_eval` (Scheibler, ICASSP 2022) implements it, and
`src/live_model_metric/separation.py` already has a `decompose`.

**We can do this and most cannot:** it needs target, interferer and noise as
separate references. All three are on disk (noise = mixture - target -
interferer, the `remix_gains` path).

**Register the direction.** The literature says punish artefact harder, but
that is from speech-plus-noise work; below 0 dB SIR **65 % of our wrong words
are leakage**. Run `a = 2.0` and `a = 0.5` and let the SIR bands decide.

## 6. An internal difficulty / confidence head

**A confidence head separate from the mask, not a better mask.** OM-LSA
(Cohen, 2003) has computed a speech-presence probability independently of the
gain since 2003 and blends `G = G_LSA^p * G_min^(1-p)`; Deep Xi (Nicolson &
Paliwal, 2019) is the causal neural per-bin a-priori-SNR form. **A binary,
frame-varying target does not collapse to one number per frame the way the
regression mask did**, and supervision is free -- `sir_db`, `snr_db`,
`overlap_achieved`, `t60_s` are all in the manifest.

It is also the principled replacement for what exists: `alpha_t` *is* already an
internal difficulty signal, just the wrong one.

**The validation blocker is dissolved.** SNRi Target Training (Koizumi et al.,
Interspeech 2022) makes processing strength an auxiliary scalar *input* and
trains a predictor for it by backpropagating a downstream loss -- they never
measured optimal strength either. Our differentiable proxies sit where their
ASR loss sits. Validate the head with **sparsification curves**, not the judge.

**Where it pays concretely:** above +5 dB SIR we sit 25 points above the
clean-target ceiling with only 8 % leakage -- self-inflicted damage. PercepNet+
(Ge et al., Interspeech 2022) fixes this with one bit: predict frame SNR, and
above threshold **bypass post-processing entirely**.

*Cautions:* the adaptive-computation literature measures FLOPs saved, not
quality gained -- do not let the two be conflated. And PRESS (ICLR 2026) found
confidence heads trained on fixed-length crops go miscalibrated on full-length
streaming audio, which is our exact setup. Report discrimination and calibration
separately and say which was measured.

---

## Dropped, with reasons

| item | why |
|---|---|
| **Particle filter / SMC** | The line died because SMC does exact inference in a weak state-space model while nets do approximate inference in a strong one, and it cannot absorb supervision. Particles survive in audio only for 2-6 dimensional state (localisation). The multimodal-posterior advantage is already standard as a categorical salience head + HMM forward recursion -- cheaper, trainable, causal. **Arithmetic kills it independently:** a *perfect* target-selective fill is worth ~2-3 points against a 1.57-point irrelevance floor, and harmonic methods reach only voiced frames (~55-60 %), putting the ceiling at or below the noise floor. Write up as a scoped negative with that arithmetic. |
| **Deep-filtering mask head** | Premise refuted -- the output is already more frequency-structured than the ideal mask. |
| **Observation adding / mix-back** | Measured and refuted (alpha sweep, n=103, 515 judge calls). Mechanism: it was built for speech-plus-noise, where what is added back is noise Whisper tolerates. Here it is a competing talker -- intelligible words that become insertions. **Worth a paragraph as a transfer failure we measured.** |
| **New SSL / frozen-ASR feature-matching loss** | Braun & Gamper (ICASSP 2022) tested pretrained MOS, WER and ASR-embedding losses against a strong magnitude-regularised complex compressed spectral loss -- essentially what we run -- and found no improvement. Lowers G1's priority relative to 1-5. |

## Unlogged obligations this surfaced

1. **D19 ran on 2026-09-15** and its answer is in
   `experiments/results/2026-09-15-effective-mask-flatness/`, not in any
   decision log, while the index still calls it a proposal. The results
   directory is gitignored and unbacked-up.
2. **"R is inert"** (line ~3143) is wrong in the domains that matter.
3. **The LR scheduler steps on `val_loss["total"]`** (`train.py:844`) -- the
   metric `selection_score()` documents as unfit and which cost a checkpoint on
   2026-08-30. Selection was fixed; the schedule was not.
