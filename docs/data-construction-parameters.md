# Mixture generator parameters

**Written 2026-08-10.** Every knob in the generator: what to set it to, what it
means. Values marked *pilot* are provisional until §11 confirms them.

Reference values from local clones: `LibriMix/scripts/create_librimix_metadata.py`,
`data/raw/whamr_scripts/`. Cite Cosentino et al. 2020 and Maciejewski et al. 2020
for anything borrowed.

---

## 1. Levels

### `sir_db` — *pilot*
**Sample U(−5, +15) dB per trial. Record it.**
Target speaker's level relative to the **interfering speaker**. Higher = target
easier to hear. This is the main difficulty axis.
*Neither reference gives a usable range: LibriMix has no explicit SIR (its ±8 dB
is an accident of two independent loudness draws, `:356-364`); WHAMR! inherits
±2.5 dB from wsj0-2mix.*

### `snr_db` — *pilot*
**Sample U(0, +20) dB per trial. Record it.**
Target's level relative to the **noise bed**. Independent of `sir_db` — noise and
interfering speech damage a downstream ASR differently, so they must be separate
axes.
*WHAMR! uses U(−3, +6) dB (`constants.py`), too harsh for a content metric.*

### `level_measure`
**BS.1770 integrated loudness (`pyloudnorm`).**
How levels are measured when setting SIR/SNR. Not peak, not RMS — loudness
matches perceived level and is what both references use.

### `target_loudness_lufs`
**U(−33, −25) LUFS.**
Absolute level of the mixture. Prevents every trial sitting at identical volume.
*LibriMix constants `MIN_LOUDNESS`/`MAX_LOUDNESS` (`:14-21`).* Also known as background noise.

### `clip_ceiling`
**0.95, common-gain rescale across all stems.**
If any signal exceeds this after summing, scale mixture *and every stem* by the
same scalar. Same scalar = SIR and SNR preserved; per-stem rescaling would
silently change them.
*WHAMR! `MAX_SAMPLE_AMP` (`run_scale_factors.py:76-91`); stricter than LibriMix's 0.9.*

---

## 2. Timing and overlap

### `mixture_length_s` — *pilot*
**U(15, 20) s.**
Length of each trial. Matches REAL-TSE's ~17–18 s so conditions are comparable
in kind.

### `overlap_ratio` — *pilot*
**U(0.2, 0.8), mean ~0.5. Record it.**
Fraction of the mixture where more than one person is talking. An experimental
variable, not a setting.
*REAL-TSE averages ~0.48–0.53 (`literature/review_synthesis.md:50`).*

### `target_activity_ratio`
**~0.75.**
Fraction of the mixture where the target is talking at all. Matches REAL-TSE's
~0.73–0.75.

### `target_absent_fraction`
**~0.35 of training examples.**
Examples where the target never speaks. Trains the model to output silence
rather than hallucinate the interferer. Needs a split loss (masked SI-SDR when
present, push-to-silence when absent).
*CARTSE uses ~38% (Li & Seki, 2026).*

### `length_mode`
**Pad, do not truncate. Extend the window past the last source by ≥ T60.**
How sources of unequal length are fitted together. Truncating to the shortest
source (LibriMix `min` mode) cuts the reverb tail, so reference and mixture stop
corresponding.

---

## 3. Room and reverberation

### `room_dims_m`
**U(5, 10) × U(5, 10) × U(3, 4) m, shoebox.**
Room size. *WHAMR! `sample_reverb.py:6-8`.*

### `t60_s` — *pilot*
**U(0.15, 0.6) s. Record it.**
Reverberation time — how long echoes take to decay 60 dB. Higher = more
reverberant.
*WHAMR!'s low+medium tiers. Their high tier reaches 1.0 s, which is a cathedral,
not a meeting room.*

### `source_distance_m`
**U(0.66, 2.0) m from the mic.**
How far each speaker is from the microphone. *WHAMR! `sample_reverb.py:21-35`.*

### `source_height_m` / `mic_height_m`
**U(0.9, 1.8) m, drawn independently.**
Vertical placement. *WHAMR!, same lines.*

### `shared_room`
**True. Non-negotiable.**
Both speakers must be placed in **one** room with **one** RIR generation. Separate
rooms give each speaker a distinct reverb signature, so the model separates on
room acoustics rather than voice and the eval number is meaningless.
*WHAMR! `wham_room.py:25-28` — both sources added to one room, one
`generate_rirs()` call.*

### `min_angular_separation`
**None.**
Not applied. We are mono; a single-channel model cannot use spatial cues, so the
constraint buys nothing. *WHAMR! also has none.*

---

## 4. Reference signal (the clean target)

### `target_reference` — **decide before generating**
**Proposed: direct path + early reflections (~first 50 ms).**
Defines what the model is trained to output, i.e. what task it is being given:

| Option | Model must learn to |
|---|---|
| Dry source | separate + fully dereverberate |
| Direct path only (`max_order=0`) | separate + dereverberate, keep delay and 1/r gain |
| **Direct + early (~50 ms)** | separate, remove late reverb only |
| Full reverberant | separate only, keep the room |

*Rationale: full dereverberation is a second hard task inside a 200–300 ms causal
budget, and late-reverb removal is where artefacts appear — artefacts are exactly
what the thesis says live models mishear. Early reflections fuse perceptually
with the direct sound, so keeping them costs no intelligibility.*

*WHAMR! chose direct-path-only (`wham_room.py:47-60`). Generate both and compare
in the pilot; whichever wins needs a `decisions.md` entry citing Maciejewski,
Wichern & Le Roux (2020).*

---

## 5. Enrollment

### `enrollment_length_s`
**≥5 s.** *Required by `docs/metric-definitions.md:43`.*

### `enrollment_source`
**A different recording from the mixture utterance. Assert it.**
Same chapter means the model can match on content instead of voice, which builds
a much easier task than you think you have.

### `enrollment_eq_augmentation`
**On for ~50% of training examples.**
Random RMS-preserving EQ curves on the enrollment only, simulating a different
capture device. Teaches the conditioning path device invariance.
*CARTSE "channel-gap" augmentation (Li & Seki, 2026).*

### `enrollment_device_mismatch`
**Recorded per trial as on/off.** An experimental variable for the results table.

---

## 6. Speaker pairing

### `n_interferers`
**1 for the main condition. 2 as a stress condition if time allows.**

### `same_gender_fraction`
**~0.5. Record per trial.**
Same-gender pairs are the hard case and must be reportable separately.

### `pairing_guards`
**Assert all three:** target ≠ interferer; target and interferer from different
books; enrollment recording ≠ mixture recording.
Shared vocabulary between target and interferer inflates apparent contamination
in the metric.

---

## 7. Noise

### `noise_source`
**WHAM! noise, 16 kHz mono FLAC, `data/wham_noise_16k/`.**

### `noise_split_mapping`
**WHAM! tr → train, cv → val, tt → eval. Never cross.**
A noise clip heard in training must not appear in eval.

### `noise_speech_rejection`
**Reject any noise segment whose own speech content exceeds −6 dB. Resample.**
Critical for this project specifically: intelligible speech in the noise bed
enters as an unlabelled third talker, and the metric would score those words as
hallucination when the extractor faithfully passed through what was there.
*WHAMR! `SNR_THRESH` (`noisesampler.py:45-62`).*

---

## 8. Transcripts

### `transcript_alignment`
**Cut `t` and `d` to match any audio truncation. Assert every retained word has
audio.**
A transcript containing words not present in the audio makes the metric score
the extractor for the generator's bug.

### `text_normalisation`
**Decide once, apply to both sides.**
LibriSpeech transcripts are uppercase with no punctuation; the judge outputs
both. The comparison is unfair unless normalised consistently.

---

## 9. Format

### `sample_rate` / `channels` / `bit_depth`
**16 kHz / mono / 16-bit PCM.** *Fixed project-wide — `decisions.md` 2026-08-10.*

### `stems_per_trial`
**Emit five: mixture `x`, clean target, enrollment `e`, target text `t`,
interferer text `d`.** *Required by `docs/metric-definitions.md:39-49`.*

---

## 10. Determinism

### `trial_seed`
**Derive per trial from `hash(trial_id)`. Not one global RNG stream.**
A global stream makes reproducibility depend on draw *order*, so regenerating a
subset produces different audio. Both references have this flaw (LibriMix
`random.seed(72)`; WHAMR! `np.random.seed(17)`).

### `manifest_fields`
**Record per trial:** all sampled values above, source file IDs, offsets, gains,
RIR seed. The manifest must be sufficient to regenerate the trial bit-exactly.

---

## 11. How to fix the *pilot* values

Do not pick these from papers. Generate and measure.

1. **Listen to 40 trials** spanning the ranges, transcript in hand. Can you
   follow the target? Does it sound like a room? Any speech in the noise bed?
   Catches more bugs than any automated check.
2. **Calibrate on floor and ceiling.** Run an off-the-shelf ASR on the clean
   target (WER should be ~0 — if not, the reference definition is wrong) and on
   the unprocessed mixture. **The gap is your metric's dynamic range.** Aim for a
   floor around 60–80% WER: at 100% nothing can be ranked, at 30% the task is too
   easy.
3. **Check conditions separate.** Bin by SIR and by overlap ratio; floor WER
   should vary monotonically. If not, the range is too narrow to be an
   experimental variable.
4. **Freeze.** Write to `experiments/configs/generator.yaml`, log the pilot in
   `experiments/results/` with config, commit, seed and date. Then stop tuning.

Needs no model, no GPU, no training. Do it before writing the generator's final form.

## 12. Ask the supervisor

1. Reference signal: direct-path-only or direct + early reflections? (§4)
2. Is 60–80% floor WER the right calibration target, or should it be harder?
3. Target-absent trials in *eval* as well as training? They test false alarms,
   but there is no reference text to score when the target says nothing.
