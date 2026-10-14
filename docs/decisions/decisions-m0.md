# Decision Log — M0 (data preparation)

**Closed 2026-08-18.** Covers the spec, the metric framing and every data-construction decision. Later decisions are split by milestone: `decisions-m1.md` (architecture), `decisions-m2.md` (training), `decisions-m3.md` (conventional evaluation), `decisions-m4.md` (the metric and the judge). Open questions stay in `decisions-pending.md`.

Renamed from `decisions.md` on 2026-08-18 when the log was split; references elsewhere in the repo were updated in the same commit.

---

## 2026-08-04 — Repo workflow: branch protection + PR review
PRs are enforced on main (no direct commits), even as a solo project. Every code change then gets a deliberate diff review, since all submitted code must be individually understood and defensible.

## 2026-08-04 — Eval data: REAL-TSE dev/eval set unavailable
**Status: partially superseded by 2026-08-07 re-scope. Retained for history.**
Challenge registration closed 31 May 2026, before we could register, so we have no official dev/eval set. Decision: replicate baseline training on Libri2Mix-100 + WHAM! (the baselines' own training data), and build our own REAL-T-style eval set from public AMI training splits (then AliMeeting/AISHELL-4) with the organisers' documented method (overlapping segments as mixtures, ≥5s non-overlapping segments as enrollment). Score with the official REAL-TSE-Challenge code to keep numbers methodologically comparable. Any comparison to published baselines is caveated as "different eval audio, same eval method", not like-for-like. Emailed the organisers [date] about academic access; will update if granted.

*What survives:* constructed training data from LibriSpeech-derived mixtures, and an AMI-derived real-audio set. *What does not:* the "replicate the baselines" framing, and the official scoring code as primary evaluation. The official set is now only a nice-to-have external validation set.

---

## 2026-08-07 — Re-scope: downstream live-model content fidelity is the objective
Supervisor meeting notes 8-10 (docs/decisions/specification.md) replace "replicate the REAL-TSE online baselines, then explore an architecture." with:

> Build a streaming TSE model that maximises how accurately a live speech-to-speech model recovers what the target speaker said.

Perceptual or signal quality of the separated audio is **not** optimised. It may help, but the two can diverge: the supervisors observed that conventional TSE can improve ASR accuracy while making audio *harder* for live speech-to-speech models. Measuring and characterising that divergence is a core result.

Consequences, each logged below: challenge replication and the on-device leg are dropped, the metric becomes the primary contribution rather than a novel add-on, and the latency budget is re-derived.

## 2026-08-07 — Dropped: replication of REAL-TSE baselines and eval pipeline
Spec note 8: "I do NOT need to reimplement anything from the 2026 challenge but I can borrow ideas from it." Dropped accordingly. Without eval data a replication could not be validated anyway, so it would have been a reimplementation claimed as a replication. Its compute goes to the metric work and proxy-objective training, the actual contributions.

**Still borrowed, with citation:** the REAL-T trial-construction method (Li et al., Interspeech 2025) for the AMI set; the BSRNN-family causal architecture (Luo & Yu, TASLP 2023) and TF-Map conditioning (Zhang et al., ICASSP 2025) for the model; CARTSE's target-absent training and channel-gap enrollment augmentation (Li & Seki, 2026); and the challenge's DNSMOS-gaming episode as motivating evidence for gaming-resistant metric design.

**Never claim comparability** with published REAL-TSE numbers: different data, metric and protocol.

## 2026-08-07 — Dropped: on-device / small-model deployment leg
Spec note 10: "We can assume we are also working with something as big as a server, and not necessarily optimise for a small device like a wearable." The parameter/MAC budget study, quantisation and the TF-MLPNet-class on-device backbone are out of scope.

TF-MLPNet and related work stay in the literature review as background and named future work (the sequential-vs-parallel argument still matters under a latency budget), but are not built. This reverses the "highest-value gap" in literature/review_synthesis.md, written under the old brief.

## 2026-08-07 — Training: differentiable proxies, live model as held-out judge
A live speech-to-speech API model is a black box that cannot be backpropagated through, so it is only a held-out judge at evaluation time.

Training optimises differentiable proxies for the same goal: frozen-ASR/SSL encoder feature matching as the primary proxy (cheaper than full ASR cross-entropy; precedented by CARTSE's Zipformer feature-matching loss and PS4's proxy-supervised objectives), plus signal-level, speaker-similarity and target-activity terms.

**Non-negotiable rule: the proxy model must be a different model family from the judge.** Otherwise the benchmark measures overfitting to one evaluator, the failure the challenge's DNSMOS-OVRL episode showed. The judge must also never be used as a training-data filter.

## 2026-08-07 — Latency budget: ~200-300 ms, derived not inherited
Spec note 10 makes latency secondary and assumes server-class compute, so the REAL-TSE online track's 100 ms cap no longer applies.

Decision: a ~200-300 ms end-to-end streaming budget, justified by turn-taking tolerance in live conversational agents rather than challenge rules. The model must still be causal/streaming, and measured algorithmic latency plus RTF are reported for every system.

**Open caveat: this number is an assumption, not a result.** There is no evidence yet for what latency live speech-to-speech models tolerate. Find published evidence or measure it in the metric work; until then, present it as a stated assumption.

## 2026-08-07 — Data: constructed primary, AMI secondary
**Training must be constructed.** Differentiable proxies need a clean target signal (signal-level loss) and exact verbatim text (ASR-based proxies). Real conversational corpora provide neither: AMI's headset channel carries cross-talk bleed from other speakers, and its human transcripts are not verbatim. Training on real audio would need an offline pseudo-label teacher (the CARTSE pipeline), which is out of budget.

**Primary eval: constructed mixtures**, same construction family as training but speaker-disjoint and held out. They give exact target text, a true clean-target ceiling, and controllable overlap ratio, SNR and device mismatch.

**Secondary eval: AMI-derived trials**, the real-audio transfer check. Without it the spec's real-world motivation goes untested.

**Caveat for code comments and the thesis:** on AMI the ceiling is *approximate*, computed from the individual headset (IHM) channel, which has cross-talk bleed and a different channel response from the distant mic used as the mixture. It is never ground truth.

## 2026-08-07 — Output modality: audio is the build target, text is a reference condition
Spec note 10, final two sentences: the live model accepts **either text or audio**, so the output modality is a design choice.

Decision: **the system we build outputs audio.** Reasons, in order:

1. **Latency.** A text path must run a streaming ASR to completion and endpoint before sending anything. That adds a decoding stage and endpointing delay the ~200-300 ms budget cannot fit. The spec agrees.
2. **It keeps the research question intact.** The premise is that live speech-to-speech models are unusually sensitive to *processing artefacts in audio*. With text there is nothing to mishear, and the project becomes streaming target-speaker ASR, a different thesis.
3. **Text discards everything non-lexical.** Prosody, emphasis, hesitation, emotion and speaker identity are lost at the ASR boundary, yet a speech-to-speech model uses them to interpret the turn and shape its reply. Our lexical metric would not see that loss, so a good text-path score is a reason for suspicion.

**The text path becomes a reference condition:** the leg-3 extractor feeds an off-the-shelf streaming ASR, whose text goes to the judge. No extra training, one extra results row, and it answers "how much of the content is recoverable at all once the extractor has run?" Measured and reported, not optimised for.

**Do not call the text row an upper bound.** An ASR error is unrecoverable once in text, whereas the audio path leaves the judge the acoustic evidence. Which path wins, and when, is itself a result.

Consequence for the metric: **output modality is recorded per trial**, and both paths are scored on the same end-to-end question: what did the assistant recover? See `docs/data/metric-definitions.md` §3.5. Cross-modality comparisons are valid on that number but say nothing about the judge's listening, because in the text condition the judge is close to a pass-through.

## 2026-08-07 — Metric is the primary contribution
Spec note 1 ("the actual score values from my defined metric do not matter — the metric itself matters more") plus note 8 (metric first, then architecture). The metric definition and its harness must not be cut under any schedule pressure. Full specification in docs/data/metric-definitions.md.

## 2026-08-10 — Sample rate: 16 kHz mono, 16-bit, everywhere
Every audio path (source corpora, generated mixtures, enrollment, model I/O, judge input) is **16 kHz, mono, 16-bit PCM**, as a hard project-wide invariant.

Reasons, most costly to get wrong first:

1. **Every frozen model in the loop is 16 kHz.** The proxies (frozen ASR/SSL feature matching), speaker encoder and VAD (WavLM, Whisper, Zipformer, ECAPA-TDNN, WeSpeaker) all assume it, and the proxy losses *are* our training signal.
2. **The judge takes 16 kHz.** Gemini Live's input is raw 16-bit PCM, 16 kHz, mono, little-endian. It resamples other rates, but an uncontrolled resample between extractor and judge is a confound, so we control that boundary.
3. **Intelligibility content is below 8 kHz.** Nyquist gives 8 kHz of bandwidth. Vowels, formants and voicing sit well under 4 kHz; above 8 kHz is mainly the upper energy of sibilant fricatives (/s/, /ʃ/, /f/), which affects crispness more than word identity, and our metric is lexical.
4. **It is the field standard**, so conditions are comparable *in kind* to REAL-TSE, LibriSpeech, WHAM! and the TSE literature, though not in number, per the standing rule.

**Consequences.** The 200-300 ms budget is measured on 16 kHz frames. A 48 kHz or wideband extension is a new experiment with new proxies, not a config change. A judge preferring another rate needs an explicit resample, logged per trial and named in the results table.

**Note on the WHAM! conversion in `docs/data/data-setup.md` step 1e:** WHAM! noise is *already* 16 kHz, so `-ar 16000` there is a no-op guard. The 4× storage reduction comes entirely from stereo to mono and 32-bit float to 16-bit, not from sample rate.

**16-bit is a deliberate, mildly lossy choice.** 32-bit float sources are quantised to 16-bit on ingest. Its ~96 dB of dynamic range is far more than these signals use at our SNRs, and the judge accepts 16-bit anyway.

Sources: <https://ai.google.dev/gemini-api/docs/live-api/capabilities>, <https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs>, <http://wham.whisper.ai/>, and `docs/data/definitions.md:60`.

## 2026-08-10 — Data licensing: the constructed set inherits CC BY-NC
Our three corpora do not share a licence:

| Corpus | Licence | Commercial use |
|---|---|---|
| LibriSpeech | CC BY 4.0 | permitted |
| AMI | CC BY 4.0 | permitted |
| **WHAM! noise** | **CC BY-NC 4.0** | **prohibited** |

Every constructed mixture contains WHAM! noise, so the constructed train/val/eval sets are **derivative works of an NC-licensed corpus**. The most restrictive input governs.

**The AMI secondary eval set is exempt.** It holds no WHAM! material, so AMI-derived trials stay CC BY 4.0 and may be released permissively, as a separate release decision.

Decision: accept NC rather than substitute the noise corpus. WHAM! is the field-standard, real recorded (not synthetic) noise set, and swapping it would cost comparability for a freedom a masters project does not need.

**What this binds us to:**
- We may **not** redistribute generated mixture audio under a permissive licence, sell it, or use it in any commercial product.
- The "public trial split" required by `docs/data/metric-definitions.md:186` must be **CC BY-NC 4.0**, attributing all three upstream corpora.
- Preferred release: the **generation code, manifests and seeds**, not the audio, so users regenerate from corpora they obtain themselves. This avoids redistribution and helps reproducibility. Audio release is a fallback.
- **Model weights are not clearly restricted** by NC. A trained model is generally not considered a derivative work of its training data, but this is unsettled and jurisdiction-dependent. If weights are released, note the training data's NC status and, if it matters, get an actual opinion. I am not a lawyer and this is not legal advice.

Obligations, per-corpus attribution text and the required thesis wording: `docs/data/data-licences.md`.

## 2026-08-11 — Target-absent trials: the interferer is the level anchor

`sir_db` is blank when `target_absent = 1`. But `snr_db` and `target_loudness_lufs` are still recorded relative to the missing target, so nothing fixes the interferer or noise gain.

Decision: **in target-absent trials the interferer becomes the anchor.** Normalise it to `target_loudness_lufs` and place the noise at `snr_db` relative to it. `sir_db` stays blank.

This keeps both columns' meaning and absent trials at a volume comparable to present ones, with no manifest change. A fixed absolute level was rejected: it would make the condition identifiable by loudness alone, which the model could exploit instead of listening to the enrollment.

## 2026-08-11 — Target-absent trials: interferer activity must match present trials

Found in `smoke_train`: absent trials had `interferer_activity` of 0.75-0.85 every time, present trials 0.29-0.82. A gap of 1.4 standard deviations, with barely overlapping ranges.

Cause: the absent branch set it to `target_activity_ratio` (a scalar, so exactly 0.75, then filled to `activity_tolerance`); present trials draw it from the §2.2 feasibility band.

So absence was detectable from activity alone. A model could learn "one voice talking near-continuously ⇒ emit silence" and score 35% of the training set without consulting the enrollment.

Decision: **absent trials draw interferer activity from the same distribution a present trial would have produced.** `build_manifest.py` takes a shadow overlap draw and a shadow target activity, then applies the present branch's `uniform(overlap, 1 - t_act + overlap)` band.

Consequence: low activity is harder to satisfy from long LibriSpeech utterances, so `n_failed` may rise. Check it on every regenerated manifest.

General lesson for any condition flag: **a condition the model is supposed to infer by listening must not be inferable from the manifest's summary statistics.** Compare every column across the two groups.

## 2026-08-11 — Noise is shorter than the mixtures: wrap around

WHAM! clips in the `tr` split run 3.4-47.7 s (median 10.0 s), against mixtures of 15-20 s. Only 86 of 20000 clips (0.43%) reach 20 s, and all 50 `smoke_train` trials need more noise than their clip holds.

Decision: **the noise bed wraps.** `noise_offset_s` is a phase into a looped stream, not a slice index. The renderer reads `mixture_length_s` seconds from it, returning to the clip start when it runs out.

Rejected: only long-enough clips (discards 99.6% of the library and biases toward scenes recorded for longer), and shorter mixtures (breaks comparability with REAL-TSE's ~17-18 s).

Cost: a faint repeat at each seam. On `smoke_train`, trials cross the clip end **1.9 times on average** (1 seam in 12 trials, 2 in 33, 3 in 5). That is more than a median 10 s clip in a 17.5 s window suggests, because `noise_offset_s` is uniform over the whole clip and so usually starts partway through. It is a known artefact of the constructed set; the AMI leg has none.

If 1.9 seams is audible in the §11 listening pass, draw `noise_offset_s` over `[0, max(0, duration - mixture_length_s)]` where the clip allows. This removes one seam without touching the wrap logic.

Renderer note: a naive `noise[offset : offset + length]` silently yields short or zero-padded noise, so the wrap is mandatory.

Was `decisions-pending.md` A2.

## 2026-08-11 — Enrollment must come from a different book, not just a different chapter

The interferer must read a **different book** from the target, so shared vocabulary is not mistaken for contamination by the metric. Enrollment only had to be a different **chapter**. LibriSpeech speakers usually read consecutive chapters of one book, so a same-book chapter shares narrative, characters, proper nouns and register. On `smoke_train`, **11 of 36 present trials** had enrollment from the same book as the mixture.

Decision: **the enrollment guard now matches the interferer guard: different book.** `build_manifest.py` filters on `book[chapter]` instead of `chapter`, and the trial assertion was tightened to match. The same argument justifies both guards, and the weaker one let enrollment match the target on topic instead of voice, making the task easier than the thesis claims.

Cost: a smaller candidate pool, so `n_failed` rises, and speakers who read only one book yield no valid present trial. Check `n_failed` and per-speaker trial counts on the next build; if a material number of speakers drop out entirely, revisit as a book-preferred-with-chapter-fallback rule.

Was `decisions-pending.md` B8.

## 2026-08-11 — val carries target-absent trials; the eval splits do not

`val` and `smoke_val` had `target_absent_fraction: 0.0`. The push-to-silence half of the split loss was therefore trained on 35% of training batches and measured on none, so a collapse in it would not show in any val curve.

Decision: **`val` and `smoke_val` move to 0.35, matching train. The eval splits stay at 0.0.** Val must cover every loss term to show whether training works. Whether absent trials belong in eval is still open (`decisions-pending.md` B4, deliberately not pre-empted here).

## 2026-08-12 — Enrollment carries no room

The enrollment clip (a few seconds of the target alone, telling the model which voice to follow) could be rendered dry or convolved with the mixture's room.

Decision (supervisor, 2026-08-12): **no room on the enrollment at all. It stays a clean sample, so the model knows what the target sounds like clean.**

Why: a shared echo signature would let the model match on room acoustics instead of voice. It is the same argument that makes `shared_room` non-negotiable and that set the different-book guard (2026-08-11): every route to the target other than the voice must be closed.

Cost: a channel mismatch between enrollment and mixture. This is realistic (a stored voice profile is captured elsewhere), and `enrollment_eq_augmentation` already trains the conditioning path to tolerate it.

No code change: the renderer is unwritten and LibriSpeech enrollment is already dry, so this constrains what the renderer must *not* do.

Was `decisions-pending.md` A4.

## 2026-08-12 — Enrollment length fixed at 5 s, and stays configurable

Longer enrollment makes the target easier to identify, so length varying per trial is a confound.

Decision (supervisor, 2026-08-12): **fixed at exactly 5 s everywhere, held as a parameter that can be changed rather than a hardcoded constant.**

5 s is the minimum `docs/data/metric-definitions.md` allows, so the headline result is a worst case. Any later variation is a deliberate experiment: one value per run, recorded, not sampled per trial.

No code change: `experiments/configs/generator.yaml` already has `enrollment_length_s: 5.0` under `defaults`, and `draw()` returns a scalar unchanged or samples uniformly from a `[lo, hi]` list, so a range is a one-line config edit.

Was `decisions-pending.md` B3.

## 2026-08-12 — Levels are measured as BS.1770 integrated loudness

`sir_db`, `snr_db` and `target_loudness_lufs` all state a level, which must be defined before rendering.

Decision: **BS.1770-4 integrated loudness, via `pyloudnorm`.**

RMS was chosen first in the same 2026-08-12 meeting and reversed that day, before implementation, as the *more* expensive option:

1. **It forces a gating rule.** BS.1770 gates out quiet passages by specification. Plain RMS averages over everything, so a target talking for a fifth of the window measures far quieter than one talking for most of it at the same level. Identical speech at 16 kHz, continuous and padded to 20 s:

   | | BS.1770 | plain RMS |
   |---|---|---|
   | 3 s continuous | −23.35 LUFS | −26.01 dBFS |
   | same 3 s, 14 s of silence around it | −23.77 LUFS | −33.55 dBFS |

   0.4 dB apart against 7.5 dB. Under RMS, `sir_db` would mean something different in every trial and level would correlate with speech activity. That is the class of shortcut §7 of the manifest notebook hunts, and a new leak-scoreboard entry. Gating RMS by hand just rebuilds a cruder BS.1770. It worsens if `decisions-pending.md` B9 is accepted, since a variable `target_activity_ratio` widens the activity spread.

2. **It would make `target_loudness_lufs` a misnomer.** LUFS is the BS.1770 unit. The column, config key and range `[-33.0, -25.0]` came from LibriMix's `pyloudnorm` constants. RMS would change their physical meaning and need a rename across the config, `build_manifest.py`, six manifest CSVs and the notebook, plus recalibration.

3. **Both reference implementations use loudness.** LibriMix and WHAMR! do. Our numbers are still not comparable to theirs (different data, different protocol), but the *method* needs no separate defence.

RMS's one advantage, being defined where gated loudness returns `-inf` on a silent stem, does not apply: under the 2026-08-11 level-anchor decision a silent target stem is never measured.

Renderer constraints, both verified against `pyloudnorm` 0.2.0 at 16 kHz:

- **A stem shorter than 400 ms raises `ValueError`** (BS.1770 block size). Guard with a minimum target-speech duration in the generator, needed anyway once B9 lets activity vary.
- **A fully silent stem returns `-inf`.** Only reachable through a bug given the anchor rule, so assert rather than handle.

Environment: `pyloudnorm` 0.2.0 in `tse_venv` (Python 3.12.3, numpy 2.5.2, scipy 1.18.0). Pin it. The repo still has no requirements file, a reproducibility gap in itself.

Was `decisions-pending.md` A3.

## 2026-08-13 — Reference signal: the target's reverberant image ("what the mic heard")

Decision (GB, 2026-08-13, pending supervisor sign-off): **`target_reference` = full reverberant: the target speaker convolved with their own room impulse response, i.e. what the microphone received from them. The interferer and the noise bed are excluded, so separation and denoising remain in the task; dereverberation does not.**

Dereverberation becomes an **ablation only, if time allows** (see below).

### Why

**1. Removing echo cannot be done inside the latency budget.** `t60_s` reaches 0.6 s against a 200–300 ms causal window, and 66.8 % of trials have a tail longer than the model may see (`decisions-pending.md` B11). A model cannot cancel what it has not heard, so it would learn a room prior, not cancellation.

**2. Trying anyway costs more than leaving it.** Enhancement error splits into residue left behind and what the attempt destroyed, and the destruction ("artifacts") dominates ASR degradation (Iwamoto et al., 2022; Ochiai et al., 2024). Sato et al. (2021) show artifacts can make a separated signal recognised *worse* than the unprocessed mixture, so they gate on whether to separate at all. A project claiming live models mishear artefacts should pick the task with the smaller artefact surface.

**3. Early reflections should not be removed under any option.** Reflections within ~50 ms help intelligibility as much as the direct sound (Bradley & Sato, 2003). The only real choice was whether to remove the late tail, and (1) rules that out.

### What this concedes, stated plainly

Late reverberation does cost recognition accuracy (the REVERB challenge work is unambiguous; Kinoshita et al., 2016), and spectral smearing also degrades the separation we *are* keeping (Maciejewski et al., 2020). The claim is not that reverb is harmless, only that removal is unaffordable online and costs more than it returns. **The write-up must state this as a known limitation.**

### Why the noise bed stays out of the reference

Rejected: keeping the noise in the reference (removing only the competing voice). `snr_db` runs down to 0.0, where the bed is as loud as the target and masks words though it contains none. `noise_speech_rejection` (implemented 2026-08-16) screens intelligible speech out of it, but does not guarantee its absence. Non-intelligible noise cannot make the judge hear *wrong* words but can stop it hearing the right ones. Denoising also needs no lookahead, so unlike dereverberation it is affordable online.

### Consequences

- **A5 (tail padding) is now mandatory**: the reference contains the tail, so the window must extend past the last speech by at least the room's decay, or reference and mixture stop matching.
- **B11 largely dissolves.** The model is not asked to suppress a tail it cannot hear. What remains, whether reverb limits separation, is covered by B13's stratified reporting (T60 above/below budget).
- **Ablation, if time allows:** render the direct+early stem as a second reference, train the same architecture against it and report both. One cheap render and one extra training run measure the artefact/residue trade instead of assuming it.
- The renderer records the RIR per trial so any reference variant is reproducible without re-drawing rooms.

### References

- Bradley, J. S. & Sato, H. (2003). On the importance of early reflections for speech in rooms. *JASA* 113(6), 3233–3244.
- Iwamoto, K., Ochiai, T., Delcroix, M., et al. (2022). How bad are artifacts?: Analyzing the impact of speech enhancement errors on ASR. *Interspeech 2022*, 5418–5422.
- Kinoshita, K., Delcroix, M., Gannot, S., et al. (2016). A summary of the REVERB challenge. *EURASIP J. Adv. Signal Process.* 2016:7.
- Maciejewski, M., Wichern, G., McQuinn, E. & Le Roux, J. (2020). WHAMR!: Noisy and reverberant single-channel speech separation. *ICASSP 2020*.
- Ochiai, T., Iwamoto, K., Delcroix, M., et al. (2024). Rethinking processing distortions: Disentangling the impact of speech enhancement errors on speech recognition performance. arXiv:2404.14860.
- Sato, H., Ochiai, T., Delcroix, M., et al. (2021). Should we always separate?: Switching between enhanced and observed signals for overlapping speech recognition. *Interspeech 2021*. arXiv:2106.00949.

Note on borrowing: WHAMR! uses direct-path-only references (`wham_room.py:47-60`). We diverge because WHAMR! is an offline separation benchmark scored on signal quality, and this is a causal streaming system scored on downstream content fidelity. Not comparable to WHAMR! numbers.

Was `decisions-pending.md` A1.

## 2026-08-13 — B1 closed: overlap range is a dial setting, not a decision

`overlap_ratio` stays `[0.2, 0.7]` for now. It is not pinned: it is one of the 14 parameters ranked in `docs/data/difficulty-dial.md` and moves on request once B12 lands. Two constraints on moving it:

1. **The 0.7 ceiling is deliberate.** It matches REAL-TSE's ~0.5 average overlap, our anchor. An ordinary meeting overlaps 0.10–0.15, so the set is harder than daily conversation *by design*. Narrowing the ceiling diverges from the anchor and needs its own decision entry, so the difficulty dial lists it as the narrowing to do **last**.
2. **The 0.2 floor is a separate problem (B9, not B1).** No present trial can have zero overlap, which makes silent-target trials detectable without listening (AUC 1.000). B9 fixes it with `target_only_fraction` and a variable `target_activity_ratio`, not by moving this range.

Was `decisions-pending.md` B1.

## 2026-08-13 — B12 architecture: two regimes, a sampler layer, no relational constraints

Decision (GB, 2026-08-13). Settles the shape of all four things B12 asked for. **Not yet implemented**; see the PR order at the end.

### 1. Two named regimes, sampled per trial, recorded as a `regime` column

**One regime is drawn per trial, and all that trial's parameters come from its bands.** Weights `base: 0.6`, `hard: 0.4`.

Per trial, not per parameter: with six parameters each independently 50/50, only ~1.6 % of trials would be base-case on all six. One draw makes it exactly 60 %.

`hard` inherits `defaults` wholesale, so **`base` is a sub-range of `hard`** and the column records *provenance, not difficulty*. On the proposed bands a `hard` draw lands inside the base band 50–71 % of the time per parameter, and inside all five at once 7.1 % of the time. So:

- **`regime` must not be a reporting stratum (B13).** It would compare overlapping populations and understate the gap. Report on value bands (SIR band, overlap band, T60 above/below budget), as B13 specifies.
- **Filtering `regime == base` is sufficient.** It keeps 60.0 % of rows against 62.8 % for a full value-based filter, only 2.8 points apart.

`train` mixes both regimes. If the data proves too hard, filter to `regime == base` with **no rebuild required**.

`regime` is renderer metadata the model never sees, so it cannot be a shortcut. The §7 leak audit must still be re-run **within** each regime, since a pooled AUC can hide a shortcut inside `base`.

### 2. Regime-scoped parameters: 6. Everything else global

| Regime-scoped | Global |
|---|---|
| `sir_db`, `snr_db`, `overlap_ratio`, `t60_s`, `source_distance_m`, `target_activity_ratio` | `same_gender_fraction`, `enrollment_length_s`, `enrollment_eq_prob`, `mixture_length_s`, `target_loudness_lufs`, room dimensions, mic/source heights, `clip_ceiling` |

The global ones are deliberate worst cases or variables B13 stratifies on independently, so they do not move with difficulty. `target_activity_ratio` is regime-scoped in the schema but **identical in both regimes until B9 lands**, since B9 decides what varying it means.

Band values: `difficulty-dial.md` §2, "Narrow to" column. Provisional by design, as a rebuild takes 58 s.

### 3. Distributions: `fixed`, `uniform`, `truncnorm`. Not beta

Beta (proposed in B12) is dropped: no parameter needs skew-within-bounds, and an unused shape is one more thing to defend.

**Shapes are not introduced in the same PR as regimes.** Every parameter starts `uniform`, as today, so the regime PR's diff is only the new column plus band values and can be checked against the current manifest. `truncnorm` comes later, one parameter at a time, each with a recorded reason.

`truncnorm` must use inverse-CDF over a single `rng.uniform` draw (`scipy.stats.norm.ppf`), **not** `scipy.stats.truncnorm.rvs(random_state=rng)`. A scipy upgrade can change how many uniforms `rvs` consumes per sample, silently changing what `seed: 42` means.

### 4. Relational constraints: deferred entirely

B12 point 3 is not implemented. Nothing in the data forces it, and it can be added later without a second rebuild.

B12's own example, *the target is always closer to the mic than the interferer*, **must not be implemented as a hard rule.** Proximity would then predict target identity perfectly, so the model could learn "nearer voice = target" and ignore the enrollment. That is the failure class of B9's overlap leak and B10's speaker leak, and a third §7 scoreboard entry. If ever wanted, it must be probabilistic (`p_target_closer: 0.6`).

More generally, constraints enforced by rejection bend distributions, which §4 and §7 show is what creates shortcuts. Prefer constraints satisfiable by construction.

### Implementation order

1. **PR1:** `src/data/sampling.py` with `draw(rng, spec)` and `resolve(cfg, regime)`, plus unit tests. No behaviour change, nothing wired in. Tested alone because every later number comes out of it.
2. **PR2:** wire into `build_manifest.py`, add the `regime` column, raise the `t60_s` floor from 0.15 to 0.25 (`difficulty-dial.md` §1: T60 0.15 in a room this size implies anechoic-grade wall absorption).
3. **PR3:** B9 + B10 + B4, then re-run the §7 leak scoreboard per regime.

Doing B12 before the B9/B10/B4 rebuild avoids rebuilding twice. Measured 2026-08-13: a full 20,000-trial `train` rebuild takes **58 s** and reproduces the committed manifest byte-identically, so this is cheap while no audio exists.

Was `decisions-pending.md` B12.

## 2026-08-13 — A5: the mixture keeps running after the last word

Echo continues after speech stops, so cutting the file at the last word cuts the echo, and the reference no longer matches the mixture.

**Decision: pad the rendered output by `t60_s` beyond the nominal `mixture_length_s`.** No speech occurs in the tail, so transcripts are unaffected.

Why not place speech earlier: forbidding late onsets would tie *when* people talk to *how echoey the room is*, a new confound of the kind §7 of the manifest notebook exists to catch.

Now mandatory: A1 (2026-08-13) puts the reverberant tail in the reference, so a truncated tail is a mismatch, not a cosmetic issue.

Was `decisions-pending.md` A5.

## 2026-08-13 — A6: clipping is fixed by rescaling everything together

**Decision: if any sample exceeds 0.95 of full scale after summing, scale the mixture and every stem by the same factor.**

A common gain preserves the trial's SIR and SNR; scaling only the offending track would silently change them. Already in `data-construction-parameters.md`; recorded here so it is a decision, not an undocumented default.

Was `decisions-pending.md` A6.

## 2026-08-13 — B2: overlap is measured from detected speech, not file boundaries

A read sentence contains pauses, so overlap measured from where recordings start and stop overstates how much talking coincides.

**Decision: first detect where speech actually is, and measure overlap from that.** One pass over the corpus, cached alongside the existing utterance index. It costs nothing after the first run and makes the overlap figure defensible and comparable to REAL-TSE's, which the current inflated number is not. The detector is an implementation choice, named in the PR that adds it.

Was `decisions-pending.md` B2.

## 2026-08-13 — B4: target-absent trials in eval, scored on their own row

**Decision: the eval splits carry the same target-absent fraction as `train`. Those trials are excluded from the main content score and reported separately as how often the system invents speech that was not there.**

Excluded because with nobody speaking there is no correct text to score. Same fraction as train so the strata match between what the model saw and what it is judged on. A separate row because inventing speech and mishearing speech are different failures with different fixes, and B13 (2026-08-13) forbids one number hiding both.

Supersedes the eval splits' `target_absent_fraction: 0.0`. The exact fraction follows B9, which sets it for `train`.

Was `decisions-pending.md` B4.

## 2026-08-13 — B5: text is normalised with Whisper's English normaliser

Corpus transcripts are capitals with no punctuation; a live model replies in ordinary prose with digits. Compared unchanged, every system looks worse than it is.

**Decision: adopt Whisper's `EnglishTextNormalizer` unchanged, applied identically to the reference and to the system output.** Cite Radford et al. (2023).

A published, widely used normaliser that is not ours cannot be suspected of being tuned to flatter our numbers, and spares us defending every choice about case, punctuation, digits and contractions. Two rules:

- **Frozen before the first judge result is recorded.**
- **Never adjusted per system.**

This does not put Whisper in the pipeline: the normaliser is a text function touching neither the training loop nor the judge, so the `CLAUDE.md` rules about model families are unaffected.

Was `decisions-pending.md` B5.

## 2026-08-13 — B13: results are reported condition by condition, never combined

A single overall score averages over very different trials (the target dominant in a dry room, or a louder interferer over it in a reverberant one) and hides which is which.

**Decision: report each condition on its own. No combinations.**

Six conditions, in priority order:

| | Condition | Split on |
|---|---|---|
| **Primary** | Which voice is louder | `sir_db` band |
| | How much they talk over each other | `overlap_achieved` band |
| | Whether the target speaks at all | `target_absent` |
| **Secondary** | How echoey the room is | `t60_s` above / below the latency budget |
| | Same or different gender | `same_gender` |
| | How content-disjoint the enrollment is | `enrollment_guard` tier (B10): `book` / `chapter` / `utterance` |

**Minimum 100 trials per bucket.** Each trial counts toward every condition, so a two-way split of *N* scored trials gives *N*/2 per bucket: 200 scored trials is the floor.

No combinations because six crossed conditions give 64 buckets, roughly 8 trials each at 500 trials, and 100 per bucket would need ~6,400 trials and about 13× the judge budget. The primary/secondary order says what to protect if the budget is tight.

The original entry's interruption condition is deferred: nothing in the data marks an interruption yet, and defining one needs B9's turn-taking trials.

Was `decisions-pending.md` B13.

## 2026-08-13 — Eval splits draw independently, with no regimes

Amends the B12 entry above, prompted by B13.

**Decision: the eval splits omit the `regimes:` block entirely and draw every parameter independently from the wide ranges. `regime` records `none`. Training keeps regimes as decided.**

Under regimes a `hard` trial draws a wide SIR *and* a wide overlap together, correlating them: `P(high overlap)` is 0.20 overall but 0.50 given a louder interferer, a 2.5× enrichment. B13's per-condition tables would then report one effect twice as two findings. Regimes exist so training data can be filtered when too hard; eval is never filtered, so it gains nothing from them.

## 2026-08-13 — B6: 500 trials generated, 200 the minimum scored

**Decision: keep 500 trials per eval split. Score at least 200; grow as the judge budget allows.**

Rows are generated in a fixed order, so the first 200 are contained in the first 300. Scoring more later extends the set, and numbers stay comparable as the budget grows. The 200 floor comes from B13: 100 trials per bucket across a two-way split.

Was `decisions-pending.md` B6.

## 2026-08-13 — B7: training mixtures are not resampled each pass

Rebuilding mixtures with fresh loudness and room draws every epoch gives the model more variety, but the run can then no longer be reproduced from the manifest alone.

**Decision: off for the main run. Available as a config switch.** The variety can be recovered as a reported ablation if it matters.

Was `decisions-pending.md` B7.

## 2026-08-13 — B11: latency is reported as a decay curve, not a threshold

Two thirds of trials have reverberation longer than the streaming window lets the model see.

**Decision: never cap `t60_s` to make the task fit. Score the same model at 100, 200, 300, 400 and 500 ms of allowed latency and report how performance decays.**

Capping would make the data less like a real room, the one thing constructed data must get right, while latency is only a secondary objective in the specification and changing the metric costs just several evaluation passes.

Largely defused by A1 (2026-08-13), which stopped asking the model to remove the tail. What remains is whether reverberation limits *separation*, which the T60 condition in B13 reports directly.

Was `decisions-pending.md` B11.

## 2026-08-13 — B9: a target speaking uninterrupted becomes a quarter of the data

### The problem

35 % of training trials have the target never speaking. They teach the model to **stay quiet when the enrolled voice is not there**, since a system that invents speech is worse than useless to a live model.

But in those trials only one person talks, and in every target-present trial two voices overlap at some point. So *"did two voices ever overlap?"* answers *"is the target absent?"* on **every trial in the set** (AUC 1.000 on `train`, 1.000 on `val`), without the enrollment. The behaviour is never trained, and no training curve would show it.

It also inverts in deployment. Someone talking uninterrupted looks like "target absent", so a model that learned *no overlap means emit silence* would go quiet on the most common real condition, which currently appears in **zero** trials.

### The decision

| Trial type | Before | After |
|---|---|---|
| Both speaking | 65 % | **50 %** |
| Target absent | 35 % | **25 %** — of which 5 % is noise bed only, neither speaker |
| Target only, nobody interrupting | 0 % | **25 %** |

**`target_activity_ratio` also varies** instead of being pinned at 0.75: base `[0.45, 0.85]`, hard `[0.15, 0.85]` (regime-scoped, per B12).

### Why these numbers

**The 25/25 split is symmetric because the two failure modes are weighed equally**: inventing and dropping speech are both content-fidelity failures. It puts `P(target absent | no overlap)` at **0.50**, a coin flip, so the shortcut carries no information. The originally proposed 30/15 would have left it at 0.67.

**The two changes compound.** With variable talkativeness, some both-speaking trials become turn-taking with near-zero overlap. These sit on the *present* side of the zero-overlap pool, pushing the figure below 0.50.

**Widening `overlap_ratio` instead would not work.** Overlap is floored by `overlap >= target_activity + interferer_activity - 1`. With the target pinned at 0.75–0.85, zero overlap needs an interferer at ≤ 0.20 activity (roughly 3.5 s against a current mean of 10.0 s), fitted inside a gap in the target's timeline, where the largest median gap is 2.68 s. That gives a narrow corner of near-silent-interferer trials, a *new* giveaway, plus heavy rejection. **The binding constraint is `target_activity_ratio`.**

### What else this fixes

- **The residual `interferer_activity` leak** (§7.3, AUC 0.614), at source. Present trials get rejected when a contiguous interferer block cannot reach the requested overlap; target-absent trials never face that test. A less talkative target widens the achievable overlap range, so the tolerance check stops firing asymmetrically.
- **The realism gap.** The target speaks a mean of 1.1 utterances per trial: one unbroken ~14 s monologue filling 80 % of the window.
- **B4's open number.** Eval splits take `target_absent_fraction: 0.25`, matching train, as B4 requires.

### Cost

The larger change touches `pick_run`, `best_onset` and the overlap bounds, plus a new `build_trial` branch mirroring the target-absent one. It needs a full manifest rebuild. §2, §7 and the health checks of the manifest notebook assume the current timing model and need revising; §3–§6 do not.

The §7 leak scoreboard must be re-run **per regime as well as pooled** (B12) and show the AUCs dropped.

Was `decisions-pending.md` B9.

## 2026-08-13 — B10: enrollment falls back through three tiers, recorded per trial

### The problem

B8 (2026-08-11) requires enrollment from a different **book** than the mixture, so a present target must have read two or more books. Most LibriSpeech readers read one: of the 1,172 speakers in `train`, only **467 (39.8 %)** own two books. Only they could be a present target; the other 705 appeared only in target-absent trials.

Scoring each trial by how often its target speaker is silent elsewhere gives **AUC 0.795** on `train` and 0.756 on `val`, against 0.502 for interferer identity. The silence decision could be made by recognising the speaker instead of listening.

**This is not a reversal of B8.** B8's cost note set the remedy and its trigger: *"if a material number of speakers drop out entirely, revisit as a book-preferred-with-chapter-fallback rule."* 60.2 % is material.

### The decision

**Enrollment falls back through three tiers, and which tier fired is recorded per trial in an `enrollment_guard` column.**

| Tier | Rule | Speakers in `train` | |
|---|---|---|---|
| `book` | Different book from the mixture | 467 | 39.8 % |
| `chapter` | Different chapter, same book | 469 | 40.0 % |
| `utterance` | Same chapter, utterances not used in the mixture | 236 | 20.1 % |

Book-preferred alone would leave the 236 single-chapter speakers with no valid enrollment (no other book *and* no other chapter), recreating the absent-only leak at reduced strength.

**Within the `utterance` tier, pick the enrollment from the utterances furthest in index from those used in the mixture.** LibriSpeech numbers utterances sequentially within a chapter, so this costs nothing and minimises shared narrative.

**Assert that no utterance appears in both the enrollment and the mixture.** In the third tier this is the only thing separating them.

### Why keep all 1,172 speakers

Two alternatives were rejected:

- **Dropping the 236.** Leak-free, but discards a fifth of the speaker pool for a speaker-conditioned model.
- **Using them as interferers only.** A voice that is *never* the target teaches "suppress this speaker" without the enrollment, creating a leak where interferer identity now has none (AUC 0.502).

**Recording the tier** makes the content-leak cost measurable: if `utterance`-tier trials score conspicuously better than `book`-tier ones, that is the leak, quantified. B8 called same-book enrollment indefensible as a *default*; as a recorded, measured fallback for speakers with no alternative, it is defensible.

### Eval splits are redrawn

`eval_public` has 8 of 20 speakers in the weakest tier and `eval_private` 3 of 20, which would make `eval_public` systematically easier before any system is measured.

**Decision: `make_splits.py` redraws the eval pools so the guard-tier composition matches between `eval_public` and `eval_private`.** Speaker-disjointness from `train` is unchanged. This invalidates the current manifests, but PR3 rebuilds them anyway.

### Consequences

- **B13 gains a three-level condition.** Its "one-book vs two-book target speaker" condition becomes **guard tier: `book` / `chapter` / `utterance`**, which is what actually varies.
- New column `enrollment_guard`; `make_splits.py` and `splits.yaml` both change.
- Folds into the PR3 rebuild.

Was `decisions-pending.md` B10.

## 2026-08-14 — B12 PR2: sampler wired in; two bands deliberately not narrowed

Implements the 2026-08-13 B12 architecture. Code only; the manifests on disk keep the old schema until PR3 rebuilds them.

**Wiring.** `build_manifest.py` draws the regime from a fifth RNG stream (`rngs(trial_id, 5)`). `SeedSequence.spawn(5)` returns the same first four children as `spawn(4)`, so no existing draw moves and the regime cannot shift the values it selects. A split opts out with `regimes: null` (the eval splits do); `draw_regime` then consumes nothing and `regime` records `none`.

**Acceptance test passed.** With the config unchanged, all six splits (21,270 rows) are byte-identical to the previous manifests, ignoring the new `regime` column. `scripts/check_manifest_parity.py`, run 2026-08-14.

**Two bands from `difficulty-dial.md` §2 were not applied to `base`:**

- **`overlap_ratio` keeps `[0.2, 0.7]`.** §3 ranks this narrowing last and requires supervisor agreement and a decision entry, not a config edit, because the 0.7 ceiling is deliberately matched to REAL-TSE. In the rebuild, mean `overlap_achieved` is 0.294 in `base` and 0.293 in `hard`.
- **`target_activity_ratio` stays fixed at 0.75**, identical in both regimes until B9 lands, as the B12 entry specifies.

So `base` narrows four parameters: `sir_db` `[0, 12]`, `snr_db` `[8, 20]`, `t60_s` `[0.25, 0.5]`, `source_distance_m` `[0.66, 1.4]`.

**Conflict recorded, not resolved.** `difficulty-dial.md` §2 proposes a `base` `overlap_ratio` floor of 0.1 against a global floor of 0.2, so `base` would not be a sub-range of `hard` for that parameter. The 0.2 floor is itself the B9 bug, so this is B9's call in PR3. `resolve()` therefore does not enforce containment; a named unit test pins the exception.

**Measured after the config change** (`train`, 19,569 rows, unchanged count): the `t60_s` floor moved from 0.15 to 0.25 and the regime mix is 0.60 / 0.40 exactly. In the eval splits only the room columns moved, because `t60_s` is drawn from the `room` stream; speech, levels, enrollment and noise are bit-identical.

## 2026-08-14 — PR3: B9, B10 and B4 implemented; both label leaks closed

Implements B9, B10 and B4 in one rebuild, as the milestone requires. On `train` (19,950 trials), before against after:

| leak | before | after |
|---|---|---|
| enrolled-speaker identity predicts absence | **0.795** | **0.508** |
| `overlap_achieved` predicts absence | **0.000** (perfect) | 0.169 |
| P(target absent \| no overlap at all) | 1.000 | **0.500** |
| distinct speakers that ever appear as a present target | 467 of 1,172 | **1,172** |
| unsatisfiable trials | 431 | **50** |

**B10 is fully closed:** the three-tier guard keeps every speaker, so speaker identity is at chance. Guard tiers on present `train` trials: `chapter` 6,040, `book` 5,866, `utterance` 2,992.

**B9 hits its stated target.** P(absent | no overlap) is 0.500 over 10,102 trials, and 0.497 / 0.505 within `base` / `hard`, as B12 requires.

### Two decisions taken inside PR3

**The `overlap_ratio` floor drops from 0.2 to 0.0**, since B9 needs zero-overlap trials on the *present* side. This is **not** the ceiling narrowing from 0.7 to 0.45 that `difficulty-dial.md` §3 defers to supervisor agreement; the ceiling and its match to REAL-TSE are untouched. The band is clipped per trial to `target_activity_ratio`, as a speaker cannot overlap more of the window than they speak in.

**Absent trials record the strongest guard tier their speaker could support**, not an `n/a` sentinel, which would itself separate absent from present rows (the giveaway B10 removes).

### What PR3 did NOT fix, stated plainly

The composition balances the *overlap* shortcut but cannot balance every shortcut at once. What a listener can infer without identifying the voice:

| voices heard | share | P(target absent) |
|---|---|---|
| 2 | 49.4 % | **0.000** |
| 1 | 45.7 % | 0.447 |
| 0 | 4.9 % | **1.000** |

- **Two voices means the target is present, always.** Unavoidable while every absent trial has exactly one interferer; removing it needs a two-interferer condition, which B9 did not specify. The model still needs the enrollment to know *which* voice to keep, so this shortens the silence decision but not the extraction one.
- **No voices means absent, always.** Benign: silence is then the correct output.
- **`interferer_activity` rose from 0.614 to 0.648**, against B9's expectation. `target_only` trials have no interferer, so their `interferer_activity` is 0 on the *present* side, widening the gap. The cause is the four-way composition, not the shadow-draw logic B9 blamed.

The algebra is forced: with `t` = target_only and `n` = noise_only, requiring both P(absent | no overlap) = 0.5 and P(absent | one voice) = 0.5 gives `n = 0`. B9 chose to balance the overlap shortcut, at the price of 0.05 noise-only trials.

### Other changes

- `make_splits.py` stratifies the eval halves by guard tier as well as sex. The weakest tier was 8/20 vs 3/20 and is now 6 vs 5, every tier within 1. The old single running counter could not balance two axes, so halves are now chosen by a deterministic search over which side each stratum starts on.
- `index_utterances` verifies its cache against `splits.yaml`. Redrawn eval pools left caches describing the previous speakers, and every downstream disjointness check reads that file, so a stale cache would have passed silently.
- **New columns:** `condition` (B9's four types), `enrollment_guard` (B10), `interrupted` (B13's deferred condition: an interferer onset strictly inside a target utterance; 53.2 % of both-speaking trials).
- `target_absent_fraction` is now **derived** from `composition` in `split_config`, so stated and drawn rates cannot drift apart.

## 2026-08-14 — Scope: the task is two-person conversation. At most one interferer

**Decision (GB, 2026-08-14): the constructed set models a conversation between two people. A trial contains the target, at most one other speaker, and noise. Two simultaneous non-target speakers never occur, and that is a declared boundary of the task rather than a gap in the data.**

Taken in response to the PR3 measurement below, with the cost seen. A fifth `two_interferers` condition was considered and rejected.

### Why

- **It is a scope boundary, so declare it rather than approximate it.** A token 10 % of three-speaker trials would leave the system neither trained for multi-party nor properly scoped, and the headline number would quietly claim more than the data supports.
- **It matches the anchor.** Libri2Mix and the REAL-TSE two-speaker framing are two-talker constructions, and `overlap_ratio` and `target_activity_ratio` are tuned to that. A third talker diverges on a second axis, without the supervisor agreement `difficulty-dial.md` §3 requires for the first.
- **The interference model stays interpretable.** `sir_db` is a ratio between two voices. With two interferers it becomes a ratio to a sum, silently changing every SIR figure in every table.

### Consequences — all of these must be carried, not forgotten

**1. "Two overlapping voices" proves the target is present, permanently.** On `train` (19,950 trials), 9,858 trials have two voices and `P(target absent | two voices) = 0.000`. This is now a property of the task, not a defect.

**2. Our own benchmark cannot detect a model that exploits it.** The eval splits share the composition, so a system that learns *"two voices → produce output"* scores normally. **No number this project reports will reveal this behaviour**, and this must be stated wherever the results are.

**3. Every claim must be scoped in words, not just in the appendix.** Results are about *target speaker extraction from two-speaker mixtures*, never meetings, multi-party audio, or "conversation" in general.

**4. The AMI secondary check changes meaning.** AMI is multi-party, so it contains the excluded condition. Choose one option before running it:
   - restrict AMI trials to segments with at most two active speakers, testing transfer to real audio within the declared scope; or
   - keep AMI as-is and report it as an **out-of-scope generalisation probe**, where a poor result is expected and is not evidence of a bad extractor.
   Only the first is a transfer check; the second measures something else.

**5. Hallucination on unseen multi-party audio is an untested failure mode.** A model may emit an interferer's words as the target's when two non-target voices overlap. Under the content-fidelity metric this is the most damaging error class, and nothing in this project measures it. It belongs in the thesis limitations as a named, quantified gap, not a caveat sentence.

**6. `noise_only` and one-voice trials are unaffected.** The other two composition residuals are benign: *no voices means absent* is correct behaviour (silence is the right output when nobody speaks), and *one voice gives 44.7 % absent* is close enough to a coin flip to carry almost no information.

### What would reopen this

Evidence that hallucinated interferer speech dominates the live judge's content-fidelity score, or a supervisor requirement for multi-party. Reopening costs a fifth condition and one rebuild: cheap while no audio exists, increasingly expensive once eval audio is rendered and judged.

## 2026-08-15 — B2 evidence: LibriSpeech is 86 % speech; overlap is overstated by 25 %

Measurement only: no code changed, no manifest rebuilt. This is the evidence the B2 implementation rests on, recorded before it is written.

**Detector:** `silero-vad` 6.2.1 (Silero Team, 2021, `snakers4/silero-vad`), as a pip package so the weights are local and pinned (`torch.hub` would refetch at runtime and could silently change every number below). Settings `threshold=0.5`, `min_speech_duration_ms=100`, `speech_pad_ms=30`, with `min_silence_duration_ms` swept.

**Measured against** `data/manifests/` as built at commit `110e64e` (PR3), seed 42. Part 1: 2,000 utterances sampled from the 137,876 in `data/index/` (6.92 h). Part 2: 400 `both` trials from `train.csv` (864 distinct utterances).

### Part 1 — how much of an utterance file is speech

**Numbers below are from the reproducible run of 2026-08-16**, not the original spike (see *Reproduction* at the end of this entry). They differ from the spike's in the third decimal and change no conclusion.

| `min_silence_ms` | speech/dur | median | p10 | p90 | segments | lead_s | trail_s |
|---|---|---|---|---|---|---|---|
| 100 (silero default) | 0.850 | 0.860 | 0.754 | 0.934 | 3.76 | 0.326 | 0.161 |
| 200 | 0.858 | 0.870 | 0.760 | 0.943 | 3.24 | 0.326 | 0.135 |
| **250 (chosen)** | **0.862** | **0.873** | **0.762** | **0.947** | **3.09** | **0.326** | **0.121** |
| 300 | 0.870 | 0.882 | 0.772 | 0.954 | 2.79 | 0.326 | 0.102 |
| 500 | 0.899 | 0.911 | 0.805 | 0.979 | 2.08 | 0.326 | 0.005 |

Threshold sensitivity at 250 ms: 0.873 at 0.3, 0.862 at 0.5, 0.854 at 0.7.

**`min_silence_duration_ms` = 250 ms**, recorded in `generator.yaml`. The whole 100–500 ms range spans only 0.850–0.899, so the headline is robust to the choice. 500 ms is rejected: trailing silence collapses to 0.005 s, i.e. end-of-file silence is absorbed into speech.

**Two systematic offsets, which cause everything in Part 2:**

- **Leading silence 0.326 s per utterance**, near-identical at every setting (0.313–0.337 across the threshold sweep).
- **Trailing silence 0.121 s** at the chosen setting.

These are biases, not noise, and they compound: an adjacent target and interferer are each offset inward, so the generator believes they coincide more than they do. Carry these figures wherever an overlap number is quoted.

### Part 2 — the label error in the manifests as they stand

Onsets held **unchanged**; only the measurement differs. Sanity check: recomputing overlap the old way from the recorded onsets reproduces the stored `overlap_achieved` to max |diff| 0.00006.

| quantity | file-boundary | VAD | change |
|---|---|---|---|
| `overlap_achieved` (mean) | 0.285 | **0.215** | **−24.6 %** |
| `overlap_achieved` (median) | 0.264 | 0.195 | −26.1 % |
| `target_activity` (mean) | 0.642 | 0.555 | −13.5 % |
| `interferer_activity` (mean) | 0.467 | 0.407 | −12.9 % |
| `interrupted` (rate) | 0.570 | **0.505** | **−11.4 %** |

Per-trial |file − VAD| overlap: mean 0.070, p90 0.148, max 0.270.

**The `interrupted` row was wrong until 2026-08-16:** it reported 0.725, definition **B**, which Part 3 *rejected*. Under the chosen definition A it **falls** to 0.505. Consequence 1 below was rewritten to match.

**The per-trial spread is why a correction factor cannot fix this.** A uniform −25 % would leave the B13 overlap strata intact, but an error varying from 0 to 0.270 per trial puts individual trials in the wrong bucket, and the per-condition table is the thesis's central artefact.

**Overlap collapsing to zero: 12/400 (3.0 %).** Both speakers talk but never simultaneously: the shared interval falls entirely inside one speaker's leading silence and the other's trailing silence, so turn-taking is recorded as talking over each other. That this is *small* is reassuring: the overlap is mostly real, just consistently overstated.

### Three consequences for the implementation

**1. `interrupted` moves, and the cause is definitional. Resolved: option A.** The old test checks one moment per interferer utterance (its file onset). A VAD test could check *every* speech-segment onset, ~3 per utterance, so an interferer *resuming after their own breath pause* mid-target-sentence would count as an interruption.

Measured side by side (Part 3, 400 `both` trials):

| definition | rate | |
|---|---|---|
| old (file onsets) | 0.570 | what the manifests carry today |
| **A: first speech onset per utterance** | **0.505** | **CHOSEN** |
| A′: first speech onset per trial | 0.495 | only the interferer's very first word |
| B: every speech onset | 0.725 | a breath pause starts a new turn |

**Decision (GB, 2026-08-15): option A.** It is the minimal correction to the 2026-08-14 definition: the same number of events as the old test, each moved by the ~0.326 s of leading silence, so `interrupted` keeps meaning "began a turn while you were mid-sentence". B *widens* the definition instead, so its rise from 0.570 to 0.725 would be an artefact, not a measurement. Implemented as `vad.onsets_of(..., first_only=True)`.

**The spread across definitions (0.495–0.725) is wider than most effects this project will report**, on a B13 reporting condition. So it is pinned here, and any `interrupted` figure must state which definition produced it.

**2. The "69 % outside `overlap_tolerance`" figure is NOT a rejection rate**, as the spike labelled it. It counts trials whose overlap drifts outside tolerance *with placement held fixed*, but the implementation re-runs `best_onset` against the new measurement, re-placing most of them. It measures how far placement must move and is an upper bound on rejections; the true rate cannot be estimated without building it.

**3. The top of `target_activity_ratio` becomes unreachable.** Realised footprint activity already reaches p99 0.910 and max 0.946 (`activity_tolerance: 0.1` carries it past the 0.85 config ceiling). 0.85 of *speech* would need 0.85/0.86 = 0.988 of the window filled with audio, beyond anything achieved and leaving no room for the gaps `lay_out` requires. The practical ceiling is near 0.78–0.80. The REAL-TSE anchor of ~0.75 still fits, so the band needs adjusting, by a recorded decision rather than a silent config edit. **Done 2026-08-16: lowered to 0.78.** See that entry.

### Why this is done before the renderer, not after

The audio is unaffected (mixtures still contain the pauses, as real speech does); only labels and placement change. But every overlap and activity column in all six manifests moves, forcing a rebuild, which is cheap only while no audio exists. `milestones.md` already places B2 before the rebuild for this reason.

### Precision footnote

These numbers were first measured with Silero's `return_seconds=True`, which also rounds boundaries to `time_resolution` decimal places, defaulting to **one** (0.1 s). Re-measured on 600 utterances with sample-accurate boundaries, mean speech/duration moves only from 0.8655 to 0.8668 and leading silence from 0.321 to 0.317, so the figures stand. Individual utterances move by up to 0.042, though, and overlap bucketing needs per-trial precision, so `src/data/vad.py` takes sample indices and divides itself rather than using `return_seconds`.

### Reproduction

Promoted from a throwaway spike to `scripts/measure_vad_impact.py`, so a supervisor can re-derive every figure:

    ../tse_venv/bin/python scripts/measure_vad_impact.py

**Run 2026-08-16, 16 min**, against `train.csv` as built at `42a3854`, config md5 `d8bf16d`, seed 42. The sanity check passed at max |diff| 0.00006. Output (`report.txt`, plus `meta.yaml` with commit hash, seed, detector version and the manifest's own build commit) is in `experiments/results/2026-08-15-vad-impact/`. The directory keeps the 2026-08-15 date of the decision it supports; `meta.yaml` carries the true run date.

**Every table above comes from that run.** It differs from the spike only in the third decimal, except that the spike's `interrupted` row reported definition B rather than A (corrected in Part 2). Re-run after PR2 rebuilds the manifests, because Part 2 measures label error in manifests that will then no longer exist in that form.

## 2026-08-15 — Training audio is pre-rendered to disk, not generated on the fly

**Decision (GB, 2026-08-15): render the training mixtures to disk once, the same way `val` and the two eval splits already are. On-the-fly generation is removed from the plan and kept only as the mechanism B7 would need if it were ever switched on.**

Supersedes `docs/data/data-setup.md` and `docs/data/data-map.md`, which both stated "0 GB, generated on the fly, never stored" as settled fact.

### Why the original reason no longer holds

`data-setup.md` chose on-the-fly generation so the training set would "cost no disk and **never repeat**". B7, decided the same day, turned the repetition off:

> Rebuilding mixtures with fresh loudness and room draws every epoch gives the model more variety, but the run can then no longer be reproduced from the manifest alone. **Decision: off for the main run.**

With B7 off, every draw is fixed in the manifest, so on-the-fly rendering produces byte-identical audio every epoch. What remains is recomputing the same 20,000 mixtures for every training run.

### The other two arguments, checked rather than assumed

**Disk.** 251 GB free. The rendered training set is ~26 GB as 16-bit PCM at 16 kHz (20,000 trials x ~40.8 s of audio each: mixture ~17.9 s including the A5 tail, clean target ~17.9 s, enrollment 5 s), or ~16-18 GB as FLAC. Not binding.

**Portability, which inverts.** `data-setup.md:440` argued on-the-fly meant "no 70 GB training set to move, only ~36 GB of public corpora". But a rendered trial is self-contained and needs **no LibriSpeech and no WHAM!** on the training machine. Pre-rendering moves ~26 GB; on-the-fly needs the full ~36 GB of corpora wherever training runs.

### The risk this actually removes

Each trial needs two RIRs (target and interferer), image-source method, `t60_s` up to 0.6 s. On the fly, the dataloader pays that every epoch while trying to feed the GPU. On a 4-vCPU Kaggle instance a CPU-bound loader starves the GPU and burns session hours, the failure `milestones.md` M1 guards against when it says discovering a problem at hour 11 of a 12-hour session costs a week. Pre-rendering makes the training step a disk read.

**Taken without the RIR benchmark.** The argument does not depend on the per-trial RIR cost. That number is still needed to *size* the render job, not to choose it.

### What this concedes

- **No augmentation variety**, which B7 already conceded deliberately.
- **Every manifest rebuild invalidates the audio.** Render *after* B2's rebuild lands, never before; rendering ~21,200 trials and then rebuilding would waste the pass.
- **~26 GB.** Cheap at 251 GB free. Exclude it by name from any `rsync` to a cluster, as the corpora already are.

### What stays

`render_trial(row, cfg) -> stems in memory` stays a pure function with the writer as a thin caller, so a PyTorch `Dataset` can call it directly if B7 is ever switched on as the ablation it was reserved as. The decision changes how the renderer is *invoked*, not what it is.

## 2026-08-16 — B2: `target_activity_ratio` ceiling lowered 0.85 -> 0.78

**Decision (GB, 2026-08-16): the top of `target_activity_ratio` comes down from 0.85 to 0.78, in both the global band and `base`.** Config only; it takes effect at B2 PR2's rebuild, not before.

### Why

`target_activity_ratio` is how much of the clip the target spends talking. PR2 changes "talking" from file length to detected speech.

A LibriSpeech file is only ~86 % speech, the rest being the pause before the reader starts, the pause at the end, and gaps between phrases. 0.85 of the window as voice would need ~0.99 of it packed with back-to-back audio, but `lay_out` must leave gaps between utterances and has never achieved more than 0.946. **0.85 is therefore impossible, not merely rare.**

### What leaving it at 0.85 would have done

Nothing visible: `build_trial` tries 20 times to find a long enough run of utterances, fails, returns `None`, and the trial goes into `failed_ids`. The build would quietly produce fewer trials, all lost from the talkative end of the band, so the realised distribution would stop matching the configured one. That is the "rejection bends distributions" problem of 2026-08-13, through the back door.

### Why 0.78 and not lower

The REAL-TSE anchor of ~0.75 still sits inside the band.

**Limit:** 0.78 is close to the edge. Realised footprint reaches p99 0.910 and max 0.946, which at ~86 % speech is roughly 0.79 and 0.82 of speech. The top of the band is reachable only in the best cases, and a few top-of-band trials will still fail. **The check is PR2's `n_failed`.** If it is material, lower again, with an entry here rather than a silent edit.

## 2026-08-16 — B2: noise beds containing speech are rejected at 0.5 s

**Decision (GB, 2026-08-16): drop any WHAM! noise clip whose longest unbroken run of detected speech reaches 0.5 s.** Config `noise_speech_rejection.max_speech_run_s`; applied in `build_manifest.py` by `reject_speech_clips()`. This closes `noise_speech_rejection`, which `data-construction-parameters.md` has called critical and unimplemented since it was written.

### Why

WHAM! was recorded in real cafes, bars and restaurants, so some beds have audible background talkers. In a trial these cause two problems:

1. **The metric punishes the model for being right.** Those words are in the mixture but in no transcript, so if the extractor passes them through and the judge hears them, they score as invented words.
2. **A third talker appears**, breaking the two-speaker task CLAUDE.md declares.

### Why the longest run, and why 0.5 s

Rejection is on `max_segment_s`, the longest *unbroken* run, not the total. Half a second in one piece can be a word; the same half second spread over five 100 ms blips is the detector twitching at a laugh, a door or a clatter. Below ~0.5 s there is not enough continuous voice to become text, so a lower cutoff would cost pool for no gain.

### Borrowed in intent from WHAMR!, but not the same rule

`data-construction-parameters.md` took this parameter from WHAMR!'s `SNR_THRESH` (`noisesampler.py:45-62`), which rejects a noise segment when its speech **energy** exceeds −6 dB. **We diverge:** we threshold the **duration of a detected speech run**, using the Silero pass B2 already pays for. Our failure mode is words being *transcribed*, and a quiet but clear background talker is a transcription risk at an energy WHAMR!'s test would pass. We also reject whole clips rather than resampling the offset, so nothing is written back to the manifest. Cite the idea as borrowed; do not describe the rule as WHAMR!'s.

### Cost

Measured, not projected (`scripts/screen_noise_speech.py`, 2026-08-15):

| pool | clips | dropped | kept |
|---|---|---|---|
| tr | 20,000 | 821 (4.1 %) | 19,179 |
| cv | 5,000 | 98 (2.0 %) | 4,902 |
| tt | 3,000 | 39 (1.3 %) | 2,961 |

### Where it is applied, and why not in the renderer

`milestones.md` put this in the renderer, assuming it needed audio. It does not: the screening pass already measured every clip, and the manifest *names* each trial's noise clip. Filtering the pool before selection is the only place it works; filtered later, the manifest could still point at a clip that should not exist.

### What this does not promise

The cutoff inherits Silero's mistakes, so a kept clip may still hold faint or short speech: contamination is much lower, not proven zero. The write-up should say "screened", not "clean".

### Consequences

- `data/index/noise_speech_{split}.csv` is now a required **build input**, not a report. `build_manifest.py` fails loudly if it is missing or does not cover the pool.
- The `vad:` block already invalidates the manifests; it now also invalidates this screening index, which the same detector settings produced.
- Each split's `meta.yaml` records `noise_clips_screened`, `noise_clips_kept` and `noise_max_speech_run_s`, because the cutoff is a config value and manifests are not in git.

## 2026-08-16 — B2 PR2: the generator measures speech, not file boundaries

**Implemented (GB, 2026-08-16): `build_manifest.py` now measures activity, overlap and interruption from `data/index/vad_segments.csv` instead of from file durations, and all six manifests are rebuilt.** Implements the evidence entry above. Branch `m0-b2-pr2-vad-wire`.

### The one distinction the code now makes

One quantity became two, and confusing them is the way to break this file:

| | what it is | what it drives |
|---|---|---|
| **footprint** | timeline the audio occupies, silence included | `lay_out`, `best_onset`'s slide range, the "does it fit" assertions |
| **speech** | detected voice inside that audio | `target_activity`, `interferer_activity`, `overlap_achieved`, `interrupted` |

They differ by a measured factor of **0.8622** on the rebuilt train split, matching the 0.862 corpus figure. Recorded per trial as the new `target_footprint_s` and `interferer_footprint_s` columns, so the distinction is visible in the data, not only in the code.

### What was wrong, in one number

Recomputing true speech overlap from the *old* manifests' own recorded onsets:

| | overlap label | true speech overlap | rows whose label was wrong by >0.01 |
|---|---|---|---|
| before | 0.2843 | 0.2118 | **9,419 / 9,858 (95.5 %)** |
| after | 0.2750 | 0.2750 | **0 / 9,846 (0.0 %)** |

Mean label error was 0.0725, reaching **0.485** on individual trials. After the rebuild the maximum disagreement is 0.0001, the column's rounding.

**The audio changed too, not just the labels.** Real speech overlap in `both` trials rose from 0.212 to 0.275, because `best_onset` now slides the interferer to hit the *requested* overlap measured in speech. The requested distribution is unchanged; the generator now delivers what it always asked for.

### Changes

- `pick_run` accumulates speech and gained a **footprint cap**: selecting on speech lets a run reach the wanted amount while its audio overruns the window, and `lay_out` would then scatter negative gaps.
- `lay_out` is deliberately **unchanged**: a file occupies its whole duration whether or not anyone is speaking in it.
- `spans` / `shared_seconds` / `is_interrupted` deleted in favour of `src/data/vad.py`; `tests/test_vad.py` asserts they do not come back.
- `best_onset` slides the interferer's real speech spans rather than one solid rectangle, so gaps inside the interferer count as not-talking.
- `interrupted` uses option A (`vad.onsets_of(..., first_only=True)`), verified against all four candidate definitions on the rebuilt data.
- One utterance of 137,876 has no detected speech and is dropped from the pools.
- Each manifest's `meta.yaml` records the `vad:` settings, the index build date and the detector version, without which an overlap figure means nothing.

### Results of the rebuild

| split | rows before → after | `interrupted` before → after |
|---|---|---|
| train | 19,950 → 19,938 | 0.532 → 0.523 |
| val | 200 → 200 | 0.694 → 0.667 |
| eval_public | 500 → 500 | 0.600 → 0.661 |
| eval_private | 499 → 500 | 0.645 → 0.679 |

**The 0.78 ceiling holds.** Unsatisfiable train trials went from 50 to 62 of 20,000 (0.25 % to 0.31 %). The 2026-08-16 entry said "watch `n_failed`, lower again if it is material"; +12 trials is not material, so 0.78 stands.

**The leak scoreboard is unchanged**, as B9 and B10's properties had to survive. On train: `interferer_activity` AUC from 0.648 to 0.651, speaker-identity prior from 0.508 to 0.503, P(absent | no overlap) from 0.500 to 0.500, and per regime 0.649 `base` / 0.653 `hard`. The `interferer_activity` leak is the pre-existing one the notebook examined and accepted, neither worsened nor fixed here. `interferer_footprint_s` carries the same 0.651: a new column exposing a known leak, not a new leak.

**`check_manifest_parity.py` fails on all six splits, as designed.** Every overlap and activity number was supposed to move.

### Still outstanding

- The notebook (`src/exploratory/data_setup.ipynb`) still shows the old numbers. The scoreboard above was reproduced standalone, not by re-running it; §2, §7 and the health checks are a separate milestone item.
- **`measure_vad_impact.py` Part 2 was rewritten the same day** (entry below), so re-running it is now worthwhile rather than fatal.
- PR3 (enrollment offset) is untouched and still optional.

## 2026-08-16 — `measure_vad_impact.py` Part 2 detects which manifest it is given

**Problem: PR2 broke the measurement script that justified PR2.** Part 2's sanity check recomputed overlap the file-boundary way and asserted it matched the stored `overlap_achieved`. After the rebuild that column is speech overlap, so the check compared two deliberately different quantities, measured `max|diff| = 0.553` against a 0.001 threshold, and exited with "results are void". Caught before the re-run finished, not after acting on a bad number.

**Fix: detect the convention instead of assuming it.** PR2 added `target_footprint_s` when it changed what `overlap_achieved` means, so that column's presence identifies the manifest's generation. Part 2 branches on it:

| manifest | check asserts | what the file-vs-VAD gap means |
|---|---|---|
| pre-PR2 | file-boundary recompute == stored column | **live label error** — the measurement B2 was argued from |
| post-PR2 | VAD recompute == stored column | **what B2 is worth** — how wrong these labels would be without it |

Verified both ways: against the backed-up pre-PR2 manifests the check passes at 0.00006 and reproduces the recorded ~25 % overstatement; against the rebuilt ones it passes at 0.00005.

**Why not delete Part 2.** That would make the 2026-08-15 entry's numbers impossible to re-derive, and a figure nobody can reproduce is a claim, not a measurement. With both paths a supervisor can re-derive the before *and* the after from one script.

Two smaller corrections from the same pass:

- **The activity rows were degenerate on a post-PR2 manifest.** They read `target_activity` as the "file-bound" value, which is now the speech figure, printing a 0.0 % change under a heading that said otherwise. Activity is now recomputed from `target_footprint_s`, and the row reads −14.2 %, matching the corpus speech/footprint ratio.
- **"outside `overlap_tolerance`"** meant "how far placement must move" before the rebuild and now means "trials that missed the overlap they requested". It should be ~0 post-PR2 and measures 0/60 on the sample. Since the reading flips, the printed explanation switches with the branch.

### Standing consequence

Part 2 is now also a **regression check**: it fails loudly if the manifests and the detector ever drift apart.

### Result of the re-run, 2026-08-16 (14 min)

`CHECK` passed at 0.00007, convention auto-detected as `speech`. Part 1's corpus figures are unchanged, as the detector and corpus did not move.

Part 2 against the rebuilt `train.csv`, and against the pre-PR2 report preserved in commit `6612414`:

| | pre-PR2 | post-PR2 | reading |
|---|---|---|---|
| file-bound vs VAD overlap | 0.285 → 0.215 (−24.6 %) | 0.387 → 0.275 (−28.8 %) | what B2 is worth |
| activity gap | −13.5 % | −14.3 % | tracks the 0.862 corpus ratio |
| overlap collapses to zero | 12/400 (3.0 %) | **0/400** | |
| outside `overlap_tolerance` | 273/400 (68.2 %) | **0/400** | |
| per-trial \|file − VAD\| | mean 0.070, max 0.270 | mean 0.111, max 0.527 | |

**Two predictions from the 2026-08-15 entry came true.** It said the 273/400 figure was "NOT a rejection rate ... PR2 re-runs `best_onset` against the new measurement and re-places most of these". It re-placed *all* of them: 0/400 now miss their requested overlap. The 12 trials whose overlap was only two speakers' silences overlapping are also gone, because placement now optimises against speech.

**B2 is worth more after the rebuild, not less.** The per-trial gap grew from 0.070 to 0.111 mean. Reaching a speech target takes more audio than the same file-duration target, so the corrected manifests carry a longer footprint per trial, and file-boundary measurement would now be *more* wrong than on the data that motivated the fix.

Part 3, on the rebuilt data: option A 0.500 (what the manifests carry), A′ 0.463, old file-onset reading 0.555, B 0.797. The spread is as wide as before, so any `interrupted` figure must still state which definition produced it.

## 2026-08-16 — The renderer: manifest row to audio

**Implemented (GB, 2026-08-16): `src/data/render.py` (pure) plus `scripts/render_trials.py` (writer).** The last M0 code item. Nothing is rendered to `data/rendered/` yet beyond timing batches; this records the design and the measurement, not a completed render.

### Measured, so the M0 date stops being a guess

`render_trials.py` was the only row in `run_times.md` marked **unknown**. 100 train trials, 8 workers: **23.4 s**, 126 MB.

| | measured | full set |
|---|---|---|
| rate | 100 trials / 23.4 s | **~83 min** for all 21,208 |
| size | 1.26 MB / trial | **~27 GB** |

27 GB matches the ~26 GB the docs already budget, so `data-map.md` and `data-setup.md` need no revision. `milestones.md` feared an overnight job; it takes an hour and a half and is resumable, so the renderer is no longer the schedule risk it was written up as.

### Format: 16-bit WAV at 16 kHz

Decision (GB, 2026-08-16). FLAC would roughly halve the transfer to a cluster. WAV wins because decode-free reads are the purpose of pre-rendering (2026-08-15): a training step is a pure disk read, and on a 4-vCPU box a CPU-bound loader starves the GPU. Disk is not scarce at 251 GB free.

### What each trial gets

    data/rendered/<split>/<trial_id>/
        mixture.wav      what the model hears
        target.wav       A1's reference: the target through its own room, alone
        enrollment.wav   dry, no room (A4)
        meta.json        both transcripts, every gain applied, the EQ curve

`meta.json` carries `target_text` and `interferer_text` verbatim. `d` in `metric-definitions.md` §2 needs the interferer's words too, and adding them later would cost a full re-render.

### Three things the specification did not settle

Called out rather than silently chosen: no existing entry contradicts them, but none decided them.

**1. The noise bed covers A5's tail as well as the window.** A2 (2026-08-11) says the renderer reads `mixture_length_s` of noise; A5 (2026-08-13) later made the output `mixture_length_s + t60_s` long. Taken literally, the last `t60_s` would be reverb over silence: an audible edge and a free cue for where the speech ended. Noise runs the full padded length.

**2. `noise_only` trials place the noise where an anchor would have put it.** Nobody speaks, but `target_loudness_lufs` and `snr_db` are still recorded. The noise sits at `target_loudness_lufs - snr_db`, the level it would have had if someone had spoken, so (as in the 2026-08-11 anchor decision) the condition is not identifiable by loudness alone.

**3. The enrollment is levelled to `target_loudness_lufs`.** No decision covers the enrollment's level. LibriSpeech's native level would put a loudness spread on the conditioning path for no reason, and level is a cue closed everywhere else.

**Also interpreted: the EQ curve itself.** `enrollment_eq_augmentation` specifies "random RMS-preserving EQ curves" (CARTSE channel-gap, Li & Seki 2026) without fixing them. Three peaking biquads, log-uniform 120 Hz–6.4 kHz, ±6 dB, Q=1.0, RMS restored after filtering. The drawn bands go into each trial's `meta.json`, so the filter is reproducible and reportable rather than an undocumented default. Seeded from a SHA-1 of the trial id, not Python's `hash()`, which is salted per process and would change the audio between runs.

### Verification

The renderer's failure mode is audio that is wrong but plays fine, so "it ran" is not enough. Checked on 150 trials across all four conditions:

| property | result |
|---|---|
| output length == `mixture_length_s + t60_s` (A5) | exact, 0 samples error |
| target stem at `target_loudness_lufs` (A3) | max error 0.066 LU |
| SNR, measured as `mixture − target` on `target_only` trials | max error 0.0002 LU |
| SIR, recomputed through the render chain | max error 0.051 dB |
| nothing exceeds the 0.95 ceiling (A6) | max peak 0.71, 0 trials rescaled |
| target stem silent **iff** `target_absent` (A1) | holds on all 150 |
| both transcripts present and non-empty when the speaker talks | holds on all 150 |

18 unit tests in `tests/test_render.py` pin each decision to a property, including that BS.1770 stays gated (A3's argument over RMS), that a silent stem raises rather than returning infinite gain, and that the RIRs are deterministic.

### Sequencing, unchanged

Any manifest rebuild invalidates the audio. `render_trials.py` records the manifest's `config_md5` and refuses to extend a directory built from a different one without `--force`.

---

## 2026-09-03 — M0 housekeeping: three data-construction items cut

Closing M0's tail so an open checklist stops reading as unfinished work. **All three are cut because the data is rendered**, not because they were wrong.

| item | cut because |
|---|---|
| **Narrow `overlap_ratio` in `base`** | B1 said narrow it *last*. That point has come and the answer is no: every split is rendered and `sir0_train` is mid-extension to 9,955, so changing the sampler invalidates all of it and breaks comparability with every result to date. The 0.7 ceiling stays, matched to REAL-TSE |
| **Enrollment offset (B2 PR3)** | Real but optional from the outset. `enroll_offset` is uniform and `long_enough` filters on file duration, so a 5.2 s file with 1.2 s of leading silence gives a "5 s enrolment" holding 4 s of voice, against B10/A1's ≥5 s. Fixing it re-renders every enrolment plus its 3-variant bank and breaks comparability with every trained checkpoint. **Carried as a known enrolment-quality caveat instead** |
| **`length_mode`** | Never implemented, referenced by nothing downstream, and every split was rendered without it. Stays documented as not-implemented in `data-construction-parameters.md` |

**The enrolment-offset caveat must reach the thesis.** Some enrolments carry less than the nominal 5 s of voice, weakening the conditioning signal on those trials, which matters when reading enrolment sensitivity.

Also closed, not cut: **HPC access** (never secured; every run is on Kaggle T4, and it stopped mattering once M5 became a per-band gate fine-tuned from an existing checkpoint) and **the two-speaker thesis text** (tracked in M7, its right home).

Still open in M0: the three interpreted renderer points (supervisor glance), per-parameter EDA, and refreshing the pre-rebuild figures in `src/exploratory/data_setup.ipynb`.

## 2026-10-06 — AMI secondary eval set built: 300 two-speaker trials
Settles the 2026-08-14 choice: **restricted to two active speakers**, so AMI tests real audio inside the declared scope. Eval only (`ami` is render-only in `scripts/train.py`); never trained, filtered or selected on.

**Construction** (`scripts/build_ami_trials.py`, `experiments/configs/ami_eval.yaml`). Borrowed from REAL-T (Li et al., Interspeech 2025), cited as borrowed.
- Source: AMI scenario meetings (Carletta et al., 2005), CC BY 4.0. 15 groups, 5 per site (Edinburgh, IDIAP, TNO), seeded draw from groups with ≥ 20 usable windows.
- Window: 10–20 s, exactly two people active (vocal sounds count), both ≥ 5 words, no `gap`/`transformerror` marks, cut inside a ≥ 0.3 s all-silent gap. 20 per group.
- Mixture = table mic `Array1-01`; `target.wav` = target headset (**approximate**); enrolment = 5.0 s solo stretch from the target's headset in a **sibling meeting**.
- Level as recorded. No gain, no matching to the constructed sets.

**Differences from REAL-T:** two speakers only; enrolment from a different meeting; scored by the LCF judge. Not comparable to REAL-T numbers.

**Found while building:**
- IS1003b has no `Array1-01` on the server, so it is an enrolment source only.
- ES2005 has 19 usable windows, so 1 is topped up from other groups, seeded.
- NXT letters A–D change person across a group's meetings (4 of 15 groups), so enrolment is matched on `global_name`.
- TNO speakers are absent from `participants.xml`: sex from the ID code (agrees for all 189 recorded), native language inferred (Dutch), age unknown.
- Natural overlap is low: median 7.5 % vs ~25 % in the constructed sets.

Set: 300 trials, 77.9 min, 58 meetings, 60 speakers (45 M / 15 F, 21 native English). Report: Appendix D, `tab:ami`.

## 2026-10-12 — AMI is no longer eval-only (pointer)
The 2026-08-07 "AMI secondary eval only" rule is superseded for the 20 AMI scenario groups `ami_eval` does not use: they become `ami_train` for M8. `ami_eval` stays held out. Decision and reasons: `decisions-m8.md` 2026-10-12.
