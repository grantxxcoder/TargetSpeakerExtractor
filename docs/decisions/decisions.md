# Decision Log

## 2026-08-04 — Repo workflow: branch protection + PR review
Chose to enforce PRs on main (no direct commits) even as a solo project,
so every code change gets a deliberate diff-review checkpoint before
becoming part of the "real" codebase — needed since all submitted code
must be individually understood and defensible.

## 2026-08-04 — Eval data: REAL-TSE dev/eval set unavailable
**Status: partially superseded by 2026-08-07 re-scope. Retained for history.**
Challenge registration closed 31 May 2026, before we could register, so
we do not have access to the official dev/eval set. Decision: replicate
baseline training on Libri2Mix-100 + WHAM! (identical to the baselines'
own training data), and construct our own REAL-T-style eval set from
public training splits of AMI (then AliMeeting/AISHELL-4), following the
organizers' documented construction method (overlapping segments as
mixtures, ≥5s non-overlapping segments as enrollment). We use the
official REAL-TSE-Challenge scoring code for evaluation to keep numbers
methodologically comparable. Any comparison to published baseline
numbers will be caveated as "different eval audio, same eval method" —
not treated as a direct apples-to-apples result.
Also emailed organizers [date] to ask about academic access to the
official set; will update if granted.

*What survives:* constructed training data from LibriSpeech-derived
mixtures, and an AMI-derived real-audio set. *What does not:* the
"replicate the baselines" framing and the use of the official scoring
code as our primary evaluation. Access to the official dev/eval set is
no longer on the critical path — it would be a nice-to-have external
validation set, nothing more.

---

## 2026-08-07 — Re-scope: downstream live-model content fidelity is the objective
Supervisor meeting notes 8-10 (docs/decisions/specification.md) redefine the goal.
The project is no longer "replicate the REAL-TSE online baselines, then
explore an architecture." It is:

> Build a streaming TSE model that maximises how accurately a live
> speech-to-speech model recovers what the target speaker said.

Explicitly **not** optimising the perceptual or signal quality of the
separated audio. Better quality may help downstream understanding, but it
is a means, not the objective, and the two can diverge — the supervisors
observed that conventional TSE can improve ASR transcription accuracy
while making audio *harder* for live speech-to-speech models. Measuring
and characterising that divergence is a core result of this project.

Consequences, each logged separately below: challenge replication is
dropped, the on-device leg is dropped, the metric becomes the primary
contribution rather than a novel add-on, and the latency budget is
re-derived.

## 2026-08-07 — Dropped: replication of REAL-TSE baselines and eval pipeline
Spec note 8: "I do NOT need to reimplement anything from the 2026
challenge but I can borrow ideas from it." Dropped accordingly.

Rationale beyond the instruction: we cannot validate a replication we
have no eval data for, so it would have been a reimplementation claimed
as a replication — a weak position to defend. The compute it would have
consumed is redirected to the metric work and to proxy-objective
training, which are the actual contributions.

**What we still borrow, with citation:** the REAL-T trial-construction
method (Li et al., Interspeech 2025) for the AMI set; BSRNN-family causal
architecture (Luo & Yu, TASLP 2023) and TF-Map conditioning (Zhang et
al., ICASSP 2025) for the model; CARTSE's target-absent training and
channel-gap enrollment augmentation (Li & Seki, 2026); and the
challenge's DNSMOS-gaming episode as motivating evidence for
gaming-resistant metric design.

**Never claim comparability** with published REAL-TSE numbers. Different
data, different metric, different protocol.

## 2026-08-07 — Dropped: on-device / small-model deployment leg
Spec note 10: "We can assume we are also working with something as big as
a server, and not necessarily optimise for a small device like a
wearable." The parameter/MAC budget study, quantisation, and the
TF-MLPNet-class on-device backbone route are all out of scope.

TF-MLPNet and related work stay in the literature review as background
and as named future work — the sequential-vs-parallel argument is still
architecturally relevant under a latency budget — but they are no longer
a build target. Note this reverses the "highest-value gap" identified in
literature/review_synthesis.md, which was written under the old brief.

## 2026-08-07 — Training: differentiable proxies, live model as held-out judge
A live speech-to-speech API model is a black box and cannot be
backpropagated through. It is therefore used **only** as a held-out judge
at evaluation time.

Training optimises differentiable proxies for the same underlying goal:
frozen-ASR/SSL encoder feature matching as the primary proxy (cheaper
than full ASR cross-entropy and precedented by CARTSE's Zipformer
feature-matching loss and PS4's proxy-supervised objectives), plus
signal-level, speaker-similarity and target-activity terms.

**Non-negotiable rule: the proxy model must be a different model family
from the judge.** If we optimise against the same model that scores us,
the benchmark measures overfitting to one evaluator rather than genuine
downstream intelligibility — which is exactly the failure mode the
challenge's DNSMOS-OVRL episode demonstrated. The judge must also never
be used as a training-data filter.

## 2026-08-07 — Latency budget: ~200-300 ms, derived not inherited
Spec note 10 makes latency a secondary objective and assumes server-class
compute, so the REAL-TSE online track's 100 ms cap no longer binds us.

Decision: target a ~200-300 ms end-to-end streaming budget, justified by
turn-taking tolerance in live conversational agents rather than by the
challenge rules. The model must still be causal/streaming, and measured
algorithmic latency plus RTF must be reported for every system.

**Open caveat — this number is currently an assumption, not a result.**
We do not yet have evidence for what latency live speech-to-speech models
actually tolerate before interaction degrades. Either find published
evidence or measure it as part of the metric work; until then the budget
must be presented as a stated assumption. Do not present 200-300 ms as
grounded until it is.

## 2026-08-07 — Data: constructed primary, AMI secondary
**Training must be constructed.** Differentiable proxies need a clean
target signal (for signal-level loss) and exact verbatim text (for
ASR-based proxies). Real conversational corpora provide neither: AMI has
no clean target — the headset channel carries cross-talk bleed from other
speakers — and its human transcripts are not verbatim. Training proxies
on real audio would require an offline pseudo-label teacher, i.e. the
CARTSE pipeline, which is out of budget.

**Primary eval: constructed mixtures**, same construction family as
training but speaker-disjoint and held out. Gives exact ground-truth
target text, a true clean-target ceiling, and controllable overlap ratio,
SNR and device mismatch as experimental variables.

**Secondary eval: AMI-derived trials**, as the real-audio transfer check.
Without this leg the project only ever measures itself in the conditions
it was trained for, and the spec's real-world motivation goes untested.

**Caveat that must appear in code comments and in the thesis:** on AMI
there is no clean target, so the ceiling condition is *approximate*,
computed from the individual headset (IHM) channel. IHM has cross-talk
bleed and a different channel response from the distant mic used as the
mixture. It is an approximate reference, never ground truth.

## 2026-08-07 — Output modality: audio is the build target, text is a reference condition
Spec note 10, final two sentences: the live model accepts **either text or
audio**, so the extractor's output modality is an open design choice.

Decision: **the system we build outputs audio.** Reasons, in order:

1. **Latency.** A text path must run a streaming ASR to completion and
   endpoint before anything can be sent. That serialises a whole decoding
   stage behind extraction and adds endpointing delay on top, which the
   ~200-300 ms budget does not have room for. The spec reaches the same
   conclusion.
2. **It keeps the research question intact.** The project's premise is that
   live speech-to-speech models are unusually sensitive to *processing
   artefacts in audio*. Hand the judge text and that premise no longer
   applies — there is no audio for it to mishear. A text-only project would
   be a streaming target-speaker-ASR project, which is a different thesis.
3. **Text discards everything non-lexical.** Prosody, emphasis, hesitation,
   emotion and speaker identity are all thrown away at the ASR boundary, and
   a speech-to-speech model uses them both to interpret the turn and to
   shape its spoken reply. Our metric is lexical and would not see that
   loss, which is a reason to be suspicious of a good text-path score, not a
   reason to prefer the path.

**But the text path is not dropped — it is demoted to a reference
condition** in the benchmark: leg-3 extractor → an off-the-shelf streaming
ASR → text → judge. No extra training, one extra row in the results table,
and it answers "how much of the content is recoverable at all once the
extractor has run?" It is measured, reported, and not optimised for.

**Do not call the text row an upper bound.** It is not guaranteed to be one.
An ASR error is unrecoverable once committed to text, whereas the audio path
leaves the judge the acoustic evidence to work from. Expect it to win
sometimes and lose sometimes; which, and under what conditions, is itself a
result.

Consequence for the metric: **output modality becomes a recorded property of
every trial**, and the metric is defined so both paths are scored by the same
end-to-end question — what did the assistant recover? See
`docs/data/metric-definitions.md` §3.5. Cross-modality comparisons are valid on
that end-to-end number and invalid as statements about the judge's listening
ability, because in the text condition the judge is close to a pass-through.

## 2026-08-07 — Metric is the primary contribution
Spec note 1 ("the actual score values from my defined metric do not
matter — the metric itself matters more") plus note 8 (metric first, then
architecture) make this explicit. The metric definition and its harness
are the deliverable that must not be cut under any schedule pressure.
Full specification in docs/data/metric-definitions.md.

## 2026-08-10 — Sample rate: 16 kHz mono, 16-bit, everywhere
Every audio path in this project — source corpora, generated mixtures,
enrollment, model I/O, and the audio handed to the judge — is **16 kHz,
mono, 16-bit PCM**. This is a hard project-wide invariant, not a per-stage
choice.

Reasons, in descending order of how costly they'd be to get wrong:

1. **Every frozen model in the loop is a 16 kHz model.** The differentiable
   proxies (frozen ASR/SSL feature matching), the speaker encoder and the
   VAD are all trained at 16 kHz — WavLM, Whisper, Zipformer, ECAPA-TDNN
   and WeSpeaker all assume it. Since the proxy losses *are* our training
   signal, the pipeline is pinned to 16 kHz by that alone.
2. **The judge takes 16 kHz.** Gemini Live's input format is raw 16-bit
   PCM, 16 kHz, mono, little-endian. It will resample other rates, but an
   uncontrolled resample sitting between our extractor and our evaluator
   is a confound in a benchmark whose whole purpose is measuring what the
   judge hears. We control that boundary explicitly.
3. **Intelligibility content is below 8 kHz.** Nyquist gives 8 kHz of
   bandwidth at this rate. Vowels, formants and voicing sit well under
   4 kHz; what is lost above 8 kHz is mainly the upper energy of sibilant
   fricatives (/s/, /ʃ/, /f/), which affects perceived crispness more than
   word identity. Since our metric is lexical content fidelity, this is the
   band that matters.
4. **It is the field standard**, so our conditions are comparable *in kind*
   to REAL-TSE, LibriSpeech, WHAM! and the published TSE literature — while
   remaining non-comparable in number, per the standing rule.

**Consequences.** The 200-300 ms latency budget is measured on 16 kHz
frames. Any future 48 kHz or wideband extension is a new experiment with
new proxies, not a config change. If a judge is added that prefers a
different rate, the resample must be explicit, logged per trial and named
in the results table.

**Note on the WHAM! conversion in `docs/data/data-setup.md` step 1e:** WHAM!
noise is *already* 16 kHz, so `-ar 16000` there is a no-op guard. The 4×
storage reduction comes entirely from stereo → mono and 32-bit float →
16-bit. Do not defend it as a sample-rate saving.

**16-bit is a deliberate, mildly lossy choice.** 32-bit float sources are
quantised to 16-bit on ingest. 16-bit gives ~96 dB of dynamic range, far
more than these signals use at the SNRs we mix at, and it matches what the
judge accepts anyway. Chosen, not defaulted to.

Sources: <https://ai.google.dev/gemini-api/docs/live-api/capabilities>,
<https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs>,
<http://wham.whisper.ai/>, and `docs/data/definitions.md:60`.

## 2026-08-10 — Data licensing: the constructed set inherits CC BY-NC
Our three corpora do not share a licence:

| Corpus | Licence | Commercial use |
|---|---|---|
| LibriSpeech | CC BY 4.0 | permitted |
| AMI | CC BY 4.0 | permitted |
| **WHAM! noise** | **CC BY-NC 4.0** | **prohibited** |

Because every constructed mixture contains WHAM! noise, the constructed
train/val/eval sets are **derivative works of an NC-licensed corpus**. The
most restrictive input governs the output.

**The AMI secondary eval set is exempt.** It contains no WHAM! material —
it is real meeting audio only — so AMI-derived trials stay CC BY 4.0 and
may be released permissively. The NC blanket covers the constructed leg,
not the AMI leg. These are two separate release decisions.

Decision: accept the NC constraint rather than substitute the noise corpus.
It is the right trade for a masters project — WHAM! is the field-standard
noise set, real recorded (not synthetic), and swapping it would cost
comparability for a freedom we do not need.

**What this binds us to:**
- We may **not** redistribute generated mixture audio under a permissive
  licence, sell it, or use it in any commercial product.
- Publishing the "public trial split" required by
  `docs/data/metric-definitions.md:198-200` must be **CC BY-NC 4.0**, with
  attribution to all three upstream corpora.
- Safer alternative for release: publish the **generation code, manifests
  and seeds** rather than the audio, so users regenerate locally from
  corpora they obtain themselves. This sidesteps redistribution entirely
  and is better for reproducibility. Prefer this; treat audio release as a
  fallback.
- **Model weights are a separate question and are not clearly restricted**
  by NC. A trained model is generally not considered a derivative work of
  its training data, but this is unsettled and jurisdiction-dependent. If
  weights are ever released, note the training data's NC status and, if it
  matters, get an actual opinion — I am not a lawyer and this is not legal
  advice.

Obligations, per-corpus attribution text and the required thesis wording
are in `docs/data/data-licences.md`.

## 2026-08-11 — Target-absent trials: the interferer is the level anchor

`sir_db` is blank when `target_absent = 1` — with no target there is no
target-to-interferer ratio. But `snr_db` and `target_loudness_lufs` are still
recorded, and both are defined relative to that missing target, so nothing in
the manifest fixes the interferer or the noise gain.

Decision: **in target-absent trials the interferer becomes the anchor.**
Normalise the interferer to `target_loudness_lufs`, then place the noise at
`snr_db` relative to the interferer. `sir_db` stays blank and unused.

Why: both recorded columns keep their meaning, absent trials sit at comparable
volume to present ones, and no manifest changes are needed. The alternative — a
fixed absolute level for absent trials — would make the condition identifiable
by loudness alone, which the model could exploit instead of listening to the
enrollment.

## 2026-08-11 — Target-absent trials: interferer activity must match present trials

Found while exploring `smoke_train`: absent trials had `interferer_activity` of
0.75-0.85 in every case, while present trials spanned 0.29-0.82. A gap of 1.4
standard deviations, with barely overlapping ranges.

Cause: the absent branch set interferer activity to `target_activity_ratio`
(a scalar, so exactly 0.75, then filled to `activity_tolerance`). Present trials
draw it from the §2.2 feasibility band instead. Two different distributions.

Why it matters: the condition became detectable from activity alone. A model
could learn "one voice talking near-continuously ⇒ emit silence" and score 35%
of the training set without ever consulting the enrollment — the shortcut these
trials exist to prevent.

Decision: **absent trials draw interferer activity from the same distribution a
present trial would have produced.** `build_manifest.py` now takes a shadow
overlap draw and a shadow target activity, then applies the same
`uniform(overlap, 1 - t_act + overlap)` band the present branch uses.

Consequence: low activity values are harder to satisfy from long LibriSpeech
utterances, so `n_failed` may rise. Check it on every regenerated manifest.

General lesson, worth applying to any future condition flag: **a condition the
model is supposed to infer by listening must not be inferable from the
manifest's summary statistics.** Compare every column across the two groups.

## 2026-08-11 — Noise is shorter than the mixtures: wrap around

WHAM! clips are shorter than the mixtures. In the `tr` split they run 3.4-47.7 s
with a median of 10.0 s, against mixtures of 15-20 s. Only 86 of 20000 clips
(0.43%) reach 20 s, and in `smoke_train` all 50 trials need more noise than
their clip holds.

Decision: **the noise bed wraps.** `noise_offset_s` is a phase into a looped
stream, not a slice index. The renderer reads `mixture_length_s` seconds from
that offset, returning to the start of the clip when it runs out.

Rejected: using only clips long enough (discards 99.6% of the library and biases
toward whichever scenes were recorded for longer), and shortening the mixtures
(breaks comparability with REAL-TSE's ~17-18 s).

Cost: a faint repeat at each seam. Measured on `smoke_train`, trials cross the
clip end **1.9 times on average** (1 seam in 12 trials, 2 in 33, 3 in 5) — more
than the "wraps once" a median 10 s clip in a 17.5 s window suggests, because
`noise_offset_s` is uniform over the whole clip and so usually starts the read
partway through. Record it as a known artefact of the constructed set; the AMI
leg has no such artefact.

If 1.9 seams proves audible in the §11 listening pass, the cheap mitigation is to
draw `noise_offset_s` over `[0, max(0, duration - mixture_length_s)]` where the
clip allows it, which removes one seam without touching the wrap logic.

Renderer note: a naive `noise[offset : offset + length]` silently yields short or
zero-padded noise. The wrap is mandatory, not an optimisation.

Was `decisions-pending.md` A2.

## 2026-08-11 — Enrollment must come from a different book, not just a different chapter

The interferer is required to read a **different book** from the target, so that
shared vocabulary cannot be mistaken for contamination by the metric. The
enrollment clip was only required to come from a different **chapter**.

In LibriSpeech a speaker usually reads consecutive chapters of one book, so a
different chapter of the same book shares narrative, characters, proper nouns and
register. Measured on `smoke_train`: **11 of 36 present trials** had enrollment
from the same book as the mixture.

Decision: **the enrollment guard now matches the interferer guard — different
book.** `build_manifest.py` filters candidates on `book[chapter]` rather than on
`chapter`, and the trial assertion was tightened to match.

Why: the same argument justifies both guards. One being weaker than the other is
not defensible in writing, and the shortcut it leaves open — matching enrollment
to target on topic instead of on voice — makes the task easier than the thesis
claims it is.

Cost: a smaller candidate pool, so `n_failed` rises. Speakers who read only one
book now yield no valid present trial. Check `n_failed` and the per-speaker trial
counts on the next build; if a material number of speakers drop out entirely,
revisit as a book-preferred-with-chapter-fallback rule.

Was `decisions-pending.md` B8.

## 2026-08-11 — val carries target-absent trials; the eval splits do not

`val` and `smoke_val` were `target_absent_fraction: 0.0`, so validation could not
exercise the push-to-silence half of the split loss. That term would have been
trained on 35% of training batches and measured on none — a collapse in it would
not have shown up in any val curve.

Decision: **`val` and `smoke_val` move to 0.35, matching train. The eval splits
stay at 0.0.**

Why the asymmetry: validation and evaluation are different jobs. Val exists to
tell you whether training is working, so it must cover every loss term. Eval
reports results, and whether absent trials belong there is still open — see
`decisions-pending.md` B4, which this decision deliberately does not pre-empt.



## 2026-08-12 — Enrollment carries no room

The enrollment clip is the few seconds of the target speaking alone that tells
the model which voice to follow. It could be rendered dry, or convolved with the
same room as the mixture.

Decision (supervisor, 2026-08-12): **no room on the enrollment at all. It stays a
clean sample, so the model knows what the target sounds like clean.**

Why: if the enrollment carried the same echo signature as the mixture, the model
could match on room acoustics instead of on voice, and the score would stop
measuring what it claims to measure. This is the same argument that makes
`shared_room` non-negotiable, applied to the conditioning path — and the same
argument behind the different-book guard (2026-08-11): every route to identifying
the target other than the voice itself has to be closed.

Cost: a channel mismatch between enrollment and mixture, which is realistic
rather than a defect — a stored voice profile genuinely is captured elsewhere.
`enrollment_eq_augmentation` already trains the conditioning path to tolerate it.

No code change: the renderer is unwritten, and LibriSpeech enrollment audio is
already dry, so this constrains what the renderer must *not* do.

Was `decisions-pending.md` A4.

## 2026-08-12 — Enrollment length fixed at 5 s, and stays configurable

Longer enrollment makes the target easier to identify, so length is a confound if
it varies per trial.

Decision (supervisor, 2026-08-12): **fixed at exactly 5 s everywhere, held as a
parameter that can be changed rather than a hardcoded constant.**

Why: 5 s is the minimum `docs/data/metric-definitions.md` allows, so the headline
result is a worst case rather than a flattering one. Keeping it configurable
leaves the door open to varying it later as a deliberate experiment, which is the
only defensible way to vary it — one value per run, recorded, not sampled per
trial.

No code change: `experiments/configs/generator.yaml` already has
`enrollment_length_s: 5.0` under `defaults`, and `draw()` returns a scalar
unchanged while sampling uniformly from a `[lo, hi]` list. Setting it to a range
is therefore a one-line config edit whenever that experiment is wanted.

Was `decisions-pending.md` B3.

## 2026-08-12 — Levels are measured as BS.1770 integrated loudness

`sir_db`, `snr_db` and `target_loudness_lufs` all state a level, and something has
to define what "level" means before the renderer can apply them.

Decision: **BS.1770-4 integrated loudness, via `pyloudnorm`.**

Recorded for the trail: RMS was chosen first in the same 2026-08-12 meeting and
reversed the same day, before anything was implemented. The reversal is not a
preference — RMS turned out to be the *more* expensive option:

1. **It forces a gating rule.** BS.1770 gates out quiet passages by
   specification. Plain RMS averages over everything handed to it, so a target
   talking for a fifth of the window measures far quieter than one talking for
   most of it at the identical speaking level. Measured at 16 kHz on identical
   speech, once continuous and once padded to 20 s:

   | | BS.1770 | plain RMS |
   |---|---|---|
   | 3 s continuous | −23.35 LUFS | −26.01 dBFS |
   | same 3 s, 14 s of silence around it | −23.77 LUFS | −33.55 dBFS |

   0.4 dB apart against 7.5 dB apart. Under RMS, `sir_db` would mean something
   different in every trial and level would correlate with speech activity — the
   same class of shortcut §7 of the manifest notebook exists to find, and it
   would land on the leak scoreboard as a new entry. Gating RMS by hand fixes it
   but only by rebuilding a cruder version of what the standard already defines.
   This gets worse if `decisions-pending.md` B9 is accepted, since variable
   `target_activity_ratio` widens the activity spread that drives it.

2. **It would have made `target_loudness_lufs` a misnomer.** LUFS is the BS.1770
   unit. The column, the config key and the range `[-33.0, -25.0]` all came from
   LibriMix's `pyloudnorm` constants, so under RMS the name is wrong and the
   numbers denote a different physical level — a rename across the config,
   `build_manifest.py`, six manifest CSVs and the notebook, plus recalibration.
   Keeping BS.1770 costs none of that.

3. **Both reference implementations use loudness.** LibriMix and WHAMR! both set
   levels this way. Our numbers are still not comparable to theirs — different
   data, different protocol — but the *method* needs no separate defence.

The one advantage RMS had, that it is always defined where gated loudness returns
`-inf` on a silent stem, does not apply: the 2026-08-11 level-anchor decision
makes the interferer the anchor in target-absent trials, so a silent target stem
is never measured.

Renderer constraints this imposes, both verified against `pyloudnorm` 0.2.0 at
16 kHz:

- **A stem shorter than 400 ms raises `ValueError`** (BS.1770 block size). Guard
  with a minimum target-speech duration in the generator — needed anyway once B9
  lets activity vary.
- **A fully silent stem returns `-inf`.** Reachable only through a bug given the
  anchor rule, so assert rather than handle.

Environment: `pyloudnorm` 0.2.0 is installed in `tse_venv` (Python 3.12.3,
numpy 2.5.2, scipy 1.18.0). Pin it — there is still no requirements file in the
repo, which is a reproducibility gap in its own right.

Was `decisions-pending.md` A3.

## 2026-08-13 — Reference signal: the target's reverberant image ("what the mic heard")

Decision (GB, 2026-08-13, pending supervisor sign-off): **`target_reference` =
full reverberant. The reference is the target speaker convolved with their own
room impulse response — exactly what the microphone received from that person.
The interferer and the noise bed are excluded, so separation and denoising remain
in the task; dereverberation does not.**

Dereverberation becomes an **ablation only, if time allows** (see below).

### Why

**1. Removing echo cannot be done inside the latency budget.** `t60_s` reaches
0.6 s against a 200–300 ms causal window; 66.8 % of trials have a tail longer than
the model is allowed to see (`decisions-pending.md` B11). A model cannot cancel
what it has not yet heard, so the demand trains a room prior, not cancellation.

**2. Trying anyway costs more than leaving it.** Enhancement error splits into
what was left behind and what was destroyed in the attempt, and the destruction —
"artifacts" — is the dominant cause of ASR degradation, not the residue
(Iwamoto et al., 2022; Ochiai et al., 2024). Sato et al. (2021) push this further:
processing artifacts can make a separated signal recognised *worse* than the
unprocessed mixture, so they gate on whether to separate at all. Since this
project's whole claim is that live models mishear artefacts, deliberately choosing
the task with the smaller artefact surface is the consistent choice.

**3. Early reflections should not be removed under any option.** Reflections
arriving within ~50 ms are as useful to intelligibility as the direct sound itself
(Bradley & Sato, 2003). So the real choice was only ever "remove the late tail or
not", and (1) says we cannot.

### What this concedes, stated plainly

Late reverberation genuinely does cost recognition accuracy — the REVERB challenge
line of work is unambiguous (Kinoshita et al., 2016), and spectral smearing also
degrades the separation itself, which is the job we *are* keeping (Maciejewski
et al., 2020). This decision does not claim reverb is harmless. It claims removal
is unaffordable online and that attempting it costs more than it returns. **The
write-up must state this as a known limitation, not omit it.**

### Why the noise bed stays out of the reference

Considered and rejected: leaving the noise in the reference too (remove only the
competing voice). `snr_db` runs down to 0.0, where the bed is as loud as the target
and masks words even though it contains none — `noise_speech_rejection`
(implemented 2026-08-16) screens intelligible speech out of it, though it screens
rather than guarantees. Non-intelligible noise cannot
make the judge hear *wrong* words but can stop it hearing the right ones, and
denoising needs no lookahead, so it is affordable online in a way dereverberation
is not.

### Consequences

- **A5 (tail padding) is now mandatory**, not optional: the reference contains the
  tail, so the window must extend past the last speech by at least the room's decay
  or reference and mixture stop matching.
- **B11 largely dissolves.** The model is no longer asked to suppress a tail it
  cannot hear. What survives is the weaker question of whether reverb limits
  separation, which the stratified reporting in B13 (T60 above/below budget) covers.
- **Ablation, if time allows:** render the direct+early stem as a second reference
  and train the same architecture against it. Report both. That measures the
  artefact/residue trade rather than assuming it. Cheap to render, one extra
  training run.
- Renderer must record the RIR per trial so any reference variant is reproducible
  without re-drawing rooms.

### References

- Bradley, J. S. & Sato, H. (2003). On the importance of early reflections for
  speech in rooms. *JASA* 113(6), 3233–3244.
- Iwamoto, K., Ochiai, T., Delcroix, M., et al. (2022). How bad are artifacts?:
  Analyzing the impact of speech enhancement errors on ASR. *Interspeech 2022*,
  5418–5422.
- Kinoshita, K., Delcroix, M., Gannot, S., et al. (2016). A summary of the REVERB
  challenge. *EURASIP J. Adv. Signal Process.* 2016:7.
- Maciejewski, M., Wichern, G., McQuinn, E. & Le Roux, J. (2020). WHAMR!: Noisy and
  reverberant single-channel speech separation. *ICASSP 2020*.
- Ochiai, T., Iwamoto, K., Delcroix, M., et al. (2024). Rethinking processing
  distortions: Disentangling the impact of speech enhancement errors on speech
  recognition performance. arXiv:2404.14860.
- Sato, H., Ochiai, T., Delcroix, M., et al. (2021). Should we always separate?:
  Switching between enhanced and observed signals for overlapping speech
  recognition. *Interspeech 2021*. arXiv:2106.00949.

Note on borrowing: WHAMR! chose direct-path-only references
(`wham_room.py:47-60`); we deliberately diverge, because WHAMR! is an offline
separation benchmark scored on signal quality and this is a causal streaming system
scored on downstream content fidelity. Not comparable to WHAMR! numbers.

Was `decisions-pending.md` A1.

## 2026-08-13 — B1 closed: overlap range is a dial setting, not a decision

`overlap_ratio` stays `[0.2, 0.7]` for now. It is not pinned by a decision — it is
one of the 14 parameters ranked in `docs/data/difficulty-dial.md` and moves on
request once B12 lands.

Two constraints on moving it, recorded so they are not rediscovered:

1. **The 0.7 ceiling is deliberate, not accidental.** It matches REAL-TSE's ~0.5
   average overlap, our anchor benchmark. An ordinary meeting overlaps 0.10–0.15,
   so the set is already harder than daily conversation *by design*. Narrowing the
   ceiling is a divergence from the anchor and needs its own decision entry — which
   is why the difficulty dial lists it as the narrowing to do **last**.
2. **The 0.2 floor is a separate problem and does not belong to B1.** No present
   trial can have zero overlap, which is what makes silent-target trials detectable
   without listening (AUC 1.000). That is B9, and it is fixed by
   `target_only_fraction` and variable `target_activity_ratio`, not by moving this
   range.

Was `decisions-pending.md` B1.

## 2026-08-13 — B12 architecture: two regimes, a sampler layer, no relational constraints

Decision (GB, 2026-08-13). B12 asked for four things; this settles the shape of all
four. **Not yet implemented** — see the PR order at the end.

### 1. Two named regimes, sampled per trial, recorded as a `regime` column

**One regime is drawn per trial, then every parameter for that trial comes from that
regime's bands.** Weights `base: 0.6`, `hard: 0.4`.

Why per trial and not per parameter: with six parameters each independently 50/50,
only ~1.6 % of trials would be base-case on all six, so the base condition would
barely exist. Drawing the regime once makes it exactly 60 %.

`hard` inherits `defaults` wholesale, so **`base` is a sub-range of `hard`** and the
column records *provenance, not difficulty*. Measured on the proposed bands, a
`hard` draw lands inside the base band 50–71 % of the time depending on the
parameter, and inside all five simultaneously 7.1 % of the time. Two consequences:

- **`regime` must not be a reporting stratum (B13).** Base vs hard would compare
  overlapping populations and understate the gap. Report on value bands — SIR band,
  overlap band, T60 above/below budget — as B13 already specifies.
- **Filtering `regime == base` is sufficient.** It keeps 60.0 % of rows against
  62.8 % for a full value-based filter: 2.8 points apart, not worth the complexity.

`train` carries both regimes mixed. If the data proves too hard, filter to
`regime == base` — **no rebuild required**, which is the main thing this buys.

`regime` is renderer metadata. The model receives the mixture and the enrollment and
nothing else, so the column cannot be a shortcut. What it does require is that the
§7 leak audit be re-run **within** each regime, since a shortcut inside `base` can be
diluted to invisibility in a pooled AUC.

### 2. Regime-scoped parameters: 6. Everything else global

| Regime-scoped | Global |
|---|---|
| `sir_db`, `snr_db`, `overlap_ratio`, `t60_s`, `source_distance_m`, `target_activity_ratio` | `same_gender_fraction`, `enrollment_length_s`, `enrollment_eq_prob`, `mixture_length_s`, `target_loudness_lufs`, room dimensions, mic/source heights, `clip_ceiling` |

The global ones are either deliberate worst cases or the experimental variables B13
stratifies on independently; they should not move with difficulty.

`target_activity_ratio` is regime-scoped in the schema but **identical in both
regimes until B9 lands**, because what varying it means is B9's decision.

Band values: `difficulty-dial.md` §2, "Narrow to" column. Provisional by design — a
rebuild is 58 s.

### 3. Distributions: `fixed`, `uniform`, `truncnorm`. Not beta

Beta was proposed in B12 and is dropped: no parameter has a stated need for
skew-within-bounds, and an unused shape is one more thing to defend.

**Shapes are not introduced in the same PR as regimes.** Every parameter starts
`uniform`, i.e. behaviourally identical to today, so the regime PR's diff is exactly
the new column plus the band values and can be verified against the current manifest.
`truncnorm` is added later, one parameter at a time, each with a recorded reason.

`truncnorm` must be implemented by inverse-CDF over a single `rng.uniform` draw
(`scipy.stats.norm.ppf`), **not** `scipy.stats.truncnorm.rvs(random_state=rng)`. Under
`rvs`, a scipy upgrade can change how many uniforms are consumed per sample, which
silently changes what `seed: 42` means and makes the logged seed worthless.

### 4. Relational constraints: deferred entirely

B12 point 3 is not implemented. Nothing in the data forces it, so it can be added
later without a second rebuild.

Recorded so the reasoning is not lost: B12's own example, *the target is always
closer to the mic than the interferer*, **must not be implemented as a hard rule.**
It would make proximity a perfect predictor of target identity, so the model could
learn "nearer voice = target" and never consult the enrollment — the same failure
class as B9's overlap leak and B10's speaker leak, and it would land on the §7
scoreboard as a third entry. If it is ever wanted, it must be probabilistic
(`p_target_closer: 0.6`) so it is a prior rather than a rule.

More generally: constraints enforced by rejection bend distributions, which §4 and §7
show is precisely what creates shortcuts. Prefer constraints satisfiable by
construction.

### Implementation order

1. **PR1** — `src/data/sampling.py`: `draw(rng, spec)` and `resolve(cfg, regime)`,
   with unit tests. No behaviour change, nothing wired in. Tested on its own because
   every number in every later experiment comes out of it.
2. **PR2** — wire into `build_manifest.py`, add the `regime` column, raise the
   `t60_s` floor from 0.15 to 0.25 (`difficulty-dial.md` §1: T60 0.15 in a room this
   size implies anechoic-grade wall absorption).
3. **PR3** — B9 + B10 + B4, then re-run the §7 leak scoreboard per regime.

Doing B12 before the B9/B10/B4 rebuild avoids rebuilding twice. Timing measured
2026-08-13: a full 20,000-trial `train` rebuild is **58 s** and reproduces the
committed manifest byte-identically, so all of this is cheap while no audio exists.

Was `decisions-pending.md` B12.

## 2026-08-13 — A5: the mixture keeps running after the last word

Echo continues after speech stops. Cutting the file at the last word cuts the echo
with it, so the reference signal and the mixture stop matching.

**Decision: pad the rendered output by `t60_s` beyond the nominal
`mixture_length_s`.** Transcripts are unaffected — no speech occurs in the tail.

Why padding rather than placing speech earlier: forbidding late onsets would make
speech placement depend on the room's reverberation time, so *when* people talk
would correlate with *how echoey the room is* — a new confound of exactly the kind
§7 of the manifest notebook exists to catch. Padding introduces no such link.

Now mandatory rather than optional: the A1 decision (2026-08-13) puts the reverberant
tail inside the reference signal, so a truncated tail is a mismatch, not a cosmetic
issue.

Was `decisions-pending.md` A5.

## 2026-08-13 — A6: clipping is fixed by rescaling everything together

**Decision: if any sample exceeds 0.95 of full scale after summing, scale the mixture
and every stem by the same factor.**

Why the same factor: quietening only the offending track would change the loudness
differences the trial was constructed to have, silently altering its SIR and SNR.
A common gain preserves both.

Already specified in `data-construction-parameters.md`; recorded here so it is a
decision rather than an undocumented default.

Was `decisions-pending.md` A6.

## 2026-08-13 — B2: overlap is measured from detected speech, not file boundaries

A read sentence contains pauses, so measuring overlap from where recordings start and
stop overstates how much genuine talking coincides.

**Decision: detect where speech actually is first, and measure overlap from that.**
One pass over the corpus, cached alongside the existing utterance index.

Why: it costs nothing after the first run and it makes the overlap figure defensible
and comparable to REAL-TSE's. The current inflated number cannot be quoted next to
the anchor benchmark's.

The specific detector is an implementation choice, to be named in the PR that adds it.

Was `decisions-pending.md` B2.

## 2026-08-13 — B4: target-absent trials in eval, scored on their own row

**Decision: the eval splits carry the same target-absent fraction as `train`, and
those trials are excluded from the main content score and reported separately as how
often the system invents speech that was not there.**

Why excluded from the main score: there is no correct text to compare against when
nobody speaks, so any content-fidelity figure computed on them would be meaningless.

Why the same fraction as train: it keeps the strata comparable between what the model
saw and what it is judged on.

Why a separate row rather than folded in: inventing speech and mis-hearing speech are
different failures with different fixes. One number hiding both is unusable, and
B13 (2026-08-13) forbids it.

Supersedes the eval splits' `target_absent_fraction: 0.0`. The exact fraction follows
B9, which sets it for `train`.

Was `decisions-pending.md` B4.

## 2026-08-13 — B5: text is normalised with Whisper's English normaliser

The corpus transcripts are capitals with no punctuation. A live model replies in
ordinary prose with digits. Compared unchanged, every system looks worse than it is.

**Decision: adopt Whisper's `EnglishTextNormalizer` unchanged, applied identically to
the reference and to the system output.** Cite Radford et al. (2023).

Why an off-the-shelf normaliser: it is published, widely used and not ours, so it
cannot be suspected of having been tuned to flatter our numbers. Writing our own
would mean defending every individual choice about case, punctuation, digits and
contractions.

Two rules that come with it:

- **Frozen before the first judge result is recorded.** Changing it afterwards
  invalidates every comparison.
- **Never adjusted per system.** Both sides of every comparison get the identical
  function.

Note this does not make Whisper part of the pipeline. The normaliser is a text
function, so it touches neither the training loop nor the judge, and the rules in
`CLAUDE.md` about model families are unaffected.

Was `decisions-pending.md` B5.

## 2026-08-13 — B13: results are reported condition by condition, never combined

A single overall score averages over trials that are nothing like each other — some
with the target as the dominant voice in a dry room, some with a louder interferer
talking over it in a reverberant one. The average says nothing about which.

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

**Minimum 100 trials per bucket.** Each trial counts toward every condition at once,
so a two-way split of *N* scored trials gives *N*/2 per bucket — 200 scored trials is
the floor.

Why no combinations: six conditions crossed together is 64 buckets, roughly 8 trials
each at 500 trials. Unreadable, and reaching 100 per bucket would need ~6,400 trials
and about 13× the judge budget. The primary/secondary order says what to protect if
the budget turns out tight.

The interruption condition named in the original entry is deferred: nothing in the
data marks an interruption today, and defining one needs the turn-taking trials B9
introduces.

Was `decisions-pending.md` B13.

## 2026-08-13 — Eval splits draw independently, with no regimes

Amends the B12 entry above, prompted by B13.

**Decision: the eval splits omit the `regimes:` block entirely and draw every
parameter independently from the wide ranges. `regime` records `none`. Training keeps
regimes as decided.**

Why: under regimes, a `hard` trial draws a wide SIR *and* a wide overlap together, so
the two become correlated — `P(high overlap)` is 0.20 overall but 0.50 given a louder
interferer, a 2.5× enrichment. B13's per-condition tables would then partly describe
the same trials, reporting one effect twice as two findings. That is the precise
failure the stratified reporting exists to avoid.

Regimes exist so training data can be filtered when it proves too hard. Eval is never
filtered, so it gains nothing from them and loses independent conditions.

## 2026-08-13 — B6: 500 trials generated, 200 the minimum scored

**Decision: keep 500 trials per eval split. Score at least 200; grow as the judge
budget allows.**

Why this works without a budget figure: rows are generated in a fixed order, so the
first 200 are contained in the first 300. Scoring more later extends the set instead
of replacing it, and the numbers stay comparable as the budget grows.

The 200 floor comes from B13: 100 trials per bucket across a two-way split.

Was `decisions-pending.md` B6.

## 2026-08-13 — B7: training mixtures are not resampled each pass

Rebuilding mixtures with fresh loudness and room draws every epoch gives the model
more variety, but the run can then no longer be reproduced from the manifest alone.

**Decision: off for the main run. Available as a config switch.**

Why: reproducibility of the headline result is worth more than the extra variety, and
the variety can be recovered as a reported ablation if it turns out to matter.

Was `decisions-pending.md` B7.

## 2026-08-13 — B11: latency is reported as a decay curve, not a threshold

Two thirds of trials have reverberation lasting longer than the streaming window lets
the model see.

**Decision: never cap `t60_s` to make the task fit. Score the same model at 100, 200,
300, 400 and 500 ms of allowed latency and report how performance decays.**

Why not cap it: capping would make the data less like a real room, which is the one
thing constructed data has to get right, and latency is a secondary objective in the
specification. Changing the metric costs several evaluation passes; changing the data
costs the realism the whole set exists to provide.

Largely defused by A1 (2026-08-13), which stopped asking the model to remove the
tail at all. What remains is whether reverberation limits *separation*, which the
T60 condition in B13 reports directly.

Was `decisions-pending.md` B11.

## 2026-08-13 — B9: a target speaking uninterrupted becomes a quarter of the data

### The problem

35 % of training trials have the target never speaking. They exist to teach one
behaviour: **stay quiet when the enrolled voice is not there** — a system that invents
speech is worse than useless to a live model.

But in every one of those trials only one person talks, and in every trial where the
target *does* speak, two voices overlap at some point. So *"did two voices ever
overlap?"* answers *"is the target absent?"* on **every trial in the set** (AUC 1.000
on `train`, 1.000 on `val`), without ever consulting the enrollment. The behaviour is
never actually trained, and nothing in any training curve would reveal it.

It also inverts in deployment. Someone talking with nobody interrupting looks exactly
like "target absent", so a model that learned *no overlap → emit silence* would go
quiet on the most common real condition there is. That condition currently appears in
**zero** trials.

### The decision

| Trial type | Before | After |
|---|---|---|
| Both speaking | 65 % | **50 %** |
| Target absent | 35 % | **25 %** — of which 5 % is noise bed only, neither speaker |
| Target only, nobody interrupting | 0 % | **25 %** |

And **`target_activity_ratio` varies** instead of being pinned at 0.75:
base `[0.45, 0.85]`, hard `[0.15, 0.85]` (regime-scoped, per B12).

### Why these numbers

**The 25/25 split is symmetric because the two failure modes are weighed equally.**
Inventing speech and dropping speech are both content-fidelity failures, and neither
is privileged. Making the two zero-overlap cases equally frequent puts
`P(target absent | no overlap)` at exactly **0.50** — a coin flip, so the shortcut
carries no information at all. Any other ratio leaves it partly predictive: the
originally proposed 30/15 would have left it 0.67, still right two thirds of the time.

**The two changes compound rather than merely stack.** Once the target's talkativeness
varies, some both-speaking trials become genuine turn-taking with near-zero overlap.
Those sit on the *present* side of the zero-overlap pool, pushing the figure below
0.50 and leaving no residual signal to exploit.

**Widening `overlap_ratio` instead would not work.** Overlap is floored by
`overlap >= target_activity + interferer_activity - 1`. With the target pinned at
0.75–0.85, zero overlap requires an interferer at ≤ 0.20 activity — roughly 3.5 s
against a current mean of 10.0 s — and that block must fit inside a gap in the
target's timeline, where the largest median gap is 2.68 s. The result would be a
narrow corner of near-silent-interferer trials, which is a *new* giveaway, plus heavy
rejection. **The binding constraint is `target_activity_ratio`, not `overlap_ratio`.**

### What else this fixes

- **The residual `interferer_activity` leak** (§7.3, AUC 0.614) at source. It exists
  because present trials get rejected when a contiguous interferer block cannot reach
  the requested overlap, while target-absent trials never face that test. A less
  talkative target widens the achievable overlap range, so the tolerance check stops
  firing asymmetrically.
- **The realism gap.** The target currently speaks in a mean of 1.1 utterances per
  trial — one unbroken ~14 s monologue filling 80 % of the window. That is not
  conversation.
- **B4's open number.** Eval splits take `target_absent_fraction: 0.25`, matching
  train, as B4 requires.

### Cost

The larger of the two changes touches `pick_run`, `best_onset` and the overlap bounds
together, plus a new branch in `build_trial` mirroring the target-absent branch. A
full manifest rebuild is required. §2, §7 and the health checks of the manifest
notebook assume the current timing model and will need revising; §3–§6 will not.

The §7 leak scoreboard must be re-run **per regime as well as pooled** (B12), and must
show the AUCs dropped.

Was `decisions-pending.md` B9.

## 2026-08-13 — B10: enrollment falls back through three tiers, recorded per trial

### The problem

B8 (2026-08-11) requires the enrollment clip to come from a different **book** than
the mixture. A present trial therefore needs a target who read two or more books —
but in LibriSpeech most readers read one. Of the 1,172 speakers in `train`, only
**467 (39.8 %)** own two books, so those 467 were the only speakers that could ever
be a present target. The other 705 appeared exclusively in target-absent trials.

Scoring each trial by how often its target speaker is silent elsewhere gives
**AUC 0.795** on `train` and 0.756 on `val`. Interferer identity gives 0.502, so the
leak is specifically about the voice the model is told to listen for: the silence
decision could be made by recognising the speaker instead of by listening.

**This is not a reversal of B8.** B8's own cost note specified the remedy and its
trigger: *"if a material number of speakers drop out entirely, revisit as a
book-preferred-with-chapter-fallback rule."* 60.2 % is material.

### The decision

**Enrollment falls back through three tiers, and which tier fired is recorded per
trial in an `enrollment_guard` column.**

| Tier | Rule | Speakers in `train` | |
|---|---|---|---|
| `book` | Different book from the mixture | 467 | 39.8 % |
| `chapter` | Different chapter, same book | 469 | 40.0 % |
| `utterance` | Same chapter, utterances not used in the mixture | 236 | 20.1 % |

Book-preferred alone would have left the 236 single-chapter speakers with no valid
enrollment at all — no different book *and* no different chapter — recreating the same
absent-only leak at reduced strength.

**Within the `utterance` tier, pick the enrollment from the utterances furthest in
index from those used in the mixture.** LibriSpeech numbers utterances sequentially
within a chapter, so maximising that distance costs nothing and minimises how much
narrative the enrollment shares with the mixture.

**Assert that no utterance appears in both the enrollment and the mixture.** In the
first two tiers this is automatic; in the third it is the only thing separating them.

### Why keep all 1,172 speakers

Two alternatives were rejected:

- **Dropping the 236.** Clean and leak-free, but discards a fifth of the speaker pool
  for a speaker-conditioned model.
- **Using them as interferers only.** This mirrors the leak it is meant to close: a
  voice that is *never* the target teaches "suppress this speaker" without consulting
  the enrollment. Interferer identity currently sits at AUC 0.502 — no leak — and this
  would manufacture one.

Keeping every speaker and **recording the tier** means the content-leak cost is
measurable rather than assumed. If `utterance`-tier trials score conspicuously better
than `book`-tier ones, that is the leak, quantified. B8 called same-book enrollment
indefensible as a *default*; as a recorded, measured fallback for speakers who leave
no alternative, it is defensible.

### Eval splits are redrawn

The three tiers are unevenly distributed across the eval pools: `eval_public` has 8 of
20 speakers in the weakest tier against `eval_private`'s 3 of 20. That would make
`eval_public` systematically the easier set — a confound between the two eval sets
before any system is measured.

**Decision: `make_splits.py` redraws the eval pools so the guard-tier composition
matches between `eval_public` and `eval_private`.** Speaker-disjointness from `train`
is unchanged. This invalidates the current manifests, but PR3 rebuilds them anyway, so
the timing is free.

### Consequences

- **B13 gains a three-level condition.** The "one-book vs two-book target speaker"
  condition in B13 becomes **guard tier: `book` / `chapter` / `utterance`**, which is
  what actually varies.
- New column `enrollment_guard`; `make_splits.py` and `splits.yaml` both change.
- Folds into the PR3 rebuild.

Was `decisions-pending.md` B10.

## 2026-08-14 — B12 PR2: sampler wired in; two bands deliberately not narrowed

Implements the 2026-08-13 B12 architecture. Code only — the manifests on disk are
still the old schema and are rebuilt in PR3.

**Wiring.** `build_manifest.py` draws the regime from a fifth RNG stream
(`rngs(trial_id, 5)`). `SeedSequence.spawn(5)` returns the same first four children
as `spawn(4)`, so the new stream moves no existing draw, and the regime can never
shift the values of the parameters it selects. A split opts out with `regimes: null`,
which the eval splits do; `draw_regime` then consumes nothing and `regime` records
`none`.

**Acceptance test passed.** With the config unchanged, all six splits — 21,270 rows —
are byte-identical to the previous manifests once the new `regime` column is ignored.
`scripts/check_manifest_parity.py`, run 2026-08-14.

**Two bands from `difficulty-dial.md` §2 were NOT applied to `base`:**

- **`overlap_ratio` keeps `[0.2, 0.7]`.** §3 ranks this narrowing last and states it
  needs supervisor agreement and a decision entry, not a config edit, because the 0.7
  ceiling is deliberately matched to REAL-TSE. Applying it would have been a research
  decision smuggled in as an implementation detail. Confirmed in the rebuild: mean
  `overlap_achieved` is 0.294 in `base` and 0.293 in `hard`.
- **`target_activity_ratio` stays fixed at 0.75.** Regime-scoped in the schema,
  identical in both regimes until B9 lands, exactly as the B12 entry specifies.

So `base` narrows four parameters: `sir_db` `[0, 12]`, `snr_db` `[8, 20]`, `t60_s`
`[0.25, 0.5]`, `source_distance_m` `[0.66, 1.4]`.

**Conflict recorded, not resolved.** `difficulty-dial.md` §2 proposes a `base`
`overlap_ratio` floor of 0.1 against a global floor of 0.2, so `base` would not be a
sub-range of `hard` and the provenance argument above would not hold for that one
parameter. The 0.2 floor is itself the B9 bug. Whichever document is wrong, it is
B9's call in PR3. `resolve()` therefore does not enforce containment; the exception
is pinned by a named unit test instead.

**Measured after the config change** (`train`, 19,569 rows, unchanged count):
`t60_s` floor 0.15 → 0.25 as intended; regime mix 0.60 / 0.40 exactly. In the eval
splits only the room columns moved, because `t60_s` is drawn from the `room` stream —
speech, levels, enrollment and noise are bit-identical to before.

## 2026-08-14 — PR3: B9, B10 and B4 implemented; both label leaks closed

Implements B9, B10 and B4 in one rebuild, as the milestone requires. Measured on
`train` (19,950 trials), before against after:

| leak | before | after |
|---|---|---|
| enrolled-speaker identity predicts absence | **0.795** | **0.508** |
| `overlap_achieved` predicts absence | **0.000** (perfect) | 0.169 |
| P(target absent \| no overlap at all) | 1.000 | **0.500** |
| distinct speakers that ever appear as a present target | 467 of 1,172 | **1,172** |
| unsatisfiable trials | 431 | **50** |

**B10 is fully closed.** The three-tier guard keeps every speaker, so speaker
identity is at chance. Guard tiers on present `train` trials: `chapter` 6,040,
`book` 5,866, `utterance` 2,992.

**B9 hits its stated target exactly.** P(absent | no overlap) is 0.500 over 10,102
trials, and 0.497 / 0.505 within `base` / `hard` separately, as B12 requires.

### Two decisions taken inside PR3

**The `overlap_ratio` floor drops 0.2 → 0.0.** B9's fix needs zero-overlap trials on
the *present* side, which the old floor forbade. This is **not** the ceiling
narrowing 0.7 → 0.45 that `difficulty-dial.md` §3 defers to supervisor agreement —
the ceiling is untouched, so the deliberate match to REAL-TSE still holds. The band
is clipped per trial to `target_activity_ratio`, because a speaker cannot overlap
more of the window than they speak in.

**Absent trials record the strongest guard tier their speaker could support**, rather
than a `n/a` sentinel. A value appearing only on absent rows would itself separate
absent from present — the exact class of giveaway B10 exists to remove.

### What PR3 did NOT fix, stated plainly

The composition balances the *overlap* shortcut perfectly but cannot balance every
shortcut at once. What a listener can infer without identifying whose voice it is:

| voices heard | share | P(target absent) |
|---|---|---|
| 2 | 49.4 % | **0.000** |
| 1 | 45.7 % | 0.447 |
| 0 | 4.9 % | **1.000** |

- **Two voices ⇒ target present, always.** Unavoidable while every absent trial has
  exactly one interferer; removing it needs a two-interferer condition, which B9 did
  not specify. The model must still consult the enrollment to know *which* voice to
  keep, so this shortens the silence decision without answering the extraction one.
- **No voices ⇒ absent, always.** Benign: when nobody speaks at all, emitting silence
  is the correct behaviour, so the shortcut and the right answer coincide.
- **`interferer_activity` rose 0.614 → 0.648**, against B9's expectation that it would
  fall. Cause: `target_only` trials have no interferer at all, so their
  `interferer_activity` is 0 on the *present* side, widening the gap rather than
  closing it. It is a consequence of the four-way composition, not of the shadow-draw
  logic B9 blamed.

The algebra is forced: with `t` = target_only and `n` = noise_only, requiring both
P(absent | no overlap) = 0.5 and P(absent | one voice) = 0.5 gives `n = 0`. B9 chose
to balance the overlap shortcut, and 0.05 of noise-only trials is the price.

### Other changes

- `make_splits.py` stratifies the eval halves by guard tier as well as sex. Weakest
  tier was 8/20 vs 3/20; it is now 6 vs 5, every tier within 1. The old single
  running counter could not balance two axes at once, so the halves are now chosen by
  a deterministic search over which side each stratum starts on.
- `index_utterances` verifies its cache against `splits.yaml`. Redrawing the eval
  pools left caches describing the previous speakers, and every downstream
  disjointness check reads that same file, so a stale cache would have passed silently.
- **New columns:** `condition` (B9's four types), `enrollment_guard` (B10),
  `interrupted` (B13's deferred condition — an interferer onset strictly inside a
  target utterance; 53.2 % of both-speaking trials).
- `target_absent_fraction` is now **derived** from `composition` in `split_config`,
  so the stated and drawn rates cannot drift apart.

## 2026-08-14 — Scope: the task is two-person conversation. At most one interferer

**Decision (GB, 2026-08-14): the constructed set models a conversation between two
people. A trial contains the target, at most one other speaker, and noise. Two
simultaneous non-target speakers never occur, and that is a declared boundary of the
task rather than a gap in the data.**

Taken in response to the PR3 measurement below, having seen the cost. The
alternative — a fifth `two_interferers` condition — was considered and rejected.

### Why

- **It is a scope boundary, so declare it rather than approximate it.** A system
  evaluated on two-speaker mixtures and reported as a two-speaker system is honest. A
  system given a token 10 % of three-speaker trials would be neither properly trained
  for multi-party nor properly scoped, and the headline number would quietly claim
  more than the data supports.
- **It matches the anchor.** Libri2Mix and the REAL-TSE two-speaker framing are both
  two-talker constructions, and `overlap_ratio` and `target_activity_ratio` are
  already tuned to that anchor. Adding a third talker diverges from it on a second
  axis, without the supervisor agreement that `difficulty-dial.md` §3 requires for the
  first.
- **The interference model stays interpretable.** `sir_db` is defined as a ratio
  between two voices. With two interferers it becomes a ratio to a sum, and every SIR
  figure in every table would silently change meaning.

### Consequences — all of these must be carried, not forgotten

**1. "Two overlapping voices" proves the target is present, permanently.** Measured on
`train` (19,950 trials): 9,858 trials have two voices, and `P(target absent | two
voices) = 0.000`. This is now a property of the task definition, not a defect.

**2. Our own benchmark cannot detect a model that exploits it.** The eval splits carry
the same composition, so a system that learns *"two voices → produce output"* scores
normally. **No number this project reports will reveal this behaviour.** That is the
price of the decision and it must be stated wherever the results are.

**3. Every claim must be scoped in words, not just in the appendix.** Results are
about *target speaker extraction from two-speaker mixtures*. They must never be
written as though they generalise to meetings, multi-party audio, or "conversation" in
general.

**4. The AMI secondary check changes meaning.** AMI is multi-party, so it contains
exactly the condition this decision excludes. Two options, and one must be chosen
before that check is run:
   - restrict the AMI trial set to segments where at most two speakers are active, so
     it tests transfer to real audio within the declared scope; or
   - keep AMI as-is and report it explicitly as an **out-of-scope generalisation
     probe**, where a poor result is expected and is not evidence of a bad extractor.
   The first is the honest transfer check; the second measures something else. Do not
   run it without deciding which.

**5. Hallucination on unseen multi-party audio is an untested failure mode.** A model
may emit an interferer's words as the target's when two non-target voices overlap.
Under the content-fidelity metric that is the most damaging error class there is, and
nothing in this project measures it. It belongs in the thesis limitations section as
a named, quantified gap — not as a caveat sentence.

**6. `noise_only` and one-voice trials are unaffected.** For completeness, the other
two composition residuals are benign and stay as they are: *no voices → absent* is
correct behaviour rather than a shortcut (silence is the right output when nobody
speaks), and *one voice → 44.7 % absent* is close enough to a coin flip to carry
almost no information.

### What would reopen this

Evidence that the live judge's content-fidelity score is dominated by hallucinated
interferer speech, or a supervisor requirement for multi-party. Reopening means a
fifth condition and one rebuild — cheap while no audio exists, and progressively more
expensive once eval audio is rendered and judged.

## 2026-08-15 — B2 evidence: LibriSpeech is 86 % speech; overlap is overstated by 25 %

Measurement only. No code changed, no manifest rebuilt. This entry is the evidence
the B2 implementation rests on, recorded before it is written.

**Detector:** `silero-vad` 6.2.1 (Silero Team, 2021, `snakers4/silero-vad`), pip
package so the weights are local and pinned — `torch.hub` would refetch at runtime and
could silently change every number below. Settings `threshold=0.5`,
`min_speech_duration_ms=100`, `speech_pad_ms=30`, and `min_silence_duration_ms` swept.

**Measured against** `data/manifests/` as built at commit `110e64e` (PR3), seed 42.
Part 1: 2,000 utterances sampled from the 137,876 in `data/index/` (6.92 h).
Part 2: 400 `both` trials from `train.csv` (864 distinct utterances).

### Part 1 — how much of an utterance file is speech

**Numbers below are the reproducible run of 2026-08-16**, not the original spike —
see *Reproduction* at the end of this entry. They move the spike's figures in the
third decimal and change no conclusion.

| `min_silence_ms` | speech/dur | median | p10 | p90 | segments | lead_s | trail_s |
|---|---|---|---|---|---|---|---|
| 100 (silero default) | 0.850 | 0.860 | 0.754 | 0.934 | 3.76 | 0.326 | 0.161 |
| 200 | 0.858 | 0.870 | 0.760 | 0.943 | 3.24 | 0.326 | 0.135 |
| **250 (chosen)** | **0.862** | **0.873** | **0.762** | **0.947** | **3.09** | **0.326** | **0.121** |
| 300 | 0.870 | 0.882 | 0.772 | 0.954 | 2.79 | 0.326 | 0.102 |
| 500 | 0.899 | 0.911 | 0.805 | 0.979 | 2.08 | 0.326 | 0.005 |

Threshold sensitivity at 250 ms: 0.873 at 0.3, 0.862 at 0.5, 0.854 at 0.7.

**`min_silence_duration_ms` = 250 ms**, recorded in `generator.yaml`. The whole
100–500 ms range spans only 0.850–0.899, so the headline is robust to the choice —
that is the main reason to be comfortable with it. 500 ms is rejected: trailing
silence collapses to 0.005 s, i.e. it absorbs end-of-file silence into speech.

**Two systematic offsets, and they are the cause of everything in Part 2:**

- **Leading silence 0.326 s per utterance**, near-identical at every setting
  (0.313–0.337 across the threshold sweep). Every LibriSpeech utterance begins about
  a third of a second after its file does.
- **Trailing silence 0.121 s** at the chosen setting.

These are biases, not noise, and they compound: a target and an interferer placed
adjacent are each offset inward, so the generator systematically believes they
coincide more than they do. Carry these figures wherever an overlap number is quoted.

### Part 2 — the label error in the manifests as they stand

Onsets held **unchanged**; only the measurement differs. Sanity check: recomputing
overlap the old way from the recorded onsets reproduces the stored
`overlap_achieved` to max |diff| 0.00006, so the placement code is being read
correctly.

| quantity | file-boundary | VAD | change |
|---|---|---|---|
| `overlap_achieved` (mean) | 0.285 | **0.215** | **−24.6 %** |
| `overlap_achieved` (median) | 0.264 | 0.195 | −26.1 % |
| `target_activity` (mean) | 0.642 | 0.555 | −13.5 % |
| `interferer_activity` (mean) | 0.467 | 0.407 | −12.9 % |
| `interrupted` (rate) | 0.570 | **0.505** | **−11.4 %** |

Per-trial |file − VAD| overlap: mean 0.070, p90 0.148, max 0.270.

**The `interrupted` row was wrong until 2026-08-16** and is corrected here. It
reported 0.725, which is definition **B** — the reading Part 3 *rejected*. Under
the chosen definition A the rate **falls** to 0.505. Consequence 1 below was
written against the old row and has been rewritten to match.

**The per-trial spread is the reason this cannot be patched with a correction
factor.** A uniform −25 % would leave the B13 overlap strata intact; an error that
varies from 0 to 0.270 per trial puts individual trials in the wrong bucket, and the
per-condition table is the thesis's central artefact.

**Overlap collapsing to zero: 12/400 (3.0 %).** Both speakers are talking, but never
simultaneously — the shared interval falls entirely inside one speaker's leading
silence and the other's trailing silence. Turn-taking recorded as talking over each
other. That this is *small* is the reassuring part: the overlap in the data is mostly
genuine, just consistently overstated.

### Three consequences for the implementation

**1. `interrupted` moves, and the cause is definitional. RESOLVED — option A.**
The old test checks one moment per interferer utterance (its file onset). A VAD test
could check *every* speech-segment onset, ~3 per utterance, which makes an interferer
*resuming after their own breath pause* mid-target-sentence count as an interruption.

Measured side by side (Part 3, 400 `both` trials):

| definition | rate | |
|---|---|---|
| old (file onsets) | 0.570 | what the manifests carry today |
| **A: first speech onset per utterance** | **0.505** | **CHOSEN** |
| A′: first speech onset per trial | 0.495 | only the interferer's very first word |
| B: every speech onset | 0.725 | a breath pause starts a new turn |

**Decision (GB, 2026-08-15): option A.** It is the minimal correction to the
2026-08-14 definition — the same number of events as the old test, each moved by the
~0.326 s of leading silence — so `interrupted` keeps meaning "began a turn while you
were mid-sentence". B does not correct the old definition, it *widens* it, and the
0.570 → 0.725 rise would be an artefact of that widening rather than a measurement.
Implemented as `vad.onsets_of(..., first_only=True)`.

**The spread across definitions (0.495–0.725) is wider than most effects this
project will report**, on a B13 reporting condition. That is why it is pinned here
rather than left to the implementation, and why any `interrupted` figure must state
which definition produced it.

**2. The "69 % outside `overlap_tolerance`" figure is NOT a rejection rate.** It was
labelled that in the spike; that label is wrong. It measures trials whose overlap
drifts outside tolerance *with placement held fixed*, and the implementation re-runs
`best_onset` against the new measurement, which re-places most of them. It is
therefore a measure of how far placement must move, and an upper bound on rejections.
The true rate cannot be estimated without building it.

**3. The top of `target_activity_ratio` becomes unreachable.** Realised footprint
activity already reaches p99 0.910 and max 0.946 (`activity_tolerance: 0.1` carries it
past the 0.85 config ceiling). Achieving 0.85 of *speech* would need 0.85/0.86 = 0.988
of the window filled with audio, beyond anything currently achieved and leaving no room
for the gaps `lay_out` requires. The practical ceiling lands near 0.78–0.80. The
REAL-TSE anchor of ~0.75 still sits inside that, so the band needs adjusting rather
than abandoning — but it is a recorded decision, not a silent config edit.
**Done 2026-08-16: lowered to 0.78.** See that entry.

### Why this is done before the renderer, not after

The audio is unaffected — mixtures still contain the pauses, which is what real speech
sounds like. Only the labels and the placement change. But every overlap and activity
column in all six manifests moves, so it forces a rebuild, and a rebuild is cheap only
while no audio exists. `milestones.md` already places B2 before the rebuild for this
reason.

### Precision footnote

These numbers were first measured with Silero's `return_seconds=True`, which also
rounds boundaries to `time_resolution` decimal places — defaulting to **one**, i.e.
0.1 s. Re-measured on 600 utterances with sample-accurate boundaries, mean
speech/duration moves only 0.8655 -> 0.8668 and leading silence 0.321 -> 0.317, so
the figures above stand. Individual utterances move by up to 0.042 though, and
per-trial precision is exactly what overlap bucketing needs, so `src/data/vad.py`
takes sample indices and divides itself rather than using `return_seconds`.

### Reproduction

Promoted from a throwaway spike to `scripts/measure_vad_impact.py`, so a supervisor
can re-derive every figure rather than take them on trust:

    ../tse_venv/bin/python scripts/measure_vad_impact.py

**Actually run 2026-08-16, 16 min**, against `train.csv` as built at
`42a3854`, config md5 `d8bf16d`, seed 42. The sanity check passed at max |diff|
0.00006. Output — `report.txt` plus `meta.yaml` with commit hash, seed, detector
version and the manifest's own build commit — is in
`experiments/results/2026-08-15-vad-impact/`. The directory keeps the 2026-08-15
date of the decision it supports; `meta.yaml` carries the true run date.

**Every table above now comes from that run**, replacing the spike's figures. The
differences are third-decimal and change no conclusion, with one exception: the
spike's `interrupted` row reported definition B rather than the chosen A, and is
corrected in Part 2. This must be re-run after PR2 rebuilds the manifests, because
Part 2 measures label error in manifests that will no longer exist in that form.

## 2026-08-15 — Training audio is pre-rendered to disk, not generated on the fly

**Decision (GB, 2026-08-15): render the training mixtures to disk once, the same
way `val` and the two eval splits already are. On-the-fly generation is removed
from the plan and kept only as the mechanism B7 would need if it were ever
switched on.**

Supersedes `docs/data/data-setup.md` and `docs/data/data-map.md`, both of which
asserted "0 GB, generated on the fly, never stored" as settled fact.

### Why the original reason no longer holds

`data-setup.md` justified on-the-fly generation because the training set would
then "cost no disk and **never repeat**". B7, decided the same day, turned the
repetition off:

> Rebuilding mixtures with fresh loudness and room draws every epoch gives the
> model more variety, but the run can then no longer be reproduced from the
> manifest alone. **Decision: off for the main run.**

With B7 off every draw is fixed in the manifest, so on-the-fly rendering produces
byte-identical audio on every epoch. It repeats exactly. The variety argument
therefore justifies nothing, and what remains is recomputing the same 20,000
mixtures for the lifetime of every training run.

### The other two arguments, checked rather than assumed

**Disk.** 251 GB free. The rendered training set is ~26 GB as 16-bit PCM at
16 kHz (20,000 trials x ~40.8 s of audio each: mixture ~17.9 s including the A5
tail, clean target ~17.9 s, enrollment 5 s), ~16-18 GB as FLAC. Not binding.

**Portability, and this one inverts.** `data-setup.md:440` argued on-the-fly
meant "no 70 GB training set to move, only ~36 GB of public corpora". But a
rendered trial is self-contained: shipping it needs **no LibriSpeech and no
WHAM!** on the training machine. Pre-rendering moves ~26 GB; on-the-fly requires
the full ~36 GB of corpora present wherever training runs. Pre-rendering moves
*less*.

### The risk this actually removes

Each trial needs two RIRs (target and interferer), image-source method, `t60_s`
up to 0.6 s. Generated on the fly that cost is paid every epoch by the dataloader,
competing with feeding the GPU. On a 4-vCPU Kaggle instance a CPU-bound loader
starves the GPU and burns session hours — the failure `milestones.md` M1 already
guards against when it says discovering a problem at hour 11 of a 12-hour session
costs a week. Pre-rendering makes the training step a disk read.

**Taken without the RIR benchmark.** The argument above does not depend on the
per-trial RIR cost, so the decision did not wait for it. That number is still
needed to *size* the render job, not to choose it.

### What this concedes

- **No augmentation variety**, but B7 already conceded that deliberately.
- **Every manifest rebuild invalidates the audio.** Sequencing consequence: render
  *after* B2's rebuild lands, never before. Rendering ~21,200 trials and then
  rebuilding would waste the whole pass.
- **~26 GB.** Cheap at 251 GB free, and it should be excluded from any `rsync` to a
  cluster by name, the same way the corpora already are.

### What stays

`render_trial(row, cfg) -> stems in memory` remains a pure function with the
writer as a thin caller, so a PyTorch `Dataset` can call it directly if B7 is ever
switched on as the ablation it was reserved as. The decision changes how the
renderer is *invoked*, not what it is.

## 2026-08-16 — B2: `target_activity_ratio` ceiling lowered 0.85 -> 0.78

**Decision (GB, 2026-08-16): the top of `target_activity_ratio` comes down from
0.85 to 0.78, in both the global band and `base`.** Config only; it takes effect
at B2 PR2's rebuild, not before.

### Why

`target_activity_ratio` is how much of the clip the target spends talking. PR2
changes what "talking" means: today the generator counts file length, after PR2 it
counts detected speech.

A LibriSpeech file is only ~86 % speech — the rest is the pause before the reader
starts, the pause at the end, and the gaps between phrases. So to get 0.85 of the
window as actual voice you would need ~0.99 of it packed with back-to-back audio.
`lay_out` has to leave gaps between utterances, and the most it has ever achieved
is 0.946. **0.85 is therefore not merely rare, it is impossible.**

### What leaving it at 0.85 would have done

Nothing visible. `build_trial` tries 20 times to find a long enough run of
utterances, fails every time, returns `None`, and the trial is dropped into
`failed_ids`. The build would just quietly produce fewer trials, and every trial
lost would be from the talkative end of the band — so the realised distribution
would stop matching the configured one. That is the same "rejection bends
distributions" problem recorded on 2026-08-13, arriving through the back door.

### Why 0.78 and not lower

The REAL-TSE anchor of ~0.75 still sits inside the band, so nothing we cared about
is given up.

**Honest limit:** 0.78 is close to the edge, not comfortably inside it. Realised
footprint reaches p99 0.910 and max 0.946, which at ~86 % speech is roughly 0.79
and 0.82 of speech. So the top of the new band is reachable only for the best
cases, and a small number of top-of-band trials will still fail. **The check is
PR2's `n_failed`.** If it is material, lower again — with an entry here, not a
silent edit.

## 2026-08-16 — B2: noise beds containing speech are rejected at 0.5 s

**Decision (GB, 2026-08-16): drop any WHAM! noise clip whose longest unbroken run
of detected speech reaches 0.5 s.** Config `noise_speech_rejection.max_speech_run_s`;
applied in `build_manifest.py` by `reject_speech_clips()`. This closes
`noise_speech_rejection`, which `data-construction-parameters.md` has called
critical and unimplemented since it was written.

### Why

WHAM! was recorded in real cafes, bars and restaurants, so some beds have audible
background talkers. Two things go wrong when one of those lands in a trial:

1. **The metric punishes the model for being right.** Those words are genuinely in
   the mixture, but they are in no transcript. If the extractor passes them through
   and the judge hears them, they score as words the model invented.
2. **A third talker appears.** CLAUDE.md declares this a two-speaker task. An
   unlabelled voice in the noise bed quietly breaks that.

### Why the longest run, and why 0.5 s

Rejection is on `max_segment_s`, the longest *unbroken* run, not the total. Half a
second in one piece can be a word; the same half second spread over five 100 ms
blips is the detector twitching at a laugh, a door or a clatter. Below ~0.5 s there
is not enough continuous voice to become text, so rejecting there would cost pool
for no gain.

### Borrowed in intent from WHAMR!, but not the same rule

`data-construction-parameters.md` took this parameter from WHAMR!'s `SNR_THRESH`
(`noisesampler.py:45-62`), which rejects a noise segment when its speech **energy**
exceeds −6 dB. **We diverge:** we threshold the **duration of a detected speech
run** instead, using the Silero pass B2 already pays for. Our failure mode is words
being *transcribed*, and a quiet but clear background talker is a transcription
risk at an energy WHAMR!'s test would pass. We also reject whole clips rather than
resampling the offset, so nothing has to be written back to the manifest. Cite the
idea as borrowed; do not describe the rule as WHAMR!'s.

### Cost

Measured, not projected — `scripts/screen_noise_speech.py`, 2026-08-15:

| pool | clips | dropped | kept |
|---|---|---|---|
| tr | 20,000 | 821 (4.1 %) | 19,179 |
| cv | 5,000 | 98 (2.0 %) | 4,902 |
| tt | 3,000 | 39 (1.3 %) | 2,961 |

### Where it is applied, and why not in the renderer

`milestones.md` put this in the renderer, on the assumption it needed audio. It
does not: the screening pass already measured every clip, and the manifest is what
*names* the noise clip for each trial. Filtering the pool before selection is
therefore the only place it works — filter later and the manifest can still point
at a clip that should not exist. The renderer just reads what the manifest says.

### What this does not promise

The cutoff is a detector output, so it inherits Silero's mistakes. A kept clip may
still hold faint or short speech. This lowers contamination a long way; it does not
prove it is zero, and the write-up should say "screened", not "clean".

### Consequences

- `data/index/noise_speech_{split}.csv` is now a **build input**, not a report. It
  is required, and `build_manifest.py` fails loudly if it is missing or does not
  cover the pool.
- The `vad:` block already invalidates the manifests; it now invalidates this
  screening index too, since the same detector settings produced it.
- Each split's `meta.yaml` records `noise_clips_screened`, `noise_clips_kept` and
  `noise_max_speech_run_s`, because the cutoff is a config value and manifests are
  not in git.
