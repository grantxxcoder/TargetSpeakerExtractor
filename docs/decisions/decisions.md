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
and masks words even though it contains none — `noise_speech_rejection` (not yet
implemented) guarantees no intelligible speech in it. Non-intelligible noise cannot
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
