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
Supervisor meeting notes 8-10 (docs/specification.md) redefine the goal.
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
`docs/metric-definitions.md` §3.5. Cross-modality comparisons are valid on
that end-to-end number and invalid as statements about the judge's listening
ability, because in the text condition the judge is close to a pass-through.

## 2026-08-07 — Metric is the primary contribution
Spec note 1 ("the actual score values from my defined metric do not
matter — the metric itself matters more") plus note 8 (metric first, then
architecture) make this explicit. The metric definition and its harness
are the deliverable that must not be cut under any schedule pressure.
Full specification in docs/metric-definitions.md.

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

**Note on the WHAM! conversion in `docs/data-setup.md` step 1e:** WHAM!
noise is *already* 16 kHz, so `-ar 16000` there is a no-op guard. The 4×
storage reduction comes entirely from stereo → mono and 32-bit float →
16-bit. Do not defend it as a sample-rate saving.

**16-bit is a deliberate, mildly lossy choice.** 32-bit float sources are
quantised to 16-bit on ingest. 16-bit gives ~96 dB of dynamic range, far
more than these signals use at the SNRs we mix at, and it matches what the
judge accepts anyway. Chosen, not defaulted to.

Sources: <https://ai.google.dev/gemini-api/docs/live-api/capabilities>,
<https://firebase.google.com/docs/ai-logic/live-api/limits-and-specs>,
<http://wham.whisper.ai/>, and `docs/definitions.md:60`.

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
  `docs/metric-definitions.md:198-200` must be **CC BY-NC 4.0**, with
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
are in `docs/data-licences.md`.

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
