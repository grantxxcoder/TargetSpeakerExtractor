# Decision Log — M4 (the metric is computable end to end)

Covers **LCF — Live-model Content Fidelity**, the project's primary
contribution: the three scores (LCF-WER, ICR, NRR), the judge, the fixed prompt,
the anchors, and the audio/text modality split. Defined in
`docs/data/metric-definitions.md`; this log records the decisions that document
does not fix.

Created 2026-08-31 when the log was split by milestone. See `milestones.md` M4.

**The conventional metrics are NOT here.** SI-SDR, DNSMOS, offline-WER and the
offline ASR that produces them are M3 (`decisions-m3.md`). Keeping them apart is
what stops the divergence result of `metric-definitions.md` 6 from being an
argument about one instrument measuring itself.

**Standing constraint on everything in this file (CLAUDE.md).** The judge model
never appears in the training loop, in any form, including as a proxy or a data
filter. Every judge result records the exact model ID, the exact prompt, the
input modality and the run date, because closed models change silently.

---

## 2026-08-31 — J1 CLOSED. The judge is audio-in / text-out, not full-duplex

**Decision: the judge is an audio-in / text-out model. Full duplex is not
required.** Closes J1 in `decisions-pending.md`. Unblocks J2 and therefore all
of M4.

**Why.** The property LCF measures is the judge's **audio encoder**, not its
turn-taking. Nothing in LCF-WER, ICR or NRR reads duplexing, and
`metric-definitions.md` 3.1 step 2 already makes audio *output* optional — "*if*
the model responds in audio, transcribe the response". An audio-in / text-out
model has exactly the learned encoder and the wide training distribution that
the artefact hypothesis in 3.1 is about.

**Three consequences that are gains, not compromises.**

1. **The response-transcription ASR disappears from the measuring instrument.**
   3.1 step 2 calls that ASR "a component of the measuring instrument, and
   changing it invalidates comparisons". A text response deletes it, and with it
   a whole class of invalidation. Only the *front-end* ASR of the text reference
   condition (3.5) remains, and it belongs to the system under test.
2. **A wider candidate field**, open and closed. Full duplex narrows the open
   field to roughly Moshi, whose task adherence is 1.26/5 on FullDuplexBench —
   below the bar our own ceiling gate sets.
3. **A less noisy instrument.** A full-duplex conversational model is tuned to
   *converse*, not to *report*. Handed "report what you heard" it may answer the
   content or chat about it instead of repeating it. That lands in **NRR**, where
   a degenerate judge is indistinguishable from a degenerate extractor. Removing
   the conversational tuning removes that confound.

**The cost, stated plainly.** This is a deviation from the project's stated
objective — CLAUDE.md and spec note 10 both say "live speech-to-speech model
(Gemini Live and similar)". The defensible sentence, which must appear in the
write-up and not merely in this log: *we used an audio-in model because the
measurement depends on the judge's audio encoder rather than on its duplexing.*
If that argument is not made explicitly, a reviewer is entitled to say the
thesis measured something other than what it set out to.

**How the deviation gets bought off rather than argued away: a small
full-duplex confirmation run.** The headline benchmark is audio-in / text-out.
Alongside it, ~50 trials are scored on a genuine full-duplex live API. At the
prices measured in J2 that is a few dollars. It converts the claim from "we
relaxed the objective and here is why that is acceptable" into "the ranking was
confirmed on a real full-duplex model at n=50". **Recorded here as part of the
decision, not as an optional extra**, because the argument above is weaker
without it.

**What this does NOT change.** The judge is still held out from training
absolutely — never a proxy, never a data filter (CLAUDE.md). Modality is still a
recorded property of every trial (3.5). The open-weight reproducibility anchor
is still required, and for reproducibility rather than cost (J2). Every judge
result still records the exact model ID, exact prompt, modality and run date.

**Still open after this:** J2a — which closed model for the headline; J2b —
which open-weight anchor. Both are now decided by the candidate gate rather than
by argument. J1 no longer blocks either.
