
---

## 2026-08-31 — Floor and ceiling broken into error types: the floor is insertion-dominated

**Measured, not decided.** Recorded because the error split changes what the
extractor's job actually is, and because a bare word error rate cannot support
the claim below.

Same data as the C2 entry above — `eval_public`, `condition=both`, n=230, 6,471
reference words, `faster-whisper small.en`, Whisper `EnglishTextNormalizer`,
scored from `experiments/results/transcripts.csv` with no new ASR run.

Rates are each error type as a share of reference words, so the three sum to the
word error rate.

| system | LCF-WER | substitutions | deletions | insertions |
|---|---|---|---|---|
| **floor** (raw mixture) | 57.4 % | 23.2 | **3.5** | **30.8** |
| **ceiling** (clean target) | 6.1 % | 2.7 | 0.4 | 2.9 |

### What it means

**Doing nothing means the listener hears too much, not too little.** Only 3.5 %
of the target's words are lost outright. The dominant damage — 30.8 points, more
than half the floor's total error — is words being **added**: the other speaker
being reported as if they were the target.

**This reframes the task.** At these signal-to-interference ratios (mean +4.9 dB
on `eval_public`) the target is rarely buried. The job is overwhelmingly
*removing the other voice*, not *recovering hidden speech*. An extractor that
improves audibility without suppressing the interferer cannot move this number
much.

**It is also the empirical case for ICR.** Word error rate counts an inserted
interferer word and an inserted hallucination identically. Since insertions are
the floor's largest error type, the metric most needed alongside LCF-WER is the
one that says *whether the inserted words were the other speaker's* — which is
ICR (`metric-definitions.md` 3.2). Without it, the largest component of the
headline number is undiagnosed.

### Caveat that must travel with these numbers

**The alignment is chosen to minimise total edits, so the split between
substitutions and insertions is the cheapest explanation, not necessarily the
true mechanism.** When the ASR transcribes the target for a stretch and then
switches to the other speaker's sentence — observed on inspection, 2026-08-30 —
whether those words score as substitutions or insertions depends on the relative
lengths of the two utterances, not on what physically happened. Treat the
decomposition as suggestive and directional, and do not build a mechanism claim
on it alone.

### Why it is worth recording anyway

The two error types move in **opposite** directions as masking is made more
aggressive: harder masking removes the interferer (fewer insertions) and
introduces artefacts (more substitutions). So the decomposition is a cheap probe
of the artefact-versus-residue trade-off that `metric-definitions.md` 1
hypothesises about, and it needs no metric beyond the one already defined.
