"""LCF-WER — the primary score of Live-model Content Fidelity.

Implements docs/data/metric-definitions.md 3.1. This is the ONLY metric in this
file, deliberately: ICR (3.2) and NRR (3.3) come later, as separate modules.

WHAT IT MEASURES, IN ONE SENTENCE
---------------------------------
Of the words the target speaker actually said, how many did the live model fail
to report back? Lower is better. 0 % means it recovered every word.

WHAT MAKES IT DIFFERENT FROM ORDINARY ASR WER
---------------------------------------------
Ordinary WER scores a transcriber. LCF-WER scores an ASSISTANT LISTENING THROUGH
our system. The chain is:

    mixture ──► our extractor ──► audio ──► live model ──► "what I heard" ──► WER

So the number contains three things at once: what the extractor destroyed, what
the live model mishears, and how well the live model follows the instruction to
report rather than converse. That is a feature, not a confound — it is the whole
point of the metric — but it is why the CEILING anchor is mandatory. Without it
you cannot tell "our system damaged the audio" from "this judge cannot listen".

WHY WORD ERROR RATE AND NOT A LEARNED QUALITY SCORE
---------------------------------------------------
metric-definitions.md 4: the score must be gaming-resistant. WER is discrete,
semantic and non-differentiable — there is no gradient for an adversarial
waveform perturbation to hill-climb. The cautionary tale is the REAL-TSE
Challenge, where teams over-optimised DNSMOS-OVRL (a smooth learned predictor)
until its correlation with human MOS was essentially zero, LCC +0.003, and the
organisers had to swap the official metric after the fact.

Borrowed: word error rate is standard ASR practice; the corpus-level convention
(total edits over total reference words) follows Morris et al. (2004). Text
normalisation is Whisper's, Radford et al. (2022). Neither is our contribution
and both are cited as borrowed — see decisions-m3.md and CLAUDE.md.

NOT COMPARABLE to published REAL-TSE TER numbers: different data, different
metric, different protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

# --- 1. Normalisation ------------------------------------------------------
#
# B5, decisions-m0.md 2026-08-13. Whisper's EnglishTextNormalizer, pinned at
# whisper-normalizer==0.1.15.
#
# WHY THIS IS PART OF THE METRIC AND NOT A DETAIL. Without it, "twenty five"
# against "25", or "don't" against "do not", or a trailing full stop, all count
# as errors the system did not make. The normaliser is therefore a COMPONENT OF
# THE MEASURING INSTRUMENT: metric-definitions.md 3.1 requires it to be fixed,
# published, applied identically to both sides, and never hand-tuned per system.
# Tuning it per system is how you would flatter your own model without improving
# it.

NORMALISER_ID = "whisper-normalizer==0.1.15:EnglishTextNormalizer"

_normaliser = None


def normalise(text) -> str:
    """Anything -> normalised lowercase word string. None/NaN -> "". Never raises.

    Returns "" rather than raising on missing values because a missing response
    is a real outcome (the judge said nothing) and must be SCORED, not skipped.
    Skipping it would quietly delete the worst trials from the average.
    """
    global _normaliser
    if text is None:
        return ""
    if isinstance(text, float) and text != text:      # NaN from a CSV cell
        return ""
    s = str(text)
    if not s.strip():
        return ""
    if _normaliser is None:
        from whisper_normalizer.english import EnglishTextNormalizer
        _normaliser = EnglishTextNormalizer()
    return _normaliser(s)


# --- 2. One trial: count the edits ----------------------------------------

@dataclass
class Edits:
    """The three error types, kept separate. `n_ref` is the reference length.

    KEEPING S/D/I SEPARATE IS THE POINT. A single WER number cannot tell you
    which failure you have, and they mean opposite things about a TSE system:

      deletions     target words that never came back  -> the system SUPPRESSED
                                                          the target
      insertions    words that were not said           -> the system LEAKED
                                                          something (the other
                                                          speaker, or noise the
                                                          judge misheard) or the
                                                          judge hallucinated
      substitutions the right slot, the wrong word     -> the system DEGRADED
                                                          the audio; phonetic
                                                          detail was lost

    A muting extractor produces almost all deletions. A pass-through produces
    insertions. Reporting only the total hides the difference between the two
    failures this project exists to tell apart.
    """
    substitutions: int = 0
    deletions: int = 0
    insertions: int = 0
    n_ref: int = 0

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def edit_counts(reference, hypothesis) -> Edits:
    """Compare one reference to one hypothesis. Both normalised here, not by the
    caller, so the two sides can never be normalised differently.

    Returns COUNTS, not a rate. This matters — see `corpus_wer`.
    """
    import jiwer
    ref, hyp = normalise(reference), normalise(hypothesis)
    n_ref = len(ref.split())
    if n_ref == 0:
        # No reference words: WER is 0/0, undefined. The caller decides what to
        # do; this function refuses to invent a number.
        return Edits(n_ref=0)
    m = jiwer.process_words([ref], [hyp])
    return Edits(substitutions=m.substitutions, deletions=m.deletions,
                 insertions=m.insertions, n_ref=n_ref)


# --- 3. Many trials: the corpus rate --------------------------------------

@dataclass
class LcfWer:
    """The result. Percentages, lower is better."""
    wer: float | None = None          # the headline
    n_scored: int = 0                 # present trials with reference text
    n_excluded_absent: int = 0        # B4: target never spoke
    n_excluded_no_ref: int = 0        # present but empty reference (a data bug)
    ref_words: int = 0
    edits: int = 0
    # The same three error types, as a share of reference words. They sum to `wer`.
    sub_rate: float | None = None
    del_rate: float | None = None
    ins_rate: float | None = None

    def as_dict(self):
        return asdict(self)

    def __str__(self):
        if self.wer is None:
            return "LCF-WER: undefined (no scorable trials)"
        return (f"LCF-WER {self.wer:.2f} %  "
                f"(sub {self.sub_rate:.2f} + del {self.del_rate:.2f} + "
                f"ins {self.ins_rate:.2f})  "
                f"over {self.n_scored} trials / {self.ref_words} reference words"
                + (f", {self.n_excluded_absent} absent trials excluded"
                   if self.n_excluded_absent else ""))


def lcf_wer(responses, references, target_absent=None) -> LcfWer:
    """Score a set of trials.

    responses      what the judge reported hearing, one per trial (`r`)
    references     what the target actually said, one per trial (`t`)
    target_absent  per-trial flag; None means "no absent trials in this set"

    CORPUS-LEVEL, NOT THE MEAN OF PER-TRIAL RATES.
    ----------------------------------------------
    We sum all edits and divide by all reference words:

        WER = (S + D + I) / N        summed over trials

    The alternative — averaging each trial's own WER — weights a 3-word trial
    the same as a 40-word one, so a handful of very short utterances can swing
    the headline. Our trials vary from a few words to ~40, so the choice is not
    cosmetic. This is the standard convention and it is stated here because it
    is a methodological choice that must be defensible, not a default.

    WER CAN EXCEED 100 %.
    ---------------------
    Insertions are not bounded by the reference length, so a system that adds
    more words than were spoken scores above 100 %. This is not a bug and it is
    diagnostic: `tiny.en` was disqualified as our offline ASR precisely because
    its floor was 123 % — it invented more words than existed, which makes it
    un-rankable (decisions-m3.md 2026-08-28).

    ABSENT TRIALS ARE EXCLUDED, NOT SCORED AS ZERO.
    -----------------------------------------------
    B4, decisions-m0.md 2026-08-13. When the target never speaks there is no
    reference text, so WER is 0/0 — undefined, not perfect. Folding those trials
    in as 0 % would reward a system for saying nothing on trials where nothing
    was said, and dilute the headline with trials the metric cannot judge. They
    are counted and reported separately, and they get their own invented-speech
    score later.

    A NON-RESPONSE COUNTS AS ALL-DELETIONS, ON PURPOSE.
    ---------------------------------------------------
    If the judge reports nothing, every reference word is a deletion and the
    trial scores ~100 %. That is the honest reading: a model that said nothing
    recovered nothing. It also means LCF-WER already punishes the degenerate
    mute — so NRR does not exist to catch it here. NRR exists to protect ICR,
    which a silent system would otherwise score perfectly on.
    """
    responses = list(responses)
    references = list(references)
    n = len(responses)
    if len(references) != n:
        raise ValueError(f"{n} responses but {len(references)} references")
    if target_absent is None:
        target_absent = [False] * n
    else:
        target_absent = [bool(a) for a in target_absent]
        if len(target_absent) != n:
            raise ValueError(f"{n} responses but {len(target_absent)} absent flags")

    out = LcfWer()
    for r, t, absent in zip(responses, references, target_absent):
        if absent:
            out.n_excluded_absent += 1
            continue
        e = edit_counts(t, r)
        if e.n_ref == 0:
            # Marked present but carries no reference text. That is a data
            # problem, not a score of 0 — counted so it cannot hide.
            out.n_excluded_no_ref += 1
            continue
        out.n_scored += 1
        out.ref_words += e.n_ref
        out.edits += e.total
        out.sub_rate = (out.sub_rate or 0) + e.substitutions
        out.del_rate = (out.del_rate or 0) + e.deletions
        out.ins_rate = (out.ins_rate or 0) + e.insertions

    if out.ref_words:
        pct = 100.0 / out.ref_words
        out.wer = round(out.edits * pct, 2)
        out.sub_rate = round(out.sub_rate * pct, 2)
        out.del_rate = round(out.del_rate * pct, 2)
        out.ins_rate = round(out.ins_rate * pct, 2)
    else:
        out.sub_rate = out.del_rate = out.ins_rate = None
    return out


# --- 4. Reading the number ------------------------------------------------
#
# A bare LCF-WER is close to meaningless. metric-definitions.md 3.4 makes two
# anchors mandatory on every results table, and they are what give it a scale:
#
#   FLOOR    the unprocessed mixture. What you get for doing nothing.
#   CEILING  the clean target. The best any extraction could achieve ON THIS
#            JUDGE — so it also measures the judge's own competence.
#
# Measured with the offline ASR standing in for a judge, on eval_public `both`
# trials (n=230, 2026-08-30): floor 57.4 %, ceiling 6.1 %. The 51-point gap is
# the room a system has to work in. A system at 50 % has captured 14 % of the
# available headroom, not "half the words wrong in isolation".
#
# `headroom_captured` puts a system on that scale. It is also exactly the
# normalisation any multi-metric composite would need, because it is
# dimensionless and 0 = doing nothing, 1 = the best achievable (see
# decisions-pending.md J4).


def headroom_captured(system_wer, floor_wer, ceiling_wer) -> float:
    """Where a system sits between doing nothing (0.0) and clean audio (1.0).

    Values below 0 mean the system is WORSE THAN DOING NOTHING — which has
    happened in this project (the epoch-24 checkpoint fell below pass-through),
    so the function returns the real number rather than clipping it. Clip at the
    plotting layer, where hiding it is a display choice, not a measurement.
    """
    span = floor_wer - ceiling_wer
    if span == 0:
        raise ValueError("floor equals ceiling: no headroom, nothing to measure")
    return (floor_wer - system_wer) / span
