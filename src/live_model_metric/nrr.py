"""NRR, the non-response rate of docs/data/metric-definitions.md 3.3.

Of the trials where the target actually spoke, what fraction came back with no
words from the judge? Lower is better.

WHY IT EXISTS. A system that outputs silence scores 0 % on ICR -- nothing came
out, so nothing leaked -- a perfect score for doing nothing. More usefully: NRR
on clean target audio is the JUDGE'S own defect rate, and it is the only score
here that separates "the judge declined" from "the system failed".

It is a tripwire, not a quality measure. It will not rank two working systems.

DETECTION. The prompt instructs the judge: if you cannot identify any speech,
return nothing. An empty response is therefore the signal. No pattern matching:
searching responses for "nothing", "i cannot", "no speech" was measured against
the ground-truth texts and matched 7.6 % of eval_public trials, because the
speakers say those words. decisions-m4.md 2026-08-31.

Scoring works on TEXT, like lcf_wer and icr.
"""

from dataclasses import dataclass, field

from .lcf_wer import normalise_text

# Measured, not assumed: faster-whisper small.en emits "you" on digital silence,
# 8 of 8 absent trials (decisions-m3.md 2026-08-28). Same class of artefact for
# the rest. A transcriber cannot be instructed, so it needs this; a prompted
# judge returns nothing and lands in "silence".
SILENCE_ARTEFACTS = frozenset({"you", "thank you", "thanks for watching", "bye"})

REASONS = ("silence", "artefact")


def non_response_reason(response_text):
    """Why this response counts as a non-response, or None if it is a real one.

    A reason rather than a bool: an empty response and a transcriber
    hallucinating on silence are different events, and one rate hides which.
    """
    normalised = normalise_text(response_text).strip()
    if not normalised:
        return "silence"
    if normalised in SILENCE_ARTEFACTS:
        return "artefact"
    return None


@dataclass
class NrrResult:
    nrr: float = None
    trials_scored: int = 0
    trials_excluded_absent: int = 0
    non_responses: int = 0
    counts_by_reason: dict = field(default_factory=dict)

    def __str__(self):
        if self.nrr is None:
            return "NRR: undefined, no scorable trials"
        breakdown = ", ".join(f"{reason} {count}" for reason, count
                              in sorted(self.counts_by_reason.items())) or "none"
        return (f"NRR {self.nrr:.2f} %  "
                f"({self.non_responses} of {self.trials_scored} trials; {breakdown})"
                + (f"; {self.trials_excluded_absent} absent trials excluded"
                   if self.trials_excluded_absent else ""))


def compute_nrr(response_texts, target_texts):
    """Score a set of trials. `target_texts` is REQUIRED.

    Absent trials are excluded here rather than by the caller remembering to.
    Where the target never speaks, reporting nothing is the CORRECT answer, so
    counting it as a non-response inverts the metric -- an earlier draft scored
    32.8 % NRR on clean target audio for exactly that reason. Absent trials are
    measured by the invented-speech row instead (B4).
    """
    response_texts = list(response_texts)
    target_texts = list(target_texts)
    if len(response_texts) != len(target_texts):
        raise ValueError(
            f"{len(response_texts)} responses but {len(target_texts)} targets"
        )

    result = NrrResult()
    for response_text, target_text in zip(response_texts, target_texts):
        if not normalise_text(target_text).strip():
            result.trials_excluded_absent += 1
            continue
        result.trials_scored += 1
        reason = non_response_reason(response_text)
        if reason is not None:
            result.non_responses += 1
            result.counts_by_reason[reason] = result.counts_by_reason.get(reason, 0) + 1

    if result.trials_scored:
        result.nrr = result.non_responses * 100.0 / result.trials_scored
    return result


def system_attributable_nrr(system_nrr, ceiling_nrr):
    """The part of a system's NRR that is not the judge's own doing.

    NRR on the clean target IS the judge's defect rate: it declined on perfect
    audio. Only the excess above that is attributable to the extractor. Without
    the subtraction, a judge with poor instruction-following makes every system
    look degenerate -- J2's warning that a degenerate judge and a degenerate
    extractor are indistinguishable in the numbers.
    """
    return system_nrr - ceiling_nrr


def silence_compliance(response_texts):
    """Fraction of responses that are a non-response, for judge selection.

    Run on SILENT input. A judge that keeps talking when there is nothing to
    hear cannot be scored on NRR, and that disqualifies the judge rather than
    calling for a cleverer detector.
    """
    response_texts = list(response_texts)
    if not response_texts:
        return None
    quiet = sum(1 for text in response_texts if non_response_reason(text) is not None)
    return quiet * 100.0 / len(response_texts)


def score_audio_files(audio_file_paths, target_texts, transcribe_one):
    response_texts = [transcribe_one(path) for path in audio_file_paths]
    return compute_nrr(response_texts, target_texts)
