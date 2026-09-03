"""FR, the fabrication rate of docs/data/metric-definitions.md 3.3b.

Answers one question: did the listener report words that NOBODY said?

The third disjoint failure mode. LCF-WER measures content RECOVERED, ICR
measures content LEAKED from the other speaker, and FR measures content
INVENTED -- present in the response but in neither speaker's script. WER pools
invented words with leaked ones as insertions and ICR only sees words traceable
to the interferer, so before this the invented residue was not isolated by
anything in the per-system protocol.

WHY IT EXISTS. metric-definitions.md 3.3 records that NRR "detects a judge that
DECLINES, not one that CONFABULATES", and that the judge measured 2026-09-02
does not decline -- it invents, 17-42 words on clips containing no speech at
all. That made the invention rate a property measured once OF THE JUDGE rather
than per system, so a system whose artefacts PROVOKE invention was invisible.
Measured on the frozen prompt over 103 `sir0_val` `both` trials, 2026-09-03:
clean target 0.20 invented content words per trial, mixture 1.24, our baseline
1.83, WeSep 1.80. **Both extractors raise fabrication ~48 % above doing nothing,
and the two are indistinguishable from each other despite a 25-point WER gap** --
so fabrication is an axis of its own, not a by-product of extraction quality.
That is the effect that was going unmeasured. decisions-m4.md 2026-09-03.

Scoring works on TEXT, like lcf_wer and icr. Audio enters only through
`transcribe_one`, so the same swap takes it from the offline ASR to the judge.

**COMPARE SYSTEMS ON `invented_per_trial`, NOT ON `mean_invented_percent`.**
The percentage divides by the RESPONSE's own length, so a terser listener scores
worse on it for saying less rather than for inventing more. Measured 2026-09-03,
this is not hypothetical: our baseline and WeSep invent 1.83 and 1.80 content
words per trial -- indistinguishable -- but read 10.4 % and 13.8 % because WeSep's
responses average 15.1 content words against 17.9. **A system-to-system claim
made on the percentage would have been an artefact of verbosity.** The percentage
is retained because it is the right reading WITHIN a system (how much of what was
said was made up); the per-trial count is the right one ACROSS systems.

**FR IS AN UPPER BOUND ON FABRICATION, AND ITS CEILING IS NOT ZERO.** A response
word can be absent from both scripts for three reasons: the listener invented it;
the listener misheard a word that was said and produced a different real word;
or the reference script does not match what was uttered. Only the first is
fabrication. This is why the ceiling row -- the listener given the CLEAN TARGET,
where by construction there is nothing to invent from -- is mandatory: it
calibrates how much this measure fires on perfect input, and only the excess
above it is attributable to the system under test. Never quote FR without it.
Contrast ICR, whose ceiling genuinely is 0.0 because a word is either in the
interferer's script or it is not.
"""

from dataclasses import dataclass, field

from .icr import content_words

K_VALUES = (1, 2, 3, 5)
HEADLINE_K = 2

# A fraction over a tiny denominator is not a fraction: 1 invented word in a
# 1-word response reads 100 %. Same guard, and same value, as ICR's
# MINIMUM_AVAILABLE_FOR_FRACTION -- kept equal on purpose so the two rates are
# read on the same footing.
MINIMUM_RESPONSE_WORDS_FOR_FRACTION = 5


def invented_content(response_text, target_text, interferer_text):
    """Response content words attributable to neither speaker.

    `interferer_text` is "" on a target-only trial and `target_text` is "" on a
    target-absent one; set difference handles both without a special case.
    """
    return (content_words(response_text)
            - content_words(target_text)
            - content_words(interferer_text))


@dataclass
class TrialFabrication:
    invented_count: int = 0
    response_count: int = 0
    invented_words: set = field(default_factory=set)

    @property
    def invented_fraction(self):
        if self.response_count == 0:
            return None
        return self.invented_count / self.response_count


def measure_fabrication(response_text, target_text, interferer_text):
    invented = invented_content(response_text, target_text, interferer_text)
    return TrialFabrication(len(invented),
                            len(content_words(response_text)),
                            invented)


@dataclass
class FrResult:
    fr_at_k: dict = field(default_factory=dict)
    mean_invented_percent: float = None
    invented_per_trial: float = None
    trials_for_fraction: int = 0
    trials_empty: int = 0
    trials_scored: int = 0
    trials_total: int = 0

    @property
    def headline(self):
        return self.fr_at_k.get(HEADLINE_K)

    def __str__(self):
        if not self.trials_scored:
            return "FR: undefined, no trials with a non-empty response"
        at_k = "  ".join(f"@{k} {self.fr_at_k[k]:.1f}%"
                         for k in K_VALUES if k in self.fr_at_k)
        mean = ("—" if self.mean_invented_percent is None
                else f"{self.mean_invented_percent:.1f}%")
        per = ("—" if self.invented_per_trial is None
               else f"{self.invented_per_trial:.2f}")
        return (f"FR {at_k}  (n={self.trials_scored})\n"
                f"  {per} invented content words per trial  <-- COMPARE SYSTEMS ON THIS\n"
                f"  mean invented {mean} of response content words, over "
                f"{self.trials_for_fraction} trials with "
                f">={MINIMUM_RESPONSE_WORDS_FOR_FRACTION} words\n"
                f"  {self.trials_empty} of {self.trials_total} trials had no "
                f"response content (see NRR)")


def compute_fr(response_texts, target_texts, interferer_texts):
    """FR@k and the mean invented fraction over parallel text sequences.

    NO PER-K ELIGIBILITY, and the difference from ICR is structural rather than
    an oversight. ICR's k is capped by how many exclusive words the interferer
    actually said, so a trial with one available word cannot reach @2 and is
    excluded from that denominator. Fabrication has no such ceiling -- any word
    at all can be invented -- so every trial with a non-empty response is
    eligible at every k, and one denominator serves them all.

    An EMPTY response is excluded rather than scored 0 % invented. It is a
    non-response, which is NRR's measurement, and scoring it clean here would
    let a muting system lower its fabrication rate by saying nothing.
    """
    response_texts = list(response_texts)
    target_texts = list(target_texts)
    interferer_texts = list(interferer_texts)
    if not len(response_texts) == len(target_texts) == len(interferer_texts):
        raise ValueError(
            f"{len(response_texts)} responses, {len(target_texts)} targets, "
            f"{len(interferer_texts)} interferers -- must be parallel"
        )

    fabrications = [
        measure_fabrication(response_text, target_text, interferer_text)
        for response_text, target_text, interferer_text
        in zip(response_texts, target_texts, interferer_texts)
    ]

    result = FrResult(trials_total=len(fabrications))
    scored = [f for f in fabrications if f.response_count > 0]
    result.trials_empty = len(fabrications) - len(scored)
    result.trials_scored = len(scored)
    if not scored:
        return result

    for k in K_VALUES:
        fired = sum(1 for f in scored if f.invented_count >= k)
        result.fr_at_k[k] = fired * 100.0 / len(scored)

    # ABSOLUTE count first, because it is the only length-safe reading. See the
    # module docstring's warning.
    result.invented_per_trial = sum(f.invented_count for f in scored) / len(scored)

    for_fraction = [f for f in scored
                    if f.response_count >= MINIMUM_RESPONSE_WORDS_FOR_FRACTION]
    result.trials_for_fraction = len(for_fraction)
    if for_fraction:
        result.mean_invented_percent = (
            sum(f.invented_fraction for f in for_fraction) * 100.0 / len(for_fraction)
        )
    return result


def score_audio_files(audio_file_paths, target_texts, interferer_texts, transcribe_one):
    response_texts = [transcribe_one(path) for path in audio_file_paths]
    return compute_fr(response_texts, target_texts, interferer_texts)
